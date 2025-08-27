import os
import requests
from openai import OpenAI
import asyncio
import uuid, time
import hashlib
import gradio as gr
from typing import Optional, List, Dict, Any, Type, TypedDict, Annotated
from pydantic import BaseModel, Field, ConfigDict
from langchain.tools import BaseTool
from langchain.callbacks.manager import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun
from sentence_transformers import CrossEncoder, SentenceTransformer

import re
import chromadb
from dotenv import load_dotenv

from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
import operator
from chromadb.utils import embedding_functions
from langchain_openai import ChatOpenAI

SYSTEM_PROMPT = """You are a highly specialized programming assistant. Your SOLE PURPOSE is to use the provided tools to find the most accurate and up-to-date information.

**CRITICAL RULES:**
1. **DO NOT answer programming questions from your own knowledge.** You are forbidden from answering technical questions, especially those involving code, errors, or specific libraries, from memory.
2. **ALWAYS prioritize using a tool.** Your primary function is to route the user's query to the correct tool.
3. For any question about **Oracle, SQL, PL/SQL, or ORA-errors**, you **MUST** use the `internal_knowledge_search` tool. No exceptions.
4. For all other general programming questions, error messages, or library usage, you **MUST** use the `stack_exchange_search` tool.
5. For questions requiring **current, real-time information, recent news, trending topics, or information about events after 2023**, you **MUST** use the `google_search_rag` tool.
6. After getting results from a tool, synthesize them into a helpful answer. If the tools return no results, you must state that you could not find an answer.
7. **ALWAYS call a tool - never respond without using at least one tool first.**
"""


class GoogleSearchRAGInput(BaseModel):
    """Input schema for Google Search RAG tool."""
    query: str = Field(description="The user's question to search for on the web")
    n_results: int = Field(
        default=5,
        description="Number of search results to retrieve (max 10)"
    )


class GoogleSearchRAGTool(BaseTool):
    """Tool for web search using Google Custom Search API with RAG capabilities."""
    name: str = "google_search_rag"
    description: str = (
        "Use this tool for questions requiring current, real-time information from the web, "
        "recent news, current events, trending topics, or when Stack Exchange and internal knowledge "
        "don't have sufficient information. This tool searches the web and provides comprehensive "
        "answers based on multiple authoritative sources with proper citations."
    )
    args_schema: Type[BaseModel] = GoogleSearchRAGInput

    client: OpenAI = Field(default=None, exclude=True)
    llm_model_name: str = Field(default="openai/gpt-4o-mini", exclude=True)
    google_api_key: str = Field(exclude=True)
    search_engine_id: str = Field(exclude=True)
    openrouter_api_key: str = Field(exclude=True)

    def __init__(self, google_api_key: str, search_engine_id: str, openrouter_api_key: str, **kwargs):
        super().__init__(
            google_api_key=google_api_key,
            search_engine_id=search_engine_id,
            openrouter_api_key=openrouter_api_key,
            **kwargs
        )
        try:
            if not openrouter_api_key or "YOUR_OPENROUTER_API_KEY" in openrouter_api_key:
                raise ValueError("OpenRouter API Key is not set or is a placeholder.")

            client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=openrouter_api_key
            )
            object.__setattr__(self, 'client', client)
        except Exception as e:
            raise ValueError(f"Error configuring OpenAI client for GoogleSearchRAGTool: {e}")

    def _validate_credentials(self) -> bool:
        if not self.google_api_key or "KEY1" in self.google_api_key or "YOUR_GOOGLE_API_KEY" in self.google_api_key:
            return False
        if not self.search_engine_id or "Key2" in self.search_engine_id or "YOUR_SEARCH_ENGINE_ID" in self.search_engine_id:
            return False
        return True

    def _retrieve_context_from_web(self, user_question: str, n_results: int = 5) -> List[Dict[str, str]]:
        if not self._validate_credentials():
            return []

        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            'key': self.google_api_key,
            'cx': self.search_engine_id,
            'q': user_question,
            'num': min(n_results, 10)  # Google API max is 10
        }

        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            search_results = response.json().get('items', [])

            if not search_results:
                return []

            retrieved_data = []
            for item in search_results:
                retrieved_data.append({
                    "title": item.get('title', 'Title Not Available'),
                    "snippet": item.get('snippet', 'Snippet Not Available'),
                    "link": item.get('link', 'Link Not Available'),
                    "source": item.get('displayLink', 'Unknown Source')
                })
            return retrieved_data
        except requests.exceptions.RequestException as e:
            print(f"Error during web search: {e}")
            return []

    def _generate_answer(self, user_question: str, retrieved_data: List[Dict[str, str]]) -> str:
        if not retrieved_data:
            return "I cannot answer the question based on the web search results."

        context_blocks = []
        for i, item in enumerate(retrieved_data, 1):
            context_blocks.append(
                f"""--- Web Search Result {i} ---
Source: {item['source']} - "{item['title']}"
Link: {item['link']}
Content:
{item['snippet']}
--- End of Result {i} ---"""
            )
        context_str = "\n\n".join(context_blocks)

        system_prompt = """You are an expert assistant. Your task is to answer the user's question *exclusively* from the provided web search context.
**CRITICAL RULES:**
1. Only use information from the provided web search context below.
2. If the context doesn't contain relevant information, say so clearly.
3. Be specific and synthesize information from multiple sources when possible.
4. At the end of your response, you MUST cite the full Link for each source you used under a "Sources:" heading."""

        user_prompt = f"""--- CONTEXT FROM WEB SEARCH ---
{context_str}
--- END OF CONTEXT ---
Based on the context above, please answer the following question:
**User's Question: "{user_question}"**"""

        try:
            response = self.client.chat.completions.create(
                model=self.llm_model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=1500
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating response: {str(e)}"

    def _run(self, query: str, n_results: int = 5, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        try:
            print(f" GoogleSearchRAGTool: Starting search for: '{query}'")
            retrieved_data = self._retrieve_context_from_web(query, n_results)
            print(f" GoogleSearchRAGTool: Retrieved {len(retrieved_data)} results")

            if not retrieved_data:
                return "No search results found for your query."

            answer = self._generate_answer(query, retrieved_data)
            print(f" GoogleSearchRAGTool: Generated answer of length {len(answer)}")
            return answer
        except Exception as e:
            error_msg = f"Error in Google Search RAG: {str(e)}"
            print(f" GoogleSearchRAGTool: {error_msg}")
            return error_msg

    async def _arun(self, query: str, n_results: int = 5,
                    run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        return await asyncio.to_thread(self._run, query, n_results=n_results)


def create_google_search_rag_tool(google_api_key: str, search_engine_id: str, openrouter_api_key: str,
                                  **kwargs) -> GoogleSearchRAGTool:
    return GoogleSearchRAGTool(
        google_api_key=google_api_key,
        search_engine_id=search_engine_id,
        openrouter_api_key=openrouter_api_key,
        **kwargs
    )


# --- STACK EXCHANGE TOOL DEFINITION ---

class StackExchangeSearchInput(BaseModel):
    """Input schema for Stack Exchange search tool."""
    query: str = Field(description="The user's programming question, code snippet, or error message to search for.")


class StackExchangeTool(BaseTool):
    """Stack Exchange search tool with multilingual support."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        extra='forbid'
    )

    name: str = "stack_exchange_search"
    description: str = (
        "Use this tool for ALL general programming questions, code snippets, library usage, and error messages "
        "that are NOT related to Oracle databases. Supports queries in any language and returns responses "
        "in the same language as the input. Best source for up-to-date, real-world solutions and code examples."
    )
    args_schema: Type[BaseModel] = StackExchangeSearchInput

    api_key: Optional[str] = Field(default=None, description="Stack Exchange API key")
    base_url: str = Field(default="https://api.stackexchange.com/2.3", description="Base URL for Stack Exchange API")
    llm_model_name: str = Field(default='openai/gpt-4o-mini', description="LLM model for multilingual support")
    confidence_threshold: float = Field(default=0.8, description="Minimum confidence score", ge=0.0, le=1.0)
    similarity_save_threshold: float = Field(
        default=0.75,
        description="Cosine similarity threshold for preventing duplicate answer saves",
        ge=0.0,
        le=1.0
    )
    db_path: str = Field(default="./chroma_db", description="ChromaDB path")

    # Runtime instances
    client: Optional[OpenAI] = Field(default=None, exclude=True)
    reranker: Optional[CrossEncoder] = Field(default=None, exclude=True)
    chroma_client: Optional[Any] = Field(default=None, exclude=True)
    embedder: Optional[Any] = Field(default=None, exclude=True)
    approved_stackoverflow_results_collection: Optional[Any] = Field(default=None, exclude=True)

    def __init__(self, openrouter_api_key: str, stack_exchange_api_key: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)

        if not openrouter_api_key or "YOUR_OPENROUTER_API_KEY" in openrouter_api_key:
            raise ValueError("OpenRouter API key is required.")

        try:
            client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=openrouter_api_key
            )
            object.__setattr__(self, 'client', client)
            object.__setattr__(self, 'api_key', stack_exchange_api_key)
            print("OpenAI client initialized successfully")
        except Exception as e:
            raise ValueError(f"Failed to initialize OpenAI client: {e}")

        self._initialize_reranker()
        self._initialize_chromadb()

    def _initialize_reranker(self) -> None:
        """Initialize the reranker model for result scoring"""
        reranker_model_name = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
        print(f"Loading reranker model: {reranker_model_name}...")

        try:
            reranker = CrossEncoder(reranker_model_name)
            object.__setattr__(self, 'reranker', reranker)
            print("Reranker model loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load reranker model: {e}")
            object.__setattr__(self, 'reranker', None)

    def _initialize_chromadb(self) -> None:
        """Initialize ChromaDB collections for caching"""
        try:
            chroma_client = chromadb.PersistentClient(path=self.db_path)
            object.__setattr__(self, 'chroma_client', chroma_client)

            embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
            object.__setattr__(self, 'embedder', embedder)

            collection = chroma_client.get_or_create_collection(
                name="stackoverflow_approved_results",
                embedding_function=embedder,
                metadata={"hnsw:space": "cosine"}
            )
            object.__setattr__(self, 'approved_stackoverflow_results_collection', collection)
            print("✓ ChromaDB initialized successfully")

        except Exception as e:
            print(f"Warning: ChromaDB initialization failed: {e}")
            object.__setattr__(self, 'chroma_client', None)
            object.__setattr__(self, 'approved_stackoverflow_results_collection', None)

    def _detect_and_get_language_info(self, query: str) -> Dict[str, str]:
        """Detect input language and get language information"""
        print(f"-> Detecting language for query: '{query[:50]}...'")

        detection_prompt = """You are a language detection expert. Analyze the given text and return ONLY a JSON object with this exact format:

{
    "detected_language": "language_name_in_english",
    "language_code": "two_letter_code",
    "confidence": "high|medium|low"
}

Examples:
- Turkish text -> {"detected_language": "Turkish", "language_code": "tr", "confidence": "high"}
- English text -> {"detected_language": "English", "language_code": "en", "confidence": "high"}

If unsure, default to English."""

        try:
            response = self.client.chat.completions.create(
                model=self.llm_model_name,
                messages=[
                    {"role": "system", "content": detection_prompt},
                    {"role": "user", "content": f"Detect language: {query}"}
                ],
                temperature=0.1,
                max_tokens=1500
            )

            result_text = response.choices[0].message.content.strip()
            # Extract JSON from response
            import json

            # Find JSON in the response
            json_start = result_text.find('{')
            json_end = result_text.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                json_text = result_text[json_start:json_end]
                lang_info = json.loads(json_text)
            else:
                # Fallback if JSON parsing fails
                lang_info = {"detected_language": "English", "language_code": "en", "confidence": "low"}

            print(f"   -> Detected: {lang_info['detected_language']} ({lang_info['language_code']})")
            return lang_info

        except Exception as e:
            print(f"   -> Language detection failed, defaulting to English: {e}")
            return {"detected_language": "English", "language_code": "en", "confidence": "low"}

    def _translate_for_search(self, query: str, source_language: str) -> str:
        """Translate query to English for Stack Exchange search if needed"""
        if source_language.lower() in ['english', 'en']:
            print("   -> Query already in English, no translation needed")
            return query

        print(f"-> Translating {source_language} query to English for search")

        translation_prompt = f"""You are a technical translation expert. Translate the following programming-related query from {source_language} to English.

IMPORTANT RULES:
1. Focus on technical terms and programming concepts
2. Keep error messages and code snippets in original form
3. Create a concise, searchable English query
4. Preserve technical accuracy
5. Remove unnecessary words for better search results

Examples:
- "Python listede eleman nasıl aranır?" -> "Python list search element"
- "JavaScript async await error" -> "JavaScript async await error" (already English)

Return ONLY the English translation, no explanations."""

        try:
            response = self.client.chat.completions.create(
                model=self.llm_model_name,
                messages=[
                    {"role": "system", "content": translation_prompt},
                    {"role": "user", "content": query}
                ],
                temperature=0.1,
                max_tokens=200
            )

            english_query = response.choices[0].message.content.strip().strip('"').strip("'")
            print(f"   -> English search terms: '{english_query}'")
            return english_query

        except Exception as e:
            print(f"   -> Translation failed, using original query: {e}")
            return query

    def _search_approved_answers(self, query: str, initial_candidates: int = 10, top_n_after_rerank: int = 5) -> \
    Optional[List[Dict]]:
        """
        Performs a two-stage search in the approved answers database.

        Args:
            query (str): The user's search query.
            initial_candidates (int): The number of potential matches to retrieve from the vector database in the first stage.
            top_n_after_rerank (int): The final number of top results to return after the reranking stage.

        Returns:
            Optional[List[Dict]]: A list of dictionaries, each representing a search result,
                                 sorted by rerank_score. Returns None if no results are found
                                 or if the database/reranker is unavailable.
        """
        if not self.approved_stackoverflow_results_collection or not self.reranker:
            print("   -> Vector database or reranker not available for searching.")
            return None

        print(f"-> Starting two-stage search for: '{query[:50]}...'")

        try:
            # Stage 1: Retrieve initial candidates using cosine similarity
            print(f"   -> Stage 1: Retrieving up to {initial_candidates} candidates with cosine similarity...")
            results = self.approved_stackoverflow_results_collection.query(
                query_texts=[query],
                n_results=initial_candidates,
                include=['documents', 'distances', 'metadatas']
            )

            if not results or not results.get('documents') or not results['documents'][0]:
                print("   -> No cached results found in the vector database.")
                return None

            documents = results['documents'][0]
            distances = results.get('distances', [[]])[0] if results.get('distances') else []
            metadatas = results.get('metadatas', [[]])[0] if results.get('metadatas') else []

            found_count = len(documents)
            print(f"   -> Found {found_count} potential matches in the vector database.")

            if found_count == 0:
                return None

            # Stage 2: Rerank the retrieved candidates to find the best matches
            print(f"   -> Stage 2: Reranking the {found_count} candidates to select the best {top_n_after_rerank}...")
            pairs = [[query, doc] for doc in documents]
            rerank_scores = self.reranker.predict(pairs, show_progress_bar=False)

            # Combine initial results with their new rerank scores
            combined_results = []
            similarities = [1 - dist for dist in distances] if distances else [0] * found_count
            for i, (doc, cosine_sim, rerank_score) in enumerate(zip(documents, similarities, rerank_scores)):
                metadata = metadatas[i] if i < len(metadatas) else {}
                combined_results.append({
                    'document': doc,
                    'cosine_similarity': cosine_sim,
                    'rerank_score': float(rerank_score),
                    'metadata': metadata,
                    'original_index': i + 1
                })

            # Sort by the new rerank score in descending order
            combined_results.sort(key=lambda x: x['rerank_score'], reverse=True)

            # Return the top N results after reranking
            top_reranked_results = combined_results[:top_n_after_rerank]

            print(f"   -> Top {len(top_reranked_results)} results after reranking:")
            for i, result in enumerate(top_reranked_results):
                original_query_preview = result['metadata'].get('original_query', 'N/A')[:50]
                print(f"      -> Rank {i + 1}: Rerank Score = {result['rerank_score']:.3f} | "
                      f"Cosine Sim = {result['cosine_similarity']:.3f} | "
                      f"Original Query: '{original_query_preview}...''")

            return top_reranked_results

        except Exception as e:
            print(f"   -> An error occurred during the two-stage search: {e}")
            return None

    def _retrieve_from_stackoverflow(self, english_query: str) -> List[Dict]:
        """Fetch search results from Stack Exchange API"""
        print(f"-> Retrieving from Stack Overflow API: '{english_query}'")
        params = {
            'q': english_query,
            'site': 'stackoverflow',
            'sort': 'relevance',
            'pagesize': 5,
            'filter': 'withbody'
        }
        if self.api_key:
            params['key'] = self.api_key

        try:
            response = requests.get("https://api.stackexchange.com/2.3/search/advanced", params=params)
            response.raise_for_status()
            items = response.json().get('items', [])

            if not items:
                print("   -> No results found from Stack Overflow API")
                return []

            results = []
            for item in items:
                content = re.sub('<[^<]+?>', '', item.get('body', ''))
                if len(content) > 2000:
                    content = content[:2000] + "..."

                results.append({
                    "title": item.get('title', 'No Title'),
                    "score": item.get('score', 0),
                    "link": item.get('link', ''),
                    "content": content
                })

            print(f"   -> Retrieved {len(results)} posts from Stack Overflow")
            return results

        except Exception as e:
            print(f"   -> Error retrieving from Stack Overflow API: {e}")
            return []

    def _generate_answer(self, original_query: str, context_posts: List[Dict], target_language: str,
                         language_code: str) -> str:
        """Generate answer in the original language of the query"""
        print(f"-> Generating answer in {target_language} from {len(context_posts)} posts")

        if not context_posts:
            return f"No relevant Stack Overflow posts were found for this query." if target_language == "English" else "Bu sorgu için Stack Overflow'da ilgili sonuç bulunamadı."

        context_str = "\n\n".join([
            f"""--- Stack Overflow Post ---
Title: {post['title']}
Score: {post['score']} upvotes
Link: {post['link']}
Content:
{post['content']}
--- End of Post ---"""
            for post in context_posts
        ])

        if target_language.lower() in ['english', 'en']:
            system_prompt = """You are an expert programmer. Answer the user's question by synthesizing information exclusively from the provided Stack Overflow context.

Rules:
1. Use only information from the Stack Overflow posts provided
2. Synthesize information from multiple posts when possible
3. Include relevant code examples from the context
4. Provide practical, actionable solutions
5. At the end, cite the links under a "Sources:" heading"""

            user_prompt = f"""--- STACK OVERFLOW CONTEXT ---
{context_str}
--- END CONTEXT ---

Based only on the above context, answer: "{original_query}" """

        else:
            system_prompt = f"""You are an expert programmer who speaks {target_language} fluently. Answer the user's question in {target_language} by synthesizing information exclusively from the provided Stack Overflow context.

CRITICAL RULES:
1. Answer in {target_language} language only
2. Use only information from the Stack Overflow posts provided
3. Synthesize information from multiple posts when possible
4. Include relevant code examples from the context (code should remain in original language)
5. Provide practical, actionable solutions in {target_language}
6. At the end, cite the links under a "Sources:" or equivalent heading in {target_language}

The user asked in {target_language}, so your entire response must be in {target_language}."""

            user_prompt = f"""--- STACK OVERFLOW CONTEXT ---
{context_str}
--- END CONTEXT ---

Based only on the above context, answer this question in {target_language}: "{original_query}" """

        try:
            response = self.client.chat.completions.create(
                model=self.llm_model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=1500
            )

            answer = response.choices[0].message.content
            print(f"   -> Generated {len(answer)} character answer in {target_language}")
            return answer

        except Exception as e:
            error_msg = f"Error generating answer: {e}"
            print(f"   -> {error_msg}")
            return error_msg

    def _save_approved_answer(self, original_query: str, answer: str, language_info: Dict) -> bool:
        """
        Save approved answer to cache after checking for exact duplicates and similarity.
        Uses content hashing to prevent exact duplicates and similarity checking for near-duplicates.

        Returns:
            bool: True if answer was saved, False if duplicate was found or error occurred
        """
        if not self.approved_stackoverflow_results_collection:
            print("Warning: Database collection not available for saving")
            return False

        try:

            print(f"-> Checking for duplicates before saving: '{original_query[:50]}...'")

            # Step 1: Create content hash for exact duplicate detection
            normalized_query = original_query.strip().lower()
            normalized_answer = answer.strip()
            content_hash = hashlib.sha256(f"{normalized_query}|{normalized_answer}".encode('utf-8')).hexdigest()

            print(f"   -> Generated content hash: {content_hash[:16]}...")

            # Step 2: Check for exact duplicates using content hash
            try:
                existing_exact = self.approved_stackoverflow_results_collection.get(
                    where={"content_hash": content_hash}
                )

                if existing_exact and len(existing_exact.get('ids', [])) > 0:
                    print(f"   -> Exact duplicate found with hash {content_hash[:16]}... - not saving")
                    return False

            except Exception as e:
                print(f"   -> Warning: Could not check exact duplicates: {e}")
                # Continue with similarity check if exact duplicate check fails

            # Step 3: Check for near-duplicates using similarity search
            existing_results = self._search_approved_answers(original_query)

            if existing_results:
                # Check the top result's cosine similarity
                top_similarity = existing_results[0]['cosine_similarity']
                print(f"   -> Found existing answer with similarity: {top_similarity:.3f}")

                # If similarity is above threshold, don't save
                if top_similarity >= self.similarity_save_threshold:
                    print(
                        f"   -> Skipping save - similarity ({top_similarity:.3f}) >= threshold ({self.similarity_save_threshold})")
                    return False

                # Additional check: Compare the actual answer content for very high similarity
                # This handles cases where questions are different but answers are nearly identical
                if self.reranker:
                    try:
                        existing_answer = existing_results[0]['document']
                        answer_similarity_score = self.reranker.predict([[answer, existing_answer]])[0]
                        print(f"   -> Answer content similarity score: {answer_similarity_score:.3f}")

                        # If answers are very similar (threshold can be adjusted)
                        answer_similarity_threshold = 0.85
                        if answer_similarity_score >= answer_similarity_threshold:
                            print(
                                f"   -> Skipping save - answer content too similar ({answer_similarity_score:.3f} >= {answer_similarity_threshold})")
                            return False

                    except Exception as e:
                        print(f"   -> Warning: Could not check answer similarity: {e}")

            # Step 4: No duplicates found, proceed with saving
            doc_id = str(uuid.uuid4())
            self.approved_stackoverflow_results_collection.add(
                ids=[doc_id],
                documents=[answer],
                metadatas=[{
                    "original_query": original_query,
                    "language": language_info.get('detected_language', 'Unknown'),
                    "language_code": language_info.get('language_code', 'unknown'),
                    "approval_timestamp": time.time(),
                    "source": "stackoverflow_approved_results",
                    "doc_id": doc_id,
                    "content_hash": content_hash  # Store content hash for exact duplicate detection
                }]
            )

            print(f"   -> Successfully saved new answer to cache (ID: {doc_id}, Hash: {content_hash[:16]}...)")
            return True

        except Exception as e:
            print(f"Warning: Could not save to cache due to error: {e}")
            return False

    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        """Main execution method"""
        try:
            print(f"\n{'=' * 60}")
            print(f"Processing Stack Exchange search")
            print(f"{'=' * 60}")

            # Step 1: Detect language
            language_info = self._detect_and_get_language_info(query)

            # Step 2: Check cache first
            cached_results = self._search_approved_answers(query)
            if cached_results and cached_results[0]['rerank_score'] >= self.confidence_threshold:
                top_result = cached_results[0]
                print(f"-> Returning cached result with high confidence ({top_result['rerank_score']:.2f})")
                return top_result['document']

            # Step 3: Translate to English for search if needed
            english_query = self._translate_for_search(query, language_info['detected_language'])

            # Step 4: Search Stack Overflow
            posts = self._retrieve_from_stackoverflow(english_query)
            if not posts:
                no_results_msg = {
                    'tr': "Bu sorgu için Stack Overflow'da ilgili sonuç bulunamadı.",
                    'en': "No relevant Stack Overflow results found for this query."
                }
                return no_results_msg.get(language_info['language_code'], no_results_msg['en'])

            # Step 5: Generate answer in original language
            answer = self._generate_answer(
                query, posts,
                language_info['detected_language'],
                language_info['language_code']
            )

            return answer

        except Exception as e:
            error_msg = f"Error in Stack Exchange tool: {str(e)}"
            print(f"ERROR: {error_msg}")
            return error_msg

    async def _arun(self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        """Async version of the run method"""
        return await asyncio.to_thread(self._run, query=query)


# --- INTERNAL KNOWLEDGE SEARCH TOOL DEFINITION ---

class InternalKnowledgeSearchInput(BaseModel):
    """Input schema for the Internal Knowledge Search tool."""
    query: str = Field(description="The user's technical or programming question.")
    n_results: int = Field(default=3, description="The number of relevant documents to retrieve.")


class InternalKnowledgeSearchTool(BaseTool):
    """A tool that answers technical questions by searching a curated vector database."""
    name: str = "internal_knowledge_search"
    description: str = (
        "Use this tool FIRST for ANY question related to Oracle Database, SQL, PL/SQL, database performance, "
        "or specific ORA-error codes. This is a mandatory first step for all Oracle-related queries as it contains "
        "trusted, curated information."
    )
    args_schema: Type[BaseModel] = InternalKnowledgeSearchInput

    client: OpenAI = Field(default=None, exclude=True)
    collection: Any = Field(default=None, exclude=True)
    llm_model_name: str = Field(default="openai/gpt-4o-mini", exclude=True)

    def __init__(self, openrouter_api_key: str, db_path: str = "./chroma_db",
                 collection_name: str = "stackoverflow_results", llm_model_name: str = "openai/gpt-4o-mini", **kwargs):
        super().__init__(**kwargs)

        # 1. Set up the LLM Client
        try:
            if not openrouter_api_key or "YOUR_KEY" in openrouter_api_key:
                raise ValueError("OpenRouter API Key is not set or is a placeholder.")

            llm_client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=openrouter_api_key)
            object.__setattr__(self, 'client', llm_client)
            object.__setattr__(self, 'llm_model_name', llm_model_name)
            print(" InternalKnowledgeTool: OpenRouter client configured successfully.")

        except Exception as e:
            raise ValueError(f"Error initializing OpenAI client for InternalKnowledgeSearchTool: {e}")

        # 2. Set up the ChromaDB Client (Retriever)
        try:
            print(" InternalKnowledgeTool: Connecting to ChromaDB...")
            embedder = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
            db_client = chromadb.PersistentClient(path=db_path)
            collection = db_client.get_collection(name=collection_name, embedding_function=embedder)
            object.__setattr__(self, 'collection', collection)
            print(f" InternalKnowledgeTool: Successfully connected to ChromaDB collection '{collection_name}'.")

        except Exception as e:
            raise ValueError(f"Fatal Error: Could not connect to ChromaDB for InternalKnowledgeSearchTool: {e}")

    def _retrieve_context(self, user_question: str, n_results: int) -> List[Dict[str, str]]:
        """
        Retrieves the most relevant Question and Answer pairs from ChromaDB
        based on the user's question, displaying only the titles of retrieved sources.
        """
        print(f"1. [RETRIEVAL] Searching database for context related to: '{user_question}'")
        if not self.collection:
            print("   -> Database collection is not available. Cannot retrieve context.")
            return []

        results = self.collection.query(
            query_texts=[user_question],
            n_results=n_results,
            include=["documents", "metadatas"]
        )

        retrieved_data = []
        if not results or not results["documents"][0]:
            print("   -> No relevant documents found in the database for this query.")
            return []

        print(f"   -> Found {len(results['documents'][0])} potential documents. Processing...")

        # Process each retrieved document
        for i, doc in enumerate(results["documents"][0]):
            doc_parts = doc.split('\n\n')
            question_text = "No question text found in document."
            answer_text = "No accepted answer found in document."

            for part in doc_parts:
                if part.startswith('Question:'):
                    question_text = part.replace('Question:', '', 1).strip()
                elif part.startswith('Answer:'):
                    answer_text = part.replace('Answer:', '', 1).strip()

            # We only process documents that have a valid answer
            if answer_text and "No accepted answer found" not in answer_text:
                metadata = results['metadatas'][0][i]
                title = metadata.get('title', 'Title Not Available')

                print(f"   -> Retrieved source: '{title}'")

                retrieved_data.append({"question": question_text, "answer": answer_text, "title": title})

        print(f"   -> Successfully extracted {len(retrieved_data)} valid contexts to be used for answer generation.")
        return retrieved_data

    def _generate_answer(self, user_question: str, retrieved_data: List[Dict[str, str]]) -> str:
        """
        Generates a final answer by sending the user's question and the retrieved
        database context to the LLM, with improved instructions for synthesis.
        """
        print("\n2. [GENERATION] Sending question and context to the AI...")
        if not retrieved_data:
            print("   -> No context was found in the database to answer the question.")
            return "I cannot answer the question based on the information available in the database."

        # Build the context string from the retrieved data
        context_blocks = []
        for i, item in enumerate(retrieved_data, 1):
            context_blocks.append(
                f"""--- Context From Database {i} ---
    Source Title: "{item['title']}"
    Source Question: "{item['question']}"

    Answer Provided:
    {item['answer']}
    --- End of Context {i} ---"""
            )
        context_str = "\n\n".join(context_blocks)

        # A more flexible system prompt that encourages synthesis and helpfulness
        system_prompt = """You are a helpful expert assistant. Your task is to answer the user's question by synthesizing information *exclusively* from the provided context below.

    **CRITICAL INSTRUCTIONS:**
    1. Base your answer **primarily** on the information within the "Context From Database" sections. Do not use any of your own prior knowledge.
    2. Synthesize information from multiple sources if possible to provide a comprehensive and helpful answer.
    3. If the context provides related information but not a direct answer to the user's specific question, **summarize what the context *does* say and explain how it might be relevant**. It is better to provide related, helpful information from the context than to simply state you cannot answer.
    4. If the context is completely irrelevant and cannot help answer the question in any way, then and only then should you state that you cannot find a relevant answer in the database.
    5. At the end of your response, you **MUST** cite the original "Source Title" for each piece of information you used. List them clearly under a "Sources:" heading.
    """

        user_prompt = f"""--- CONTEXT FROM DATABASE ---\n{context_str}\n--- END OF CONTEXT ---\n\nBased on the rules and context above, please answer the following question.\n\n**User's Question: "{user_question}"**"""

        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

        try:
            response = self.client.chat.completions.create(
                model=self.llm_model_name,
                messages=messages,
                temperature=0.1,
                max_tokens=1500
            )
            print("   -> AI call succeeded.")
            generated_text = response.choices[0].message.content
            return generated_text if generated_text else "The model returned an empty response."

        except Exception as e:
            print(f"   -> An error occurred during AI generation: {e}")
            return "Sorry, an error occurred while generating a response from the AI model."

    def _run(self, query: str, n_results: int = 3, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        """
        The main execution function for the tool.
        It orchestrates the retrieval and generation steps.
        """
        # Step 1: Retrieve relevant context from the database.
        retrieved_data = self._retrieve_context(query, n_results)

        # Step 2: Generate a final answer using only the retrieved context.
        final_answer = self._generate_answer(query, retrieved_data)

        return final_answer

    async def _arun(self, query: str, n_results: int = 3,
                    run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        # For simplicity, we'll just use the synchronous version in an async context.
        # For a production environment, you might want to use async libraries for I/O.
        return self._run(query, n_results=n_results)


# --- GRADIO INTERFACE AND LOGIC ---
def run_gradio_interface(agent_executor, tools):
    """
    Runs the agent in an interactive Gradio web interface.
    """
    with gr.Blocks(theme=gr.themes.Soft(), title="Programming Assistant") as demo:
        # Session state to store the conversation history in LangChain message format
        state = gr.State([SystemMessage(content=SYSTEM_PROMPT)])

        gr.Markdown(
            """
            # Specialized Programming Assistant
            Ask any programming question. The assistant will use specialized tools to find the best answer.
            - For **Oracle, SQL, PL/SQL**, it uses an internal knowledge base.
            - For **general programming questions**, it searches Stack Exchange.
            - For **recent or real-time info**, it uses Google Search.
            """
        )

        chatbot = gr.Chatbot(label="Conversation", height=500, type="messages")

        with gr.Row():
            with gr.Column(scale=12):
                msg = gr.Textbox(
                    label="Your Question",
                    placeholder="e.g., How to solve ORA-00942? or How to iterate over a list in Python?",
                    show_label=False,
                    container=False
                )
            with gr.Column(scale=1, min_width=50):
                submit_btn = gr.Button("➤", variant="primary")

        with gr.Row():
            clear_btn = gr.Button("🗑️ Clear Conversation")
            with gr.Column(scale=1, min_width=100):
                with gr.Row():
                    thumb_up_btn = gr.Button("👍 Helpful")
                    thumb_down_btn = gr.Button("👎 Not Helpful")

        def _convert_to_gradio_messages(history: List[AnyMessage]):
            """Converts LangChain messages to a list of dicts for Gradio Chatbot."""
            messages = []
            for msg in history:
                if isinstance(msg, HumanMessage):
                    messages.append({"role": "user", "content": msg.content})
                elif isinstance(msg, AIMessage):
                    content = msg.content if isinstance(msg.content, str) else "Calling tools..."
                    messages.append({"role": "assistant", "content": content})
            return messages

        def respond(message, messages_state):
            """Main function to handle user input and agent response."""
            print(f"\n{'=' * 50}")
            print(f"Processing: \"{message}\"")

            # Add user message to state
            messages_state.append(HumanMessage(content=message))
            gradio_display_history = _convert_to_gradio_messages(messages_state)
            # Return empty string for msg input to clear it
            yield gradio_display_history, messages_state, ""

            # Show "thinking" message BEFORE agent execution
            messages_state.append(AIMessage(content="Thinking and searching for the best answer..."))
            gradio_display_history = _convert_to_gradio_messages(messages_state)
            yield gradio_display_history, messages_state, ""

            try:
                # Remove the "thinking" message before invoking agent
                messages_state.pop()  # Remove the temporary thinking message

                final_state = agent_executor.invoke({"messages": messages_state}, {"recursion_limit": 10})
                messages_state = final_state['messages']
            except Exception as e:
                # Remove the "thinking" message in case of error too
                messages_state.pop()  # Remove the temporary thinking message
                error_message = f"An error occurred: {e}"
                messages_state.append(AIMessage(content=error_message))
                print(f"ERROR: {error_message}")

            print("-" * 50)
            final_answer = messages_state[-1].content
            print(f"Final Answer:\n{final_answer}")
            print("=" * 50 + "\n")

            gradio_display_history = _convert_to_gradio_messages(messages_state)
            # Return empty string for msg input to keep it cleared
            yield gradio_display_history, messages_state, ""

        def process_feedback(messages_state, feedback_type):
            """Handles user feedback, checks for duplicates before saving helpful answers."""
            if not messages_state or len(messages_state) < 2:
                gr.Warning("No conversation to provide feedback on.")
                return

            last_user_msg = next((msg for msg in reversed(messages_state) if isinstance(msg, HumanMessage)), None)
            last_ai_msg = next((msg for msg in reversed(messages_state) if isinstance(msg, AIMessage)), None)

            if not last_user_msg or not last_ai_msg:
                gr.Warning("Could not find a valid question/answer pair to save.")
                return

            if feedback_type == "helpful":
                stack_exchange_tool = next((t for t in tools if isinstance(t, StackExchangeTool)), None)

                if stack_exchange_tool and isinstance(last_ai_msg.content, str):
                    print("-> User marked answer as helpful. Checking for similar answers before saving...")

                    # Step 1: Search for existing answers using the user's question.
                    # We use the cosine_similarity score as it's a direct measure of semantic similarity.
                    cached_results = stack_exchange_tool._search_approved_answers(last_user_msg.content)

                    # Step 2: Check if a highly similar answer already exists.
                    if (cached_results and
                            cached_results[0]['cosine_similarity'] >= stack_exchange_tool.similarity_save_threshold):

                        similarity_score = cached_results[0]['cosine_similarity']
                        print(f"   -> Found a highly similar answer (Score: {similarity_score:.3f}). Skipping save.")
                        gr.Info(
                            f"Thank you for the feedback! A very similar answer already exists, so it wasn't saved again.")

                    else:
                        # Step 3: If no duplicate is found, proceed with saving.
                        print("   -> No highly similar answer found. Proceeding with save.")
                        try:
                            language_info = stack_exchange_tool._detect_and_get_language_info(last_user_msg.content)
                            stack_exchange_tool._save_approved_answer(last_user_msg.content, last_ai_msg.content,
                                                                      language_info)
                            gr.Info("Thank you! This helpful answer has been saved for future reference.")
                        except Exception as e:
                            gr.Warning(f"Could not save the answer: {e}")
                else:
                    gr.Warning("StackExchange tool not found or AI response was not text. Cannot save answer.")

            else:  # Feedback was "not_helpful"
                gr.Info("Thank you for your feedback. The answer was not saved.")

        # Wire up components - MODIFIED: Added msg as output to clear the input
        msg.submit(respond, [msg, state], [chatbot, state, msg])
        submit_btn.click(respond, [msg, state], [chatbot, state, msg])

        # Clear button logic - MODIFIED: Added msg clearing
        def clear_chat():
            return None, [SystemMessage(content=SYSTEM_PROMPT)], ""

        clear_btn.click(clear_chat, None, [chatbot, state, msg], queue=False)

        # Feedback buttons logic
        thumb_up_btn.click(lambda s: process_feedback(s, "helpful"), [state], None, queue=False)
        thumb_down_btn.click(lambda s: process_feedback(s, "not_helpful"), [state], None, queue=False)

    # Launch the Gradio app
    print("\nProgramming Assistant Ready! Launching Gradio interface...")
    demo.queue()
    demo.launch()


# --- MAIN EXECUTION BLOCK ---

if __name__ == "__main__":
    # Disable LangSmith tracing
    os.environ["LANGCHAIN_TRACING_V2"] = "false"

    # Load environment variables from .env file
    load_dotenv()

    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    STACK_EXCHANGE_API_KEY = os.getenv("STACK_EXCHANGE_API_KEY")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    SEARCH_ENGINE_ID = os.getenv("SEARCH_ENGINE_ID")

    # Validate required API key
    if not OPENROUTER_API_KEY:
        raise ValueError("FATAL: OPENROUTER_API_KEY environment variable must be set.")

    # --- Create and configure tools ---
    tools = []

    # 1. Stack Exchange Tool
    tools.append(
        StackExchangeTool(stack_exchange_api_key=STACK_EXCHANGE_API_KEY, openrouter_api_key=OPENROUTER_API_KEY))
    print(" Created Stack Exchange tool")

    # 2. Internal Knowledge Tool
    try:
        tools.append(InternalKnowledgeSearchTool(openrouter_api_key=OPENROUTER_API_KEY))
        print(" Created Internal Knowledge Search tool")
    except Exception as e:
        print(f"️  Warning: Could not initialize internal knowledge tool: {e}")

    # 3. Google Search RAG Tool
    if GOOGLE_API_KEY and SEARCH_ENGINE_ID:
        try:
            tools.append(create_google_search_rag_tool(
                google_api_key=GOOGLE_API_KEY,
                search_engine_id=SEARCH_ENGINE_ID,
                openrouter_api_key=OPENROUTER_API_KEY
            ))
            print(" Created Google Search RAG tool")
        except Exception as e:
            print(f"  Warning: Could not initialize Google Search RAG tool: {e}")
    else:
        print("️  Warning: GOOGLE_API_KEY or SEARCH_ENGINE_ID not found. Google Search tool will be disabled.")

    print(f" Total tools loaded: {len(tools)}")

    llm = ChatOpenAI(
        model="openai/gpt-4o-mini",
        temperature=0.1,
        max_tokens=1500,
        api_key=OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1"
    )

    # CRITICAL FIX: Bind tools to the LLM
    llm_with_tools = llm.bind_tools(tools)
    print(" LLM initialized and bound with tools successfully")


    class AgentState(TypedDict):
        messages: Annotated[list[AnyMessage], operator.add]


    # Create the agent graph
    graph_builder = StateGraph(AgentState)


    def agent_node(state: AgentState):
        """Invokes the LLM to decide the next action."""
        messages = state['messages']
        # Ensure the system message is always first
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages

        # Use the LLM with bound tools
        response = llm_with_tools.invoke(messages)

        # Debug: Print tool calls if any
        if hasattr(response, 'tool_calls') and response.tool_calls:
            print(f" Agent is calling tools: {[call['name'] for call in response.tool_calls]}")
        else:
            print("️  Agent did not call any tools - this may indicate an issue")

        return {"messages": [response]}


    graph_builder.add_node("agent", agent_node)

    # 2. Define the tool node
    tool_node = ToolNode(tools)
    graph_builder.add_node("tools", tool_node)

    # 3. Define the graph's edges
    graph_builder.set_entry_point("agent")
    graph_builder.add_conditional_edges("agent", tools_condition)
    graph_builder.add_edge("tools", "agent")

    # 4. Compile the graph into a runnable executor
    agent_executor = graph_builder.compile()
    print(" LangGraph Agent initialized successfully with tool binding")

    # Start the Gradio interface instead of the console loop
    run_gradio_interface(agent_executor, tools)