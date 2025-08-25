import os
import requests
from openai import OpenAI
import asyncio
from typing import Optional, List, Dict, Any, Type, TypedDict, Annotated
from pydantic import BaseModel, Field
from langchain.tools import BaseTool
from langchain.callbacks.manager import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun

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
    query: str = Field(description="The search query for Stack Exchange")
    site: str = Field(default="stackoverflow", description="Stack Exchange site to search")
    tagged: Optional[str] = Field(default=None, description="Semicolon delimited list of tags")
    sort: str = Field(default="relevance", description="Sort order: activity, votes, creation, relevance")
    order: str = Field(default="desc", description="Sort direction: desc or asc")
    pagesize: int = Field(default=10, description="Number of results to return (max 100)")


class StackExchangeTool(BaseTool):
    """Tool for searching Stack Exchange sites."""
    name: str = "stack_exchange_search"
    description: str = (
        "Use this tool for ALL general programming questions, code snippets, library usage, and error messages "
        "that are NOT related to Oracle databases. It is the best source for finding up-to-date, real-world solutions "
        "and code examples for languages like Python, JavaScript, Java, etc."
    )
    args_schema: type = StackExchangeSearchInput
    api_key: Optional[str] = Field(default=None, description="Stack Exchange API key")
    base_url: str = Field(default="https://api.stackexchange.com/2.3", description="Base URL for Stack Exchange API")

    def _run(self, query: str, site: str = "stackoverflow", tagged: Optional[str] = None, sort: str = "relevance",
             order: str = "desc", pagesize: int = 10, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        try:
            print(f" StackExchangeTool: Searching for: '{query}' on {site}")
            params = {'q': query, 'site': site, 'sort': sort, 'order': order, 'pagesize': min(pagesize, 100),
                      'filter': 'withbody'}
            if tagged: params['tagged'] = tagged
            if self.api_key: params['key'] = self.api_key

            response = requests.get(f"{self.base_url}/search/advanced", params=params)
            response.raise_for_status()
            data = response.json()

            if 'items' not in data or not data['items']:
                print(f" StackExchangeTool: No results found for query: '{query}' on {site}")
                return f"No results found for query: '{query}' on {site}"

            results = []
            for item in data['items'][:pagesize]:
                body_text = re.sub('<[^<]+?>', '', item.get('body', ''))
                excerpt = body_text[:200] + '...' if len(body_text) > 200 else body_text
                results.append(
                    f"Title: {item.get('title', 'No title')}\nScore: {item.get('score', 0)} | Answers: {item.get('answer_count', 0)} | Answered: {item.get('is_answered', False)}\nTags: {', '.join(item.get('tags', []))}\nLink: {item.get('link', '')}\nQuestion: {excerpt}\n---\n")

            result_text = f"Found {len(results)} results on {site}:\n" + "".join(results)
            print(f" StackExchangeTool: Retrieved {len(results)} results")
            return result_text
        except requests.exceptions.RequestException as e:
            error_msg = f"Error making API request: {str(e)}"
            print(f" StackExchangeTool: {error_msg}")
            return error_msg
        except Exception as e:
            error_msg = f"Error processing Stack Exchange search: {str(e)}"
            print(f" StackExchangeTool: {error_msg}")
            return error_msg

    async def _arun(self, query: str, **kwargs) -> str:
        return await asyncio.to_thread(self._run, query, **kwargs)


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
    tools.append(StackExchangeTool(api_key=STACK_EXCHANGE_API_KEY))
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

    # FIXED: Initialize LLM with proper tool binding
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
    graph_builder.add_conditional_edges("agent", tools_condition)  # Automatically routes to "tools" or END
    graph_builder.add_edge("tools", "agent")

    # 4. Compile the graph into a runnable executor
    agent_executor = graph_builder.compile()
    print(" LangGraph Agent initialized successfully with tool binding")


    def run_interactive(agent_executor):
        """
        Runs the agent in an interactive loop with proper conversation history management.
        """
        print("\n Programming Assistant Ready! Type 'quit' or 'exit' to stop.\n")
        messages: List[AnyMessage] = [SystemMessage(content=SYSTEM_PROMPT)]

        while True:
            try:
                user_input = input(" Your question: ").strip()

                if user_input.lower() in ['quit', 'exit', 'q']:
                    print(" Goodbye!")
                    break
                if not user_input:
                    continue

                print("\n" + "=" * 50)
                print(f" Processing: \"{user_input}\"")
                messages.append(HumanMessage(content=user_input))
                initial_state = {"messages": messages}
                print("  Agent is thinking...\n")

                # Invoke the agent graph
                final_state = agent_executor.invoke(initial_state, {"recursion_limit": 10})
                messages = final_state['messages']

                # Get the final response
                final_message = messages[-1]
                if isinstance(final_message, AIMessage):
                    final_answer = final_message.content
                else:
                    final_answer = str(final_message.content)

                print("-" * 50)
                print(f" Final Answer:\n{final_answer}")
                print("=" * 50 + "\n")

            except KeyboardInterrupt:
                print("\n Goodbye!")
                break
            except Exception as e:
                print(f"\n An unexpected error occurred: {e}")
                print(f"Error type: {type(e).__name__}")
                import traceback
                traceback.print_exc()
                messages = [SystemMessage(content=SYSTEM_PROMPT)]
                print("\n Conversation history has been reset due to an error. Please try again.")


    # Start the interactive loop
    run_interactive(agent_executor)