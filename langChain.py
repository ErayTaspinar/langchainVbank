import os
import tempfile

import datetime
from contextlib import contextmanager

import psycopg
import requests
import timedelta
from openai import OpenAI
import asyncio
import uuid, time
import hashlib
import gradio as gr
import base64
from typing import Optional, List, Dict, Any, Type, TypedDict, Annotated

from psycopg.rows import dict_row
from pydantic import BaseModel, Field, ConfigDict
from langchain.tools import BaseTool
from langchain.callbacks.manager import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun
from sentence_transformers import CrossEncoder, SentenceTransformer

import re
import chromadb
from dotenv import load_dotenv

from langchain.memory import ConversationBufferMemory
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage, AIMessage, BaseMessage
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


class PostgreSQLMemoryBuffer:
    """PostgreSQL-based conversation memory buffer with ConversationBufferMemory integration."""

    def __init__(self,
                 connection_params: Dict[str, Any],
                 max_messages_per_session: int = 50,
                 cleanup_days: int = 7,
                 max_token_limit: int = 2000):
        """
        Initialize PostgreSQL memory buffer with ConversationBufferMemory integration.

        Args:
            connection_params: Dict with keys: host, database, user, password, port.
            max_messages_per_session: Maximum messages to keep per session.
            cleanup_days: Days after which to clean up old sessions.
            max_token_limit: Token limit for ConversationBufferMemory.
        """
        self.connection_params = connection_params
        self.max_messages_per_session = max_messages_per_session
        self.cleanup_days = cleanup_days
        self.max_token_limit = max_token_limit

        # Dictionary to store ConversationBufferMemory instances for each session
        self.session_memories: Dict[str, ConversationBufferMemory] = {}

        self._initialize_database()

    @contextmanager
    def get_db_connection(self):
        """Context manager for database connections."""
        conn = None
        try:
            conn = psycopg.connect(**self.connection_params)
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            print(f"Database connection error: {e}")
            raise
        finally:
            if conn:
                conn.close()

    def _initialize_database(self):
        """Verify connection and that the required tables exist."""
        try:
            with self.get_db_connection() as conn:
                with conn.cursor() as cur:
                    # Check for the existence of the main tables
                    cur.execute("SELECT to_regclass('public.unique_ids')")
                    if cur.fetchone()[0] is None:
                        raise RuntimeError("Database schema error: 'unique_ids' table not found.")

                    cur.execute("SELECT to_regclass('public.chats')")
                    if cur.fetchone()[0] is None:
                        raise RuntimeError("Database schema error: 'chats' table not found.")

                    cur.execute("SELECT to_regclass('public.images')")
                    if cur.fetchone()[0] is None:
                        raise RuntimeError("Database schema error: 'images' table not found.")

                    cur.execute("SELECT to_regclass('public.chat_images')")
                    if cur.fetchone()[0] is None:
                        raise RuntimeError("Database schema error: 'chat_images' table not found.")

                    print("Database schema verified successfully.")
        except Exception as e:
            raise Exception(f"Failed to initialize and verify database connection: {e}")

    def get_langchain_memory(self, session_id: str) -> ConversationBufferMemory:
        """Get or create ConversationBufferMemory for a session"""
        if session_id not in self.session_memories:
            memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                max_token_limit=self.max_token_limit
            )

            # Load existing messages from PostgreSQL into the buffer
            existing_messages = self.get_messages_as_langchain(session_id)
            for msg in existing_messages:
                if isinstance(msg, HumanMessage):
                    content = self._extract_text_from_message(msg)
                    memory.chat_memory.add_user_message(content)
                elif isinstance(msg, AIMessage):
                    memory.chat_memory.add_ai_message(str(msg.content))

            self.session_memories[session_id] = memory
            print(f"Created ConversationBufferMemory for session {session_id}")

        return self.session_memories[session_id]

    def _extract_text_from_message(self, message: BaseMessage) -> str:
        """Extract text content from potentially multimodal message"""
        if isinstance(message.content, list):
            text_parts = []
            for item in message.content:
                if isinstance(item, dict) and item.get('type') == 'text':
                    text_parts.append(item.get('text', ''))
            return ' '.join(text_parts)
        return str(message.content)

    def get_memory_variables(self, session_id: str) -> Dict[str, Any]:
        """Get LangChain memory variables for a session"""
        memory = self.get_langchain_memory(session_id)
        return memory.load_memory_variables({})

    def ensure_session_exists(self, session_id: str) -> None:
        """Ensure session ID exists in the unique_ids table."""
        with self.get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO unique_ids (unique_id) VALUES (%s) ON CONFLICT (unique_id) DO NOTHING",
                    (session_id,)
                )
                conn.commit()

    def save_message(self, session_id: str, message: BaseMessage, image_path: Optional[str] = None) -> int:
        """
        Save a message and its image associations to both PostgreSQL and ConversationBufferMemory.

        Args:
            session_id: The unique identifier for the conversation session.
            message: The LangChain message object (e.g., HumanMessage, AIMessage).
            image_path: The Base64 data URL of an image associated with the message.

        Returns:
            The ID of the newly created chat message record.
        """
        # Step 1: Ensure the session ID exists in the parent table to satisfy foreign key constraints.
        self.ensure_session_exists(session_id)

        # Step 2: Determine the role ('human', 'ai', 'system') from the message object type.
        if isinstance(message, HumanMessage):
            message_type = 'human'
        elif isinstance(message, AIMessage):
            message_type = 'ai'
        elif isinstance(message, SystemMessage):
            message_type = 'system'
        else:
            message_type = 'unknown'

        # Step 3: Extract the text content from the LangChain message object.
        # It can be a simple string or a list of content blocks (for multimodal messages).
        text_content = ""
        if isinstance(message.content, list):
            for item in message.content:
                if isinstance(item, dict) and item.get('type') == 'text':
                    text_content += item.get('text', '')
        else:
            text_content = str(message.content)

        # Step 4: Open a database connection and perform the transaction.
        with self.get_db_connection() as conn:
            with conn.cursor() as cur:
                # Step 4a: Insert the main chat message into the 'chats' table and retrieve its generated ID.
                cur.execute(
                    """INSERT INTO chats (unique_id, chat_text, message_type)
                       VALUES (%s, %s, %s) RETURNING id""",
                    (session_id, text_content, message_type)
                )
                chat_id = cur.fetchone()[0]

                # Step 4b: If an image URL was provided, insert it into the 'images' table.
                image_id = None
                if image_path:
                    cur.execute(
                        """INSERT INTO images (unique_id, image_url)
                           VALUES (%s, %s) RETURNING id""",
                        (session_id, image_path)
                    )
                    image_id = cur.fetchone()[0]

                # A set to keep track of processed image tokens to avoid duplicates.
                processed_tokens = set()

                # Step 4c: Search the message text for image placeholders (e.g., "{image}").
                # This is the crucial step for creating the association.
                for match in re.finditer(r'\{image\d*\}', text_content):
                    token = match.group(0)
                    start_pos = match.start()
                    end_pos = match.end()

                    # Avoid processing the same token at the same position multiple times.
                    if (token, start_pos) in processed_tokens:
                        continue
                    processed_tokens.add((token, start_pos))

                    # Step 4d: If an image was saved (we have an image_id), insert the
                    # link into the `chat_images` association table.
                    if image_id:
                        cur.execute(
                            """INSERT INTO chat_images (chat_id, image_id, image_token, start_pos, end_pos)
                               VALUES (%s, %s, %s, %s, %s)
                               ON CONFLICT (chat_id, start_pos) DO UPDATE SET
                               image_id = EXCLUDED.image_id,
                               image_token = EXCLUDED.image_token,
                               end_pos = EXCLUDED.end_pos""",
                            (chat_id, image_id, token, start_pos, end_pos)
                        )
                    else:
                        # If for some reason there's a token but no image, record the token's position
                        # without linking to an image_id. This handles edge cases.
                        cur.execute(
                            """INSERT INTO chat_images (chat_id, image_token, start_pos, end_pos)
                               VALUES (%s, %s, %s, %s)
                               ON CONFLICT (chat_id, start_pos) DO NOTHING""",
                            (chat_id, token, start_pos, end_pos)
                        )

                # Step 5: Commit the entire transaction to the database.
                conn.commit()

                # Step 6: Add to ConversationBufferMemory
                try:
                    langchain_memory = self.get_langchain_memory(session_id)

                    if isinstance(message, HumanMessage):
                        content = self._extract_text_from_message(message)
                        langchain_memory.chat_memory.add_user_message(content)
                    elif isinstance(message, AIMessage):
                        langchain_memory.chat_memory.add_ai_message(str(message.content))

                    print(f"Added message to ConversationBufferMemory for session {session_id}")

                except Exception as e:
                    print(f"Warning: Could not add message to ConversationBufferMemory: {e}")

                return chat_id

    def get_conversation_history(self, session_id: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get conversation history for a session from the new schema."""
        limit = limit or self.max_messages_per_session

        with self.get_db_connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute("""
                    SELECT
                        c.id,
                        c.chat_text,
                        c.message_type,
                        c.created_at,
                        i.image_url,
                        ci.image_token,
                        ci.start_pos,
                        ci.end_pos
                    FROM chats c
                    LEFT JOIN chat_images ci ON c.id = ci.chat_id
                    LEFT JOIN images i ON ci.image_id = i.id
                    WHERE c.unique_id = %s
                    ORDER BY c.created_at ASC, ci.start_pos ASC
                """, (session_id,))

                all_chat_rows = cur.fetchall()

                message_ids = sorted(list(set(row['id'] for row in all_chat_rows)), reverse=True)
                limited_message_ids = set(message_ids[:limit])

                rows = [row for row in all_chat_rows if row['id'] in limited_message_ids]
                rows.sort(key=lambda x: (x['created_at'], x['start_pos']))

                messages = {}
                for row in rows:
                    msg_id = row['id']
                    if msg_id not in messages:
                        messages[msg_id] = {
                            'id': msg_id,
                            'text': row['chat_text'],
                            'type': row['message_type'],
                            'created_at': row['created_at'],
                            'images': []
                        }

                    if row['image_url'] or row['image_token']:
                        messages[msg_id]['images'].append({
                            'url': row['image_url'],
                            'token': row['image_token'],
                            'start_pos': row['start_pos'],
                            'end_pos': row['end_pos']
                        })
                return sorted(list(messages.values()), key=lambda x: x['created_at'])

    def get_messages_as_langchain(self, session_id: str, limit: Optional[int] = None) -> List[BaseMessage]:
        """Convert database messages to LangChain message objects."""
        history = self.get_conversation_history(session_id, limit)
        langchain_messages = []

        for msg_data in history:
            content = msg_data['text']
            if msg_data['images']:
                multimodal_content = [{"type": "text", "text": content}]
                for img in msg_data['images']:
                    if img['url']:
                        multimodal_content.append({
                            "type": "image_url",
                            "image_url": {"url": img['url']}
                        })
                content = multimodal_content

            # Create appropriate LangChain message type
            if msg_data['type'] == 'human':
                langchain_messages.append(HumanMessage(content=content))
            elif msg_data['type'] == 'ai':
                langchain_messages.append(AIMessage(content=content))
            elif msg_data['type'] == 'system':
                langchain_messages.append(SystemMessage(content=content))

        return langchain_messages

    def get_context_summary(self, session_id: str, max_chars: int = 2000) -> str:
        """Get a text summary of the conversation history."""
        history = self.get_conversation_history(session_id)
        if not history:
            return "No previous conversation history."

        context_parts = []
        total_chars = 0
        for msg in reversed(history):  # Start from the most recent message
            msg_type = msg['type'].title()
            text = msg['text']

            # Indicate if images are present
            if msg['images']:
                image_count = len([img for img in msg['images'] if img['url']])
                if image_count > 0:
                    text += f" [Contains {image_count} image(s)]"

            part = f"{msg_type}: {text}"
            if total_chars + len(part) > max_chars:
                break

            context_parts.insert(0, part)
            total_chars += len(part)

        return "\n".join(context_parts)

    def clear_session(self, session_id: str) -> bool:
        """Clear all data for a session by deleting the unique_id and clearing ConversationBufferMemory."""
        try:
            with self.get_db_connection() as conn:
                with conn.cursor() as cur:
                    # ON DELETE CASCADE will handle cleaning up related table entries
                    cur.execute("DELETE FROM unique_ids WHERE unique_id = %s", (session_id,))
                    conn.commit()

                    # Clear from ConversationBufferMemory
                    if session_id in self.session_memories:
                        del self.session_memories[session_id]
                        print(f"Cleared ConversationBufferMemory for session {session_id}")

                    return cur.rowcount > 0
        except Exception as e:
            print(f"Error clearing session {session_id}: {e}")
            return False

    def cleanup_old_sessions(self) -> int:
        """Remove sessions older than the specified cleanup_days."""
        cutoff_date = datetime.now() - timedelta(days=self.cleanup_days)
        try:
            with self.get_db_connection() as conn:
                with conn.cursor() as cur:
                    # Find unique_ids of sessions with the last message older than the cutoff
                    cur.execute("""
                        SELECT unique_id FROM (
                            SELECT
                                unique_id,
                                MAX(created_at) as last_message_time
                            FROM chats
                            GROUP BY unique_id
                        ) as session_activity
                        WHERE last_message_time < %s
                    """, (cutoff_date,))

                    old_session_ids = [row[0] for row in cur.fetchall()]

                    if old_session_ids:
                        # Clear from memory buffers first
                        for session_id in old_session_ids:
                            if session_id in self.session_memories:
                                del self.session_memories[session_id]

                        # Delete from database
                        cur.execute("""
                            DELETE FROM unique_ids
                            WHERE unique_id = ANY(%s)
                        """, (old_session_ids,))

                        deleted_count = cur.rowcount
                        conn.commit()

                        if deleted_count > 0:
                            print(f"Cleaned up {deleted_count} old session(s).")
                        return deleted_count

                    return 0
        except Exception as e:
            print(f"Error during old session cleanup: {e}")
            return 0

    def get_session_stats(self, session_id: str) -> Dict[str, Any]:
        """Get statistics for a specific session."""
        with self.get_db_connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                # Get chat stats
                cur.execute("""
                    SELECT
                        COUNT(*) as message_count,
                        MIN(created_at) as first_message,
                        MAX(created_at) as last_message,
                        COUNT(*) FILTER (WHERE message_type = 'human') as human_messages,
                        COUNT(*) FILTER (WHERE message_type = 'ai') as ai_messages
                    FROM chats
                    WHERE unique_id = %s
                """, (session_id,))
                stats = cur.fetchone()

                cur.execute("""
                    SELECT COUNT(*) as image_count
                    FROM images
                    WHERE unique_id = %s
                """, (session_id,))
                image_stats = cur.fetchone()

                if stats and image_stats:
                    stats.update(image_stats)

                return stats if stats else {}

    def get_all_sessions(self) -> List[Dict[str, Any]]:
        """Get all session IDs with their metadata for the sidebar."""
        with self.get_db_connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute("""
                    SELECT 
                        u.unique_id,
                        COUNT(c.id) as message_count,
                        MIN(c.created_at) as first_message_time,
                        MAX(c.created_at) as last_message_time,
                        COALESCE(
                            SUBSTRING(
                                MIN(CASE WHEN c.message_type = 'human' THEN c.chat_text END)
                                FROM 1 FOR 50
                            ), 
                            'No messages'
                        ) as first_user_message_preview
                    FROM unique_ids u
                    LEFT JOIN chats c ON u.unique_id = c.unique_id
                    GROUP BY u.unique_id
                    ORDER BY MAX(c.created_at) DESC NULLS LAST
                """)

                sessions = cur.fetchall()

                # Format the sessions for display
                formatted_sessions = []
                for session in sessions:
                    # Create a display title from the first message or use timestamp
                    if session['first_user_message_preview'] and session['first_user_message_preview'] != 'No messages':
                        title = session['first_user_message_preview']
                        if len(title) > 50:
                            title = title[:47] + "..."
                    else:
                        title = f"Chat {session['unique_id'][:8]}"

                    # Format the last message time
                    if session['last_message_time']:
                        time_str = session['last_message_time'].strftime("%m/%d %H:%M")
                    else:
                        time_str = "No messages"

                    formatted_sessions.append({
                        'unique_id': session['unique_id'],
                        'title': title,
                        'message_count': session['message_count'],
                        'last_message_time': time_str,
                        'display_text': f"{title}\n{session['message_count']} messages • {time_str}"
                    })

                return formatted_sessions

    def load_session_chat_history(self, session_id: str) -> List[tuple]:
        """
        Load chat history for UI display, correctly rendering images and cleaning text.

        This method fetches the conversation history and formats it into the
        specific tuple structure that the Gradio Chatbot component requires to
        properly display user-uploaded images and their corresponding text.

        Args:
            session_id: The unique ID of the session to load.

        Returns:
            A list of tuples formatted for the Gradio Chatbot UI.
        """
        # Step 1: Get the raw conversation data from the database.
        history = self.get_conversation_history(session_id)
        ui_history = []

        for msg in history:
            if msg['type'] == 'system':
                continue

            is_human = msg['type'] == 'human'
            text_content = re.sub(r'\s*\{image\d*\}\s*$', '', msg['text']).strip()
            image_urls = [img['url'] for img in msg.get('images', []) if img.get('url')]

            if is_human:
                # First, process and display any images from the database.
                for url in image_urls:
                    # THE FIX IS HERE: Decode the Base64 string and save to a temp file
                    if url and url.startswith('data:image'):
                        try:
                            # Split the metadata from the actual base64 data
                            header, encoded = url.split(",", 1)
                            # Get the file extension (e.g., .jpeg, .png)
                            file_ext = "." + header.split('/')[1].split(';')[0]

                            # Decode the base64 string
                            image_data = base64.b64decode(encoded)

                            # Create a temporary file to store the image
                            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
                                tmp_file.write(image_data)
                                # Get the path to the temporary file
                                temp_file_path = tmp_file.name

                            # Append the FILE PATH to the history, not the raw data
                            ui_history.append(((temp_file_path,), None))

                        except Exception as e:
                            print(f"Error decoding or saving Base64 image: {e}")
                            # Optionally, add a placeholder for a broken image
                            ui_history.append(("[Error loading image]", None))

                # After handling images, add the text content.
                if text_content:
                    ui_history.append((text_content, None))

            else:  # AI message
                if ui_history and ui_history[-1][1] is None:
                    last_turn = ui_history[-1]
                    ui_history[-1] = (last_turn[0], text_content)
                else:
                    ui_history.append((None, text_content))

        # Final cleanup
        ui_history = [turn for turn in ui_history if turn[0] or turn[1]]
        return ui_history

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

class ImageInterpreterInput(BaseModel):
    """Input schema for the Code Extractor From Image tool."""
    query: str = Field(description="The user's original question about the image, "
                                   "providing context for what to look for.")


class SmartImageInterpreterTool(BaseTool):
    """
    A specialized tool to extract text, code, and error messages from an image, or describe it if non-technical.
    This is the first step in any workflow involving an image.
    """
    name: str = "code_extractor_from_image"
    description: str = (
        "Use this tool FIRST for ANY query that includes an image. "
        "Its purpose is to transcribe code/errors or describe the image content. The output of this tool "
        "will be used by other tools to find a solution or to answer the user's question directly."
    )
    args_schema: Type[BaseModel] = ImageInterpreterInput
    client: OpenAI = Field(default=None, exclude=True)
    llm_model_name: str = Field(default="openai/gpt-4o-mini", exclude=True)

    def __init__(self, openrouter_api_key: str, **kwargs):
        super().__init__(**kwargs)
        if not openrouter_api_key or "YOUR_KEY" in openrouter_api_key:
            raise ValueError("OpenRouter API Key is required for CodeExtractorFromImageTool.")
        client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=openrouter_api_key)
        object.__setattr__(self, 'client', client)

    def _run(self, query: str, image_url: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        print(f" CodeExtractorFromImageTool: Extracting content from image for query: '{query}'")
        system_prompt = """You are a highly specialized visual analysis expert. Your SOLE PURPOSE is to analyze the provided image and return a text-based representation of it.

**CRITICAL RULES:**
1.  **Analyze the image content first.** Determine if it is a technical screenshot (code, error, terminal) or a general photograph (object, animal, scene).
2.  **If it is a technical screenshot:** Accurately transcribe ALL visible code, commands, and error messages. DO NOT try to solve or explain the error. Your only job is transcription. Start your response with: "The image contains a technical screenshot. Here is the transcription:"
3.  **If it is a general photograph:** Describe the main subject and scene of the image clearly and concisely. Start your response with: "The image does not contain code. It shows:"
4.  **Return ONLY the transcription or the description.** Do not add any other commentary.
"""
        try:
            message = HumanMessage(
                content=[
                    {"type": "text", "text": f"User's original query for context (do not answer it): '{query}'"},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ]
            )
            response = self.client.chat.completions.create(
                model=self.llm_model_name,
                messages=[{"role": "system", "content": system_prompt}, message.to_dict()],
                temperature=0.0,
                max_tokens=1500
            )
            extracted_text = response.choices[0].message.content
            print(f" CodeExtractorFromImageTool: Successfully extracted content.")
            return extracted_text
        except Exception as e:
            error_msg = f"Error in Code Extractor Tool: {str(e)}"
            print(f" CodeExtractorFromImageTool: {error_msg}")
            return error_msg

    async def _arun(self, query: str, image_url: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        return self._run(query, image_url)

# --- GRADIO INTERFACE AND LOGIC ---
def run_gradio_interface(agent_executor, memory: PostgreSQLMemoryBuffer):
    with gr.Blocks(theme=gr.themes.Soft(), title="Programming Assistant") as demo:

        # State management
        session_id = gr.State(str(uuid.uuid4()))
        print(f"New Gradio session started with ID: {session_id.value}")

        # Initialize with system message
        initial_messages = [SystemMessage(content=SYSTEM_PROMPT)]
        agent_state = gr.State(initial_messages)

        gr.Markdown("# Programming Assistant")

        with gr.Row():
            # Left sidebar for chat history (This section is unchanged)
            with gr.Column(scale=2, min_width=250):
                gr.Markdown("### Chat History")
                new_chat_btn = gr.Button("New Chat", variant="primary", size="sm")
                initial_sessions = memory.get_all_sessions()
                session_choices = [(session['display_text'], session['unique_id']) for session in initial_sessions]
                session_dropdown = gr.Dropdown(
                    choices=session_choices,
                    label="Previous Chats",
                    interactive=True,
                    visible=len(session_choices) > 0
                )
                refresh_btn = gr.Button("Refresh", size="sm")
                delete_btn = gr.Button("Delete Selected", variant="stop", size="sm")
                session_info = gr.Markdown("**Current Session:** New Chat")

            # Main chat area
            with gr.Column(scale=5):
                chatbot = gr.Chatbot(label="Conversation", height=600, bubble_full_width=False, render_markdown=True)

                # Image preview area (shows when image is selected)
                with gr.Row(visible=False) as image_preview_row:
                    with gr.Column(scale=1):
                        gr.Markdown("**Attached:**")
                    with gr.Column(scale=4):
                        image_preview = gr.Image(
                            height=80, show_label=False, interactive=False, container=False
                        )
                    with gr.Column(scale=1):
                        remove_image_btn = gr.Button("Remove", variant="stop", size="sm")

                # Chat input area with integrated attachment button
                with gr.Row():
                    attach_btn = gr.UploadButton(
                        "+", file_types=["image"], size="lg", variant="secondary", min_width=50
                    )
                    msg = gr.Textbox(
                        placeholder="Type your message here...",
                        show_label=False, container=False, lines=1, max_lines=5, scale=10
                    )
                    submit_btn = gr.Button("Send", variant="primary", size="lg")

                clear_current_btn = gr.Button("Clear Current Chat", size="sm", variant="stop")

        # Add a state to store the current image path
        current_image = gr.State(None)

        def respond(message, image_path, current_session_id, agent_messages, ui_history):
            """Handle user message and generate response"""
            print(f"\nProcessing for session {current_session_id}: \"{message}\" | Image: {image_path}")

            if not message and not image_path:
                # If both message and image are empty, do nothing.
                yield ui_history, agent_messages, "", None, gr.update(visible=False)
                return

            image_url = None
            # This variable will hold the text sent to the agent and database.
            text_for_agent = message

            if image_path:
                # Encode the image for storage and agent processing.
                with open(image_path, "rb") as f:
                    encoded_string = base64.b64encode(f.read()).decode('utf-8')
                image_url = f"data:image/jpeg;base64,{encoded_string}"

                text_for_agent = f"{message} {{image}}".strip()

                # Add the user's image to the UI immediately for responsiveness.
                ui_history.append(((image_path,), None))

            # Add the user's text message to the UI.
            if message:
                # If an image was just added, update its turn with the text.
                if ui_history and ui_history[-1][1] is None:
                     ui_history[-1] = (ui_history[-1][0], message)
                else: # Otherwise, add a new turn for the text.
                     ui_history.append((message, None))

            # Update the UI to show the user's full input.
            yield ui_history, agent_messages, "", None, gr.update(visible=False)

            # Prepare the LangChain message object
            content_list = []
            if text_for_agent:  # Use the potentially modified text
                content_list.append({"type": "text", "text": text_for_agent})
            if image_url:
                content_list.append({"type": "image_url", "image_url": {"url": image_url}})

            human_message = HumanMessage(content=content_list)
            agent_messages.append(human_message)

            memory.save_message(session_id=current_session_id, message=human_message, image_path=image_url)
            print(f"Saved HumanMessage to DB for session {current_session_id}")

            # Invoke the agent
            try:
                final_state = agent_executor.invoke({"messages": agent_messages}, {"recursion_limit": 10})
                agent_messages = final_state['messages']
                final_answer_message = agent_messages[-1]
                print(f"Final Answer: {final_answer_message.content}")

                memory.save_message(session_id=current_session_id, message=final_answer_message)
                print(f"Saved AIMessage to DB for session {current_session_id}")

                # Update the last entry in the UI history with the AI's response
                if ui_history:
                    ui_history[-1] = (ui_history[-1][0], final_answer_message.content)

            except Exception as e:
                error_msg = f"Error processing request: {str(e)}"
                print(f"ERROR: {error_msg}")
                if ui_history:
                    ui_history[-1] = (ui_history[-1][0], error_msg)

            # Final yield to update UI and clear inputs
            yield ui_history, agent_messages, "", None, gr.update(visible=False)

        def handle_image_upload(uploaded_file):
            """Handle when an image is uploaded - show preview."""
            if uploaded_file:
                print(f"Image uploaded to temporary path: {uploaded_file.name}")
                return (
                    gr.Row(visible=True),  # Show preview row
                    uploaded_file.name,  # Update preview image with the temp file path
                    uploaded_file.name  # Store the file path in the state
                )
            return gr.Row(visible=False), None, None

        def remove_image():
            """Remove the selected image and hide preview"""
            return (
                gr.Row(visible=False),  # Hide preview row
                None,  # Clear preview image
                None  # Clear stored file path
            )

        # Session management functions (unchanged)
        def load_session(selected_session_id, current_session_id):
            if not selected_session_id:
                # Return current state if dropdown is cleared
                return chatbot.value, agent_state.value, current_session_id, f"**Current Session:** {current_session_id[:8]}..."
            print(f"Loading session: {selected_session_id}")
            ui_history = memory.load_session_chat_history(selected_session_id)
            agent_messages = memory.get_messages_as_langchain(selected_session_id)
            if not any(isinstance(m, SystemMessage) for m in agent_messages):
                agent_messages.insert(0, SystemMessage(content=SYSTEM_PROMPT))
            session_info_text = f"**Current Session:** {selected_session_id[:8]}..."
            return ui_history, agent_messages, selected_session_id, session_info_text

        def refresh_sessions():
            sessions = memory.get_all_sessions()
            choices = [(session['display_text'], session['unique_id']) for session in sessions]
            return gr.Dropdown(choices=choices, visible=len(choices) > 0)

        def create_new_chat():
            new_session_id = str(uuid.uuid4())
            print(f"Creating new chat session: {new_session_id}")
            initial_msgs = [SystemMessage(content=SYSTEM_PROMPT)]
            return [], initial_msgs, "", None, gr.update(visible=False), new_session_id, "**Current Session:** New Chat"

        def delete_selected_session(selected_session_id, current_session_id):
            if not selected_session_id:
                return gr.update(), current_session_id, gr.update()
            success = memory.clear_session(selected_session_id)
            if success:
                print(f"Deleted session: {selected_session_id}")
                if selected_session_id == current_session_id:
                    new_id = str(uuid.uuid4())
                    return refresh_sessions(), new_id, "**Current Session:** New Chat"
            return refresh_sessions(), current_session_id, gr.update()

        def clear_current_chat(current_session_id):
            memory.clear_session(current_session_id)
            print(f"Cleared current session {current_session_id} from the database.")
            return create_new_chat()

        # --- MODIFICATION: Updated Event Handlers ---

        # The UploadButton's "upload" event triggers the preview
        attach_btn.upload(
            handle_image_upload,
            inputs=[attach_btn],
            outputs=[image_preview_row, image_preview, current_image]
        )

        remove_image_btn.click(
            remove_image,
            outputs=[image_preview_row, image_preview, current_image]
        )

        submit_btn.click(
            respond,
            [msg, current_image, session_id, agent_state, chatbot],
            [chatbot, agent_state, msg, current_image, image_preview_row]
        )
        msg.submit(
            respond,
            [msg, current_image, session_id, agent_state, chatbot],
            [chatbot, agent_state, msg, current_image, image_preview_row]
        )

        # Session management events
        session_dropdown.change(
            load_session,
            [session_dropdown, session_id],
            [chatbot, agent_state, session_id, session_info]
        )
        new_chat_btn.click(
            create_new_chat,
            [],
            [chatbot, agent_state, msg, current_image, image_preview_row, session_id, session_info]
        )
        refresh_btn.click(refresh_sessions, outputs=[session_dropdown])
        delete_btn.click(
            delete_selected_session,
            [session_dropdown, session_id],
            [session_dropdown, session_id, session_info]
        )
        clear_current_btn.click(
            clear_current_chat,
            [session_id],
            [chatbot, agent_state, msg, current_image, image_preview_row, session_id, session_info]
        )

    print("\nProgramming Assistant with Chat History Ready! Launching Gradio interface...")
    demo.queue().launch()


# --- MAIN EXECUTION BLOCK ---

if __name__ == "__main__":
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    load_dotenv()

    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    STACK_EXCHANGE_API_KEY = os.getenv("STACK_EXCHANGE_API_KEY")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    SEARCH_ENGINE_ID = os.getenv("SEARCH_ENGINE_ID")

    if not OPENROUTER_API_KEY:
        raise ValueError("FATAL: OPENROUTER_API_KEY environment variable must be set.")

    db_params = {
        "host": os.getenv("DB_HOST"),
        "dbname": os.getenv("DB_NAME"),
        "user": os.getenv("DB_USER"),
        "password": os.getenv("DB_PASSWORD"),
        "port": os.getenv("DB_PORT", 5432)
    }

    try:
        memory = PostgreSQLMemoryBuffer(connection_params=db_params)
        print(" PostgreSQLMemoryBuffer initialized successfully.")
    except Exception as e:
        raise RuntimeError(f"FATAL: Could not connect to the database. Please check your .env settings. Error: {e}")

    tools = [
        StackExchangeTool(
            openrouter_api_key=OPENROUTER_API_KEY,
            stack_exchange_api_key=STACK_EXCHANGE_API_KEY
        ),
        InternalKnowledgeSearchTool(
            openrouter_api_key=OPENROUTER_API_KEY
        ),
        SmartImageInterpreterTool(
            openrouter_api_key=OPENROUTER_API_KEY
        )
    ]
    if GOOGLE_API_KEY and SEARCH_ENGINE_ID:
        tools.append(GoogleSearchRAGTool(
            openrouter_api_key=OPENROUTER_API_KEY,
            google_api_key=GOOGLE_API_KEY,
            search_engine_id=SEARCH_ENGINE_ID
        ))

    print(f" Total tools loaded: {len(tools)}")

    llm = ChatOpenAI(
        model="openai/gpt-4o-mini",
        temperature=0.1,
        max_tokens=1500,
        api_key=OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1"
    )

    llm_with_tools = llm.bind_tools(tools)
    print("LLM initialized and bound with tools successfully")

    class AgentState(TypedDict):
        messages: Annotated[list[AnyMessage], operator.add]

    graph_builder = StateGraph(AgentState)

    def agent_node(state: AgentState):
        response = llm_with_tools.invoke(state['messages'])
        return {"messages": [response]}

    graph_builder.add_node("agent", agent_node)
    tool_node = ToolNode(tools)
    graph_builder.add_node("tools", tool_node)
    graph_builder.set_entry_point("agent")
    graph_builder.add_conditional_edges("agent", tools_condition)
    graph_builder.add_edge("tools", "agent")

    agent_executor = graph_builder.compile()
    print(" LangGraph Agent initialized successfully")

    run_gradio_interface(agent_executor, memory)