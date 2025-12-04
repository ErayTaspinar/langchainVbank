import os
import uuid
import mimetypes
import base64
from contextlib import contextmanager
from typing import Optional, List, Dict, Any, Type, TypedDict, Annotated
import operator
import traceback

from flask import Flask, request, jsonify
from flask_cors import CORS
import psycopg
import requests
from openai import OpenAI
from psycopg.rows import dict_row
from pydantic import BaseModel, Field
from langchain.tools import BaseTool
import chromadb
from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from chromadb.utils import embedding_functions
from langchain_openai import ChatOpenAI

SYSTEM_PROMPT = """You are a specialized technical assistant. 

**PROTOCOL:**
1. **Oracle/SQL Questions:** Use `internal_knowledge_search`.
2. **Modern/General Questions:** Use `google_search_rag`.
3. **Legacy Code Questions:** Use `stack_exchange_search`.

**CRITICAL RULES:**
- **ONE SEARCH LIMIT:** Do not search for the same thing twice. If a tool returns "No results" or "Maintenance", **IMMEDIATELY** switch to `google_search_rag` or say you don't know.
- **STOP IMMEDIATELY:** Once you have enough info to answer the user's specific question, output the answer. Do not keep searching for "more" context unless asked.
- **IF STACK EXCHANGE FAILS:** Do not retry it. Do not apologize. Just use Google.
"""


class PostgreSQLMemoryBuffer:
    def __init__(self, connection_params: Dict[str, Any], max_messages_per_session: int = 50,
                 max_token_limit: int = 2000):
        self.connection_params = connection_params
        self.max_messages_per_session = max_messages_per_session
        self.max_token_limit = max_token_limit
        self.session_memories: Dict[str, ConversationBufferMemory] = {}
        self._initialize_database()

    @contextmanager
    def get_db_connection(self):
        conn = None
        try:
            conn = psycopg.connect(**self.connection_params)
            yield conn
        except Exception as e:
            if conn: conn.rollback()
            raise

    def _initialize_database(self):
        try:
            with self.get_db_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT to_regclass('public.users')")
                    if cur.fetchone()[0] is None: raise RuntimeError("'users' table not found.")
                    cur.execute("SELECT to_regclass('public.unique_ids')")
                    if cur.fetchone()[0] is None: raise RuntimeError("'unique_ids' table not found.")
                    cur.execute("SELECT to_regclass('public.chats')")
                    if cur.fetchone()[0] is None: raise RuntimeError("'chats' table not found.")
        except Exception as e:
            raise Exception(f"Failed to initialize database: {e}")

    def create_session(self, session_id: str, user_id: int):
        with self.get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO unique_ids (unique_id, user_id) VALUES (%s, %s)",
                    (session_id, user_id)
                )
                conn.commit()

    def get_langchain_memory(self, session_id: str) -> ConversationBufferMemory:
        if session_id not in self.session_memories:
            memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True,
                                              max_token_limit=self.max_token_limit)
            for msg in self.get_messages_as_langchain(session_id):
                if isinstance(msg, HumanMessage):
                    memory.chat_memory.add_user_message(self._extract_text_from_message(msg))
                elif isinstance(msg, AIMessage):
                    memory.chat_memory.add_ai_message(str(msg.content))
            self.session_memories[session_id] = memory
        return self.session_memories[session_id]

    def _extract_text_from_message(self, message: BaseMessage) -> str:
        if isinstance(message.content, list):
            return ' '.join(item.get('text', '') for item in message.content if
                            isinstance(item, dict) and item.get('type') == 'text')
        return str(message.content)

    def save_message(self, session_id: str, message: BaseMessage, image_path: Optional[str] = None) -> int:
        with self.get_db_connection() as conn:
            with conn.cursor() as cur:
                message_type_map = {HumanMessage: 'human', AIMessage: 'ai'}
                message_type = message_type_map.get(type(message), 'unknown')
                text_content = self._extract_text_from_message(message)
                cur.execute("INSERT INTO chats (unique_id, chat_text, message_type) VALUES (%s, %s, %s) RETURNING id",
                            (session_id, text_content, message_type))
                chat_id = cur.fetchone()[0]
                if image_path:
                    cur.execute("INSERT INTO images (unique_id, image_url) VALUES (%s, %s) RETURNING id",
                                (session_id, image_path))
                    image_id = cur.fetchone()[0]
                    cur.execute("INSERT INTO chat_images (chat_id, image_id, start_pos, end_pos) VALUES (%s, %s, 0, 0)",
                                (chat_id, image_id))
                conn.commit()
                return chat_id

    def get_conversation_history(self, session_id: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        limit = limit or self.max_messages_per_session
        with self.get_db_connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute("""
                    SELECT c.id, c.chat_text, c.message_type, c.created_at, array_agg(i.image_url) FILTER (WHERE i.image_url IS NOT NULL) as image_urls
                    FROM chats c
                    LEFT JOIN chat_images ci ON c.id = ci.chat_id
                    LEFT JOIN images i ON ci.image_id = i.id
                    WHERE c.unique_id = %s
                    GROUP BY c.id, c.chat_text, c.message_type, c.created_at
                    ORDER BY c.created_at ASC
                """, (session_id,))
                rows = cur.fetchall()
                history = [
                    {
                        'id': row['id'],
                        'text': row['chat_text'],
                        'type': row['message_type'],
                        'created_at': row['created_at'].isoformat(),
                        'images': [{'url': url} for url in row['image_urls']] if row['image_urls'] else []
                    }
                    for row in rows
                ]
                return history[-limit:]

    def get_messages_as_langchain(self, session_id: str, limit: Optional[int] = None) -> List[BaseMessage]:
        history = self.get_conversation_history(session_id, limit)
        messages = []
        for msg_data in history:
            content = [{"type": "text", "text": msg_data['text']}]
            if msg_data['images']:
                for img in msg_data['images']:
                    content.append({"type": "image_url", "image_url": {"url": img['url']}})

            if msg_data['type'] == 'human':
                messages.append(HumanMessage(content=content))
            elif msg_data['type'] == 'ai':
                messages.append(AIMessage(content=msg_data['text']))
        return messages

    def get_sessions_for_user(self, email: str) -> List[Dict[str, Any]]:
        with self.get_db_connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute("""
                    SELECT u_ids.unique_id, MAX(c.created_at) as last_message_time,
                           (SELECT chat_text FROM chats WHERE unique_id = u_ids.unique_id AND message_type = 'human' AND chat_text IS NOT NULL AND chat_text != '' ORDER BY created_at ASC LIMIT 1) as title
                    FROM unique_ids u_ids
                    JOIN users u ON u_ids.user_id = u.id
                    LEFT JOIN chats c ON u_ids.unique_id = c.unique_id
                    WHERE u.email = %s
                    GROUP BY u_ids.unique_id
                    ORDER BY last_message_time DESC NULLS LAST
                """, (email,))
                results = cur.fetchall()
                return [{'session_id': s['unique_id'], 'title': (
                    s['title'][:47] + "..." if s['title'] and len(s['title']) > 50 else (
                        s['title'] if s['title'] else "Chat " + s['unique_id'][:8]))} for s in results]


class GoogleSearchRAGInput(BaseModel):
    query: str = Field(description="The user's question to search for on the web")
    n_results: int = Field(default=5, description="Number of search results to retrieve (max 10)")


class InternalKnowledgeSearchInput(BaseModel):
    query: str = Field(description="The user's technical or programming question.")


class SmartImageInterpreterInput(BaseModel):
    query: str = Field(
        description="The user's original question about the image, providing context for what to look for.")
    image_url: str = Field(description="The data URL of the image to be analyzed.")


class StackExchangeSearchInput(BaseModel):
    query: str = Field(description="The user's programming question to search on Stack Exchange.")


class GoogleSearchRAGTool(BaseTool):
    name: str = "google_search_rag"
    description: str = "A tool for performing Google searches and generating answers."
    args_schema: Type[BaseModel] = GoogleSearchRAGInput
    client: OpenAI = Field(default=None, exclude=True)
    google_api_key: str = Field(exclude=True)
    search_engine_id: str = Field(exclude=True)

    def __init__(self, **data):
        super().__init__(**data)
        self.client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=data.get("openrouter_api_key"))

    def _run(self, query: str, n_results: int = 5) -> str:
        if not self.google_api_key or not self.search_engine_id:
            return "Search is unavailable (Missing configuration)."

        url = "https://www.googleapis.com/customsearch/v1"
        params = {'key': self.google_api_key, 'cx': self.search_engine_id, 'q': query, 'num': n_results}
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            items = response.json().get('items', [])
            if not items: return "No search results found."
            context = "\n".join([f"Source: {item['link']}\nSnippet: {item['snippet']}" for item in items])
            system_prompt = "Answer the user's query based ONLY on the provided context. Cite your sources."
            response = self.client.chat.completions.create(model="openai/gpt-4o-mini",
                                                           messages=[{"role": "system", "content": system_prompt},
                                                                     {"role": "user",
                                                                      "content": f"Context:\n{context}\n\nQuery: {query}"}])
            return response.choices[0].message.content
        except Exception as e:
            return f"Search failed: {e}"


class InternalKnowledgeSearchTool(BaseTool):
    name: str = "internal_knowledge_search"
    description: str = "Searches an internal knowledge base for Oracle/SQL questions."
    args_schema: Type[BaseModel] = InternalKnowledgeSearchInput
    client: OpenAI = Field(default=None, exclude=True)
    collection: Any = Field(default=None, exclude=True)

    def __init__(self, **data):
        super().__init__(**data)
        self.client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=data.get("openrouter_api_key"))
        embedder = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        db_client = chromadb.PersistentClient(path="./chroma_db")
        self.collection = db_client.get_collection(name="stackoverflow_results", embedding_function=embedder)

    def _run(self, query: str) -> str:
        results = self.collection.query(query_texts=[query], n_results=3, include=["documents"])
        if not results['documents'][0]: return "No relevant documents found in the internal knowledge base."
        context = "\n\n".join(results['documents'][0])
        system_prompt = "Answer the user's query based ONLY on the provided context."
        response = self.client.chat.completions.create(model="openai/gpt-4o-mini",
                                                       messages=[{"role": "system", "content": system_prompt},
                                                                 {"role": "user",
                                                                  "content": f"Context:\n{context}\n\nQuery: {query}"}])
        return response.choices[0].message.content


class SmartImageInterpreterTool(BaseTool):
    name: str = "code_extractor_from_image"
    description: str = "Interprets an image to extract code or describe its content."
    args_schema: Type[BaseModel] = SmartImageInterpreterInput
    client: OpenAI = Field(default=None, exclude=True)

    def __init__(self, **data):
        super().__init__(**data)
        self.client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=data.get("openrouter_api_key"))

    def _run(self, query: str, image_url: str) -> str:
        system_prompt = "Analyze the image. If it's technical, transcribe the text/code. If it's a photo, describe it. Be concise."
        messages = [{"role": "user", "content": [{"type": "text", "text": query},
                                                 {"type": "image_url", "image_url": {"url": image_url}}]}]
        response = self.client.chat.completions.create(model="openai/gpt-4o-mini", messages=messages, max_tokens=1000)
        return response.choices[0].message.content


class StackExchangeTool(BaseTool):
    name: str = "stack_exchange_search"
    description: str = "Searches Stack Exchange for programming questions."
    args_schema: Type[BaseModel] = StackExchangeSearchInput
    google_tool: Optional[GoogleSearchRAGTool] = None

    def _run(self, query: str) -> str:
        if self.google_tool:
            return f"Stack Exchange is under maintenance. I have automatically searched Google instead. Results: {self.google_tool._run(query)}"
        return "Stack Exchange is under maintenance and no backup search is available."


app = Flask(__name__)
CORS(app)
load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
SEARCH_ENGINE_ID = os.getenv("SEARCH_ENGINE_ID")
db_params = {"host": os.getenv("DB_HOST"), "dbname": os.getenv("DB_NAME"), "user": os.getenv("DB_USER"),
             "password": os.getenv("DB_PASSWORD"), "port": os.getenv("DB_PORT", 5432)}
memory = PostgreSQLMemoryBuffer(connection_params=db_params)

internal_tool = InternalKnowledgeSearchTool(openrouter_api_key=OPENROUTER_API_KEY)
image_tool = SmartImageInterpreterTool(openrouter_api_key=OPENROUTER_API_KEY)
google_tool = None

tools = [internal_tool, image_tool]

if GOOGLE_API_KEY and SEARCH_ENGINE_ID:
    google_tool = GoogleSearchRAGTool(google_api_key=GOOGLE_API_KEY, search_engine_id=SEARCH_ENGINE_ID,
                                      openrouter_api_key=OPENROUTER_API_KEY)
    tools.append(google_tool)

stack_tool = StackExchangeTool(google_tool=google_tool)
tools.append(stack_tool)

llm = ChatOpenAI(model="openai/gpt-4o-mini", temperature=0, api_key=OPENROUTER_API_KEY,
                 base_url="https://openrouter.ai/api/v1")
llm_with_tools = llm.bind_tools(tools)


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]


graph_builder = StateGraph(AgentState)


def agent_node(state: AgentState):
    return {"messages": [llm_with_tools.invoke(state['messages'])]}


graph_builder.add_node("agent", agent_node)
tool_node = ToolNode(tools)
graph_builder.add_node("tools", tool_node)
graph_builder.set_entry_point("agent")
graph_builder.add_conditional_edges("agent", tools_condition)
graph_builder.add_edge("tools", "agent")
agent_executor = graph_builder.compile()


def get_user_id_from_email(email: str) -> Optional[int]:
    with memory.get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM users WHERE email = %s", (email,))
            result = cur.fetchone()
            return result[0] if result else None


@app.route('/chat/new', methods=['POST'])
def new_chat():
    data = request.get_json()
    email = data.get('email')

    if not email:
        return jsonify({"error": "Email is required."}), 400

    session_id = str(uuid.uuid4())
    try:
        user_id = get_user_id_from_email(email)
        if user_id is None:
            return jsonify({"error": "User not found."}), 404

        memory.create_session(session_id, user_id)
        return jsonify({"session_id": session_id}), 201
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "Failed to create session."}), 500


@app.route('/chats/user/<string:email>', methods=['GET'])
def get_user_chats(email):
    try:
        sessions = memory.get_sessions_for_user(email)
        return jsonify(sessions), 200
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "Failed to retrieve chats."}), 500


@app.route('/chat/<string:session_id>', methods=['GET'])
def get_chat_history(session_id):
    return jsonify(memory.get_conversation_history(session_id)), 200


@app.route('/chat/<string:session_id>', methods=['POST'])
def chat(session_id):
    user_message = ""
    image_url = None

    if request.is_json:
        data = request.get_json()
        user_message = data.get('message', '')
    else:
        user_message = request.form.get('message', '')
        image_file = request.files.get('image')

        if image_file:
            mimetype = image_file.mimetype
            if not mimetype or mimetype == 'application/octet-stream':
                mimetype, _ = mimetypes.guess_type(image_file.filename)

            allowed_types = {'image/jpeg', 'image/jpg', 'image/png', 'image/webp', 'image/gif'}
            if mimetype == 'image/jpg': mimetype = 'image/jpeg'

            if mimetype not in allowed_types:
                return jsonify({"error": f"Unsupported image type: {mimetype}."}), 400

            try:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                image_url = f"data:{mimetype};base64,{encoded_string}"
            except Exception:
                return jsonify({"error": "Failed to process image file."}), 400

    if not user_message and not image_url:
        return jsonify({"error": "Empty message"}), 400

    try:
        content_list = [{"type": "text", "text": user_message}]
        if image_url:
            content_list.append({"type": "image_url", "image_url": {"url": image_url}})
        human_message = HumanMessage(content=content_list)

        memory.save_message(session_id=session_id, message=human_message, image_path=image_url)

        agent_messages = [SystemMessage(content=SYSTEM_PROMPT)] + memory.get_messages_as_langchain(session_id)

        final_state = agent_executor.invoke({"messages": agent_messages}, {"recursion_limit": 15})
        ai_response_message = final_state['messages'][-1]

        memory.save_message(session_id=session_id, message=ai_response_message)
        return jsonify({"response": ai_response_message.content})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Server Error: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001)