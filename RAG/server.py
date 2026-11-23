import json
import datetime
import time
import logging
from typing import Optional, List
from fastapi import Form, UploadFile, File, HTTPException, Request, Response
from pathlib import Path 
import os, uuid
from tempfile import NamedTemporaryFile
from langchain_core.runnables import RunnableLambda
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import Document as LangchainDocument
from langchain_core.prompts import PromptTemplate
import asyncio
from concurrent.futures import ThreadPoolExecutor
from filelock import FileLock
from dotenv import load_dotenv

# Load configuration before importing project modules that read environment
# variables at import time (for example dbhelper and agent).
auth_env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "auth", ".env"))
load_dotenv(dotenv_path=auth_env_path)
load_dotenv()

from langchain_openai import ChatOpenAI
import requests

from vectorestore import build_vectorstore, load_vectorstore
from wordloader import process_document 

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, RedirectResponse, JSONResponse
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from dbhelper import get_connection, get_threads_by_user, get_user_by_email, create_thread, add_message, get_messages_by_thread, update_thread_title

from prometheus_fastapi_instrumentator import Instrumentator
from agent import ConversationalAgent, format_messages
import time

# ============================================================
# AUTH DEPENDENCIES  (merged from main.py / Flask auth server)
# ============================================================
from starlette.middleware.sessions import SessionMiddleware
from authlib.integrations.starlette_client import OAuth
from sqlalchemy import create_engine, Column, String, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from passlib.context import CryptContext

# ---------- Session middleware (replaces Flask sessions) ----------
SECRET_KEY = os.getenv("SESSION_SECRET_KEY", "your-secret-key-here")

# ---------- SQLAlchemy (auth DB) ----------
# Use PostgreSQL if DATABASE_URL is set, otherwise fall back to SQLite
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    # Fallback to shared SQLite database only if PostgreSQL is not configured
    db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app.db"))
    DATABASE_URL = f"sqlite:///{db_path}"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
)
Base = declarative_base()
SessionLocal = sessionmaker(bind=engine)

class User(Base):
    __tablename__ = "users"
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    email = Column(String(120), unique=True, nullable=False)
    name = Column(String(120), nullable=False)
    password_hash = Column(String(256))
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    auth_provider = Column(String(50), default="google")

Base.metadata.create_all(bind=engine)

# Support BOTH scrypt (from werkzeug/Flask) and bcrypt for password verification
# bcrypt is default for new passwords
# Added pbkdf2_sha256 for Flask/werkzeug compatibility
pwd_context = CryptContext(schemes=["bcrypt", "bcrypt_sha256", "pbkdf2_sha256", "scrypt"], deprecated="auto")

# ---------- Google OAuth (Authlib / Starlette) ----------
oauth = OAuth()
oauth.register(
    name="google",
    client_id=os.getenv("GOOGLE_CLIENT_ID"),
    client_secret=os.getenv("GOOGLE_CLIENT_SECRET"),
    server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
    client_kwargs={"scope": "openid email profile"},
)

ALLOWED_DOMAIN = os.getenv("ALLOWED_DOMAIN", "nmit.ac.in")
FRONTEND_URL  = os.getenv("FRONTEND_URL", "http://localhost:3000")
BACKEND_URL   = os.getenv("BACKEND_URL",  "http://localhost:9096")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

db_type = "PostgreSQL" if DATABASE_URL.startswith("postgresql") else "SQLite"
logger.info(f"✅ Using {db_type} DATABASE: {DATABASE_URL}")

# ============================================================
# FASTAPI APP
# ============================================================
app = FastAPI()

# Session middleware must be added BEFORE CORS
app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY, max_age=31 * 24 * 3600)

# Enable Prometheus metrics exporter
Instrumentator().instrument(app).expose(app)

# ---------- CORS ----------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://dev.nmit.ac.in:3001",
        "http://dev.nmit.ac.in:9095",
        "http://dev.nmit.ac.in:9096",
        "http://192.168.190.28:3001",
        "http://192.168.190.28:9095",
        "http://192.168.190.28:9096",
        "http://localhost:3001",
        "http://localhost:3000",
        "http://localhost:9095",
        "http://localhost:9096",
        "http://127.0.0.1:3001",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:9095",
        "http://127.0.0.1:9096",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------Models-----------
MODEL_REGISTRY = {
    "llama": "./meta-llama/Llama-3.1-8B-Instruct-awq",
    "deepseek": "./DeepSeek-Coder-V2-Lite-Instruct-awq",
    "gpt-oss": "openai/gpt-oss-20b",
}

API_PORT_MAPPING = {
    "llama": os.getenv("LLAMA_API_URL", "http://127.0.0.1:8000"),
    "deepseek": os.getenv("DEEPSEEK_API_URL", "http://127.0.0.1:8002"),
    "gpt-oss": os.getenv("GPT_OSS_API_URL", "http://127.0.0.1:8004")
}

# ---------- Schemas ----------
class MessageInput(BaseModel):
    thread_id: str
    message: str
    model: Optional[str] = "llama"
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 2048
    top_p: Optional[float] = 0.95
    top_k: Optional[int] = 40
    system_prompt: Optional[str] = ""

class CreateThreadInput(BaseModel):
    email: str
    title: Optional[str] = "New Chat"
    metadata: Optional[dict] = {}

class UserThreadsInput(BaseModel):
    email: str

class RagInput(BaseModel):
    thread_id: str
    document_id: str
    query: str
    model: Optional[str] = "llama"
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 2048
    top_p: Optional[float] = 0.95
    top_k: Optional[int] = 40
    system_prompt: Optional[str] = ""

class TestVectorstoreInput(BaseModel):
    thread_id: str
    document_id: str
    query: Optional[str] = "what is india goal for 2030"

class UpdateThreadTitleInput(BaseModel):
    thread_id: str
    title: str

class RegisterInput(BaseModel):
    email: str
    password: str
    name: Optional[str] = "User"

class LoginInput(BaseModel):
    email: str
    password: str

# ---------- Agent cache ----------
agent_clients = {}

def get_agent(model_key: str) -> ConversationalAgent:
    if model_key not in agent_clients:
        agent_clients[model_key] = ConversationalAgent(model_key)
    return agent_clients[model_key]

# Base directories
BASE_DATA_DIR = "llm_net_data"
UPLOAD_DIR = os.path.join(BASE_DATA_DIR, "uploads")
VECTORSTORE_DIR = os.path.join(BASE_DATA_DIR, "vectorstores")
Path(UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
Path(VECTORSTORE_DIR).mkdir(parents=True, exist_ok=True)

# ============================================================
# AUTH ROUTES  (previously main.py)
# ============================================================

@app.get("/login")
async def login(request: Request):
    """Redirect to Google OAuth."""
    redirect_uri = f"{BACKEND_URL}/callback"
    return await oauth.google.authorize_redirect(request, redirect_uri, prompt="select_account")


@app.get("/callback")
async def callback(request: Request):
    """Handle Google OAuth callback."""
    try:
        token = await oauth.google.authorize_access_token(request)
        userinfo = token.get("userinfo") or await oauth.google.userinfo(token=token)
    except Exception as e:
        logger.error("Authorization error: %s", e)
        raise HTTPException(status_code=400, detail=f"Authorization failed: {str(e)}")

    email = userinfo.get("email")
    if not email:
        raise HTTPException(status_code=400, detail="No email found in Google response")

    domain = email.split("@")[-1]
    if ALLOWED_DOMAIN and domain != ALLOWED_DOMAIN:
        return JSONResponse(
            status_code=403,
            content={"message": f"Access denied for domain: {domain}. Only {ALLOWED_DOMAIN} is allowed."}
        )

    request.session["user"] = {
        "email": email,
        "name": userinfo.get("name"),
        "sub": userinfo.get("sub"),
        "picture": userinfo.get("picture"),
    }

    # Upsert user in DB
    db = SessionLocal()
    try:
        existing = db.query(User).filter_by(email=email).first()
        if not existing:
            db.add(User(email=email, name=userinfo.get("name", "Unknown"), auth_provider="google"))
            db.commit()
    except Exception as e:
        logger.error("DB error: %s", e)
    finally:
        db.close()

    return RedirectResponse(url=FRONTEND_URL)


@app.get("/logout")
async def logout(request: Request):
    request.session.clear()
    return RedirectResponse(url=FRONTEND_URL)


@app.get("/api/user/me")
async def get_current_user(request: Request):
    user = request.session.get("user")
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    # Get actual auth provider from database
    email = user.get("email")
    db = SessionLocal()
    try:
        db_user = db.query(User).filter_by(email=email).first()
        auth_provider = db_user.auth_provider if db_user else "unknown"
        has_password = bool(db_user and db_user.password_hash and db_user.password_hash.strip()) if db_user else False
    finally:
        db.close()
    
    return {
        "authenticated": True,
        "email": user.get("email"),
        "name": user.get("name"),
        "google_id": user.get("sub"),
        "picture": user.get("picture"),
        "auth_provider": auth_provider,
        "can_update_password": True if not has_password else auth_provider == "local"
    }


@app.post("/api/register")
async def register(data: RegisterInput):
    email = data.email.strip().lower()
    if not email.endswith(f"@{ALLOWED_DOMAIN}"):
        raise HTTPException(status_code=403, detail=f"Only @{ALLOWED_DOMAIN} emails are allowed")

    db = SessionLocal()
    try:
        if db.query(User).filter_by(email=email).first():
            raise HTTPException(status_code=409, detail="Email already registered")
        user = User(
            email=email,
            name=data.name,
            password_hash=pwd_context.hash(data.password),
            auth_provider="local",
        )
        db.add(user)
        db.commit()
    finally:
        db.close()

    return JSONResponse(status_code=201, content={"message": "Registration successful"})


@app.post("/api/login")
async def login_email(data: LoginInput, request: Request):
    email = data.email.strip().lower()
    db = SessionLocal()
    try:
        user = db.query(User).filter_by(email=email, auth_provider="local").first()
        if not user or not pwd_context.verify(data.password, user.password_hash):
            raise HTTPException(status_code=401, detail="Invalid credentials")
        request.session["user"] = {
            "email": user.email,
            "name": user.name,
            "sub": user.id,
            "picture": None,
        }
    finally:
        db.close()

    return {"message": "Login successful"}


class PasswordUpdateInput(BaseModel):
    email: str
    new_password: str
    current_password: Optional[str] = None


@app.post("/api/update-password")
async def update_password(data: PasswordUpdateInput):
    """Update user password."""
    email = data.email.strip().lower()
    
    logger.info(f"Password update request for: {email}")
    
    if not email.endswith(f"@{ALLOWED_DOMAIN}"):
        logger.warning(f"Invalid domain: {email}")
        raise HTTPException(status_code=403, detail=f"Only @{ALLOWED_DOMAIN} emails are allowed")
    
    if len(data.new_password) < 6:
        logger.warning(f"Password too short for: {email}")
        raise HTTPException(status_code=400, detail="New password must be at least 6 characters")
    
    db = SessionLocal()
    try:
        user = db.query(User).filter_by(email=email).first()
        
        if not user:
            logger.warning(f"User not found: {email}")
            raise HTTPException(status_code=404, detail="User not found")
        
        logger.info(f"User found: {email}, auth_provider: {user.auth_provider}, has_password: {bool(user.password_hash)}")
        
        has_existing_password = bool(user.password_hash and user.password_hash.strip())
        
        if has_existing_password:
            # User has existing password - verify current password before changing
            if not data.current_password:
                logger.warning(f"Current password required for user {email} who has existing password")
                raise HTTPException(status_code=400, detail="Current password is required to update your password")
            
            logger.info(f"User {email} has existing password, verifying current password")
            try:
                if not pwd_context.verify(data.current_password, user.password_hash):
                    logger.warning(f"Invalid current password for: {email}")
                    raise HTTPException(status_code=401, detail="Current password is incorrect")
            except HTTPException:
                raise
            except Exception as e:
                if "hash could not be identified" in str(e):
                    logger.error(f"Password hash for {email} is corrupted: {str(e)}")
                    raise HTTPException(status_code=400, detail="Your password hash is corrupted. Please contact support to reset your password.")
                raise
        else:
            # User has no existing password - first time setting up
            logger.info(f"User {email} ({user.auth_provider}) setting password for first time")
        
        # Update password
        user.password_hash = pwd_context.hash(data.new_password)
        db.commit()
        
        logger.info(f"✅ Password updated successfully for user: {email}")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Password update error for {email}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to update password")
    finally:
        db.close()
    
    return JSONResponse(status_code=200, content={"message": "Password updated successfully"})

# ============================================================
# HEALTH / ROOT
# ============================================================

@app.get("/")
def root():
    return {"message": "✅ Chat Agent running!"}

@app.get("/health")
def health():
    return {"status": "healthy", "timestamp": datetime.datetime.utcnow().isoformat()}

# ============================================================
# USER & THREAD
# ============================================================

@app.post("/user/create")
def create_user(data: UserThreadsInput):
    try:
        user_id = get_user_by_email(data.email)
        return {"user_id": user_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"User creation failed: {str(e)}")

@app.post("/chat/create_thread")
def create_chat(data: CreateThreadInput):
    try:
        user_id = get_user_by_email(data.email)
        thread_id = create_thread(user_id, data.title, data.metadata or {})
        return {"thread_id": thread_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Thread creation failed: {str(e)}")

@app.get("/chat/threads")
def list_threads(email: str):
    try:
        user_id = get_user_by_email(email)
        threads = get_threads_by_user(user_id)
        return [{"thread_id": t_id, "title": title, "created_at": created_at} for t_id, title, created_at in threads]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list threads: {str(e)}")

@app.get("/chat/history")
def chat_history(thread_id: str):
    return get_messages_by_thread(thread_id)

@app.post("/chat/update_thread_title")
def update_thread_title_endpoint(data: UpdateThreadTitleInput):
    """
    Update the title of a chat thread.
    
    Args:
        thread_id: The ID of the thread to update
        title: The new title for the thread
    
    Returns:
        Success/error response with updated thread info
    """
    try:
        thread_id = data.thread_id.strip()
        new_title = data.title.strip()
        
        # Validate inputs
        if not thread_id:
            raise HTTPException(status_code=400, detail="thread_id is required")
        if not new_title:
            raise HTTPException(status_code=400, detail="title is required")
        if len(new_title) > 200:
            raise HTTPException(status_code=400, detail="title must be 200 characters or less")
        
        # Update the thread title in database
        result = update_thread_title(thread_id, new_title)
        
        if result:
            logger.info(f"✅ Thread title updated: {thread_id} -> '{new_title}'")
            return {
                "success": True,
                "message": "Thread title updated successfully",
                "thread_id": thread_id,
                "new_title": new_title
            }
        else:
            logger.warning(f"Thread title update returned None for: {thread_id}")
            raise HTTPException(status_code=500, detail="Failed to update thread title")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error updating thread title: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to update thread title: {str(e)}")

# ============================================================
# DOCUMENT UPLOAD
# ============================================================

@app.post("/chat/upload_doc")
async def upload_document_with_thread_context(thread_id: str = Form(), file: UploadFile = File(...)):
    logger.info(f"=== Starting upload for thread {thread_id}, file: {file.filename} ===")

    if not thread_id:
        raise HTTPException(status_code=400, detail="thread_id is required")

    thread_vectorstore_dir = os.path.join(VECTORSTORE_DIR, thread_id)
    os.makedirs(thread_vectorstore_dir, exist_ok=True)

    doc_id = str(uuid.uuid4())
    ext = os.path.splitext(file.filename)[1] or ""
    save_path = os.path.join(UPLOAD_DIR, f"{doc_id}{ext}")

    try:
        content = await file.read()
        with open(save_path, "wb") as f:
            f.write(content)

        file_size = os.path.getsize(save_path)
        logger.info(f"Saved {file.filename} -> {save_path} ({file_size} bytes)")

        if not os.path.exists(save_path):
            raise HTTPException(status_code=500, detail=f"File was not saved properly: {save_path}")

        documents = process_document(save_path, max_sentences=10)

        if not documents:
            preview_text = ""
            try:
                with open(save_path, "rb") as f:
                    preview_bytes = f.read(2048)
                preview_text = preview_bytes.decode("utf-8", errors="ignore")
            except Exception as e:
                preview_text = f"<failed to read preview: {e}>"

            raise HTTPException(
                status_code=500,
                detail=f"Document processed but no content found. File: {file.filename}, Size: {file_size} bytes, Preview: {preview_text[:200]}"
            )

        for doc in documents:
            doc.metadata.update({
                "source_doc_id": doc_id,
                "original_filename": file.filename,
                "upload_timestamp": datetime.datetime.now().isoformat(),
            })

        vectorstore = build_vectorstore(docs=documents, thread_id=thread_id, document_id=doc_id)

        if not vectorstore:
            raise HTTPException(status_code=500, detail="Vectorstore creation failed")

        add_message(thread_id, role="system", content=f"📄 Document '{file.filename}' uploaded successfully. Ready for Q&A!")

        return {
            "message": "Document processed successfully",
            "thread_id": thread_id,
            "document_id": doc_id,
            "filename": file.filename,
            "chunks": len(documents),
            "original_file_path": save_path,
            "vectorstore_path": thread_vectorstore_dir
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Upload processing failed for {file.filename}")
        try:
            if os.path.exists(save_path):
                os.remove(save_path)
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

# ============================================================
# TEST VECTORSTORE
# ============================================================

@app.post("/chat/test_vectorstore")
async def test_vectorstore(data: TestVectorstoreInput):
    thread_id = data.thread_id
    document_id = data.document_id
    query = data.query

    try:
        thread_vectorstore_dir = os.path.join(VECTORSTORE_DIR, thread_id)

        if not os.path.exists(thread_vectorstore_dir):
            return {
                "status": "error",
                "message": f"Thread vectorstore directory doesn't exist: {thread_vectorstore_dir}",
                "available_threads": os.listdir(VECTORSTORE_DIR) if os.path.exists(VECTORSTORE_DIR) else []
            }

        thread_contents = os.listdir(thread_vectorstore_dir)

        start = time.perf_counter()
        vectorstore = load_vectorstore(thread_id, document_id)
        load_time = time.perf_counter() - start

        if not vectorstore:
            return {
                "status": "error",
                "message": "load_vectorstore returned None",
                "thread_directory": thread_vectorstore_dir,
                "thread_contents": thread_contents
            }

        start = time.perf_counter()
        results = vectorstore.similarity_search(query, k=5)
        search_time = time.perf_counter() - start

        collection = vectorstore._collection
        all_data = collection.get()
        metadatas = all_data.get("metadatas", [])
        documents = all_data.get("documents", [])

        stats = {
            "total_chunks": collection.count(),
            "metadata_fields": list(metadatas[0].keys()) if metadatas and metadatas[0] else [],
            "sample_chunk": documents[0][:100] + "..." if documents else "No documents found"
        }

        return {
            "status": "success",
            "stats": stats,
            "timings": {
                "vectorstore_load_time": f"{load_time:.3f}s",
                "similarity_search_time": f"{search_time:.3f}s"
            },
            "test_query_results": {
                "query": query,
                "top_match": results[0].page_content[:200] + "..." if results else "No results",
                "metadata": results[0].metadata if results else {}
            },
            "debug_info": {
                "thread_directory": thread_vectorstore_dir,
                "thread_contents": thread_contents
            }
        }
    except Exception as e:
        return {"status": "error", "message": f"Test failed: {str(e)}", "error_type": type(e).__name__}

# ============================================================
# RAG ENDPOINT
# ============================================================

@app.post("/chat/rag")
async def context_aware_rag_streaming(data: RagInput):
    start_time = time.perf_counter()

    thread_id = data.thread_id
    document_id = data.document_id
    query = data.query
    model_key = data.model or "llama"
    temperature = data.temperature or 0.7
    max_tokens = data.max_tokens or 3000
    system_prompt = data.system_prompt or ""

    model = MODEL_REGISTRY.get(model_key, model_key)

    add_message(thread_id, role="human", content=query)

    raw_history = get_messages_by_thread(thread_id)
    user_messages = [msg for msg in raw_history if msg["role"] == "human"]
    if len(user_messages) == 1:
        try:
            update_thread_title(thread_id, query[:60])
        except Exception as e:
            print(f"Failed to update thread title: {e}")

    vectorstore = load_vectorstore(thread_id, document_id)
    if not vectorstore:
        return {"status": "error", "message": "Could not load vectorstore"}

    retrieved_docs = vectorstore.similarity_search(query, k=5)
    raw_history = get_messages_by_thread(thread_id)

    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    conversation_history = "\n\nPrevious conversation:\n" + "\n".join(
        f"{'User' if msg['role'] == 'human' else 'Assistant'}: {msg['content']}"
        for msg in raw_history[:-1]
    ) if len(raw_history) > 1 else ""

    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "token-abc123")
    YOUR_API_KEY = os.getenv("YOUR_API_KEY", "token-abc123")
    api_gateway_url = API_PORT_MAPPING.get(model_key, API_PORT_MAPPING["gpt-oss"])

    llm = ChatOpenAI(
        model=model,
        openai_api_base=f"{api_gateway_url}/v1",
        openai_api_key=YOUR_API_KEY,
        temperature=temperature,
        max_tokens=max_tokens,
        streaming=True,
    )

    prompt = f"""Document Context:
{context}
{conversation_history}
Current Question: {query}"""

    def stream_response():
        full_content = ""
        try:
            for chunk in llm.stream(prompt):
                token = chunk.content if hasattr(chunk, "content") else str(chunk)
                full_content += token
                yield f"data: {json.dumps({'token': token})}\n\n"

            if full_content.strip():
                add_message(thread_id, role="ai", content=full_content.strip())

            yield "data: [DONE]\n\n"

        except Exception as e:
            add_message(thread_id, role="ai", content=f"Error: {str(e)}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        stream_response(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )

# ============================================================
# RAG HISTORY
# ============================================================

@app.get("/chat/rag_history")
def get_rag_history(thread_id: str):
    try:
        messages = get_messages_by_thread(thread_id)
        formatted_history = [
            {
                "role": msg["role"],
                "content": msg["content"],
                "timestamp": msg["created_at"],
                "message_type": "rag_conversation"
            }
            for msg in messages
        ]
        return {"thread_id": thread_id, "message_count": len(formatted_history), "messages": formatted_history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve history: {str(e)}")

@app.delete("/chat/rag_history/{thread_id}")
def clear_rag_history(thread_id: str):
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM messages WHERE thread_id = %s", (thread_id,))
        count_before = cursor.fetchone()[0]
        cursor.execute("DELETE FROM messages WHERE thread_id = %s", (thread_id,))
        conn.commit()
        cursor.close()
        conn.close()
        return {"message": f"Cleared {count_before} messages from thread {thread_id}", "thread_id": thread_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear history: {str(e)}")

# ============================================================
# AGENTIC RAG (stub)
# ============================================================

@app.post("/chat/agentic_rag")
def agentic_rag(payload: RagInput):
    return {"status": "error", "message": "This endpoint is not implemented yet."}

# ============================================================
# CHAT STREAMING
# ============================================================

@app.post("/chat/send")
async def chat_send(payload: MessageInput):
    t0 = time.time()
    thread_id = payload.thread_id
    question = payload.message
    selected_model = payload.model or "llama"
    temperature = payload.temperature or 0.7
    max_tokens = payload.max_tokens or 4000
    system_prompt = payload.system_prompt or "You are a helpful assistant."

    add_message(thread_id, role="human", content=question)

    raw_history = get_messages_by_thread(thread_id)
    user_messages = [msg for msg in raw_history if msg["role"] == "human"]
    if len(user_messages) == 1:
        try:
            update_thread_title(thread_id, question[:60])
        except Exception as e:
            print(f"Failed to update thread title: {e}")

    messages = format_messages(raw_history, system_prompt)
    agent = get_agent(selected_model)

    async def sse_stream():
        full_content = ""
        try:
            async for item in agent.astream_response(messages, temperature=temperature, max_tokens=max_tokens):
                if isinstance(item, dict):
                    yield f"data: {json.dumps(item)}\n\n"
                    if "token" in item:
                        full_content += item["token"]
                else:
                    full_content += str(item)
                    yield f"data: {json.dumps({'token': str(item)})}\n\n"

            if full_content:
                add_message(thread_id, role="ai", content=full_content)

            print(f"TOTAL /chat/send time: {time.time() - t0:.3f} sec")
            yield "data: [DONE]\n\n"

        except Exception as exc:
            print(f"Error in SSE stream: {exc}")
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"

    return StreamingResponse(
        sse_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )

# ============================================================
# SHUTDOWN
# ============================================================

@app.on_event("shutdown")
async def shutdown_event():
    for agent in agent_clients.values():
        await agent.close()
