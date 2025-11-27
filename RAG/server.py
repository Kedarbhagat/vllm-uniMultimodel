import json
import datetime
import time
import logging
from typing import Optional, List
from fastapi import Form, UploadFile, File, HTTPException
from pathlib import Path 
import os, uuid
from tempfile import NamedTemporaryFile
from langchain_core.runnables import RunnableLambda
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import Document as LangchainDocument
from langchain_core.prompts import PromptTemplate

from langchain_openai import ChatOpenAI
import requests

from vectorestore import build_vectorstore, load_vectorstore
from wordloader import process_document 

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from agent import graph, reconstruct_history, ConversationalAgent, prompt_template
from dbhelper import get_connection, get_threads_by_user, get_user_by_email, create_thread, add_message, get_messages_by_thread, update_thread_title

from prometheus_fastapi_instrumentator import Instrumentator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI()

# Enable Prometheus metrics exporter
Instrumentator().instrument(app).expose(app)

# -----------Models-----------
MODEL_REGISTRY = {
    "llama": "./meta-llama/Llama-3.1-8B-Instruct-awq",
    "deepseek": "./DeepSeek-Coder-V2-Lite-Instruct-awq",
    "qwen2.5": "/mnt/c/Users/STUDENT/qwen2.5-coder-14b-instruct-awq-final"
}

# ---------- CORS ----------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

# Base directories
BASE_DATA_DIR = "llm_net_data"
UPLOAD_DIR = os.path.join(BASE_DATA_DIR, "uploads")
VECTORSTORE_DIR = os.path.join(BASE_DATA_DIR, "vectorstores")
Path(UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
Path(VECTORSTORE_DIR).mkdir(parents=True, exist_ok=True)

# ---------- Health Check ----------
@app.get("/")
def root():
    return {"message": "âœ… Chat Agent running!"}

# ---------- User & Thread ----------
@app.post("/user/create")
def create_user(data: UserThreadsInput):
    """Create or get user safely"""
    try:
        user_id = get_user_by_email(data.email)
        return {"user_id": user_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"User creation failed: {str(e)}")

@app.post("/chat/create_thread")
def create_chat(data: CreateThreadInput):
    try:
        user_id = get_user_by_email(data.email)
        return {"thread_id": create_thread(user_id, data.title, data.metadata or {})}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Thread creation failed: {str(e)}")

@app.get("/chat/threads")
def list_threads(email: str):
    """List threads for a user safely"""
    try:
        user_id = get_user_by_email(email)
        threads = get_threads_by_user(user_id)
        return [{"thread_id": t_id, "title": title, "created_at": created_at} for t_id, title, created_at in threads]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list threads: {str(e)}")

@app.get("/chat/history")
def chat_history(thread_id: str):
    return get_messages_by_thread(thread_id)

# ---------- Document Upload ----------
@app.post("/chat/upload_doc")
async def upload_document_with_thread_context(thread_id: str = Form(), file: UploadFile = File(...)):
    """
    Save uploaded file, run process_document, return helpful diagnostics if processing fails.
    """
    logger.info(f"=== Starting upload for thread {thread_id}, file: {file.filename} ===")
    
    if not thread_id:
        raise HTTPException(status_code=400, detail="thread_id is required")

    thread_vectorstore_dir = os.path.join(VECTORSTORE_DIR, thread_id)
    os.makedirs(thread_vectorstore_dir, exist_ok=True)

    doc_id = str(uuid.uuid4())
    ext = os.path.splitext(file.filename)[1] or ""
    save_path = os.path.join(UPLOAD_DIR, f"{doc_id}{ext}")

    logger.info(f"File extension detected: {ext}")
    logger.info(f"Save path: {save_path}")

    try:
        # Save file
        content = await file.read()
        with open(save_path, "wb") as f:
            f.write(content)

        file_size = os.path.getsize(save_path)
        logger.info(f"Saved upload {file.filename} -> {save_path} ({file_size} bytes)")

        # Check if file exists and is readable
        if not os.path.exists(save_path):
            raise HTTPException(status_code=500, detail=f"File was not saved properly: {save_path}")

        # Run document processing
        logger.info(f"Starting document processing for {save_path}")
        documents = process_document(save_path, max_sentences=10)
        logger.info(f"Document processing completed. Found {len(documents) if documents else 0} documents")

        if not documents:
            logger.error(f"No documents extracted from {file.filename}")
            
            preview_text = ""
            try:
                with open(save_path, "rb") as f:
                    preview_bytes = f.read(2048)
                try:
                    preview_text = preview_bytes.decode("utf-8", errors="ignore")
                except Exception:
                    preview_text = str(preview_bytes[:200])
            except Exception as e:
                preview_text = f"<failed to read preview: {e}>"

            raise HTTPException(
                status_code=500,
                detail=f"Document processed but no content found. File: {file.filename}, Size: {file_size} bytes, Preview: {preview_text[:200]}"
            )

        # Attach metadata and build vectorstore
        logger.info(f"Adding metadata to {len(documents)} documents")
        for doc in documents:
            doc.metadata.update({
                "source_doc_id": doc_id,
                "original_filename": file.filename,
                "upload_timestamp": datetime.datetime.now().isoformat(),
            })

        logger.info(f"Building vectorstore for thread {thread_id}")
        vectorstore = build_vectorstore(
            docs=documents,
            thread_id=thread_id,
            document_id=doc_id
        )

        if not vectorstore:
            raise HTTPException(status_code=500, detail="Vectorstore creation failed")

        # Log document upload as a system message
        upload_message = f"ðŸ“„ Document '{file.filename}' uploaded successfully. Ready for Q&A!"
        add_message(thread_id, role="system", content=upload_message)

        logger.info(f"=== Upload completed successfully for {file.filename} ===")

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
        # Clean up on error
        try:
            if os.path.exists(save_path):
                os.remove(save_path)
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

# ---------- Test Vectorstore ----------
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
        return {
            "status": "error",
            "message": f"Test failed: {str(e)}",
            "error_type": type(e).__name__
        }

# ---------- RAG Endpoint ----------
@app.post("/chat/rag")
async def context_aware_rag_streaming(data: RagInput):
    print(f"\n=== Starting RAG processing at {datetime.datetime.now().isoformat()} ===")
    start_time = time.perf_counter()
    
    thread_id = data.thread_id
    document_id = data.document_id
    query = data.query
    model = data.model or "llama"
    temperature = data.temperature or 0.7
    max_tokens = data.max_tokens or 3000
    top_p = data.top_p or 0.95
    top_k = data.top_k or 40
    system_prompt = data.system_prompt or ""

    if model == "llama":
        model = MODEL_REGISTRY["llama"]
    elif model == "deepseek":
        model = MODEL_REGISTRY["deepseek"]
    elif model == "qwen2.5":
        model = MODEL_REGISTRY["qwen2.5"]

    print(f"\n[1/6] Parameters received - Thread: {thread_id}, Doc: {document_id}, Query: '{query}'")

    # Store user query
    db_start = time.perf_counter()
    add_message(thread_id, role="human", content=query)
    print(f"[2/6] Database insert time: {(time.perf_counter() - db_start)*1000:.2f}ms")

    # Update thread title if this is the first user message
    raw_history = get_messages_by_thread(thread_id)
    user_messages = [msg for msg in raw_history if msg["role"] == "human"]
    if len(user_messages) == 1:
        try:
            update_thread_title(thread_id, query[:60])
        except Exception as e:
            print(f"Failed to update thread title: {e}")

    # Load vectorstore
    print("\n[3/6] Loading vectorstore...")
    vs_load_start = time.perf_counter()
    vectorstore = load_vectorstore(thread_id, document_id)
    vs_load_time = time.perf_counter() - vs_load_start
    print(f"  â†’ Vectorstore load completed in {vs_load_time:.2f}s")
    
    if not vectorstore:
        print("  âŒ Error: Vectorstore load failed!")
        return {"status": "error", "message": "Could not load vectorstore"}

    # Similarity search
    print("\n[4/6] Performing similarity search...")
    search_start = time.perf_counter()
    retrieved_docs = vectorstore.similarity_search(query, k=5)
    search_time = time.perf_counter() - search_start
    print(f"  â†’ Found {len(retrieved_docs)} docs in {search_time:.2f}s")
    print(f"  â†’ First doc content preview: {retrieved_docs[0].page_content[:100]}...")

    # Get chat history
    print("\n[5/6] Retrieving chat history...")
    history_start = time.perf_counter()
    raw_history = get_messages_by_thread(thread_id)
    history_time = time.perf_counter() - history_start
    print(f"  â†’ Retrieved {len(raw_history)} messages in {history_time:.2f}s")

    # Prepare context
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    conversation_history = "\n\nPrevious conversation:\n" + "\n".join(
        f"{'User' if msg['role'] == 'human' else 'Assistant'}: {msg['content']}" 
        for msg in raw_history[:-1]
    ) if len(raw_history) > 1 else ""

    # LLM setup
    print("\n[6/6] Setting up LLM...")
    llm_setup_start = time.perf_counter()
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "token-abc123")
    YOUR_API_KEY = os.getenv("YOUR_API_KEY", "token-abc123")
    API_GATEWAY_URL = "http://192.168.190.28:8082"

    llm = ChatOpenAI(
        model=model,
        openai_api_base=f"{API_GATEWAY_URL}/v1",
        openai_api_key=YOUR_API_KEY,
        temperature=temperature,
        max_tokens=max_tokens,
        streaming=True,
    )
    print(f"  â†’ LLM configured in {(time.perf_counter() - llm_setup_start)*1000:.2f}ms")

    # Build prompt
    prompt = f"""Document Context:
{context}
{conversation_history}
Current Question: {query}"""

    print(f"\n=== Pre-streaming phase completed in {(time.perf_counter() - start_time):.2f}s ===")
    print(f"Prompt length: {len(prompt)} characters")
    print(f"Starting streaming response...\n")

    def stream_response():
        full_content = ""
        stream_start = time.perf_counter()
        first_token_received = False
        
        try:
            for chunk in llm.stream(prompt):
                token = chunk.content if hasattr(chunk, "content") else str(chunk)
                full_content += token
                
                if not first_token_received:
                    first_token_time = (time.perf_counter() - stream_start)*1000
                    print(f"  â†’ First token received in {first_token_time:.2f}ms")
                    first_token_received = True
                
                yield f"data: {json.dumps({'token': token})}\n\n"
            
            total_stream_time = time.perf_counter() - stream_start
            print(f"\n  â†’ Streaming completed in {total_stream_time:.2f}s")
            print(f"  â†’ Total response length: {len(full_content)} characters")
            
            # Store response
            db_store_start = time.perf_counter()
            if full_content.strip():
                add_message(thread_id, role="ai", content=full_content.strip())
            print(f"  â†’ Database storage time: {(time.perf_counter() - db_store_start)*1000:.2f}ms")
            
            yield f"data: [DONE]\n\n"
            
        except Exception as e:
            error_time = time.perf_counter() - start_time
            print(f"\nâŒ Error after {error_time:.2f}s: {str(e)}")
            add_message(thread_id, role="ai", content=f"Error: {str(e)}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        stream_response(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "*",
        }
    )

# ---------- RAG History ----------
@app.get("/chat/rag_history")
def get_rag_history(thread_id: str):
    """Get chat history for a specific thread with RAG context"""
    try:
        messages = get_messages_by_thread(thread_id)
        
        formatted_history = []
        for msg in messages:
            formatted_msg = {
                "role": msg["role"],
                "content": msg["content"],
                "timestamp": msg["created_at"],
                "message_type": "rag_conversation"
            }
            formatted_history.append(formatted_msg)
        
        return {
            "thread_id": thread_id,
            "message_count": len(formatted_history),
            "messages": formatted_history
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve history: {str(e)}")

@app.delete("/chat/rag_history/{thread_id}")
def clear_rag_history(thread_id: str):
    """Clear all messages for a specific thread"""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM messages WHERE thread_id = %s", (thread_id,))
        count_before = cursor.fetchone()[0]
        
        cursor.execute("DELETE FROM messages WHERE thread_id = %s", (thread_id,))
        conn.commit()
        
        cursor.close()
        conn.close()
        
        return {
            "message": f"Cleared {count_before} messages from thread {thread_id}",
            "thread_id": thread_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear history: {str(e)}")

# ---------- Helper Functions ----------
def get_conversation_summary(thread_id: str, max_messages: int = 10) -> str:
    """Get a summary of recent conversation for context"""
    try:
        messages = get_messages_by_thread(thread_id)
        
        if not messages:
            return ""
        
        recent_messages = [
            msg for msg in messages[-max_messages:] 
            if msg["role"] in ["human", "ai"]
        ]
        
        if not recent_messages:
            return ""
        
        summary = "Recent conversation context:\n"
        for msg in recent_messages:
            role = "User" if msg["role"] == "human" else "Assistant"
            content = msg["content"][:200] + "..." if len(msg["content"]) > 200 else msg["content"]
            summary += f"{role}: {content}\n"
        
        return summary
    except Exception:
        return ""

# ---------- Agentic RAG (Not implemented) ----------
@app.post("/chat/agentic_rag")
def agentic_rag(payload: RagInput):
    thread_id = payload.thread_id
    document_id = payload.document_id
    query = payload.query
    selected_model = payload.model or "llama"
    
    return {"status": "error", "message": "This endpoint is not implemented yet."}

# ---------- Chat Streaming Endpoint ----------
@app.post("/chat/send")
def chat_send(payload: MessageInput):
    t0 = time.time()
    thread_id = payload.thread_id
    question = payload.message
    selected_model = payload.model or "llama"
    temperature = payload.temperature or 0.7
    max_tokens = payload.max_tokens or 2048
    top_p = payload.top_p or 0.95
    top_k = payload.top_k or 40
    system_prompt = payload.system_prompt or ""
    
    print("selected_model:", selected_model)
    
    # Add human message to DB
    t1 = time.time()
    add_message(thread_id, role="human", content=question)
    print(f"DB insert (human message): {time.time() - t1:.3f} sec")

    # Retrieve conversation history from DB
    t2 = time.time()
    raw_history = get_messages_by_thread(thread_id)
    print(f"DB retrieval (messages): {time.time() - t2:.3f} sec")

    # Update thread title if this is the first user message
    user_messages = [msg for msg in raw_history if msg["role"] == "human"]
    if len(user_messages) == 1:
        try:
            update_thread_title(thread_id, question[:60])
        except Exception as e:
            print(f"Failed to update thread title: {e}")

    # Reconstruct history
    t3 = time.time()
    history = reconstruct_history(raw_history)
    print(f"History reconstruction: {time.time() - t3:.3f} sec")

    # Initialize agent
    t4 = time.time()
    agent = ConversationalAgent(prompt_template, selected_model)
    print(f"Agent initialization: {time.time() - t4:.3f} sec")

    def sse_stream():
        t5 = time.time()
        full_content = ""
        try:
            for token in agent.stream_response_for_web(history):
                full_content += token
                yield f"data: {json.dumps({'token': token})}\n\n"
            print(f"LLM streaming time: {time.time() - t5:.3f} sec")

            # Save AI message to DB
            t6 = time.time()
            if full_content:
                add_message(thread_id, role="ai", content=full_content)
            print(f"DB insert (AI message): {time.time() - t6:.3f} sec")

            print(f"TOTAL /chat/send time: {time.time() - t0:.3f} sec")

            yield "data: [DONE]\n\n"
        except GeneratorExit:
            print("âš ï¸ Client disconnected during SSE stream")
            raise
        except Exception as exc:
            print(f"Error in SSE stream: {exc}")
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"

    return StreamingResponse(
        sse_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "*",
        }
    )