import json
import datetime
import time
from typing import Optional
from fastapi import Form, UploadFile, File, HTTPException, logger
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

from vectorestore import build_vectorstore
from wordloader import process_document 

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from fastapi import HTTPException
from vectorestore import load_vectorstore
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from agent import graph, reconstruct_history, ConversationalAgent, prompt_template
from dbhelper import get_connection, get_threads_by_user, get_user_by_email, create_thread, add_message, get_messages_by_thread, update_thread_title

from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI(title="Simple Conversational Agent")
Instrumentator().instrument(app).expose(app)
# -----------Models-----------
MODEL_REGISTRY = {
    "llama": "./meta-llama/Llama-3.1-8B-Instruct-awq",
    "deepseek": "./DeepSeek-Coder-V2-Lite-Instruct-awq"
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


os.environ["LANGCHAIN_TRACING_V2"] = "true" # Enable LangSmith tracing for LangChain/LangGraph
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "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" # Replace with your actual key
os.environ["LANGCHAIN_PROJECT"] = "pr-healthy-nephew-28"

# ---------- Health Check ----------
@app.get("/")
def root():
    return {"message": "✅ Chat Agent running!"}

# ---------- User & Thread ----------
@app.post("/user/create")
def create_user(data: UserThreadsInput):
    """Create or get user safely"""
    try:
        user_id = get_user_by_email(data.email)  # already a UUID string
        return {"user_id": user_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"User creation failed: {str(e)}")


@app.post("/chat/create_thread")
def create_chat(data: CreateThreadInput):
    try:
        user_id = get_user_by_email(data.email)  # UUID string only
        return {"thread_id": create_thread(user_id, data.title, data.metadata or {})}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Thread creation failed: {str(e)}")


@app.get("/chat/threads")
def list_threads(email: str):
    """List threads for a user safely"""
    try:
        user_id = get_user_by_email(email)  # UUID string only
        threads = get_threads_by_user(user_id)
        return [{"thread_id": t_id, "title": title,"created_at" :created_at} for t_id, title,created_at in threads]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list threads: {str(e)}")


@app.get("/chat/history")
def chat_history(thread_id: str):
    return get_messages_by_thread(thread_id)



BASE_DATA_DIR = "llm_net_data"
UPLOAD_DIR = os.path.join(BASE_DATA_DIR, "uploads")
VECTORSTORE_DIR = os.path.join(BASE_DATA_DIR, "vectorstores")
Path(UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
Path(VECTORSTORE_DIR).mkdir(parents=True, exist_ok=True)
'''
# ---------- Document Upload ----------
@app.post("/chat/upload_doc")
async def upload_document(thread_id: str = Form(), file: UploadFile = File(...)):
    if not thread_id:
        raise HTTPException(status_code=400, detail="thread_id is required")

    # Create thread-specific vectorstore directory
    thread_vectorstore_dir = os.path.join(VECTORSTORE_DIR, thread_id)
    os.makedirs(thread_vectorstore_dir, exist_ok=True)

    # Save original file
    doc_id = str(uuid.uuid4())
    ext = os.path.splitext(file.filename)[1]
    save_path = os.path.join(UPLOAD_DIR, f"{doc_id}{ext}")

    try:
        with open(save_path, "wb") as f:
            f.write(await file.read())

        documents = process_document(save_path, max_sentences=10)
        if not documents:
            raise HTTPException(status_code=500, detail="Document processed but no content found.")

        # Add metadata to each document
        for doc in documents:
            doc.metadata.update({
                "source_doc_id": doc_id,
                "original_filename": file.filename,
                "upload_timestamp": datetime.datetime.now().isoformat(),
            })

        vectorstore = build_vectorstore(
                        docs=documents,
                        thread_id=thread_id,
                        document_id=doc_id
                    )


        if not vectorstore:
            raise HTTPException(status_code=500, detail="Vectorstore creation failed")

        return {
            "message": "Document processed successfully",
            "thread_id": thread_id,
            "document_id": doc_id,
            "filename": file.filename,
            "chunks": len(documents),
            "original_file_path": save_path,
            "vectorstore_path": thread_vectorstore_dir
        }

    except Exception as e:
        # Clean up if something failed
        if os.path.exists(save_path):
            os.remove(save_path)
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
    '''
# --- test the vectorestore---





# Add this schema class with your other schemas
class TestVectorstoreInput(BaseModel):
    thread_id: str
    document_id: str
    query: Optional[str] = "what is india goal for 2030"

# Fixed test endpoint

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

# ---------- Simple RAG Endpoint ----------
# Updated RAG endpoint with chat history context awareness

@app.post("/chat/rag")
async def context_aware_rag_streaming(data: RagInput):
    print(f"\n=== Starting RAG processing at {datetime.datetime.now().isoformat()} ===")
    start_time = time.perf_counter()
    
    thread_id = data.thread_id
    document_id = data.document_id
    query = data.query
    model = data.model or "llama"

    if(model=="llama"):
        model = MODEL_REGISTRY["llama"]
    elif(model=="deepseek"):
        model = MODEL_REGISTRY["deepseek"]

    print(f"\n[1/6] Parameters received - Thread: {thread_id}, Doc: {document_id}, Query: '{query}'")

    # Store user query
    db_start = time.perf_counter()
    add_message(thread_id, role="human", content=query)
    print(f"[2/6] Database insert time: {(time.perf_counter() - db_start)*1000:.2f}ms")

    # --- Update thread title if this is the first user message ---
    raw_history = get_messages_by_thread(thread_id)
    user_messages = [msg for msg in raw_history if msg["role"] == "human"]
    if len(user_messages) == 1:
        try:
            update_thread_title(thread_id, query[:60])  # Limit title length
        except Exception as e:
            print(f"Failed to update thread title: {e}")

    # Load vectorstore
    print("\n[3/6] Loading vectorstore...")
    vs_load_start = time.perf_counter()
    vectorstore = load_vectorstore(thread_id, document_id)
    vs_load_time = time.perf_counter() - vs_load_start
    print(f"  → Vectorstore load completed in {vs_load_time:.2f}s")
    
    if not vectorstore:
        print("  ❌ Error: Vectorstore load failed!")
        return {"status": "error", "message": "Could not load vectorstore"}

    # Similarity search
    print("\n[4/6] Performing similarity search...")
    search_start = time.perf_counter()
    retrieved_docs = vectorstore.similarity_search(query, k=5)
    search_time = time.perf_counter() - search_start
    print(f"  → Found {len(retrieved_docs)} docs in {search_time:.2f}s")
    print(f"  → First doc content preview: {retrieved_docs[0].page_content[:100]}...")

    # Get chat history
    print("\n[5/6] Retrieving chat history...")
    history_start = time.perf_counter()
    raw_history = get_messages_by_thread(thread_id)
    history_time = time.perf_counter() - history_start
    print(f"  → Retrieved {len(raw_history)} messages in {history_time:.2f}s")

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
    API_GATEWAY_URL = "http://172.17.35.82:8082"

    llm = ChatOpenAI(
        model=model,
        openai_api_base=f"{API_GATEWAY_URL}/v1",
        openai_api_key=YOUR_API_KEY,
        temperature=0.7,
        max_tokens=2048,
        streaming=True,
    )
    print(f"  → LLM configured in {(time.perf_counter() - llm_setup_start)*1000:.2f}ms")

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
                    print(f"  → First token received in {first_token_time:.2f}ms")
                    first_token_received = True
                
                yield f"data: {json.dumps({'token': token})}\n\n"
            
            total_stream_time = time.perf_counter() - stream_start
            print(f"\n  → Streaming completed in {total_stream_time:.2f}s")
            print(f"  → Total response length: {len(full_content)} characters")
            
            # Store response
            db_store_start = time.perf_counter()
            if full_content.strip():
                add_message(thread_id, role="ai", content=full_content.strip())
            print(f"  → Database storage time: {(time.perf_counter() - db_store_start)*1000:.2f}ms")
            
            yield f"data: [DONE]\n\n"
            
        except Exception as e:
            error_time = time.perf_counter() - start_time
            print(f"\n❌ Error after {error_time:.2f}s: {str(e)}")
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
# Additional endpoint to get RAG chat history with document context
@app.get("/chat/rag_history")
def get_rag_history(thread_id: str):
    """Get chat history for a specific thread with RAG context"""
    try:
        messages = get_messages_by_thread(thread_id)
        
        # Format messages for better readability
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

# Enhanced endpoint to clear RAG conversation history
@app.delete("/chat/rag_history/{thread_id}")
def clear_rag_history(thread_id: str):
    """Clear all messages for a specific thread"""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # Get count before deletion
        cursor.execute("SELECT COUNT(*) FROM messages WHERE thread_id = %s", (thread_id,))
        count_before = cursor.fetchone()[0]
        
        # Delete messages
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

# Enhanced document upload with thread association
@app.post("/chat/upload_doc")
async def upload_document_with_thread_context(thread_id: str = Form(), file: UploadFile = File(...)):
    if not thread_id:
        raise HTTPException(status_code=400, detail="thread_id is required")

    # Create thread-specific vectorstore directory
    thread_vectorstore_dir = os.path.join(VECTORSTORE_DIR, thread_id)
    os.makedirs(thread_vectorstore_dir, exist_ok=True)

    # Save original file
    doc_id = str(uuid.uuid4())
    ext = os.path.splitext(file.filename)[1]
    save_path = os.path.join(UPLOAD_DIR, f"{doc_id}{ext}")

    try:
        with open(save_path, "wb") as f:
            f.write(await file.read())

        documents = process_document(save_path, max_sentences=10)
        if not documents:
            raise HTTPException(status_code=500, detail="Document processed but no content found.")

        # Add metadata to each document
        for doc in documents:
            doc.metadata.update({
                "source_doc_id": doc_id,
                "original_filename": file.filename,
                "upload_timestamp": datetime.datetime.now().isoformat(),
                "thread_id": thread_id,  # Associate with thread
            })

        vectorstore = build_vectorstore(
            docs=documents,
            thread_id=thread_id,
            document_id=doc_id
        )

        if not vectorstore:
            raise HTTPException(status_code=500, detail="Vectorstore creation failed")

        # Log document upload as a system message
        upload_message = f"📄 Document '{file.filename}' uploaded successfully. Ready for Q&A!"
        add_message(thread_id, role="system", content=upload_message)

        return {
            "message": "Document processed successfully",
            "thread_id": thread_id,
            "document_id": doc_id,
            "filename": file.filename,
            "chunks": len(documents),
            "original_file_path": save_path,
            "vectorstore_path": thread_vectorstore_dir
        }

    except Exception as e:
        # Clean up if something failed
        if os.path.exists(save_path):
            os.remove(save_path)
        
        # Log error as system message
        error_message = f"❌ Failed to upload document '{file.filename}': {str(e)}"
        add_message(thread_id, role="system", content=error_message)
        
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

# Helper function to get conversation summary
def get_conversation_summary(thread_id: str, max_messages: int = 10) -> str:
    """Get a summary of recent conversation for context"""
    try:
        messages = get_messages_by_thread(thread_id)
        
        if not messages:
            return ""
        
        # Get last max_messages (excluding system messages)
        recent_messages = [
            msg for msg in messages[-max_messages:] 
            if msg["role"] in ["human", "ai"]
        ]
        
        if not recent_messages:
            return ""
        
        summary = "Recent conversation context:\n"
        for msg in recent_messages:
            role = "User" if msg["role"] == "human" else "Assistant"
            # Truncate long messages
            content = msg["content"][:200] + "..." if len(msg["content"]) > 200 else msg["content"]
            summary += f"{role}: {content}\n"
        
        return summary
    except Exception:
        return ""
#-------------------agentic rag endpoint-------------------
@app.post("/chat/agentic_rag")
def agentic_rag(payload: RagInput):
    thread_id = payload.thread_id
    document_id = payload.document_id
    query = payload.query
    selected_model = payload.model or "llama"

    # too much latency in this endpoint, so not implemented yet
    #use simple rag for now 
    return {"status": "error", "message": "This endpoint is not implemented yettt."}


# ---------- Chat Streaming Endpoint ----------
import time
import logging

logging.basicConfig(level=logging.INFO)

@app.post("/chat/send")
def chat_send(payload: MessageInput):
    t0 = time.time()
    thread_id = payload.thread_id
    question = payload.message
    selected_model = payload.model or "llama"

    # 1. Add human message to DB
    t1 = time.time()
    add_message(thread_id, role="human", content=question)
    print(f"DB insert (human message): {time.time() - t1:.3f} sec")

    # 2. Retrieve conversation history from DB
    t2 = time.time()
    raw_history = get_messages_by_thread(thread_id)
    print(f"DB retrieval (messages): {time.time() - t2:.3f} sec")

    # --- Update thread title if this is the first user message ---
    user_messages = [msg for msg in raw_history if msg["role"] == "human"]
    if len(user_messages) == 1:
        try:
            update_thread_title(thread_id, question[:60])  # Limit title length
        except Exception as e:
            print(f"Failed to update thread title: {e}")

    # 3. Reconstruct history
    t3 = time.time()
    history = reconstruct_history(raw_history)
    print(f"History reconstruction: {time.time() - t3:.3f} sec")

    # 4. Initialize agent
    t4 = time.time()
    agent = ConversationalAgent(prompt_template, selected_model)
    print(f"Agent initialization: {time.time() - t4:.3f} sec")

    def sse_stream():
        # 5. LLM streaming
        t5 = time.time()
        full_content = ""
        try:
            for token in agent.stream_response_for_web(history):
                full_content += token
                yield f"data: {json.dumps({'token': token})}\n\n"
            print(f"LLM streaming time: {time.time() - t5:.3f} sec")

            # 6. Save AI message to DB
            t6 = time.time()
            if full_content:
                add_message(thread_id, role="ai", content=full_content)
            print(f"DB insert (AI message): {time.time() - t6:.3f} sec")

            # 7. Total end-to-end time
            print(f"TOTAL /chat/send time: {time.time() - t0:.3f} sec")

            yield "data: [DONE]\n\n"
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

