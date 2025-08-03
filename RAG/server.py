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
from langchain.chat_models import ChatOpenAI
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
from dbhelper import (
    get_or_create_user, create_thread, get_threads_by_user,
    add_message, get_messages_by_thread
)

app = FastAPI(title="Simple Conversational Agent")
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
    return {"user_id": get_or_create_user(data.email)}

@app.post("/chat/create_thread")
def create_chat(data: CreateThreadInput):
    user_id = get_or_create_user(data.email)
    return {"thread_id": create_thread(user_id, data.title, data.metadata or {})}

@app.get("/chat/threads")
def list_threads(email: str):
    uid = get_or_create_user(email)
    return [{"thread_id": t_id, "title": title} for t_id, title in get_threads_by_user(uid)]

@app.get("/chat/history")
def chat_history(thread_id: str):
    return get_messages_by_thread(thread_id)




BASE_DATA_DIR = "llm_net_data"
UPLOAD_DIR = os.path.join(BASE_DATA_DIR, "uploads")
VECTORSTORE_DIR = os.path.join(BASE_DATA_DIR, "vectorstores")
Path(UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
Path(VECTORSTORE_DIR).mkdir(parents=True, exist_ok=True)

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
    
# --- test the vectorestore---





# Add this schema class with your other schemas
class TestVectorstoreInput(BaseModel):
    thread_id: str
    document_id: str
    query: Optional[str] = "what is india goal for 2030"

# Fixed test endpoint
@app.post("/chat/test_vectorstore")
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

@app.post("/chat/rag")
async def simple_rag_streaming(data: RagInput):
    thread_id = data.thread_id
    document_id = data.document_id
    query = data.query
    model = data.model or "llama"

    # Load vectorstore
    vectorstore = load_vectorstore(thread_id, document_id)
    if not vectorstore:
        return {"status": "error", "message": "Could not load vectorstore"}

    # Retrieve relevant chunks
    retrieved_docs = vectorstore.similarity_search(query, k=5)
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    # LLM setup
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

    prompt = f"""Use the context below to answer the question.

Context:
{context}

Question:
{query}

Answer:"""

    def stream_response():
        full_content = ""
        try:
            for chunk in llm.stream(prompt):
                token = chunk.content if hasattr(chunk, "content") else str(chunk)
                full_content += token
                yield f"data: {json.dumps({'token': token})}\n\n"
            yield f"data: [DONE]\n\n"
        except Exception as e:
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

# ---------- Chat Streaming Endpoint ----------
@app.post("/chat/send")
def chat_send(payload: MessageInput):
    thread_id = payload.thread_id
    question = payload.message
    selected_model = payload.model or "llama"

    add_message(thread_id, role="human", content=question)
    raw_history = get_messages_by_thread(thread_id)
    history = reconstruct_history(raw_history)
    agent = ConversationalAgent(prompt_template, selected_model)


    def sse_stream():
        full_content = ""
        try:
            for token in agent.stream_response_for_web(history):
                full_content += token
                yield f"data: {json.dumps({'token': token})}\n\n"
            if full_content:
                add_message(thread_id, role="ai", content=full_content)
            yield "data: [DONE]\n\n"
        except Exception as exc:
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
