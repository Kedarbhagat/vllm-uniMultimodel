import os
import uuid
import logging
from typing import Optional, List, Dict, Any
import json

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from pydantic import BaseModel
from tempfile import NamedTemporaryFile

from wordloader import process_document
from embeddingmodel import MiniLMEmbeddings
from vectorestore import build_vectorstore

from langchain.schema import Document, HumanMessage, AIMessage, BaseMessage
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# --- Configure logging and environment ---
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["LANGCHAIN_PROJECT"] = "vllm-simple-rag"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Simple RAG API",
    description="API for document upload and simple conversational RAG (no agent, no workflow)",
    version="0.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Global State ---
embedding_model = MiniLMEmbeddings()
vectorstore = None
document_store: Dict[str, Dict[str, Any]] = {}
session_chat_history: Dict[str, List[BaseMessage]] = {}

# --- LLM Initialization ---
API_GATEWAY_URL = os.getenv("API_GATEWAY_URL", "http://172.17.35.82:8082")
YOUR_MODEL_BASE_URL = f"{API_GATEWAY_URL}/v1"
YOUR_MODEL_NAME = os.getenv("VLLM_MODEL_NAME", "./DeepSeek-Coder-V2-Lite-Instruct-awq")
YOUR_API_KEY = os.getenv("YOUR_API_KEY", "token-abc123")

llm = ChatOpenAI(
    model=YOUR_MODEL_NAME,
    openai_api_base=YOUR_MODEL_BASE_URL,
    openai_api_key=YOUR_API_KEY,
    temperature=0.7,
    max_tokens=2048,
    streaming=True,
)

# --- Pydantic Models ---
class ChatRequest(BaseModel):
    question: str
    document_id: Optional[str] = None
    session_id: Optional[str] = None
    stream: bool = False

class DocumentUploadResponse(BaseModel):
    document_id: str
    filename: str
    status: str
    metadata: Optional[Dict[str, Any]] = None

class ChatResponse(BaseModel):
    answer: str
    document_id: Optional[str] = None
    session_id: Optional[str] = None

class Message(BaseModel):
    type: str
    content: str

class ChatHistoryResponse(BaseModel):
    session_id: str
    history: List[Message]

# --- FastAPI Event Handlers ---
@app.on_event("startup")
async def startup_event():
    global vectorstore
    logger.info("Initializing empty vectorstore...")
    empty_docs = [Document(page_content="")]
    vectorstore = build_vectorstore(empty_docs)
    logger.info("Vectorstore initialized.")

# --- API Endpoints ---

@app.post("/upload-document/", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)):
    global vectorstore
    try:
        file_ext = os.path.splitext(file.filename)[1]
        with NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
            contents = await file.read()
            tmp_file.write(contents)
            tmp_file_path = tmp_file.name

        logger.info(f"Processing document: {file.filename} saved to {tmp_file_path}")
        documents = process_document(tmp_file_path)
        if not documents:
            raise ValueError("No content extracted from document.")

        doc_id = str(uuid.uuid4())
        document_store[doc_id] = {
            "path": tmp_file_path,
            "filename": file.filename,
            "documents": documents
        }

        # Update the global vectorstore with all documents
        all_docs = []
        for info in document_store.values():
            all_docs.extend(info["documents"])
        vectorstore = build_vectorstore(all_docs)
        logger.info(f"Document {doc_id} processed and vectorstore updated.")

        return {
            "document_id": doc_id,
            "filename": file.filename,
            "status": "processed",
            "metadata": {"pages": len(documents)}
        }
    except Exception as e:
        logger.error(f"Document processing failed for {file.filename}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Document processing failed: {str(e)}")

@app.post("/chat/")
async def chat_endpoint(request: ChatRequest):
    global vectorstore
    if vectorstore is None:
        raise HTTPException(status_code=500, detail="Vectorstore not initialized.")

    session_id = request.session_id if request.session_id else str(uuid.uuid4())
    current_chat_history = session_chat_history.setdefault(session_id, [])

    try:
        # Retrieve context from vectorstore
        context = ""
        if vectorstore:
            docs = vectorstore.similarity_search(request.question, k=4)
            context = "\n\n".join([doc.page_content for doc in docs])

        # Prepare chat history string
        chat_hist_str = "\n".join(
            [
                f"{'Human' if isinstance(msg, HumanMessage) else 'Assistant'}: {msg.content}"
                for msg in current_chat_history[-8:]
            ]
        )

        prompt = ChatPromptTemplate.from_template(
            """
You are a helpful assistant. Answer the question based on the provided context and chat history.

Context:
{context}

Chat History:
{chat_history}

Question: {question}

If you don't know the answer, say so honestly.
            """.strip()
        ).format(
            context=context,
            chat_history=chat_hist_str,
            question=request.question
        )

        messages = [HumanMessage(content=prompt)]

        def token_stream():
            answer = ""
            for chunk in llm.stream(messages):
                token = ""
                if hasattr(chunk, 'content') and chunk.content:
                    token = chunk.content
                elif isinstance(chunk, dict):
                    token = chunk.get('content', '')
                if token:
                    answer += token
                    yield f"data: {json.dumps({'token': token})}\n\n"
            # Update chat history after streaming is done
            current_chat_history.append(HumanMessage(content=request.question))
            current_chat_history.append(AIMessage(content=answer))
            if len(current_chat_history) > 20:
                session_chat_history[session_id] = current_chat_history[-20:]
            yield "data: [DONE]\n\n"

        if request.stream:
            return StreamingResponse(token_stream(), media_type="text/event-stream")
        else:
            # Non-streaming fallback
            answer = ""
            for chunk in llm.stream(messages):
                if hasattr(chunk, 'content') and chunk.content:
                    answer += chunk.content
                elif isinstance(chunk, dict):
                    content = chunk.get('content', '')
                    if content:
                        answer += content
            current_chat_history.append(HumanMessage(content=request.question))
            current_chat_history.append(AIMessage(content=answer))
            if len(current_chat_history) > 20:
                session_chat_history[session_id] = current_chat_history[-20:]
            return ChatResponse(
                answer=answer,
                document_id=request.document_id,
                session_id=session_id
            )
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {str(e)}")

@app.get("/documents/")
async def list_documents():
    return [
        {
            "document_id": doc_id,
            "filename": info["filename"],
            "pages": len(info.get("documents", []))
        }
        for doc_id, info in document_store.items()
    ]

@app.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    if document_id not in document_store:
        raise HTTPException(status_code=404, detail="Document not found.")
    doc_info = document_store.pop(document_id)
    try:
        if os.path.exists(doc_info["path"]) and os.path.isfile(doc_info["path"]):
            os.remove(doc_info["path"])
            logger.info(f"Deleted temporary file for document {document_id}: {doc_info['path']}")
    except Exception as e:
        logger.warning(f"Error deleting temporary file for document {document_id}: {e}")
    logger.info(f"Document {document_id} deleted from store.")
    return {"status": "deleted", "document_id": document_id}

@app.get("/chat-history/{session_id}", response_model=ChatHistoryResponse)
async def get_chat_history(session_id: str):
    history = session_chat_history.get(session_id, [])
    serializable_history = [
        Message(type=msg.type, content=msg.content) for msg in history
    ]
    logger.info(f"Retrieved history for session {session_id}. Messages: {len(history)}")
    return ChatHistoryResponse(session_id=session_id, history=serializable_history)

@app.delete("/chat-history/{session_id}")
async def clear_chat_history(session_id: str):
    if session_id in session_chat_history:
        del session_chat_history[session_id]
        logger.info(f"Chat history cleared for session {session_id}.")
        return {"status": "cleared", "session_id": session_id}
    logger.warning(f"Attempted to clear non-existent session history: {session_id}")
    raise HTTPException(status_code=404, detail="Session ID not found.")

# --- Run the Application ---
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting FastAPI application...")
    uvicorn.run(app, host="0.0.0.0", port=9000)