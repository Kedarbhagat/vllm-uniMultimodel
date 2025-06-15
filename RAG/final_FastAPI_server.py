from fastapi import FastAPI, HTTPException, UploadFile, File, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any, Generator, Union
import os
import uuid
from tempfile import NamedTemporaryFile
from wordloader import process_document
from embeddingmodel import MiniLMEmbeddings
from vectorestore import build_vectorstore
from langchain.schema import Document, HumanMessage, AIMessage, BaseMessage
import json
import uvicorn
import logging
from fastapi.responses import StreamingResponse

# Import the EnhancedRAGAgent from our implementation
from finalRAG_Web import EnhancedRAGAgent, llm_chat, llm_summary

# Configure logging
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["LANGCHAIN_PROJECT"] = "vllm-enhanced-rag-new-agent"

import os
os.environ["TAVILY_API_KEY"] = "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"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Enhanced RAG API with Mermaid Support",
    description="API for document processing and conversational RAG system with diagram generation",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
global_rag_agent: Optional[EnhancedRAGAgent] = None
document_store: Dict[str, Dict[str, Any]] = {}
session_chat_history: Dict[str, List[BaseMessage]] = {}

# --- Pydantic Models ---
class ChatRequest(BaseModel):
    question: str
    document_id: Optional[str] = None
    session_id: Optional[str] = None
    stream: bool = False
    enable_web_search: bool = False

class DocumentUploadResponse(BaseModel):
    document_id: str
    filename: str
    status: str
    metadata: Optional[Dict[str, Any]] = None

class ChatResponse(BaseModel):
    answer: str
    document_id: Optional[str] = None
    session_id: Optional[str] = None
    is_diagram: bool = False

class Message(BaseModel):
    type: str
    content: str
    is_diagram: bool = False

class ChatHistoryResponse(BaseModel):
    session_id: str
    history: List[Message]

class ToolResponse(BaseModel):
    type: str
    tool: Optional[str] = None
    content: Optional[Union[str, Dict]] = None
    complete: Optional[bool] = None

# --- FastAPI Event Handlers ---
@app.on_event("startup")
async def startup_event():
    global global_rag_agent
    logger.info("Initializing Enhanced RAG agent with Mermaid support...")
    
    embedding_model = MiniLMEmbeddings()
    empty_docs = [Document(page_content="")]
    vectorstore = build_vectorstore(empty_docs)
    
    global_rag_agent = EnhancedRAGAgent(
        llm_chat=llm_chat,
        llm_summary=llm_summary,
        vectorstore=vectorstore,
        embedding_model=embedding_model
    )
    logger.info("Enhanced RAG agent initialized with Mermaid support")

# --- API Endpoints ---
@app.post("/upload-document/", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)):
    global global_rag_agent
    if not global_rag_agent:
        raise HTTPException(status_code=500, detail="RAG agent not initialized")

    try:
        file_ext = os.path.splitext(file.filename)[1]
        with NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
            contents = await file.read()
            tmp_file.write(contents)
            tmp_file_path = tmp_file.name
        
        logger.info(f"Processing document: {file.filename}")
        documents = process_document(tmp_file_path)
        
        if not documents:
            raise ValueError("No content extracted from document")

        doc_id = str(uuid.uuid4())
        document_store[doc_id] = {
            "path": tmp_file_path,
            "filename": file.filename,
            "documents": documents
        }

        # Update vectorstore with new documents
        global_rag_agent.vectorstore = build_vectorstore(documents)
        logger.info(f"Document {doc_id} processed and vectorstore updated")

        return {
            "document_id": doc_id,
            "filename": file.filename,
            "status": "processed",
            "metadata": {"pages": len(documents)}
        }
    except Exception as e:
        logger.error(f"Document processing failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Document processing failed: {str(e)}")

@app.post("/chat/")
async def chat_endpoint(request: ChatRequest):
    global global_rag_agent
    if global_rag_agent is None:
        raise HTTPException(status_code=500, detail="RAG agent not initialized")

    # Session management
    session_id = request.session_id if request.session_id else str(uuid.uuid4())
    current_chat_history = session_chat_history.setdefault(session_id, [])
    
    try:
        document_path = None
        documents_for_rag = None

        if request.document_id:
            if request.document_id not in document_store:
                raise HTTPException(status_code=404, detail="Document not found")
            doc_info = document_store[request.document_id]
            document_path = doc_info["path"]
            documents_for_rag = doc_info["documents"]

        if request.stream:
            def generate_stream():
                full_response = ""
                is_diagram = False
                
                for chunk in global_rag_agent.stream_chat(
                    question=request.question,
                    chat_history=current_chat_history,
                    enable_web_search=request.enable_web_search,
                    document_path=document_path,
                    documents=documents_for_rag
                ):
                    if isinstance(chunk, dict):  # Tool response
                        yield f"data: {json.dumps(chunk)}\n\n"
                        if chunk.get("type") == "tool_output" and chunk.get("tool") == "mermaid":
                            is_diagram = True
                    else:  # Text response
                        full_response += chunk
                        yield f"data: {json.dumps({'type': 'text', 'content': chunk})}\n\n"
                
                # Update history after streaming completes
                if full_response:
                    current_chat_history.extend([
                        HumanMessage(content=request.question),
                        AIMessage(content=full_response)
                    ])
                    
                    if len(current_chat_history) > 20:
                        session_chat_history[session_id] = current_chat_history[-20:]
                
                yield "data: {}\n\n"  # Final empty message

            return StreamingResponse(
                generate_stream(),
                media_type="text/event-stream"
            )
        else:
            # Non-streaming response
            answer = global_rag_agent.chat(
                question=request.question,
                chat_history=current_chat_history,
                enable_web_search=request.enable_web_search,
                document_path=document_path,
                documents=documents_for_rag
            )
            
            # Check if answer contains a diagram
            is_diagram = "```mermaid" in answer
            
            current_chat_history.extend([
                HumanMessage(content=request.question),
                AIMessage(content=answer)
            ])
            
            if len(current_chat_history) > 20:
                session_chat_history[session_id] = current_chat_history[-20:]

            return ChatResponse(
                answer=answer,
                document_id=request.document_id,
                session_id=session_id,
                is_diagram=is_diagram
            )
            
    except Exception as e:
        logger.error(f"Chat error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {str(e)}")

@app.get("/documents/")
async def list_documents():
    """List all uploaded documents."""
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
    """Delete an uploaded document."""
    if document_id not in document_store:
        raise HTTPException(status_code=404, detail="Document not found")

    doc_info = document_store.pop(document_id)
    try:
        if os.path.exists(doc_info["path"]):
            os.remove(doc_info["path"])
    except Exception as e:
        logger.warning(f"Error deleting document file: {e}")

    return {"status": "deleted", "document_id": document_id}

@app.get("/chat-history/{session_id}", response_model=ChatHistoryResponse)
async def get_chat_history(session_id: str):
    """Retrieve chat history for a session."""
    history = session_chat_history.get(session_id, [])
    
    serializable_history = []
    for msg in history:
        is_diagram = "```mermaid" in msg.content if isinstance(msg, AIMessage) else False
        serializable_history.append(
            Message(
                type="human" if isinstance(msg, HumanMessage) else "ai",
                content=msg.content,
                is_diagram=is_diagram
            )
        )

    return ChatHistoryResponse(
        session_id=session_id,
        history=serializable_history
    )

@app.delete("/chat-history/{session_id}")
async def clear_chat_history(session_id: str):
    """Clear chat history for a session."""
    if session_id in session_chat_history:
        del session_chat_history[session_id]
        return {"status": "cleared", "session_id": session_id}
    raise HTTPException(status_code=404, detail="Session not found")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9000)