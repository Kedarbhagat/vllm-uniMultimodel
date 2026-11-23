# main.py

from fastapi import FastAPI, HTTPException, UploadFile, File, Response

from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel

from typing import Optional, List, Dict, Any, Generator

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



# Assuming your LangGraphAgenticRAG class is in 'new_agen.py'

from new_agen import LangGraphAgenticRAG, llm_chat, llm_summary



# Configure logging

os.environ["LANGCHAIN_TRACING_V2"] = "true" # Enable LangSmith tracing for LangChain/LangGraph
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "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" # Replace with your actual key
os.environ["LANGCHAIN_PROJECT"] = "vllm-enhanced-rag_new_Agen"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)



app = FastAPI(

    title="LangGraph Agentic RAG API",

    description="API for document processing and conversational RAG system with session management",

    version="0.2.0"

)



# CORS configuration

app.add_middleware(

    CORSMiddleware,

    allow_origins=["*"],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"],

)



# Global instances for the RAG agent and document storage

# The RAG agent itself is now stateless regarding chat history

global_rag_agent_instance: Optional[LangGraphAgenticRAG] = None

# This will store processed documents mapped by their ID

document_store: Dict[str, Dict[str, Any]] = {}



# In-memory store for chat history per session/user

# For production, replace this with a proper database (Redis, PostgreSQL, etc.)

session_chat_history: Dict[str, List[BaseMessage]] = {}



# --- Pydantic Models ---

class ChatRequest(BaseModel):

    question: str

    document_id: Optional[str] = None

    session_id: Optional[str] = None # Optional session ID for continuation

    stream: bool = False



class DocumentUploadResponse(BaseModel):

    document_id: str

    filename: str

    status: str

    metadata: Optional[Dict[str, Any]] = None



class ChatResponse(BaseModel):

    answer: str

    document_id: Optional[str] = None

    context_used: Optional[str] = None

    session_id: Optional[str] = None # Return session_id to client



class Message(BaseModel):

    type: str

    content: str



class ChatHistoryResponse(BaseModel):

    session_id: str

    history: List[Message]



# --- FastAPI Event Handlers ---

@app.on_event("startup")

async def startup_event():

    """

    Initialize a SINGLE RAG agent instance.

    This agent instance does NOT manage chat history internally.

    It will receive chat history as a parameter for each request.

    """

    global global_rag_agent_instance

    logger.info("Initializing RAG agent instance...")

    embedding_model = MiniLMEmbeddings()

    # Provide an initial empty vectorstore or a shared one if relevant

    # The vectorstore will be updated dynamically upon document upload.

    empty_docs = [Document(page_content="")]

    vectorstore = build_vectorstore(empty_docs)

    global_rag_agent_instance = LangGraphAgenticRAG(llm_chat, llm_summary, vectorstore, embedding_model)

    logger.info("RAG agent initialized.")



# --- API Endpoints ---



@app.post("/upload-document/", response_model=DocumentUploadResponse)

async def upload_document(file: UploadFile = File(...)):

    """Upload and process a document for later use in the RAG system."""

    global global_rag_agent_instance

    if not global_rag_agent_instance:

        raise HTTPException(status_code=500, detail="RAG agent not initialized.")



    try:

        # Create a temporary file to save the uploaded content

        file_ext = os.path.splitext(file.filename)[1]

        with NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:

            contents = await file.read()

            tmp_file.write(contents)

            tmp_file_path = tmp_file.name

        

        logger.info(f"Processing document: {file.filename} saved to {tmp_file_path}")

        documents = process_document(tmp_file_path) # Your custom loader

        

        if not documents:

            raise ValueError("No content extracted from document.")



        doc_id = str(uuid.uuid4())



        # Store document information

        document_store[doc_id] = {

            "path": tmp_file_path,

            "filename": file.filename,

            "documents": documents # Store pre-chunked LangChain Documents

        }



        # Update the global RAG agent's vectorstore with the new document's chunks.

        # This design choice means the vectorstore always reflects the LAST uploaded document.

        # For multiple documents or per-user document contexts, a more advanced vectorstore

        # management (e.g., separate vectorstores per document, or a combined one) would be needed.

        global_rag_agent_instance.vectorstore = build_vectorstore(documents)

        logger.info(f"Document {doc_id} processed and vectorstore updated.")



        return {

            "document_id": doc_id,

            "filename": file.filename,

            "status": "processed",

            "metadata": {"pages": len(documents)} if documents else None

        }

    except Exception as e:

        logger.error(f"Document processing failed for {file.filename}: {e}", exc_info=True)

        raise HTTPException(status_code=500, detail=f"Document processing failed: {str(e)}")

    finally:

        # Optionally, remove the temporary file if you're not storing it long-term

        # For now, we'll keep it for the document_store's 'path' reference.

        pass



@app.post("/chat/", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    if global_rag_agent_instance is None:
        raise HTTPException(status_code=500, detail="RAG agent not initialized.")

    # Get or create session ID
    session_id = request.session_id if request.session_id else str(uuid.uuid4())
    current_chat_history = session_chat_history.setdefault(session_id, [])
    
    try:
        document_path = None
        documents_for_rag = None

        if request.document_id:
            if request.document_id not in document_store:
                raise HTTPException(status_code=404, detail="Document not found.")
            doc_info = document_store[request.document_id]
            document_path = doc_info["path"]
            documents_for_rag = doc_info["documents"]

        if request.stream:
            def generate_stream():
                full_response = ""
                
                # Call the updated stream_chat with chat_history
                for token in global_rag_agent_instance.stream_chat(
                    question=request.question,
                    document_path=document_path,
                    documents=documents_for_rag,
                    chat_history=current_chat_history  # Pass history here
                ):
                    full_response += token
                    yield f"data: {json.dumps({'token': token})}\n\n"
                
                # Update history after streaming completes
                current_chat_history.extend([
                    HumanMessage(content=request.question),
                    AIMessage(content=full_response)
                ])
                
                if len(current_chat_history) > 20:
                    session_chat_history[session_id] = current_chat_history[-20:]
                
                yield "data: [DONE]\n\n"

            return StreamingResponse(generate_stream(), media_type="text/event-stream")
        else:
            # Call the updated chat with chat_history
            answer = global_rag_agent_instance.chat(
                question=request.question,
                document_path=document_path,
                documents=documents_for_rag,
                chat_history=current_chat_history  # Pass history here
            )
            
            current_chat_history.extend([
                HumanMessage(content=request.question),
                AIMessage(content=answer)
            ])
            
            if len(current_chat_history) > 20:
                session_chat_history[session_id] = current_chat_history[-20:]

            context_type = "full_document" if global_rag_agent_instance.last_used_full_document else "retrieved_context_or_direct"
            
            return ChatResponse(
                answer=answer,
                document_id=request.document_id,
                context_used=context_type,
                session_id=session_id
            )
            
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}", exc_info=True)
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

    """Delete an uploaded document and its temporary file."""

    if document_id not in document_store:

        raise HTTPException(status_code=404, detail="Document not found.")



    doc_info = document_store.pop(document_id)

    try:

        # Attempt to delete the temporary file associated with the document

        if os.path.exists(doc_info["path"]) and os.path.isfile(doc_info["path"]):

            os.remove(doc_info["path"])

            logger.info(f"Deleted temporary file for document {document_id}: {doc_info['path']}")

    except Exception as e:

        logger.warning(f"Error deleting temporary file for document {document_id}: {e}")



    logger.info(f"Document {document_id} deleted from store.")

    return {"status": "deleted", "document_id": document_id}



@app.get("/chat-history/{session_id}", response_model=ChatHistoryResponse)

async def get_chat_history(session_id: str):

    """Retrieve chat history for a given session ID."""

    history = session_chat_history.get(session_id, [])

    # Convert BaseMessage objects to a serializable format for response

    serializable_history = [

        Message(type=msg.type, content=msg.content) for msg in history

    ]

    logger.info(f"Retrieved history for session {session_id}. Messages: {len(history)}")

    return ChatHistoryResponse(session_id=session_id, history=serializable_history)



@app.delete("/chat-history/{session_id}")

async def clear_chat_history(session_id: str):

    """Clear chat history for a given session ID."""

    if session_id in session_chat_history:

        del session_chat_history[session_id]

        logger.info(f"Chat history cleared for session {session_id}.")

        return {"status": "cleared", "session_id": session_id}

    logger.warning(f"Attempted to clear non-existent session history: {session_id}")

    raise HTTPException(status_code=404, detail="Session ID not found.")



# --- Run the Application ---

if __name__ == "__main__":

    logger.info("Starting FastAPI application...")

    uvicorn.run(app, host="0.0.0.0", port=9000)
