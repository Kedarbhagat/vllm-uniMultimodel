from docx import Document
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Request
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Generator
import os
import shutil
import logging
import asyncio

# Import your LangGraphAgenticRAG class and its dependencies
from new_agen import LangGraphAgenticRAG, llm_chat, llm_summary
from vectorestore import build_vectorstore
from embeddingmodel import MiniLMEmbeddings
from wordloader import process_document # Ensure this is compatible with file paths

# --- Configuration and Initialization ---
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

UPLOAD_DIR = "uploaded_documents"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Initialize RAG components (these should be singleton instances for efficiency)
try:
    logger.info("Initializing embedding model...")
    embedding_model = MiniLMEmbeddings() # Assuming this is lightweight
    logger.info("Building initial vector store (can be empty initially or pre-populated)...")
    # You might want to load an existing vectorstore here or build an empty one.
    # For this example, let's assume it's built from scratch or can be updated.
    # We'll create a dummy one for now, as the prompt didn't specify persistence.
    vectorstore = build_vectorstore([]) # Start with an empty vector store

    logger.info("Initializing LangGraphAgenticRAG agent...")
    rag_agent = LangGraphAgenticRAG(llm_chat, llm_summary, vectorstore, embedding_model)
    logger.info("LangGraphAgenticRAG agent initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize RAG components: {e}", exc_info=True)
    # In a real production app, you might want to exit or enter a degraded state.
    rag_agent = None # Indicate that agent is not available

app = FastAPI(
    title="RAG Agent API",
    description="A FastAPI server for a LangGraph-powered RAG agent with chat, document upload, and health endpoints.",
    version="1.0.0",
)

# --- Pydantic Models for Request/Response ---
class ChatRequest(BaseModel):
    prompt: str = Field(..., description="The user's prompt or question.")
    document_path: Optional[str] = Field(None, description="Optional path to a document for summarization/analysis.")
    # Add a flag to indicate if the document needs to be processed
    process_document_flag: bool = Field(False, description="Set to true if document_path should trigger full document processing.")


class DocumentUploadResponse(BaseModel):
    message: str
    file_path: str
    file_size_bytes: int
    chunks_processed: Optional[int] = Field(None, description="Number of chunks processed if document was indexed.")

# --- Dependencies (for injecting rag_agent into endpoints) ---
def get_rag_agent():
    if rag_agent is None:
        raise HTTPException(status_code=500, detail="RAG agent not initialized. Check server logs.")
    return rag_agent

# --- Endpoints ---

@app.get("/health", summary="Health Check", response_model=Dict[str, str])
async def health_check():
    """
    Returns the health status of the application.
    """
    status = "healthy" if rag_agent else "degraded (RAG agent not initialized)"
    return {"status": status, "message": "RAG Agent API is operational."}

@app.post("/upload-document", summary="Upload and Process Document", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(..., description="The document file to upload (e.g., PDF, DOCX, TXT)."),
    agent: LangGraphAgenticRAG = Depends(get_rag_agent) # Inject the RAG agent
):
    """
    Uploads a document, saves it, and optionally processes it into the vector store.
    Note: For simplicity, this re-indexes the document. In production,
    you'd have a more robust indexing pipeline.
    """
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    file_size = 0
    chunks_processed = None

    try:
        # Save the uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        file_size = os.path.getsize(file_path)
        logger.info(f"Uploaded file '{file.filename}' saved to '{file_path}'. Size: {file_size} bytes.")

        # Process the document using your custom loader and update vectorstore
        # IMPORTANT: This is a simplified in-memory vector store update.
        # For production, you'd want persistent storage and proper indexing management.
        logger.info(f"Processing document '{file.filename}' for indexing...")
        documents: List[Document] = process_document(file_path) # Your custom loader
        if documents:
            # Clear existing documents for this simple example, or manage document IDs
            # If you want to add to the existing vectorstore, use add_documents
            # For a proper production setup, you'd manage this with document IDs
            # and potentially a dedicated indexing service.
            # agent.vectorstore.add_documents(documents) # if you want to add
            agent.vectorstore = build_vectorstore(documents) # Rebuild for simplicity
            chunks_processed = len(documents)
            logger.info(f"Document '{file.filename}' processed and indexed. Chunks: {chunks_processed}.")
            message = f"Document '{file.filename}' uploaded and indexed successfully."
        else:
            message = f"Document '{file.filename}' uploaded, but no content extracted or indexed."
            logger.warning(f"No content extracted from '{file.filename}'.")

        return DocumentUploadResponse(
            message=message,
            file_path=file_path,
            file_size_bytes=file_size,
            chunks_processed=chunks_processed
        )
    except Exception as e:
        logger.error(f"Error processing document '{file.filename}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to process document: {e}")
    finally:
        file.file.close() # Ensure the uploaded file stream is closed

@app.post("/chat", summary="Chat with the RAG Agent (Streaming Response)")
async def chat_endpoint(
    request: ChatRequest,
    agent: LangGraphAgenticRAG = Depends(get_rag_agent)
):
    """
    Sends a prompt to the RAG agent and receives a streaming response.
    Optionally, provide a document_path to summarize/analyze an uploaded document.
    """
    logger.info(f"Received chat request for prompt: '{request.prompt}' with document_path: '{request.document_path}' and process_flag: {request.process_document_flag}")

    # Initial validation for document processing flag
    if request.process_document_flag and not request.document_path:
        raise HTTPException(
            status_code=400,
            detail="'process_document_flag' is true but 'document_path' is not provided."
        )
    if request.document_path and not os.path.exists(request.document_path):
         raise HTTPException(
            status_code=404,
            detail=f"Document not found at path: {request.document_path}"
        )

    async def generate_stream():
        try:
            # Determine if we should pass the document path for full processing
            doc_path_for_agent = request.document_path if request.process_document_flag else None

            # Stream the response from your RAG agent
            # Ensure your agent's stream_chat method handles the document_path correctly
            # and that it correctly yields tokens.
            for chunk in agent.stream_chat(question=request.prompt, document_path=doc_path_for_agent):
                # Ensure the chunk is a string and encode it for SSE
                yield f"data: {chunk.strip()}\n\n"
            yield "data: [DONE]\n\n" # Signal end of stream
        except Exception as e:
            logger.error(f"Error during streaming chat response: {e}", exc_info=True)
            yield f"data: {{'error': 'An internal server error occurred: {e}'}}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(generate_stream(), media_type="text/event-stream")

# --- Example of a Non-Streaming Chat Endpoint (Optional, for comparison) ---
# @app.post("/chat-non-streaming", summary="Chat with the RAG Agent (Non-Streaming)")
# async def chat_non_streaming_endpoint(
#     request: ChatRequest,
#     agent: LangGraphAgenticRAG = Depends(get_rag_agent)
# ):
#     """
#     Sends a prompt to the RAG agent and receives a complete response (non-streaming).
#     """
#     logger.info(f"Received non-streaming chat request for prompt: '{request.prompt}'")
#     try:
#         doc_path_for_agent = request.document_path if request.process_document_flag else None
#         response_content = agent.chat(question=request.prompt, document_path=doc_path_for_agent)
#         return JSONResponse(content={"answer": response_content})
#     except Exception as e:
#         logger.error(f"Error during non-streaming chat response: {e}", exc_info=True)
#         raise HTTPException(status_code=500, detail=f"Failed to get response: {e}")

if __name__ == "__main__":
    import uvicorn
    logger.info("FastAPI application starting...")
    uvicorn.run(app, host="0.0.0.0", port=9000)