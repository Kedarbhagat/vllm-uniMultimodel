# In your FastAPI app.py

from fastapi import FastAPI, File, Request, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from typing import AsyncGenerator, Optional, List, Dict, Any
import json
import asyncio
import os
import shutil
import logging

# Import your LangGraph agent components
from agentic_rag import build_graph, AgentState # Assuming AgentState is defined in agentic_rag.py
from wordloader import process_document
from vectorestore import build_vectorstore
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage # Import if not already done, for chat history type hinting

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Document Assistant API",
    description="API for RAG, summarization, QA generation, and document analysis using LangGraph.",
    version="1.0.0"
)

class GlobalDocState:
    def __init__(self):
        self.documents: Optional[List[Document]] = None
        self.vectorstore: Optional[object] = None
        self.doc_path: Optional[str] = None
        # Use LangChain's Message types for clarity in chat history
        self.chat_history: List[Dict[str, str]] = [] # Storing as dicts, matching AgentState

global_doc_state = GlobalDocState()

# Compile the LangGraph graph once at application startup for efficiency
global_graph = build_graph()

# --- Your existing /upload-document/ endpoint goes here ---
@app.post("/upload-document/", summary="Upload a document for processing")
async def upload_document(file: UploadFile = File(...)):
    upload_dir = "uploaded_documents"
    os.makedirs(upload_dir, exist_ok=True)

    file_location = os.path.join(upload_dir, file.filename)
    try:
        with open(file_location, "wb+") as file_object:
            shutil.copyfileobj(file.file, file_object)
        logger.info(f"File saved to {file_location}")

        logger.info("Processing document and building vector store...")
        docs = process_document(file_location, max_sentences=7)
        if not docs:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="No documents could be processed from the uploaded file. Ensure it's a valid PDF/DOCX."
            )

        vs = build_vectorstore(docs)
        if not vs:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to build vector store from the document."
            )

        # Update global state
        global_doc_state.documents = docs
        global_doc_state.vectorstore = vs
        global_doc_state.doc_path = file_location
        global_doc_state.chat_history = [] # Reset history for new document

        logger.info(f"Document processed and indexed successfully. Chunks: {len(docs)}")

        return JSONResponse(content={
            "message": "Document uploaded and processed successfully!",
            "filename": file.filename,
            "doc_count": len(docs),
            "total_chars": sum(len(doc.page_content) for doc in docs)
        }, status_code=status.HTTP_200_OK)
    except Exception as e:
        logger.error(f"Error during document upload and processing: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred during file upload or processing: {str(e)}"
        )

# --- NEW STREAMING CHAT ENDPOINT ---
class StreamingChatRequest(BaseModel):
    question: str
    # If you want to send history from client, it should match your AgentState's history format
    # chat_history: Optional[List[Dict[str, str]]] = None

@app.post("/stream-chat/", summary="Stream chat responses from the document assistant")
async def stream_chat_response(request: StreamingChatRequest):
    """
    Streams the response from the LangGraph agent token-by-token based on the last uploaded document.
    """
    if global_doc_state.documents is None or global_doc_state.vectorstore is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No document has been uploaded yet. Please upload a document first using /upload-document/."
        )

    logger.info(f"Received streaming chat question: '{request.question}'")

    async def generate_response_chunks() -> AsyncGenerator[str, None]:
        full_answer_text_accumulator = "" # To rebuild the final answer for chat history

        # Prepare initial state for LangGraph
        initial_state = AgentState(
            question=request.question,
            doc_path=global_doc_state.doc_path,
            documents=global_doc_state.documents,
            vectorstore=global_doc_state.vectorstore,
            answer="",
            route="rag", # This initial route will be re-evaluated by the graph's safe_route
            error=None,
            metadata={},
            chat_history=global_doc_state.chat_history # Pass the current chat history
        )

        try:
            # Use astream_events to get granular events, including LLM token streams
            async for s in global_graph.astream_events(initial_state, version="v1"):
                event_type = s["event"]
                # logger.debug(f"Received event: {event_type} - {s.get('node')}") # Debugging

                # Look for LLM token streams from any node
                if event_type == "on_chat_model_stream" and "chunk" in s["data"]:
                    chunk = s["data"]["chunk"]
                    # If it's a content chunk from the LLM
                    if hasattr(chunk, 'content') and chunk.content:
                        yield chunk.content # Yield the raw text token
                        full_answer_text_accumulator += chunk.content
                # You can also stream other events if the client needs more granular info
                # elif event_type == "on_node_start":
                #     yield f"\n[Node Started: {s['node']}]\n"
                # elif event_type == "on_node_end" and "output" in s["data"]:
                #     # This captures the *final* answer from a node, but the LLM tokens are above
                #     if s['node'] in ['rag', 'summary', 'qa_gen', 'analysis']:
                #         # This is redundant if you're streaming tokens, but good for diagnostics
                #         # yield f"\n[Node {s['node']} finished]\n"
                #         pass
                elif event_type == "on_error":
                    error_message = str(s["data"].get("exception", "An unknown error occurred during processing."))
                    logger.error(f"Error event from LangGraph: {error_message}")
                    yield f"\n\n[ERROR]: {error_message}\n"
                    # It's good to break here or yield a final error message and then stop.
                    break # Stop streaming on critical error

            # After the stream finishes, update the global chat history
            # This is critical for multi-turn conversations
            if full_answer_text_accumulator:
                global_doc_state.chat_history.append({"role": "human", "content": request.question})
                global_doc_state.chat_history.append({"role": "ai", "content": full_answer_text_accumulator})
            logger.info("Streaming response complete. Chat history updated.")

        except Exception as e:
            logger.error(f"Error during streaming response generation: {e}", exc_info=True)
            yield f"\n\n[API STREAMING ERROR]: {str(e)}\n"

    # Return FastAPI's StreamingResponse
    return StreamingResponse(generate_response_chunks(), media_type="text/plain") # text/plain is simplest for raw tokens

# --- Your existing /status/ endpoint goes here ---
@app.get("/status/", summary="Check API status and loaded document info")
async def get_status():
    if global_doc_state.documents and global_doc_state.vectorstore:
        return JSONResponse(content={
            "status": "active",
            "document_loaded": True,
            "filename": os.path.basename(global_doc_state.doc_path),
            "doc_chunks": len(global_doc_state.documents),
            "total_characters_in_doc": sum(len(doc.page_content) for doc in global_doc_state.documents)
        })
    else:
        return JSONResponse(content={
            "status": "active",
            "document_loaded": False,
            "message": "No document currently loaded. Please upload one."
        })


if __name__ == "__main__":
    import uvicorn

    os.makedirs("uploaded_documents", exist_ok=True)
    logger.info("FastAPI application starting...")
    uvicorn.run(app, host="0.0.0.0", port=9000)