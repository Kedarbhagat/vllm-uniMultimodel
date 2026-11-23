"""
Standalone Title Generation Service
Separate from chat server to avoid blocking chat operations.
Can be run independently or as a background service.
"""

import asyncio
import logging
from typing import Optional, List
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
from transformers import pipeline
import json
from datetime import datetime

from dbhelper import update_thread_title, get_messages_by_thread, get_connection

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load model ONCE at startup
_title_summarizer = pipeline(
    task="text2text-generation",
    model="google/flan-t5-base",
    device=-1,  # CPU only (change to 0 if you have GPU)
)

logger.info("✅ Title summarizer model loaded")

# FastAPI app for this service
app = FastAPI(title="Title Generation Service")

# ========== Schemas ==========
class TitleGenerationRequest(BaseModel):
    thread_id: str
    messages: Optional[List[dict]] = None  # Pre-filtered messages (optional)
    async_mode: Optional[bool] = True  # Return immediately or wait for result

class TitleGenerationResponse(BaseModel):
    thread_id: str
    title: str
    status: str  # "success", "failed", "queued"
    timestamp: str

# ========== Service Functions ==========
async def generate_title_from_messages(messages: List[dict]) -> Optional[str]:
    """
    Generate a concise title from conversation messages.
    
    Args:
        messages: List of message dicts with 'role' and 'content' keys
    
    Returns:
        Title string or None if generation fails
    """
    if not messages or len(messages) == 0:
        logger.warning("No messages provided for title generation")
        return None

    # Extract first user message (most indicative of conversation topic)
    user_message = None
    for msg in messages:
        if msg.get("role") == "human":
            user_message = msg.get("content", "")
            break

    if not user_message:
        logger.warning("No user message found in conversation")
        return None

    # Truncate to reasonable length (avoid huge context)
    user_message_truncated = user_message[:500]

    prompt = (
        "Summarize the user question to a title of a chat sessioin 4 to 6 words. "
        "Do not use punctuation.\n\n"
        f"Question: {user_message_truncated}"
    )

    try:
        result = _title_summarizer(
            prompt,
            max_new_tokens=12,
            do_sample=False,
            temperature=0.0,
            repetition_penalty=1.1,
        )
        
        title = result[0]["generated_text"].strip()
        logger.info(f"Generated title: '{title}'")
        return title
    except Exception as e:
        logger.error(f"Title generation failed: {e}")
        return None

async def update_thread_title_in_db(thread_id: str, title: str):
    """
    Update thread title in database.
    Runs in background - doesn't block caller.
    """
    try:
        update_thread_title(thread_id, title)
        logger.info(f"✅ Updated thread {thread_id} with title: '{title}'")
    except Exception as e:
        logger.error(f"❌ Failed to update thread {thread_id}: {e}")

# ========== HTTP Endpoints ==========

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Title Generation Service",
        "model": "google/flan-t5-base"
    }

@app.post("/generate_title")
async def generate_title_endpoint(
    request: TitleGenerationRequest,
    background_tasks: BackgroundTasks
):
    """
    Generate and update thread title.
    
    Can be called in two modes:
    1. async_mode=True: Returns immediately, updates DB in background
    2. async_mode=False: Waits for title generation and returns result
    
    Example curl:
        curl -X POST "http://localhost:8001/generate_title" \
          -H "Content-Type: application/json" \
          -d '{"thread_id": "abc123", "async_mode": true}'
    """
    thread_id = request.thread_id
    
    logger.info(f"Title generation request for thread: {thread_id}")

    # Fetch messages from DB if not provided
    messages = request.messages
    if not messages:
        try:
            db_messages = get_messages_by_thread(thread_id)
            messages = db_messages
            logger.info(f"Fetched {messages} messages from DB")
            logger.info(f"Fetched {len(messages)} messages from DB")
        except Exception as e:
            logger.error(f"Failed to fetch messages for {thread_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to fetch messages: {str(e)}")

    if not messages:
        return TitleGenerationResponse(
            thread_id=thread_id,
            title="New Chat",
            status="failed",
            timestamp=datetime.now().isoformat()
        )

    # ASYNC MODE: Generate title without blocking
    if request.async_mode:
        logger.info(f"Running in ASYNC mode for thread {thread_id}")
        background_tasks.add_task(
            _generate_and_update_async,
            thread_id,
            messages
        )
        return TitleGenerationResponse(
            thread_id=thread_id,
            title="Generating...",
            status="queued",
            timestamp=datetime.now().isoformat()
        )

    # SYNC MODE: Wait for result
    else:
        logger.info(f"Running in SYNC mode for thread {thread_id}")
        title = await generate_title_from_messages(messages)
        
        if title:
            try:
                await update_thread_title_in_db(thread_id, title)
                status = "success"
            except Exception as e:
                logger.error(f"Failed to update DB: {e}")
                status = "failed"
        else:
            title = "New Chat"
            status = "failed"

        return TitleGenerationResponse(
            thread_id=thread_id,
            title=title,
            status=status,
            timestamp=datetime.now().isoformat()
        )

async def _generate_and_update_async(thread_id: str, messages: List[dict]):
    """Internal helper for async background processing"""
    try:
        title = await generate_title_from_messages(messages)
        if title:
            await update_thread_title_in_db(thread_id, title)
        logger.info(f"✅ Background title generation completed for {thread_id}")
    except Exception as e:
        logger.error(f"❌ Background task failed for {thread_id}: {e}")

@app.post("/batch_generate_titles")
async def batch_generate_titles(
    thread_ids: List[str],
    background_tasks: BackgroundTasks
):
    """
    Generate titles for multiple threads in background.
    Useful for batch processing or backfill operations.
    
    Example:
        curl -X POST "http://localhost:8001/batch_generate_titles" \
          -H "Content-Type: application/json" \
          -d '{"thread_ids": ["thread1", "thread2", "thread3"]}'
    """
    logger.info(f"Batch title generation for {len(thread_ids)} threads")
    
    for thread_id in thread_ids:
        background_tasks.add_task(_batch_process_thread, thread_id)
    
    return {
        "status": "queued",
        "thread_count": len(thread_ids),
        "message": "Titles are being generated in background"
    }

async def _batch_process_thread(thread_id: str):
    """Helper for batch processing"""
    try:
        messages = get_messages_by_thread(thread_id)
        title = await generate_title_from_messages(messages)
        if title:
            await update_thread_title_in_db(thread_id, title)
    except Exception as e:
        logger.error(f"Batch processing failed for {thread_id}: {e}")

@app.get("/stats")
def get_service_stats():
    """Get service statistics"""
    return {
        "service": "Title Generation Service",
        "model": "google/flan-t5-base",
        "device": "CPU (device=-1)",
        "version": "1.0",
        "status": "running"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
