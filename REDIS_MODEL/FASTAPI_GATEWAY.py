import os
import json
import uuid
import logging
import asyncio
import time
from typing import Dict, Any, Optional, List, Union

import redis.asyncio as redis
from fastapi import FastAPI, HTTPException, Depends, status, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# --- Configuration ---
REDIS_HOST = os.getenv("REDIS_HOST", "172.31.21.186")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
API_PORT = int(os.getenv("API_PORT", 8082))

# Updated: Define model configurations with their respective backend URLs and queue names
MODEL_CONFIG: Dict[str, Dict[str, str]] = {
    "./meta-llama/Llama-3.1-8B-Instruct-awq": {
        "backend_url": "http://172.31.21.186:8000/v1/chat/completions",
        "queue_name": "llm_task_queue:llama3_1_8b"
    },
    "./DeepSeek-Coder-V2-Lite-Instruct-awq": {
        "backend_url": "http://172.31.21.186:8002/v1/chat/completions",
        "queue_name": "llm_task_queue:deepseek_v2_lite"
    },
    # Add other models with their specific backend and queue names if needed

    "/mnt/c/Users/STUDENT/qwen2.5-coder-14b-instruct-awq-final":{
        "backend_url": "http://172.31.21.186:8010/v1/chat/completions",
        "queue_name": "llm_task_queue:qwen2_5_coder_14b"
    }
}

STREAM_RESPONSE_PREFIX = "llm:response:stream:"

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("api_gateway")

# --- FastAPI Initialization ---
app = FastAPI(
    title="vLLM API Gateway with Redis Queue",
    description="Gateway for LLM inference, routing through Redis queues.",
    version="1.0.0"
)

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models ---
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False

class TaskResponse(BaseModel):
    task_id: str
    status: str

# --- Global Redis Client ---
redis_client: Optional[redis.Redis] = None

@app.on_event("startup")
async def startup_event():
    global redis_client
    try:
        redis_client = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            db=0,
            decode_responses=True,
            socket_connect_timeout=10,
            socket_timeout=10
        )
        await redis_client.ping()
        logger.info(f"Connected to Redis at {REDIS_HOST}:{REDIS_PORT}")
    except Exception as e:
        logger.error(f"Redis connection failed: {e}")
        raise RuntimeError(f"Could not connect to Redis: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    global redis_client
    if redis_client:
        await redis_client.aclose()
        logger.info("Redis connection closed.")

async def get_redis_client_dep():
    if redis_client is None:
        raise HTTPException(status_code=500, detail="Redis client not initialized.")
    return redis_client

# --- Health Check ---
@app.get("/health", status_code=200)
async def health_check():
    if redis_client:
        try:
            await redis_client.ping()
            return {"status": "ok", "redis_connected": True}
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Redis error: {e}")
    raise HTTPException(status_code=503, detail="Redis client not initialized.")

# --- Main Endpoint ---
@app.post("/v1/chat/completions", response_model=Union[TaskResponse, Dict[str, Any]])
async def create_chat_completion(
    request: ChatRequest,
    r: redis.Redis = Depends(get_redis_client_dep)
):
    # Retrieve model configuration including backend URL and queue name
    model_config = MODEL_CONFIG.get(request.model)
    print("\n -------------------------------\n")
    print("model config:",model_config)
    if not model_config:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model: {request.model}. Available models: {list(MODEL_CONFIG.keys())}"
        )

    task_id = str(uuid.uuid4())
    backend_url = model_config["backend_url"]
    queue_name = model_config["queue_name"] # Get the specific queue name for this model

    task_data = request.dict()
    task_data.update({
        "task_id": task_id,
        "vllm_backend_url": backend_url,
        "original_model": request.model # Keep the original model name in task data
    })

    # Push the task to the model-specific queue
    await r.rpush(queue_name, json.dumps(task_data))
    logger.info(f"Task {task_id} queued to {queue_name}.")

    if request.stream:
        stream_name = f"{STREAM_RESPONSE_PREFIX}{task_id}"

        async def event_stream():
            last_id = "0-0"
            try:
                # Set a reasonable timeout for the stream to exist initially
                # The worker needs to create the stream. If it doesn't appear, something is wrong.
                initial_wait_time = 45 # seconds
                start_time = time.time()
                stream_ready = False
                while time.time() - start_time < initial_wait_time:
                    if await r.exists(stream_name):
                        stream_ready = True
                        break
                    await asyncio.sleep(0.1) # Check frequently at first

                if not stream_ready:
                    logger.error(f"Stream {stream_name} not created by worker within {initial_wait_time}s timeout.")
                    yield f"data: {json.dumps({'error': 'Stream initiation timeout', 'finish_reason': 'error'})}\n\n".encode()
                    yield b"data: [DONE]\n\n"
                    return

                while True:
                    # Block for a long time, as the worker will push to this stream
                    messages = await r.xread({stream_name: last_id}, block=20000, count=1)
                    if not messages:
                        # If block timed out, and no messages, continue waiting
                        continue
                    
                    # messages will be a list of tuples: [(stream_name, [(msg_id, fields), ...])]
                    stream_msgs = messages[0][1]
                    for msg_id, fields in stream_msgs:
                        content = fields.get("content")
                        is_done = fields.get("done") == "true"
                        is_error = fields.get("error")

                        if is_error:
                            logger.error(f"Worker reported error for task {task_id}: {is_error}")
                            yield f"data: {json.dumps({'error': is_error, 'finish_reason': 'error'})}\n\n".encode()
                            yield b"data: [DONE]\n\n"
                            # Clean up the stream key after error
                            await r.delete(stream_name)
                            return

                        if content:
                            chunk = {
                                "id": f"chatcmpl-{task_id}",
                                "object": "chat.completion.chunk",
                                "created": int(time.time()),
                                "model": request.model,
                                "choices": [{
                                    "index": 0,
                                    "delta": {"content": content},
                                    "logprobs": None,
                                    "finish_reason": None
                                }]
                            }
                            yield f"data: {json.dumps(chunk)}\n\n".encode()

                        if is_done:
                            logger.info(f"Stream for task {task_id} completed.")
                            yield b"data: [DONE]\n\n"
                            # Clean up the stream key once done
                            await r.delete(stream_name)
                            return

                        last_id = msg_id # Update last_id for the next read

            except asyncio.CancelledError:
                logger.warning(f"Client disconnected for stream task {task_id}.")
            except Exception as e:
                logger.exception(f"Stream error for {task_id}: {e}")
                yield f"data: {json.dumps({'error': 'Stream error', 'finish_reason': 'error'})}\n\n".encode()
                yield b"data: [DONE]\n\n"
            finally:
                # Ensure stream key is cleaned up even on unexpected exits
                if await r.exists(stream_name):
                    await r.delete(stream_name)

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no" # Important for SSE with proxies like Nginx
            }
        )

    else:
        # Non-streaming logic remains the same, as task_id keys are generic
        result_key = f"task:{task_id}:result"
        status_key = f"task:{task_id}:status"
        timeout = 300 # seconds
        start = time.time()

        await r.set(status_key, "pending", ex=timeout + 60) # Set initial status with extended expiry

        while True:
            status = await r.get(status_key)
            if status == "completed":
                result_str = await r.get(result_key)
                if result_str:
                    await r.delete(result_key)
                    await r.delete(status_key)
                    return JSONResponse(content=json.loads(result_str))
                # This case indicates a logic error where status is completed but result is missing
                logger.error(f"Task {task_id} completed but result missing.")
                raise HTTPException(status_code=500, detail=f"Task completed but result missing for {task_id}")
            elif status == "failed":
                error_str = await r.get(result_key) # Assuming result_key stores error message on failure
                logger.error(f"Task {task_id} failed: {error_str}")
                await r.delete(result_key)
                await r.delete(status_key)
                raise HTTPException(status_code=500, detail=f"Task failed: {error_str or 'Unknown error'}")
            elif time.time() - start > timeout:
                # If timeout, remove status key to prevent zombie tasks
                logger.warning(f"Timeout for task {task_id}.")
                await r.delete(status_key)
                await r.delete(result_key) # Also remove result key in case worker partially wrote it
                raise HTTPException(status_code=504, detail=f"Timeout for task {task_id}")
            
            # Add a small delay to prevent busy-waiting
            await asyncio.sleep(0.5)

# --- Result Retrieval (for non-streaming tasks that might be polled separately) ---
@app.get("/v1/chat/result/{task_id}", response_model=Dict[str, Any])
async def get_result(task_id: str, r: redis.Redis = Depends(get_redis_client_dep)):
    status_key = f"task:{task_id}:status"
    result_key = f"task:{task_id}:result"
    status = await r.get(status_key)

    if not status:
        raise HTTPException(status_code=404, detail="Task not found or expired. Ensure it was a non-streaming request.")

    result_str = await r.get(result_key)
    
    response_data = {
        "task_id": task_id,
        "status": status,
        "result": None,
        "error": None
    }

    if status == "completed":
        if result_str:
            response_data["result"] = json.loads(result_str)
            # Optionally delete keys here if client polling is the final retrieval
            # await r.delete(result_key)
            # await r.delete(status_key)
        else:
            response_data["error"] = "Result missing despite completed status."
            logger.error(f"Result missing for completed task {task_id} during explicit retrieval.")
    elif status == "failed":
        response_data["error"] = result_str if result_str else "Unknown error during task execution."
    else: # e.g., 'pending'
        response_data["status"] = "pending" # Explicitly set for clarity if it's not completed/failed
        response_data["message"] = "Task is still being processed."

    return response_data

# --- Optional: Run directly ---
if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting FastAPI API Gateway on 0.0.0.0:{API_PORT}")
    uvicorn.run(app, host="0.0.0.0", port=API_PORT)