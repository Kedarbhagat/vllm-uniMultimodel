import os
import json
import uuid
import logging
import asyncio
import time # Added for int(time.time())
from typing import Dict, Any, Optional, List, Union

import redis.asyncio as redis
from fastapi import FastAPI, HTTPException, Depends, status, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ValidationError

# --- Configuration ---
REDIS_HOST = os.getenv("REDIS_HOST", "172.31.21.186")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
API_PORT = int(os.getenv("API_PORT", 8082))
# VLLM_BACKEND_URL is now primarily used by the Worker, but kept for model mapping
VLLM_BACKEND_URL = os.getenv("VLLM_BACKEND_URL", "http://172.31.21.186:8000/v1/chat/completions")

# Model to backend mapping (Gateway still validates models, Worker uses the URL)
MODEL_TO_BACKEND: Dict[str, str] = {
    "./meta-llama/Llama-3.1-8B-Instruct-awq": VLLM_BACKEND_URL,
    # Add other models as needed
}

# --- Redis Stream Name Prefix for responses ---
STREAM_RESPONSE_PREFIX = "llm:response:stream:"

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO, # Set to logging.DEBUG for more verbose streaming logs
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("api_gateway")

# --- FastAPI App Initialization ---
app = FastAPI(
    title="vLLM API Gateway with Redis Queue for All Requests",
    description="Gateway for LLM inference, routing all requests through Redis queues.",
    version="1.0.0"
)

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Be more restrictive in production for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models ---
class ChatMessage(BaseModel):
    role: str = Field(..., description="Role of the message sender (e.g., 'user', 'assistant', 'system').")
    content: str = Field(..., description="Content of the message.")

class ChatRequest(BaseModel):
    model: str = Field(..., description="The model to use for completion.")
    messages: List[ChatMessage] = Field(..., description="List of messages forming the conversation.")
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature.")
    max_tokens: Optional[int] = Field(default=1024, ge=1, description="Maximum number of tokens to generate.")
    stream: Optional[bool] = Field(default=False, description="Whether to stream the response (Server-Sent Events).")
    # No stream_id here, it's generated internally by the gateway for streaming requests

class TaskResponse(BaseModel):
    task_id: str = Field(..., description="Unique identifier for the asynchronous task.")
    status: str = Field(..., description="Current status of the task.")
    # For streaming, this will only contain task_id and status 'pending'

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
        logger.info(f"Successfully connected to Redis at {REDIS_HOST}:{REDIS_PORT}")
    except redis.RedisError as e:
        logger.error(f"Failed to connect to Redis at {REDIS_HOST}:{REDIS_PORT}: {e}")
        raise RuntimeError(f"Could not connect to Redis: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during startup: {e}")
        raise RuntimeError(f"Startup failed: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    global redis_client
    if redis_client:
        await redis_client.close()
        logger.info("Redis connection closed.")

# Dependency for Redis client
async def get_redis_client_dep():
    if redis_client is None:
        logger.error("Redis client not initialized when requested by dependency.")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Redis client not initialized.")
    return redis_client

# --- API Endpoints ---

@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    """Basic health check endpoint. Checks Redis connectivity."""
    try:
        if redis_client:
            await redis_client.ping()
        else:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Redis client not initialized.")
        return {"status": "ok", "redis_connected": True}
    except redis.RedisError as e:
        logger.error(f"Health check failed: Redis connection error: {e}")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"Redis connection error: {e}")
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Unexpected error during health check: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Internal server error during health check: {str(e)}")

@app.post("/v1/chat/completions", response_model=Union[TaskResponse, Dict[str, Any]])
async def create_chat_completion(
    request: ChatRequest,
    r: redis.Redis = Depends(get_redis_client_dep)
):
    """
    Handles chat completions. All requests are queued to Redis.
    Streaming requests subscribe to a Redis Stream for real-time chunks.
    Non-streaming requests poll for results.
    """
    if request.model not in MODEL_TO_BACKEND:
        logger.warning(f"Attempted request for unknown model: {request.model}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown model: {request.model}. Available models: {list(MODEL_TO_BACKEND.keys())}"
        )

    task_id = str(uuid.uuid4())
    logger.info(f"Received request: stream={request.stream}, task_id: {task_id}, model: {request.model}")

    # Prepare task data to be pushed to Redis queue
    task_data = request.model_dump(mode='json')
    task_data["task_id"] = task_id
    # Ensure worker knows it's a streaming request if client asked for it
    task_data["stream"] = request.stream

    try:
        # Push all requests to a common task queue
        await r.rpush("llm_task_queue", json.dumps(task_data))
        logger.info(f"Task {task_id} pushed to Redis 'llm_task_queue'.")

        if request.stream:
            # For streaming, the Gateway immediately subscribes to a dedicated Redis Stream
            # where the worker will publish results.
            response_stream_name = f"{STREAM_RESPONSE_PREFIX}{task_id}"
            logger.info(f"Setting up stream reader for task_id: {task_id} on stream: {response_stream_name}")

            async def stream_from_redis_queue():
                last_id = '0-0' # Start reading from the beginning of the stream
                try:
                    while True:
                        # XREAD BLOCK 0 waits indefinitely for new messages
                        # count=1 reads one message at a time for fine-grained streaming
                        messages = await r.xread(
                            {response_stream_name: last_id},
                            block=20000, # Block for up to 20 seconds for new messages
                            count=1
                        )

                        if not messages:
                            # Timeout occurred, still waiting for first chunk or next chunk
                            logger.debug(f"Gateway: Timeout waiting for stream {response_stream_name} chunk.")
                            continue # Keep trying

                        # Messages format: [[stream_name, [[msg_id, {field: value}]]]]
                        stream_messages = messages[0][1]
                        for msg_id, fields in stream_messages:
                            content_chunk = fields.get('content')
                            is_done = fields.get('done') == 'true' # Convert from Redis string 'true'/'false'
                            is_error = fields.get('error') is not None

                            if is_error:
                                error_message = fields.get('error', 'Unknown error from worker.')
                                logger.error(f"Gateway: Received error from worker for {task_id}: {error_message}")
                                # Send error as an OpenAI-like chunk
                                yield f"data: {json.dumps({'error': error_message, 'finish_reason': 'error'})}\n\n".encode('utf-8')
                                yield b"data: [DONE]\n\n"
                                # Optionally clean up stream if error means no further data
                                await r.delete(response_stream_name)
                                return # Exit the generator
                            
                            if content_chunk:
                                # Re-create OpenAI-compatible chunk for the client (test script)
                                # This is the crucial part that ensures the client can parse the JSON.
                                openai_chunk_payload = {
                                    "id": f"chatcmpl-{task_id}",
                                    "object": "chat.completion.chunk",
                                    "created": int(time.time()),
                                    "model": request.model,
                                    "choices": [
                                        {
                                            "index": 0,
                                            "delta": {
                                                "content": content_chunk
                                            },
                                            "logprobs": None,
                                            "finish_reason": None
                                        }
                                    ]
                                }
                                yield f"data: {json.dumps(openai_chunk_payload)}\n\n".encode('utf-8')
                                logger.debug(f"Gateway: Yielded chunk for {task_id}. msg_id: {msg_id}")

                            if is_done:
                                yield b"data: [DONE]\n\n"
                                logger.info(f"Gateway: Received [DONE] for stream {task_id}. Exiting stream.")
                                # Clean up the Redis Stream after successful completion
                                await r.delete(response_stream_name)
                                return # Exit the generator

                            last_id = msg_id # Update last_id to continue from the next message

                except asyncio.CancelledError:
                    logger.info(f"Gateway: Streaming for {task_id} cancelled by client or timeout.")
                    # Optionally, notify worker to stop if possible (more advanced)
                except redis.RedisError as e:
                    logger.error(f"Gateway: Redis error during stream reading for {task_id}: {e}")
                    yield f"data: {json.dumps({'error': 'Redis stream error: ' + str(e), 'finish_reason': 'error'})}\n\n".encode('utf-8')
                    yield b"data: [DONE]\n\n"
                except Exception as e:
                    logger.exception(f"Gateway: Unexpected error in stream_from_redis_queue for {task_id}.")
                    yield f"data: {json.dumps({'error': 'Internal gateway streaming error', 'finish_reason': 'error'})}\n\n".encode('utf-8')
                    yield b"data: [DONE]\n\n"

            return StreamingResponse(
                stream_from_redis_queue(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no" # Essential for Nginx/other reverse proxies
                }
            )
        else:
            # For non-streaming, return task_id for polling
            return TaskResponse(task_id=task_id, status="pending")

    except redis.RedisError as e:
        logger.error(f"Redis error when queuing task {task_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to queue task due to Redis error: {str(e)}"
        )
    except Exception as e:
        logger.exception(f"Unexpected error when processing request for task {task_id}.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error when queuing task: {str(e)}"
        )

@app.get("/v1/chat/result/{task_id}", response_model=Dict[str, Any])
async def get_result(
    task_id: str,
    r: redis.Redis = Depends(get_redis_client_dep)
):
    """
    Retrieves the result of a non-streaming asynchronous task from Redis.
    """
    logger.info(f"Fetching result for task_id: {task_id}")
    try:
        result_key = f"task:{task_id}:result"
        status_key = f"task:{task_id}:status"

        status = await r.get(status_key)
        if not status:
            logger.warning(f"Status not found for task_id: {task_id}")
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Task not found or expired.")

        if status == "completed" or status == "failed":
            result_data_str = await r.get(result_key)
            if result_data_str:
                result = json.loads(result_data_str)
                logger.info(f"Result for task_id {task_id}: Status={status}")
                return {
                    "task_id": task_id,
                    "status": status,
                    "result": result if status == "completed" else None,
                    "error": result if status == "failed" else None
                }
            else:
                logger.error(f"Result data missing for completed/failed task_id: {task_id}")
                return {
                    "task_id": task_id,
                    "status": "failed",
                    "error": "Result data missing after completion signal."
                }
        else:
            logger.info(f"Task {task_id} status: {status}")
            return {"task_id": task_id, "status": status}

    except redis.RedisError as e:
        logger.error(f"Redis error when fetching result for task {task_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve result due to Redis error: {str(e)}"
        )
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error for task {task_id} result: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to decode result data: {str(e)}"
        )
    except Exception as e:
        logger.exception(f"Unexpected error when fetching result for task {task_id}.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error when fetching task result: {str(e)}"
        )

# --- Main Execution ---
if __name__ == "__main__":
    logger.info(f"Starting FastAPI API Gateway on 0.0.0.0:{API_PORT}")
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=API_PORT)