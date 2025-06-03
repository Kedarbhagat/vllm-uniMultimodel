import os
import json
import uuid
import logging
import asyncio
from typing import Dict, Any, Optional, List, Union

import redis.asyncio as redis # Use async Redis client
import httpx
from fastapi import FastAPI, HTTPException, Depends, status, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware # For CORS
from pydantic import BaseModel, Field, ValidationError

# --- Configuration (using Environment Variables) ---
# It's good practice to load configuration from environment variables for production
REDIS_HOST = os.getenv("REDIS_HOST", "172.31.21.186")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
API_PORT = int(os.getenv("API_PORT", 8082))
VLLM_BACKEND_URL = os.getenv("VLLM_BACKEND_URL", "http://172.31.21.186:8000/v1/chat/completions")

# Model to backend mapping - In a more complex setup, this might be dynamic or from a DB
# For this example, we assume one primary vLLM backend for all models it serves.
MODEL_TO_BACKEND: Dict[str, str] = {
    "./meta-llama/Llama-3.1-8B-Instruct-awq": VLLM_BACKEND_URL,
    # Add other models if your vLLM server serves multiple and you want
    # to enforce which ones are accepted by the gateway.
    # Note: vLLM's /v1/chat/completions endpoint handles the model selection internally
    # if you pass 'model' in the request body. This mapping mostly serves for validation.
}

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("api_gateway")

# --- FastAPI App Initialization ---
app = FastAPI(
    title="vLLM API Gateway",
    description="Gateway for LLM inference, supporting both streaming and async batch processing.",
    version="1.0.0"
)

# --- CORS Middleware (adjust origins as needed for your frontend) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Be more restrictive in production, e.g., ["http://your-frontend-domain.com"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models ---
class ChatMessage(BaseModel):
    role: str = Field(..., description="Role of the message sender (e.g., 'user', 'assistant', 'system').")
    content: str = Field(..., description="Content of the message.")

class ChatRequest(BaseModel):
    model: str = Field(..., description="The model to use for completion (e.g., './meta-llama/Llama-3.1-8B-Instruct-awq').")
    messages: List[ChatMessage] = Field(..., description="List of messages forming the conversation.")
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature.")
    max_tokens: Optional[int] = Field(default=1024, ge=1, description="Maximum number of tokens to generate.")
    stream: Optional[bool] = Field(default=False, description="Whether to stream the response (Server-Sent Events).")
    # Add any other vLLM specific parameters you want to expose, e.g., top_p, presence_penalty, etc.

class TaskResponse(BaseModel):
    task_id: str = Field(..., description="Unique identifier for the asynchronous task.")
    status: str = Field(..., description="Current status of the task (e.g., 'pending', 'completed', 'failed', 'cancelled').")
    # For a completed task, you might add Optional[Any] result field later,
    # but for initial pending response, this is fine.

# --- Global Redis Client (initialized in startup, closed in shutdown) ---
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
        # In a real app, you might want to exit or health check failure here.
        raise RuntimeError(f"Could not connect to Redis: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    global redis_client
    if redis_client:
        await redis_client.close()
        logger.info("Redis connection closed.")

# Dependency for Redis client
async def get_redis_client_dep():
    if redis_client is None:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Redis client not initialized.")
    return redis_client

# --- API Endpoints ---

@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    """
    Basic health check endpoint.
    Checks Redis connectivity.
    """
    try:
        if redis_client:
            await redis_client.ping()
        else:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Redis client not initialized.")
        return {"status": "ok", "redis_connected": True}
    except redis.RedisError as e:
        logger.error(f"Health check failed: Redis connection error: {e}")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"Redis connection error: {e}")

@app.post("/v1/chat/completions", response_model=Union[TaskResponse, Dict[str, Any]])
async def create_chat_completion(
    request: ChatRequest,
    req: Request, # Access original request object if needed
    r: redis.Redis = Depends(get_redis_client_dep)
):
    """
    Handles chat completions. If `stream` is True, proxies directly to vLLM.
    If `stream` is False, queues the task for asynchronous processing via Redis.
    """
    # Validate if the model is configured
    if request.model not in MODEL_TO_BACKEND:
        logger.warning(f"Attempted request for unknown model: {request.model}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown model: {request.model}. Available models: {list(MODEL_TO_BACKEND.keys())}"
        )

    vllm_backend_url = MODEL_TO_BACKEND[request.model]

    if request.stream:
        # --- Handle Streaming Request Directly to vLLM ---
        logger.info(f"Stream request received for task_id: N/A (direct streaming) - Model: {request.model}")
        try:
            # Prepare payload for vLLM, ensure stream=True for vLLM
            # model_dump() handles nested Pydantic models correctly
            vllm_payload = request.model_dump(mode='json')
            vllm_payload['stream'] = True

            # Use httpx.AsyncClient for asynchronous HTTP requests
            # It's more efficient to reuse an httpx client across requests if possible
            # For simplicity here, we create one per request. For high concurrency,
            # consider making client a global/app-level instance or using `httpx.AsyncClient(app=app)`.
            async with httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=10.0)) as client:
                async with client.stream("POST", vllm_backend_url, json=vllm_payload) as response:
                    # Raise an exception for HTTP errors (4xx or 5xx) from backend
                    response.raise_for_status()

                    async def generate_from_backend():
                        # Iterate over the async byte chunks from the httpx response
                        async for chunk in response.aiter_bytes():
                            if chunk:
                                # vLLM's /v1/chat/completions endpoint already sends data: and [DONE]
                                # You can directly yield the bytes.
                                yield chunk

                    return StreamingResponse(
                        generate_from_backend(),
                        media_type="text/event-stream",
                        headers={
                            "Cache-Control": "no-cache",
                            "Connection": "keep-alive",
                            "X-Accel-Buffering": "no" # Disable Nginx buffering if applicable
                        }
                    )
        except httpx.RequestError as e:
            logger.error(f"HTTPX Request Error to vLLM backend for streaming: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to connect to LLM backend: {e.request.url} - {e.__class__.__name__}"
            )
        except httpx.HTTPStatusError as e:
            logger.error(f"vLLM backend returned error for streaming: {e.response.status_code} - {e.response.text}")
            raise HTTPException(
                status_code=e.response.status_code, # Forward the backend's status code
                detail=f"LLM backend error: {e.response.text}"
            )
        except Exception as e:
            logger.exception(f"Unexpected error during streaming request processing.")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Internal server error processing streaming request: {str(e)}"
            )

    else:
        # --- Handle Non-Streaming Request (Queue for Worker) ---
        task_id = str(uuid.uuid4())
        logger.info(f"Non-stream request received. Queuing task_id: {task_id} - Model: {request.model}")

        try:
            # Prepare data for Redis queue. Use model_dump to convert Pydantic model to dict.
            task_data = request.model_dump(mode='json')
            task_data["task_id"] = task_id
            task_data["stream"] = False # Explicitly set for worker clarity

            # Push task to Redis queue
            await r.rpush("task_queue", json.dumps(task_data))
            logger.info(f"Task {task_id} pushed to Redis 'task_queue'.")

            return TaskResponse(task_id=task_id, status="pending")

        except redis.RedisError as e:
            logger.error(f"Redis error when queuing task {task_id}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to queue task due to Redis error: {str(e)}"
            )
        except Exception as e:
            logger.exception(f"Unexpected error when processing non-streaming request for task {task_id}.")
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
    Retrieves the result of a non-streaming asynchronous task by task_id.
    """
    try:
        result_key = f"result:{task_id}"
        result = await r.get(result_key)

        if result:
            logger.info(f"Result found for task {task_id}")
            return json.loads(result)
        else:
            logger.info(f"Result not found for task {task_id}, status is pending.")
            return {"task_id": task_id, "status": "pending"}

    except redis.RedisError as e:
        logger.error(f"Redis error when retrieving result for task {task_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve task result due to Redis error: {str(e)}"
        )
    except json.JSONDecodeError:
        logger.error(f"Failed to decode JSON result for task {task_id}. Data: {result}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Corrupted task result found in Redis."
        )
    except Exception as e:
        logger.exception(f"Unexpected error when retrieving result for task {task_id}.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error retrieving result: {str(e)}"
        )

# --- Main Execution ---
if __name__ == "__main__":
    logger.info(f"Starting FastAPI API Gateway on 0.0.0.0:{API_PORT}")
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=API_PORT)