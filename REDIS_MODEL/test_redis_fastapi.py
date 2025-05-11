from fastapi import FastAPI, HTTPException, Request, Depends, BackgroundTasks
from fastapi.responses import StreamingResponse
import redis
import json
import uuid
import logging
import requests
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
import asyncio

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("api_server")

# FastAPI app
app = FastAPI(title="LLM Task Processing API")

# Redis configuration
REDIS_HOST = "172.31.21.186"
REDIS_PORT = 6379

# Model to backend mapping
MODEL_TO_BACKEND = {
    "./meta-llama/Llama-3.1-8B-Instruct-awq": "http://172.31.21.186:8000/v1/chat/completions"
}

# Pydantic models
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    temperature: Optional[float] = Field(default=0.7)
    max_tokens: Optional[int] = Field(default=1024)
    stream: Optional[bool] = Field(default=False)

class TaskResponse(BaseModel):
    task_id: str
    status: str

# Redis client helper
def get_redis_client():
    try:
        client = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            decode_responses=True,
            socket_timeout=5,
            socket_connect_timeout=5
        )
        client.ping()
        return client
    except redis.RedisError as e:
        logger.error(f"Redis connection error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Redis connection error: {str(e)}")

@app.post("/v1/chat/completions", response_model=TaskResponse)
async def create_chat_completion(
    request: ChatRequest,
    background_tasks: BackgroundTasks,
    r: redis.Redis = Depends(get_redis_client)
):
    try:
        if request.model not in MODEL_TO_BACKEND:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown model: {request.model}. Available models: {list(MODEL_TO_BACKEND.keys())}"
            )

        task_id = str(uuid.uuid4())

        task_data = {
            "task_id": task_id,
            "model": request.model,
            "messages": [msg.dict() for msg in request.messages],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "stream": request.stream
        }

        # Push task to Redis
        r.rpush("task_queue", json.dumps(task_data))

        # Add task to background for processing
        background_tasks.add_task(check_worker_status, r)

        return {"task_id": task_id, "status": "pending"}

    except redis.RedisError as e:
        logger.error(f"Redis error in chat completion: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Redis error: {str(e)}")
    except Exception as e:
        logger.error(f"Server error in chat completion: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

@app.get("/v1/chat/result/{task_id}")
async def get_result(task_id: str, r: redis.Redis = Depends(get_redis_client)):
    try:
        result = r.get(f"result:{task_id}")
        if result:
            logger.info(f"Result found for task {task_id}")
            return json.loads(result)

        return {"status": "pending", "task_id": task_id}

    except redis.RedisError as e:
        logger.error(f"Redis error in get result: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Redis error: {str(e)}")

@app.post("/v1/chat/completions/stream")
async def stream_chat_completion(
    request: ChatRequest,
    r: redis.Redis = Depends(get_redis_client)
):
    try:
        if request.model not in MODEL_TO_BACKEND:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown model: {request.model}. Available models: {list(MODEL_TO_BACKEND.keys())}"
            )

        backend_url = MODEL_TO_BACKEND[request.model]

        # Forward the stream request to vLLM's backend
        response = requests.post(
            backend_url,
            json=request.dict(),
            stream=True
        )

        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Error from model backend: {response.text}"
            )

        # Fixed streaming generator function
        async def generate():
            for chunk in response.iter_lines():
                if chunk:
                    # Properly format each response chunk with "data: " prefix
                    # This is crucial for SSE (Server-Sent Events) format
                    yield f"data: {chunk.decode('utf-8')}\n\n"
            
            # Send a final DONE message
            yield "data: [DONE]\n\n"

        # Return StreamingResponse with proper event-stream media type
        return StreamingResponse(
            generate(), 
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"  # Disable Nginx buffering if applicable
            }
        )

    except requests.exceptions.RequestException as e:
        logger.error(f"Error communicating with model backend: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error communicating with model backend: {str(e)}")

def check_worker_status(r: redis.Redis):
    try:
        initial_size = r.llen("task_queue")
        import time
        time.sleep(2)
        new_size = r.llen("task_queue")
        if new_size >= initial_size and initial_size > 0:
            logger.warning("Worker may not be processing tasks. Queue size not decreasing.")
            r.publish("worker_health_check", json.dumps({"timestamp": time.time()}))
    except Exception as e:
        logger.error(f"Error checking worker status: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting FastAPI server on port 8082")
    uvicorn.run(app, host="0.0.0.0", port=8082)