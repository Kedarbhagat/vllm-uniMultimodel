from fastapi import FastAPI, HTTPException, Request, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import redis
import json
import uuid
import logging
import requests
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("api_server")

# FastAPI app
app = FastAPI(title="LLM Task Processing API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

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

class ResultResponse(BaseModel):
    status: str
    task_id: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

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

@app.get("/health")
async def health_check():
    try:
        r = get_redis_client()
        queue_size = r.llen("task_queue")
        backends_status = {}
        for model, url in MODEL_TO_BACKEND.items():
            try:
                health_url = url.rsplit('/', 1)[0] + "/health"
                response = requests.get(health_url, timeout=2)
                backends_status[model] = "available" if response.status_code == 200 else "error"
            except Exception:
                backends_status[model] = "unavailable"
        return {
            "status": "healthy",
            "redis": "connected",
            "queue_size": queue_size,
            "backends": backends_status
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }

@app.get("/")
async def root():
    return {"status": "online", "message": "LLM Task Processing API"}

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

        r.rpush("task_queue", json.dumps(task_data))
        logger.info(f"Task created - ID: {task_id}, Model: {request.model}")
        background_tasks.add_task(check_worker_status, r)

        return {"task_id": task_id, "status": "pending"}

    except redis.RedisError as e:
        logger.error(f"Redis error in chat completion: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Redis error: {str(e)}")
    except Exception as e:
        logger.error(f"Server error in chat completion: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

@app.get("/v1/chat/result/{task_id}", response_model=ResultResponse)
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

@app.post("/v1/chat/cancel/{task_id}")
async def cancel_task(task_id: str, r: redis.Redis = Depends(get_redis_client)):
    try:
        r.set(f"cancel:{task_id}", "1", ex=3600)
        logger.info(f"Task {task_id} marked as cancelled.")
        return {"task_id": task_id, "status": "cancelled"}
    except redis.RedisError as e:
        logger.error(f"Redis error cancelling task {task_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Redis error: {str(e)}")

@app.get("/v1/queue/status")
async def queue_status(r: redis.Redis = Depends(get_redis_client)):
    try:
        queue_length = r.llen("task_queue")
        pending_tasks = []
        for i in range(min(5, queue_length)):
            task_json = r.lindex("task_queue", i)
            if task_json:
                task = json.loads(task_json)
                is_cancelled = r.get(f"cancel:{task.get('task_id')}") == "1"
                pending_tasks.append({
                    "task_id": task.get("task_id"),
                    "model": task.get("model"),
                    "cancelled": is_cancelled
                })

        return {
            "queue_length": queue_length,
            "pending_tasks": pending_tasks
        }
    except Exception as e:
        logger.error(f"Error fetching queue status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching queue status: {str(e)}")

@app.get("/v1/models")
async def list_models():
    models = []
    for model_name, url in MODEL_TO_BACKEND.items():
        try:
            health_url = url.rsplit('/', 1)[0] + "/health"
            response = requests.get(health_url, timeout=2)
            available = response.status_code == 200
        except Exception:
            available = False

        models.append({
            "id": model_name,
            "available": available
        })

    return {"models": models}

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
    logger.info("Starting FastAPI server on port 8081")
    uvicorn.run(app, host="0.0.0.0", port=8081)
