import os
import json
import logging
import asyncio
from typing import Dict, Any, Optional

import redis.asyncio as redis
import httpx

# --- Configuration for DeepSeek Coder V2 Lite Worker ---
REDIS_HOST = os.getenv("REDIS_HOST", "172.31.21.186")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))

# THIS WORKER'S SPECIFIC QUEUE AND BACKEND
MODEL_QUEUE_TO_LISTEN = "llm_task_queue:deepseek_v2_lite"
VLLM_BACKEND_URL = "http://172.31.21.186:8002/v1/chat/completions" # DeepSeek vLLM instance

VLLM_API_KEY = os.getenv("VLLM_API_KEY", "token-abc123") # Ensure this matches your vLLM setup

STREAM_RESPONSE_PREFIX = "llm:response:stream:"

# --- Logging Setup ---
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("deepseek_worker") # Unique logger name for this worker

# --- Global Clients ---
redis_client: Optional[redis.Redis] = None
http_client: Optional[httpx.AsyncClient] = None

async def initialize_clients():
    global redis_client, http_client

    logger.info(f"Initializing DeepSeek worker for queue: {MODEL_QUEUE_TO_LISTEN}, backend: {VLLM_BACKEND_URL}")

    try:
        redis_client = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            db=0,
            decode_responses=True,
            socket_connect_timeout=10,
            socket_timeout=3600, # for blocking BLPOP
        )
        await redis_client.ping()
        logger.info(f"Connected to Redis at {REDIS_HOST}:{REDIS_PORT}")

        http_client = httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=10.0))
        logger.info("HTTPX client initialized")

    except Exception as e:
        logger.error(f"Failed to initialize clients for DeepSeek worker: {e}")
        raise

async def close_clients():
    global redis_client, http_client
    if redis_client:
        await redis_client.close()
        logger.info("DeepSeek worker: Redis connection closed")
    if http_client:
        await http_client.aclose()
        logger.info("DeepSeek worker: HTTPX client closed")

async def process_task(task_data: Dict[str, Any], r: redis.Redis, http_client: httpx.AsyncClient):
    task_id = task_data.get("task_id")
    model_name = task_data.get("model") # This should be "./DeepSeek-Coder-V2-Lite-Instruct-awq"
    messages = task_data.get("messages")
    max_tokens = task_data.get("max_tokens")
    temperature = task_data.get("temperature")
    is_streaming = task_data.get("stream", False)

    logger.info(f"DeepSeek worker processing task {task_id}, model={model_name}, streaming={is_streaming}")

    status_key = f"task:{task_id}:status"
    result_key = f"task:{task_id}:result"

    await r.set(status_key, "processing", ex=3600)

    # Construct payload for vLLM API - ensure it matches vLLM's expected format
    payload = {
        "model": model_name,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": is_streaming,
    }

    headers = {"Authorization": f"Bearer {VLLM_API_KEY}"}

    try:
        if is_streaming:
            response_stream_name = f"{STREAM_RESPONSE_PREFIX}{task_id}"
            logger.info(f"DeepSeek worker: Starting streaming request to vLLM for task {task_id}")

            async with http_client.stream("POST", VLLM_BACKEND_URL, json=payload, headers=headers) as response:
                response.raise_for_status()
                await r.expire(response_stream_name, 300) # Set expiry on stream key

                async for line in response.aiter_lines():
                    if not line:
                        continue
                    if line.startswith("data: "):
                        data_str = line[len("data: "):].strip()
                        if not data_str:
                            continue

                        if data_str == "[DONE]":
                            await r.xadd(response_stream_name, {"content": "", "done": "true"})
                            logger.debug(f"DeepSeek worker: Published [DONE] for {task_id}")
                            break
                        try:
                            chunk = json.loads(data_str)
                            choices = chunk.get("choices", [])
                            if choices:
                                delta = choices[0].get("delta", {})
                                content = delta.get("content")
                                if content:
                                    await r.xadd(response_stream_name, {"content": content, "done": "false"})
                                    logger.debug(f"DeepSeek worker: Published chunk for {task_id}: {content[:50]}...")
                            else:
                                logger.warning(f"DeepSeek worker: Unexpected chunk format (no choices) for {task_id}: {data_str}")
                        except json.JSONDecodeError:
                            error_msg = f"DeepSeek worker: JSON decode error from vLLM for {task_id}: {data_str}"
                            logger.error(error_msg)
                            await r.xadd(response_stream_name, {"error": error_msg, "done": "true"})
                            await r.set(status_key, "failed", ex=3600)
                            return
                    else:
                        logger.debug(f"DeepSeek worker: Non-data line from vLLM for {task_id}: {line}")

            await r.set(status_key, "completed", ex=3600)
            logger.info(f"DeepSeek worker: Completed streaming task {task_id}")

        else: # Non-streaming request
            logger.info(f"DeepSeek worker: Making non-streaming request to vLLM for task {task_id}")
            response = await http_client.post(VLLM_BACKEND_URL, json=payload, headers=headers)
            response.raise_for_status()
            result = response.json()

            await r.set(result_key, json.dumps(result), ex=3600)
            await r.set(status_key, "completed", ex=3600)
            logger.info(f"DeepSeek worker: Completed non-streaming task {task_id}")

    except httpx.RequestError as e:
        error_message = f"DeepSeek worker: HTTPX request error for task {task_id}: {e}"
        logger.error(error_message)
        if is_streaming:
            await r.xadd(response_stream_name, {"error": error_message, "done": "true"})
        await r.set(status_key, "failed", ex=3600)
        await r.set(result_key, json.dumps({"error": error_message}), ex=3600)
    except httpx.HTTPStatusError as e:
        error_message = f"DeepSeek worker: HTTP error {e.response.status_code} for task {task_id}: {e.response.text}"
        logger.error(error_message)
        if is_streaming:
            await r.xadd(response_stream_name, {"error": error_message, "done": "true"})
        await r.set(status_key, "failed", ex=3600)
        await r.set(result_key, json.dumps({"error": error_message}), ex=3600)
    except Exception as e:
        error_message = f"DeepSeek worker: An unexpected error occurred for task {task_id}: {e}"
        logger.exception(error_message)
        if is_streaming:
            await r.xadd(response_stream_name, {"error": error_message, "done": "true"})
        await r.set(status_key, "failed", ex=3600)
        await r.set(result_key, json.dumps({"error": error_message}), ex=3600)

async def main():
    await initialize_clients()
    logger.info(f"DeepSeek worker started, listening on queue: {MODEL_QUEUE_TO_LISTEN}")
    try:
        while True:
            _queue_name, task_json = await redis_client.blpop(MODEL_QUEUE_TO_LISTEN, timeout=0)
            if task_json:
                task_data = json.loads(task_json)
                asyncio.create_task(process_task(task_data, redis_client, http_client))
    except asyncio.CancelledError:
        logger.info("DeepSeek worker processing cancelled.")
    except Exception as e:
        logger.exception(f"Fatal error in DeepSeek worker main loop: {e}")
    finally:
        await close_clients()

if __name__ == "__main__":
    logger.info("Starting DeepSeek LLM Worker.")
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("DeepSeek worker shutdown requested by user (KeyboardInterrupt).")
    except Exception as e:
        logger.critical(f"DeepSeek worker: Fatal startup error: {e}")