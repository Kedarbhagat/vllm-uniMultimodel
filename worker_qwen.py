import os
import json
import logging
import asyncio
from typing import Dict, Any, Optional

import redis.asyncio as redis
import httpx

# --- Configuration for Qwen 2.5 Coder 14B Worker ---
REDIS_HOST = os.getenv("REDIS_HOST", "172.31.21.186")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))

MODEL_QUEUE_TO_LISTEN = "llm_task_queue:qwen2_5_coder_14b"  # CHANGED: Match gateway config
VLLM_BACKEND_URL = "http://172.31.21.186:8010/v1/chat/completions"
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "token-abc123")

STREAM_RESPONSE_PREFIX = "llm:response:stream:"

# --- Logging Setup ---
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("qwen_worker")

# --- Global Clients ---
redis_client: Optional[redis.Redis] = None
http_client: Optional[httpx.AsyncClient] = None


async def initialize_clients():
    """Initialize Redis and HTTP clients."""
    global redis_client, http_client

    logger.info(
        f"Initializing Qwen 2.5 Coder 14B worker for queue: {MODEL_QUEUE_TO_LISTEN}, backend: {VLLM_BACKEND_URL}"
    )

    redis_client = redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        db=0,
        decode_responses=True,
        socket_connect_timeout=10,
        socket_timeout=None,  # allow BLPOP blocking safely
        socket_keepalive=True,
    )
    await redis_client.ping()
    logger.info(f"Connected to Redis at {REDIS_HOST}:{REDIS_PORT}")

    http_client = httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=10.0))
    logger.info("HTTPX client initialized")


async def close_clients():
    """Gracefully close clients."""
    global redis_client, http_client
    if redis_client:
        await redis_client.aclose()
        logger.info("Qwen worker: Redis connection closed")
        redis_client = None
    if http_client:
        await http_client.aclose()
        logger.info("Qwen worker: HTTPX client closed")
        http_client = None


async def process_task(task_data: Dict[str, Any], r: redis.Redis, http_client: httpx.AsyncClient):
    """Process a single LLM task for Qwen model."""
    task_id = task_data.get("task_id")
    model_name = task_data.get("model")
    messages = task_data.get("messages")
    max_tokens = task_data.get("max_tokens")
    temperature = task_data.get("temperature")
    is_streaming = task_data.get("stream", False)

    logger.info(
        f"Qwen worker processing task {task_id}, model={model_name}, streaming={is_streaming}"
    )

    status_key = f"task:{task_id}:status"
    result_key = f"task:{task_id}:result"
    response_stream_name = f"{STREAM_RESPONSE_PREFIX}{task_id}"
    
    await r.set(status_key, "processing", ex=3600)

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
            logger.info(f"Qwen worker: Starting streaming request to vLLM for task {task_id}")

            async with http_client.stream("POST", VLLM_BACKEND_URL, json=payload, headers=headers) as response:
                response.raise_for_status()
                # Create stream immediately so gateway knows worker is processing
                await r.xadd(response_stream_name, {"status": "started", "done": "false"})
                await r.expire(response_stream_name, 300)

                async for line in response.aiter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    data_str = line[len("data: "):].strip()
                    if not data_str:
                        continue

                    if data_str == "[DONE]":
                        await r.xadd(response_stream_name, {"content": "", "done": "true"})
                        logger.debug(f"Qwen worker: Published [DONE] for {task_id}")
                        break

                    try:
                        chunk = json.loads(data_str)
                        choices = chunk.get("choices", [])
                        if choices:
                            delta = choices[0].get("delta", {})
                            content = delta.get("content")
                            if content:
                                await r.xadd(
                                    response_stream_name, {"content": content, "done": "false"}
                                )
                                logger.debug(
                                    f"Qwen worker: Published chunk for {task_id}: {content[:50]}..."
                                )
                        else:
                            logger.warning(
                                f"Qwen worker: Unexpected chunk format (no choices) for {task_id}: {data_str}"
                            )
                    except json.JSONDecodeError:
                        error_msg = f"Qwen worker: JSON decode error from vLLM for {task_id}: {data_str}"
                        logger.error(error_msg)
                        await r.xadd(response_stream_name, {"error": error_msg, "done": "true"})
                        await r.set(status_key, "failed", ex=3600)
                        return

            await r.set(status_key, "completed", ex=3600)
            logger.info(f"Qwen worker: Completed streaming task {task_id}")

        else:
            logger.info(f"Qwen worker: Making non-streaming request to vLLM for task {task_id}")
            response = await http_client.post(VLLM_BACKEND_URL, json=payload, headers=headers)
            response.raise_for_status()
            result = response.json()

            await r.set(result_key, json.dumps(result), ex=3600)
            await r.set(status_key, "completed", ex=3600)
            logger.info(f"Qwen worker: Completed non-streaming task {task_id}")

    except Exception as e:
        error_message = f"Qwen worker: Error in task {task_id}: {e}"
        logger.exception(error_message)
        if is_streaming:
            await r.xadd(response_stream_name, {"error": error_message, "done": "true"})
        await r.set(status_key, "failed", ex=3600)
        await r.set(result_key, json.dumps({"error": error_message}), ex=3600)


async def main_loop():
    """Main loop with Redis reconnect + retry."""
    global redis_client, http_client

    while True:
        try:
            if redis_client is None or http_client is None:
                await initialize_clients()

            logger.debug(f"Qwen worker: Waiting for tasks on queue {MODEL_QUEUE_TO_LISTEN}...")
            result = await redis_client.blpop(MODEL_QUEUE_TO_LISTEN, timeout=30)
            if result is None:
                # no task within timeout
                logger.debug("Qwen worker: No tasks received (timeout)")
                continue

            _queue_name, task_json = result
            if task_json:
                task_data = json.loads(task_json)
                logger.info(f"Qwen worker: Received task {task_data.get('task_id')}")
                asyncio.create_task(process_task(task_data, redis_client, http_client))

        except (ConnectionError, OSError) as e:
            logger.warning(f"Qwen worker: Redis connection error: {e}, reconnecting in 5s...")
            await close_clients()
            await asyncio.sleep(5)

        except Exception as e:
            logger.exception(f"Qwen worker: Unexpected error in main loop: {e}")
            await close_clients()
            await asyncio.sleep(5)


if __name__ == "__main__":
    logger.info("Starting Qwen 2.5 Coder 14B LLM Worker.")
    try:
        asyncio.run(main_loop())
    except KeyboardInterrupt:
        logger.info("Qwen worker shutdown requested by user (KeyboardInterrupt).")
    except Exception as e:
        logger.critical(f"Qwen worker: Fatal startup error: {e}")
