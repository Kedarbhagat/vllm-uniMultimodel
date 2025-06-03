import os
import json
import logging
import asyncio
import time
from typing import Dict, Any, Optional

import redis.asyncio as redis
import httpx # For making requests to vLLM

# --- Configuration ---
REDIS_HOST = os.getenv("REDIS_HOST", "172.31.21.186")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
VLLM_BACKEND_URL = os.getenv("VLLM_BACKEND_URL", "http://172.31.21.186:8000/v1/chat/completions")
WORKER_POLL_INTERVAL = float(os.getenv("WORKER_POLL_INTERVAL", 0.5)) # seconds

# NEW: VLLM API Key - ensure this matches what you set in your vLLM server startup
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "token-abc123") 

# --- Redis Stream Name Prefix for responses ---
STREAM_RESPONSE_PREFIX = "llm:response:stream:"

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO, # Set to logging.DEBUG for more verbose processing logs
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("llm_worker")

# --- Global Clients ---
redis_client: Optional[redis.Redis] = None
http_client: Optional[httpx.AsyncClient] = None # Global httpx client for reuse

async def initialize_clients():
    """Initializes Redis and HTTPX clients."""
    global redis_client, http_client
    try:
        redis_client = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            db=0,
            decode_responses=True,
            socket_connect_timeout=10,
            # INCREASE THIS TIMEOUT SIGNIFICANTLY FOR WORKER'S BLPOP
            socket_timeout=3600 # For example, 1 hour (3600 seconds)
        )
        await redis_client.ping()
        logger.info(f"Worker connected to Redis at {REDIS_HOST}:{REDIS_PORT}")

        http_client = httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=10.0))
        logger.info("Worker HTTPX client initialized for vLLM requests.")

    except redis.RedisError as e:
        logger.error(f"Worker failed to connect to Redis: {e}")
        raise
    except Exception as e:
        logger.error(f"Worker failed to initialize HTTPX client: {e}")
        raise

async def close_clients():
    """Closes Redis and HTTPX clients."""
    global redis_client, http_client
    if redis_client:
        await redis_client.close()
        logger.info("Worker Redis connection closed.")
    if http_client:
        await http_client.aclose()
        logger.info("Worker HTTPX client closed.")

async def process_task(task_data: Dict[str, Any], r: redis.Redis, http_client: httpx.AsyncClient):
    """
    Processes a single LLM task, handling both streaming and non-streaming requests.
    """
    task_id = task_data.get("task_id")
    model_name = task_data.get("model")
    messages = task_data.get("messages")
    max_tokens = task_data.get("max_tokens")
    temperature = task_data.get("temperature")
    is_streaming = task_data.get("stream", False)

    logger.info(f"Worker: Processing task_id: {task_id}, streaming: {is_streaming}")

    result_key = f"task:{task_id}:result"
    status_key = f"task:{task_id}:status"

    await r.set(status_key, "processing", ex=3600) # Task status expires in 1 hour

    try:
        # Prepare payload for vLLM
        vllm_payload = {
            "model": model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": is_streaming # Tell vLLM whether to stream or not
        }

        # NEW: Headers for vLLM API Key
        vllm_headers = {
            "Authorization": f"Bearer {VLLM_API_KEY}"
        }

        if is_streaming:
            # --- Handle Streaming Request from Worker to vLLM ---
            response_stream_name = f"{STREAM_RESPONSE_PREFIX}{task_id}"
            logger.info(f"Worker: Starting streaming request to vLLM for task_id: {task_id}")
            
            try:
                # Use httpx.stream to consume vLLM's stream
                # Pass headers to the httpx.stream call for API Key authentication
                async with http_client.stream(
                    "POST", 
                    VLLM_BACKEND_URL, 
                    json=vllm_payload, 
                    headers=vllm_headers # <-- API KEY HEADER ADDED HERE
                ) as vllm_response:
                    vllm_response.raise_for_status() # Check for HTTP errors from vLLM

                    async for line in vllm_response.aiter_lines():
                        if not line: # Skip empty lines
                            continue

                        # All lines from aiter_lines() are already strings
                        if line.startswith("data: "):
                            json_str = line[len("data: "):].strip() # Use .strip() to remove leading/trailing whitespace
                            
                            if not json_str: # Skip if data part is empty after stripping
                                continue

                            # Process [DONE] or actual content from vLLM
                            if json_str == "[DONE]":
                                # Publish a [DONE] message to the Redis Stream
                                await r.xadd(response_stream_name, {'content': '', 'done': 'true'})
                                logger.debug(f"Worker: Published [DONE] for {task_id} to Redis Stream.")
                                break # Exit the vLLM stream loop

                            try:
                                chunk = json.loads(json_str)
                                if chunk.get("choices") and len(chunk["choices"]) > 0:
                                    delta_content = chunk["choices"][0].get("delta", {}).get("content")
                                    if delta_content:
                                        # Publish each content chunk to the Redis Stream
                                        await r.xadd(response_stream_name, {'content': delta_content, 'done': 'false'})
                                        logger.debug(f"Worker: Published chunk for {task_id} to Redis Stream: {delta_content[:50]}...")
                                else:
                                    logger.warning(f"Worker: Received non-standard chunk from vLLM for {task_id}: {json_str}")

                            except json.JSONDecodeError:
                                logger.error(f"Worker: Could not decode JSON from vLLM chunk for {task_id}: {json_str}")
                                # Publish error to Redis Stream and mark task as failed
                                await r.xadd(response_stream_name, {'error': f"JSON decode error in worker: {json_str}", 'done': 'true'})
                                await r.set(status_key, "failed", ex=3600) # Mark main task as failed
                                return # Stop processing this stream for this task
                        else:
                            # Log unexpected lines from vLLM if not "data: " prefixed
                            logger.debug(f"Worker: Received non-SSE data line from vLLM for {task_id}: {line}")
                
                # If stream completes without error, mark main task as completed
                await r.set(status_key, "completed", ex=3600)
                logger.info(f"Worker: Successfully completed streaming response for task_id: {task_id}.")

            except httpx.RequestError as e:
                logger.error(f"Worker: HTTPX Request Error to vLLM for streaming task {task_id}: {e}")
                # Publish error to Redis Stream and mark task as failed
                await r.xadd(response_stream_name, {'error': f"LLM backend connection error: {e}", 'done': 'true'})
                await r.set(status_key, "failed", ex=3600)
            except httpx.HTTPStatusError as e:
                logger.error(f"Worker: vLLM backend returned HTTP error for streaming task {task_id}: {e.response.status_code} - {e.response.text}")
                await r.xadd(response_stream_name, {'error': f"LLM backend returned error: {e.response.text}", 'done': 'true'})
                await r.set(status_key, "failed", ex=3600)
            except Exception as e:
                logger.exception(f"Worker: Unexpected error during streaming processing for task_id {task_id}.")
                await r.xadd(response_stream_name, {'error': f"Internal worker error during streaming: {str(e)}", 'done': 'true'})
                await r.set(status_key, "failed", ex=3600)

        else:
            # --- Handle Non-Streaming Request ---
            logger.info(f"Worker: Making non-streaming request to vLLM for task_id: {task_id}")
            # Pass headers to the httpx.post call for API Key authentication
            response = await http_client.post(
                VLLM_BACKEND_URL, 
                json=vllm_payload, 
                headers=vllm_headers # <-- API KEY HEADER ADDED HERE
            )
            response.raise_for_status()
            llm_result = response.json()

            # Store the final result in Redis
            await r.set(result_key, json.dumps(llm_result), ex=3600)
            await r.set(status_key, "completed", ex=3600)
            logger.info(f"Worker: Completed non-streaming task_id: {task_id}. Result stored.")

    except httpx.RequestError as e:
        logger.error(f"Worker: HTTPX Request Error to vLLM for task {task_id}: {e}")
        await r.set(status_key, "failed", ex=3600)
        await r.set(result_key, json.dumps({"error": f"Failed to connect to LLM backend: {e}"}), ex=3600)
    except httpx.HTTPStatusError as e:
        logger.error(f"Worker: vLLM backend returned HTTP error for task {task_id}: {e.response.status_code} - {e.response.text}")
        await r.set(status_key, "failed", ex=3600)
        await r.set(result_key, json.dumps({"error": f"LLM backend error: {e.response.text}"}), ex=3600)
    except Exception as e:
        logger.exception(f"Worker: Unexpected error processing task_id: {task_id}.")
        await r.set(status_key, "failed", ex=3600)
        await r.set(result_key, json.dumps({"error": f"Internal worker error: {str(e)}"}), ex=3600)

async def main():
    """Main loop for the LLM worker, polling Redis for tasks."""
    await initialize_clients()
    try:
        while True:
            # BLPOP is blocking pop, waits indefinitely for an item.
            # Using 'llm_task_queue' now for all requests.
            # The long socket_timeout in initialize_clients() handles long waits.
            queue_name, task_json = await redis_client.blpop("llm_task_queue", timeout=0) 
            
            if task_json:
                task_data = json.loads(task_json)
                # Run task processing concurrently to not block the main loop
                asyncio.create_task(process_task(task_data, redis_client, http_client))
            else:
                # This else block is theoretically not hit with timeout=0, but kept for clarity.
                logger.debug("Worker: No tasks in queue. Waiting...")
                await asyncio.sleep(WORKER_POLL_INTERVAL) 

    except asyncio.CancelledError:
        logger.info("Worker: Task queue processing cancelled.")
    except Exception as e:
        logger.exception(f"Worker: Unhandled exception in main loop: {e}")
    finally:
        await close_clients()

if __name__ == "__main__":
    logger.info("Starting LLM Worker.")
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Worker: Shutting down due to KeyboardInterrupt.")
    except Exception as e:
        logger.critical(f"Worker: Fatal error during startup or main loop: {e}")