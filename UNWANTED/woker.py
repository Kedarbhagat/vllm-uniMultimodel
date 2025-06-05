import os
import json
import time
import signal # For graceful shutdown
import logging
import requests # Synchronous requests are fine here as it's a blocking worker
from typing import Dict, Any, Optional
import redis
from redis.exceptions import ConnectionError, TimeoutError, RedisError

# --- Configuration (using Environment Variables) ---
REDIS_HOST = os.getenv("REDIS_HOST", "172.31.21.186")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
VLLM_BACKEND_URL = os.getenv("VLLM_BACKEND_URL", "http://172.31.21.186:8000/v1/chat/completions")
TASK_QUEUE_NAME = os.getenv("TASK_QUEUE_NAME", "task_queue")
RESULT_KEY_PREFIX = os.getenv("RESULT_KEY_PREFIX", "result:")
CANCEL_KEY_PREFIX = os.getenv("CANCEL_KEY_PREFIX", "cancel:")
TASK_EXPIRATION_SECONDS = int(os.getenv("TASK_EXPIRATION_SECONDS", 3600)) # Results expire after 1 hour

# Model to backend mapping (Worker also needs to know which models it serves)
MODEL_TO_BACKEND: Dict[str, str] = {
    "./meta-llama/Llama-3.1-8B-Instruct-awq": VLLM_BACKEND_URL,
}

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('worker')

# --- Graceful Shutdown Flag ---
SHUTDOWN_REQUESTED = False

def signal_handler(signum, frame):
    """Handles OS signals for graceful shutdown."""
    global SHUTDOWN_REQUESTED
    logger.info(f"Signal {signum} received. Initiating graceful shutdown...")
    SHUTDOWN_REQUESTED = True

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
signal.signal(signal.SIGTERM, signal_handler) # docker stop, kill

# --- Redis Connection Helper ---
def connect_to_redis(host: str, port: int, retries: int = 5, delay: int = 2) -> redis.Redis:
    """Connect to Redis with retry mechanism."""
    current_retries = retries
    while current_retries > 0:
        try:
            client = redis.Redis(
                host=host,
                port=port,
                db=0,
                decode_responses=True,
                socket_timeout=30,  # Longer timeouts for worker as tasks can take time
                socket_connect_timeout=30,
            )
            client.ping()
            logger.info("Successfully connected to Redis.")
            return client
        except (ConnectionError, TimeoutError) as e:
            logger.error(f"Error connecting to Redis: {e}. Retrying in {delay}s... ({current_retries-1} attempts left)")
            current_retries -= 1
            time.sleep(delay)
    raise ConnectionError(f"Unable to connect to Redis after {retries} retries.")

def check_cancelled(task_id: str, r: redis.Redis) -> bool:
    """Check if a task has been explicitly cancelled."""
    return r.get(f"{CANCEL_KEY_PREFIX}{task_id}") == "1"

def store_result(task_id: str, status: str, result_data: Optional[Any], r: redis.Redis, error: Optional[str] = None):
    """Helper to store task results in Redis."""
    data_to_store = {
        "task_id": task_id,
        "status": status,
        "result": result_data,
        "error": error
    }
    try:
        r.set(f"{RESULT_KEY_PREFIX}{task_id}", json.dumps(data_to_store), ex=TASK_EXPIRATION_SECONDS)
        logger.info(f"Task {task_id} status '{status}' and result stored successfully.")
    except RedisError as e:
        logger.error(f"Failed to store result for task {task_id} in Redis: {e}")

def process_task(task: Dict[str, Any], r: redis.Redis):
    """Processes a single non-streaming LLM inference task."""
    task_id = task.get("task_id")
    if not task_id:
        logger.error(f"Received task with no 'task_id': {task}. Skipping.")
        return

    logger.info(f"Attempting to process task: {task_id}")

    try:
        # Check for cancellation before expensive LLM call
        if check_cancelled(task_id, r):
            logger.info(f"Task {task_id} was cancelled before starting processing. Marking as cancelled.")
            store_result(task_id, "cancelled", None, r)
            return

        model_name = task.get("model")
        if not model_name or model_name not in MODEL_TO_BACKEND:
            logger.error(f"Invalid or unsupported model '{model_name}' for task {task_id}. Skipping.")
            store_result(task_id, "failed", None, r, error=f"Invalid or unsupported model: {model_name}")
            return

        backend_url = MODEL_TO_BACKEND[model_name]
        logger.info(f"Forwarding task {task_id} to vLLM backend: {backend_url}")

        # Prepare payload for vLLM backend
        # Ensure 'stream' is explicitly False for non-streaming worker processing
        vllm_payload = {
            "model": model_name,
            "messages": task.get("messages", []),
            "temperature": task.get("temperature"),
            "max_tokens": task.get("max_tokens"),
            "stream": False # Important: Worker asks for non-streaming response
            # Add other parameters like top_p, presence_penalty, etc. if supported and required
        }
        # Remove None values from payload to avoid issues with some APIs
        vllm_payload = {k: v for k, v in vllm_payload.items() if v is not None}

        headers = {"Content-Type": "application/json"}
        # Timeout for the vLLM call. Adjust based on expected max generation time.
        request_timeout = 300 # seconds

        response = requests.post(
            backend_url,
            json=vllm_payload,
            headers=headers,
            timeout=request_timeout
        )
        response.raise_for_status() # Raises HTTPError for bad responses (4xx or 5xx)

        llm_result = response.json()
        store_result(task_id, "completed", llm_result, r)
        logger.info(f"Task {task_id} successfully completed by vLLM backend.")

    except requests.exceptions.Timeout:
        logger.error(f"Task {task_id} failed: Request to vLLM backend timed out after {request_timeout}s.")
        store_result(task_id, "failed", None, r, error=f"LLM backend request timed out.")
    except requests.exceptions.ConnectionError as e:
        logger.error(f"Task {task_id} failed: Connection error to vLLM backend: {e}")
        store_result(task_id, "failed", None, r, error=f"Connection error to LLM backend: {e}")
    except requests.exceptions.HTTPError as e:
        logger.error(f"Task {task_id} failed: vLLM backend returned HTTP error {e.response.status_code}: {e.response.text}")
        store_result(task_id, "failed", None, r, error=f"LLM backend returned error: {e.response.status_code} - {e.response.text}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Task {task_id} failed: General request error to vLLM backend: {e}")
        store_result(task_id, "failed", None, r, error=f"General LLM backend request error: {e}")
    except json.JSONDecodeError:
        logger.error(f"Task {task_id} failed: Failed to decode JSON response from vLLM backend.")
        store_result(task_id, "failed", None, r, error="Invalid JSON response from LLM backend.")
    except Exception as e:
        logger.exception(f"An unexpected error occurred while processing task {task_id}.")
        store_result(task_id, "failed", None, r, error=f"Unexpected worker error: {str(e)}")

def main():
    redis_client = None
    logger.info(f"Worker starting. Listening on Redis queue: '{TASK_QUEUE_NAME}'")
    while not SHUTDOWN_REQUESTED:
        try:
            if redis_client is None:
                redis_client = connect_to_redis(REDIS_HOST, REDIS_PORT)

            # blpop blocks until an item is available or timeout occurs
            queue_item = redis_client.blpop(TASK_QUEUE_NAME, timeout=1) # Short timeout to check shutdown flag
            if queue_item:
                _, task_data = queue_item
                try:
                    task = json.loads(task_data)
                    process_task(task, redis_client)
                except json.JSONDecodeError:
                    logger.error(f"Failed to decode task from queue: {task_data}. Skipping.")
                except Exception as e:
                    logger.exception(f"Error processing task from queue: {e}")
            else:
                # No task in queue, check shutdown flag
                pass

        except redis.exceptions.ConnectionError:
            logger.warning("Lost connection to Redis. Attempting to reconnect...")
            redis_client = None # Force reconnection
            time.sleep(5) # Wait before retrying connection to prevent tight loop
        except RedisError as e:
            logger.error(f"Redis specific error in main loop: {e}. Reconnecting...")
            redis_client = None
            time.sleep(5)
        except Exception as e:
            logger.exception(f"Unhandled error in worker main loop: {e}")
            time.sleep(5) # Prevent tight loop on unhandled errors

    logger.info("Graceful shutdown complete. Exiting worker.")

if __name__ == "__main__":
    main()