import redis
import json
import time
import requests
import logging
from redis.exceptions import ConnectionError, TimeoutError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('worker')

# Model to backend mapping
MODEL_TO_BACKEND = {
    "./meta-llama/Llama-3.1-8B-Instruct-awq": "http://172.31.21.186:8000/v1/chat/completions",
}

def connect_to_redis(host="172.31.21.186", port=6379):
    retries = 5
    while retries > 0:
        try:
            redis_client = redis.Redis(
                host=host,
                port=port,
                db=0,
                decode_responses=True,
                socket_timeout=30,
                socket_connect_timeout=30,
            )
            redis_client.ping()
            logger.info("Connected to Redis")
            return redis_client
        except (ConnectionError, TimeoutError) as e:
            logger.error(f"Error connecting to Redis: {e}, retrying... {retries} attempts left.")
            retries -= 1
            time.sleep(2)
    raise ConnectionError("Unable to connect to Redis after several retries")

def check_cancelled(task_id, redis_client):
    return redis_client.get(f"cancel:{task_id}") == "1"

def process_task(task, redis_client):
    try:
        task_id = task["task_id"]

        # Cancelled before processing
        if check_cancelled(task_id, redis_client):
            logger.info(f"Task {task_id} was cancelled before processing.")
            redis_client.set(f"result:{task_id}", json.dumps({
                "task_id": task_id,
                "status": "cancelled",
                "result": None
            }), ex=3600)
            return

        backend_url = MODEL_TO_BACKEND.get(task["model"])
        if not backend_url:
            redis_client.set(f"result:{task_id}", json.dumps({
                "task_id": task_id,
                "status": "failed",
                "error": "Invalid model"
            }), ex=3600)
            logger.error(f"Invalid model for task {task_id}.")
            return

        logger.info(f"Streaming task {task_id} using backend {backend_url}")
        headers = {"Content-Type": "application/json"}

        response = requests.post(
            backend_url,
            json=task,
            headers=headers,
            stream=True,
            timeout=300
        )
        response.raise_for_status()

        stream_key = f"stream:{task_id}"
        redis_client.delete(stream_key)  # Clear old stream data if any

        for line in response.iter_lines(decode_unicode=True):
            if line:
                if check_cancelled(task_id, redis_client):
                    logger.info(f"Task {task_id} was cancelled during streaming.")
                    redis_client.rpush(stream_key, "__cancelled__")
                    return
                redis_client.rpush(stream_key, line)

        redis_client.rpush(stream_key, "__done__")
        redis_client.set(f"result:{task_id}", json.dumps({
            "task_id": task_id,
            "status": "completed",
            "result": "streamed"
        }), ex=3600)
        logger.info(f"Task {task_id} completed with streamed response.")

    except requests.exceptions.RequestException as e:
        redis_client.set(f"result:{task['task_id']}", json.dumps({
            "task_id": task["task_id"],
            "status": "failed",
            "error": f"API request failed: {str(e)}"
        }), ex=3600)
        logger.error(f"API error for task {task['task_id']}: {str(e)}")

    except Exception as e:
        redis_client.set(f"result:{task['task_id']}", json.dumps({
            "task_id": task["task_id"],
            "status": "failed",
            "error": f"Task processing error: {str(e)}"
        }), ex=3600)
        logger.error(f"Error processing task {task['task_id']}: {str(e)}")

def main():
    redis_client = None
    while True:
        try:
            if redis_client is None:
                redis_client = connect_to_redis()

            queue_item = redis_client.blpop("task_queue", timeout=5)
            if queue_item:
                _, task_data = queue_item
                task = json.loads(task_data)
                process_task(task, redis_client)

        except redis.exceptions.ConnectionError:
            logger.warning("Redis connection lost. Reconnecting...")
            redis_client = None
            time.sleep(2)

        except (TimeoutError, Exception) as e:
            logger.error(f"Worker error: {str(e)}")
            time.sleep(2)

        except KeyboardInterrupt:
            logger.info("Worker shutdown requested. Exiting.")
            break

if __name__ == "__main__":
    main()
