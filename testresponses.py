import requests
import json
import threading
import time

API_URL = "http://172.17.25.83:8081/v1/chat/completions"
RESULT_URL_BASE = "http://172.17.25.83:8081/v1/chat/result/"
NUM_REQUESTS = 10
TASK_IDS = []

def send_request(request_id):
    payload = {
        "model": "./meta-llama/Llama-3.1-8B-Instruct-awq",
        "messages": [
            {
                "role": "user",
                "content": "tell me a story about greek mythology"
            }
        ],
        "temperature": 0.7,
        
    }
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()
        task_data = response.json()
        task_id = task_data.get("task_id")
        if task_id:
            print(f"Request {request_id}: Task ID - {task_id}")
            TASK_IDS.append(task_id)
        else:
            print(f"Request {request_id}: Error getting task ID")
    except requests.exceptions.RequestException as e:
        print(f"Request {request_id}: Error - {e}")

def get_result(task_id):
    result_url = RESULT_URL_BASE + task_id
    while True:
        try:
            response = requests.get(result_url)
            response.raise_for_status()
            result_data = response.json()
            if result_data.get("status") == "completed":
                print(f"Task {task_id}: Result - {result_data.get('result')}")
                break
            elif result_data.get("status") == "failed":
                print(f"Task {task_id}: Failed - {result_data.get('error')}")
                break
            else:
                print(f"Task {task_id}: Pending...")
                time.sleep(2)
        except requests.exceptions.RequestException as e:
            print(f"Task {task_id}: Error getting result - {e}")
            time.sleep(2)

if __name__ == "__main__":
    threads = []
    for i in range(NUM_REQUESTS):
        thread = threading.Thread(target=send_request, args=(i + 1,))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    print("\n--- Retrieving Results ---")
    for task_id in TASK_IDS:
        get_result(task_id)