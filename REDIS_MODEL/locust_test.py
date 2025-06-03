from locust import HttpUser, task, between
import requests  # Import requests to handle the cancellation

class LLMUser(HttpUser):
    wait_time = between(1, 3)
    api_url = "http://172.17.25.83:8081/v1/chat/completions"
    result_url_base = "http://172.17.25.83:8081/v1/chat/result/"
    cancel_url_base = "http://172.17.25.83:8081/v1/chat/cancel/"
    model = "./meta-llama/Llama-3.1-8B-Instruct-awq"
    task_id = None  # To store the task_id of the current task
    
    @task
    def create_chat_completion(self):
        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": "what is quantum computing"}
            ],
            "temperature": 0.7,
            "max_tokens": 50
        }
        headers = {"Content-Type": "application/json"}
        response = self.client.post(self.api_url, json=payload, headers=headers)
        if response.status_code == 200:
            self.task_id = response.json().get("task_id")  # Store task ID
            if self.task_id:
                self.schedule_get_result(self.task_id)

    def schedule_get_result(self, task_id):
        import time
        # Simulate a delay before checking the result
        time.sleep(5)
        self.client.get(self.result_url_base + task_id)

    def cancel_task(self):
        """Trigger the cancellation of the task"""
        if self.task_id:
            cancel_url = f"{self.cancel_url_base}{self.task_id}"
            response = requests.post(cancel_url)
            if response.status_code == 200:
                print(f"Task {self.task_id} cancelled successfully!")
            else:
                print(f"Failed to cancel task {self.task_id}. Status Code: {response.status_code}")
        else:
            print("No task to cancel.")
    
    def on_start(self):
        print("Locust user started.")

    def on_stop(self):
        """Cancel the task when the user stops"""
        print("Locust user stopping...")
        self.cancel_task()  # Cancel the task when stopping the Locust user
