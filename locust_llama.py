from locust import HttpUser, task, between
import json

class LlamaLoadTest(HttpUser):
    wait_time = between(1, 2)
    host = "http://172.17.25.83:8080"  # ✅ Match your `url` base

    @task
    def chat_completion(self):
        headers = {
            "Content-Type": "application/json"
        }

        payload = {
            "model": "meta-llama/Llama-3.1-8B-Instruct",
            "messages": [
                {"role": "user", "content": " what are neural networks. explain perceptron training rule    "}
            ],
            "temperature": 0.7
        }

        with self.client.post(
            url="/v1/chat/completions",
            headers=headers,
            data=json.dumps(payload),
            catch_response=True
        ) as response:
            if response.status_code != 200:
                response.failure(f"{response.status_code} - {response.text}")
