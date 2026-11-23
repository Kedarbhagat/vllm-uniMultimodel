# locustfile.py
import json
import random
import time
import uuid
from locust import HttpUser, task, between

def safe_json_resp(resp):
    try:
        return resp.json()
    except Exception:
        return {}

class ChatUser(HttpUser):
    wait_time = between(1.0, 3.5)  # realistic think time between actions

    def on_start(self):
        self.email = f"user_{uuid.uuid4().hex[:8]}@loadtest.test"
        self.threads = []
        try:
            with self.client.post("/chat/create_thread",
                                  json={"email": self.email, "title": "loadtest-init"},
                                  catch_response=True, timeout=30) as res:
                if res.status_code in (200, 201):
                    data = safe_json_resp(res)
                    thread_id = data.get("thread_id") if isinstance(data, dict) else None
                    if thread_id:
                        self.threads.append(thread_id)
                        res.success()  # ADD THIS
                    else:
                        res.failure("no thread_id in response")
                else:
                    res.failure(f"status {res.status_code}")
        except Exception as e:
            # Let Locust record the exception as a failure
            raise

    @task(7)
    def followup_same_thread(self):
        if not self.threads:
            self.on_start()
            if not self.threads:
                return

        thread_id = random.choice(self.threads)
        messages = [
            "What is the capital of France?",
        ]
        message = random.choice(messages)

        payload = {
            "thread_id": thread_id,
            "message": message,
            "model": "llama",
            "temperature": 0.7,
        
        }

        start = time.time()
        try:
            # streaming SSE-style response; use catch_response to mark failure if needed
            with self.client.post("/chat/send", json=payload, stream=True, catch_response=True, timeout=600) as resp:
                if resp.status_code != 200:
                    resp.failure(f"status {resp.status_code}")
                    return

                # consume lines until [DONE]
                success = False
                for raw_line in resp.iter_lines(decode_unicode=True):
                    if not raw_line:
                        continue
                    line = raw_line.strip()
                    if line in ("data: [DONE]", "[DONE]"):
                        success = True
                        break

                # Mark as success only if we got [DONE]
                if success:
                    resp.success()
                else:
                    resp.failure("stream incomplete - no [DONE] marker")
        except Exception:
            # allow Locust to capture the exception/failure automatically
            raise

    @task(3)
    def new_thread_and_question(self):
        try:
            with self.client.post("/chat/create_thread",
                                  json={"email": self.email, "title": f"thread-{int(time.time())}"},
                                  catch_response=True, timeout=30) as res:
                if res.status_code not in (200, 201):
                    res.failure(f"status {res.status_code}")
                    return
                data = safe_json_resp(res)
                thread_id = data.get("thread_id") if isinstance(data, dict) else None
                if not thread_id:
                    res.failure("no thread_id")
                    return
                self.threads.append(thread_id)
                res.success()  # ADD THIS
        except Exception:
            raise

        payload = {
            "thread_id": thread_id,
            "message": "Start a new topic: give an overview and key points.",
            "model": "llama",
            "temperature": 0.3,
            "max_tokens": 512
        }

        try:
            with self.client.post("/chat/send", json=payload, stream=True, catch_response=True, timeout=600) as resp:
                if resp.status_code != 200:
                    resp.failure(f"status {resp.status_code}")
                    return
                success = False
                for raw_line in resp.iter_lines(decode_unicode=True):
                    if not raw_line:
                        continue
                    if raw_line.strip() in ("data: [DONE]", "[DONE]"):
                        success = True
                        break
                
                if success:
                    resp.success()
                else:
                    resp.failure("stream incomplete - no [DONE] marker")
        except Exception:
            raise

    @task(1)
    def small_pause(self):
        time.sleep(random.uniform(0.5, 2.0))