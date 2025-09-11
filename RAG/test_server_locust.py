import json
import time
import uuid
from locust import HttpUser, task, between
from locust.exception import RescheduleTask
import random

class ChatAPIUser(HttpUser):
    wait_time = between(1, 3)  # Wait 1-3 seconds between requests
    
    def on_start(self):
        self.user_email = f"testuser{uuid.uuid4().hex[:8]}@example.com"
        self.thread_id = None
        self.thread_created = False
        
        self.test_messages = [
            "write an essay about ethics of AI",
            "explain quantum computing in simple terms",
            "what are the benefits of renewable energy?",
            "describe the impact of climate change",
            "how does machine learning work?",
            "what is the future of space exploration?",
            "explain blockchain technology",
            "what are the pros and cons of social media?",
            "describe the importance of cybersecurity",
            "how will AI change the job market?"
        ]
        
        self.models = ["deepseek"]
        
        self.create_test_thread()
        print(f"User initialized with email: {self.user_email}, thread_id: {self.thread_id}")
    
    def create_test_thread(self):
        payload = {
            "email": self.user_email,
            "title": f"Load Test Thread - {uuid.uuid4().hex[:6]}",
            "metadata": {"test": True, "created_by": "locust"}
        }
        
        try:
            with self.client.post("/chat/create_thread", json=payload, catch_response=True) as response:
                if response.status_code == 200:
                    data = response.json()
                    if "thread_id" in data:
                        self.thread_id = data["thread_id"]
                        self.thread_created = True
                        response.success()
                        print(f"✓ Thread created successfully: {self.thread_id}")
                    else:
                        response.failure("No thread_id in create_thread response")
                        self.thread_created = False
                else:
                    response.failure(f"Create thread failed with status {response.status_code}")
                    self.thread_created = False
        except Exception as e:
            print(f"✗ Failed to create thread: {str(e)}")
            self.thread_created = False
    
    def ensure_thread_exists(self):
        if not self.thread_created or not self.thread_id:
            print("Thread not available, creating new one...")
            self.create_test_thread()
        return self.thread_created

    def send_chat_message(self, message, model):
        if not self.ensure_thread_exists():
            return
        
        payload = {
            "thread_id": self.thread_id,
            "message": message,
            "model": model
        }
        
        headers = {
            'Accept': 'text/event-stream',
            'Content-Type': 'application/json'
        }
        
        start_time = time.time()
        ttft_ms = None
        
        try:
            with self.client.post(
                "/chat/send",
                json=payload,
                headers=headers,
                stream=True,
                catch_response=True
            ) as response:
                
                if response.status_code != 200:
                    response.failure(f"Expected 200, got {response.status_code}")
                    return
                
                tokens_received = 0
                full_response = ""
                
                for line in response.iter_lines():
                    if line:
                        line_str = line.decode('utf-8')
                        if line_str.startswith('data: '):
                            data_part = line_str[6:]
                            if data_part == '[DONE]':
                                break
                            try:
                                data = json.loads(data_part)
                                if 'token' in data:
                                    if ttft_ms is None:
                                        ttft_ms = (time.time() - start_time) * 1000
                                    full_response += data['token']
                                    tokens_received += 1
                                elif 'error' in data:
                                    response.failure(f"API returned error: {data['error']}")
                                    return
                            except json.JSONDecodeError:
                                continue
                
                total_time_ms = (time.time() - start_time) * 1000
                
                if tokens_received == 0:
                    response.failure("No tokens received in SSE stream")
                elif len(full_response.strip()) == 0:
                    response.failure("Empty response received")
                else:
                    response.success()
                    print(
                        f"✓ Received {tokens_received} tokens | "
                        f"TTFT: {ttft_ms:.0f} ms | "
                        f"Total: {total_time_ms:.0f} ms | "
                        f"Length: {len(full_response)}"
                    )
        
        except Exception as e:
            self.client.get("/", catch_response=True).failure(f"Request failed: {str(e)}")

    @task(3)
    def test_chat_send_basic(self):
        message = random.choice(self.test_messages)
        model = random.choice(self.models)
        self.send_chat_message(message, model)
    
    @task(2)
    def test_chat_send_short_message(self):
        short_messages = [
            "Hello",
            "How are you?",
            "What is AI?",
            "Explain Python",
            "Define machine learning"
        ]
        message = random.choice(short_messages)
        model = random.choice(self.models)
        self.send_chat_message(message, model)

    @task(1)
    def test_health_check(self):
        with self.client.get("/", catch_response=True) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    if "message" in data and "running" in data["message"].lower():
                        response.success()
                    else:
                        response.failure("Unexpected health check response")
                except:
                    response.failure("Invalid JSON in health check")
            else:
                response.failure(f"Health check failed with status {response.status_code}")

    @task(1)
    def test_list_threads(self):
        if not self.ensure_thread_exists():
            return
        with self.client.get(f"/chat/threads?email={self.user_email}", catch_response=True) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    if isinstance(data, list):
                        response.success()
                        thread_ids = [thread.get("thread_id") for thread in data]
                        if self.thread_id not in thread_ids:
                            print(f"Warning: Our thread {self.thread_id} not found in thread list")
                    else:
                        response.failure("Expected list of threads")
                except:
                    response.failure("Invalid JSON response")
            else:
                response.failure(f"List threads failed with status {response.status_code}")

    @task(1)
    def test_chat_history(self):
        if not self.ensure_thread_exists():
            return
        with self.client.get(f"/chat/history?thread_id={self.thread_id}", catch_response=True) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    if isinstance(data, list):
                        response.success()
                    else:
                        response.failure("Expected list of messages")
                except:
                    response.failure("Invalid JSON response")
            else:
                response.failure(f"Chat history failed with status {response.status_code}")

class HighVolumeUser(ChatAPIUser):
    wait_time = between(0.5, 1)
    weight = 1

class NormalUser(ChatAPIUser):
    wait_time = between(2, 5)
    weight = 3

class LongConversationUser(ChatAPIUser):
    wait_time = between(3, 6)
    weight = 2
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conversation_count = 0
    
    @task(3)
    def continue_conversation(self):
        if not self.ensure_thread_exists():
            return
        self.conversation_count += 1
        follow_up_messages = [
            f"Can you elaborate on that? (message #{self.conversation_count})",
            f"What else can you tell me about this topic? (message #{self.conversation_count})",
            f"Give me more details please. (message #{self.conversation_count})",
            f"How does this relate to other concepts? (message #{self.conversation_count})"
        ]
        message = random.choice(follow_up_messages)
        model = random.choice(self.models)
        self.send_chat_message(message, model)

class ThreadManagementUser(ChatAPIUser):
    weight = 1
    
    @task(2)
    def test_create_multiple_threads(self):
        for i in range(3):
            thread_payload = {
                "email": self.user_email,
                "title": f"Test Thread {i+1} - {uuid.uuid4().hex[:4]}",
                "metadata": {"thread_number": i+1, "test": True}
            }
            with self.client.post("/chat/create_thread", json=thread_payload, catch_response=True) as response:
                if response.status_code == 200:
                    try:
                        data = response.json()
                        if "thread_id" in data:
                            response.success()
                        else:
                            response.failure("No thread_id in response")
                    except:
                        response.failure("Invalid JSON response")
                else:
                    response.failure(f"Thread creation failed with status {response.status_code}")
            time.sleep(0.1)
