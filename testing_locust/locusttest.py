# locustfile.py - Production-ready version
import json
import random
import time
import uuid
from locust import HttpUser, task, between, events
from locust.exception import RescheduleTask

# Track custom metrics
request_times = []
stream_token_counts = []

@events.request.add_listener
def on_request(request_type, name, response_time, response_length, exception, **kwargs):
    """Custom metric tracking"""
    if exception is None and "chat/send" in name:
        request_times.append(response_time)

def safe_json_resp(resp):
    """Safely parse JSON response"""
    try:
        return resp.json()
    except Exception as e:
        print(f"JSON parse error: {e}, response: {resp.text[:200]}")
        return {}

class ChatUser(HttpUser):
    wait_time = between(1.0, 3.5)  # Think time between actions
    
    def on_start(self):
        """Initialize user with email and first thread"""
        self.email = f"user_{uuid.uuid4().hex[:8]}@loadtest.test"
        self.threads = []
        self.message_count = 0
        self.error_count = 0
        
        # Create initial thread
        self._create_thread("loadtest-init")
        
        if not self.threads:
            print(f"⚠️ User {self.email} failed to create initial thread")
            raise RescheduleTask()

    def _create_thread(self, title="New Thread"):
        """Helper to create a thread"""
        try:
            with self.client.post(
                "/chat/create_thread",
                json={"email": self.email, "title": title},
                catch_response=True,
                timeout=30,
                name="create_thread"
            ) as res:
                if res.status_code in (200, 201):
                    data = safe_json_resp(res)
                    thread_id = data.get("thread_id")
                    if thread_id:
                        self.threads.append(thread_id)
                        res.success()
                        return thread_id
                    else:
                        res.failure("No thread_id in response")
                        return None
                else:
                    res.failure(f"HTTP {res.status_code}: {res.text[:100]}")
                    return None
        except Exception as e:
            print(f"❌ Thread creation failed: {e}")
            return None

    def _send_message(self, thread_id, message, model="llama", temp=0.7, max_tokens=512):
        """Helper to send a message and consume stream"""
        payload = {
            "thread_id": thread_id,
            "message": message,
            "model": model,
            "temperature": temp,
            "max_tokens": max_tokens
        }
        
        token_count = 0
        start_time = time.time()
        first_token_time = None
        
        try:
            with self.client.post(
                "/chat/send",
                json=payload,
                stream=True,
                catch_response=True,
                timeout=120,  # 2 min timeout for streaming
                name="chat_send"
            ) as resp:
                if resp.status_code != 200:
                    resp.failure(f"HTTP {resp.status_code}: {resp.text[:200]}")
                    self.error_count += 1
                    return False
                
                # Consume streaming response
                for raw_line in resp.iter_lines(decode_unicode=True):
                    if not raw_line:
                        continue
                    
                    line = raw_line.strip()
                    
                    # Check for completion
                    if line in ("data: [DONE]", "[DONE]"):
                        break
                    
                    # Parse SSE data
                    if line.startswith("data: "):
                        try:
                            data = json.loads(line[6:])
                            if "token" in data:
                                token_count += 1
                                if first_token_time is None:
                                    first_token_time = time.time() - start_time
                            elif "error" in data:
                                resp.failure(f"Stream error: {data['error']}")
                                self.error_count += 1
                                return False
                        except json.JSONDecodeError:
                            continue
                
                total_time = time.time() - start_time
                
                # Success metrics
                resp.success()
                self.message_count += 1
                stream_token_counts.append(token_count)
                
                # Log metrics for debugging
                if self.message_count % 10 == 0:
                    print(f"✅ User {self.email[:8]}: {self.message_count} messages, "
                          f"TTFT: {first_token_time:.2f}s, Total: {total_time:.2f}s, "
                          f"Tokens: {token_count}")
                
                return True
                
        except Exception as e:
            print(f"❌ Stream error for {self.email[:8]}: {e}")
            self.error_count += 1
            return False

    @task(7)
    def followup_same_thread(self):
        """Send a follow-up message in existing thread"""
        if not self.threads:
            self._create_thread()
            if not self.threads:
                return
        
        thread_id = random.choice(self.threads)
        
        messages = [
            "What is the capital of France?",
            "Explain quantum computing in simple terms.",
            "What are the benefits of exercise?",
            "How does photosynthesis work?",
            "What is machine learning?",
            "Explain blockchain technology.",
            "What causes climate change?",
            "How do vaccines work?",
        ]
        
        message = random.choice(messages)
        self._send_message(thread_id, message, temp=0.7, max_tokens=512)

    @task(3)
    def new_thread_and_question(self):
        """Create new thread and ask a question"""
        thread_id = self._create_thread(f"topic-{int(time.time())}")
        
        if not thread_id:
            return
        
        topics = [
            "Give an overview of artificial intelligence and its key applications.",
            "Explain the basics of web development and popular frameworks.",
            "Describe the fundamentals of data science and analytics.",
            "What are the main concepts in cloud computing?",
            "Explain microservices architecture and its benefits.",
        ]
        
        message = random.choice(topics)
        self._send_message(thread_id, message, temp=0.3, max_tokens=512)

    @task(2)
    def multi_turn_conversation(self):
        """Simulate a multi-turn conversation"""
        if not self.threads:
            return
        
        thread_id = random.choice(self.threads)
        
        conversation = [
            ("What is Python?", 0.5),
            ("Give me an example.", 0.7),
            ("Explain more about that.", 0.8),
        ]
        
        for msg, temp in conversation:
            if not self._send_message(thread_id, msg, temp=temp, max_tokens=256):
                break
            time.sleep(random.uniform(0.5, 1.5))  # Realistic typing delay

    @task(1)
    def check_thread_history(self):
        """Retrieve chat history (tests read performance)"""
        if not self.threads:
            return
        
        thread_id = random.choice(self.threads)
        
        try:
            with self.client.get(
                "/chat/history",
                params={"thread_id": thread_id},
                catch_response=True,
                timeout=10,
                name="get_history"
            ) as resp:
                if resp.status_code == 200:
                    resp.success()
                else:
                    resp.failure(f"HTTP {resp.status_code}")
        except Exception as e:
            print(f"History fetch error: {e}")

    @task(1)
    def small_pause(self):
        """Simulate user reading/thinking"""
        time.sleep(random.uniform(1.0, 3.0))

    def on_stop(self):
        """Cleanup when user stops"""
        print(f"\n📊 User {self.email[:8]} final stats:")
        print(f"   Messages sent: {self.message_count}")
        print(f"   Errors: {self.error_count}")
        if self.message_count > 0:
            success_rate = ((self.message_count - self.error_count) / self.message_count) * 100
            print(f"   Success rate: {success_rate:.1f}%")


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Print summary statistics when test ends"""
    if request_times:
        avg_time = sum(request_times) / len(request_times)
        print(f"\n{'='*60}")
        print(f"📊 LOAD TEST SUMMARY")
        print(f"{'='*60}")
        print(f"Total requests: {len(request_times)}")
        print(f"Average response time: {avg_time:.2f}ms")
        print(f"Min response time: {min(request_times):.2f}ms")
        print(f"Max response time: {max(request_times):.2f}ms")
        
    if stream_token_counts:
        avg_tokens = sum(stream_token_counts) / len(stream_token_counts)
        print(f"Average tokens per response: {avg_tokens:.1f}")
        print(f"{'='*60}\n")