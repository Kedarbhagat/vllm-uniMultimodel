import time
import json
import random
from locust import HttpUser, task, between
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChatUser(HttpUser):
    """
    Simulates users making streaming chat requests to the FastAPI Gateway.
    """
    wait_time = between(1, 5)  # Users wait 1-5 seconds between requests
    host = "http://172.17.35.82:8082"  # FastAPI Gateway host

    supported_models = [
        "./meta-llama/Llama-3.1-8B-Instruct-awq",
        "./DeepSeek-Coder-V2-Lite-Instruct-awq",
    ]

    generic_messages = [
        {
            "role": "user",
            "content": (
                "A concise, high-level overview that explains the purpose of the joint project, "
                "its impact across three industries (biotechnology, financial modeling, and aerospace engineering), "
                "and the ethical implications of using AI to accelerate innovation in each domain."
            )
        },
    ]

    def on_start(self):
        logger.info(f"Starting Locust user, targeting host: {self.host}")

    @task
    def chat_completion_streaming(self):
        model = random.choice(self.supported_models)
        messages = self.generic_messages

        payload = {
            "model": model,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 1024,
            "stream": True
        }

        request_name = f"/v1/chat/completions (Streaming) - {model.split('/')[-1]}"
        start_time = time.monotonic()

        try:
            with self.client.post(
                "/v1/chat/completions",
                json=payload,
                stream=True,
                name=request_name,
                catch_response=True,
                timeout=60  # ✅ Critical fix: prevents premature timeout under load
            ) as response:
                response.raise_for_status()

                received_chunks = []
                time_to_first_chunk_ms = -1
                first_chunk_processed = False

                for chunk_bytes in response.iter_lines():
                    current_time = time.monotonic()

                    if chunk_bytes:
                        line = chunk_bytes.decode('utf-8')
                        if line.startswith("data: "):
                            json_str = line[len("data: "):].strip()

                            if json_str == "[DONE]":
                                break

                            try:
                                data = json.loads(json_str)

                                if not first_chunk_processed:
                                    if data.get("choices") and data["choices"][0].get("delta", {}).get("content"):
                                        time_to_first_chunk_ms = (current_time - start_time) * 1000
                                        self.environment.events.request.fire(
                                            request_type="Custom",
                                            name=f"TTFC - {model.split('/')[-1]}",
                                            response_time=time_to_first_chunk_ms,
                                            response_length=0,
                                            context=self.context,
                                            exception=None
                                        )
                                        first_chunk_processed = True

                                if data.get("error"):
                                    response.failure(f"Streaming error for {model}: {data['error']}")
                                    logger.error(f"Streaming error for {request_name}: {data['error']}")
                                    return

                                if "choices" in data and len(data["choices"]) > 0:
                                    delta_content = data["choices"][0].get("delta", {}).get("content")
                                    if delta_content:
                                        received_chunks.append(delta_content)

                            except json.JSONDecodeError:
                                response.failure(f"JSON decode error for {model}: {line}")
                                logger.error(f"JSON decode error for {request_name}: {line}")
                                return

                if not received_chunks:
                    response.failure(f"No content received from {model}")
                    logger.warning(f"No chunks from {request_name} at user count {self.environment.runner.user_count}")
                elif not first_chunk_processed:
                    response.failure(f"No valid first content chunk from {model}")
                    logger.warning(f"No first chunk from {request_name} at user count {self.environment.runner.user_count}")
                else:
                    response.success()
                    logger.info(f"Streamed response from {request_name} with {len(received_chunks)} chunks")

        except Exception as e:
            total_response_time = (time.monotonic() - start_time) * 1000
            self.environment.events.request.fire(
                request_type="POST",
                name=request_name,
                response_time=total_response_time,
                response_length=0,
                exception=e,
                context=self.context
            )
            logger.error(f"Exception during streaming request for {request_name}: {e}", exc_info=True)
