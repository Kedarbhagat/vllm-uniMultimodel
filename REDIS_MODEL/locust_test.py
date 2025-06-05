import time
import json
from locust import HttpUser, task, between, events
import uuid
import logging
# Ensure sseclient-py is installed: pip install sseclient-py
# import sseclient # Not strictly needed if manually parsing iter_lines for SSE data:

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChatUser(HttpUser):
    """
    User class that makes streaming requests to the FastAPI Gateway.
    """
    wait_time = between(1, 5) # Users wait between 1 and 5 seconds between tasks

    host = "http://172.17.35.82:8082" # Your specific Gateway host

    # Define your supported model(s) here, matching MODEL_TO_BACKEND in your gateway
    supported_models = [
        "./meta-llama/Llama-3.1-8B-Instruct-awq",
    ]

    def on_start(self):
        """ On start, print out the host being used """
        logger.info(f"Starting Locust user, targeting host: {self.host}")

    @task # By default, @task has a weight of 1, making this the only task executed
    def chat_completion_streaming(self):
        """
        Simulates a streaming chat completion request.
        """
        model = self.supported_models[0] # Using the first model for simplicity
        messages = [
            {"role": "user", "content": "A concise, high-level overview that explains the purpose of the joint project, its impact across three industries (biotechnology, financial modeling, and aerospace engineering), and the ethical implications of using AI to accelerate innovation in each domain."},
        ]

        payload = {
            "model": model,
            "messages": messages,
            "temperature": 0.7,
            "stream": True
        }

        request_name = f"/v1/chat/completions (Streaming)" # Consistent naming for stats

        start_time = time.monotonic() # Manually track start time for TTFC

        try:
            with self.client.post(
                "/v1/chat/completions",
                json=payload,
                stream=True,  # Crucial for consuming SSE
                name=request_name,
                catch_response=True
            ) as response:
                response.raise_for_status() # Raise for initial HTTP errors (4xx, 5xx)

                received_chunks = []
                # Flag to ensure TTFC is only recorded once
                time_to_first_chunk_ms = -1
                first_chunk_processed = False 
                
                for chunk_bytes in response.iter_lines():
                    current_time = time.monotonic()

                    if chunk_bytes:
                        line = chunk_bytes.decode('utf-8')
                        
                        # Process only lines that start with "data: "
                        if line.startswith("data: "):
                            json_str = line[len("data: "):].strip()
                            
                            if json_str == "[DONE]":
                                break # End of stream
                            
                            try:
                                data = json.loads(json_str)
                                
                                # Record Time To First Chunk (TTFC)
                                if not first_chunk_processed:
                                    time_to_first_chunk_ms = (current_time - start_time) * 1000
                                    self.environment.events.request.fire(
                                        request_type="Custom", # Use "Custom" for TTFC
                                        name="TTFC",
                                        response_time=time_to_first_chunk_ms,
                                        response_length=0,
                                        context=self.context,
                                        exception=None # Mark as success
                                    )
                                    first_chunk_processed = True
                                
                                # Check for errors in the streaming data (e.g., from worker)
                                if data.get("error"):
                                    response.failure(f"Streaming: Error chunk received: {data['error']}")
                                    logger.error(f"Streaming error for {request_name}: {data['error']}")
                                    return # Exit task on error
                                
                                if "choices" in data and len(data["choices"]) > 0:
                                    delta_content = data["choices"][0].get("delta", {}).get("content")
                                    if delta_content:
                                        received_chunks.append(delta_content)
                                # else:
                                #     logger.warning(f"Unexpected chunk format for {request_name}: {line}") # Debug
                            except json.JSONDecodeError:
                                response.failure(f"Failed to decode JSON from stream: {line}")
                                logger.error(f"Failed to decode JSON from stream for {request_name}: {line}")
                                return # Exit task on JSON error
                        # else:
                        #     logger.debug(f"Skipping non-data line: {line}") # For debugging raw lines

                if not received_chunks:
                    response.failure("No content chunks received in streaming response (stream might be empty or invalid).")
                    logger.warning(f"No content chunks received for {request_name} for user count {self.environment.runner.user_count}")
                elif not first_chunk_processed:
                    # This means we received data but no meaningful first content chunk
                    response.failure("Stream ended without valid first content chunk or [DONE] signal.")
                    logger.warning(f"Stream for {request_name} ended without first content for user count {self.environment.runner.user_count}")
                else:
                    response.success()
                    logger.info(f"Streamed response for {request_name} completed. Total chunks: {len(received_chunks)}")
        except Exception as e:
            # Handle any exceptions that occur outside the response processing (e.g., network errors, timeouts)
            total_response_time = (time.monotonic() - start_time) * 1000 # ms
            self.environment.events.request.fire(
                request_type="POST",
                name=request_name,
                response_time=total_response_time,
                response_length=0,
                exception=e, # Pass the exception to mark as failure
                context=self.context
            )
            logger.error(f"Exception during streaming request for {request_name}: {e}", exc_info=True)

    # You can re-enable non-streaming with a weight > 0 if needed
    # @task(0) # This task has a weight of 0, meaning it will never be picked
    # def chat_completion_non_streaming(self):
    #     logger.info("Non-streaming task (disabled)")
    #     pass
