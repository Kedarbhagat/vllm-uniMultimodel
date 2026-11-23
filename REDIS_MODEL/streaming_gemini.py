import httpx
import asyncio
import json
import os
import time
import sys

# --- Configuration ---
API_GATEWAY_URL = os.getenv("API_GATEWAY_URL", "http://localhost:8082")
VLLM_MODEL_NAME = os.getenv("VLLM_MODEL_NAME", "./meta-llama/Llama-3.1-8B-Instruct-awq")
NON_STREAMING_POLLING_INTERVAL = 1 # seconds
NON_STREAMING_MAX_POLLING_ATTEMPTS = 30
STREAMING_MAX_DURATION = 60 # seconds, for streaming test timeout

def print_separator():
    print("\n" + "="*80 + "\n")

async def test_health_check(client: httpx.AsyncClient):
    print("--- Testing Health Check ---")
    try:
        response = await client.get(f"{API_GATEWAY_URL}/health")
        response.raise_for_status()
        health_response = response.json()
        print(f"Health Check Response: {json.dumps(health_response, indent=2)}")
        assert health_response.get("status") == "ok"
        assert health_response.get("redis_connected") is True
        print("Health Check PASSED.\n")
    except httpx.RequestError as e:
        print(f"Health Check FAILED: Request error: {e}")
        sys.exit(1)
    except httpx.HTTPStatusError as e:
        print(f"Health Check FAILED: HTTP error: {e.response.status_code} - {e.response.text}")
        sys.exit(1)
    except AssertionError as e:
        print(f"Health Check FAILED: Assertion error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Health Check FAILED: Unexpected error: {e}")
        sys.exit(1)
    print_separator()

async def test_non_streaming_completion(client: httpx.AsyncClient):
    print("--- Testing Non-Streaming Completion (Async via Redis Queue) ---")
    prompt = "What is the capital of France?"
    payload = {
        "model": VLLM_MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": 50,
        "stream": False # Explicitly non-streaming
    }

    print(f"Sending non-streaming request for model: {VLLM_MODEL_NAME}")
    try:
        response = await client.post(f"{API_GATEWAY_URL}/v1/chat/completions", json=payload)
        response.raise_for_status()
        initial_response = response.json()
        print(f"Initial Response: Task ID: {initial_response.get('task_id')}, Status: {initial_response.get('status')}")

        task_id = initial_response.get("task_id")
        if not task_id:
            raise ValueError("No task_id received in initial response.")

        # Poll for the result
        print(f"Polling for result of task ID: {task_id}")
        for i in range(1, NON_STREAMING_MAX_POLLING_ATTEMPTS + 1):
            await asyncio.sleep(NON_STREAMING_POLLING_INTERVAL)
            poll_response = await client.get(f"{API_GATEWAY_URL}/v1/chat/result/{task_id}")
            poll_response.raise_for_status()
            task_status = poll_response.json()
            print(f"Polling attempt {i}/{NON_STREAMING_MAX_POLLING_ATTEMPTS}: Status: {task_status.get('status')}")

            if task_status.get("status") == "completed":
                print("Task Completed!")
                full_result = task_status
                print(f"Full Result:\n{json.dumps(full_result, indent=2)}")
                assert "Paris" in full_result["result"]["choices"][0]["message"]["content"]
                print("Non-Streaming Completion PASSED.\n")
                return
            elif task_status.get("status") == "failed":
                print(f"Task Failed! Error: {task_status.get('error')}")
                raise Exception(f"Task {task_id} failed: {task_status.get('error')}")
        
        raise TimeoutError(f"Non-streaming task {task_id} did not complete within {NON_STREAMING_MAX_POLLING_ATTEMPTS * NON_STREAMING_POLLING_INTERVAL} seconds.")

    except (httpx.RequestError, httpx.HTTPStatusError, ValueError, TimeoutError, AssertionError) as e:
        print(f"Non-Streaming Completion FAILED: {e}\n")
        sys.exit(1)
    except Exception as e:
        print(f"Non-Streaming Completion FAILED: Unexpected error: {e}\n")
        sys.exit(1)
    finally:
        print_separator()

async def test_streaming_completion(client: httpx.AsyncClient):
    print("--- Testing Streaming Completion (Via Redis Queue & Stream) ---")
    prompt = "Tell me a very short, simple story about a curious cat named Whiskers."
    payload = {
        "model": VLLM_MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 100, # Keep max_tokens reasonable
        "stream": True # Explicitly streaming
    }

    print(f"Sending streaming request for model: {VLLM_MODEL_NAME}")
    full_response_content = ""
    chunk_count = 0
    start_time = time.time()

    try:
        # For streaming, we expect a StreamingResponse immediately
        # The initial response will be a TaskResponse if the gateway queues it first
        response = await client.post(f"{API_GATEWAY_URL}/v1/chat/completions", json=payload, timeout=STREAMING_MAX_DURATION)
        response.raise_for_status()

        # Check if the initial response is a TaskResponse (meaning it's queued)
        # This will be true if your gateway returns TaskResponse immediately then streams
        # If it directly returns StreamingResponse, this parsing might fail/not be needed
        try:
            initial_response_json = response.json()
            if initial_response_json.get("task_id") and initial_response_json.get("status") == "pending":
                print(f"Initial Gateway Response: Task ID: {initial_response_json.get('task_id')}, Status: {initial_response_json.get('status')}")
                # The actual streaming will happen as the connection stays open.
                # The httpx client will then iterate over the streamed bytes.
                # No separate polling for streaming, the response *is* the stream.
                print("Gateway accepted streaming request. Now receiving stream chunks...")
                
                # We need to re-make the request as a stream if the first one was just task_id
                # However, with the current design, the Gateway either returns JSON or starts stream.
                # So the logic below assumes the first response IS the stream.
                # If your gateway returns the TaskResponse then closes, this will need adjustment.
                # Assuming the gateway directly streams after queuing for this example.
                pass # Continue to iter_bytes if this was a StreamingResponse

        except json.JSONDecodeError:
            # This is expected if the response is immediately a stream, not JSON
            print("Gateway responded with a direct stream (as expected for streaming).")
            pass

        # Now, iterate over the stream of bytes
        async for chunk_bytes in response.aiter_bytes():
            if time.time() - start_time > STREAMING_MAX_DURATION:
                raise TimeoutError(f"Streaming exceeded max duration of {STREAMING_MAX_DURATION} seconds.")

            decoded_chunk = chunk_bytes.decode('utf-8')
            # Look for Server-Sent Events 'data: ' lines
            for line in decoded_chunk.splitlines():
                if line.startswith("data: "):
                    json_str = line[len("data: "):].strip()
                    if json_str == "[DONE]":
                        print("\nStream [DONE] event received.")
                        assert chunk_count > 0, "Received [DONE] without any prior content chunks."
                        print("Streaming Completion PASSED.\n")
                        return

                    try:
                        chunk_data = json.loads(json_str)
                        if chunk_data.get("choices") and len(chunk_data["choices"]) > 0:
                            delta_content = chunk_data["choices"][0].get("delta", {}).get("content")
                            if delta_content:
                                full_response_content += delta_content
                                sys.stdout.write(delta_content)
                                sys.stdout.flush()
                                chunk_count += 1
                        elif chunk_data.get("error"):
                            print(f"\nStream ERROR event received: {chunk_data['error']}")
                            raise Exception(f"Stream error: {chunk_data['error']}")
                        else:
                            print(f"\n[Warning] Received unexpected chunk format: {json_str}")

                    except json.JSONDecodeError:
                        print(f"\n[Warning] Could not decode JSON from chunk: {json_str}")
                # else:
                #     print(f"\n[Debug] Non-data line: {line}") # Uncomment for deeper debugging

        # If the loop finishes without a [DONE]
        raise RuntimeError("Streaming finished without a [DONE] event.")

    except TimeoutError as e:
        print(f"\nStreaming Completion FAILED: {e}")
        sys.exit(1)
    except (httpx.RequestError, httpx.HTTPStatusError) as e:
        print(f"\nStreaming Completion FAILED: HTTP Request Error: {e}")
        if e.response:
            print(f"Response status: {e.response.status_code}")
            print(f"Response content: {e.response.text}")
        sys.exit(1)
    except AssertionError as e:
        print(f"\nStreaming Completion FAILED: Assertion error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nStreaming Completion FAILED: Unexpected error: {e}")
        sys.exit(1)
    finally:
        print_separator()
        print(f"Total characters received in streaming: {len(full_response_content)}")


async def main():
    print(f"Starting LLM Application Test Script.")
    print(f"API Gateway URL: {API_GATEWAY_URL}")
    print(f"vLLM Model Name: {VLLM_MODEL_NAME}\n")

    print("================================================================================")
    print("Giving worker and gateway a moment to fully initialize...")
    time.sleep(5) # Give the services time to spin up
    print("Proceeding with tests.\n")

    # Use a single httpx client for all tests
    async with httpx.AsyncClient() as client:
        await test_health_check(client)
        await test_non_streaming_completion(client)
        await test_streaming_completion(client)

    print("All tests completed.")

if __name__ == "__main__":
    asyncio.run(main())