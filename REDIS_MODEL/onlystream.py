import asyncio
import httpx
import json
import sys
import os

API_GATEWAY_URL = os.getenv("API_GATEWAY_URL", "http://172.17.35.82:8082")
MODEL = os.getenv("VLLM_MODEL_NAME", "./meta-llama/Llama-3.1-8B-Instruct-awq")

async def main():
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": "Tell me a very short, simple story about a curious cat named Whiskers."}
        ],
        "temperature": 0.7,
        "max_tokens": 100,
        "stream": True
    }

    headers = {
        "Content-Type": "application/json"
    }

    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream("POST", f"{API_GATEWAY_URL}/v1/chat/completions", json=payload, headers=headers) as response:
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[len("data: "):].strip()
                    if data == "[DONE]":
                        print("\n\n[Done]")
                        break
                    try:
                        chunk = json.loads(data)
                        delta = chunk["choices"][0]["delta"].get("content", "")
                        sys.stdout.write(delta)
                        sys.stdout.flush()
                    except Exception as e:
                        print(f"\n[Error decoding chunk]: {data}\n{e}")

if __name__ == "__main__":
    asyncio.run(main())
