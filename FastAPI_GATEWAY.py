from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import requests
import json
import asyncio

app = FastAPI()

# CORS setup for cross-origin access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Map of supported models and their backend URLs
MODEL_TO_BACKEND = {
    "microsoft/Phi-4-mini-instruct": "http://172.17.25.83:8000/v1/chat/completions",
    "./meta-llama/Llama-3.1-8B-Instruct-awq": "http://172.31.21.186:8000/v1/chat/completions",
    "./mistral-instruct-v0.2-awq": "http://172.17.29.25:8000/v1/chat/completions"
}

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    model = body.get("model")
    messages = body.get("messages")
    stream = bool(body.get("stream", False))

    # Validate model
    if not model or model not in MODEL_TO_BACKEND:
        raise HTTPException(status_code=400, detail=f"Unknown or missing model: {model}")
    
    # Validate messages
    if not messages or not isinstance(messages, list):
        raise HTTPException(status_code=400, detail="`messages` must be a non-empty list")

    backend_url = MODEL_TO_BACKEND[model]
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": body.get("max_tokens", 7000),
        "temperature": body.get("temperature", 0.7),
        "top_p": body.get("top_p", 1.0),
        "stream": stream
    }

    # Handle Streaming Response
    if stream:
        def stream_vllm():
            try:
                with requests.post(backend_url, headers=headers, json=payload, stream=True, timeout=60) as resp:
                    if resp.status_code != 200:
                        raise HTTPException(status_code=resp.status_code, detail=resp.text)
                    for line in resp.iter_lines():
                        if not line:
                            continue
                        text = line.decode("utf-8")
                        if text.startswith("data: "):
                            text = text[len("data: "):]
                        if text.strip() == "[DONE]":
                            yield "data: [DONE]\n\n"
                            break
                        yield f"data: {text}\n\n"
            except Exception as e:
                yield f"data: {{\"error\": \"{str(e)}\"}}\n\n"

        return StreamingResponse(stream_vllm(), media_type="text/event-stream")

    # Handle Non-streaming Response
    else:
        try:
            resp = await asyncio.to_thread(
                lambda: requests.post(backend_url, headers=headers, json=payload, timeout=60)
            )
            return JSONResponse(content=resp.json(), status_code=resp.status_code)
        except requests.RequestException as e:
            raise HTTPException(status_code=500, detail=f"Request error: {str(e)}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
