# app.py
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import requests
import json
import asyncio

app = FastAPI()

# 🌐 Allow CORS if you’ll call from JS frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # in prod, lock this down to your frontends
    allow_methods=["*"],
    allow_headers=["*"],
)

# ▼ Change these per-deployment ▼
VLLM_ENDPOINT = "http://172.31.16.1:8000/v1/chat/completions"
DEFAULT_MODEL  = "microsoft/Phi-4-mini-instruct"

def _stream_vllm(messages, model):
    """
    Generator that yields Server-Sent Events (SSE) 
    coming back from vLLM’s streaming API.
    """
    headers = {"Content-Type": "application/json"}
    payload = {"model": model, "messages": messages, "stream": True}

    # Fire the request to vLLM
    resp = requests.post(VLLM_ENDPOINT, headers=headers, json=payload, stream=True, timeout=60)
    if resp.status_code != 200:
        # bubble up any erro
        raise HTTPException(resp.status_code, detail=resp.text)

    # stream back each chunk as SSE
    for line in resp.iter_lines():
        if not line:
            continue
        text = line.decode("utf-8")
        # strip the “data: ” prefix if present
        if text.startswith("data: "):
            text = text[len("data: "):]
        # break on the sentinel
        if text.strip() == "[DONE]":
            break
        yield f"data: {text}\n\n"


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """
    Proxy endpoint for both streaming and non-streaming completions.
    
    Body JSON schema:
    {
      "model": "<optional model name>",
      "stream": true|false,
      "messages": [ { "role": "...", "content": "..." }, ... ]
    }
    """
    body = await request.json()
    messages = body.get("messages")
    if not messages or not isinstance(messages, list):
        raise HTTPException(400, detail="`messages` must be a non-empty list")

    model  = body.get("model", DEFAULT_MODEL)
    stream = bool(body.get("stream", False))

    if stream:
        # Server-Sent Events: text/event-stream
        return StreamingResponse(
            _stream_vllm(messages, model),
            media_type="text/event-stream",
        )
    else:
        # Simple pass-through for non-streaming
        headers = {"Content-Type": "application/json"}
        payload = {"model": model, "messages": messages, "stream": False}
        resp = await asyncio.to_thread(
            lambda: requests.post(VLLM_ENDPOINT, headers=headers, json=payload, timeout=60)
        )
        return JSONResponse(content=resp.json(), status_code=resp.status_code)
