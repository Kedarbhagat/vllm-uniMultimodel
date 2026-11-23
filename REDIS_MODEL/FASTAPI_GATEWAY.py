import os
import json
import uuid
import logging
import asyncio
from typing import Dict, Any, List, Union, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx

# Config: backend URL where vllm serve listens. Default to localhost:8000
BACKEND_URL = os.getenv("BACKEND_URL", "http://172.31.21.186:8000/v1/chat/completions")
API_PORT = int(os.getenv("API_PORT", 8082))

# NEW: API key to forward to backend (e.g. token-abc123)
BACKEND_API_KEY = os.getenv("BACKEND_API_KEY", "token-abc123")
# NEW: header name to use. If "Authorization", we send "Bearer <key>".
# If you prefer a raw header like x-api-key, set BACKEND_API_KEY_HEADER="x-api-key"
BACKEND_API_KEY_HEADER = os.getenv("BACKEND_API_KEY_HEADER", "Authorization")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("simple_gateway")

app = FastAPI(title="Simple vLLM Gateway", version="0.1")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False

def model_to_dict(m: BaseModel) -> Dict[str, Any]:
    # Support both Pydantic v1 and v2
    if hasattr(m, "model_dump"):
        return m.model_dump()
    return m.dict()

def build_backend_headers() -> Dict[str, str]:
    headers: Dict[str, str] = {}
    if BACKEND_API_KEY:
        if BACKEND_API_KEY_HEADER.lower() == "authorization":
            headers["Authorization"] = f"Bearer {BACKEND_API_KEY}"
        else:
            headers[BACKEND_API_KEY_HEADER] = BACKEND_API_KEY
        logger.info("Gateway will forward backend API key using header: %s", BACKEND_API_KEY_HEADER)
    return headers

@app.get("/health")
async def health():
    return {"status": "ok", "backend": BACKEND_URL, "api_key_forwarded": bool(BACKEND_API_KEY)}

@app.post("/v1/chat/completions", response_model=Union[Dict[str, Any], Dict[str, str]])
async def chat_completions(req: ChatRequest):
    # Always forward to BACKEND_URL (single model)
    logger.info("Forwarding request to backend: %s", BACKEND_URL)
    payload = model_to_dict(req)
    # attach a trace id for visibility (backend may ignore)
    payload.setdefault("task_id", str(uuid.uuid4()))

    headers = build_backend_headers()

    # Streaming: open streaming POST to backend and proxy bytes to client
    if payload.get("stream"):
        async def stream_proxy():
            async with httpx.AsyncClient() as client:
                try:
                    # no timeout so backend can stream indefinitely
                    async with client.stream("POST", BACKEND_URL, json=payload, headers=headers, timeout=None) as resp:
                        if resp.status_code >= 400:
                            text = (await resp.aread()).decode(errors="ignore")
                            logger.error("Backend error %s: %s", resp.status_code, text)
                            yield f"data: {json.dumps({'error':'backend_error','detail':text})}\n\n".encode()
                            yield b"data: [DONE]\n\n"
                            return
                        # Forward raw bytes as-is (works if backend uses SSE)
                        async for chunk in resp.aiter_bytes():
                            if not chunk:
                                continue
                            yield chunk
                except (httpx.ConnectError, httpx.ConnectTimeout) as e:
                    logger.error("Connect error to backend: %s", e)
                    yield f"data: {json.dumps({'error':'backend_unreachable','detail':str(e)})}\n\n".encode()
                    yield b"data: [DONE]\n\n"
                except asyncio.CancelledError:
                    logger.info("Client disconnected while streaming")
                except Exception as e:
                    logger.exception("Unexpected proxy error: %s", e)
                    yield f"data: {json.dumps({'error':'gateway_error','detail':str(e)})}\n\n".encode()
                    yield b"data: [DONE]\n\n"

        return StreamingResponse(stream_proxy(), media_type="text/event-stream",
                                 headers={"Cache-Control":"no-cache","Connection":"keep-alive","X-Accel-Buffering":"no"})
    # Non-streaming: POST and return backend JSON
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.post(BACKEND_URL, json=payload, headers=headers, timeout=300.0)
        except httpx.RequestError as e:
            logger.exception("Backend request failed: %s", e)
            raise HTTPException(status_code=502, detail=str(e))
        if resp.status_code >= 400:
            logger.error("Backend error %s: %s", resp.status_code, resp.text)
            raise HTTPException(status_code=resp.status_code, detail=resp.text)
        try:
            return JSONResponse(content=resp.json(), status_code=200)
        except Exception:
            return JSONResponse(content={"result": resp.text}, status_code=200)

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting simple gateway on 0.0.0.0:%s -> backend %s", API_PORT, BACKEND_URL)
    uvicorn.run(app, host="0.0.0.0", port=API_PORT)