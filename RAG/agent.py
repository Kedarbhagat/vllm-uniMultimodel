# agent.py - Pure Python version
import os
import json
import httpx
from typing import List, AsyncIterator

MODEL_REGISTRY = {
    "llama": "./meta-llama/Llama-3.1-8B-Instruct-awq",
    "deepseek": "./DeepSeek-Coder-V2-Lite-Instruct-awq",
    "qwen2.5": "/mnt/c/Users/STUDENT/qwen2.5-coder-14b-instruct-awq-final",
    "gpt-oss": "openai/gpt-oss-20b"
}

# Map models to their respective API gateway ports
API_PORT_MAPPING = {
    "llama": os.getenv("LLAMA_API_URL", "http://127.0.0.1:8000"),
    "deepseek": os.getenv("DEEPSEEK_API_URL", "http://127.0.0.1:8002"),
    "gpt-oss": os.getenv("GPT_OSS_API_URL", "http://127.0.0.1:8004")
}

# Default fallback
API_GATEWAY_URL = API_PORT_MAPPING["gpt-oss"]



class ConversationalAgent:
    def __init__(self, model_key: str = "llama"):
        print(f"🚀 ConversationalAgent.__init__ called with model_key='{model_key}'")
        self.model_key = model_key
        self.model = MODEL_REGISTRY.get(model_key, MODEL_REGISTRY["llama"])
        
        # Get the correct API gateway URL for this model
        self.api_gateway_url = API_PORT_MAPPING.get(model_key, API_GATEWAY_URL)
        self.api_url = f"{self.api_gateway_url}/v1/chat/completions"
        print(f"   Model: {self.model}")
        print(f"   API Gateway URL: {self.api_gateway_url}")
        print(f"   API Endpoint: {self.api_url}")
        self.api_key = os.getenv("YOUR_API_KEY", "token-abc123")
        
        # Reuse HTTP client with connection pooling
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(300.0, connect=10.0),
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=100)
        )
    
    async def astream_response(
        self, 
        messages: List[dict],
        temperature: float = 0.7,
        max_tokens: int = 2048
    ) -> AsyncIterator[str]:
        """
        Stream tokens from vLLM using async HTTP
        
        Args:
            messages: List of dicts like [{"role": "user", "content": "Hello"}]
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True
        }
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            async with self.client.stream(
                "POST", 
                self.api_url, 
                json=payload, 
                headers=headers
            ) as response:
                response.raise_for_status()
                
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:]  # Remove "data: " prefix
                        
                        if data_str.strip() == "[DONE]":
                            break
                        
                        try:
                            chunk = json.loads(data_str)
                            delta = chunk.get("choices", [{}])[0].get("delta", {})
                            content = delta.get("content", "")
                            reasoning = delta.get("reasoning_content", "") or chunk.get("reasoning_content", "")

                            # Forward content and reasoning as structured items for server to pass through
                            if content:
                                yield {"token": content}
                            if reasoning:
                                yield {"reasoning_content": reasoning}
                                
                        except json.JSONDecodeError:
                            continue
                            
        except Exception as e:
            print(f"❌ Streaming error: {e}")
            yield f"\n❌ Error: {str(e)}"
    
    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()


def format_messages(history: List[dict], system_prompt: str = "") -> List[dict]:
    """
    Convert DB history to vLLM message format
    
    Args:
        history: List of {"role": "human"|"ai", "content": "..."}
        system_prompt: Optional system message
    
    Returns:
        List of {"role": "system"|"user"|"assistant", "content": "..."}
    """
    messages = []
    
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    
    for msg in history:
        if msg["role"] == "human":
            messages.append({"role": "user", "content": msg["content"]})
        elif msg["role"] == "ai":
            messages.append({"role": "assistant", "content": msg["content"]})
    
    return messages
