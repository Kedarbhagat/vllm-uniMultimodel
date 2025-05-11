from typing import Any, Dict, List, Optional, Mapping, Union, Iterator
from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
import requests
import json

class CustomFastAPILLM(LLM):
    """Custom LLM class that interfaces with a FastAPI endpoint."""
    
    api_url: str = "http://172.17.25.83:8080/v1/chat/completions"
    model: str = "microsoft/Phi-4-mini-instruct"
    temperature: float = 0.7
    
    @property
    def _llm_type(self) -> str:
        """Return type of LLM."""
        return "custom_fastapi_llm"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call the FastAPI endpoint."""
        messages = [{"role": "user", "content": prompt}]
        
        # Merge instance parameters with call parameters
        params = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        params.update(kwargs)
        
        # Send request to FastAPI endpoint
        headers = {"Content-Type": "application/json"}
        response = requests.post(
            self.api_url,
            headers=headers,
            json=params,
            timeout=60,
            stream=True
        )
        
        if response.status_code != 200:
            raise ValueError(f"Error from API: {response.text}")
        
        result = response.json()
        
        # Extract content from the response
        # The structure matches OpenAI-compatible completion formats
        if "choices" in result and len(result["choices"]) > 0:
            if "message" in result["choices"][0]:
                return result["choices"][0]["message"]["content"]
            elif "text" in result["choices"][0]:
                return result["choices"][0]["text"]
        
        raise ValueError(f"Unexpected response structure: {result}")
    
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            "api_url": self.api_url,
            "model": self.model,
            "temperature": self.temperature,
        }


# Example usage - this is the only part we need for now
if __name__ == "__main__":
    llm = CustomFastAPILLM(
        api_url="http://172.17.25.83:8080/v1/chat/completions",
        model="microsoft/Phi-4-mini-instruct",
    )
    
    result = llm.invoke("what do think my name would be, my name is kedar what is its meaning")
    print(result)