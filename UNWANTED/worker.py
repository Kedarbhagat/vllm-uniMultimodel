# celery_worker.py
from UNWANTED.worker import Celery
import requests

# Connect to Redis (this assumes Redis is running on the default port 6379)
celery_app = Celery(
    "fastapi_gateway",  # Celery app name
    broker="redis://localhost:6379/0",  # Redis broker URL
    backend="redis://localhost:6379/0",  # Result backend
)

# Optional: Configure Celery settings
celery_app.conf.update(
    task_serializer="json",  # Use JSON for task serialization
    result_serializer="json",
    accept_content=["json"],  # Allow only JSON
    result_expires=3600,  # Results expire in 1 hour
)


url = "http://localhost:8000/v1/chat/completions"  # FastAPI gateway
model = "microsoft/Phi-4-mini-instruct"  # Model name, which is mapped in FastAPI
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Can you help me with my project?"}
]
stream = False  # Set to True if you want streaming

payload = {
    "model": model,
    "messages": messages,
    "stream": stream
}

response = requests.post(url, json=payload)

# Handle response
if response.status_code == 200:
    print("Response:", response.json())  # Print the response from the model
else:
    print(f"Error: {response.status_code}, {response.text}")
