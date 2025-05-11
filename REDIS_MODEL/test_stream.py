import requests
import json

# URL of your FastAPI endpoint for streaming
url = "http://127.17.25.83:8082/v1/chat/completions/stream"

# Your request data
data = {
    "model": "./meta-llama/Llama-3.1-8B-Instruct-awq",
    "messages": [{"role": "user", "content": "tell me about life "}],
    "stream": True
}

headers = {
    "Content-Type": "application/json"
}

# Send the POST request with stream=True to enable streaming
response = requests.post(url, json=data, headers=headers, stream=True)

# Check if the response is valid
if response.status_code == 200:
    print("Streaming response started:")
    
    full_response = ""  # To store the complete response
    # Iterate over the streamed content
    for chunk in response.iter_lines(decode_unicode=True):
        if chunk:
            try:
                # Parse the chunk as JSON to extract the content
                json_chunk = chunk.strip().lstrip("data: ")
                data = json.loads(json_chunk)
                content = data['choices'][0]['delta'].get('content', '')
                full_response += content  # Append the content from this chunk
                print(content, end='')  # Optionally print each chunk's content
            except Exception as e:
                print(f"Error parsing chunk: {e}")
    
    print("\n\nComplete response:", full_response)

else:
    print(f"Failed to connect: {response.status_code}")
