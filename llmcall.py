import requests
import json

url = "http://localhost:8081/v1/chat/completions"
responses = []

headers = {
    "Content-Type": "application/json"
}

data = {
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"}
    ],
    "stream": True
}

response = requests.post(url, headers=headers, data=json.dumps(data), stream=True)

if response.status_code == 200:
    for chunk in response.iter_lines():
        if chunk:
            # Remove "data: " prefix if present
            decoded_line = chunk.decode('utf-8')
            if decoded_line.startswith("data: "):
                decoded_line = decoded_line[len("data: "):]
            
            try:
                chunk_data = json.loads(decoded_line)
                delta = chunk_data['choices'][0].get('delta', {})
                assistant_response = delta.get('content', '')
                if assistant_response:
                    responses.append(assistant_response)
                    print(assistant_response, end='', flush=True)
            except json.JSONDecodeError:
                print("Error decoding JSON chunk:", decoded_line)
else:
    print(f"Request failed with status code {response.status_code}")

# Print final response
final_response = "".join(responses)
print("\n\nFull Response:", final_response)
