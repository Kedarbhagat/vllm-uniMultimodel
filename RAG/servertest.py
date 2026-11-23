import requests
from sseclient import SSEClient

BASE_URL = "http://127.0.0.1:9095"

# --- 1. Create or Get User ---
email = "venkat@example.com"
resp = requests.post(f"{BASE_URL}/user/create", json={"email": email})
user_id = resp.json()["user_id"]
print(f"✅ User ID: {user_id}")

# --- 2. Create Thread ---
resp = requests.post(f"{BASE_URL}/chat/create_thread", json={"email": email, "title": "Test Chat"})
thread_id = resp.json()["thread_id"]
print(f"🧵 Thread ID: {thread_id}")

# --- 3. Send a Message ---
question = "Write a story about a robot learning to dance."
print(f"\n💬 {question}\n🤖 AI says:")

resp = requests.post(
    f"{BASE_URL}/chat/send",
    json={"thread_id": thread_id, "message": question},
    stream=True,
    headers={"Accept": "text/event-stream"}
)

client = SSEClient(resp)

for event in client.events():
    data = event.data.strip()
    if data == "[DONE]":
        break
    print(data, end="", flush=True)  # Just print the raw streamed content

print("\n✅ Done.")
