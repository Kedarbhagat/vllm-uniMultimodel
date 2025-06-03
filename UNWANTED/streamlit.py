import UNWANTED.streamlit as st
import requests
import json

st.set_page_config(page_title="Chat with Phi-4", page_icon="💬")
st.title("💬 Chat with Phi-4-mini-instruct")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Function to stream response from vLLM
def get_streaming_response(user_input):
    url = "http://localhost:8080/v1/chat/completions"  # Use your vLLM IP:port if running remotely
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": "microsoft/Phi-4-mini-instruct",  # Must match what you used in `vllm serve`
        "messages": st.session_state.messages + [{"role": "user", "content": user_input}],
        "stream": True
    }

    response = requests.post(url, headers=headers, data=json.dumps(payload), stream=True)
    full_reply = ""
    for line in response.iter_lines():
        if line:
            if line.startswith(b"data: "):
                line = line[len(b"data: "):]
            if line.strip() == b"[DONE]":
                break
            try:
                data = json.loads(line)
                delta = data.get("choices", [{}])[0].get("delta", {})
                content = delta.get("content", "")
                full_reply += content
                yield content
            except Exception:
                continue
    st.session_state.messages.append({"role": "assistant", "content": full_reply})

# User input
user_input = st.chat_input("Type your message...")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        response = st.empty()
        stream_output = ""
        for chunk in get_streaming_response(user_input):
            stream_output += chunk
            response.markdown(stream_output)

# Show previous messages
for msg in st.session_state.messages[:-2]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
