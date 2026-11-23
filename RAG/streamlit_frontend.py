import json
import time
import requests
import streamlit as st

# Configure this to your FastAPI server base URL
API_BASE = "http://192.168.190.28:9096"

# Helpers for API calls
def api_post(path, json_body):
    url = f"{API_BASE}{path}"
    try:
        r = requests.post(url, json=json_body, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}

def api_post_stream(path, json_body):
    """
    Post and return a generator that yields parsed SSE 'data' payloads.
    The server sends lines like 'data: {"token":"..."}' and 'data: [DONE]'.
    """
    url = f"{API_BASE}{path}"
    headers = {"Accept": "text/event-stream", "Cache-Control": "no-cache"}
    try:
        with requests.post(url, json=json_body, stream=True, headers=headers, timeout=None) as resp:
            resp.raise_for_status()
            for raw_line in resp.iter_lines(decode_unicode=True):
                if raw_line is None:
                    continue
                line = raw_line.strip()
                if not line:
                    continue
                # SSE data lines start with "data:"
                if line.startswith("data:"):
                    payload = line[len("data:"):].strip()
                    if payload == "[DONE]":
                        yield {"done": True}
                        break
                    # sometimes server may send non-json or strings
                    try:
                        data = json.loads(payload)
                    except Exception:
                        data = {"raw": payload}
                    yield {"data": data}
            # end
    except requests.exceptions.RequestException as e:
        yield {"error": str(e)}
    except Exception as e:
        yield {"error": str(e)}

# Session state defaults
if "user_email" not in st.session_state:
    st.session_state.user_email = ""
if "user_id" not in st.session_state:
    st.session_state.user_id = None
if "threads" not in st.session_state:
    st.session_state.threads = []
if "selected_thread" not in st.session_state:
    st.session_state.selected_thread = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of dicts {"role":"human|ai|system","content": "..."}
if "last_ai_stream" not in st.session_state:
    st.session_state.last_ai_stream = ""

st.title("Simple Streamlit Chat Client (for testing)")

st.markdown("Small testing frontend to create a user/thread and run streaming chat against the FastAPI server.")

# Left column: user + threads
col1, col2 = st.columns([1, 2])

with col1:
    st.header("User & Threads")
    email = st.text_input("Your email", value=st.session_state.user_email)
    if st.button("Set email / Create user"):
        st.session_state.user_email = email.strip()
        if not st.session_state.user_email:
            st.error("Provide a valid email.")
        else:
            res = api_post("/user/create", {"email": st.session_state.user_email})
            if res is None or "error" in res:
                st.error(f"User create/get failed: {res}")
            else:
                st.session_state.user_id = res.get("user_id")
                st.success(f"Got user_id: {st.session_state.user_id}")
                # refresh threads list
                threads_res = None
                try:
                    r = requests.get(f"{API_BASE}/chat/threads", params={"email": st.session_state.user_email}, timeout=10)
                    r.raise_for_status()
                    threads_res = r.json()
                except Exception as e:
                    st.warning(f"Could not list threads: {e}")
                if threads_res:
                    st.session_state.threads = threads_res

    st.markdown("Create a new thread (chat)")
    new_title = st.text_input("Thread title", value="New Chat")
    if st.button("Create thread"):
        if not st.session_state.user_email:
            st.error("Set an email first.")
        else:
            payload = {"email": st.session_state.user_email, "title": new_title or "New Chat", "metadata": {}}
            res = api_post("/chat/create_thread", payload)
            if res and "thread_id" in res:
                st.success(f"Created thread: {res['thread_id']}")
                # refresh
                try:
                    r = requests.get(f"{API_BASE}/chat/threads", params={"email": st.session_state.user_email}, timeout=10)
                    r.raise_for_status()
                    st.session_state.threads = r.json()
                except Exception as e:
                    st.warning(f"Could not refresh threads: {e}")
            else:
                st.error(f"Thread creation failed: {res}")

    st.markdown("Select thread to chat")
    thread_options = {t.get("title", t.get("thread_id")) + " — " + t.get("thread_id"): t.get("thread_id") for t in st.session_state.threads}
    if thread_options:
        sel = st.selectbox("Threads", options=list(thread_options.keys()))
        st.session_state.selected_thread = thread_options.get(sel)
        st.write(f"Selected thread_id: `{st.session_state.selected_thread}`")
    else:
        st.info("No threads found. Create one to begin.")

with col2:
    st.header("Conversation")
    # show chat history
    chat_container = st.container()
    with chat_container:
        if not st.session_state.selected_thread:
            st.info("Select a thread to start chatting.")
        else:
            st.subheader(f"Thread: {st.session_state.selected_thread}")
            # retrieve existing history from server
            if st.button("Load history"):
                try:
                    r = requests.get(f"{API_BASE}/chat/history", params={"thread_id": st.session_state.selected_thread}, timeout=10)
                    r.raise_for_status()
                    st.session_state.chat_history = r.json()
                except Exception as e:
                    st.warning(f"Failed to load history: {e}")

            # Display messages
            if st.session_state.chat_history:
                for m in st.session_state.chat_history:
                    role = m.get("role", "unknown")
                    ts = m.get("created_at") or m.get("timestamp") or ""
                    content = m.get("content") or m.get("message") or ""
                    if role == "human":
                        st.markdown(f"**You:** {content}")
                    elif role == "ai":
                        st.markdown(f"**Assistant:** {content}")
                    else:
                        st.markdown(f"**{role.title()}:** {content}")

            # Input area for new message
            user_message = st.text_area("Your message", key="user_message", height=100)
            col_send1, col_send2 = st.columns([1, 4])
            with col_send1:
                if st.button("Send & Stream"):
                    if not st.session_state.selected_thread:
                        st.error("Select a thread first.")
                    elif not user_message.strip():
                        st.error("Type a message.")
                    else:
                        # Append human message locally and to server
                        st.session_state.chat_history.append({"role": "human", "content": user_message})
                        # Call streaming endpoint
                        payload = {
                            "thread_id": st.session_state.selected_thread,
                            "message": user_message,
                            # you can add other fields like model, temperature if desired
                        }
                        # push the human message to server DB (endpoint already stores on server side inside /chat/send)
                        # Start streaming from server and append tokens
                        ai_placeholder = st.empty()
                        ai_text = ""
                        status = st.empty()
                        status.info("Streaming...")
                        for event in api_post_stream("/chat/send", payload):
                            if event.get("error"):
                                status.error(f"Stream error: {event['error']}")
                                st.session_state.chat_history.append({"role": "ai", "content": f"Error: {event['error']}"})
                                break
                            if event.get("done"):
                                status.success("Stream complete.")
                                # store final assistant message in history
                                st.session_state.chat_history.append({"role": "ai", "content": ai_text})
                                break
                            data = event.get("data")
                            if not data:
                                continue
                            # Some servers send {"token": "..."} or {"error": "..."}
                            if isinstance(data, dict):
                                if "token" in data:
                                    ai_text += data["token"]
                                    ai_placeholder.markdown(f"**Assistant:** {ai_text}")
                                elif "error" in data:
                                    status.error(f"LLM error: {data['error']}")
                                    st.session_state.chat_history.append({"role": "ai", "content": f"Error: {data['error']}"})
                                    break
                                elif "raw" in data:
                                    # fallback raw data
                                    ai_text += str(data["raw"])
                                    ai_placeholder.markdown(f"**Assistant:** {ai_text}")
                            else:
                                ai_text += str(data)
                                ai_placeholder.markdown(f"**Assistant:** {ai_text}")
                        # clear input
                        st.session_state.user_message = ""
                        # reload server-side history to be consistent (optional)
                        try:
                            r = requests.get(f"{API_BASE}/chat/history", params={"thread_id": st.session_state.selected_thread}, timeout=10)
                            r.raise_for_status()
                            st.session_state.chat_history = r.json()
                        except Exception:
                            pass

            with col_send2:
                st.write("")  # spacing

st.markdown("---")
st.markdown("Run notes:")
st.markdown("- Make sure the FastAPI server is running and `API_BASE` at top points to it.")
st.markdown("- This simple frontend does not implement authentication.")

st.caption("Streamlit app for quick testing of the FastAPI SSE streaming chat.")