#!/usr/bin/env python3
"""
Minimal CLI chat client for the FastAPI RAG server.

Defaults to server at http://127.0.0.1:9096 and endpoint /chat/send.

Usage examples (PowerShell):
  pip install requests
  python .\chat_client_9096.py --server http://127.0.0.1:9096 --email you@example.com
  python .\chat_client_9096.py --server http://127.0.0.1:9096 --thread <thread_id>

Optional:
  --auth "Bearer token-abc123"   # forwards Authorization header to the server
"""
import argparse
import json
import requests
import uuid

def create_thread(server_url: str, email: str, title: str = "Simple Chat"):
    url = server_url.rstrip("/") + "/chat/create_thread"
    payload = {"email": email, "title": title}
    try:
        r = requests.post(url, json=payload, timeout=10)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, dict) and "thread_id" in data:
            return data["thread_id"]
        return data
    except Exception as e:
        print(f"[warn] create_thread failed: {e}")
        return None

def stream_chat(server_url: str, thread_id: str, message: str, model: str = None, auth: str | None = None):
    url = server_url.rstrip("/") + "/chat/send"
    payload = {"thread_id": thread_id, "message": message}
    if model:
        payload["model"] = model
    headers = {}
    if auth:
        headers["Authorization"] = auth

    try:
        with requests.post(url, json=payload, stream=True, headers=headers, timeout=(5, None)) as resp:
            if resp.status_code != 200:
                # Print helpful debug info
                body = resp.text
                print(f"\n[error] Server returned {resp.status_code}: {body}")
                return

            print("\n> ", end="", flush=True)
            for raw_line in resp.iter_lines(decode_unicode=True):
                if raw_line is None:
                    continue
                line = raw_line.strip()
                if not line:
                    continue
                if line.startswith("data:"):
                    payload_text = line[len("data:"):].strip()
                    if payload_text == "[DONE]":
                        break
                    try:
                        obj = json.loads(payload_text)
                        if isinstance(obj, dict):
                            if "token" in obj:
                                print(obj["token"], end="", flush=True)
                            elif "error" in obj:
                                print(f"\n[error] {obj['error']}")
                                break
                            else:
                                print(json.dumps(obj, ensure_ascii=False), end=" ", flush=True)
                        else:
                            print(str(obj), end="", flush=True)
                    except json.JSONDecodeError:
                        print(payload_text, end="", flush=True)
            print("\n")
    except KeyboardInterrupt:
        print("\n[info] Interrupted by user.")
    except Exception as e:
        print(f"\n[error] Streaming failed: {e}")

def main():
    parser = argparse.ArgumentParser(description="Simple chat client for RAG server (port 9096)")
    parser.add_argument("--server", "-s", default="http://127.0.0.1:9096", help="Base URL of server (default http://127.0.0.1:9096)")
    parser.add_argument("--email", "-e", help="Email to create a thread (optional)")
    parser.add_argument("--thread", "-t", help="Existing thread_id to use (optional)")
    parser.add_argument("--model", "-m", help="Model name to request (optional)")
    parser.add_argument("--auth", help="Authorization header value (e.g. 'Bearer token-abc123') (optional)")
    args = parser.parse_args()

    server_url = args.server
    thread_id = args.thread

    if not thread_id and args.email:
        print(f"[info] Creating thread for {args.email} ...")
        thread_id = create_thread(server_url, args.email)
        if thread_id:
            print(f"[info] Created thread_id: {thread_id}")
        else:
            print("[warn] Couldn't create thread; you'll be asked for a thread_id or one will be auto-generated.")

    if not thread_id:
        tid = input("Thread id (leave empty to auto-generate): ").strip()
        if tid:
            thread_id = tid
        else:
            thread_id = str(uuid.uuid4())
            print(f"[info] Using generated thread_id: {thread_id}")

    print(f"[info] Chatting on thread: {thread_id}")
    print("[info] Type messages and press Enter. Ctrl+C to quit.\n")

    try:
        while True:
            try:
                msg = input("You: ").strip()
            except EOFError:
                break
            if not msg:
                continue
            stream_chat(server_url, thread_id, msg, model=args.model, auth=args.auth)
    except KeyboardInterrupt:
        print("\n[info] Exiting...")

if __name__ == "__main__":
    main()