import asyncio
import os
from typing import Optional
import httpx
import json

API_BASE = "http://localhost:9000"

async def upload_document(file_path: str) -> str:
    """Uploads a document to the FastAPI server and returns its ID."""
    async with httpx.AsyncClient() as client:
        with open(file_path, "rb") as f:
            files = {"file": (file_path, f, "application/octet-stream")}
            response = await client.post(f"{API_BASE}/upload-document/", files=files)
            response.raise_for_status()
            data = response.json()
            print("✅ Document Uploaded:", data)
            return data["document_id"]

async def stream_chat_response(document_id: str, question: str, session_id: str = None) -> str:
    """
    Streams a chat response from the FastAPI server.
    Returns the full concatenated response and the session_id (if received).
    """
    payload = {
        "question": question,
        "document_id": document_id,
        "stream": True,
        "session_id": session_id # Pass session_id if available
    }
    
    async with httpx.AsyncClient(timeout=None) as client:
        print(f"\n🧠 You: {question}")
        print("💬 Assistant:", end="", flush=True) # Prepare for streaming output
        full_response = ""
        try:
            async with client.stream("POST", f"{API_BASE}/chat/", json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    processed_line = line.strip()
                    if processed_line.startswith("data:"):
                        content_str = processed_line.removeprefix("data:").strip()
                        if content_str == "[DONE]":
                            print("\n✅ Streaming Complete")
                            break
                        try:
                            token_data = json.loads(content_str)
                            if "token" in token_data:
                                print(token_data["token"], end="", flush=True)
                                full_response += token_data["token"]
                        except json.JSONDecodeError:
                            print(f"\n[CLIENT ERROR] Could not decode JSON: {content_str}", end="", flush=True)
                return full_response
        except httpx.HTTPStatusError as e:
            print(f"\n❌ HTTP Error during streaming: {e.response.status_code} - {e.response.text}")
            return ""
        except httpx.RequestError as e:
            print(f"\n❌ Network Error during streaming: {e}")
            return ""

async def get_session_id_from_chat_response(question: str, document_id: Optional[str] = None) -> Optional[str]:
    """
    Makes a non-streaming chat request to get an initial session ID.
    This is a workaround for the streaming endpoint not returning session_id mid-stream.
    Ideally, your streaming endpoint would send a session_id chunk.
    """
    payload = {
        "question": question,
        "document_id": document_id,
        "stream": False # Request non-streaming to get the full JSON response
    }
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(f"{API_BASE}/chat/", json=payload)
            response.raise_for_status()
            data = response.json()
            return data.get("session_id")
        except httpx.HTTPStatusError as e:
            print(f"\n❌ Failed to get initial session ID: {e.response.status_code} - {e.response.text}")
            return None
        except httpx.RequestError as e:
            print(f"\n❌ Network error getting initial session ID: {e}")
            return None

async def list_documents():
    """Lists all documents currently uploaded to the server."""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{API_BASE}/documents/")
        response.raise_for_status()
        data = response.json()
        print("\n📄 Uploaded Documents:")
        if data:
            for doc in data:
                print(f"- ID: {doc['document_id']} | Filename: {doc['filename']} | Pages: {doc['pages']}")
        else:
            print("No documents uploaded.")

async def delete_document(document_id: str):
    """Deletes a document from the server by its ID."""
    async with httpx.AsyncClient() as client:
        response = await client.delete(f"{API_BASE}/documents/{document_id}")
        response.raise_for_status()
        print("🗑️ Document Deleted:", response.json())

async def get_chat_history(session_id: str):
    """Fetches and displays chat history for a given session ID."""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{API_BASE}/chat-history/{session_id}")
            response.raise_for_status()
            data = response.json()
            print(f"\n📜 Chat History for Session ID: {data['session_id']}")
            if data['history']:
                for msg in data['history']:
                    print(f"  {msg['type']}: {msg['content']}")
            else:
                print("  No history found for this session.")
            return data['history']
        except httpx.HTTPStatusError as e:
            print(f"\n❌ Error fetching chat history: {e.response.status_code} - {e.response.text}")
            return None
        except httpx.RequestError as e:
            print(f"\n❌ Network error fetching chat history: {e}")
            return None

async def clear_chat_history(session_id: str):
    """Clears chat history for a given session ID."""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.delete(f"{API_BASE}/chat-history/{session_id}")
            response.raise_for_status()
            print("🗑️ Chat history cleared:", response.json())
        except httpx.HTTPStatusError as e:
            print(f"\n❌ Error clearing chat history: {e.response.status_code} - {e.response.text}")
        except httpx.RequestError as e:
            print(f"\n❌ Network error clearing chat history: {e}")


async def main_continuous_conversation():
    """Main function for a continuous chat conversation in the terminal."""
    file_path = input("Enter path to PDF/DOCX document (or leave empty to skip document upload): ").strip()
    document_id = None
    if file_path:
        if not os.path.exists(file_path):
            print(f"❌ Error: Document not found at '{file_path}'. Please check the path.")
            return
        try:
            document_id = await upload_document(file_path)
            print(f"Document uploaded with ID: {document_id}")
        except Exception as e:
            print(f"❌ Failed to upload document: {e}")
            return

    session_id = None
    print("\n--- Start your continuous conversation ---")
    print("Type 'exit' to end the chat.")
    print("Type 'history' to view current session chat history.")
    print("Type 'clear' to clear current session chat history.")

    while True:
        question = input("\n> ").strip()
        if question.lower() == 'exit':
            print("Ending conversation. Goodbye!")
            break
        elif question.lower() == 'history':
            if session_id:
                await get_chat_history(session_id)
            else:
                print("No active session yet to view history.")
            continue
        elif question.lower() == 'clear':
            if session_id:
                await clear_chat_history(session_id)
                session_id = None # Reset session_id after clearing
                print("Chat history cleared. Starting a new conversation context.")
            else:
                print("No active session to clear history for.")
            continue
        elif not question:
            continue # Don't send empty questions

        # Get session_id on the first meaningful question if not already established
        if session_id is None:
            # We'll make a non-streaming request first to get the session_id
            # This is a workaround as the current streaming endpoint doesn't return session_id mid-stream.
            print("Establishing new session...")
            temp_session_id = await get_session_id_from_chat_response(question, document_id)
            if temp_session_id:
                session_id = temp_session_id
                print(f"🔗 New Session ID: {session_id}")
            else:
                print("⚠️ Failed to establish session. Conversation might not maintain context.")

        if session_id: # Only proceed if a session_id was successfully established or passed
            await stream_chat_response(document_id, question, session_id)
        else:
            print("Cannot proceed without a session ID. Please try again.")

    # Clean up
    if document_id:
        await delete_document(document_id)
        print("Document deleted.")

if __name__ == "__main__":
    asyncio.run(main_continuous_conversation())