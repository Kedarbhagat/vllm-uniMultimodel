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

async def stream_chat_response(
    question: str,
    document_id: Optional[str] = None,
    session_id: Optional[str] = None,
    enable_web_search: bool = False
) -> tuple[str, Optional[str]]:
    """
    Streams a chat response from the FastAPI server.
    Returns the full concatenated response and the session_id (if received).
    """
    payload = {
        "question": question,
        "document_id": document_id,
        "session_id": session_id,
        "stream": True,
        "enable_web_search": enable_web_search
    }
    
    async with httpx.AsyncClient(timeout=None) as client:
        print(f"\n🧠 You: {question}")
        print("💬 Assistant:", end="", flush=True)
        full_response = ""
        new_session_id = session_id
        
        try:
            async with client.stream("POST", f"{API_BASE}/chat/", json=payload) as response:
                response.raise_for_status()
                
                async for line in response.aiter_lines():
                    if line.strip() == "":
                        continue
                        
                    if line.startswith("data:"):
                        content_str = line[5:].strip()
                        
                        # Handle different response types
                        if content_str == "[DONE]":
                            break
                            
                        try:
                            data = json.loads(content_str)
                            
                            # Handle text responses
                            if data.get("type") == "text":
                                token = data.get("content", "")
                                print(token, end="", flush=True)
                                full_response += token
                                
                            # Handle tool responses
                            elif data.get("type") == "tool_output" and data.get("tool") == "mermaid":
                                diagram = data.get("content", "")
                                print("\n🔷 Mermaid Diagram:\n", diagram, "\n")
                                full_response += f"\nDiagram: {diagram}\n"
                                
                            # Handle session ID if provided
                            if "session_id" in data:
                                new_session_id = data["session_id"]
                                
                        except json.JSONDecodeError:
                            print(f"\n[CLIENT ERROR] Could not decode JSON: {content_str}")
                            
                return full_response, new_session_id
                
        except httpx.HTTPStatusError as e:
            print(f"\n❌ HTTP Error during streaming: {e.response.status_code} - {e.response.text}")
            return "", session_id
        except httpx.RequestError as e:
            print(f"\n❌ Network Error during streaming: {e}")
            return "", session_id

async def list_documents():
    """Lists all documents currently uploaded to the server."""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{API_BASE}/documents/")
        response.raise_for_status()
        return response.json()

async def delete_document(document_id: str):
    """Deletes a document from the server by its ID."""
    async with httpx.AsyncClient() as client:
        response = await client.delete(f"{API_BASE}/documents/{document_id}")
        response.raise_for_status()
        return response.json()

async def get_chat_history(session_id: str):
    """Fetches chat history for a given session ID."""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{API_BASE}/chat-history/{session_id}")
        response.raise_for_status()
        return response.json()

async def clear_chat_history(session_id: str):
    """Clears chat history for a given session ID."""
    async with httpx.AsyncClient() as client:
        response = await client.delete(f"{API_BASE}/chat-history/{session_id}")
        response.raise_for_status()
        return response.json()

async def display_chat_history(history):
    """Displays formatted chat history."""
    print(f"\n📜 Chat History (Session: {history['session_id']})")
    for msg in history['history']:
        prefix = "👤 You:" if msg['type'] == "human" else "🤖 AI:"
        if msg.get('is_diagram'):
            print(f"{prefix}\n```mermaid\n{msg['content']}\n```")
        else:
            print(f"{prefix} {msg['content']}")

async def main_continuous_conversation():
    """Main function for a continuous chat conversation with multiple modes."""
    print("\n" + "="*50)
    print("Enhanced RAG Chat System".center(50))
    print("="*50)
    print("\nChoose conversation mode:")
    print("1. Document-based RAG")
    print("2. Web Search assisted")
    print("3. Regular chat")
    
    mode = input("\nSelect mode (1-3): ").strip()
    document_id = None
    enable_web_search = False
    
    if mode == "1":
        file_path = input("Enter path to PDF/DOCX document: ").strip()
        if not os.path.exists(file_path):
            print(f"❌ Error: Document not found at '{file_path}'")
            return
        try:
            document_id = await upload_document(file_path)
            print(f"✅ Document uploaded with ID: {document_id}")
        except Exception as e:
            print(f"❌ Failed to upload document: {e}")
            return
    elif mode == "2":
        enable_web_search = True
        print("🌐 Web Search mode enabled")
    elif mode == "3":
        print("💬 Regular chat mode")
    else:
        print("❌ Invalid mode selection")
        return

    session_id = None
    print("\n" + "-"*50)
    print("Chat Commands:")
    print("• 'exit' - End conversation")
    print("• 'history' - View chat history")
    print("• 'clear' - Clear current session")
    print("• 'documents' - List uploaded documents")
    print("• 'delete' - Delete current document")
    print("-"*50 + "\n")

    while True:
        question = input("\nYou: ").strip()
        
        # Handle commands
        if question.lower() == 'exit':
            print("Ending conversation. Goodbye!")
            break
        elif question.lower() == 'history':
            if session_id:
                history = await get_chat_history(session_id)
                await display_chat_history(history)
            else:
                print("No active session to view history")
            continue
        elif question.lower() == 'clear':
            if session_id:
                await clear_chat_history(session_id)
                session_id = None
                print("Session cleared. Starting new conversation.")
            else:
                print("No active session to clear")
            continue
        elif question.lower() == 'documents':
            docs = await list_documents()
            print("\n📄 Uploaded Documents:")
            for doc in docs:
                print(f"- ID: {doc['document_id']} | {doc['filename']} ({doc['pages']} pages)")
            continue
        elif question.lower() == 'delete' and document_id:
            await delete_document(document_id)
            document_id = None
            print("Document deleted. Continuing without document context.")
            continue
        elif not question:
            continue
            
        # Process the question
        response, new_session_id = await stream_chat_response(
            question=question,
            document_id=document_id,
            session_id=session_id,
            enable_web_search=enable_web_search
        )
        
        # Update session ID if we got a new one
        if new_session_id and new_session_id != session_id:
            session_id = new_session_id
            print(f"\n🔗 Session ID: {session_id}")

    # Clean up
    if document_id:
        await delete_document(document_id)
        print("Document deleted.")

if __name__ == "__main__":
    asyncio.run(main_continuous_conversation())