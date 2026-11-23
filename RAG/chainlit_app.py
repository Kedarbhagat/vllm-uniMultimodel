import json
import chainlit as cl
import httpx
import uuid
from typing import Optional
import os

# Configuration
API_BASE_URL = "http://localhost:9095"  # Update with your FastAPI server URL
DEFAULT_MODEL = "llama"
AVAILABLE_MODELS = ["llama", "deepseek"]

os.environ["CHAINLIT_AUTH_SECRET"] = os.getenv("CHAINLIT_AUTH_SECRET", r"Oo8g%az/eje3WD$/-sqV~*Sfhm>?,%n?o:.=qyEi?:3-n,BtjV?jt8,FzwK0.dAe")
# Dummy user system (replace with real auth later)
users_db = {
    "test@example.com": {"name": "Test User"},
    "demo@example.com": {"name": "Demo User"}
}

async def get_or_create_user(email: str):
    """Get or create a user with dummy auth"""
    if email not in users_db:
        users_db[email] = {"name": email.split("@")[0]}
    return {"email": email, **users_db[email]}

async def api_request(method: str, endpoint: str, data: dict = None):
    """Helper for API requests"""
    async with httpx.AsyncClient() as client:
        try:
            if method == "GET":
                response = await client.get(f"{API_BASE_URL}{endpoint}")
            else:
                response = await client.post(f"{API_BASE_URL}{endpoint}", json=data)
            
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            print(f"API Error: {e.response.text}")
            raise Exception(f"API request failed: {e.response.text}")

@cl.on_chat_start
async def start_chat():
    # Simple login - in a real app, use proper auth
    user_email = await cl.AskUserMessage(
        content="Welcome to the Chat Agent! Please enter your email to continue:",
        timeout=30
    ).send()
    
    user = await get_or_create_user(user_email["content"])
    cl.user_session.set("user", user)
    
    # Get or create threads for this user
    threads = await api_request("GET", f"/chat/threads?email={user['email']}")
    if not threads:
        # Create first thread
        new_thread = await api_request("POST", "/chat/create_thread", {
            "email": user["email"],
            "title": "First Chat"
        })
        threads = [new_thread]
    
    cl.user_session.set("threads", threads)
    cl.user_session.set("current_thread_id", threads[0]["thread_id"])
    
    # Show thread selection
    thread_options = [{"label": f"Chat {idx+1}", "value": t["thread_id"]} 
                     for idx, t in enumerate(threads)]
    
    thread_res = await cl.AskActionMessage(
        content="Select a chat thread or create new:",
        actions=[
            *[cl.Action(name=opt["value"], value=opt["value"], label=opt["label"]) 
              for opt in thread_options],
            cl.Action(name="new_thread", value="new", label="+ New Chat")
        ]
    ).send()
    
    if thread_res.get("value") == "new":
        # Create new thread
        title_res = await cl.AskUserMessage(
            content="Enter a title for the new chat:",
            timeout=30
        ).send()
        
        new_thread = await api_request("POST", "/chat/create_thread", {
            "email": user["email"],
            "title": title_res["content"]
        })
        cl.user_session.set("current_thread_id", new_thread["thread_id"])
        threads.append(new_thread)
        cl.user_session.set("threads", threads)
    else:
        cl.user_session.set("current_thread_id", thread_res["value"])
    
    # Load history for selected thread
    thread_id = cl.user_session.get("current_thread_id")
    history = await api_request("GET", f"/chat/history?thread_id={thread_id}")
    
    # Send welcome message
    welcome_msg = f"""Welcome {user['name']}! 
You're chatting in thread: {thread_id}
Select a model and start chatting."""
    
    await cl.Message(
        content=welcome_msg,
        disable_feedback=True
    ).send()
    
    # Send history
    for msg in history:
        await cl.Message(
            content=msg["content"],
            author="User" if msg["role"] == "human" else "AI",
            disable_feedback=True
        ).send()
    
    # Model selection
    model_res = await cl.AskActionMessage(
        content="Select a model:",
        actions=[
            cl.Action(name=m, value=m, label=m.capitalize()) 
            for m in AVAILABLE_MODELS
        ]
    ).send()
    
    cl.user_session.set("model", model_res["value"])
    await cl.Message(content=f"Using {model_res['value']} model").send()


@cl.on_message
@cl.on_chat_start
async def start_chat():
    # Simple login - with proper error handling
    user_email = await cl.AskUserMessage(
        content="Welcome to the Chat Agent! Please enter your email to continue:",
        timeout=30
    ).send()

    # Handle case where user didn't provide email or closed prompt
    if not user_email or not isinstance(user_email, dict) or "content" not in user_email:
        await cl.Message(
            content="Email is required to continue. Please refresh the page to try again.",
            disable_feedback=True
        ).send()
        return

    try:
        user = await get_or_create_user(user_email["content"])
        cl.user_session.set("user", user)
        
        # Get or create threads for this user
        threads = await api_request("GET", f"/chat/threads?email={user['email']}")
        if not threads:
            # Create first thread
            new_thread = await api_request("POST", "/chat/create_thread", {
                "email": user["email"],
                "title": "First Chat"
            })
            threads = [new_thread]
        
        cl.user_session.set("threads", threads)
        
        # Default to first thread if available
        current_thread_id = threads[0]["thread_id"] if threads else None
        if not current_thread_id:
            await cl.Message(
                content="Failed to initialize chat thread. Please try again.",
                disable_feedback=True
            ).send()
            return
            
        cl.user_session.set("current_thread_id", current_thread_id)
        
        # Show thread selection UI only if we have multiple threads
        if len(threads) > 1:
            thread_options = [{"label": f"Chat {idx+1}", "value": t["thread_id"]} 
                           for idx, t in enumerate(threads)]
            
            thread_res = await cl.AskActionMessage(
                content="Select a chat thread or create new:",
                actions=[
                    *[cl.Action(name=opt["value"], value=opt["value"], label=opt["label"]) 
                      for opt in thread_options],
                    cl.Action(name="new_thread", value="new", label="+ New Chat")
                ]
            ).send()
            
            if thread_res.get("value") == "new":
                # Create new thread
                title_res = await cl.AskUserMessage(
                    content="Enter a title for the new chat:",
                    timeout=30
                ).send()
                
                if not title_res or not isinstance(title_res, dict) or "content" not in title_res:
                    await cl.Message(
                        content="Thread title is required",
                        disable_feedback=True
                    ).send()
                    return
                
                new_thread = await api_request("POST", "/chat/create_thread", {
                    "email": user["email"],
                    "title": title_res["content"]
                })
                cl.user_session.set("current_thread_id", new_thread["thread_id"])
                threads.append(new_thread)
                cl.user_session.set("threads", threads)
            else:
                cl.user_session.set("current_thread_id", thread_res["value"])
        
        # Load history for selected thread
        thread_id = cl.user_session.get("current_thread_id")
        history = await api_request("GET", f"/chat/history?thread_id={thread_id}")
        
        # Send welcome message
        welcome_msg = f"""Welcome {user['name']}! 
You're chatting in thread: {thread_id}
Select a model and start chatting."""
        
        await cl.Message(
            content=welcome_msg,
            disable_feedback=True
        ).send()
        
        # Send history if available
        if history:
            for msg in history:
                await cl.Message(
                    content=msg["content"],
                    author="User" if msg["role"] == "human" else "AI",
                    disable_feedback=True
                ).send()
        
        # Model selection
        model_res = await cl.AskActionMessage(
            content="Select a model:",
            actions=[
                cl.Action(name=m, value=m, label=m.capitalize()) 
                for m in AVAILABLE_MODELS
            ]
        ).send()
        
        if model_res and "value" in model_res:
            cl.user_session.set("model", model_res["value"])
            await cl.Message(content=f"Using {model_res['value']} model").send()
        else:
            cl.user_session.set("model", DEFAULT_MODEL)
            await cl.Message(content=f"Using default {DEFAULT_MODEL} model").send()
            
    except Exception as e:
        await cl.Message(
            content=f"An error occurred: {str(e)}. Please refresh the page to try again.",
            disable_feedback=True
        ).send()
        return 
async def process_file_upload(element: cl.File, thread_id: str):
    msg = cl.Message(content=f"Processing {element.name}...", author="AI")
    await msg.send()
    
    try:
        # Save file temporarily
        file_path = f"/tmp/{element.name}"
        with open(file_path, "wb") as f:
            f.write(element.content)
        
        # Upload to API
        async with httpx.AsyncClient() as client:
            files = {"file": open(file_path, "rb")}
            data = {"thread_id": thread_id}
            
            response = await client.post(
                f"{API_BASE_URL}/chat/upload_doc",
                files=files,
                data=data
            )
            
            result = response.json()
            if response.status_code == 200:
                await msg.update(content=f"✅ Document uploaded successfully!\n{result['filename']} is now available for questions.")
            else:
                await msg.update(content=f"❌ Failed to upload document: {result.get('message', 'Unknown error')}")
        
        os.remove(file_path)
    except Exception as e:
        await msg.update(content=f"❌ Error processing file: {str(e)}")

@cl.action_callback
async def handle_actions(action: cl.Action):
    if action.name == "rag_query":
        # Handle RAG query from a button
        thread_id = cl.user_session.get("current_thread_id")
        doc_id = action.value.split("|")[0]
        query = action.value.split("|")[1]
        
        msg = cl.Message(content="", author="AI")
        await msg.send()
        
        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                f"{API_BASE_URL}/chat/rag",
                json={
                    "thread_id": thread_id,
                    "document_id": doc_id,
                    "query": query,
                    "model": cl.user_session.get("model", DEFAULT_MODEL)
                }
            ) as response:
                async for chunk in response.aiter_text():
                    if chunk.startswith("data: "):
                        data = chunk[6:].strip()
                        if data == "[DONE]":
                            break
                        try:
                            token = json.loads(data).get("token", "")
                            await msg.stream_token(token)
                        except:
                            pass
        
        await msg.update()

@cl.password_auth_callback
def auth_callback(username: str, password: str) -> Optional[cl.User]:
    # Dummy auth - in production use real authentication
    if username in users_db:
        return cl.User(identifier=username)
    return None

