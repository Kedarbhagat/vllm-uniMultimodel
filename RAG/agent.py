import os
import datetime
from typing import Annotated, List, Dict
from typing_extensions import TypedDict
from dataclasses import dataclass

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI

#from dbhelper import get_or_create_user, create_thread, add_message, get_messages_by_thread

# --- LLM ---
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "token-abc123")
#API_GATEWAY_URL = os.getenv("API_GATEWAY_URL", "http://172.17.35.82:8082")
#YOUR_MODEL_NAME = os.getenv("VLLM_MODEL_NAME", "./meta-llama/Llama-3.1-8B-Instruct-awq")
#YOUR_API_KEY = os.getenv("YOUR_API_KEY", "token-abc123")

#llm = ChatOpenAI(
    #model=YOUR_MODEL_NAME,
   # openai_api_base=f"{API_GATEWAY_URL}/v1",
    #openai_api_key=YOUR_API_KEY,
   # temperature=0.7,
  #  max_tokens=2048,
 #   streaming=True,
#)
MODEL_REGISTRY = {
    "llama": "./meta-llama/Llama-3.1-8B-Instruct-awq",
    "deepseek": "./DeepSeek-Coder-V2-Lite-Instruct-awq"
}
API_GATEWAY_URL = "http://172.17.35.82:8082"
DEFAULT_MODEL = "llama"

# --- Prompt Template ---
prompt_template = ChatPromptTemplate.from_messages([
    ("system",'''You are a helpful assistant. conversational in your responses. ask followup questions if needed. 
     complete the each response within 1200 tokens.'''),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
]).partial(current_time=lambda: datetime.datetime.now().strftime("%A, %B %d, %Y at %I:%M:%S %p IST"))

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

class ConversationalAgent:
    def __init__(self, prompt_template: ChatPromptTemplate,model_key: str = "llama"):
        model_name=MODEL_REGISTRY.get(model_key,MODEL_REGISTRY[DEFAULT_MODEL] )
        self.llm = ChatOpenAI(
            model=model_name,
            openai_api_base=f"{API_GATEWAY_URL}/v1",
            openai_api_key=os.getenv("YOUR_API_KEY", "token-abc123"),
            temperature=0.7,
            max_tokens=5000,
            streaming=True,
        )
        self.prompt_template = prompt_template

    def stream_response(self, messages: List[BaseMessage]):
        content_buffer = []
        try:
            prepared_messages = self._prepare_messages(messages)
            for chunk in self.llm.stream(prepared_messages):
                if chunk.content:
                    content_buffer.append(chunk.content)
                    print(chunk.content, end="", flush=True)
            print()
            return AIMessage(content="".join(content_buffer))
        except Exception as e:
            print(f"\n❌ Error during streaming: {e}")
            return AIMessage(content=f"Sorry, something went wrong: {e}")

    def stream_response_for_web(self, messages: List[BaseMessage]):
        try:
            prepared_messages = self._prepare_messages(messages)
            for chunk in self.llm.stream(prepared_messages):
                if chunk.content:
                    yield chunk.content  # Yield, do NOT print
        except Exception as e:
            yield f"\n❌ Error during streaming: {e}"

    def _prepare_messages(self, messages: List[BaseMessage]) -> List[BaseMessage]:
        if not messages:
            return []

        input_msg = ""
        history = []

        for i in reversed(range(len(messages))):
            if isinstance(messages[i], HumanMessage):
                input_msg = messages[i].content
                history = messages[:i] + messages[i+1:]
                break

        return self.prompt_template.invoke({
            "chat_history": history,
            "input": input_msg,
            "current_time": datetime.datetime.now().strftime("%A, %B %d, %Y at %I:%M:%S %p IST")
        }).messages

def conversational_node(state: AgentState) -> Dict[str, List[BaseMessage]]:
    messages = state["messages"]
    agent = ConversationalAgent(prompt_template)
    response = agent.stream_response_for_web(messages)
    return {"messages": [response]}

# --- LangGraph ---
workflow = StateGraph(AgentState)
workflow.add_node("agent", conversational_node)
workflow.set_entry_point("agent")
graph = workflow.compile(checkpointer=MemorySaver())

def reconstruct_history(raw_messages):
    history = []
    for msg in raw_messages:
        if msg["role"] == "human":
            history.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "ai":
            history.append(AIMessage(content=msg["content"]))
    return history

#def run_chat():
    print("🧠 AI Assistant - Conversational Mode")
    email = input("Enter your email: ").strip() or "anonymous@example.com"
    user_id = get_or_create_user(email)
    thread_id = str(create_thread(user_id))  # Ensure thread_id is a string
    print(f"New chat started (Thread ID: {thread_id})")

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            print("👋 Bye!")
            break

        add_message(thread_id, role="human", content=user_input)
        raw_history = get_messages_by_thread(thread_id)
        history = reconstruct_history(raw_history)

        # ✅ Fixed: Separate state input and config
        current_state_input = {
            "messages": history
        }

        config = {
            "configurable": {
                "thread_id": thread_id  # ✅ Required for MemorySaver checkpointer
            }
        }

        # ✅ Fixed: Pass config as separate parameter
        final_state = graph.invoke(current_state_input, config=config)

        new_messages = final_state["messages"][len(history):]
        for msg in new_messages:
            if isinstance(msg, AIMessage):
                add_message(thread_id, role="ai", content=msg.content)

if __name__ == "__main__":
    #run_chat()
    print("This module is not meant to be run directly. Use the server.py to start the application.")