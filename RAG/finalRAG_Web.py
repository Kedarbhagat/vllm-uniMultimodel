# agent.py — Fixed LangGraph implementation

from typing import TypedDict, List, Generator, Dict, Any, Optional, Union, Literal
from langgraph.graph import StateGraph, END
from langchain.schema import BaseMessage, HumanMessage, AIMessage, Document
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
import os

from wordloader import process_document  # Assuming this is for document processing

# --- Configuration ---
API_GATEWAY_URL = os.getenv("API_GATEWAY_URL", "http://172.17.35.82:8082")
YOUR_MODEL_BASE_URL = f"{API_GATEWAY_URL}/v1"
YOUR_MODEL_NAME = os.getenv("VLLM_MODEL_NAME", "./DeepSeek-Coder-V2-Lite-Instruct-awq")
YOUR_API_KEY = os.getenv("YOUR_API_KEY", "token-abc123") 
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "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")

# Initialize LLMs
llm_chat = ChatOpenAI(
    model=YOUR_MODEL_NAME,
    openai_api_base=YOUR_MODEL_BASE_URL,
    openai_api_key=YOUR_API_KEY,
    temperature=0.7,
    max_tokens=5000,
    streaming=True,
)

llm_summary = ChatOpenAI(
    model=YOUR_MODEL_NAME,
    openai_api_base=YOUR_MODEL_BASE_URL,
    openai_api_key=YOUR_API_KEY,
    temperature=0.7,
    max_tokens=6000,
    streaming=True,
)

# --- Tools ---
@tool
def generate_mermaid_diagram(
    nodes: List[str],
    edges: List[str],
    diagram_type: Literal["flowchart", "sequenceDiagram", "classDiagram", "stateDiagram"] = "flowchart",
    direction: Literal["TD", "LR", "TB", "RL"] = "TD",
    title: Optional[str] = None
) -> str:
    """Generate Mermaid.js diagrams. Provide nodes and edges in Mermaid syntax."""
    diagram = f"{diagram_type} {direction}\n"
    if title:
        diagram += f"    title {title}\n"
    diagram += "    " + "\n    ".join(nodes) + "\n"
    diagram += "    " + "\n    ".join(edges)
    return f"```mermaid\n{diagram}\n```"

def make_retrieve_context_tool(vectorstore):
    @tool
    def retrieve_context_tool(question: str, history: str = "") -> str:
        """Retrieve relevant context from vector store."""
        query = f"{question} {history}"
        docs = vectorstore.similarity_search(query, k=4)
        return "\n\n".join(f"Doc {i+1}: {doc.page_content}" for i, doc in enumerate(docs))
    return retrieve_context_tool

@tool
def summarize_document(document_path: str) -> str:
    """Summarize long documents using the specialized summary LLM."""
    try:
        document_text = process_document(document_path)  # Assuming this processes the doc
        prompt = ChatPromptTemplate.from_template(
            "Please summarize the following document, focusing on key points "
            "and maintaining important technical details:\n\n{document}"
        )
        chain = prompt | llm_summary
        return chain.invoke({"document": document_text}).content
    except Exception as e:
        return f"Error processing document: {str(e)}"

# --- Agent State ---
class AgentState(TypedDict):
    messages: List[BaseMessage]
    question: str
    answer: str
    chat_history: List[BaseMessage]

# --- Tool-Based Agent with LangGraph ---
class ToolBasedAgent:
    def __init__(self, llm_chat: ChatOpenAI, llm_summary: ChatOpenAI, vectorstore):
        self.vectorstore = vectorstore
        self.llm_chat = llm_chat
        self.llm_summary = llm_summary
        
        # Initialize search tool with proper error handling
        try:
            self.search_tool = TavilySearchResults(
                api_key=TAVILY_API_KEY,
                tavily_api_key=TAVILY_API_KEY
            )
        except Exception as e:
            print(f"Warning: Could not initialize Tavily search tool: {e}")
            self.search_tool = None
        
        self.retrieve_context_tool = make_retrieve_context_tool(self.vectorstore)

        # Create list of available tools
        self.tools = [
            generate_mermaid_diagram,
            self.retrieve_context_tool,
            summarize_document
        ]
        
        # Add search tool if available
        if self.search_tool:
            self.tools.append(self.search_tool)

        # Bind tools to LLM
        self.tool_llm = self.llm_chat.bind_tools(self.tools, tool_choice="auto")

        # Build the workflow
        self.workflow = self._build_workflow()
        self.app = self.workflow.compile()

    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("agent", self.call_model)
        workflow.add_node("tools", ToolNode(self.tools))
        workflow.add_node("generate_final_answer", self.generate_final_answer)

        # Set entry point
        workflow.set_entry_point("agent")

        # Add conditional edges
        workflow.add_conditional_edges(
            "agent",
            self.should_continue,
            {
                "continue": "tools",
                "end": "generate_final_answer"
            }
        )

        # Add edge from tools back to agent
        workflow.add_edge("tools", "agent")
        
        # Add edge from final answer to END
        workflow.add_edge("generate_final_answer", END)

        return workflow

    def call_model(self, state: AgentState) -> AgentState:
        """Call the model with the current messages."""
        messages = state["messages"]
        
        # Get response from the tool-bound LLM
        response = self.tool_llm.invoke(messages)
        
        # Add the response to messages
        messages.append(response)
        
        return {"messages": messages}

    def should_continue(self, state: AgentState) -> str:
        """Determine whether to continue with tools or end."""
        last_message = state["messages"][-1]
        
        # If there are tool calls, continue to tools
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return "continue"
        else:
            return "end"

    def generate_final_answer(self, state: AgentState) -> AgentState:
        """Generate the final answer."""
        last_message = state["messages"][-1]
        
        # If the last message is from the AI and doesn't have tool calls, use it as the answer
        if isinstance(last_message, AIMessage):
            answer = last_message.content
        else:
            answer = "I'm sorry, I couldn't generate a proper response."
        
        return {"answer": answer}

    def stream_chat(self, question: str, chat_history: List[BaseMessage]) -> Generator[str, None, None]:
        """Streaming chat interface using LangGraph."""
        # Prepare initial state
        messages = chat_history[-6:] if chat_history else []  # Last 3 exchanges
        messages.append(HumanMessage(content=question))
        
        initial_state: AgentState = {
            "messages": messages,
            "question": question,
            "chat_history": chat_history,
            "answer": ""
        }
        
        try:
            # Stream through the workflow
            final_state = None
            for step in self.app.stream(initial_state):
                # Get the current state
                for node_name, node_state in step.items():
                    if node_name == "generate_final_answer" and "answer" in node_state:
                        final_state = node_state
                        break
            
            # Return the final answer
            if final_state and "answer" in final_state:
                yield final_state["answer"]
            else:
                # Fallback: get the last AI message
                for message in reversed(initial_state["messages"]):
                    if isinstance(message, AIMessage):
                        yield message.content
                        break
                else:
                    yield "I'm sorry, I couldn't process your request properly."
                    
        except Exception as e:
            yield f"Error processing request: {str(e)}"

    def chat(self, question: str, chat_history: List[BaseMessage]) -> str:
        """Non-streaming chat interface."""
        return "".join(self.stream_chat(question, chat_history))

if __name__ == "__main__":
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain.schema import AIMessage
    import os

    # Ensure environment variables are set properly
    if not TAVILY_API_KEY or TAVILY_API_KEY == "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":
        print("Warning: Using default/demo Tavily API key. Set TAVILY_API_KEY environment variable for production use.")

    # Initialize embedding model and vectorstore
    print("Initializing test environment...")
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Create dummy documents for testing
    dummy_docs = [
        Document(page_content="The capital of France is Paris. It is known for the Eiffel Tower.", metadata={"source": "wiki"}),
        Document(page_content="The Amazon River is the largest river by discharge volume of water in the world.", metadata={"source": "geography"}),
        Document(page_content="Python is a popular programming language for AI and web development.", metadata={"source": "programming"}),
    ]
    vectorstore = FAISS.from_documents(dummy_docs, embedding_model)

    # Initialize the agent
    print("Initializing ToolBasedAgent...")
    try:
        agent = ToolBasedAgent(llm_chat, llm_summary, vectorstore)
        print("Agent initialized successfully!")
    except Exception as e:
        print(f"Failed to initialize agent: {str(e)}")
        exit(1)

    # Simulate chat history
    chat_history = []

    def run_chat_interaction(question, documents=None):
        print(f"\nUser: {question}")
        print("Agent: ", end="", flush=True)
        
        full_response = ""
        try:
            for chunk in agent.stream_chat(question, chat_history):
                print(chunk, end="", flush=True)
                full_response += chunk
        except Exception as e:
            print(f"Error during chat: {str(e)}")
            return ""
        
        print()  # New line after response
        
        chat_history.extend([
            HumanMessage(content=question),
            AIMessage(content=full_response)
        ])
        return full_response

    # Test basic RAG functionality
    print("\n--- Testing Basic RAG Capabilities ---")
    run_chat_interaction("What is the capital of France?")
    run_chat_interaction("Tell me about Python programming")

    # Test document processing
    print("\n--- Testing Document Processing ---")
    test_document_path = r"C:\Users\STUDENT\Documents\vllm\RAG\Graduate Trainee - Technology- Aug24.pdf"
    
    if os.path.exists(test_document_path):
        print(f"Processing document: {test_document_path}")
        try:
            processed_docs = process_document(test_document_path)
            vectorstore.add_documents(processed_docs)
            
            # Test document-related queries
            run_chat_interaction("What are the key points from the uploaded document?")
            run_chat_interaction("What are the main topics covered in the document?")
        except Exception as e:
            print(f"Document processing failed: {str(e)}")
    else:
        print("Test document not found. Skipping document processing tests.")

    # Test tool usage
    print("\n--- Testing Tool Usage ---")
    if agent.search_tool:
        run_chat_interaction("what is nvidia stock price now ?")
    else:
        print("Skipping search test - Tavily API not available")
    
    run_chat_interaction("Create a flowchart diagram with nodes A, B, C and edges A-->B, B-->C")

    print("\n--- Conversation History ---")
    for i, msg in enumerate(chat_history):
        prefix = "User" if isinstance(msg, HumanMessage) else "Agent"
        print(f"{i+1}. {prefix}: {msg.content[:100]}...")

    print("\nTesting complete.")