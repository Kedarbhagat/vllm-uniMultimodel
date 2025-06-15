from typing import TypedDict, List, Generator, Dict, Any, Optional

from langgraph.graph import StateGraph, END
from langchain.schema import BaseMessage, HumanMessage, AIMessage, Document
from langchain.prompts import ChatPromptTemplate
import os

from langchain_openai import ChatOpenAI
from wordloader import process_document  

# --- Environment variables and LLM initialization ---
# These are typically handled in the main FastAPI app, but kept here for self-containment
# for demonstration purposes if this file were to be run directly.

API_GATEWAY_URL = os.getenv("API_GATEWAY_URL", "http://172.17.35.82:8082")
YOUR_MODEL_BASE_URL = f"{API_GATEWAY_URL}/v1"
YOUR_MODEL_NAME = os.getenv("VLLM_MODEL_NAME", "./DeepSeek-Coder-V2-Lite-Instruct-awq")
YOUR_API_KEY = os.getenv("YOUR_API_KEY", "token-abc123")  # Replace with real key if needed

# Define two LLM instances with different max_tokens

os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGSMITH_API_KEY"] = "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" # Replace with your actual key
os.environ["LANGSMITH_PROJECT"] = "vllm-enhanced-rag_newAgent"


llm_chat = ChatOpenAI(
    model=YOUR_MODEL_NAME,
    openai_api_base=YOUR_MODEL_BASE_URL,
    openai_api_key=YOUR_API_KEY,
    temperature=0.7,
    max_tokens=5000,  # Efficient for conversational responses
    streaming=True,
)

llm_summary = ChatOpenAI(
    model=YOUR_MODEL_NAME,
    openai_api_base=YOUR_MODEL_BASE_URL,
    openai_api_key=YOUR_API_KEY,
    temperature=0.7,
    max_tokens=6000,  # Significantly increased for document processing
    streaming=True,
)


# --- AgentState Definition ---
class AgentState(TypedDict):
    messages: List[BaseMessage]
    question: str
    context: str
    answer: str
    chat_history: List[BaseMessage]  # History for the *current* turn from the caller
    needs_retrieval: bool
    streaming_content: str
    needs_full_document_processing: bool
    full_document_content: str
    document_path: Optional[str]
    documents: Optional[List[Document]]  # Added to pass pre-chunked documents


# --- LangGraphAgenticRAG Class ---
class LangGraphAgenticRAG:
    def __init__(self, llm_chat: ChatOpenAI, llm_summary: ChatOpenAI, vectorstore, embedding_model):
        self.llm_chat = llm_chat
        self.llm_summary = llm_summary
        self.vectorstore = vectorstore
        self.embedding_model = embedding_model
        # REMOVED: self.chat_history: List[BaseMessage] = []
        self.last_used_full_document: bool = False  # Retained for context_used tracking in FastAPI

        # Build the LangGraph workflow
        self.workflow = self._build_workflow()
        self.app = self.workflow.compile()

    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("analyze_question", self.analyze_question_node)
        workflow.add_node("retrieve_context", self.retrieve_context_node)
        workflow.add_node("process_full_document", self.process_full_document_node)
        workflow.add_node("generate_answer", self.generate_answer_node)

        # Add edges
        workflow.set_entry_point("analyze_question")

        workflow.add_conditional_edges(
            "analyze_question",
            self.route_question,
            {
                "retrieve": "retrieve_context",
                "direct": "generate_answer",
                "process_full_document": "process_full_document"
            }
        )

        workflow.add_edge("retrieve_context", "generate_answer")
        workflow.add_edge("process_full_document", "generate_answer")
        workflow.add_edge("generate_answer", END)

        return workflow

    def analyze_question_node(self, state: AgentState) -> AgentState:
        """Node to analyze if retrieval, full document processing, or direct answer is needed."""
        # Use state["chat_history"] directly as it's passed into the graph state
        chat_hist_str = "\n".join(
            [
                f"{'Human' if isinstance(msg, HumanMessage) else 'AI'}: {msg.content}"
                for msg in state["chat_history"][-6:]  # Using last 6 messages from the passed history
            ]
        )

        analysis_prompt = ChatPromptTemplate.from_template(
            """
You are an intelligent router. Based on the following user question and chat history,
determine the appropriate action.

Use retrive for most of the user input. unless it is very necessary to summarize or analyze an external document  use it. 
but always try to use retrieve first. for topics like explain a topic etc.. use retrival 
if the question is about "generate question based on this document" or "summarize this document" or "analyze this document" then use summarize_or_analyze_external_doc action.  

Actions:
- "RETRIEVE": If the question requires looking up factual information in a knowledge base (vectorstore). This is for specific facts not directly mentioned in previous chat turns.
- "SUMMARIZE_OR_ANALYZE_EXTERNAL_DOC": If the question explicitly asks to summarize, analyze, or provide opinion on an *external document* that has just been provided or is clearly referenced as the primary subject of the current turn. This action should ONLY be chosen if a `document_path` or `documents` (pre-chunked content) is available and the query is about that document's content.
- "DIRECT": If the question can be answered directly using information from the current conversation history, general knowledge, or if it's a simple greeting/chit-chat. This also includes summarizing or rephrasing *previous conversational turns*.

Question: {question}

Chat History:
{chat_history}

Current document_status (if any): {document_status}

Respond with one of these exact actions: "RETRIEVE", "SUMMARIZE_OR_ANALYZE_EXTERNAL_DOC", or "DIRECT".
Do not include any other text or explanation in your response, just the action.
            """.strip()
        )

        document_status = "A document (path or pre-chunked content) is available for processing." if state.get("document_path") or state.get("documents") else "No specific document provided for this turn."

        prompt_text = analysis_prompt.format(
            question=state["question"],
            chat_history=chat_hist_str,
            document_status=document_status
        )

        messages = [HumanMessage(content=prompt_text)]
        response = self.llm_chat.invoke(messages)
        content = response.content.strip().upper()
        action = content.split('\n')[0].strip()

        state["needs_retrieval"] = "RETRIEVE" in action
        state["needs_full_document_processing"] = "SUMMARIZE_OR_ANALYZE_EXTERNAL_DOC" in action
        print(f"Analyze Question: Determined action = '{action}'. Retrieval needed: {state['needs_retrieval']}, Full doc processing needed: {state['needs_full_document_processing']}")
        return state

    def route_question(self, state: AgentState) -> str:
        """Conditional edge function to route based on analysis."""
        if state["needs_full_document_processing"]:
            print("Routing to process_full_document.")
            return "process_full_document"
        elif state["needs_retrieval"]:
            print("Routing to retrieve_context.")
            return "retrieve"
        else:
            print("Routing to direct (generate_answer without retrieval).")
            return "direct"

    def retrieve_context_node(self, state: AgentState) -> AgentState:
        """Node to retrieve context from vectorstore."""
        print("Executing retrieve_context_node...")
        recent_context = ""
        # Use state["chat_history"] directly
        if state["chat_history"]:
            recent_msgs = state["chat_history"][-4:]
            recent_context = " ".join([msg.content for msg in recent_msgs])

        enhanced_query = f"{state['question']} {recent_context}".strip()
        docs = self.vectorstore.similarity_search(enhanced_query, k=4)

        context = "\n\n".join([
            f"Document {i+1}: {doc.page_content}"
            for i, doc in enumerate(docs)
        ])
        print(f"Retrieved context (first 100 chars): {context[:100]}...")
        state["context"] = context
        return state

    def process_full_document_node(self, state: AgentState) -> AgentState:
        """
        Node to load the full document content for summarization/analysis.
        Assumes the 'document_path' or 'documents' is available in the AgentState.
        """
        print("Executing process_full_document_node...")
        langchain_docs = state.get("documents")  # Prefer pre-chunked documents

        if langchain_docs:
            print(f"Using pre-chunked documents (count: {len(langchain_docs)})")
        else:
            document_path = state.get("document_path")
            if document_path and os.path.exists(document_path):
                print(f"Loading full document from: {document_path}")
                langchain_docs: List[Document] = process_document(document_path)
            else:
                error_msg = f"Error: No valid document path or pre-chunked documents provided for full document processing."
                print(error_msg)
                state["full_document_content"] = error_msg
                state["context"] = error_msg
                return state

        if not langchain_docs:
            no_content_msg = f"Error: No content extracted from document."
            print(no_content_msg)
            state["full_document_content"] = no_content_msg
            state["context"] = no_content_msg
            return state

        full_text = "\n\n".join([doc.page_content for doc in langchain_docs])
        state["full_document_content"] = full_text
        state["context"] = full_text
        print(f"Successfully loaded full document content (chars: {len(full_text)})")
        return state

    def generate_answer_node(self, state: AgentState) -> AgentState:
        """Node to generate the final answer - this will be called within LangGraph."""
        print("Executing generate_answer_node...")
        # Use state["chat_history"] directly
        chat_hist_str = "\n".join(
            [
                f"{'Human' if isinstance(msg, HumanMessage) else 'Assistant'}: {msg.content}"
                for msg in state["chat_history"][-8:]  # Using last 8 messages from the passed history
            ]
        )

        is_full_document_task = state.get("needs_full_document_processing", False)

        if state.get("context"):
            if is_full_document_task:
                answer_prompt = ChatPromptTemplate.from_template(
                    """
You are a helpful assistant. You have been provided with the full content of a document.
Based on this full document and the chat history, answer the current question.
Your task is to summarize, analyze, or provide an opinion on the document as requested.

Full Document Content:
{context}

Chat History:
{chat_history}

Current Question: {question}

Provide a helpful, accurate answer that:
1. Addresses the question directly using the provided full document content.
2. References previous conversation when appropriate.
3. Is conversational and natural.
4. Admits if you don't have enough information from the document or history.

Answer directly without any function calls or special formatting.
                    """.strip()
                )
            else:  # Regular RAG context
                answer_prompt = ChatPromptTemplate.from_template(
                    """
You are a helpful assistant. Answer the question based on the provided context
and chat history. Be conversational and refer to previous discussion when relevant.

Context from Knowledge Base:
{context}

Chat History:
{chat_history}

Current Question: {question}

Provide a helpful, accurate answer that:
1. Uses the retrieved context when relevant.
2. References previous conversation when appropriate.
3. Is conversational and natural.
4. Admits if you don't have enough information.

Answer directly without any function calls or special formatting.
                    """.strip()
                )

            prompt_text = answer_prompt.format(
                context=state["context"],
                chat_history=chat_hist_str,
                question=state["question"]
            )
        else:  # No context (direct answer based on chat history/general knowledge)
            answer_prompt = ChatPromptTemplate.from_template(
                """
You are a helpful assistant. Answer the question based on the chat history
and your general knowledge. Be conversational and refer to previous
discussion when relevant.

Chat History:
{chat_history}

Current Question: {question}

Provide a helpful, natural response that builds on the conversation context.
Answer directly without any function calls or special formatting.
                """.strip()
            )

            prompt_text = answer_prompt.format(
                chat_history=chat_hist_str,
                question=state["question"]
            )

        messages = [HumanMessage(content=prompt_text)]

        llm_to_use = self.llm_summary if is_full_document_task else self.llm_chat
        print(f"Using LLM: {'llm_summary' if is_full_document_task else 'llm_chat'}")

        full_answer = ""
        for chunk in llm_to_use.stream(messages):
            if hasattr(chunk, 'content') and chunk.content:
                full_answer += chunk.content
            elif isinstance(chunk, dict):
                content = chunk.get('content', '')
                if content:
                    full_answer += content

        state["answer"] = full_answer
        return state

    def stream_chat(self, question: str, chat_history: List[BaseMessage],  # ADDED chat_history parameter
                    document_path: Optional[str] = None, documents: Optional[List[Document]] = None) -> Generator[str, None, None]:
        """
        Stream the chat response token by token using LangGraph workflow.
        Accepts chat_history as a parameter.
        """
        print(f"Initiating stream_chat for question: '{question}'")
        initial_state: AgentState = {
            "messages": [HumanMessage(content=question)],
            "question": question,
            "context": "",
            "answer": "",
            "chat_history": chat_history if chat_history else [],  # USE THE PASSED IN HISTORY
            "needs_retrieval": False,
            "streaming_content": "",
            "needs_full_document_processing": False,
            "full_document_content": "",
            "document_path": document_path,
            "documents": documents,
        }

        # Run the workflow to determine the processing path
        # The result of this invoke is the state *after* analyze_question, retrieve/process_full_document, and generate_answer
        # The generate_answer node will populate 'answer' and 'context' based on the workflow.
        # We need to explicitly run the whole graph to get the final state with the answer
        # The streaming logic for LLM is handled directly in generate_answer_node within the graph for better control
        # However, for external streaming, we need to re-run the LLM stream based on the determined path.
        # This part of the design can be tricky with LangGraph's default streaming if you want node-by-node streaming.
        # For simplicity, we'll run the graph to determine path and context, then stream the final LLM call.

        # Run the entire graph to get the final answer and context from the chosen path
        final_state = self.app.invoke(initial_state)

        # Update the flag based on the workflow's decision (from the final state)
        self.last_used_full_document = final_state.get("needs_full_document_processing", False)

        # Prepare the chat history string (using the history that was passed in)
        chat_hist_str = "\n".join(
            f"{'Human' if isinstance(msg, HumanMessage) else 'Assistant'}: {msg.content}"
            for msg in chat_history[-8:]  # Use the original passed history for prompt context
        )

        # Determine which LLM to use
        llm_to_use = self.llm_summary if self.last_used_full_document else self.llm_chat
        print(f"Using LLM: {'llm_summary' if self.last_used_full_document else 'llm_chat'}")

        # Prepare the prompt based on available context and the question
        # This prompt is essentially what generate_answer_node *would* have used
        if final_state.get("context"):
            if self.last_used_full_document:
                prompt = f"""
Full Document Content:
{final_state['context']}

Chat History:
{chat_hist_str}

Question: {question}

Please analyze/summarize the document content to answer the question.
"""
            else:
                prompt = f"""
Context from Knowledge Base:
{final_state['context']}

Chat History:
{chat_hist_str}

Question: {question}

Please answer using the provided context.
"""
        else:
            prompt = f"""
Chat History:
{chat_hist_str}

Question: {question}

Please answer based on the conversation history.
"""

        messages_for_llm = [HumanMessage(content=prompt.strip())]
        full_response_content = ""
        skip_tokens = False

        # Stream tokens one by one directly from the chosen LLM
        for chunk in llm_to_use.stream(messages_for_llm):
            content = ""
            if hasattr(chunk, 'content'):
                content = chunk.content
            elif isinstance(chunk, dict):
                content = chunk.get('content', '')

            if not content:
                continue

            # Handle special tokens (as before)
            if any(pattern in content for pattern in ['<｜tool ', '｜tool ', 'function<｜tool sep｜>']):
                skip_tokens = True
                continue

            if skip_tokens:
                if not any(pattern in content for pattern in ['<｜', '｜>']):
                    skip_tokens = False
                continue

            # Yield each character individually for true streaming
            for char in content:
                full_response_content += char
                yield char
        
        # The history update responsibility is now shifted to the caller (FastAPI)
        print("Stream finished. Caller is responsible for chat history update.")

    def chat(self, question: str, chat_history: List[BaseMessage],  # ADDED chat_history parameter
             document_path: Optional[str] = None, documents: Optional[List[Document]] = None) -> str:
        """
        Non-streaming chat method using LangGraph workflow.
        Accepts chat_history as a parameter.
        """
        print(f"Initiating chat (non-streaming) for question: '{question}' with document_path: {document_path} and documents: {bool(documents)}")
        initial_state: AgentState = {
            "messages": [HumanMessage(content=question)],
            "question": question,
            "context": "",
            "answer": "",
            "chat_history": chat_history if chat_history else [],  # USE THE PASSED IN HISTORY
            "needs_retrieval": False,
            "streaming_content": "",
            "needs_full_document_processing": False,
            "full_document_content": "",
            "document_path": document_path,
            "documents": documents,
        }

        # Invoke the entire graph to get the final state with the answer
        final_state = self.app.invoke(initial_state)
        print(f"Workflow invoked. Final answer (non-streaming): {final_state['answer'][:100]}...")

        # Update the flag based on the workflow's decision
        self.last_used_full_document = final_state.get("needs_full_document_processing", False)

        # The history update responsibility is now shifted to the caller (FastAPI)
        print("Chat finished. Caller is responsible for chat history update.")

        return final_state["answer"]


# -------- USAGE EXAMPLE (for testing the agent in isolation) --------
if __name__ == "__main__":
    from vectorestore import build_vectorstore  # Ensure you have this implemented properly
    from embeddingmodel import MiniLMEmbeddings  # Ensure this exists

    embedding_model = MiniLMEmbeddings()

    # Create dummy documents for the vectorstore
    dummy_docs = [
        Document(page_content="The capital of France is Paris. It is known for the Eiffel Tower.", metadata={"source": "wiki"}),
        Document(page_content="The Amazon River is the largest river by discharge volume of water in the world.", metadata={"source": "geography"}),
        Document(page_content="Python is a popular programming language for AI and web development.", metadata={"source": "programming"}),
    ]
    vectorstore_for_test = build_vectorstore(dummy_docs)

    # Initialize the agent
    rag_agent = LangGraphAgenticRAG(llm_chat, llm_summary, vectorstore_for_test, embedding_model)

    # Simulate chat history
    current_chat_history_for_test: List[BaseMessage] = []

    

    print("\n--- Simulating Document Summary (if you have a real document path) ---")
    # For this to work, you need a valid path to a PDF/DOCX that your `process_document` can handle.
    # Replace with a path to a small test document if you want to test this.
    test_document_path = r"C:\Users\STUDENT\Documents\vllm\RAG\nces (1).pdf"  # Adjust this path

    if os.path.exists(test_document_path):
        print(f"Attempting to process document: {test_document_path}")
        # When processing a document, you typically create a new vectorstore for it
        # or update the existing one in your FastAPI app's /upload-document endpoint.
        # Here, we're just testing the agent's ability to handle the document input.
        processed_docs_for_test = process_document(test_document_path)
        
        question_doc_summary = "summarise this document in 10 lines"
        full_response_doc_summary = ""
        # IMPORTANT: Passing the document_path/documents is crucial here
        for chunk in rag_agent.stream_chat(question_doc_summary, chat_history=current_chat_history_for_test, documents=processed_docs_for_test):
            full_response_doc_summary += chunk
            print(chunk, end="", flush=True)
        current_chat_history_for_test.append(HumanMessage(content=question_doc_summary))
        current_chat_history_for_test.append(AIMessage(content=full_response_doc_summary))
        print(f"\nUpdated chat history length: {len(current_chat_history_for_test)}")
        print("-" * 30)


        question_doc_summary = "what is my name?"
        for chunk in rag_agent.stream_chat(question_doc_summary, chat_history=current_chat_history_for_test, documents=processed_docs_for_test):
            full_response_doc_summary += chunk
            print(chunk, end="", flush=True)
        current_chat_history_for_test.append(HumanMessage(content=question_doc_summary))
        current_chat_history_for_test.append(AIMessage(content=full_response_doc_summary))
        print(f"\nUpdated chat history length: {len(current_chat_history_for_test)}")
        print("-" * 30)
    else:
        print(f"Skipping document summary test: Document not found at {test_document_path}")
        print("Please ensure you have a document at the specified path to test full document processing.")