# enhanced_hybrid_langgraph_agent.py
# Enhanced LangGraph agent with improved routing, error handling, and chat history
# Optimized for performance by removing unused features and strategic streaming.

from langgraph.graph import StateGraph, END
from typing import Annotated, TypedDict, Literal, Optional, List
from langchain_core.documents import Document
from langchain_core.runnables import Runnable
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
import logging
import re
import os

# --- START LangSmith Configuration (Hardcoded for convenience) ---
# IMPORTANT: For production, load these from actual environment variables
# or a secure configuration management system, NOT hardcode them.

# Set up logging



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- VLLM Model Configuration ---
API_GATEWAY_URL = os.getenv("API_GATEWAY_URL", "http://172.17.35.82:8082")
YOUR_MODEL_BASE_URL = f"{API_GATEWAY_URL}/v1"
YOUR_MODEL_NAME = os.getenv("VLLM_MODEL_NAME", "./DeepSeek-Coder-V2-Lite-Instruct-awq")
YOUR_API_KEY = os.getenv("YOUR_API_KEY", "token-abc123") # Use dummy if not enforced

# --- 1. Agent State Definition ----
class AgentState(TypedDict):
    question: str
    doc_path: str
    documents: Optional[List[Document]]
    vectorstore: Optional[object]
    answer: str
    # Removed 'comparison' as it's not implemented, simplifying routing.
    route: Literal['rag', 'summary', 'qa_gen', 'analysis']
    error: Optional[str]
    metadata: Optional[dict]
    chat_history: List[dict] # List of {"role": "human" | "ai", "content": "message"}

# ---- Helper to format chat history for prompts ----
def format_chat_history(history: List[dict]) -> str:
    """Formats a list of chat messages into a string for prompt injection."""
    formatted_lines = []
    for entry in history:
        if entry["role"] == "human":
            formatted_lines.append(f"User: {entry['content']}")
        elif entry["role"] == "ai":
            formatted_lines.append(f"Assistant: {entry['content']}")
    return "\n".join(formatted_lines) if formatted_lines else "No prior conversation."


# ---- 2. Optimized Router ----
def route_question(state: AgentState) -> str:
    """Routes questions to appropriate nodes based on keywords."""
    q = state['question'].lower()

    patterns = {
        'summary': [
            r'\b(summarize|summary|overview|brief|outline|key points)\b',
            r'\bwhat is this (document|paper|text) about\b',
            r'\bgive me (a|an) (summary|overview)\b'
        ],
        'qa_gen': [
            r'\b(generate|create|make) questions?\b',
            r'\bquestions? (for|about|from)\b',
            r'\b(quiz|test) questions?\b',
            r'\bstudy questions?\b'
        ],
        'analysis': [
            r'\b(analyze|analysis|examine|evaluate)\b',
            r'\bwhat are the (implications|consequences|benefits|drawbacks)\b',
            r'\bhow does this (relate|compare|contrast)\b'
        ]
    }

    for route_type, pattern_list in patterns.items():
        for pattern in pattern_list:
            if re.search(pattern, q):
                logger.info(f"Routed to {route_type} based on pattern: {pattern}")
                return route_type

    return "rag" # Default to RAG for specific questions

# ---- 3. Document Loading with Error Handling ----
def load_and_index(state: AgentState) -> AgentState:
    """Load and index documents, skipping if already present in state."""
    if state.get("documents") is not None and state.get("vectorstore") is not None:
        logger.info("Documents and vectorstore already pre-loaded. Skipping loader node.")
        return state

    try:
        # Assuming wordloader.py contains the process_document function
        # and vectorestore.py contains the build_vectorstore function
        from wordloader import process_document # Make sure this import is correct
        from vectorestore import build_vectorstore

        # Call the universal process_document from wordloader.py
        # Pass kwargs to process_document so max_sentences can be controlled
        docs = process_document(state["doc_path"], max_sentences=7)
        if not docs:
            return {**state, "error": "No documents could be processed from the file"}

        vs = build_vectorstore(docs)
        if not vs:
            return {**state, "error": "Failed to build vector store"}

        logger.info(f"Loaded {len(docs)} documents and built vector store")
        return {
            **state,
            "documents": docs,
            "vectorstore": vs,
            "metadata": {"doc_count": len(docs), "total_chars": sum(len(doc.page_content) for doc in docs)}
        }
    except Exception as e:
        logger.error(f"Error in load_and_index: {str(e)}")
        return {**state, "error": f"Failed to load document: {str(e)}"}

# ---- Helper function to instantiate the VLLM-configured LLM ----
def get_vllm_llm(temperature: float, max_tokens: int = 500, streaming: bool = True):
    """Instantiates the ChatOpenAI model configured for VLLM, with streaming enabled by default."""
    return ChatOpenAI(
        model=YOUR_MODEL_NAME,
        openai_api_base=YOUR_MODEL_BASE_URL,
        openai_api_key=YOUR_API_KEY,
        temperature=temperature,
        max_tokens=max_tokens,
        streaming=streaming # Streaming is kept True for performance and user experience
    )

# ---- 4. RAG Answer Generation ----
def rag_answer(state: AgentState) -> AgentState:
    """Generates RAG answer, incorporating chat history."""
    try:
        if state.get("error"):
            return state

        # Dynamic k based on question complexity (simple heuristic)
        question_words = len(state['question'].split())
        k = min(5, max(2, question_words // 10))

        retriever = state["vectorstore"].as_retriever(
    search_kwargs={"k": k}
    )

        llm = get_vllm_llm(temperature=0.2, streaming=True)

        history_str = format_chat_history(state.get("chat_history", []))

        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""You are a helpful AI assistant that answers questions based on the provided context.
            Here is the conversation history so far:
            ---
            {history_str}
            ---

            If the context doesn't contain enough information to fully answer the current question, say so explicitly.
            Always cite which parts of the context support your answer."""),
            ("human", "Context: {context}\n\nQuestion: {question}")
        ])

        chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )

        rag_parts = []
        print("\n--- RAG Answer (Streaming) ---")
        for s in chain.stream({"query": state['question']}):
            if "result" in s:
                print(s["result"], end="", flush=True)
                rag_parts.append(s["result"])
        print("\n--- End RAG Streaming ---")

        full_rag_answer = "".join(rag_parts)

        updated_history = state.get("chat_history", []) + [
            {"role": "human", "content": state['question']},
            {"role": "ai", "content": full_rag_answer}
        ]

        return {
            **state,
            "answer": full_rag_answer,
            "metadata": {**state.get("metadata", {}), "streamed_rag": True},
            "chat_history": updated_history
        }
    except Exception as e:
        logger.error(f"Error in rag_answer: {str(e)}")
        return {**state, "error": f"RAG processing failed: {str(e)}"}

# ---- 5. Document Summarization ----
def summarize_full_doc(state: AgentState) -> AgentState:
    """Summarizes the full document, using hierarchical summary for long texts."""
    try:
        if state.get("error"):
            return state

        llm = get_vllm_llm(temperature=0.3, streaming=True)
        full_text = "\n\n".join([doc.page_content for doc in state["documents"]])

        answer = ""
        # Adaptive summarization based on document length
        if len(full_text) > 15000: # Threshold for hierarchical summary
            chunks = [full_text[i:i+12000] for i in range(0, len(full_text), 10000)] # Overlapping chunks
            chunk_summaries_parts = []

            chunk_prompt = ChatPromptTemplate.from_messages([
                ("system", "Summarize this section of a document concisely:"),
                ("human", "{chunk}")
            ])

            for i, chunk in enumerate(chunks[:3]): # Limit to first 3 chunks for brevity
                current_chunk_summary_parts = []
                print(f"\n--- Summarizing Section {i+1} (Streaming) ---")
                for s_chunk in (chunk_prompt | llm | StrOutputParser()).stream({"chunk": chunk}):
                    print(s_chunk, end="", flush=True)
                    current_chunk_summary_parts.append(s_chunk)
                print(f"\n--- End Section {i+1} Streaming ---")
                chunk_summaries_parts.append(f"Section {i+1}: {''.join(current_chunk_summary_parts)}")

            final_prompt = ChatPromptTemplate.from_messages([
                ("system", "Create a comprehensive summary from these section summaries:"),
                ("human", "{summaries}")
            ])

            final_summary_parts = []
            print("\n--- Final Summary (Streaming) ---")
            for fs_chunk in (final_prompt | llm | StrOutputParser()).stream({
                "summaries": "\n\n".join(chunk_summaries_parts)
            }):
                print(fs_chunk, end="", flush=True)
                final_summary_parts.append(fs_chunk)
            print("\n--- End Final Summary Streaming ---")

            answer = f"**Document Summary (Multi-section)**\n\n{''.join(final_summary_parts)}"

        else:
            # Standard summarization for shorter documents
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful summarization assistant. Create a comprehensive yet concise summary."),
                ("human", "Summarize the following document, highlighting key points and main themes:\n\n{document}")
            ])

            summary_parts = []
            print("\n--- Document Summary (Streaming) ---")
            for s_chunk in (prompt | llm | StrOutputParser()).stream({"document": full_text[:12000]}): # Limit input context
                print(s_chunk, end="", flush=True)
                summary_parts.append(s_chunk)
            print("\n--- End Document Summary Streaming ---")
            answer = f"**Document Summary**\n\n{''.join(summary_parts)}"

        updated_history = state.get("chat_history", []) + [
            {"role": "human", "content": state['question']},
            {"role": "ai", "content": answer}
        ]

        return {
            **state,
            "answer": answer,
            "metadata": {**state.get("metadata", {}), "summary_type": "hierarchical" if len(full_text) > 15000 else "standard", "streamed_summary": True},
            "chat_history": updated_history
        }
    except Exception as e:
        logger.error(f"Error in summarize_full_doc: {str(e)}")
        return {**state, "error": f"Summarization failed: {str(e)}"}

# ---- 6. QA Generation ----
def generate_questions(state: AgentState) -> AgentState:
    """Generates diverse types of questions from the document content."""
    try:
        if state.get("error"):
            return state

        llm = get_vllm_llm(temperature=0.7, streaming=True)
        text = "\n\n".join([doc.page_content for doc in state["documents"]])

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert question generator for educational purposes.
            Create diverse types of questions including:
            - Factual recall questions
            - Analytical questions
            - Application questions
            - Critical thinking questions
            Format each question clearly with its type."""),
            ("human", "Generate 6-8 varied questions from this content:\n\n{text}")
        ])

        qa_parts = []
        print("\n--- QA Generation (Streaming) ---")
        for q_chunk in (prompt | llm | StrOutputParser()).stream({"text": text[:12000]}):
            print(q_chunk, end="", flush=True)
            qa_parts.append(q_chunk)
        print("\n--- End QA Generation Streaming ---")

        qa = "".join(qa_parts)

        updated_history = state.get("chat_history", []) + [
            {"role": "human", "content": state['question']},
            {"role": "ai", "content": f"**Generated Study Questions**\n\n{qa}"}
        ]

        return {
            **state,
            "answer": f"**Generated Study Questions**\n\n{qa}",
            "metadata": {**state.get("metadata", {}), "question_type": "educational", "streamed_qa": True},
            "chat_history": updated_history
        }
    except Exception as e:
        logger.error(f"Error in generate_questions: {str(e)}")
        return {**state, "error": f"Question generation failed: {str(e)}"}

# ---- 7. Document Analysis Node ----
def analyze_document(state: AgentState) -> AgentState:
    """Performs detailed analysis of the document content."""
    try:
        if state.get("error"):
            return state

        llm = get_vllm_llm(temperature=0.4, streaming=True)
        text = "\n\n".join([doc.page_content for doc in state["documents"]])

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert document analyst. Provide a structured analysis including:
            - Main themes and topics
            - Key arguments or points
            - Tone and style
            - Target audience
            - Strengths and limitations
            - Implications or conclusions"""),
            ("human", "Analyze this document in detail:\n\n{text}")
        ])

        analysis_parts = []
        print("\n--- Document Analysis (Streaming) ---")
        for a_chunk in (prompt | llm | StrOutputParser()).stream({"text": text[:12000]}):
            print(a_chunk, end="", flush=True)
            analysis_parts.append(a_chunk)
        print("\n--- End Document Analysis Streaming ---")

        analysis = "".join(analysis_parts)

        updated_history = state.get("chat_history", []) + [
            {"role": "human", "content": state['question']},
            {"role": "ai", "content": f"**Document Analysis**\n\n{analysis}"}
        ]

        return {
            **state,
            "answer": f"**Document Analysis**\n\n{analysis}",
            "metadata": {**state.get("metadata", {}), "analysis_type": "comprehensive", "streamed_analysis": True},
            "chat_history": updated_history
        }
    except Exception as e:
        logger.error(f"Error in analyze_document: {str(e)}")
        return {**state, "error": f"Analysis failed: {str(e)}"}

# ---- 8. Error Handling Node ----
def handle_error(state: AgentState) -> AgentState:
    """Handles errors gracefully and provides an error message."""
    error_msg = state.get("error", "Unknown error occurred")
    # No need to update chat history here, as it's an error response
    return {
        **state,
        "answer": f"I encountered an issue: {error_msg}\n\nPlease check your document path and try again."
    }

# ---- 9. Graph Builder ----
def build_graph():
    """Builds the LangGraph with defined nodes and conditional routing."""
    builder = StateGraph(AgentState)

    # Add all nodes
    builder.add_node("loader", load_and_index)
    builder.add_node("rag", rag_answer)
    builder.add_node("summary", summarize_full_doc)
    builder.add_node("qa_gen", generate_questions)
    builder.add_node("analysis", analyze_document)
    builder.add_node("error_handler", handle_error)

    # Define routing function that checks for errors first
    def safe_route(state: AgentState) -> str:
        if state.get("error"):
            return "error_handler"
        return route_question(state)

    # Add conditional edges from the loader node
    builder.add_conditional_edges("loader", safe_route, {
        "rag": "rag",
        "summary": "summary",
        "qa_gen": "qa_gen",
        "analysis": "analysis",
        "error_handler": "error_handler" # Fallback if route_question returns something unexpected or error
    })

    # Add edges from all processing nodes and error handler to END
    for node in ["rag", "summary", "qa_gen", "analysis", "error_handler"]:
        builder.add_edge(node, END)

    builder.set_entry_point("loader")
    return builder.compile()

# ---- 10. Agent Runner ----
def run_agent(
    doc_path: str,
    question: str,
    verbose: bool = True,
    initial_documents: Optional[List[Document]] = None,
    initial_vectorstore: Optional[object] = None,
    current_chat_history: Optional[List[dict]] = None
):
    """
    Runs the agent with the given question and document context.
    Pre-loaded documents/vectorstore can be passed to avoid re-indexing.
    Maintains and updates chat history.
    """
    try:
        graph = build_graph()

        if verbose:
            print(f"\n{'='*60}")
            print(f"Processing question: '{question}'")
            print(f"Document Path: {doc_path}")
            if current_chat_history:
                print("\n--- Current Chat History ---")
                print(format_chat_history(current_chat_history))
                print("--------------------------")
            print("\n(Note: LLM responses will stream to console during processing)\n")

        initial_state = {
            "question": question,
            "doc_path": doc_path,
            "documents": initial_documents,
            "vectorstore": initial_vectorstore,
            "answer": "",
            "route": "rag", # Initial route will be re-evaluated by safe_route
            "error": None,
            "metadata": {},
            "chat_history": current_chat_history if current_chat_history is not None else []
        }

        # The graph.invoke() call will block until the entire graph run is complete.
        # Internal LLM calls stream to console as they execute.
        result = graph.invoke(initial_state)

        if verbose:
            print(f"\n\n{'='*60}")
            print("\n--- Final Answer (Collected) ---")
            print(result['answer'])

            if result.get('metadata'):
                print(f"\n--- Metadata ---")
                for key, value in result['metadata'].items():
                    print(f"{key}: {value}")
            print(f"{'='*60}")

        return result

    except Exception as e:
        error_msg = f"Failed to run agent at the top level: {str(e)}"
        logger.error(error_msg)
        return {"answer": error_msg, "error": str(e), "chat_history": current_chat_history}