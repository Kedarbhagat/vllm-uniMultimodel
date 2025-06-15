# new_agen.py - The fully fixed version for true streaming

from typing import TypedDict, List, Generator, Dict, Any, Optional
from langgraph.graph import StateGraph, END
from langchain.schema import BaseMessage, HumanMessage, AIMessage, Document
from langchain.prompts import ChatPromptTemplate
import os

from langchain_openai import ChatOpenAI
from wordloader import process_document # Your custom loader

# --- Environment variables and LLM initialization ---
API_GATEWAY_URL = os.getenv("API_GATEWAY_URL", "http://172.17.35.82:8082")
YOUR_MODEL_BASE_URL = f"{API_GATEWAY_URL}/v1"
YOUR_MODEL_NAME = os.getenv("VLLM_MODEL_NAME", "./DeepSeek-Coder-V2-Lite-Instruct-awq")
YOUR_API_KEY = os.getenv("YOUR_API_KEY", "token-abc123") # Replace with real key if needed

# Define two LLM instances with different max_tokens
llm_chat = ChatOpenAI(
    model=YOUR_MODEL_NAME,
    openai_api_base=YOUR_MODEL_BASE_URL,
    openai_api_key=YOUR_API_KEY,
    temperature=0.7,
    max_tokens=5000, # Increased for flexibility
    streaming=True, # Essential for streaming
)

llm_summary = ChatOpenAI(
    model=YOUR_MODEL_NAME,
    openai_api_base=YOUR_MODEL_BASE_URL,
    openai_api_key=YOUR_API_KEY,
    temperature=0.7,
    max_tokens=6000, # Increased for flexibility
    streaming=True, # Essential for streaming
)


# --- AgentState Definition ---
class AgentState(TypedDict):
    messages: List[BaseMessage]
    question: str
    context: str
    answer: str # This will accumulate the full answer for non-streaming invokes
    chat_history: List[BaseMessage] # History passed from the caller
    needs_retrieval: bool
    needs_full_document_processing: bool
    full_document_content: str
    document_path: Optional[str]
    documents: Optional[List[Document]] # Added to pass pre-chunked documents


# --- LangGraphAgenticRAG Class ---
class LangGraphAgenticRAG:
    def __init__(self, llm_chat: ChatOpenAI, llm_summary: ChatOpenAI, vectorstore, embedding_model):
        self.llm_chat = llm_chat
        self.llm_summary = llm_summary
        self.vectorstore = vectorstore
        self.embedding_model = embedding_model
        self.last_used_full_document: bool = False

        # Build the LangGraph workflow
        self.workflow = self._build_workflow()
        self.app = self.workflow.compile() # Compiled graph for non-streaming execution

    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("analyze_question", self.analyze_question_node)
        workflow.add_node("retrieve_context", self.retrieve_context_node)
        workflow.add_node("process_full_document", self.process_full_document_node)
        # This node is now designed to be a generator itself
        workflow.add_node("generate_answer", self.generate_answer_node)

        # Add edges
        workflow.set_entry_point("analyze_question")

        workflow.add_conditional_edges(
            "analyze_question",
            self.route_question,
            {
                "retrieve": "retrieve_context",
                "direct": "generate_answer", # Direct path skips retrieval/document processing
                "process_full_document": "process_full_document"
            }
        )

        workflow.add_edge("retrieve_context", "generate_answer")
        workflow.add_edge("process_full_document", "generate_answer")
        workflow.add_edge("generate_answer", END) # End of the graph

        return workflow

    def analyze_question_node(self, state: AgentState) -> AgentState:
        """Node to analyze if retrieval, full document processing, or direct answer is needed."""
        # Use state["chat_history"] directly as it's passed into the graph state
        chat_hist_str = "\n".join(
            [
                f"{'Human' if isinstance(msg, HumanMessage) else 'AI'}: {msg.content}"
                for msg in state["chat_history"][-6:] # Using last 6 messages from the passed history
            ]
        )

        analysis_prompt = ChatPromptTemplate.from_template(
            """
You are an intelligent router. Based on the following user question and chat history,
determine the appropriate action.

Use 'RETRIEVE' for most user inputs unless it is strictly necessary to summarize or analyze an external document.
Always try to use 'RETRIEVE' first. For topics like "explain a topic," use retrieval.

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
        langchain_docs = state.get("documents") # Prefer pre-chunked documents

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

    # --- THE KEY FIX: generate_answer_node now handles streaming ---
    # ... (rest of the file remains the same until generate_answer_node)

    def generate_answer_node(self, state: AgentState) -> Generator[AgentState, None, AgentState]:
        """
        Node to generate the final answer. When the graph is streamed, this node
        will yield tokens. When the graph is invoked (non-streaming), it will
        accumulate all tokens and return the final state.
        """
        print("Executing generate_answer_node (streaming enabled)...")
        
        chat_hist_str = "\n".join(
            [
                f"{'Human' if isinstance(msg, HumanMessage) else 'Assistant'}: {msg.content}"
                for msg in state["chat_history"][-8:]
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
            else:
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
        else:
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
        print(f"Using LLM: {'llm_summary' if is_full_document_task else 'llm_chat'} for generation.")

        full_answer_accumulator = ""
        skip_tokens = False

        for chunk in llm_to_use.stream(messages):
            content = chunk.content if hasattr(chunk, 'content') else chunk.get('content', '')
            if not content:
                continue

            if any(pattern in content for pattern in ['<｜tool ', '｜tool ', 'function<｜tool sep｜>']):
                skip_tokens = True
                continue

            if skip_tokens:
                if not any(pattern in content for pattern in ['<｜', '｜>']):
                    skip_tokens = False
                continue
            
            full_answer_accumulator += content
            # Yield a dictionary that matches the AgentState structure
            # This ensures that the 'answer' field is updated incrementally
            yield {"answer": content}
        
        # This final return value is crucial when `workflow.invoke()` is called (non-streaming path)
        # It also ensures the final state of the graph has the complete answer, even for streaming.
        state["answer"] = full_answer_accumulator
        print(f"LLM generation finished in generate_answer_node. Final answer length: {len(full_answer_accumulator)}")
        return state # LangGraph captures this final state


    def stream_chat(self, question: str, chat_history: List[BaseMessage],
                    document_path: Optional[str] = None, documents: Optional[List[Document]] = None) -> Generator[str, None, None]:
        """
        Stream the chat response token by token using LangGraph workflow's stream method.
        """
        print(f"Initiating stream_chat for question: '{question}' via LangGraph stream...")
        initial_state: AgentState = {
            "messages": [HumanMessage(content=question)],
            "question": question,
            "context": "",
            "answer": "", # Will be populated by streaming
            "chat_history": chat_history, # Pass the history to the initial state
            "needs_retrieval": False,
            "needs_full_document_processing": False,
            "full_document_content": "",
            "document_path": document_path,
            "documents": documents,
        }

        accumulated_streamed_content = ""

        # FIX: Use self.app.stream, not self.workflow.stream
        for state_chunk in self.app.stream(initial_state):
            if "generate_answer" in state_chunk:
                token = state_chunk["generate_answer"].get("answer", "")
                if token:
                    accumulated_streamed_content += token
                    print(f"Yielding token: {token!r}")  # <-- Add this debug print
                    yield token

            if "analyze_question" in state_chunk:
                self.last_used_full_document = state_chunk["analyze_question"].get("needs_full_document_processing", False)

        print("LangGraph stream finished. Caller is responsible for chat history update.")
        # Note: We are not returning `accumulated_streamed_content` directly from `stream_chat`
        # as it's a generator. The FastAPI endpoint consuming this generator will accumulate.


    def chat(self, question: str, chat_history: List[BaseMessage],
             document_path: Optional[str] = None, documents: Optional[List[Document]] = None) -> str:
        """
        Non-streaming chat method using LangGraph workflow.
        """
        print(f"Initiating chat (non-streaming) for question: '{question}'...")
        initial_state: AgentState = {
            "messages": [HumanMessage(content=question)],
            "question": question,
            "context": "",
            "answer": "", # Will be populated by invoke
            "chat_history": chat_history, # Pass the history to the initial state
            "needs_retrieval": False,
            "needs_full_document_processing": False,
            "full_document_content": "",
            "document_path": document_path,
            "documents": documents,
        }

        # Run the compiled graph for non-streaming
        final_state = self.app.invoke(initial_state)
        print(f"LangGraph invoke finished. Final answer (non-streaming): {final_state['answer'][:100]}...")

        # Update the flag based on the workflow's decision
        self.last_used_full_document = final_state.get("needs_full_document_processing", False)

        print("Chat finished. Caller is responsible for chat history update.")

        # The 'answer' field will contain the full accumulated answer from generate_answer_node
        return final_state["answer"]


# -------- USAGE EXAMPLE (for testing the agent in isolation) --------
if __name__ == "__main__":
    from vectorestore import build_vectorstore # Ensure you have this implemented properly
    from embeddingmodel import MiniLMEmbeddings # Ensure this exists

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

    print("--- Streaming Answer (first turn) ---")
    question1 = "What is the capital of France?"
    full_response1 = ""
    for chunk in rag_agent.stream_chat(question1, chat_history=current_chat_history_for_test):
        full_response1 += chunk
        print(chunk, end="", flush=True)
    current_chat_history_for_test.append(HumanMessage(content=question1))
    current_chat_history_for_test.append(AIMessage(content=full_response1))
    print(f"\nUpdated chat history length: {len(current_chat_history_for_test)}")
    print("-" * 30)

    print("--- Streaming Follow-up Answer (second turn) ---")
    question2 = "And what is it famous for?"
    full_response2 = ""
    for chunk in rag_agent.stream_chat(question2, chat_history=current_chat_history_for_test):
        full_response2 += chunk
        print(chunk, end="", flush=True)
    current_chat_history_for_test.append(HumanMessage(content=question2))
    current_chat_history_for_test.append(AIMessage(content=full_response2))
    print(f"\nUpdated chat history length: {len(current_chat_history_for_test)}")
    print("-" * 30)

    print("--- Non-streaming Answer (third turn) ---")
    question3 = "Tell me about the Amazon River."
    response3 = rag_agent.chat(question3, chat_history=current_chat_history_for_test)
    print(response3)
    current_chat_history_for_test.append(HumanMessage(content=question3))
    current_chat_history_for_test.append(AIMessage(content=response3))
    print(f"\nUpdated chat history length: {len(current_chat_history_for_test)}")
    print("-" * 30)

    print("\n--- Simulating Document Summary (if you have a real document path) ---")
    # For this to work, you need a valid path to a PDF/DOCX that your `process_document` can handle.
    # Replace with a path to a small test document if you want to test this.
    test_document_path = r"C:\Users\STUDENT\Documents\vllm\RAG\nces (1).pdf" # Adjust this path

    if os.path.exists(test_document_path):
        print(f"Attempting to process document: {test_document_path}")
        # When processing a document, you typically create a new vectorstore for it
        # or update the existing one in your FastAPI app's /upload-document endpoint.
        # Here, we're just testing the agent's ability to handle the document input.
        processed_docs_for_test = process_document(test_document_path)
        
        question_doc_summary = "Summarize this document for me in 5 key points."
        full_response_doc_summary = ""
        # IMPORTANT: Passing the document_path/documents is crucial here
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



        #####################################################################################################



        # main.py - The fully fixed FastAPI application for streaming

from fastapi import FastAPI, HTTPException, UploadFile, File, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any, Generator
import os
import uuid
from tempfile import NamedTemporaryFile
from wordloader import process_document
from embeddingmodel import MiniLMEmbeddings
from vectorestore import build_vectorstore
from langchain.schema import Document, HumanMessage, AIMessage, BaseMessage
import json
import uvicorn
import logging
from fastapi.responses import StreamingResponse

# Assuming your LangGraphAgenticRAG class is in 'new_agen.py'
# Make sure new_agen.py also includes the llm_chat and llm_summary definitions
from new_agen import LangGraphAgenticRAG, llm_chat, llm_summary

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(
    title="LangGraph Agentic RAG API",
    description="API for document processing and conversational RAG system with session management",
    version="0.2.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances for the RAG agent and document storage
# The RAG agent itself is now stateless regarding chat history
global_rag_agent_instance: Optional[LangGraphAgenticRAG] = None
# This will store processed documents mapped by their ID
document_store: Dict[str, Dict[str, Any]] = {}

# In-memory store for chat history per session/user
# For production, replace this with a proper database (Redis, PostgreSQL, etc.)
session_chat_history: Dict[str, List[BaseMessage]] = {}

# --- Pydantic Models ---
class ChatRequest(BaseModel):
    question: str
    document_id: Optional[str] = None
    session_id: Optional[str] = None # Optional session ID for continuation
    stream: bool = False # Flag to indicate streaming request

class DocumentUploadResponse(BaseModel):
    document_id: str
    filename: str
    status: str
    metadata: Optional[Dict[str, Any]] = None

# For streaming, the client will parse individual tokens.
# For non-streaming, we still use ChatResponse.
class ChatResponse(BaseModel):
    answer: str
    document_id: Optional[str] = None
    context_used: Optional[str] = None
    session_id: Optional[str] = None # Return session_id to client

class Message(BaseModel):
    type: str
    content: str

class ChatHistoryResponse(BaseModel):
    session_id: str
    history: List[Message]

# --- FastAPI Event Handlers ---
@app.on_event("startup")
async def startup_event():
    """
    Initialize a SINGLE RAG agent instance.
    This agent instance does NOT manage chat history internally.
    It will receive chat history as a parameter for each request.
    """
    global global_rag_agent_instance
    logger.info("Initializing RAG agent instance...")
    embedding_model = MiniLMEmbeddings()
    # Provide an initial empty vectorstore or a shared one if relevant
    # The vectorstore will be updated dynamically upon document upload.
    empty_docs = [Document(page_content="")]
    vectorstore = build_vectorstore(empty_docs)
    global_rag_agent_instance = LangGraphAgenticRAG(llm_chat, llm_summary, vectorstore, embedding_model)
    logger.info("RAG agent initialized.")

# --- API Endpoints ---

@app.post("/upload-document/", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """Upload and process a document for later use in the RAG system."""
    global global_rag_agent_instance
    if not global_rag_agent_instance:
        raise HTTPException(status_code=500, detail="RAG agent not initialized.")

    try:
        # Create a temporary file to save the uploaded content
        file_ext = os.path.splitext(file.filename)[1]
        with NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
            contents = await file.read()
            tmp_file.write(contents)
            tmp_file_path = tmp_file.name
            
        logger.info(f"Processing document: {file.filename} saved to {tmp_file_path}")
        documents = process_document(tmp_file_path) # Your custom loader
            
        if not documents:
            raise ValueError("No content extracted from document.")

        doc_id = str(uuid.uuid4())

        # Store document information
        document_store[doc_id] = {
            "path": tmp_file_path,
            "filename": file.filename,
            "documents": documents # Store pre-chunked LangChain Documents
        }

        # Update the global RAG agent's vectorstore with the new document's chunks.
        # This design choice means the vectorstore always reflects the LAST uploaded document.
        # For multiple documents or per-user document contexts, a more advanced vectorstore
        # management (e.g., separate vectorstores per document, or a combined one) would be needed.
        global_rag_agent_instance.vectorstore = build_vectorstore(documents)
        logger.info(f"Document {doc_id} processed and vectorstore updated.")

        return {
            "document_id": doc_id,
            "filename": file.filename,
            "status": "processed",
            "metadata": {"pages": len(documents)} if documents else None
        }
    except Exception as e:
        logger.error(f"Document processing failed for {file.filename}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Document processing failed: {str(e)}")
    finally:
        # Optionally, remove the temporary file if you're not storing it long-term
        # For now, we'll keep it for the document_store's 'path' reference.
        # In a real app, you might move it to a persistent storage and store that path.
        pass

@app.post("/chat/")
async def chat_endpoint(request: ChatRequest):
    """Handle both streaming and non-streaming chat requests."""
    if global_rag_agent_instance is None:
        raise HTTPException(status_code=500, detail="RAG agent not initialized.")

    # Get or create session ID
    session_id = request.session_id if request.session_id else str(uuid.uuid4())
    # Retrieve current chat history for this session, defaulting to empty list
    current_chat_history = session_chat_history.setdefault(session_id, [])
    logger.info(f"Chat request for session_id: {session_id}, stream: {request.stream}, question: '{request.question[:50]}'")

    try:
        document_path = None
        documents_for_rag = None # LangChain Document chunks

        if request.document_id:
            if request.document_id not in document_store:
                raise HTTPException(status_code=404, detail="Document not found.")
            doc_info = document_store[request.document_id]
            document_path = doc_info["path"]
            documents_for_rag = doc_info["documents"] # Pass the pre-chunked documents

        # Add the human's message to history *before* calling the agent
        current_chat_history.append(HumanMessage(content=request.question))

        if request.stream:
            from typing import AsyncGenerator
            async def generate_stream() -> AsyncGenerator[str, None]:
                full_response_content = ""
                try:
                    # Call the agent's stream_chat, which is now a generator
                    for token in global_rag_agent_instance.stream_chat(
                        question=request.question,
                        chat_history=current_chat_history, # Pass the history to the agent
                        document_path=document_path,
                        documents=documents_for_rag
                    ):
                        full_response_content += token
                        # Yield each token as a Server-Sent Event (SSE)
                        # We are sending raw tokens, client needs to concatenate
                        yield f"data: {json.dumps({'token': token})}\n\n"
                except Exception as e:
                    logger.error(f"Error during streaming for session {session_id}: {e}", exc_info=True)
                    # Send an error event to the client
                    yield f"data: {json.dumps({'error': str(e)})}\n\n"
                finally:
                    # After the stream finishes (or errors), update the session's chat history
                    # with the full accumulated AI response
                    if full_response_content:
                        current_chat_history.append(AIMessage(content=full_response_content))
                        # Truncate history to avoid excessive memory usage
                        if len(current_chat_history) > 20: # Keep last 10 turns (question + answer)
                            session_chat_history[session_id] = current_chat_history[-20:]
                        logger.info(f"Stream finished for session {session_id}. History updated. Answer: {full_response_content[:50]}...")
                    else:
                        logger.warning(f"Stream finished for session {session_id} but no content was generated.")
                    yield "data: [DONE]\n\n" # Signal end of stream to client

            return StreamingResponse(generate_stream(), media_type="text/event-stream")
        else:
            # For non-streaming, get the full answer at once
            answer = global_rag_agent_instance.chat(
                question=request.question,
                chat_history=current_chat_history, # Pass the history to the agent
                document_path=document_path,
                documents=documents_for_rag
            )

            # Update the session's chat history for non-streaming
            current_chat_history.append(AIMessage(content=answer))
            
            # Truncate history
            if len(current_chat_history) > 20:
                session_chat_history[session_id] = current_chat_history[-20:]
                
            logger.info(f"Non-streaming chat finished for session {session_id}. History updated. Answer: {answer[:50]}...")

            context_used_type = "full_document" if global_rag_agent_instance.last_used_full_document else "retrieved_context_or_direct"

            return ChatResponse(
                answer=answer,
                document_id=request.document_id,
                context_used=context_used_type,
                session_id=session_id # Return session_id to client
            )
    except HTTPException: # Re-raise FastAPI HTTPExceptions
        raise
    except Exception as e:
        logger.error(f"Chat processing failed for session {session_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {str(e)}")

@app.get("/documents/")
async def list_documents():
    """List all uploaded documents."""
    return [
        {
            "document_id": doc_id,
            "filename": info["filename"],
            "pages": len(info.get("documents", []))
        }
        for doc_id, info in document_store.items()
    ]

@app.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """Delete an uploaded document and its temporary file."""
    if document_id not in document_store:
        raise HTTPException(status_code=404, detail="Document not found.")

    doc_info = document_store.pop(document_id)
    try:
        # Attempt to delete the temporary file associated with the document
        if os.path.exists(doc_info["path"]) and os.path.isfile(doc_info["path"]):
            os.remove(doc_info["path"])
            logger.info(f"Deleted temporary file for document {document_id}: {doc_info['path']}")
    except Exception as e:
        logger.warning(f"Error deleting temporary file for document {document_id}: {e}")

    logger.info(f"Document {document_id} deleted from store.")
    return {"status": "deleted", "document_id": document_id}

@app.get("/chat-history/{session_id}", response_model=ChatHistoryResponse)
async def get_chat_history(session_id: str):
    """Retrieve chat history for a given session ID."""
    history = session_chat_history.get(session_id, [])
    # Convert BaseMessage objects to a serializable format for response
    serializable_history = [
        Message(type=msg.type, content=msg.content) for msg in history
    ]
    logger.info(f"Retrieved history for session {session_id}. Messages: {len(history)}")
    return ChatHistoryResponse(session_id=session_id, history=serializable_history)

@app.delete("/chat-history/{session_id}")
async def clear_chat_history(session_id: str):
    """Clear chat history for a given session ID."""
    if session_id in session_chat_history:
        del session_chat_history[session_id]
        logger.info(f"Chat history cleared for session {session_id}.")
        return {"status": "cleared", "session_id": session_id}
    logger.warning(f"Attempted to clear non-existent session history: {session_id}")
    raise HTTPException(status_code=404, detail="Session ID not found.")

# --- Run the Application ---
if __name__ == "__main__":
    logger.info("Starting FastAPI application...")
    uvicorn.run(app, host="0.0.0.0", port=9000)