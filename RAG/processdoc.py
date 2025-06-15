from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
import asyncio # Although not explicitly used for streaming output in this exact snippet, good to keep if you plan to use it later
import sys

 # Make sure this file is named document_processor.py
from wordloader import process_document
from vectorestore import build_vectorstore    # Make sure this file is named vectorestore.py

# --- Configuration ---
# Set these as environment variables or update directly
API_GATEWAY_URL = os.getenv("API_GATEWAY_URL", "http://172.17.35.82:8082")
YOUR_MODEL_BASE_URL = f"{API_GATEWAY_URL}/v1"
YOUR_MODEL_NAME = os.getenv("VLLM_MODEL_NAME", "./DeepSeek-Coder-V2-Lite-Instruct-awq")
YOUR_API_KEY = os.getenv("YOUR_API_KEY", "token-abc123") # Use dummy if not enforced

# 1. Instantiate the LLM for streaming (even if not explicitly streaming output to console here)
# IMPORTANT: Set streaming=True for stream/astream calls for models that support it
llm_streaming = ChatOpenAI(
    model=YOUR_MODEL_NAME,
    openai_api_base=YOUR_MODEL_BASE_URL,
    openai_api_key=YOUR_API_KEY,
    temperature=0.7, # Adjust creativity/randomness
    max_tokens=500,  # Max tokens for the LLM's response
    streaming=True
)

from langchain.chains import RetrievalQA

# --- Document Ingestion and Processing ---
# Define the path to your document
# Using raw string (r"...") is good for Windows paths to avoid issues with backslashes
document_path = r"C:\Users\STUDENT\Documents\vllm\RAG\Graduate Trainee - Technology- Aug24.pdf"

print(f"--- Step 1: Processing document from {document_path} ---")

# 1. Call your universal document processing function
if not os.path.exists(document_path):
    print(f"Error: Document not found at '{document_path}'. Please check the path.")
    sys.exit(1)

# You can pass additional kwargs that your process_document function accepts,
# e.g., max_sentences for chunking, ocr_threshold for PDFs.
documents = process_document(document_path, max_sentences=7, ocr_threshold=20)

if not documents:
    print(f"Error: No documents were successfully processed from '{document_path}'.")
    print("Please check your document_processor.py for errors and ensure Tesseract-OCR is installed and configured correctly if it's a scanned PDF.")
    sys.exit(1)

print(f"Successfully processed {len(documents)} chunks from the document.")

# --- Vector Store Building ---
print("\n--- Step 2: Building vector store (embedding and storing chunks) ---")
# 2. Build the vector store using your documents
# This function will handle embedding the text and storing it in Redis (as per vectorestore.py)
vectorstore = build_vectorstore(documents)

if not vectorstore:
    print("Error: Failed to build the vector store. Check your Redis connection and embedding model setup in vectorestore.py.")
    sys.exit(1)

print("Vector store built successfully.")

# --- Retriever Setup ---
# 3. Create a retriever from your vector store
# search_kwargs={"k": 3} means it will retrieve the top 3 most relevant documents
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
print("Retriever configured to fetch top 3 relevant documents.")

# --- RetrievalQA Chain Creation ---
print("\n--- Step 4: Creating RetrievalQA chain with LLM and Retriever ---")
# 4. Create the RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm_streaming,
    retriever=retriever,
    return_source_documents=True # This ensures you get the source chunks back for verification
)
print("RetrievalQA chain created.")

# --- Asking a Question ---
print("\n--- Step 5: Asking a question and getting an answer ---")
# 5. Ask a question relevant to your document's content
# IMPORTANT: Adjust this query to something that can be answered by the PDF content!
query = "What are the main responsibilities of a Graduate Trainee in Technology at IG Group, as described in the document?"
# Example of a query relevant to a job description PDF.
# Other examples:
# query = "What kind of projects will a Graduate Trainee work on?"
# query = "What company is described in the document?"
# query = "Where are the offices located?"


print(f"Query: \"{query}\"")

try:
    # Invoke the QA chain
    result = qa_chain({"query": query})

    # Print the answer and source documents
    print("\n--- Answer from LLM ---")
    print(result["result"])

    print("\n--- Retrieved Source Documents ---")
    if result["source_documents"]:
        for i, doc in enumerate(result["source_documents"]):
            print(f"\nSource Document {i+1}:")
            print(f"  Filename: {doc.metadata.get('filename', 'N/A')}")
            print(f"  Page: {doc.metadata.get('page', 'N/A')}") # Relevant for PDFs
            print(f"  Section: {doc.metadata.get('section', 'N/A')}") # Relevant for DOCX/DOC
            print(f"  Extraction Method: {doc.metadata.get('extraction_method', 'N/A')}")
            print(f"  Content Snippet (first 500 chars):\n---")
            print(doc.page_content[:500] + ("..." if len(doc.page_content) > 500 else ""))
            print("---\n")
    else:
        print("No relevant source documents were retrieved for this query.")

except Exception as e:
    print(f"\nAn error occurred during the QA chain execution: {e}")
    print("Please verify your LLM configuration, API Gateway URL, and ensure the vLLM model is running correctly.")
    print("Also check if Redis is active and accessible for the vector store.")