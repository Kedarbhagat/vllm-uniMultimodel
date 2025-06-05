from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
import asyncio # For running the async function
import sys
from wordloader import process_document
from vectorestore import build_vectorstore # For sys.stdout.write and flush

# --- Configuration ---
# Make sure these match your actual API Gateway URL and model name
API_GATEWAY_URL = os.getenv("API_GATEWAY_URL", "http://172.17.35.82:8082")
YOUR_MODEL_BASE_URL = f"{API_GATEWAY_URL}/v1"
YOUR_MODEL_NAME = os.getenv("VLLM_MODEL_NAME", "./DeepSeek-Coder-V2-Lite-Instruct-awq")
YOUR_API_KEY = os.getenv("YOUR_API_KEY", "token-abc123") # Use dummy if not enforced

# 1. Instantiate the LLM for streaming
# IMPORTANT: Set streaming=True for stream/astream calls
llm_streaming = ChatOpenAI(
    model=YOUR_MODEL_NAME,
    openai_api_base=YOUR_MODEL_BASE_URL,
    openai_api_key=YOUR_API_KEY,
    temperature=0.7,
    max_tokens=300,
    streaming=True   # Disable streaming for testing
)



from langchain.chains import RetrievalQA


# 1. Ingest some example documents
documents = [
    "The Nile is the longest river in Africa.",
    "Mount Everest is the tallest mountain on Earth.",
    "Paris is the capital city of France."
]



documents=process_document("C:\Users\STUDENT\Documents\vllm\REDIS_MODEL\Graduate Trainee - Technology- Aug24.pdf")




# 2. Create vectorstore and retriever
vectorstore = build_vectorstore(documents)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# 3. Create RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm_streaming,
    retriever=retriever,
    return_source_documents=True
)

# 4. Ask a question
query = "What is the longest river in Africa?"
result = qa_chain(query)

# Print final result
print("Answer:", result["result"])
print("Sources:")
for doc in result["source_documents"]:
    print("-", doc.page_content)
