from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
import asyncio # For running the async function

# --- Configuration ---
# Make sure these match your actual API Gateway URL and model name
API_GATEWAY_URL = os.getenv("API_GATEWAY_URL", "http://172.17.35.82:8082")
YOUR_MODEL_BASE_URL = f"{API_GATEWAY_URL}/v1"
YOUR_MODEL_NAME = os.getenv("VLLM_MODEL_NAME", "./meta-llama/Llama-3.1-8B-Instruct-awq")
YOUR_API_KEY = os.getenv("YOUR_API_KEY", "token-abc123") # Use dummy if not enforced

# 1. Instantiate the LLM for non-streaming
# IMPORTANT: Set streaming=False for invoke/ainvoke calls
llm_non_streaming = ChatOpenAI(
    model=YOUR_MODEL_NAME,
    openai_api_base=YOUR_MODEL_BASE_URL,
    openai_api_key=YOUR_API_KEY,
    temperature=0.7,
    max_tokens=200, # A reasonable max_tokens for a full response
    streaming=False # Key for non-streaming behavior
)

# 2. Define a simple Prompt Template
prompt = ChatPromptTemplate.from_template("Tell me a short story about {animal}.")

# 3. Define an Output Parser (to get plain text)
output_parser = StrOutputParser()

# 4. Create a LangChain Chain
# Prompt -> LLM -> Output Parser
chain = prompt | llm_non_streaming | output_parser

# 5. Define an async function to run the chain.invoke call
async def test_chain_invoke():
    print("--- Testing chain.invoke() ---")
    animal = "a magical cat"
    print(f"Asking for a story about: {animal}")

    # Invoke the chain. This will send the request to your API Gateway
    # and wait for the full response before returning.
    response = await chain.ainvoke({"animal": animal})

    print("\n--- Full Story Received ---")
    print(response)
    print("\n--------------------------")

# Run the async test
if __name__ == "__main__":
    print(f"Using API Gateway: {API_GATEWAY_URL}")
    print(f"Model: {YOUR_MODEL_NAME}")
    print("\nInitializing...")
    asyncio.run(test_chain_invoke())
    print("\nTest finished.")