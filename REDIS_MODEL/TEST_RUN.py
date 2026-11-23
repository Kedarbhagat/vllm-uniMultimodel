from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
import asyncio # For running the async function
import sys # For sys.stdout.write and flush

# --- Configuration ---
# Make sure these match your actual API Gateway URL and model name
API_GATEWAY_URL = os.getenv("API_GATEWAY_URL", "http://172.17.35.82:8082")
YOUR_MODEL_BASE_URL = f"{API_GATEWAY_URL}/v1"
YOUR_MODEL_NAME = os.getenv("VLLM_MODEL_NAME", "./meta-llama/Llama-3.1-8B-Instruct-awq")
YOUR_API_KEY = os.getenv("YOUR_API_KEY", "token-abc123") # Use dummy if not enforced

# 1. Instantiate the LLM for streaming
# IMPORTANT: Set streaming=True for stream/astream calls
llm_streaming = ChatOpenAI(
    model=YOUR_MODEL_NAME,
    openai_api_base=YOUR_MODEL_BASE_URL,
    openai_api_key=YOUR_API_KEY,
    temperature=0.7,
    max_tokens=300, # A reasonable max_tokens for a full response
    streaming=True # Key for streaming behavior
)
 
# 2. Define a simple Prompt Template
prompt = ChatPromptTemplate.from_template("A concise, high-level overview that explains the purpose of the joint project, its impact across three industries (biotechnology, financial modeling, and aerospace engineering), and the ethical implications of using AI to accelerate innovation in each domain.")

# 3. Define an Output Parser (to get plain text)
# StrOutputParser is suitable for streaming text
output_parser = StrOutputParser()

# 4. Create a LangChain Chain
# Prompt -> LLM -> Output Parser
chain_streaming = prompt | llm_streaming | output_parser

# 5. Define an async function to run the chain.astream call
async def test_chain_stream():
    print("--- Testing chain.astream() ---")
    animal = "a curious red panda"
    print(f"Asking for a streaming story about: {animal}")
    print("\n--- Story Chunks Received ---")

    full_story_content = ""
    try:
        # Astream the chain. This will send the request to your API Gateway
        # and yield chunks as they are received.
        async for chunk in chain_streaming.astream({"animal": animal}):
            sys.stdout.write(chunk)
            sys.stdout.flush() # Ensure the output is printed immediately
            full_story_content += chunk
        
        print("\n--- Streaming Complete ---")
        print(f"Total characters received: {len(full_story_content)}")
        # Optional: Add an assertion to check if the story contains expected words
        assert "red panda" in full_story_content.lower() or "curious" in full_story_content.lower(), \
            "The streamed story did not contain expected keywords."
        print("\nStreaming test PASSED.")

    except Exception as e:
        print(f"\n--- Streaming Test FAILED ---")
        print(f"An error occurred during streaming: {e}")
        # Print any partial content received before the error
        if full_story_content:
            print(f"\nPartial content received:\n{full_story_content}")
        print("\n--------------------------")
        sys.exit(1) # Exit with an error code

# Run the async test
if __name__ == "__main__":
    print(f"Using API Gateway: {API_GATEWAY_URL}")
    print(f"Model: {YOUR_MODEL_NAME}")
    print("\nInitializing...")
    asyncio.run(test_chain_stream())
    print("\nTest finished.")