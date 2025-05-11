# Import your custom LLM class
from Custom_LLMwrapper import CustomFastAPILLM

# Import any LangChain components you need
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Initialize the LLM
llm = CustomFastAPILLM(
    api_url="http://172.17.25.83:8080/v1/chat/completions",
    model="./mistral-instruct-v0.2-awq",
    temperature=0.7
)

# Simple invocation
direct_result = llm.invoke("What is artificial intelligence?")
print("Direct result:", direct_result)

# Using modern chain pattern with RunnableSequence
prompt = PromptTemplate(
    input_variables=["query"],
    template="Please answer the following question: {query}"
)

# Create a chain using the pipe operator
chain = prompt | llm | StrOutputParser()

# Use invoke instead of run
chain_result = chain.invoke({"query": "Explain quantum computing in simple terms"})
print("Chain result:", chain_result)