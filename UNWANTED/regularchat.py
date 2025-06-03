from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from Custom_LLMwrapper import CustomFastAPILLM

models = {
    0: "microsoft/Phi-4-mini-instruct",
    1: "meta-llama/Llama-3.1-8B-Instruct",
    2: "./mistral-instruct-v0.2-awq"
}

user_prompt = input("Enter prompt: ")

model_index = int(input("Enter the model index to chat (0, 1, or 2): "))

selected_model = models.get(model_index, None)

if selected_model is None:
    print("Invalid model index. Please enter 0, 1, or 2.")
else:
    llm = CustomFastAPILLM(model=selected_model)

    final_prompt = PromptTemplate(
        input_variables=["user_input"],
        template="You are an AI assistant for question answering tasks. "
                 "Answer the following user query based on your knowledge: "
                 "---------------{user_input}"
    )

    chain = final_prompt | llm | StrOutputParser()

    # Streaming the response
    try:
        response = chain.invoke({"user_input": user_prompt})
        print("AI Response (streaming):")
        print(response)
    except Exception as e:
        print(f"Error: {e}")
