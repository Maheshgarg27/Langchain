from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.messages import HumanMessage, AIMessage


llm = HuggingFacePipeline.from_model_id(
    model_id="Qwen/Qwen2.5-0.5B-Instruct",
    task="text-generation",
    model_kwargs={
        "torch_dtype": "float32"
    },
    pipeline_kwargs={
        "max_new_tokens": 100,
        "do_sample": False,
        "return_full_text": False
    }
)

model = ChatHuggingFace(llm=llm)


chat_history = []


while True:

    user_input = input("You: ")

    if user_input.lower() == "exit":
        break

    # Add user message to history
    chat_history.append(
        HumanMessage(content=user_input)
    )

    # Send complete conversation history to model
    result = model.invoke(chat_history)

    # Add AI response to history
    chat_history.append(
        AIMessage(content=result.content)
    )

    print("AI:", result.content)


print("\nChat History:")
print(chat_history)