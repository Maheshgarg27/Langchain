from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

llm = HuggingFacePipeline.from_model_id(
    model_id="Qwen/Qwen2.5-0.5B-Instruct",
    task="text-generation",
    pipeline_kwargs={
        "max_new_tokens": 50,
        "temperature": 0.5,
        "return_full_text": False
    }
)

model = ChatHuggingFace(llm=llm)

result = model.invoke("How many states are there in India?")

print(result.content)