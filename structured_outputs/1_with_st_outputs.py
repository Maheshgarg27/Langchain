from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from typing import TypedDict


# Load Hugging Face model
llm = HuggingFacePipeline.from_model_id(
    model_id="Qwen/Qwen2.5-0.5B-Instruct",
    task="text-generation",
    pipeline_kwargs={
        "max_new_tokens": 100,
        "do_sample": False,
        "return_full_text": False
    }
)

model = ChatHuggingFace(llm=llm)


# Schema
class Review(TypedDict):
    summary: str
    sentiment: str


# Structured output
structured_model = model.with_structured_output(Review)


# Input
result = structured_model.invoke(
    """The hardware is great, but the software feels bloated.
    There are too many pre-installed apps that I can't remove.
    Also, the UI looks outdated compared to other brands.
    Hoping for a software update to fix this."""
)

print(result)