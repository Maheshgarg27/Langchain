from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
import json


# Load model
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


prompt = """
Analyze this product review.

Return ONLY valid JSON in exactly this format:

{
    "summary": "short summary of the review",
    "sentiment": "positive, negative, or neutral"
}

Review:
The hardware is great, but the software feels bloated.
There are too many pre-installed apps that I can't remove.
The UI looks outdated compared to other brands.
Hoping for a software update to fix this.
"""


result = model.invoke(prompt)

print(result.content)