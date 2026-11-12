from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser


# 1. Load Hugging Face model
print("Loading model...")

llm = HuggingFacePipeline.from_model_id(
    model_id="Qwen/Qwen2.5-0.5B-Instruct",
    task="text-generation",
    pipeline_kwargs={
        "max_new_tokens": 150,
        "do_sample": False,
        "return_full_text": False
    }
)

model = ChatHuggingFace(llm=llm)

print("Model loaded successfully!")


# 2. Create JSON Output Parser
parser = JsonOutputParser()


# 3. Create Prompt Template
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an expert product review analyzer.

        Analyze the given review and return:
        1. A short summary of the review.
        2. Its sentiment (positive, negative, or neutral).

        Return ONLY a valid JSON object.
        Do not include markdown or explanations.

        Required JSON format:
        {{
            "summary": "Short summary here",
            "sentiment": "positive/negative/neutral"
        }}
        """
    ),
    (
        "human",
        "Analyze this review:\n{review}"
    )
])


# 4. Create Structured Output Chain
structured_model = prompt | model | parser

print("Chain created successfully!")


# 5. Input Review
review = """
The hardware is great, but the software feels bloated.
There are too many pre-installed apps that I can't remove.
Also, the UI looks outdated compared to other brands.
Hoping for a software update to fix this.
"""


# 6. Invoke Model
print("\nInvoking model...")

try:
    result = structured_model.invoke({
        "review": review
    })

    print("\nModel execution completed!")

    # 7. Display Result
    print("\nStructured Output:")
    print(result)

    print("\nSummary:", result["summary"])
    print("Sentiment:", result["sentiment"])

except Exception as e:
    print("\nError occurred:")
    print(e)