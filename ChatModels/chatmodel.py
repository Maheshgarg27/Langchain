from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

# model = ChatGoogleGenerativeAI(
#     model="gemini-3.8-flash",
#     temperature =1.5,
#     max_output_tokens=10
# )
model = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    thinking_level="low"
)
result = model.invoke("Write a poem on cricket?")

print(result.content)