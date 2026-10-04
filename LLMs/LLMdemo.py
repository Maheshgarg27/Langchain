from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash"
)

result = llm.invoke("Who is the Prime Minister of India?")

print(result.content[0]["text"])