from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()

embedding = OpenAIEmbeddings(model='text-embedding-3-large',dimensions=32)
documents =[
    "delhi is capital of india"
    "chandigarh is capital of haryana and punjab"
]
result= embedding.embed_documents(documents)
print(str(result))