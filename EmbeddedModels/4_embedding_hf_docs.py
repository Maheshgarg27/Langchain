from langchain_huggingface import HuggingFaceEmbeddings

embedding=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

document=[
"Chandigarh is the capital of Haryana",
"Delhi is the capital of India."
]
vector=embedding.embed_documents(document)
print(str(vector))