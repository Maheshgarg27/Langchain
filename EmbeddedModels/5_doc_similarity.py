from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

embedding = HuggingFaceEmbeddings( model_name="sentence-transformers/all-MiniLM-L6-v2")

documents = [
    "Virat Kohli is a famous Indian cricketer known for his batting.",
    "MS Dhoni is a former Indian cricket captain and wicketkeeper.",
    "Sachin Tendulkar is one of the greatest batsmen in cricket history.",
    "Rohit Sharma is an Indian cricketer famous for his batting and captaincy.",
    "Jasprit Bumrah is an Indian fast bowler known for his unique bowling action."
]

query = "Tell me about Virat Kohli"

doc_embeddings = embedding.embed_documents(documents)
query_embedding = embedding.embed_query(query)

scores = cosine_similarity([query_embedding],doc_embeddings)[0]

index, score = sorted(list(enumerate(scores)),key=lambda x: x[1])[-1]


print(query)
print(documents[index])
print("Similarity score is:", score)