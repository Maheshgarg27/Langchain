from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline


# -----------------------------
# 1. Load Hugging Face model
# -----------------------------

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


# -----------------------------
# 2. Chat prompt template
# -----------------------------

chat_template = ChatPromptTemplate.from_messages([
    (
        "system","You are a helpful customer support agent."
    ),

    MessagesPlaceholder(
        variable_name="chat_history"
    ),
    (
        "human",  "{query}"
    )
])


# -----------------------------
# 3. Load chat history from file
# -----------------------------

chat_history = []

with open("prompts/chat_history.txt", "r", encoding="utf-8") as f:

    for line in f:
        line = line.strip()

        if line.startswith("HumanMessage(content="):
            content = line[len("HumanMessage(content="):-2]
            chat_history.append(HumanMessage(content=content))

        elif line.startswith("AIMessage(content="):
            content = line[len("AIMessage(content="):-2]
            chat_history.append(AIMessage(content=content))


# -----------------------------
# 4. Create prompt
# -----------------------------

prompt = chat_template.invoke({
    "chat_history": chat_history,
    "query": "How long will my refund take?"
})


# -----------------------------
# 5. Send prompt to HF model
# -----------------------------

result = model.invoke(prompt)


# -----------------------------
# 6. Print response
# -----------------------------

print("AI:", result.content)