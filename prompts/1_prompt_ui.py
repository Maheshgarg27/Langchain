from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
import streamlit as st


st.title("LangChain Prompt UI")


@st.cache_resource
def load_model():

    llm = HuggingFacePipeline.from_model_id(
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        task="text-generation",
        model_kwargs={
            "torch_dtype": "float32"
        },
        pipeline_kwargs={
            "max_new_tokens": 30,
            "do_sample": False,
            "return_full_text": False
        }
    )

    return ChatHuggingFace(llm=llm)


model = load_model()


user_input = st.text_input("Enter your prompt")


if st.button("Submit"):

    with st.spinner("Generating..."):
        result = model.invoke(user_input)

    st.write(result.content)