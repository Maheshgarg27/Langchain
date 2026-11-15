
from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate
from langchain_classic.output_parsers import StructuredOutputParser, ResponseSchema

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

load_dotenv()

# 1. Define the output schema
response_schemas = [
    ResponseSchema(
        name="content",
        description="Write exactly 5 lines about the given topic"
    )
]

parser = StructuredOutputParser.from_response_schemas(response_schemas)

# 2. Create the prompt
prompt = PromptTemplate(
    template="""
    Write exactly 5 lines on the topic: {topic}

    {format_instructions}
    """,
    input_variables=["topic"],
    partial_variables={
        "format_instructions": parser.get_format_instructions()
    }
)

# 3. Load Hugging Face model
llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    max_new_tokens=256,
    temperature=0.3,
    huggingfacehub_api_token=None
)

model = ChatHuggingFace(llm=llm)

# 4. Create the chain
chain = prompt | model | parser

# 5. Invoke the chain
result = chain.invoke({"topic": "AI"})

print(result)

# 6. Print chain graph
chain.get_graph().print_ascii()
