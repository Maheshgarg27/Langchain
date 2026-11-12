from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langcahin_core.prompts import PromptTemplate
from langchain_core.output_parsers import StructuredOutputParser

load_dotenv()
prompt =PromptTemplate(
    template = '5 line on topic{topic}',
    input_variables = ['topic'],    
)

model=chatopenAI()
praser=structuredOutputParser()

chain=prompt|model|praser


result = chain.invoke({'topic':'AI'})
print(result.content)
chain.get_graph().print_ascii()