from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StructuredOutputParser

load_dotenv()
promt1=PromptTemplate(
    template = '5 line on topic{topic}',
    input_variables = ['topic'],
)

promt2=PromptTemplate(
    template = '5 line summary on topic{text}',
    input_variables = ['text'],
)
model=ChatOpenAI()
parser=StructuredOutputParser()
chain =promt1|model|parser|promt2|model|parser

result = chain.invoke({'topic':'AI'})
print(result.content)
