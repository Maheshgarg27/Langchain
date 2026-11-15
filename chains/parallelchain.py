from langchain_ophenai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StructuredOutputParser
from langchain_antropic import chatAnthropic
from langchain.schema.runnable import RunnableParallel

load_dotenv()

model = chatAnthropic()
model2 = ChatOpenAI()

prompt1 = PromptTemplate(
    template = '5 line on topic{topic}',
    input_variables = ['topic'],
)
prompt2 = PromptTemplate(
    template = '5 line summary on topic{text}',
    input_variables = ['text'],
)
prompt3 = PromptTemplate(
    template= 'merge the two text {text1} and {text2}',
    input_variables = ['text1','text2'],
)
parser = StructuredOutputParser()

parallel_chain = RunnableParallel(
  'cahin1': prompt1 | model | parser,
    'chain2': prompt2 | model2 | parser
)

merger_chain = prompt3 | model | parser

chain=parallel_chain | merger_chain

chain.invoke({'topic':'AI'})



