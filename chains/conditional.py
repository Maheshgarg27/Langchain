from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StructuredOutputParser
from langchain_anthropic import chatAnthropic
from langchain.schema.runnable import RunnableParallel,RuunnableBranch,RunnableLambda
from langchain_core.output_parsers import pydantic_output_parser
from pydantic import BaseModel,field
from typing import literal



load_dotenv()
model = ChatOpenAI()
parser = StructuredOutputParser()

class feedback(BaseModel):
    sentiment: literal['positive','negative']
    reason: str = field(description='reason for the sentiment')

paser2=pydantic_output_parser(pydantic_object=feedback)

prompt1 = PromptTemplate(
    template ='classeify as positive or negative {text},{instructions}',
    input_variables = ['text'],
    partial_variables = {
        'instructions': paser2.get_format_instructions()
    }


)

classifier_chain = prompt1 | model | parser

prompt2 = PromptTemplate(
    template='write a positive review on the topic {text}',
    input_variables = ['text'],
)
prompt3 = PromptTemplate(
    template='write a negative review on the topic {text}',
    input_variables = ['text'],
)
branch_chain = RuunnableBranch(
    (lambda x: x['sentiment'] == 'positive',prompt2 | model | parser),
    (lambda x: x['sentiment'] == 'negative',prompt3 | model | parser)
  ,RunnableLambda(lambda x: "unable to find")
)

chain=classifier_chain | branch_chain
chain.invoke({'text':'AI is the future of technology'});