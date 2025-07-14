from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import  StructuredOutputParser, ResponseSchema
import os
load_dotenv()


model = ChatGoogleGenerativeAI(model='gemini-2.0-flash')



schema = [
    ResponseSchema(name = 'fact_1', description='fact_1 about the topic'),
    ResponseSchema(name = 'fact_2', description='fact_2 about the topic'),
    ResponseSchema(name = 'fact_3', description='fact_3 about the topic'),
]
parser = StructuredOutputParser.from_response_schemas(schema)

templete = PromptTemplate(
    template='Give 3 fact about the {topic} \n {format_instruction}',
    input_variables=['topic'],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)

# prompt = templete.invoke({'topic':'black hole'})

# result = model.invoke(prompt)

# final_result = parser.parse(result.content)
chain = templete | model | parser
result = chain.invoke({'topic':'black hole'})

print(result)