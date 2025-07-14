from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser 
import os
load_dotenv()


model = ChatGoogleGenerativeAI(model='gemini-2.0-flash')

parser = JsonOutputParser()

template = PromptTemplate(
    template= "Give me the name, age and city of a fictional person \n{format_instruction}",
    input_variables=[],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)

# prompt = template.format()

# result = model.invoke(prompt)

# final_result = parser.parse(result.content)

chain = template | model | parser

result = chain.invoke({})

print(result)