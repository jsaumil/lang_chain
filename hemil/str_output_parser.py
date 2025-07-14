from langchain_google_genai import GoogleGenerativeAI
# from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser 
# import os
# load_dotenv()


model = GoogleGenerativeAI(model='gemini-2.0-flash')
# 1st prompt -> detailed report
templet1 = PromptTemplate(
    template="Write a detailrs report on {topic}",
    input_variables=['topic']
)

# 2md prompt -> summary
templet2 = PromptTemplate(
    template="Write a 5 line summary on the following text. /n {text}",
    input_variables=['text']
)

# prompt1 = templet1.invoke({'topic':'black hole'})

# result = model.invoke(prompt1)

# prompt2 = templet2.invoke({'text':result})

# result1 = model.invoke(prompt2)

parser = StrOutputParser()

chain = templet1 | model| parser | templet2 | model | parser

result = chain.invoke({'topic':'black hole'})

print(result)