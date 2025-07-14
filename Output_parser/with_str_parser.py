import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
load_dotenv()

# model = ChatOpenAI(
#     base_url="https://openrouter.ai/api/v1",
#     model_name="qwen/qwen3-14b:free",
#     openai_api_key=os.getenv("OPENROUTER_API_KEY"),
# )

model = ChatGoogleGenerativeAI(model='gemini-2.0-flash')

# 1st prompt
template1 = PromptTemplate(
    template='Write a detailed report on {topic}',
    input_variables=['topic']
)

# 2nd prompt ->
template2 = PromptTemplate(
    template='Write a 5 line summary on the following text./ {text}',
    input_variables=['text']
)

prompt1 = template1.invoke({'topic':'black hole'})
result = model.invoke(prompt1)

prompt2 = template2.invoke({'text':result.content})
result1 = model.invoke(prompt2)

print(result1.content)