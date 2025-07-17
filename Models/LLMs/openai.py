# from langchain_openai import OpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash')

result = llm.invoke("i completeed my 12th science what should i do next")

print(result.content)