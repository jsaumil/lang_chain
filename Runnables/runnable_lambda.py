from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableLambda

load_dotenv()

def word_counter(text):
    return len(text.split())

runnable_word_counter = RunnableLambda(word_counter)

print(runnable_word_counter.invoke("This is a test sentence to count words."))