from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model='google-1.5-flash')

prompt = PromptTemplate(
    template='Write a summary for the following doc -\n {doc}',
    input_variables=['poem']
)

parser = StrOutputParser()

loader = TextLoader("Admission_process.txt",encoding='utf-8')

docs = loader.load()

print(type(docs))

print(len(docs))
print(docs[0].page_content)
print(docs[0].metadata)

chain = prompt | model | parser

print(chain.invoke({'doc':docs[0].page_content}))