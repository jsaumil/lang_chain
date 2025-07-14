from langchain_community.chat_models import ChatOllama
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema.runnable import RunnableParallel

load_dotenv()

# model1 = ChatOllama(
#     base_url="https://6fe462576267.ngrok-free.app",
#     model="qwen3:0.6b",
#     strem = False,
#     think = False
# )

# Replace 'your_file.txt' with the path to your file
with open('Admission_process.txt', 'r', encoding='utf-8') as file:
    admission = file.read()

with open('gisd_courses.txt', 'r', encoding='utf-8') as file:
    courses = file.read()


model = ChatGoogleGenerativeAI(model='gemini-2.0-flash')

prompt1 = PromptTemplate(
    template='Extract details of courses from this text \n {text}',
    input_variables=['text']
)

prompt2 = PromptTemplate(
    template='Extract details of courses from this text \n {text1}',
    input_variables=['text1']
)

prompt3 = PromptTemplate(
    template='Compare all the courses from {t1} and {t2} is there any difference between these text',
    input_variables=['t1','t2']
)

parser = StrOutputParser()

paralle_chain = RunnableParallel({
    't1': prompt1 | model | parser,
    't2' : prompt2 | model | parser
})

merge_chain = prompt3 | model | parser

chain = paralle_chain | merge_chain


result = chain.invoke({"text":admission,"text1":courses})

print(result)

# chain.get_graph().print_ascii()


# BSc and BSc Computer application