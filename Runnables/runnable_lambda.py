from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableSequence, RunnableLambda, RunnableParallel, RunnablePassthrough

load_dotenv()

def word_count(text):
    return len(text.split())

prompt = PromptTemplate(
    template = 'Write a joke about {topic}',
    input_variables = ['topic']
)

model = ChatGoogleGenerativeAI(model = 'gemini-2.0-flash')

parser = StrOutputParser()

joke_gen_chain = RunnableSequence(prompt, model, parser)

parallel_chain = RunnableParallel({
    'joke':RunnableParallel(),
    'word_count': RunnableLambda(lambda x: len(x.split()))
})