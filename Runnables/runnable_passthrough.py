from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableParallel, RunnableSequence, RunnablePassthrough

load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-2.0-flash')

passthrough = RunnablePassthrough()

print(passthrough.invoke('This is a passthrough test.'))

prompt1 = PromptTemplate(
    template = 'Write a joke about {topic}',
    input_variables = ['topic']
)

model = ChatGoogleGenerativeAI(model = 'gemini-2.0-flash')

parser = StrOutputParser()

prompt2 = PromptTemplate(
    template= 'Explain the following joke - {text}',
    input_variables = ['text']
)

joke_gen_chain = RunnableSequence(prompt1, model, parser)

parallel_chain = RunnableParallel({
    'joke': RunnableParallel(),
    'explanatoin':RunnableParallel(prompt2, model, parser)
})

final_chain = RunnableSequence(joke_gen_chain, parallel_chain)

print(final_chain.invoke({'topic': 'AI'}))