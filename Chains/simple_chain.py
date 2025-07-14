from langchain_community.chat_models import ChatOllama
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt = PromptTemplate(
    template='Generate  interesting facts about {topic}',
    input_variables=['topic']
)

model = ChatOllama(
    base_url="https://6fe462576267.ngrok-free.app",
    model="qwen3:0.6b",
    strem = False,
    think = False
)

parser = StrOutputParser()

chain = prompt | model | parser

result = chain.invoke({'topic':'cricket'})

print(result)

chain.get_graph().print_ascii()