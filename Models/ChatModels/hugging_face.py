from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
# import os


load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3-Reranker-4B",
    task="text-generation",
)

model = ChatHuggingFace(llm=llm)

result = model.invoke("What is the capital of India")

print(result)