from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
# import os
# os.environ["HUGGINGFACE_HUB_TOKEN"] = "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"

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-2-7b-chat-hf",
    task="text-generation",
)

model = ChatHuggingFace(llm=llm)

result = model.invoke("What is the capital of India")

print(result)