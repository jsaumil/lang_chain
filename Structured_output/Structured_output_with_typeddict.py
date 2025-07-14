import os
from dotenv import load_dotenv
# from langchain_openai import ChatOpenAI
from typing import TypedDict, Annotated, Optional, Literal
from langchain_google_genai import ChatGoogleGenerativeAI
load_dotenv()

# model = ChatOpenAI(
#     base_url="https://openrouter.ai/api/v1",
#     model_name="qwen/qwen3-14b:free",
#     openai_api_key=os.getenv("OPENROUTER_API_KEY"),
# )

model = ChatGoogleGenerativeAI(model='gemini-2.0-flash')
# Simple TypedDict
#Schema
# class Review(TypedDict):

#     summary: str
#     sentiment: str

class Review(TypedDict):

    key_themes: Annotated[list[str], "Write down all the keys themes discussed in the review in a list"]
    summary: Annotated[str,"A brief summary of the review"]
    sentiment: Annotated[Literal["pos","neg"],"Return sentiment of the review either negative, positive or neutral"]
    pros: Annotated[Optional[list[str]],"Write down all the pros inside a list"]
    cons: Annotated[Optional[list[str]],"Write down all the cons inside a list"]

structured_model = model.with_structured_output(Review)


result = structured_model.invoke("""The hardware is great, but the software feels bloated. There are too many pre-installed apps that I can't remove.
                        Also, the UI looks outdated compared to other brands. Hoping for a software update to fix this.""")

print(result)
# print(result)

