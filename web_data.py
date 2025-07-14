import os
import yaml
import requests as r
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.chat_models import ChatOllama
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_core.output_parsers import JsonOutputParser

# Load environment variables
load_dotenv()

# Step 1: Scrape web content
url = "https://gandhinagaruni.ac.in/admission-process/"
response = r.get(url)
soup = BeautifulSoup(response.content, "lxml")

# Step 2: Extract visible text from relevant blocks
text_blocks = soup.find_all("div", class_="elementor-widget-container")
full_text = "\n".join(block.get_text(strip=True, separator="\n") for block in text_blocks)

# Step 3: Define response schema
# response_schemas = [
#     ResponseSchema(name="course_name", description="Name of the course or program"),
#     ResponseSchema(name="eligibility", description="Eligibility criteria"),
#     ResponseSchema(name="admission_process", description="Details about the admission process"),
#     ResponseSchema(name="fees", description="Fee structure if available"),
#     ResponseSchema(name="duration", description="Course duration if mentioned"),
# ]

parser = JsonOutputParser()
format_instructions = parser.get_format_instructions()

# Step 4: Prompt template
prompt = PromptTemplate(
    template="""
Given the following content extracted from a university admission webpage:

\"\"\"{text}\"\"\"

Extract the following details in structured form for each course mentioned:
- Course Name
- Eligibility
- Admission Process
- Fees
- Duration

{format_instructions}
""",
    input_variables=["text"],
    partial_variables={"format_instructions": format_instructions}
)

# Step 5: Call the LLM
model = ChatGoogleGenerativeAI(model='gemini-2.0-flash')
chain = prompt | model | parser

# Step 6: Run the chain and get structured data
output = chain.invoke({"text": full_text})
# print(output)
yaml_data = yaml.dump(output, allow_unicode=True, sort_keys=False)

# Save to file
with open("courses.yaml", "w", encoding="utf-8") as f:
    f.write(yaml_data)

print("✅ YAML saved to courses.yaml")