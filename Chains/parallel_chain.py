from langchain_community.chat_models import ChatOllama
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema.runnable import RunnableParallel

load_dotenv()

model1 = ChatOllama(
    base_url="https://6fe462576267.ngrok-free.app",
    model="qwen3:0.6b",
    strem = False,
    think = False
)

model2 = ChatGoogleGenerativeAI(model='gemini-2.0-flash')

prompt1 = PromptTemplate(
    template='Generate short and simple notes from the following text \n {text}',
    input_variables=['text']
)

prompt2 = PromptTemplate(
    template='Generate 5 short question answers from the following text \n {text}',
    input_variables=['text']
)

prompt3 = PromptTemplate(
    template='Merger the provided notes and quiz into single document \n notes -> {notes} and quiz -> {quiz}',
    input_variables=['notes','quiz']
)

parser = StrOutputParser()

paralle_chain = RunnableParallel({
    'notes': prompt1 | model1 | parser,
    'quiz' : prompt2 | model2 | parser
})

merge_chain = prompt3 | model2 | parser

chain = paralle_chain | merge_chain

text = """
## What are periods?

A period is made up of blood and the womb lining. The first day of a woman's period is day 1 of the menstrual cycle.

Periods last around 2 to 7 days, and women lose about 20 to 90ml (about 1 to 5 tablespoons) of blood in a period.

Many women facing heavy periods:

- What are heavy periods:
    - need to change your pad or tampon every 1 to 2 hours, or empty your menstrual cup more often than is recommended
    - need to use 2 types of period product together, such as a pad and a tampon
    - have periods lasting more than 7 days
    - pass blood clots larger than about 2.5cm (the size of a 10p coin)
    - bleed through to your clothes or bedding
    - avoid daily activities, like exercise, or take time off work because of your periods
    - feel tired or short of breath a lot
- Causes of heavy periods:
    - They can sometimes be heavy at different times, like when you first start your periods, after pregnancy or approaching menopause.
    - conditions affecting your womb or ovaries, such as [polycystic ovary syndrome](https://www.nhs.uk/conditions/polycystic-ovary-syndrome-pcos/), [fibroids](https://www.nhs.uk/conditions/fibroids/), [endometriosis](https://www.nhs.uk/conditions/endometriosis/), [adenomyosis](https://www.nhs.uk/conditions/adenomyosis/) and [pelvic inflammatory disease](https://www.nhs.uk/conditions/pelvic-inflammatory-disease-pid/).
    - conditions that can make you bleed more easily, such as [Von Willebrand disease](https://www.nhs.uk/conditions/von-willebrand-disease/)
    - Some medicines and treatments, including some [anticoagulant medicines](https://www.nhs.uk/medicines/anticoagulants/) and chemotherapy medicines
- GP points:
    - heavy periods are affecting your life
    - you've had heavy periods for some time
    - you have severe pain during your periods
    - you bleed between periods or after sex
    - you have heavy periods and other symptoms, such as pain when peeing, pooing or having sex
"""

result = chain.invoke({"text":text})

print(result)

chain.get_graph().print_ascii()