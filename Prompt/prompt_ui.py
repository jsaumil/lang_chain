from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain import prompts
import streamlit as st
import os

os.environ['HF_HOME'] = "F:\python\model"
llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task='text-generation',
    pipeline_kwargs=dict(
        temperature=0.5,
        max_new_tokens=100
    )
)

model = ChatHuggingFace(llm=llm)

st.header('Reasearch Tool')

user_input = st.text_input("Enter our prompt")

if st.button:
    st.text('Some Research Tool')

if st.button('Summarize'):
    result = model.invoke(user_input)
    
    st.write(result.content)    