from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
import streamlit as st
import os
from dotenv import load_dotenv


load_dotenv()

## Langsmith tracking
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ['LANGCHAIN_API_KEY']=os.getenv("LANGCHAIN_API_KEY","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")

# Prompt template

prompt = ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant. Please respond to the user queries"),
        ("user","Question : {question}")
    ]
)

#streamlit framework 

st.title('Langchain Demo with Llama3.1 APP')
input_text = st.text_input("Search the topic you want")

# ollama LLM
llm = Ollama(model="llama3.1")
output_parser = StrOutputParser()
chain = prompt|llm|output_parser

if input_text:
    # Generate the prompt with user input
    st.write(chain.invoke({"question":input_text}))