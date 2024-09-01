from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langserve import add_routes
import uvicorn
import os
from langchain_ollama import OllamaLLM
from dotenv import load_dotenv
load_dotenv()

os.environ["OPEN_API_KEY"] = os.getenv("OPEN_API_KEY",'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')

app = FastAPI(
    title="Langchain Server",
    version="1.0",
    description="A simple API server"    
)

add_routes(
    app,
    ChatOpenAI(),
    path='/openai'
)
model=ChatOpenAI()

# ollama llama3.1
llm = OllamaLLM(model="llama3.1")

# prompt template
prompt1 = ChatPromptTemplate.from_template("Write me an essay about {topic} with 100 words.")
prompt2 = ChatPromptTemplate.from_template("Write me a poem about {topic} with 100 words.")


add_routes(
    app,
    prompt1|model,
    path='/essay'
)

add_routes(
    app,
    prompt2|llm,
    path='/poem'
)

if __name__ == "__main__":
    uvicorn.run(app,host="localhost",port=000)
    