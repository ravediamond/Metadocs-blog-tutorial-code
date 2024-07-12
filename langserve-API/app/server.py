import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langserve import add_routes
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


app = FastAPI(
    title="Metadocs Langserve Tutorial",
    version="0.1",
    description="Spin up a LangServe API for learning purpose",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

config = {
    "enable_feedback_endpoint": False,
}

template = """Answer the question as if you were Homer Simpsons.
Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)
model = ChatOpenAI(
    temperature=0,
    model_name="gpt-3.5-turbo-0125",
    openai_api_key=os.environ["OPENAI_KEY"],
)


chain = prompt | model | StrOutputParser()

add_routes(app, chain, path="/homer-chat", **config)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000, log_level="error")
