import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langserve import add_routes
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS


embedding = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_KEY"])


app = FastAPI(
    title="Knowledge Assistant LLM Serving Layer",
    version="0.1",
    description="Spin up a LangServe API for the Knowledge Assistant App",
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


def file_to_vector_store(path: str):
    splitted_data = string_data.split("\n\n")

    vectorstore = FAISS.from_texts(splitted_data, embedding=embedding)
