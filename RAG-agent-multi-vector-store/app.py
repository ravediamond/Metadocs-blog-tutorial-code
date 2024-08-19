import os

from dotenv import load_dotenv
import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Added
from langchain.tools.retriever import create_retriever_tool
from typing import Annotated, Literal, Sequence, TypedDict, List
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langchain_core.pydantic_v1 import BaseModel, Field

# Load the variables from .env
load_dotenv()

# Data model
class RouteQuery(BaseModel):
    """Route a user query to the most relevant vector store."""

    datasource: Literal["state_of_the_union", "clean_energy", "not_answerable"] = Field(
        ...,
        description="Given a user question choose to route it to the relevant vector store or say it is not answerable.",
    )


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        documents: list of documents
    """

    question: str
    generation: str
    documents: List[str]


rag_template = """Answer the question based only on the following contexts:
{context}

Question: {question}
"""

router_template = """You are an expert at routing a user question to different vector stores.
There is 2 vector stores, one about the state of the union (USA) and the other about clean_energy.
Return the corresponding vectors store depending of the topics of the question or just say that you
answer the question because it does't match with the vector stores.

Question: {question}
"""

rag_prompt = ChatPromptTemplate.from_template(rag_template)
route_prompt = ChatPromptTemplate.from_template(router_template)
model = ChatOpenAI(
    temperature=0,
    model_name="gpt-4o-2024-08-06",
    openai_api_key=os.environ["OPENAI_KEY"],
)
embedding = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_KEY"])

state_of_the_union = open("state_of_the_union.txt", "r").read()
state_of_the_union_vectorstore = FAISS.from_texts(
    state_of_the_union.split("\n\n"), embedding=embedding
)
state_of_the_union_retriever = state_of_the_union_vectorstore.as_retriever()

clean_energy = open("generated_clean_energy_discourse.txt", "r").read()
clean_energy_vectorstore = FAISS.from_texts(
    clean_energy.split("\n\n"), embedding=embedding
)
clean_energy_retriever = clean_energy_vectorstore.as_retriever()

rag_chain = (
        rag_prompt
        | model
        | StrOutputParser()
    )

structured_model_router = model.with_structured_output(RouteQuery)
question_router = route_prompt | structured_model_router



def state_of_the_union_retrieve(state):
    """
    Retrieve documents fromt the state of the union vector store

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE STATE OF THE UNION---")
    question = state["question"]

    # Retrieval
    documents = state_of_the_union_retriever.invoke(question)
    return {"documents": documents, "question": question}


def clean_energy_retrieve(state):
    """
    Retrieve documents fromt the clean energy vector store

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE CLEAN ENERGY---")
    question = state["question"]

    # Retrieval
    documents = clean_energy_retriever.invoke(question)
    return {"documents": documents, "question": question}


def generate(state):
    """
    Generate answer

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]

    # RAG generation
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}


def route_question(state):
    """
    Route question to corresponding RAG.

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """

    print("---ROUTE QUESTION---")
    question = state["question"]
    source = question_router.invoke({"question": question})
    if source.datasource == "state_of_the_union":
        print("---ROUTE QUESTION TO STATE OF THE UNION---")
        return "state_of_the_union"
    elif source.datasource == "clean_energy":
        print("---ROUTE QUESTION TO CLEAN ENERGY---")
        return "clean_energy"
    elif source.datasource == "not_answerable":
        print("---ROUTE QUESTION TO NOT ANSWERABLE---")
        return "not_answerable"


st.title("Hello, Metadocs readers!")


    question = st.text_input("Input your question for the uploaded document")

    result = chain.invoke(question)

    st.write(result)
