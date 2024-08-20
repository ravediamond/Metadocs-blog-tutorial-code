import os
from typing import Literal, TypedDict, List


from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import END, StateGraph, START


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

not_answerable_template = """The following question cannot be answered using the following vector stores:
- clean_energy: a speech to advocates for a unified commitment to transitioning to clean energy through solar, 
wind, geothermal, and energy-efficient technologies, emphasizing the importance of community action, 
education, and innovation in creating a sustainable future.
- state_of_the_union: the State of the Union address emphasizes the resilience of the American people, 
highlights strong economic recovery efforts, pledges support for Ukraine, and calls for unity in facing domestic and global challenges.

Explain to the question writer why it is not possible to answer this question using the vector store 
and give some advices if possible to make an answerable question.

Question: {question}
"""

rag_prompt = ChatPromptTemplate.from_template(rag_template)
route_prompt = ChatPromptTemplate.from_template(router_template)
not_answerable_prompt = ChatPromptTemplate.from_template(not_answerable_template)
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

rag_chain = rag_prompt | model | StrOutputParser()

structured_model_router = model.with_structured_output(RouteQuery)
question_router = route_prompt | structured_model_router

not_answerable_chain = not_answerable_prompt | model | StrOutputParser()


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


def not_answerable_generate(state):
    """
    Generate answer in case of not answerable decision

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE---")
    question = state["question"]

    # Not answerable generation
    generation = not_answerable_chain.invoke({"question": question})
    return {"documents": None, "question": question, "generation": generation}


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


# Define the nodes
workflow = StateGraph(GraphState)
workflow.add_node("state_of_the_union_retrieve", state_of_the_union_retrieve)
workflow.add_node("clean_energy_retrieve", clean_energy_retrieve)
workflow.add_node("generate", generate)
workflow.add_node("not_answerable_generate", not_answerable_generate)

# Build graph
workflow.add_conditional_edges(
    START,
    route_question,
    {
        "state_of_the_union": "state_of_the_union_retrieve",
        "clean_energy": "clean_energy_retrieve",
        "not_answerable": "not_answerable_generate",
    },
)

workflow.add_edge("state_of_the_union_retrieve", "generate")
workflow.add_edge("clean_energy_retrieve", "generate")
workflow.add_edge("generate", END)
workflow.add_edge("not_answerable_generate", END)

# Compile
graph = workflow.compile()

st.title("Hello, Metadocs readers!")

st.image(graph.get_graph(xray=True).draw_mermaid_png())

question = st.text_input("Input your question for the uploaded document")
inputs = {"question": question}

if question:
    result = None
    for output in graph.stream(inputs):
        st.write(output)
