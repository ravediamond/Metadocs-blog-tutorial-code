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
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        documents: list of documents
    """

    original_question: str
    question: str
    generation: str
    documents: List[str]
    vocabulary: str


rag_template = """Answer the question based only on the following contexts:
{context}

Question: {question}
"""

grade_template = """You are a grader assessing relevance of a retrieved document to a user question. 
    If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant.
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.

User question: {question}

Retrieved documents: {documents}
"""

rewrite_template = """You a question re-writer that converts an input question to a better version that is upgraded
     by using the given domain specific vocabulary. Look at the input and try to reason about the underlying semantic intent / meaning.

Question: {question}

Domain specific definitions: {vocabulary}
"""

rag_prompt = ChatPromptTemplate.from_template(rag_template)
grade_prompt = ChatPromptTemplate.from_template(grade_template)
rewrite_prompt = ChatPromptTemplate.from_template(rewrite_template)
model = ChatOpenAI(
    temperature=0,
    model_name="gpt-4o-2024-08-06",
    openai_api_key=os.environ["OPENAI_KEY"],
)
embedding = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_KEY"])

clean_energy_vocabulary = open("domain_specific_vocabulary.txt", "r").read()
clean_energy = open("clean_energy_domain_specific.txt", "r").read()
clean_energy_vectorstore = FAISS.from_texts(
    clean_energy.split("\n\n"), embedding=embedding
)
clean_energy_retriever = clean_energy_vectorstore.as_retriever()

rag_chain = rag_prompt | model | StrOutputParser()

structured_model_grader = model.with_structured_output(GradeDocuments)
grader = grade_prompt | structured_model_grader

rewriter = rewrite_prompt | model | StrOutputParser()


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
    # return {"documents": documents, "question": question, "generation": generation}
    return {"generation": generation}


def get_vocabulary(state):
    """
    Get domain specific vocabulary

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GET_VOCABULARY---")
    question = state["question"]

    return {"vocabulary": clean_energy_vocabulary}


def rewrite_question(state):
    """
    Rewrite the question with domain specific information

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---REWRITE QUESTION---")
    question = state["question"]
    original_question = state["original_question"]
    vocabulary = state["vocabulary"]

    # Not answerable generation
    generation = rewriter.invoke({"question": question, "vocabulary": vocabulary})

    if original_question is None:
        original_question = question

    question = generation
    return {"question": question}


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
    return {"generation": generation}


def grade_question(state):
    """
    Grade question to see if there is a need to add domain specific vocabulary.

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """

    print("---GRADE QUESTION---")
    question = state["question"]
    documents = state["documents"]
    grade = grader.invoke({"question": question, "documents": documents})
    if grade.binary_score == "yes":
        print("---GRADE QUESTION TO GENERATE---")
        return "yes"
    elif grade.binary_score == "no":
        print("---GRADE QUESTION TO ---")
        return "no"


# Define the nodes
workflow = StateGraph(GraphState)

workflow.add_node("clean_energy_retrieve", clean_energy_retrieve)
workflow.add_node("generate", generate)
workflow.add_node("get_vocabulary", get_vocabulary)
workflow.add_node("rewrite_question", rewrite_question)

# Build graph
workflow.add_edge(START, "clean_energy_retrieve")
workflow.add_conditional_edges(
    "clean_energy_retrieve",
    grade_question,
    {
        "yes": "generate",
        "no": "get_vocabulary",
    },
)

workflow.add_edge("get_vocabulary", "rewrite_question")
workflow.add_edge("rewrite_question", "clean_energy_retrieve")
workflow.add_edge("clean_energy_retrieve", "generate")
workflow.add_edge("generate", END)

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
