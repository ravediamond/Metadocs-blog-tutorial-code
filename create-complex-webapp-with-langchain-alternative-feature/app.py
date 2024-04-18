import os

from dotenv import load_dotenv
import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import ConfigurableField
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from enum import Enum


# Load the variables from .env
load_dotenv()

st.title("Hello, Metadocs readers!")


class Scenario(Enum):
    ONE_VECTOR_STORE_SINGLE_LINE = "one_vector_store_single_line"
    ONE_VECTOR_STORE_DETAILED = "one_vector_store_detailed"
    TWO_VECTOR_STORES_SINGLE_LINE = "two_vector_stores_single_line"
    TWO_VECTOR_STORES_DETAILED = "two_vector_stores_detailed"


# Base templates for vector store prompts
template_one_store_single_line = PromptTemplate.from_template(
    """Answer the question in a single line based on the following context:
{context}

Question: {question}
"""
)

template_one_store_detailed = PromptTemplate.from_template(
    """Answer the question in a detailed way with an idea per bullet point based on the following context:
{context}

Question: {question}
"""
)

base_template_two_stores_single_line = PromptTemplate.from_template(
    """Answer the question in a single line based on the following contexts:
{context1}

{context2}

Question: {question}
"""
)

base_template_two_stores_detailed = PromptTemplate.from_template(
    """Answer the question in a detailed way with an idea per bullet based on the following contexts:
{context1}

{context2}

Question: {question}
"""
)

scenario_name = []

prompt_alternatives = {
    Scenario.ONE_VECTOR_STORE_DETAILED.value: template_one_store_detailed,
    Scenario.TWO_VECTOR_STORES_SINGLE_LINE.value: base_template_two_stores_single_line,
    Scenario.TWO_VECTOR_STORES_DETAILED.value: base_template_two_stores_detailed,
}

# Configurable prompt templates with alternatives for detailed and summary answers

configurable_prompt = template_one_store_single_line.configurable_alternatives(
    which=ConfigurableField(
        id="scenario",
        name="Scenario",
        description="The scenario to use",
    ),
    default_key=Scenario.ONE_VECTOR_STORE_SINGLE_LINE.value,
    **prompt_alternatives
)


model = ChatOpenAI(
    temperature=0,
    model_name="gpt-3.5-turbo-0125",
    openai_api_key=os.environ["OPENAI_KEY"],
)
embedding = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_KEY"])

first_retriever = None
second_retriever = None

tab1, tab2 = st.tabs(["Upload Files", "Query"])

with tab1:
    st.header("Upload Files to Vector Stores")
    first_index_uploaded_file = st.file_uploader(
        "Upload a text file to the first vector store:", type="txt", key="first_index"
    )
    if first_index_uploaded_file is not None:
        string_data = first_index_uploaded_file.getvalue().decode("utf-8")
        splitted_data = string_data.split("\n\n")
        first_vectorstore = FAISS.from_texts(splitted_data, embedding=embedding)
        first_retriever = first_vectorstore.as_retriever()
        st.success("First vector store loaded successfully!")

    second_index_uploaded_file = st.file_uploader(
        "Upload a text file to the second vector store:", type="txt", key="second_index"
    )
    if second_index_uploaded_file is not None:
        string_data = second_index_uploaded_file.getvalue().decode("utf-8")
        splitted_data = string_data.split("\n\n")
        second_vectorstore = FAISS.from_texts(splitted_data, embedding=embedding)
        second_retriever = second_vectorstore.as_retriever()
        st.success("Second vector store loaded successfully!")

with tab2:
    st.header("Query a Vector Store")
    if first_retriever or second_retriever:
        vector_store_selection = st.selectbox(
            "Choose the vector store configuration:",
            ["First only", "Second only", "Both"],
        )
        output_type = st.selectbox(
            "Select the type of response:", ["detailed", "single line"]
        )

        question = st.text_input("Input your question:")
        if st.button("Get Answer"):
            if vector_store_selection == "First only" and output_type == "single line":
                scenario = Scenario.ONE_VECTOR_STORE_SINGLE_LINE.value
                context = {"context": first_retriever, "question": question}

            elif vector_store_selection == "Both" and output_type == "detailed":
                scenario = Scenario.ONE_VECTOR_STORE_DETAILED.value
                context = {"context": second_retriever, "question": question}

            elif vector_store_selection == "Both" and output_type == "single line":
                scenario = Scenario.TWO_VECTOR_STORES_SINGLE_LINE.value
                context = {
                    "context1": first_retriever,
                    "context2": second_retriever,
                    "question": question,
                }
            elif vector_store_selection == "Both" and output_type == "detailed":
                scenario = Scenario.TWO_VECTOR_STORES_DETAILED.value
                context = {
                    "context1": first_retriever,
                    "context2": second_retriever,
                    "question": question,
                }
            else:
                st.write("There is an error.")

            chain = (
                context | configurable_prompt | model | StrOutputParser()
            ).with_config(configurable={"scenario": scenario})

            result = chain.invoke(question)
            st.write(result)
    else:
        st.write("Please upload files to the vector stores before querying.")
