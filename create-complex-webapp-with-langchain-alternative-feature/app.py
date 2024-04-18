import os

from dotenv import load_dotenv
import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import ConfigurableField
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser

# Load the variables from .env
load_dotenv()

print(os.environ["OPENAI_KEY"])

st.title("Hello, Metadocs readers!")

# Base templates for vector store prompts
base_template_two_stores = """Answer the question based on the following contexts:
{context1}

{context2}

Question: {question}
"""

base_template_one_store = """Answer the question based on the following context:
{context}

Question: {question}
"""

# Configurable prompt templates with alternatives for detailed and summary answers
prompt_two_stores = PromptTemplate.from_template(
    base_template_two_stores
).configurable_alternatives(
    which=ConfigurableField(
        id="output_type",
        name="Output Type",
        description="Choose detailed or summary for the type of answer.",
    ),
    detailed=ChatPromptTemplate.from_template(
        """Provide a detailed answer to the question, based on:
{context1}

{context2}

Question: {question}
"""
    ),
    summary=ChatPromptTemplate.from_template(
        """Provide a summary answer to the question, based on:
{context1}

{context2}

Question: {question}
"""
    ),
)

prompt_one_store = PromptTemplate.from_template(
    base_template_one_store
).configurable_alternatives(
    which=ConfigurableField(
        id="output_type",
        name="Output Type",
        description="Choose detailed or summary for the type of answer.",
    ),
    detailed=ChatPromptTemplate.from_template(
        """Provide a detailed answer to the question, based on:
{context}

Question: {question}
"""
    ),
    summary=ChatPromptTemplate.from_template(
        """Provide a summary answer to the question, based on:
{context}

Question: {question}
"""
    ),
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
        use_two_stores = st.checkbox("Use two vector stores for the query", value=False)
        output_type = st.selectbox(
            "Select the type of response:", ["detailed", "summary"]
        )

        question = st.text_input("Input your question:")
        if st.button("Get Answer"):
            if use_two_stores and first_retriever and second_retriever:
                context = {
                    "context1": first_retriever,
                    "context2": second_retriever,
                    "question": question,
                }
                chosen_prompt = prompt_two_stores.choose(output_type)
            elif first_retriever:
                context = {"context": first_retriever, "question": question}
                chosen_prompt = prompt_one_store.choose(output_type)
            elif second_retriever:
                context = {"context": second_retriever, "question": question}
                chosen_prompt = prompt_one_store.choose(output_type)
            else:
                st.error(
                    "No vector store loaded to query. Please upload at least one file."
                )

            chain = context | chosen_prompt | model | StrOutputParser()

            result = chain.invoke(question)
            st.write(result)
    else:
        st.write("Please upload files to the vector stores before querying.")
