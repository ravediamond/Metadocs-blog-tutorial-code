import os

from dotenv import load_dotenv
import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Load the variables from .env
load_dotenv()

st.title("Hello, Metadocs readers!")

template = """Answer the question based only on the following contexts:
{context1}

{context2}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)
model = ChatOpenAI(temperature=0, model_name="gpt-4", openai_api_key=os.environ["OPENAI_KEY"])
embedding = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_KEY"])
first_retriever = None
second_retriever = None

st.title("First vector store index")

first_index_uploaded_file = st.file_uploader("Choose a text file", type="txt", key="first_index")

if first_index_uploaded_file is not None:
    string_data = first_index_uploaded_file.getvalue().decode("utf-8")

    splitted_data = string_data.split("\n\n")

    first_vectorstore = FAISS.from_texts(
        splitted_data,
        embedding=embedding)
    first_retriever = first_vectorstore.as_retriever()


st.title("Second vector store index")

second_vector_uploaded_file = st.file_uploader("Choose a text file", type="txt", key="second_index")

if second_vector_uploaded_file is not None:
    string_data = second_vector_uploaded_file.getvalue().decode("utf-8")

    splitted_data = string_data.split("\n\n")

    second_vectorstore = FAISS.from_texts(
        splitted_data,
        embedding=embedding)
    second_retriever = second_vectorstore.as_retriever()

if first_retriever is not None and second_retriever is not None:
    chain = (
        {"context1": first_retriever,
         "context2": second_retriever,
         "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )

    question = st.text_input("Input your question for the uploaded document")

    result = chain.invoke(question)

    st.write(result)
