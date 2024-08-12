import streamlit as st
from langchain_aws.chat_models import BedrockChat
from langchain_aws.embeddings import BedrockEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import LanceDB
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


aws_region_name = "us-east-1"
claude_3_sonnet = "anthropic.claude-3-sonnet-20240229-v1:0"
cohere_embed = "cohere.embed-multilingual-v3"
s3_bucket = "s3://vector-store-lancedb-tuto"

model = BedrockChat(
    model_id=claude_3_sonnet,
    region_name=aws_region_name,
)

cohere_embedding = BedrockEmbeddings(
    model_id=cohere_embed,
    region_name=aws_region_name,
)

template = """Answer the question based only on the following context:
{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

vector_store = LanceDB(
    uri=f"{s3_bucket}/lancedb/",
    table_name="tuto",
    embedding=cohere_embedding,
    mode="overwrite",
)
retriever = vector_store.as_retriever()

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

st.write("Hello, Metadocs readers!")

uploaded_file = st.file_uploader("Choose a text file", type="txt")

if uploaded_file is not None:
    string_data = uploaded_file.getvalue().decode("utf-8")
    split_data = string_data.split("\n\n")

    vector_store.add_texts(split_data)

question = st.text_input("Input your question for the uploaded document")

result = chain.invoke(question)

st.write(result)
