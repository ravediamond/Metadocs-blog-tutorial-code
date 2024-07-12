import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Load the variables from .env
load_dotenv()

template = """Answer the question based only on the following context:
{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)
model = ChatOpenAI(
    temperature=0, model_name="gpt-4", openai_api_key=os.environ["OPENAI_KEY"]
)
embedding = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_KEY"])


string_data = uploaded_file.getvalue().decode("utf-8")

splitted_data = string_data.split("\n\n")

vectorstore = FAISS.from_texts(splitted_data, embedding=embedding)
retriever = vectorstore.as_retriever()

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

question = st.text_input("Input your question for the uploaded document")

result = chain.invoke(question)
