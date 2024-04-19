import os
from typing import List, Optional

from dotenv import load_dotenv
import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import ConfigurableField, RunnablePassthrough
from langchain.schema import format_document, Document
from langchain.schema.runnable import (
    ConfigurableField,
    RunnableConfig,
    RunnableSerializable,
)
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser


# Load the variables from .env
load_dotenv()

st.title("Hello, Metadocs readers!")

# Base templates for vector store prompts
template_single_line = PromptTemplate.from_template(
    """Answer the question in a single line based on the following context:
{context}

Question: {question}
"""
)

template_detailed = PromptTemplate.from_template(
    """Answer the question in a detailed way with an idea per bullet point based on the following context:
{context}

Question: {question}
"""
)

prompt_alternatives = {
    "detailed": template_detailed,
}


configurable_prompt = template_single_line.configurable_alternatives(
    which=ConfigurableField(
        id="output_type",
        name="Output type",
        description="The type for the output, single line or detailed.",
    ),
    default_key="single_line",
    **prompt_alternatives,
)

model = ChatOpenAI(
    temperature=0,
    model_name="gpt-3.5-turbo-0125",
    openai_api_key=os.environ["OPENAI_KEY"],
)
embedding = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_KEY"])

politic_vector_store_path = "politic_vector_store_path.faiss"
environnetal_vector_store_path = "environnetal_vector_store_path.faiss"


class ConfigurableFaissRetriever(RunnableSerializable[str, List[Document]]):
    vector_store_topic: str

    def invoke(
        self, input: str, config: Optional[RunnableConfig] = None
    ) -> List[Document]:
        """Invoke the retriever."""

        vector_store_path = (
            politic_vector_store_path
            if "politic" in vector_store_topic
            else environnetal_vector_store_path
        )
        faiss_vector_store = FAISS.load_local(
            vector_store_path,
            embedding,
            allow_dangerous_deserialization=True,
        )
        retriever = faiss_vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 4}
        )
        return retriever.invoke(input, config=config)


configurable_faiss_vector_store = ConfigurableFaissRetriever(
    vector_store_topic="politic"
).configurable_fields(
    vector_store_topic=ConfigurableField(
        id="vector_store_topic",
        name="Vector store topic",
        description="The topic of the faiss vector store.",
    )
)

DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")


def combine_documents(
    docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"
):
    """Combine documents into a single string."""
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return document_separator.join(doc_strings)


tab1, tab2 = st.tabs(["Upload Files", "Query"])

with tab1:
    st.header("Upload Files to Vector Stores")
    politic_index_uploaded_file = st.file_uploader(
        "Upload a text file to Politic vector store:", type="txt", key="politic_index"
    )
    if politic_index_uploaded_file is not None:
        string_data = politic_index_uploaded_file.getvalue().decode("utf-8")
        splitted_data = string_data.split("\n\n")
        politic_vectorstore = FAISS.from_texts(splitted_data, embedding=embedding)
        politic_vectorstore.save_local(politic_vector_store_path)
        st.success("Politic vector store loaded successfully!")

    environnetal_index_uploaded_file = st.file_uploader(
        "Upload a text file to the Environnemental vector store:",
        type="txt",
        key="environnetal_index",
    )
    if environnetal_index_uploaded_file is not None:
        string_data = environnetal_index_uploaded_file.getvalue().decode("utf-8")
        splitted_data = string_data.split("\n\n")
        environnetal_vectorstore = FAISS.from_texts(splitted_data, embedding=embedding)
        environnetal_vectorstore.save_local(environnetal_vector_store_path)
        st.success("Environnemental vector store loaded successfully!")

with tab2:
    st.header("Query a Vector Store")
    if os.path.exists(politic_vector_store_path) or os.path.exists(
        environnetal_vector_store_path
    ):
        vector_store_topic = st.selectbox(
            "Choose the vector store configuration:",
            ["Politic", "Environnemental"],
        )
        output_type = st.selectbox(
            "Select the type of response:", ["detailed", "single_line"]
        )

        question = st.text_input("Input your question:")
        if st.button("Get Answer"):

            context = {
                "context": configurable_faiss_vector_store | combine_documents,
                "question": RunnablePassthrough(),
            }

            chain = (
                context | configurable_prompt | model | StrOutputParser()
            ).with_config(
                configurable={
                    "vector_store_topic": vector_store_topic,
                    "output_type": output_type,
                }
            )
            result = chain.invoke(question)
            st.write(result)
    else:
        st.write("Please upload files to the vector stores before querying.")
