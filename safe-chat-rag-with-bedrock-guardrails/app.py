import streamlit as st
from langchain_aws.chat_models import ChatBedrock
from langchain_core.output_parsers import StrOutputParser


aws_region_name = "eu-west-1"
claude_3_5_sonnet = "eu.anthropic.claude-3-5-sonnet-20240620-v1:0"
guardrail_id = "GUARDRAIL_ID"
guardrail_version = "1"

llm = ChatBedrock(
    model_id=claude_3_5_sonnet,
    region_name=aws_region_name,
    guardrails={
        "guardrailIdentifier": guardrail_id,
        "guardrailVersion": guardrail_version,
    },
)

chain = llm | StrOutputParser()

question = st.text_input("Input your question")

if question:
    result = chain.invoke(question)
    st.write(result)
