import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_aws.chat_models import BedrockChat


aws_region_name = "eu-west-1"
credentials_profile_name = "default"
claude_3_5_sonnet = "anthropic.claude-3-sonnet-20240229-v1:0"

llm = BedrockChat(
    model_id=claude_3_5_sonnet,
    credentials_profile_name=credentials_profile_name,
    region_name=aws_region_name,
)

question = st.text_input("Input your question")

result = llm.invoke(question)
