import os
import streamlit as st
import streamlit_mermaid as stmd
from typing import Literal
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent

# Load environment variables
load_dotenv()

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_diagram" not in st.session_state:
    st.session_state.current_diagram = """
    graph TD
        A[Web Client] --> B[API Gateway]
        B --> C[Application Server]
        C --> D[(Database)]
    """

# Initialize LLM
model = ChatOpenAI(temperature=0, openai_api_key=os.environ["OPENAI_KEY"])


# Define tools
@tool
def modify_diagram(modification: str) -> str:
    """Modifies the current diagram based on the given modification."""
    st.session_state.current_diagram = modification
    return f"Diagram modified: {modification}"


@tool
def get_current_diagram() -> str:
    """Returns the current diagram."""
    return st.session_state.current_diagram


# Create tools list
tools = [modify_diagram, get_current_diagram]

# Create ReAct agent
graph = create_react_agent(model, tools=tools)

# Streamlit app
st.title("Diagram Chat Assistant")

# Display the current diagram
stmd.st_mermaid(st.session_state.current_diagram)

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input
if prompt := st.chat_input("What's on your mind?"):
    # Add user message to chat
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Process with ReAct agent
    response = graph.invoke({"input": prompt})

    # Add assistant response
    with st.chat_message("assistant"):
        st.markdown(response["output"])
    st.session_state.messages.append(
        {"role": "assistant", "content": response["output"]}
    )
    st.rerun()
