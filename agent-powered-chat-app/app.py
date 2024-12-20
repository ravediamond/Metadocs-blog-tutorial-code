import os
import streamlit as st
import streamlit_mermaid as stmd
from typing import TypedDict, List
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langgraph.graph import END, StateGraph, START

# Load environment variables
load_dotenv()


# Data models
class MessageType(BaseModel):
    """Determine if message is feedback or new diagram request"""

    type: str = Field(description="Type of message: 'feedback' or 'new'")
    content: str = Field(description="Extracted content/modifications")


class GraphState(TypedDict):
    """State for our diagram modification graph"""

    message: str
    modifications: List[str]
    current_diagram: str
    diagram_requirements: str


# Initialize LLM and prompts
model = ChatOpenAI(temperature=0, openai_api_key=os.environ["OPENAI_KEY"])

analyze_template = """Determine if the user message is providing feedback about a diagram or requesting a new diagram.
If it's feedback, extract what needs to be modified. If it's a new diagram request, identify what kind of diagram is needed.

User message: {message}

Return in this exact format:
TYPE: [FEEDBACK or NEW]
CONTENT: [If feedback, list modifications needed, if new, describe diagram type]"""

diagram_template = """Generate a mermaid diagram based on the following requirements. 
ONLY return the mermaid diagram code, nothing else. No explanations, no markdown.

Requirements: {requirements}

Modifications to implement:
{modifications}"""

analyze_prompt = ChatPromptTemplate.from_template(analyze_template)
diagram_prompt = ChatPromptTemplate.from_template(diagram_template)

structured_model = model.with_structured_output(MessageType)
analyzer = analyze_prompt | structured_model
diagram_generator = diagram_prompt | model | StrOutputParser()


def analyze_message(state: GraphState):
    """Analyze if message is feedback or new diagram request"""
    message = state["message"]
    result = analyzer.invoke({"message": message})

    if result.type == "FEEDBACK":
        return {"modifications": state["modifications"] + [result.content]}
    else:
        return {"diagram_requirements": result.content}


def generate_diagram(state: GraphState):
    """Generate new diagram based on requirements and modifications"""
    mods = "\n".join(f"- {mod}" for mod in state["modifications"])
    new_diagram = diagram_generator.invoke(
        {"requirements": state["diagram_requirements"], "modifications": mods}
    )
    return {"current_diagram": new_diagram, "modifications": []}


def route_next_step(state: GraphState):
    """Determine next step based on message analysis"""
    message = state["message"]
    result = analyzer.invoke({"message": message})
    return "generate" if result.type == "NEW" else END


# Define workflow
workflow = StateGraph(GraphState)

# Add nodes
workflow.add_node("analyze", analyze_message)
workflow.add_node("generate", generate_diagram)

# Build graph
workflow.add_edge(START, "analyze")
workflow.add_conditional_edges(
    "analyze", route_next_step, {"generate": "generate", END: END}
)
workflow.add_edge("generate", END)

# Compile
graph = workflow.compile()

# Streamlit app
st.title("Diagram Chat Assistant")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "modifications" not in st.session_state:
    st.session_state.modifications = []
if "current_diagram" not in st.session_state:
    st.session_state.current_diagram = """
    graph TD
        A[Web Client] --> B[API Gateway]
        B --> C[Application Server]
        C --> D[(Database)]
    """
if "diagram_requirements" not in st.session_state:
    st.session_state.diagram_requirements = "Create a basic system architecture diagram"

# Display current modifications if any exist
if st.session_state.modifications:
    st.sidebar.title("Pending Modifications")
    for mod in st.session_state.modifications:
        st.sidebar.write(f"- {mod}")

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

    # Process with graph
    inputs = {
        "message": prompt,
        "modifications": st.session_state.modifications,
        "current_diagram": st.session_state.current_diagram,
        "diagram_requirements": st.session_state.diagram_requirements,
    }

    # Run graph and update state
    result = None
    for output in graph.stream(inputs):
        result = output

        # Update session state based on graph output
        if "modifications" in result:
            st.session_state.modifications = result["modifications"]
        if "current_diagram" in result:
            st.session_state.current_diagram = result["current_diagram"]
        if "diagram_requirements" in result:
            st.session_state.diagram_requirements = result["diagram_requirements"]

    # Add assistant response
    with st.chat_message("assistant"):
        if "modifications" in result and len(result["modifications"]) > len(
            inputs["modifications"]
        ):
            response = "I've added your feedback to the modifications list."
        else:
            response = "I've generated a new diagram incorporating your request!"
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})

    # Rerun to update UI
    if "current_diagram" in result:
        st.experimental_rerun()

# Button to generate new diagram with current modifications
if st.sidebar.button("Generate New Diagram") and st.session_state.modifications:
    inputs = {
        "message": "Generate new diagram",
        "modifications": st.session_state.modifications,
        "current_diagram": st.session_state.current_diagram,
        "diagram_requirements": st.session_state.diagram_requirements,
    }

    for output in graph.stream(inputs):
        if "current_diagram" in output:
            st.session_state.current_diagram = output["current_diagram"]
            st.session_state.modifications = []
            st.experimental_rerun()
