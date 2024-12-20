import os
import streamlit as st
import streamlit_mermaid as stmd
from typing import TypedDict, List, Literal
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from langgraph.graph import END, StateGraph, START

# Load environment variables
load_dotenv()

# Initialize session state first
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


# Data models
class GraphState(TypedDict):
    """State for our diagram modification graph"""

    message: str


# Initialize LLM and prompts
model = ChatOpenAI(temperature=0, openai_api_key=os.environ["OPENAI_KEY"])

check_redraw_template = """Determine if the user is asking to redraw/update/regenerate/update the diagram.
Return EXACTLY 'redraw' if they want to update/redraw/regenerate, 'modify' if it's a modification request.

User message: {message}

Think step by step:
1. Is the user asking to update/redraw/regenerate the existing diagram?
2. Or are they providing a new modification?

Return only one word, either 'redraw' or 'modify'."""

analyze_modification_template = """Extract the modification request from the user's message.
Be specific about what needs to be modified in the diagram.

User message: {message}

Return ONLY the modification needed, no additional text."""

generate_diagram_template = """Generate a mermaid flowchart diagram by modifying the current diagram to include the requested changes.
Return ONLY the mermaid diagram code starting with 'graph TD' or 'flowchart TD', no other text, no explanations.

Current diagram:
{current_diagram}

Modifications to implement:
{modifications}

Rules:
1. Start with 'graph TD' or 'flowchart TD'
2. Keep all existing components unless they need to be modified
3. Use descriptive names in square brackets (e.g., [Cache], [CDN], [API Gateway])
4. Use proper Mermaid syntax for connections (-->)
5. Add new components based on the modifications
6. Show all connections between components
7. Maintain proper node IDs (like A, B, C) but with descriptive labels

Example output format:
graph TD
    A[Component1] --> B[Component2]
    B --> C[Component3]
"""

check_redraw_prompt = ChatPromptTemplate.from_template(check_redraw_template)
analyze_mod_prompt = ChatPromptTemplate.from_template(analyze_modification_template)
diagram_prompt = ChatPromptTemplate.from_template(generate_diagram_template)


def router(state: GraphState) -> Literal["modify", "redraw"]:
    """Route to next step based on message analysis"""
    print("\n=== ROUTER STARTED ===")
    message = state["message"]
    result = (check_redraw_prompt | model | StrOutputParser()).invoke(
        {"message": message}
    )
    print(f"Router output for message '{message}': {result}")
    return "redraw" if result.strip().lower() == "redraw" else "modify"


def analyze_modification(state: GraphState):
    """Extract modification from message"""
    print("\n=== ANALYZING MODIFICATION ===")
    message = state["message"]
    modification = (analyze_mod_prompt | model | StrOutputParser()).invoke(
        {"message": message}
    )
    print(f"Extracted modification: {modification}")
    print(f"Current modifications: {st.session_state.modifications}")

    # Update session state directly
    st.session_state.modifications.append(modification)
    print(f"Updated modifications list: {st.session_state.modifications}")
    return {}


def generate_diagram(state: GraphState):
    """Generate new diagram based on modifications"""
    print("\n=== GENERATING DIAGRAM ===")
    print(f"Current modifications to implement: {st.session_state.modifications}")
    mods_text = "\n".join(f"- {mod}" for mod in st.session_state.modifications)

    new_diagram = (diagram_prompt | model | StrOutputParser()).invoke(
        {
            "current_diagram": st.session_state.current_diagram,
            "modifications": mods_text,
        }
    )
    print(f"Generated diagram:\n{new_diagram}")

    if not new_diagram.strip():
        print("Warning: Generated diagram is empty, keeping current diagram")
        return {}

    # Update session state only if we got a valid diagram
    st.session_state.current_diagram = new_diagram
    st.session_state.modifications = []
    return {}


# Define workflow
workflow = StateGraph(GraphState)

# Add nodes
workflow.add_node("modify", analyze_modification)
workflow.add_node("redraw", generate_diagram)

# Build graph
workflow.add_conditional_edges(START, router, {"modify": "modify", "redraw": "redraw"})
workflow.add_edge("modify", END)
workflow.add_edge("redraw", END)

# Compile
graph = workflow.compile()

# Streamlit app
st.title("Diagram Chat Assistant")

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
    inputs = {"message": prompt}

    # Get router decision first
    route_decision = router({"message": prompt})

    # Run graph and update state
    for output in graph.stream(inputs):
        print(f"\nGraph output: {output}")

    # Add assistant response
    with st.chat_message("assistant"):
        if route_decision == "modify":
            total_mods = len(st.session_state.modifications)
            response = f'✅ Got it! I\'ve added this modification to the list: "{st.session_state.modifications[-1]}"'
            if total_mods > 1:
                response += f"\n\nThere are now {total_mods} pending modifications. When you're done adding changes, just ask me to \"update the diagram\" and I'll implement them all."
            else:
                response += '\n\nYou can keep adding more modifications, or ask me to "update the diagram" when you\'re ready.'
        else:  # redraw
            response = "✨ I've updated the diagram with all your modifications! Let me know if you'd like to make any other changes."
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})

    st.rerun()
