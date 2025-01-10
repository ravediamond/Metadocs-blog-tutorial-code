import streamlit as st
import streamlit_mermaid as stmd
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from dotenv import load_dotenv
from typing import Literal
import json

# Load environment variables
load_dotenv()


class LLMOutput(BaseModel):
    explanation: str = Field(..., description="The text explanation for the user")
    viz_content: str = Field(..., description="The content to be visualized")
    viz_type: Literal["code", "markdown", "mermaid", "text"] = Field(...)


# Function to generate a response with the chosen format
def generate_response(messages):
    # Format conversation history
    conversation = "\n".join(
        [
            f"{'User' if msg['role']=='user' else 'Assistant'}: {msg.get('content', msg.get('explanation', ''))}"
            for msg in messages[:-1]  # Exclude current message
        ]
    )

    current_msg = messages[-1]["content"]

    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content="""You are a helpful programming assistant.
            Your will give a detailed answer.
            If needed for the clarity of the answer, you can add a vizualisation as a mermaid diagram, code, markdown or text.
            
            For mermaid diagrams, follow these rules:
            - Use '-->' for arrows (not -->)
            - Start with graph direction (TD, LR, etc.)
            - Each node definition on new line with \\n
            - Escape special characters
            
            Example mermaid format:
            {
                "explanation": "Here's a flowchart example",
                "viz_content": "graph TD\\n    A[Start] ---> B[Process]\\n    B ---> C[End]",
                "viz_type": "mermaid"
            }
            
            For markdown content, follow these rules:
            - Use valid markdown syntax
            - Use \\n for newlines
            - Escape special characters
            
            Example markdown format:
            {
                "explanation": "Here's a markdown example",
                "viz_content": "# Title\\n\\nSome **bold** text and a [link](https://example.com)",
                "viz_type": "markdown"
            }
            
            For code content, follow these rules:
            - Use valid code syntax
            - Use \\n for newlines
            - Escape special characters
            
            Example code format:
            {
                "explanation": "Here's a code example",
                "viz_content": "def hello_world():\\n    print('Hello, world!')",
                "viz_type": "code"
            }
            
            Rules for all responses:
            - Must be valid JSON
            - Use \\n for newlines
            - Use \\" for quotes
            - No raw backticks
            """
            ),
            HumanMessage(
                content=f"""
            Previous conversation:
            {conversation}
            
            Current user message: {current_msg}"""
            ),
        ]
    )

    model = ChatOpenAI(model="gpt-4", temperature=0)
    parser = JsonOutputParser(pydantic_object=LLMOutput)

    chain = prompt | model | parser
    response = chain.invoke({"messages": messages})

    print(response)

    if not isinstance(response, dict):
        raise OutputParserException(f"Invalid json output: {response}")

    return response


st.title("Copilot style chat")

# Initialize session state
if "messages" not in st.session_state:
    welcome_message = {
        "role": "assistant",
        "explanation": "Hello! I'm your AI programming assistant. How can I help you today?",
        "viz_content": "",
        "viz_type": "markdown",
    }
    st.session_state.messages = [welcome_message]

# Create two columns without specific ratios for full width
chat_col, viz_col = st.columns(2)

with chat_col:
    st.title("Chat")

    # Create a container for messages
    message_container = st.container()
    input_container = st.container()

    # Chat input at the bottom
    with input_container:
        if prompt := st.chat_input("What's on your mind?"):
            # Add user message to state
            st.session_state.messages.append({"role": "user", "content": prompt})

            # Generate and add AI response to state
            response = generate_response(st.session_state.messages)
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "explanation": response["explanation"],
                    "viz_content": response["viz_content"],
                    "viz_type": response["viz_type"],
                }
            )

    # Display all messages after state updates
    with message_container:
        for message in st.session_state.messages:
            if message["role"] == "user":
                with st.chat_message("user"):
                    st.write(message["content"])
            else:
                with st.chat_message("assistant"):
                    st.write(message["explanation"])

    # Force the input to the bottom using an empty space
    st.markdown("#")

with viz_col:
    st.title("Visualization")
    # Add some padding at the top to align with chat
    st.markdown("##")
    # Add visualization code here based on the latest message
    if st.session_state.messages and "viz_content" in st.session_state.messages[-1]:
        last_msg = st.session_state.messages[-1]
        if last_msg["viz_content"]:
            if last_msg["viz_type"] == "markdown":
                st.markdown(last_msg["viz_content"])
            elif last_msg["viz_type"] == "mermaid":
                stmd.st_mermaid(last_msg["viz_content"])
            elif last_msg["viz_type"] == "code":
                st.code(last_msg["viz_content"])
            else:
                st.write(last_msg["viz_content"])
