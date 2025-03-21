import os
from typing_extensions import TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from langgraph.types import interrupt
from langgraph.checkpoint.memory import InMemorySaver
from langgraph_supervisor import create_supervisor
from langgraph.prebuilt import create_react_agent
from langgraph.prebuilt.chat_agent_executor import AgentState


# Load environment variables
load_dotenv()

# Initialize LLM
model = ChatOpenAI(model="gpt-4o", openai_api_key=os.environ["OPENAI_KEY"])

checkpointer = InMemorySaver()


class State(AgentState):
    feedback: str


def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


def web_search(query: str) -> str:
    """Search the web for information."""
    return (
        "Here are the headcounts for each of the FAANG companies in 2024:\n"
        "1. **Facebook (Meta)**: 67,317 employees.\n"
        "2. **Apple**: 164,000 employees.\n"
        "3. **Amazon**: 1,551,000 employees.\n"
        "4. **Netflix**: 14,000 employees.\n"
        "5. **Google (Alphabet)**: 181,269 employees."
    )


def human_node(state: State):
    value = interrupt(
        # Any JSON serializable value to surface to the human.
        # For example, a question or a piece of text or a set of keys in the state
        {"text_to_revise": state["feedback"]}
    )
    # Update the state with the human's input or route the graph based on the input.
    return {"some_text": value}


math_agent = create_react_agent(
    model=model,
    tools=[add, multiply],
    name="math_expert",
    prompt="You are a math expert. Always use one tool at a time.",
)

research_agent = create_react_agent(
    model=model,
    tools=[web_search],
    name="research_expert",
    prompt="You are a world class researcher with access to web search. Do not do any math.",
)

# Create supervisor workflow
workflow = create_supervisor(
    agents=[research_agent, math_agent],
    model=model,
    prompt=(
        "You are a team supervisor managing a research expert and a math expert. "
        "For current events, use research_agent. "
        "For math problems, use math_agent."
    ),
    state_schema=State,
    output_mode="last_message",
)

# Compile and run
app = workflow.compile(
    checkpointer=checkpointer,
)
result = app.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "what's the combined headcount of the FAANG companies in 2024?",
            }
        ],
    },
    {"configurable": {"thread_id": "1", "user_id": "1"}},
)

for m in result["messages"]:
    m.pretty_print()
