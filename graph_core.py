from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph

from app.core.llm import get_sovereign_llm


class AgentState(TypedDict):
    # The 'messages' key will store our conversation history
    messages: Annotated[list[str], "The conversation history"]

    # We can add metadata like 'status' to track execution
    status: str


llm = get_sovereign_llm()


def call_model(state: AgentState) -> dict:
    """Invoke the LLM on the latest message and append its reply to state."""
    prompt = state["messages"][-1]
    response = llm.invoke(prompt)

    return {
        "messages": state["messages"] + [response.content],
        "status": "completed",
    }


# 1. Initialize the Graph with our State schema
workflow = StateGraph(AgentState)


# 2. Add our node to the graph
workflow.add_node("agent", call_model)


# 3. Define the flow: Start -> Agent -> End
workflow.add_edge(START, "agent")
workflow.add_edge("agent", END)


# 4. Compile the graph into an executable 'app'
app = workflow.compile()

if __name__ == "__main__":
    initial_input = {
        "messages": ["Hello, describe the power of agentic workflows in one sentence."],
        "status": "starting",
    }

    final_state = app.invoke(initial_input)

    print("--- Final Agent State ---")
    print(f"Status: {final_state['status']}")
    print(f"Response: {final_state['messages'][-1]}")
