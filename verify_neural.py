from typing import TypedDict

from langgraph.graph import END, StateGraph


class AgentState(TypedDict):
    status: str


def check_logic(state: AgentState) -> AgentState:
    print(f"--- Node execution: {state['status']} ---")
    return {"status": "Verified"}


workflow = StateGraph(AgentState)
workflow.add_node("verify", check_logic)
workflow.set_entry_point("verify")
workflow.add_edge("verify", END)

app = workflow.compile()


if __name__ == "__main__":
    result = app.invoke({"status": "Initializing"})
    print(f"Final state: {result}")
