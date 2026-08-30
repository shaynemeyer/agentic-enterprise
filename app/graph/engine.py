import logging
import operator
from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from app.core.context import get_request_id
from app.core.llm import get_sovereign_llm


class AgentState(TypedDict):
    # add_messages: append new messages, replace existing ones by id.
    messages: Annotated[list[BaseMessage], add_messages]

    # No reducer -> overwrite. Only the latest status matters.
    status: str

    # operator.add on a plain list -> append-only. An audit trail every
    # node can add a line to without reading the existing list first.
    internal_logs: Annotated[list[str], operator.add]


llm = get_sovereign_llm()


async def call_model(state: AgentState) -> dict:
    """Invoke the LLM on the conversation so far and append its reply to state."""
    logger = logging.getLogger("enterprise_agent.graph")
    logger.info("node=agent request_id=%s", get_request_id() or "-")

    writer = get_stream_writer()
    writer({"status": "invoking model"})

    response = await llm.ainvoke(state["messages"])

    writer({"status": "model responded"})

    return {
        "messages": [response],
        "status": "completed",
        "internal_logs": ["agent: model call complete"],
    }


# 1. Initialize the Graph with our State schema
graph_builder = StateGraph(AgentState)


# 2. Add our node to the graph
graph_builder.add_node("agent", call_model)


# 3. Define the flow: Start -> Agent -> End
graph_builder.add_edge(START, "agent")
graph_builder.add_edge("agent", END)


# 4. Compile the graph into an executable workflow
workflow = graph_builder.compile()

if __name__ == "__main__":
    import asyncio

    from langchain_core.messages import HumanMessage

    initial_input = {
        "messages": [
            HumanMessage(
                "Hello, describe the power of agentic workflows in one sentence."
            )
        ],
        "status": "starting",
    }

    final_state = asyncio.run(workflow.ainvoke(initial_input))

    print(f"Logs: {final_state['internal_logs']}")
    print("--- Final Agent State ---")
    print(f"Status: {final_state['status']}")
    print(f"Response: {final_state['messages'][-1].content}")
