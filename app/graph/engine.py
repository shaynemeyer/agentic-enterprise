import logging
import operator
from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from app.core.context import get_request_id
from app.core.llm import get_sovereign_llm
from app.graph.tools import tools


class AgentState(TypedDict):
    # add_messages: append new messages, replace existing ones by id.
    messages: Annotated[list[BaseMessage], add_messages]

    # No reducer -> overwrite. Only the latest status matters.
    status: str

    # operator.add on a plain list -> append-only. An audit trail every
    # node can add a line to without reading the existing list first.
    internal_logs: Annotated[list[str], operator.add]


llm = get_sovereign_llm().bind_tools(tools)


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


# Initialize the Graph with our State schema
graph_builder = StateGraph(AgentState)


# Add our node to the graph
graph_builder.add_node("agent", call_model)

graph_builder.add_node("tools", ToolNode(tools))

# Define the flow:
graph_builder.add_edge(START, "agent")

# After 'agent': if the last message has tool_calls -> "tools", else -> END.
graph_builder.add_conditional_edges("agent", tools_condition)

# After a tool runs, go back to 'agent' to use the result.
graph_builder.add_edge("tools", "agent")


# Compile the graph into an executable workflow
workflow = graph_builder.compile()

if __name__ == "__main__":
    import asyncio

    from langchain_core.messages import HumanMessage

    initial_input = {
        "messages": [HumanMessage("What version is agent-api running?")],
        "status": "starting",
    }

    async def _main() -> None:
        async for step in workflow.astream(initial_input, stream_mode="values"):
            step["messages"][-1].pretty_print()

    asyncio.run(_main())
