import logging
import operator
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AIMessage, BaseMessage
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

    # No reducer -> overwrite. Router writes it once; downstream reads it.
    route_to: Literal["technical", "billing", "general"]


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


async def route_request(state: AgentState) -> dict:
    """Classify the last human message and write the routing decision to state."""
    logger = logging.getLogger("enterprise_agent.graph")

    text = state["messages"][-1].content.lower()

    if any(w in text for w in ("deploy", "status", "error", "logs", "service")):
        decision = "technical"
    elif any(w in text for w in ("invoice", "payment", "billing", "charge")):
        decision = "billing"
    else:
        decision = "general"

    logger.info(
        "node=router request_id=%s route_to=%s", get_request_id() or "-", decision
    )

    writer = get_stream_writer()
    writer({"status": f"routed to {decision}"})

    return {
        "route_to": decision,
        "status": "routed",
        "internal_logs": [f"router: classified as {decision}"],
    }


async def billing_worker(state: AgentState) -> dict:
    """Terminal node for billing requests. Stubbed."""
    reply = AIMessage("This is the billing department. A stub - no records wired yet.")
    return {
        "messages": [reply],
        "status": "completed",
        "internal_logs": ["billing: stub reply"],
    }


async def general_worker(state: AgentState) -> dict:
    """Terminal node for general inquiries. Stubbed."""
    reply = AIMessage("General enquiries desk. A stub - no knowledge base wired yet.")
    return {
        "messages": [reply],
        "status": "completed",
        "internal_logs": ["general: stub reply"],
    }


def pick_route(state: AgentState) -> Literal["agent", "billing", "general"]:
    """Read the router's decision and name the next node."""
    decision = state.get("route_to", "general")
    if decision == "technical":
        return "agent"
    if decision == "billing":
        return "billing"
    return "general"


# Initialize the Graph with our State schema
graph_builder = StateGraph(AgentState)

# Add our node to the graph
graph_builder.add_node("router", route_request)
graph_builder.add_node("agent", call_model)
graph_builder.add_node("tools", ToolNode(tools))
graph_builder.add_node("billing", billing_worker)
graph_builder.add_node("general", general_worker)

# Define the flow:
graph_builder.add_edge(START, "router")

# The new conditional edge: router -> one of three nodes, on route_to.
graph_builder.add_conditional_edges("router", pick_route)

# if the last message has tool_calls -> "tools", else -> END.
graph_builder.add_conditional_edges("agent", tools_condition)
graph_builder.add_edge("tools", "agent")

graph_builder.add_edge("billing", END)
graph_builder.add_edge("general", END)

# Compile the graph into an executable workflow
workflow = graph_builder.compile()

if __name__ == "__main__":
    import asyncio

    from langchain_core.messages import HumanMessage

    prompts = [
        "What version is agent-api running?",  # -> technical -> tool loop
        "My last invoice looks wrong.",  # -> billing
        "What does this platform do?",  # -> general
    ]

    async def _main() -> None:
        for p in prompts:
            print(f"\n=== {p} ===")
            state = {"messages": [HumanMessage(p)], "status": "starting"}
            async for step in workflow.astream(state, stream_mode="values"):
                step["messages"][-1].pretty_print()

    asyncio.run(_main())
