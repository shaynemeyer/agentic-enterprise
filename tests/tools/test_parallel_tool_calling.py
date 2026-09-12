import time

import pytest
from langchain_core.messages import AIMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from app.graph.tools import weather_tool

LATENCY_TOLERANCE_S = 1.0  # generous margin over the tool's own 2s sleep


@pytest.mark.asyncio
async def test_tool_node_runs_calls_concurrently():
    # ToolNode.ainvoke() needs a Runtime object in its invocation config on
    # langgraph 1.2.11 (see [[langgraph-version-drift]] in project memory) -
    # that only exists when ToolNode runs as a node inside a compiled graph,
    # so a bare `ToolNode(...).ainvoke({...})` raises
    # "Missing required config key 'N/A' for 'tools'". A minimal one-node
    # graph supplies it while still isolating ToolNode's own concurrency
    # from the model, per this test's original intent.
    node = ToolNode([weather_tool])
    graph_builder = StateGraph(MessagesState)
    graph_builder.add_node("tools", node)
    graph_builder.add_edge(START, "tools")
    graph_builder.add_edge("tools", END)
    graph = graph_builder.compile()

    message = AIMessage(
        content="",
        tool_calls=[
            {"name": "get_weather", "args": {"city": "London"}, "id": "call_1"},
            {"name": "get_weather", "args": {"city": "Tokyo"}, "id": "call_2"},
        ],
    )

    start = time.perf_counter()
    result = await graph.ainvoke({"messages": [message]})
    duration = time.perf_counter() - start

    tool_messages = [m for m in result["messages"] if m.type == "tool"]
    assert len(tool_messages) == 2
    assert duration < 2 + LATENCY_TOLERANCE_S, (
        f"two 2s tool calls took {duration:.2f}s - expected ~2s if concurrent"
    )
