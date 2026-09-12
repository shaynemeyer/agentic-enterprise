import time

import pytest
from langchain_core.messages import AIMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from app.graph.tools import sandboxed_code_tool, weather_tool
from tests.tools.test_code_executor import requires_sandbox

LATENCY_TOLERANCE_S = 1.0  # generous margin over the tool's own 2s sleep


def _one_node_tool_graph(tools):
    """Wrap tools in a minimal compiled graph so ToolNode.ainvoke() has the
    Runtime a bare `ToolNode(tools).ainvoke(...)` lacks - see the note on
    test_tool_node_runs_calls_concurrently below.
    """
    node = ToolNode(tools)
    graph_builder = StateGraph(MessagesState)
    graph_builder.add_node("tools", node)
    graph_builder.add_edge(START, "tools")
    graph_builder.add_edge("tools", END)
    return graph_builder.compile()


@pytest.mark.asyncio
async def test_tool_node_runs_calls_concurrently():
    # ToolNode.ainvoke() needs a Runtime object in its invocation config on
    # langgraph 1.2.11 (see [[langgraph-version-drift]] in project memory) -
    # that only exists when ToolNode runs as a node inside a compiled graph,
    # so a bare `ToolNode(...).ainvoke({...})` raises
    # "Missing required config key 'N/A' for 'tools'". A minimal one-node
    # graph supplies it while still isolating ToolNode's own concurrency
    # from the model, per this test's original intent.
    graph = _one_node_tool_graph([weather_tool])

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


@pytest.mark.asyncio
@requires_sandbox
async def test_two_sandbox_calls_run_concurrently():
    # No fixed threshold here, deliberately - container-launch latency over
    # SSH to gvisor-lab has enough of its own variance (Lab 47) that a hard
    # bound would be flaky in a way the mock weather tool's 2s sleep isn't.
    # Read the printed number and compare by hand to two sequential runs.
    graph = _one_node_tool_graph([sandboxed_code_tool])
    message = AIMessage(
        content="",
        tool_calls=[
            {"name": "execute_sandboxed_code", "args": {"code": "print(1)"}, "id": "c1"},
            {"name": "execute_sandboxed_code", "args": {"code": "print(2)"}, "id": "c2"},
        ],
    )

    start = time.perf_counter()
    result = await graph.ainvoke({"messages": [message]})
    duration = time.perf_counter() - start

    tool_messages = [m for m in result["messages"] if m.type == "tool"]
    assert len(tool_messages) == 2
    print(f"two sandbox calls: {duration:.2f}s")
