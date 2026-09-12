"""The autonomous research graph (Lab 50).

A standalone cyclic graph, not an addition to app/graph/engine.py's
GraphState/route_request dispatch graph - a genuinely different control-flow
shape (an open-ended agent -> tools -> agent loop) is simpler as its own
small state than folded into the main graph's fixed Literal routing.

Composes Labs 44-49 unchanged: load_tools() already returns every MCP tool,
sandboxed_code_tool, weather_tool, and market_metrics_tool, so ToolNode here
is the same construction as engine.py's "tools" node.
"""

from typing import Annotated, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.runtime import Runtime

from app.graph.engine import RuntimeContext, checkpointer, format_tool_error
from app.graph.tools import load_tools
from app.schemas.agent_schema import ResearchReport


class ResearchState(TypedDict):
    messages: Annotated[list, add_messages]
    report: ResearchReport | None


async def researcher(state: ResearchState, runtime: Runtime[RuntimeContext]) -> dict:
    """The loop's brain: decide whether to call a tool or stop."""
    llm = runtime.context.tool_llm
    response = await llm.ainvoke(state["messages"])
    return {"messages": [response]}


def should_continue(state: ResearchState) -> Literal["tools", "report"]:
    """Real loop exit: the model's own tool_calls decide, not a counted budget."""
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return "report"


async def generate_research_report(
    state: ResearchState, runtime: Runtime[RuntimeContext]
) -> dict:
    """Same with_structured_output pattern as Lab 49's generate_structured_report -
    unbound llm, not tool_llm, for the same reason (forcing a schema and
    leaving tool choice open conflict on one invoke() call)."""
    structured_llm = runtime.context.llm.with_structured_output(ResearchReport)
    transcript = "\n".join(str(m.content) for m in state["messages"])
    report = await structured_llm.ainvoke(
        "Synthesize a ResearchReport from this research transcript. "
        "confidence must be a fraction between 0.0 and 1.0 (not a percentage, "
        f"not out of 100):\n{transcript}"
    )
    return {"report": report}


_research_builder = StateGraph(ResearchState, context_schema=RuntimeContext)
_research_builder.add_node("agent", researcher)
_research_builder.add_node("report", generate_research_report)

_research_builder.add_edge(START, "agent")
_research_builder.add_conditional_edges("agent", should_continue)
_research_builder.add_edge("tools", "agent")
_research_builder.add_edge("report", END)

_research_app = None


async def build_research_workflow():
    """The compiled research graph, built once and cached.

    Async: the "tools" node's list comes from load_tools(), an MCP call that
    can't run at module import - same reason engine.py's _ensure_tools_node
    defers its ToolNode construction.
    """
    global _research_app
    if _research_app is None:
        _research_builder.add_node(
            "tools", ToolNode(await load_tools(), handle_tool_errors=format_tool_error)
        )
        _research_app = _research_builder.compile(checkpointer=checkpointer)
    return _research_app
