import logging
from dataclasses import dataclass
from typing import Annotated, Literal, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.config import get_config, get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.runtime import Runtime
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.context import get_request_id
from app.core.llm import get_sovereign_llm
from app.graph.tools import tools

from app.graph.ownership import is_admin, owned_thread_ids
from app.memory.vector_store import search


def merge_logs(existing: list[str] | None, update: list[str]) -> list[str]:
    """Append-only merge for internal_logs.

    Concatenates like operator.add, but tolerates a None left side and skips
    lines already present so a cyclic node re-entry cannot duplicate history.
    """
    base = existing or []
    return base + [line for line in update if line not in base]


class GraphInput(TypedDict):
    """What a caller is allowed to pass to the graph. Anything else is dropped."""

    messages: Annotated[list[BaseMessage], add_messages]


class GraphState(TypedDict):
    """The full shared scratchpad. Every node reads and writes this shape"""

    messages: Annotated[list[BaseMessage], add_messages]
    status: str
    internal_logs: Annotated[list[str], merge_logs]
    route_to: Literal["technical", "billing", "general"]
    # No reducer -> overwrite. The general worker reads the current value and
    # returns current + 1; the critic reads it to decide whether to stop.
    revision_count: int
    # No reducer -> overwrite. The critic's note back to the worker; "" on the
    # first pass, "PASS" once the draft clears the bar.
    critique: str
    # No reducer -> overwrite, same convention as route_to/critique above:
    # each run's retrieval replaces the last, it does not accumulate.
    retrieved_context: list[str]


class GraphOutput(TypedDict):
    """What the compiled graph returns. The scratchpad keys are not here."""

    messages: Annotated[list[BaseMessage], add_messages]
    status: str


@dataclass
class RuntimeContext:
    """Per-request dependencies injected by the caller via `config`.

    Not part of state - the caller supplies this at invoke time, nodes only
    read it. Lets one compiled graph serve different LLM backends, DB sessions,
    or a no-network test double without touching module globals.
    """

    llm: BaseChatModel
    db: AsyncSession
    username: str


async def call_model(state: GraphState, runtime: Runtime[RuntimeContext]) -> dict:
    """Invoke the LLM on the conversation so far."""
    logger = logging.getLogger("enterprise_agent.graph")
    logger.info("node=agent request_id=%s", get_request_id() or "-")

    writer = get_stream_writer()
    writer({"status": "invoking model"})

    llm = runtime.context.llm
    messages = state["messages"]

    if state.get("retrieved_context"):
        context_block = "\n".join(f"- {fact}" for fact in state["retrieved_context"])
        system_prompt = SystemMessage(
            content=(
                "The following are past facts recorded in other conversations, "
                "not part of the current exchange. Use them only if relevant:\n"
                f"{context_block}"
            )
        )
        messages = [system_prompt, *messages]

    response = await llm.ainvoke(messages)

    writer({"status": "model responded"})

    return {
        "messages": [response],
        "status": "completed",
        "internal_logs": ["agent: model call complete"],
    }


async def route_request(state: GraphState) -> dict:
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


# Neither Ollama nor vLLM exposes a client-side tokenizer the way tiktoken
# does for OpenAI - Previously we hit the same gap for embeddings
# (check_embedding_ctx_length=False) and took the same
# position there: don't guess at a model-specific tokenizer, use the
# server's own accounting. ~4 chars/token is a widely-cited rule of thumb
# for English text (OpenAI's own tokenizer help page states it:
# https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them)
# and is model-agnostic enough to serve as a deliberately conservative
# budget - not a real count, and not claimed to be one (Roadmap).
CHARS_PER_TOKEN_ESTIMATE = 4
MAX_CONTEXT_TOKENS = 500
MAX_CONTEXT_CHARS = MAX_CONTEXT_TOKENS * CHARS_PER_TOKEN_ESTIMATE


async def retrieve_semantic_memories(
    state: GraphState, runtime: Runtime[RuntimeContext]
) -> dict:
    """Query the cross-thread vector store with the latest user turn,
    restricted to threads the calling user actually owns.

    Runs only when pick_route has already classified this turn as needing
    it - not on every message, unlike the source material's
    unconditional node. thread_id (for logging only, not the filter) comes
    from the run's own config, the same mechanism get_request_id() and
    get_stream_writer() already use for per-run values GraphState doesn't
    carry.
    """
    logger = logging.getLogger("enterprise_agent.graph")
    thread_id = get_config().get("configurable", {}).get("thread_id", "")
    query = state["messages"][-1].content

    username = runtime.context.username
    if is_admin(username):
        hits = await search(query, limit=2)
    else:
        allowed = await owned_thread_ids(runtime.context.db, username)
        hits = await search(query, limit=2, thread_ids=allowed)

    logger.info(
        "node=memory_retriever request_id=%s thread_id=%s user=%s hits=%d",
        get_request_id() or "-",
        thread_id,
        username,
        len(hits),
    )

    context: list[str] = []
    budget = MAX_CONTEXT_CHARS
    for hit in hits:
        text = hit["text"]
        if len(text) > budget:
            break
        context.append(text)
        budget -= len(text)

    return {
        "retrieved_context": context,
        "internal_logs": ["memory_retriever: fetched cross-thread context"],
    }


async def billing_worker(state: GraphState) -> dict:
    """Terminal node for billing requests. Stubbed."""
    reply = AIMessage("This is the billing department. A stub - no records wired yet.")
    return {
        "messages": [reply],
        "status": "completed",
        "internal_logs": ["billing: stub reply"],
    }


async def general_worker(state: GraphState) -> dict:
    """Draft a general-enquiries answer. Re-entered by the critic loop."""
    logger = logging.getLogger("enterprise_agent.graph")

    count = state.get("revision_count", 0)
    note = state.get("critique", "")

    # Stub answer. A real worker would call a retrieval step here; the point of
    # this is the loop, not the content. The draft "improves" only in that
    # it acknowledges the critic's note on a re-run.
    if note and note != "PASS":
        reply = AIMessage(f"General enquiries desk (revised). Addressing: {note}")
    else:
        reply = AIMessage(
            "General enquiries desk. A stub - no knowledge base wired yet."
        )

    logger.info(
        "node=general request_id=%s attempt=%d", get_request_id() or "-", count + 1
    )

    writer = get_stream_writer()
    writer({"status": f"general draft attempt {count + 1}"})

    return {
        "messages": [reply],
        "status": "drafted",
        "revision_count": count + 1,
        "internal_logs": [f"general: draft attempt {count + 1}"],
    }


GENERAL_REVISION_LIMIT = 3


async def critic(state: GraphState) -> dict:
    """Grade the general worker's latest draft. Terminal decision lives in the edge."""
    logger = logging.getLogger("enterprise_agent.graph")

    draft = state["messages"][-1].content
    count = state["revision_count"]

    # Deterministic bar: the draft must be more than a bare stub.
    # A real critic would score relevance / grounding with an LLM call.
    passes = "revised" in draft.lower() or len(draft) > 90

    if passes:
        verdict = "PASS"
    elif count >= GENERAL_REVISION_LIMIT:
        verdict = "PASS"  # out of attempts - ship what we have
        logger.warning(
            "node=critic request_id=%s revision limit hit, forcing PASS",
            get_request_id() or "-",
        )
    else:
        verdict = "draft is a bare stub; add detail"

    logger.info(
        "node=critic request_id=%s attempt=%d verdict=%s",
        get_request_id() or "-",
        count,
        verdict,
    )

    writer = get_stream_writer()
    writer({"status": f"critic verdict: {verdict}"})

    return {
        "critique": verdict,
        "status": "critiqued",
        "internal_logs": [f"critic: attempt {count} -> {verdict}"],
    }


def pick_route(state: GraphState) -> Literal["agent", "billing", "general"]:
    """Read the router's decision and name the next node."""
    decision = state.get("route_to", "general")
    if decision == "technical":
        return "agent"
    if decision == "billing":
        return "billing"
    return "general"


def after_critic(state: GraphState) -> Literal["general", "__end__"]:
    """PASS -> stop; anything else -> back to the general worker."""
    if state["critique"] == "PASS":
        return END
    return "general"


graph_builder = StateGraph(
    GraphState,
    input_schema=GraphInput,
    output_schema=GraphOutput,
    context_schema=RuntimeContext,
)

graph_builder.add_node("router", route_request)
graph_builder.add_node("agent", call_model)
graph_builder.add_node("tools", ToolNode(tools))
graph_builder.add_node("billing", billing_worker)
graph_builder.add_node("general", general_worker)
graph_builder.add_node("critic", critic)

graph_builder.add_edge(START, "router")
graph_builder.add_node("memory_retriever", retrieve_semantic_memories)

graph_builder.add_conditional_edges("router", pick_route)
# pick_route's "technical" branch now points at memory_retriever instead
# of directly at "agent" - update the branch mapping pick_route returns,
# then:
graph_builder.add_edge("memory_retriever", "agent")

graph_builder.add_conditional_edges("agent", tools_condition)
graph_builder.add_edge("tools", "agent")

graph_builder.add_edge("billing", END)

graph_builder.add_edge("general", "critic")
graph_builder.add_conditional_edges("critic", after_critic)

# In-memory thread-level memory. Volatile - state lives in this process only
# and is lost on restart.
checkpointer = InMemorySaver()
workflow = graph_builder.compile(checkpointer=checkpointer)


def graph_mermaid() -> str:
    """The compiled graph as a Mermaid flowchart. Regenerate docs from this."""
    return workflow.get_graph().draw_mermaid()


def graph_png() -> bytes:
    """PNG of the compiled graph. Calls the mermaid.ink API - needs outbound HTTPS."""
    return workflow.get_graph().draw_mermaid_png()


if __name__ == "__main__":
    import asyncio

    from langchain_core.messages import HumanMessage

    prompts = [
        "What version is agent-api running?",  # -> technical -> tool loop
        "My last invoice looks wrong.",  # -> billing
        "What does this platform do?",  # -> general
    ]

    async def _main() -> None:
        demo_llm = get_sovereign_llm().bind_tools(tools)
        for p in prompts:
            print(f"\n=== {p} ===")
            state = {"messages": [HumanMessage(p)], "status": "starting"}
            async for step in workflow.astream(
                state,
                stream_mode="values",
                context={"llm": demo_llm},
                output_keys=list(GraphState.__annotations__),
            ):
                step["messages"][-1].pretty_print()

    asyncio.run(_main())
