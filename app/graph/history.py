"""Read and resume from a thread's checkpoint history.

The compiled graph (app.graph.engine.workflow) exposes the history through
aget_state_history; this module shapes each StateSnapshot into JSON and builds
the config that pins a run to one past checkpoint. No new DB connection - every
call goes through workflow's own AsyncPostgresSaver.
"""

from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import StateSnapshot


def checkpoint_config(
    thread_id: str, checkpoint_id: str | None = None
) -> RunnableConfig:
    """Config for a thread, optionally pinned to one checkpoint.

    With checkpoint_id, an invoke replays from that point and forks a new
    branch; without it, an invoke uses the thread's latest checkpoint.
    """
    configurable: dict[str, Any] = {"thread_id": thread_id}
    if checkpoint_id:
        configurable["checkpoint_id"] = checkpoint_id
    return {"configurable": configurable}


def _message_count(values: dict[str, Any]) -> int:
    msgs = values.get("messages", [])
    return len(msgs) if isinstance(msgs, list) else 0


def snapshot_summary(snap: StateSnapshot) -> dict[str, Any]:
    """One checkpoint as JSON: its id, what was pending, a size hint, timestamp."""
    cfg = snap.config.get("configurable", {})
    return {
        "checkpoint_id": cfg.get("checkpoint_id"),
        "parent_checkpoint_id": (snap.parent_config or {})
        .get("configurable", {})
        .get("checkpoint_id"),
        "next": list(snap.next),
        "message_count": _message_count(snap.values),
        "status": snap.values.get("status"),
        "created_at": snap.created_at,
        "source": snap.metadata.get("source"),
        "step": snap.metadata.get("step"),
    }


async def thread_timeline(
    graph: CompiledStateGraph, thread_id: str, limit: int = 50
) -> list[dict[str, Any]]:
    """Every checkpoint for a thread, newest first, as summary dicts."""
    config = checkpoint_config(thread_id)
    out: list[dict[str, Any]] = []
    async for snap in graph.aget_state_history(config, limit=limit):
        out.append(snapshot_summary(snap))
    return out


async def is_interrupted(graph: CompiledStateGraph, thread_id: str) -> bool:
    """True if the thread's latest checkpoint has pending nodes to run."""
    snap = await graph.aget_state(checkpoint_config(thread_id))
    return bool(snap.next)


async def edit_checkpoint(
    graph: CompiledStateGraph,
    thread_id: str,
    checkpoint_id: str,
    values: dict[str, Any],
    as_node: str,
) -> RunnableConfig:
    """Write `values` onto the checkpoint as if node `as_node` had produced them.

    Returns the config of the new checkpoint aupdate_state creates - pass this
    straight to ainvoke(None, ...) to replay from the correction, or read it
    with aget_state to confirm the edit before replaying.
    """
    return await graph.aupdate_state(
        checkpoint_config(thread_id, checkpoint_id),
        values,
        as_node=as_node,
    )


def branch_tree(timeline: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group a flat timeline by parent_checkpoint_id.

    Keys are parent_checkpoint_id (None for the thread's root checkpoint);
    values are that parent's direct children, in the order thread_timeline
    returned them (newest first). A checkpoint with more than one child here
    is a fork or edit point - the thread diverged there.
    """
    tree: dict[str, list[dict[str, Any]]] = {}
    for entry in timeline:
        tree.setdefault(entry["parent_checkpoint_id"], []).append(entry)
    return tree
