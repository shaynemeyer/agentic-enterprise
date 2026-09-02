"""Thread history and resume against a real Postgres checkpoint store.

  export CHECKPOINT_TEST_DSN=postgresql://agent:agent@localhost:5433/agent_db
Skipped when unset.
"""

import os
import uuid

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from app.graph.engine import graph_builder
from app.graph.history import (
    branch_tree,
    checkpoint_config,
    edit_checkpoint,
    is_interrupted,
    thread_timeline,
)

DB_URL = os.getenv("CHECKPOINT_TEST_DSN")
pytestmark = pytest.mark.skipif(not DB_URL, reason="CHECKPOINT_TEST_DSN not set")


def _fake(*replies: str) -> FakeMessagesListChatModel:
    return FakeMessagesListChatModel(responses=[AIMessage(r) for r in replies])


@pytest.mark.asyncio
async def test_completed_thread_has_history_and_is_not_interrupted():
    tid = f"hist-{uuid.uuid4()}"
    cfg = {"configurable": {"thread_id": tid}}
    async with AsyncPostgresSaver.from_conn_string(DB_URL) as saver:
        await saver.setup()
        graph = graph_builder.compile(checkpointer=saver)
        await graph.ainvoke(
            {"messages": [HumanMessage("hello")]},
            context={"llm": _fake("hi there")},
            config=cfg,
        )

        timeline = await thread_timeline(graph, tid)
        assert len(timeline) > 1  # one checkpoint per node step
        assert timeline[0]["next"] == []  # newest: nothing pending
        assert await is_interrupted(graph, tid) is False

        await saver.adelete_thread(tid)


@pytest.mark.asyncio
async def test_unknown_thread_has_empty_timeline():
    async with AsyncPostgresSaver.from_conn_string(DB_URL) as saver:
        await saver.setup()
        graph = graph_builder.compile(checkpointer=saver)
        assert await thread_timeline(graph, f"nope-{uuid.uuid4()}") == []


@pytest.mark.asyncio
async def test_edit_writes_a_new_checkpoint_without_touching_the_original():
    tid = f"edit-{uuid.uuid4()}"
    cfg = {"configurable": {"thread_id": tid}}
    async with AsyncPostgresSaver.from_conn_string(DB_URL) as saver:
        await saver.setup()
        graph = graph_builder.compile(checkpointer=saver)
        await graph.ainvoke(
            {"messages": [HumanMessage("hello")]},
            context={"llm": _fake("hi there")},
            config=cfg,
        )

        before = await thread_timeline(graph, tid)
        original_cp = before[0]["checkpoint_id"]
        original_snap_before = await graph.aget_state(checkpoint_config(tid, original_cp))
        original_revision_count = original_snap_before.values["revision_count"]

        new_config = await edit_checkpoint(
            graph,
            tid,
            original_cp,
            {"revision_count": original_revision_count + 100},
            as_node="critic",
        )
        edited_cp = new_config["configurable"]["checkpoint_id"]
        assert edited_cp != original_cp

        # the edit is a new checkpoint; the original is unchanged
        original_snap = await graph.aget_state(checkpoint_config(tid, original_cp))
        assert original_snap.values["revision_count"] == original_revision_count

        edited_snap = await graph.aget_state(new_config)
        assert edited_snap.values["revision_count"] == original_revision_count + 100

        await saver.adelete_thread(tid)


@pytest.mark.asyncio
async def test_branch_tree_groups_fork_children_under_their_parent():
    tid = f"branch-{uuid.uuid4()}"
    cfg = {"configurable": {"thread_id": tid}}
    async with AsyncPostgresSaver.from_conn_string(DB_URL) as saver:
        await saver.setup()
        graph = graph_builder.compile(checkpointer=saver)
        await graph.ainvoke(
            {"messages": [HumanMessage("hello")]},
            context={"llm": _fake("hi there")},
            config=cfg,
        )

        timeline = await thread_timeline(graph, tid)
        # the thread's final checkpoint: next == [] means it has no children yet,
        # so grouping edits under it gives a count with nothing else mixed in
        leaf_cp = timeline[0]["checkpoint_id"]
        assert timeline[0]["next"] == []

        await graph.aupdate_state(
            checkpoint_config(tid, leaf_cp), {"critique": "PASS"}, as_node="critic"
        )
        await graph.aupdate_state(
            checkpoint_config(tid, leaf_cp), {"critique": "retry"}, as_node="critic"
        )

        tree = branch_tree(await thread_timeline(graph, tid))
        assert len(tree[leaf_cp]) == 2  # two edits off the same parent - a fork point

        await saver.adelete_thread(tid)
