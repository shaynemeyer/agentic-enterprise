"""Checkpoint GC: a thread past the cutoff is deleted; a fresh one is kept.

Requires a running Postgres. Set CHECKPOINT_TEST_DSN, e.g.
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
from app.graph.gc import find_stale_threads, sweep

DB_URL = os.getenv("CHECKPOINT_TEST_DSN")
pytestmark = pytest.mark.skipif(not DB_URL, reason="CHECKPOINT_TEST_DSN not set")


def _fake(*replies: str) -> FakeMessagesListChatModel:
    return FakeMessagesListChatModel(responses=[AIMessage(r) for r in replies])


@pytest.mark.asyncio
async def test_fresh_thread_is_not_stale():
    tid = f"gc-fresh-{uuid.uuid4()}"
    cfg = {"configurable": {"thread_id": tid}}
    async with AsyncPostgresSaver.from_conn_string(DB_URL) as saver:
        await saver.setup()
        app = graph_builder.compile(checkpointer=saver)
        await app.ainvoke(
            {"messages": [HumanMessage("hello")]},
            context={"llm": _fake("hi")},
            config=cfg,
        )
        stale = await find_stale_threads(saver, retention_days=1)
        assert tid not in stale
        await saver.adelete_thread(tid)  # cleanup


@pytest.mark.asyncio
async def test_sweep_deletes_a_thread_past_the_cutoff():
    tid = f"gc-stale-{uuid.uuid4()}"
    cfg = {"configurable": {"thread_id": tid}}
    async with AsyncPostgresSaver.from_conn_string(DB_URL) as saver:
        await saver.setup()
        app = graph_builder.compile(checkpointer=saver)
        await app.ainvoke(
            {"messages": [HumanMessage("hello")]},
            context={"llm": _fake("hi")},
            config=cfg,
        )
        # Backdate this thread's checkpoint ts so retention_days=1 catches it.
        await saver.conn.execute(
            "UPDATE checkpoints "
            "SET checkpoint = jsonb_set("
            "  checkpoint, '{ts}', to_jsonb((now() - interval '10 days')::text)) "
            "WHERE thread_id = %s",
            (tid,),
        )

        summary = await sweep(saver, retention_days=1)
        assert summary["deleted"] >= 1

        snapshot = await app.aget_state(cfg)
        assert snapshot.values == {}  # thread is gone
