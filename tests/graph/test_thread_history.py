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
from app.graph.history import is_interrupted, thread_timeline

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
