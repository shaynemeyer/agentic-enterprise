"""Postgres checkpointer: state is shared across savers and survives a reconnect.

Requires a running Postgres. Set CHECKPOINT_TEST_DSN, e.g.
  export CHECKPOINT_TEST_DSN=postgresql://agent:agent@localhost:5433/agent_db
Skipped when it is unset (CI without a DB, quick local runs).

The name deliberately avoids a "..._db_url" suffix: Settings forbids extra env
vars, so anything matching a field-name prefix breaks every Settings() call.

Each test uses a unique thread_id, so the tests are safe to run against a shared
dev database without a teardown.
"""

import os
import uuid

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from app.graph.engine import graph_builder

DB_URL = os.getenv("CHECKPOINT_TEST_DSN")
pytestmark = pytest.mark.skipif(not DB_URL, reason="CHECKPOINT_TEST_DSN not set")


def _fake(*replies: str) -> FakeMessagesListChatModel:
    return FakeMessagesListChatModel(responses=[AIMessage(r) for r in replies])


@pytest.mark.asyncio
async def test_thread_is_visible_to_a_second_saver_on_the_same_db():
    cfg = {"configurable": {"thread_id": f"conv-{uuid.uuid4()}"}}

    async with AsyncPostgresSaver.from_conn_string(DB_URL) as saver:
        await saver.setup()
        app = graph_builder.compile(checkpointer=saver)
        await app.ainvoke(
            {"messages": [HumanMessage("the deploy logs show an error")]},
            context={"llm": _fake("restarted it")},
            config=cfg,
        )

    # a separate connection - what a second worker / replica would use
    async with AsyncPostgresSaver.from_conn_string(DB_URL) as saver2:
        app2 = graph_builder.compile(checkpointer=saver2)
        snapshot = await app2.aget_state(cfg)

    texts = [m.content for m in snapshot.values["messages"]]
    assert "the deploy logs show an error" in texts
    assert "restarted it" in texts
    assert snapshot.values["route_to"] == "technical"


@pytest.mark.asyncio
async def test_setup_creates_the_checkpoints_table():
    async with AsyncPostgresSaver.from_conn_string(DB_URL) as saver:
        await saver.setup()
        # LangGraph opens the connection with row_factory=dict_row, so rows are
        # dicts keyed by column name - not tuples.
        async with saver.conn.cursor() as cur:
            await cur.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_name LIKE 'checkpoint%'"
            )
            tables = {row["table_name"] for row in await cur.fetchall()}

    assert "checkpoints" in tables
    assert "checkpoint_writes" in tables


@pytest.mark.asyncio
async def test_a_second_run_on_the_thread_appends():
    cfg = {"configurable": {"thread_id": f"conv-{uuid.uuid4()}"}}

    async with AsyncPostgresSaver.from_conn_string(DB_URL) as saver:
        await saver.setup()
        app = graph_builder.compile(checkpointer=saver)
        await app.ainvoke(
            {"messages": [HumanMessage("the deploy logs show an error")]},
            context={"llm": _fake("restarted it")},
            config=cfg,
        )
        result = await app.ainvoke(
            {"messages": [HumanMessage("what is the status now?")]},
            context={"llm": _fake("all green")},
            config=cfg,
        )

    texts = [m.content for m in result["messages"]]
    assert "the deploy logs show an error" in texts
    assert texts[-1] == "all green"
