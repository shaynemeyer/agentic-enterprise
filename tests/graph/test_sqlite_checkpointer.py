"""SQLite checkpointer: state is written to a file and survives a reopen.

The app binds AsyncSqliteSaver in its lifespan (Lab 32). Under pytest no lifespan runs, so these tests construct the saver directly against a tmp_path file, drive the graph with it, then reopen a fresh saver on the same file and assert the thread is still there.
"""

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from app.graph.engine import graph_builder


def _fake(*replies: str) -> FakeMessagesListChatModel:
    return FakeMessagesListChatModel(responses=[AIMessage(r) for r in replies])


@pytest.mark.asyncio
async def test_thread_survives_a_fresh_saver_on_the_same_file(tmp_path):
    db = str(tmp_path / "cp.db")
    cfg = {"configurable": {"thread_id": "conv-persist"}}

    async with AsyncSqliteSaver.from_conn_string(db) as saver:
        await saver.setup()
        app = graph_builder.compile(checkpointer=saver)
        await app.ainvoke(
            {"messages": [HumanMessage("the deploy logs show an error")]},
            context={"llm": _fake("restarted it")},
            config=cfg,
        )

    # simulate a process restart: brand-new saver, same file
    async with AsyncSqliteSaver.from_conn_string(db) as saver2:
        app2 = graph_builder.compile(checkpointer=saver2)
        snapshot = await app2.aget_state(cfg)

    texts = [m.content for m in snapshot.values["messages"]]
    assert "the deploy logs show an error" in texts
    assert "restarted it" in texts
    assert snapshot.values["route_to"] == "technical"


@pytest.mark.asyncio
async def test_setup_creates_the_checkpoints_table(tmp_path):
    db = str(tmp_path / "cp.db")
    async with AsyncSqliteSaver.from_conn_string(db) as saver:
        await saver.setup()
        cur = await saver.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        tables = {row[0] for row in await cur.fetchall()}

    assert "checkpoints" in tables
    assert "writes" in tables or "checkpoint_writes" in tables


@pytest.mark.asyncio
async def test_a_second_run_on_the_thread_appends(tmp_path):
    db = str(tmp_path / "cp.db")
    cfg = {"configurable": {"thread_id": "conv-append"}}

    async with AsyncSqliteSaver.from_conn_string(db) as saver:
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
