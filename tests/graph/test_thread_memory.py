"""Thread-level memory: state persists per thread_id, and threads are isolated.

The compiled `workflow` carries an InMemorySaver (Lab 31). These tests drive it
with a fake LLM and assert that a second call on the same thread_id sees the
first call's messages, while a different thread_id sees nothing.
"""

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage

from app.graph.engine import workflow


def _fake(*replies: str) -> FakeMessagesListChatModel:
    return FakeMessagesListChatModel(responses=[AIMessage(r) for r in replies])


@pytest.mark.asyncio
async def test_second_call_on_same_thread_sees_prior_messages():
    cfg = {"configurable": {"thread_id": "conv-a"}}

    # both turns hit the technical path so the agent node runs and consumes the
    # fake reply. "status" keyword on turn two keeps it on that path.
    await workflow.ainvoke(
        {"messages": [HumanMessage("the deploy logs show an error")]},
        context={"llm": _fake("restarted it")},
        config=cfg,
    )
    result = await workflow.ainvoke(
        {"messages": [HumanMessage("what is the status now?")]},
        context={"llm": _fake("all green")},
        config=cfg,
    )

    texts = [m.content for m in result["messages"]]
    # the first turn's human + AI messages are still there, ahead of turn two
    assert "the deploy logs show an error" in texts
    assert "restarted it" in texts
    assert texts[-1] == "all green"


@pytest.mark.asyncio
async def test_a_different_thread_id_is_isolated():
    await workflow.ainvoke(
        {"messages": [HumanMessage("the deploy logs show an error")]},
        context={"llm": _fake("restarted it")},
        config={"configurable": {"thread_id": "conv-a"}},
    )
    result = await workflow.ainvoke(
        {"messages": [HumanMessage("my invoice is wrong")]},
        context={"llm": _fake()},
        config={"configurable": {"thread_id": "conv-b"}},
    )

    texts = [m.content for m in result["messages"]]
    assert "restarted it" not in texts
    assert "the deploy logs show an error" not in texts
    assert "billing department" in texts[-1]


@pytest.mark.asyncio
async def test_missing_thread_id_is_rejected():
    with pytest.raises(ValueError, match="thread_id"):
        await workflow.ainvoke(
            {"messages": [HumanMessage("hello")]},
            context={"llm": _fake("hi")},
        )


@pytest.mark.asyncio
async def test_state_snapshot_carries_the_routing_scratchpad():
    """get_state returns the full GraphState for a thread, including the
    scratchpad keys that GraphOutput filters out of ainvoke's return."""
    cfg = {"configurable": {"thread_id": "conv-snap"}}
    await workflow.ainvoke(
        {"messages": [HumanMessage("my invoice has a wrong charge")]},
        context={"llm": _fake()},
        config=cfg,
    )

    snapshot = await workflow.aget_state(cfg)
    assert snapshot.values["route_to"] == "billing"
    assert any(
        "router: classified as billing" in line
        for line in snapshot.values["internal_logs"]
    )
    # next node is empty -> the graph ran to completion on this thread
    assert snapshot.next == ()
