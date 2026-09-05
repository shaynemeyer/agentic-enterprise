"""Module 3 milestone: every path through the assembled graph, end to end.

Node-level behaviour is covered in test_nodes.py. This drives the compiled
`workflow` once per route with a fake LLM and asserts on GraphOutput.
"""

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage

from app.graph.engine import workflow


def _fake(*replies: str) -> FakeMessagesListChatModel:
    return FakeMessagesListChatModel(responses=[AIMessage(r) for r in replies])


@pytest.mark.asyncio
async def test_technical_message_routes_through_the_agent():
    result = await workflow.ainvoke(
        {"messages": [HumanMessage("the deploy logs show an error in agent-api")]},
        context={"llm": _fake("restarted the service"), "username": "admin"},
        config={"configurable": {"thread_id": "milestone-technical"}},
    )
    assert result["messages"][-1].content == "restarted the service"
    assert result["status"] == "completed"


@pytest.mark.asyncio
async def test_billing_message_routes_to_the_billing_worker():
    # billing_worker (Lab 40) calls the LLM for real now, so the fake supplies
    # the reply - one call, one response.
    result = await workflow.ainvoke(
        {"messages": [HumanMessage("my invoice has the wrong charge on it")]},
        context={"llm": _fake("billing department: refund issued"), "username": "admin"},
        config={"configurable": {"thread_id": "milestone-billing"}},
    )
    assert "billing department" in result["messages"][-1].content
    assert result["status"] == "completed"


@pytest.mark.asyncio
async def test_general_message_runs_the_critic_loop_to_a_pass():
    # First draft is a bare stub -> critic rejects -> general re-runs and
    # emits a "revised" draft -> critic PASSes on attempt 2. The `general` path
    # has no node that sets status="completed" after a PASS, so the terminal
    # status is the critic's own "critiqued".
    result = await workflow.ainvoke(
        {"messages": [HumanMessage("hello, I have a general question")]},
        context={
            "llm": _fake(
                "general enquiries desk: draft",
                "general enquiries desk: revised draft with more detail",
            ),
            "username": "admin",
        },
        config={"configurable": {"thread_id": "milestone-general"}},
    )
    assert "general enquiries desk" in result["messages"][-1].content.lower()
    assert result["status"] == "critiqued"


@pytest.mark.asyncio
async def test_general_path_stays_within_the_revision_limit():
    """The critic loop must not run away. GENERAL_REVISION_LIMIT = 3 forces a
    PASS, so `general` runs at most 3 times - well inside the default
    recursion_limit of 25. If this hangs or raises GraphRecursionError, the
    loop-breaker from Lab 24 regressed."""
    # Every reply is a bare stub that never passes the critic's bar on its own
    # merit, so the loop should run out attempts and force a PASS.
    result = await workflow.ainvoke(
        {"messages": [HumanMessage("just a general enquiry please")]},
        context={
            "llm": _fake(*["general enquiries desk: stub"] * 4),
            "username": "admin",
        },
        config={"configurable": {"thread_id": "milestone-revision-limit"}},
    )
    # 4 general-desk turns max (initial + 3 revisions); the real cap is the
    # critic's force-pass, not recursion_limit.
    general_turns = [
        m
        for m in result["messages"]
        if isinstance(m, AIMessage) and "general enquiries desk" in m.content.lower()
    ]
    assert 1 <= len(general_turns) <= 4
