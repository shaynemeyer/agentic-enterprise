from unittest.mock import AsyncMock

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.runtime import Runtime

from app.graph.engine import RuntimeContext, call_model, critic, route_request


@pytest.mark.asyncio
async def test_call_model_returns_llm_reply_as_delta(no_stream):
    fake = FakeMessagesListChatModel(responses=[AIMessage("stubbed reply")])
    runtime = Runtime(context=RuntimeContext(llm=fake))

    result = await call_model({"messages": [HumanMessage("hi")]}, runtime)

    # The node returns only the keys it updates - assert on those, not on a
    # merged global state.
    assert result["messages"][0].content == "stubbed reply"
    assert result["status"] == "completed"
    assert result["internal_logs"] == ["agent: model call complete"]


@pytest.mark.asyncio
async def test_route_request_classifies_technical_keywords(no_stream):
    result = await route_request(
        {"messages": [HumanMessage("check the deploy logs for agent-api")]}
    )
    assert result["route_to"] == "technical"


@pytest.mark.asyncio
async def test_route_request_defaults_to_general(no_stream):
    result = await route_request({"messages": [HumanMessage("hello there")]})
    assert result["route_to"] == "general"


@pytest.mark.asyncio
async def test_call_model_invokes_injected_llm_with_message_history(no_stream):
    spy = AsyncMock()
    spy.ainvoke.return_value = AIMessage("spied reply")
    runtime = Runtime(context=RuntimeContext(llm=spy))

    history = [HumanMessage("first"), AIMessage("second"), HumanMessage("third")]
    result = await call_model({"messages": history}, runtime)

    spy.ainvoke.assert_awaited_once_with(history)
    assert result["messages"][0].content == "spied reply"


@pytest.mark.asyncio
async def test_critic_passes_a_long_draft(no_stream):
    long_draft = AIMessage("x" * 100)  # > 90 chars clears the bar
    result = await critic({"messages": [long_draft], "revision_count": 1})
    assert result["critique"] == "PASS"


@pytest.mark.asyncio
async def test_critic_rejects_a_bare_stub_under_the_limit(no_stream):
    result = await critic({"messages": [AIMessage("too short")], "revision_count": 1})
    assert result["critique"] != "PASS"


@pytest.mark.asyncio
async def test_critic_force_passes_at_the_revision_limit(no_stream):
    result = await critic({"messages": [AIMessage("still short")], "revision_count": 3})
    assert result["critique"] == "PASS"  # out of attempts - ship it
