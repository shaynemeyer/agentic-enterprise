# tests/graph/test_dependency_injection.py
import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage

from app.graph.engine import workflow


@pytest.mark.asyncio
async def test_call_model_uses_injected_llm():
    fake = FakeMessagesListChatModel(responses=[AIMessage("stubbed reply")])

    # "status" is a router keyword -> routes to `agent` -> call_model runs.
    result = await workflow.ainvoke(
        {"messages": [HumanMessage("check the deploy status of agent-api")]},
        context={"llm": fake},
        config={"configurable": {"thread_id": "di-test"}},
    )

    assert any(
        isinstance(m, AIMessage) and m.content == "stubbed reply"
        for m in result["messages"]
    )
