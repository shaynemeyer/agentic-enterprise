import os
from typing import Annotated, TypedDict

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph, add_messages
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt.tool_node import ToolInvocationError
from pydantic import ValidationError

from app.core.llm import get_sovereign_llm
from app.graph.tools import calculate_corporate_risk, tools
from app.graph.tools import tools as risk_tools


def test_rejects_industry_outside_the_enum():
    with pytest.raises(ValidationError):
        calculate_corporate_risk.invoke(
            {
                "company_name": "AI Corp",
                "industry": "Entertainment",
                "exposure_value": 50_000,
                "is_regulated": False,
            }
        )


def test_rejects_non_positive_exposure():
    with pytest.raises(ValidationError):
        calculate_corporate_risk.invoke(
            {
                "company_name": "AI Corp",
                "industry": "Tech",
                "exposure_value": 0,
                "is_regulated": False,
            }
        )


def test_valid_call_returns_deterministic_score():
    result = calculate_corporate_risk.invoke(
        {
            "company_name": "CyberBank",
            "industry": "Finance",
            "exposure_value": 200_000,
            "is_regulated": True,
        }
    )
    assert result == {
        "entity": "CyberBank",
        "risk_score": 45_000.0,
        "band": "Standard",
        "logic_version": "2026.09.0",
    }


def test_is_regulated_defaults_to_true():
    result = calculate_corporate_risk.invoke(
        {
            "company_name": "CyberBank",
            "industry": "Finance",
            "exposure_value": 200_000,
        }
    )
    assert result["risk_score"] == 45_000.0


needs_llm = pytest.mark.skipif(
    not os.getenv("LLM_BASE_URL"),
    reason="no LLM backend configured",
)


@needs_llm
def test_model_requests_the_risk_tool():
    bound = get_sovereign_llm().bind_tools(risk_tools)
    prompt = (
        "Assess the corporate risk for Initech, a Manufacturing company "
        "with 4,000,000 dollars of exposure. It is not regulated."
    )

    response = bound.invoke([HumanMessage(prompt)])

    assert response.tool_calls, "model answered directly instead of calling a tool"
    call = response.tool_calls[0]
    assert call["name"] == "calculate_corporate_risk"
    assert call["args"]["industry"] == "Manufacturing"
    assert call["args"]["exposure_value"] == 4_000_000
    assert call["args"]["is_regulated"] is False


class _MsgState(TypedDict):
    messages: Annotated[list, add_messages]


def _one_node_graph(node: ToolNode):
    g = StateGraph(_MsgState)
    g.add_node("tools", node)
    g.add_edge(START, "tools")
    g.add_edge("tools", END)
    return g.compile()


def _bad_risk_call() -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": "calculate_corporate_risk",
                "args": {
                    "company_name": "Initech",
                    "industry": "Entertainment",
                    "exposure_value": 50_000,
                },
                "id": "call_1",
            }
        ],
    )


def test_toolnode_returns_error_message_by_default():
    out = _one_node_graph(ToolNode(tools)).invoke({"messages": [_bad_risk_call()]})
    msg = out["messages"][-1]
    assert msg.status == "error"
    assert "calculate_corporate_risk" in msg.content


def test_toolnode_can_be_made_strict():
    with pytest.raises(ToolInvocationError):
        _one_node_graph(ToolNode(tools, handle_tool_errors=False)).invoke(
            {"messages": [_bad_risk_call()]}
        )
