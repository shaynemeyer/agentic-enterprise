import json
import os
from typing import Annotated, TypedDict

import httpx
import pytest
import pytest_asyncio
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph, add_messages
from langgraph.prebuilt import ToolNode

from app.core.llm import get_sovereign_llm
from app.graph.engine import format_tool_error
from app.graph.tools import load_tools
from app.mcp.risk_server import calculate_corporate_risk
from tests.graph.mcp_probe import requires_mcp

# calculate_corporate_risk moved to the MCP server (Lab 44). @mcp.tool()
# returns the function unchanged, so the pure-logic tests call it directly;
# the ToolNode tests go through load_tools() and need the server running.
_risk = calculate_corporate_risk


def test_valid_call_returns_deterministic_score():
    result = _risk(
        company_name="CyberBank",
        industry="Finance",
        exposure_value=200_000,
        is_regulated=True,
    )
    assert result == {
        "entity": "CyberBank",
        "risk_score": 45_000.0,
        "band": "Standard",
        "logic_version": "2026.09.0",
    }


def test_is_regulated_defaults_to_true():
    result = _risk(
        company_name="CyberBank",
        industry="Finance",
        exposure_value=200_000,
    )
    assert result["risk_score"] == 45_000.0


needs_llm = pytest.mark.skipif(
    not os.getenv("LLM_BASE_URL"),
    reason="no LLM backend configured",
)


@requires_mcp
@needs_llm
@pytest.mark.asyncio
async def test_model_requests_the_risk_tool():
    bound = get_sovereign_llm().bind_tools(await load_tools())
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


def _text(msg) -> str:
    """A ToolMessage from an MCP tool carries a list of content blocks, not a
    bare string. Join the text ones."""
    if isinstance(msg.content, str):
        return msg.content
    return "".join(b.get("text", "") for b in msg.content if isinstance(b, dict))


def _risk_call(**args) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": "calculate_corporate_risk", "args": args, "id": "call_x"}],
    )


@pytest_asyncio.fixture
async def graph_tools():
    return await load_tools()


@requires_mcp
@pytest.mark.asyncio
async def test_toolnode_returns_a_result_for_a_good_call(graph_tools):
    good = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "calculate_corporate_risk",
                "args": {
                    "company_name": "CyberBank",
                    "industry": "Finance",
                    "exposure_value": 200_000,
                    "is_regulated": True,
                },
                "id": "call_ok",
            }
        ],
    )
    out = await _one_node_graph(
        ToolNode(graph_tools, handle_tool_errors=format_tool_error)
    ).ainvoke({"messages": [good]})
    msg = out["messages"][-1]
    assert msg.status != "error"
    assert '"risk_score": 45000.0' in _text(msg)


@requires_mcp
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bad_args",
    [
        {
            "company_name": "AI Corp",
            "industry": "Entertainment",
            "exposure_value": 50_000,
        },
        {"company_name": "AI Corp", "industry": "Tech", "exposure_value": 0},
    ],
    ids=["industry-outside-enum", "non-positive-exposure"],
)
async def test_toolnode_reports_a_bad_argument(graph_tools, bad_args):
    # The argument constraints (Literal[...], gt=0) still run - now in
    # FastMCP's generated wrapper on the server. It catches the pydantic
    # ValidationError and returns it as an error result, so the client does
    # not raise and format_tool_error never sees a ValidationError. What
    # reaches the model is the server's "Error executing tool ..." text as an
    # error ToolMessage - enough to retry or give up on, and it does not
    # crash the run.
    out = await _one_node_graph(
        ToolNode(graph_tools, handle_tool_errors=format_tool_error)
    ).ainvoke({"messages": [_risk_call(**bad_args)]})
    msg = out["messages"][-1]
    assert msg.status == "error"
    assert "calculate_corporate_risk" in _text(msg)
    assert "validation error" in _text(msg).lower()


FS_URL = "http://127.0.0.1:8101/mcp"


def _fs_up() -> bool:
    try:
        httpx.get(FS_URL.rsplit("/mcp", 1)[0], timeout=0.5)
        return True
    except httpx.HTTPError:
        return False


requires_fs = pytest.mark.skipif(
    not _fs_up(), reason="filesystem MCP server not running"
)


def _read_call(path: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": "read_text_file", "args": {"path": path}, "id": "call_r"}],
    )


@pytest.mark.asyncio
@requires_mcp
@requires_fs
async def test_filesystem_jail_rejects_traversal():
    tools = await load_tools()
    out = await _one_node_graph(
        ToolNode(tools, handle_tool_errors=format_tool_error)
    ).ainvoke({"messages": [_read_call("../../etc/passwd")]})
    msg = out["messages"][-1]
    assert msg.status == "error"
    assert "escapes" in _text(msg)


@pytest.mark.asyncio
async def test_write_rejects_oversize_payload(monkeypatch):
    from app.mcp import fs_server

    monkeypatch.setattr(fs_server.settings, "agent_files_max_write_bytes", 10)
    with pytest.raises(ValueError, match="exceeds"):
        await fs_server.write_text_file(path="big.txt", content="x" * 50)


FINANCE_MCP_URL = "http://127.0.0.1:8103/mcp"
FINANCE_API_URL = "http://127.0.0.1:8102"


def _finance_up() -> bool:
    try:
        httpx.get(FINANCE_API_URL, timeout=0.5)
        httpx.get(FINANCE_MCP_URL.rsplit("/mcp", 1)[0], timeout=0.5)
        return True
    except httpx.HTTPError:
        return False


requires_finance = pytest.mark.skipif(
    not _finance_up(), reason="finance MCP server or mock API not running"
)


def _balance_call(account_id: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {"name": "get_account_balance", "args": {"account_id": account_id}, "id": "call_b"}
        ],
    )


@pytest.mark.asyncio
@requires_mcp
@requires_finance
async def test_get_account_balance_crosses_both_boundaries():
    tools = await load_tools()
    bal = next(t for t in tools if t.name == "get_account_balance")
    blocks = await bal.ainvoke({"account_id": "ACC-1001"})
    out = json.loads(blocks[0]["text"])
    assert out["currency"] == "USD"
    assert out["balance"] == 1_250_500.75


@pytest.mark.asyncio
@requires_mcp
@requires_finance
async def test_unknown_account_is_an_error_message():
    tools = await load_tools()
    out = await _one_node_graph(
        ToolNode(tools, handle_tool_errors=format_tool_error)
    ).ainvoke({"messages": [_balance_call("ACC-9999")]})
    msg = out["messages"][-1]
    assert msg.status == "error"


@pytest.mark.asyncio
async def test_token_cache_refreshes_near_expiry(monkeypatch):
    from app.mcp.oauth import ClientCredentialsToken

    calls = []

    async def fake_fetch(self):
        calls.append(1)
        self._access_token = f"tok-{len(calls)}"
        self._expires_at = 0.0 if len(calls) == 1 else 1e12

    monkeypatch.setattr(ClientCredentialsToken, "_fetch", fake_fetch)
    t = ClientCredentialsToken("u", "id", "sec", "scope")

    assert await t.value() == "tok-1"  # first fetch
    assert await t.value() == "tok-2"  # tok-1 already past expiry -> refetch
    assert await t.value() == "tok-2"  # tok-2 valid -> cached
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_token_cache_single_fetch_under_concurrency(monkeypatch):
    """Ten coroutines hitting a cold cache trigger exactly one POST."""
    import asyncio

    from app.mcp.oauth import ClientCredentialsToken

    calls = []

    async def fake_fetch(self):
        await asyncio.sleep(0.01)  # make the race real
        calls.append(1)
        self._access_token = "tok"
        self._expires_at = 1e12

    monkeypatch.setattr(ClientCredentialsToken, "_fetch", fake_fetch)
    t = ClientCredentialsToken("u", "id", "sec", "scope")

    await asyncio.gather(*(t.value() for _ in range(10)))
    assert len(calls) == 1
