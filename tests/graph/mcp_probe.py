"""Skip marker for tests that need the risk MCP server running.

The graph's "tools" node is built from an MCP list_tools call now (Lab 44),
so build_workflow() and graph_mermaid() reach the server at
http://127.0.0.1:8100/mcp. When it is not up, skip - the same way the
LLM-backed tests skip on an unset LLM_BASE_URL.
"""

import httpx
import pytest

MCP_URL = "http://127.0.0.1:8100/mcp"


def _mcp_up() -> bool:
    try:
        httpx.get(MCP_URL.rsplit("/mcp", 1)[0], timeout=0.5)
        return True
    except httpx.HTTPError:
        return False


requires_mcp = pytest.mark.skipif(not _mcp_up(), reason="risk MCP server not running")
