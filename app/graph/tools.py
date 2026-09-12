"""Tools the graph's agent node can call.

Each is a plain function with the @tool decorator. The decorator reads the
signature and docstring into a JSON schema that gets sent to the model on
every call, so the docstring is a prompt - write it for the model to read.
"""

import asyncio
import logging
import os

from langchain_core.tools import StructuredTool, tool
from langchain_mcp_adapters.client import MultiServerMCPClient

from app.core.config import settings
from app.tools.code_executor import (
    CodeExecutionInput,
    execute_sandboxed_code,
    execute_sandboxed_code_async,
)

logger = logging.getLogger("enterprise_agent.tools")


@tool
def get_deployment_status(service_name: str) -> str:
    """Return the current deployment status of one named service.

    Args:
        service_name: the service to look up, e.g. "agent-api" or "redis".
    """
    # A real implementation would query the orchestrator. Stubbed for now.
    known = {"agent-api": "healthy, v1.4.2", "redis": "healthy", "db": "healthy"}
    return known.get(service_name, f"unknown service: {service_name}")


sandboxed_code_tool = StructuredTool.from_function(
    func=execute_sandboxed_code,
    coroutine=execute_sandboxed_code_async,
    name="execute_sandboxed_code",
    description=(
        "Run a short Python script in an isolated, network-disabled "
        "sandbox for data analysis or calculations no other tool covers. "
        "The script must print() its result."
    ),
    args_schema=CodeExecutionInput,
)


async def get_weather(city: str) -> str:
    """Return the current weather for a city.

    Args:
        city: the city to fetch weather for, e.g. "London".
    """
    logger.info("weather lookup started city=%s", city)
    await asyncio.sleep(2)  # stands in for a real network call's latency
    logger.info("weather lookup finished city=%s", city)
    return f"The weather in {city} is 22C and sunny."


weather_tool = StructuredTool.from_function(
    coroutine=get_weather,
    name="get_weather",
    description="Return the current weather for a named city.",
)

MCP_RISK_URL = os.environ.get("MCP_RISK_URL", "http://127.0.0.1:8100/mcp")
MCP_FS_URL = os.environ.get("MCP_FS_URL", "http://127.0.0.1:8101/mcp")
MCP_FINANCE_URL = os.environ.get("MCP_FINANCE_URL", "http://127.0.0.1:8103/mcp")

_auth_headers = {"Authorization": f"Bearer {settings.mcp_auth_token}"}

_mcp_client = MultiServerMCPClient(
    {
        "risk": {
            "transport": "streamable_http",
            "url": MCP_RISK_URL,
            "headers": _auth_headers,
        },
        "filesystem": {
            "transport": "streamable_http",
            "url": MCP_FS_URL,
            "headers": _auth_headers,
        },
        "finance": {
            "transport": "streamable_http",
            "url": MCP_FINANCE_URL,
            "headers": _auth_headers,
        },
    }
)


async def load_tools() -> list:
    """The graph's tool list: the local tool plus every tool the risk
    MCP server advertises.

    Async because get_tools() opens an MCP session to call list_tools.
    Call this once at startup, not per request.
    """
    remote = await _mcp_client.get_tools()
    return [get_deployment_status, sandboxed_code_tool, weather_tool, *remote]


_tools_cache: list | None = None
_tools_lock = asyncio.Lock()


async def get_tools() -> list:
    """load_tools() result, fetched once and reused.

    Request handlers bind this to the LLM per call; without the cache each
    request would re-run the MCP list_tools round-trip.
    """
    global _tools_cache
    async with _tools_lock:
        if _tools_cache is None:
            _tools_cache = await load_tools()
    return _tools_cache
