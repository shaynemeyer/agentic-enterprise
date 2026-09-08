# tests/graph/test_visualization.py
from pathlib import Path

import pytest

from app.graph.engine import graph_mermaid
from tests.graph.mcp_probe import requires_mcp

pytestmark = [requires_mcp, pytest.mark.asyncio]


async def test_mermaid_lists_every_node():
    m = await graph_mermaid()
    for node in ("router", "agent", "tools", "billing", "general", "critic"):
        assert node in m


async def test_mermaid_shows_the_critic_cycle():
    m = await graph_mermaid()
    assert "general --> critic" in m
    assert "critic -.-> general" in m


async def test_committed_diagram_matches_the_compiled_graph():
    committed = Path("docs/graph.mmd").read_text().strip()
    assert committed == (await graph_mermaid()).strip(), (
        "docs/graph.mmd is stale - regenerate: "
        "uv run python -c 'import asyncio; from app.graph.engine import graph_mermaid; "
        "print(asyncio.run(graph_mermaid()))' > docs/graph.mmd"
    )
