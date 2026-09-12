# tests/graph/test_research_visualization.py
from pathlib import Path

import pytest

from app.graph.research import research_mermaid
from tests.graph.mcp_probe import requires_mcp

pytestmark = [requires_mcp, pytest.mark.asyncio]


async def test_mermaid_lists_every_node():
    m = await research_mermaid()
    for node in ("agent", "tools", "report"):
        assert node in m


async def test_mermaid_shows_the_research_loop():
    m = await research_mermaid()
    assert "tools --> agent" in m
    assert "agent -.-> tools" in m


async def test_committed_diagram_matches_the_compiled_graph():
    committed = Path("docs/graph-research.mmd").read_text().strip()
    assert committed == (await research_mermaid()).strip(), (
        "docs/graph-research.mmd is stale - regenerate: "
        "uv run python -c 'import asyncio; from app.graph.research import "
        "research_mermaid; print(asyncio.run(research_mermaid()))' > "
        "docs/graph-research.mmd"
    )
