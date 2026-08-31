# tests/graph/test_visualization.py
from pathlib import Path

from app.graph.engine import graph_mermaid


def test_mermaid_lists_every_node():
    m = graph_mermaid()
    for node in ("router", "agent", "tools", "billing", "general", "critic"):
        assert node in m


def test_mermaid_shows_the_critic_cycle():
    m = graph_mermaid()
    assert "general --> critic" in m
    assert "critic -.-> general" in m


def test_committed_diagram_matches_the_compiled_graph():
    committed = Path("docs/graph.mmd").read_text().strip()
    assert committed == graph_mermaid().strip(), (
        "docs/graph.mmd is stale - regenerate: "
        "uv run python -c 'from app.graph.engine import graph_mermaid; "
        "print(graph_mermaid())' > docs/graph.mmd"
    )
