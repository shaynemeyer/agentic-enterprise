# tests/graph/test_visualization.py
from app.graph.engine import graph_mermaid


def test_mermaid_lists_every_node():
    m = graph_mermaid()
    for node in ("router", "agent", "tools", "billing", "general", "critic"):
        assert node in m


def test_mermaid_shows_the_critic_cycle():
    m = graph_mermaid()
    assert "general --> critic" in m
    assert "critic -.-> general" in m
