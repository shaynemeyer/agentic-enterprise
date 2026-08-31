# tests/graph/test_reducers.py
from app.graph.engine import merge_logs


def test_merge_logs_appends_new_lines():
    assert merge_logs(["a"], ["b"]) == ["a", "b"]


def test_merge_logs_tolerates_none():
    assert merge_logs(None, ["a"]) == ["a"]


def test_merge_logs_skips_duplicates():
    assert merge_logs(["a", "b"], ["b", "c"]) == ["a", "b", "c"]
