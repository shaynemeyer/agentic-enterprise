"""remember()/search() against a real Qdrant instance and a fake embedder.

  export QDRANT_TEST_URL=http://localhost:6333
Skipped when unset.
"""

import os
import uuid

import pytest

from app.memory import vector_store

QDRANT_URL = os.getenv("QDRANT_TEST_URL")
pytestmark = pytest.mark.skipif(not QDRANT_URL, reason="QDRANT_TEST_URL not set")


class _FakeEmbeddings:
    """Deterministic 4-dim vectors keyed by text - no network, no model pull
    needed to exercise remember()/search()'s Qdrant plumbing."""

    async def aembed_query(self, text: str) -> list[float]:
        return [float(len(text) % 7), 1.0, 0.0, 0.0]


@pytest.fixture(autouse=True)
def _isolated_collection(monkeypatch):
    """A fresh collection name per test run - tests never share state, and
    never touch whatever collection a real dev session has been using."""
    from app.core.config import settings

    monkeypatch.setattr(settings, "qdrant_url", QDRANT_URL)
    monkeypatch.setattr(settings, "qdrant_collection", f"test-{uuid.uuid4().hex[:8]}")
    monkeypatch.setattr(vector_store, "get_embeddings", lambda: _FakeEmbeddings())
    monkeypatch.setattr(vector_store, "_client", None)


@pytest.mark.asyncio
async def test_remember_then_search_finds_it():
    thread_id = f"t-{uuid.uuid4()}"
    await vector_store.remember(thread_id, "the deploy key rotates every 90 days")

    hits = await vector_store.search("deploy key rotation", limit=3)
    assert any(h["thread_id"] == thread_id for h in hits)


@pytest.mark.asyncio
async def test_search_on_empty_collection_returns_no_hits():
    hits = await vector_store.search("anything", limit=3)
    assert hits == []


@pytest.mark.asyncio
async def test_remember_mints_a_distinct_point_id_per_call():
    thread_id = f"t-{uuid.uuid4()}"
    first = await vector_store.remember(thread_id, "fact one")
    second = await vector_store.remember(thread_id, "fact two")
    assert first != second
