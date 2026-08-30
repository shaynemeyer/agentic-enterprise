"""Response-cache tests: in-memory backend, stubbed graph, no Redis, no LLM."""

import pytest
from fastapi_cache import FastAPICache
from fastapi_cache.backends.inmemory import InMemoryBackend
from httpx import ASGITransport, AsyncClient

from app.api.v1 import endpoints
from app.core.config import settings
from app.main import app


class _CountingWorkflow:
    """Records how many times the 'LLM' actually ran."""

    def __init__(self):
        self.calls = 0

    async def ainvoke(self, _state):
        self.calls += 1
        return {"messages": [type("M", (), {"content": "cached answer"})()]}


@pytest.fixture(autouse=True)
def isolate(monkeypatch):
    FastAPICache.init(InMemoryBackend(), prefix="test-cache")
    app.state.limiter.reset()
    fake = _CountingWorkflow()
    monkeypatch.setattr(endpoints, "workflow", fake)
    yield fake
    app.state.limiter.reset()


@pytest.fixture
def client():
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


async def _token(c):
    r = await c.post(
        "/api/v1/token",
        data={"username": settings.demo_username, "password": settings.demo_password},
    )
    return r.json()["access_token"]


@pytest.mark.asyncio
@pytest.mark.skipif(not settings.demo_password, reason="no DEMO_PASSWORD in .env")
async def test_second_identical_query_hits_cache(client, isolate):
    async with client as c:
        headers = {"Authorization": f"Bearer {await _token(c)}"}
        r1 = await c.get("/api/v1/ask", params={"q": "same"}, headers=headers)
        r2 = await c.get("/api/v1/ask", params={"q": "same"}, headers=headers)
    assert r1.status_code == r2.status_code == 200
    assert r1.json() == r2.json()
    assert isolate.calls == 1  # LLM ran once; second call came from cache


@pytest.mark.asyncio
@pytest.mark.skipif(not settings.demo_password, reason="no DEMO_PASSWORD in .env")
async def test_different_query_is_a_miss(client, isolate):
    async with client as c:
        headers = {"Authorization": f"Bearer {await _token(c)}"}
        await c.get("/api/v1/ask", params={"q": "one"}, headers=headers)
        await c.get("/api/v1/ask", params={"q": "two"}, headers=headers)
    assert isolate.calls == 2


@pytest.mark.asyncio
async def test_ask_still_requires_auth(client):
    async with client as c:
        r = await c.get("/api/v1/ask", params={"q": "x"})
    assert r.status_code == 401
