"""Shared test fixtures: isolated app with no Postgres, no Redis, no LLM."""

import os

from langchain.messages import AIMessage, HumanMessage

from app.graph import engine

# Must run before app.core.config imports it.
os.environ.setdefault("LLM_BASE_URL", "http://localhost:11434/v1")

import pytest
import pytest_asyncio
from fastapi_cache import FastAPICache
from fastapi_cache.backends.inmemory import InMemoryBackend
from httpx import ASGITransport, AsyncClient

from app.api.v1 import endpoints
from app.core.config import settings
from app.database import get_db
from app.graph import tools as graph_tools_mod
from app.main import app


class _FakeState:
    def __init__(self, values):
        self.values = values
        self.next = ()


class _FakeSession:
    """Async session stub - the run routes only ever call .add()."""

    def add(self, _obj):
        pass


async def _fake_get_db():
    yield _FakeSession()


class _FakeWorkflow:
    """Stubbed LangGraph. `calls` counts how many times the graph really ran,
    which is how the cache tests prove a hit skipped the LLM."""

    def __init__(self):
        self.calls = 0
        self.checkpointer = None

    async def ainvoke(self, _state, *, context=None, config=None):
        self.calls += 1
        return {"messages": [type("M", (), {"content": "hi there friend"})()]}

    async def aget_state(self, _config):
        return _FakeState(
            {"messages": [HumanMessage("prior question"), AIMessage("prior answer")]}
        )


@pytest.fixture
def no_stream(monkeypatch):
    """Neutralise `get_stream_writer()` so a node can be called off-graph.

    The real writer reads a contextvar that only `workflow.astream(...)` sets;
    a direct node call has no runnable context. This swaps it for a sink.
    """
    monkeypatch.setattr(engine, "get_stream_writer", lambda: lambda _update: None)


@pytest.fixture(autouse=True)
def reset_shared_state():
    """Empty the rate-limit counters and dependency overrides around every test."""
    app.state.limiter.reset()
    yield
    app.state.limiter.reset()
    app.dependency_overrides.clear()


@pytest.fixture(autouse=True)
def fake_db():
    """No test touches a real database."""
    app.dependency_overrides[get_db] = _fake_get_db
    yield


@pytest.fixture
def fake_workflow(monkeypatch):
    """Swap the real graph for the counting stub; return it so a test can read
    `.calls`.

    The handlers call `build_workflow()` per request now (the tool list loads
    from the MCP server at build time), so patch that to hand back the stub -
    and the cached `_workflow` so nothing rebuilds the real graph.
    """
    fake = _FakeWorkflow()

    async def _fake_build():
        return fake

    async def _fake_get_tools():
        return []

    monkeypatch.setattr(engine, "_workflow", fake)
    monkeypatch.setattr(engine, "build_workflow", _fake_build)
    monkeypatch.setattr(endpoints, "build_workflow", _fake_build)
    # The handlers also call get_tools() directly to bind them to the LLM;
    # without this that reaches the real MCP server.
    monkeypatch.setattr(endpoints, "get_tools", _fake_get_tools)
    monkeypatch.setattr(graph_tools_mod, "get_tools", _fake_get_tools)
    monkeypatch.setattr(graph_tools_mod, "load_tools", _fake_get_tools)
    return fake


@pytest.fixture
def cache():
    """In-memory fastapi-cache2 backend - same semantics as Redis, no container."""
    FastAPICache.init(InMemoryBackend(), prefix="test-cache")
    yield
    FastAPICache._backend = None
    FastAPICache._init = False


@pytest_asyncio.fixture
async def client():
    """AsyncClient wired straight to the ASGI app - no network."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac


@pytest_asyncio.fixture
async def auth_headers(client):
    """A Bearer header for the demo user. Skips the test if no DEMO_PASSWORD."""
    if not settings.demo_password:
        pytest.skip("no DEMO_PASSWORD in .env")
    r = await client.post(
        "/api/v1/token",
        data={"username": settings.demo_username, "password": settings.demo_password},
    )
    return {"Authorization": f"Bearer {r.json()['access_token']}"}
