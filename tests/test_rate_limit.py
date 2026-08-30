"""Rate-limit tests, isolated from Postgres and the LLM."""

import pytest
from httpx import ASGITransport, AsyncClient

from app.api.v1 import endpoints
from app.core.config import settings
from app.database import get_db
from app.main import app

VALID_BODY = {"task_description": "Say hello in five words.", "agent_id": "probe"}


class _FakeSession:
    def add(self, _obj):
        pass


async def _fake_get_db():
    yield _FakeSession()


class _FakeWorkflow:
    async def ainvoke(self, _state):
        return {"messages": [type("M", (), {"content": "hi there friend"})()]}


@pytest.fixture(autouse=True)
def isolate(monkeypatch):
    """Empty limiter counters, stubbed db and graph for every test."""
    app.state.limiter.reset()
    app.dependency_overrides[get_db] = _fake_get_db
    monkeypatch.setattr(endpoints, "workflow", _FakeWorkflow())
    yield
    app.state.limiter.reset()
    app.dependency_overrides.clear()


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
async def test_sixth_call_in_a_minute_is_429(client):
    async with client as c:
        headers = {"Authorization": f"Bearer {await _token(c)}"}
        codes = [
            (await c.post("/api/v1/run", json=VALID_BODY, headers=headers)).status_code
            for _ in range(6)
        ]
    assert codes[:5] == [200, 200, 200, 200, 200]
    assert codes[5] == 429


@pytest.mark.asyncio
@pytest.mark.skipif(not settings.demo_password, reason="no DEMO_PASSWORD in .env")
async def test_429_body_and_retry_after(client):
    async with client as c:
        headers = {"Authorization": f"Bearer {await _token(c)}"}
        for _ in range(5):
            await c.post("/api/v1/run", json=VALID_BODY, headers=headers)
        r = await c.post("/api/v1/run", json=VALID_BODY, headers=headers)
    assert r.status_code == 429
    assert r.json()["error_code"] == "RATE_LIMIT_EXCEEDED"
    assert "Retry-After" in r.headers


@pytest.mark.asyncio
async def test_rate_limit_does_not_preempt_auth(client):
    """No token -> 401 from get_current_user, not 429."""
    async with client as c:
        r = await c.post("/api/v1/run", json=VALID_BODY)
    assert r.status_code == 401
