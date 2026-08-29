import pytest
from httpx import ASGITransport, AsyncClient

from app.core.config import settings
from app.main import app


@pytest.fixture
def client():
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


@pytest.mark.asyncio
async def test_run_requires_token(client):
    async with client as c:
        r = await c.post(
            "/api/v1/run",
            json={"task_description": "warm-up ping for auth", "agent_id": "probe"},
        )
    assert r.status_code == 401


@pytest.mark.asyncio
@pytest.mark.skipif(not settings.demo_password, reason="no DEMO_PASSWORD in .env")
async def test_token_then_protected_route(client):
    async with client as c:
        tok = await c.post(
            "/api/v1/token",
            data={
                "username": settings.demo_username,
                "password": settings.demo_password,
            },
        )
        assert tok.status_code == 200
        access = tok.json()["access_token"]

        r = await c.post(
            "/api/v1/run/stream",
            json={
                "task_description": "Say hello in five words.",
                "agent_id": "probe",
            },
            headers={"Authorization": f"Bearer {access}"},
        )
    # 200 stream (or 500 if the LLM backend is down) — the point is it is not 401.
    assert r.status_code != 401


@pytest.mark.asyncio
async def test_bad_token_rejected(client):
    async with client as c:
        r = await c.post(
            "/api/v1/run",
            json={"task_description": "warm-up ping for auth", "agent_id": "probe"},
            headers={"Authorization": "Bearer not.a.jwt"},
        )
    assert r.status_code == 401
