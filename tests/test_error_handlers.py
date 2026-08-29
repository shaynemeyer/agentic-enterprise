import pytest
from httpx import ASGITransport, AsyncClient

from app.core.config import settings
from app.core.exceptions import MaxRecursionError
from app.main import app


@pytest.mark.asyncio
async def test_agentic_exception_returns_error_schema():
    @app.get("/_test_boom")
    async def _boom():
        raise MaxRecursionError(details={"max_steps": 25, "current_step": 26})

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/_test_boom", headers={"X-Request-ID": "trace-13"})

    assert resp.status_code == 422
    body = resp.json()
    assert body["success"] is False
    assert body["error_code"] == "MAX_RECURSION_REACHED"
    assert body["details"] == {"max_steps": 25, "current_step": 26}
    assert body["trace_id"] == "trace-13"
    assert resp.headers["X-Request-ID"] == "trace-13"
    assert "timestamp" in body


@pytest.mark.asyncio
@pytest.mark.skipif(not settings.demo_password, reason="no DEMO_PASSWORD in .env")
async def test_validation_error_uses_error_schema():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        tok = await client.post(
            "/api/v1/token",
            data={
                "username": settings.demo_username,
                "password": settings.demo_password,
            },
        )
        resp = await client.post(
            "/api/v1/run",
            json={"agent_id": "x"},  # missing task_description
            headers={"Authorization": f"Bearer {tok.json()['access_token']}"},
        )

    assert resp.status_code == 422
    assert resp.json()["error_code"] == "VALIDATION_ERROR"
