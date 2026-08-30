import pytest

from app.core.exceptions import MaxRecursionError
from app.main import app


@pytest.mark.asyncio
async def test_agentic_exception_returns_error_schema(client):
    @app.get("/_test_boom")
    async def _boom():
        raise MaxRecursionError(details={"max_steps": 25, "current_step": 26})

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
async def test_validation_error_uses_error_schema(client, auth_headers):
    resp = await client.post(
        "/api/v1/run",
        json={"agent_id": "x"},  # missing task_description
        headers=auth_headers,
    )

    assert resp.status_code == 422
    assert resp.json()["error_code"] == "VALIDATION_ERROR"
