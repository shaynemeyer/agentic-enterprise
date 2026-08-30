import pytest

from app.core.context import REQUEST_ID_HEADER
from app.core.exceptions import MaxRecursionError
from app.main import app


@pytest.mark.asyncio
async def test_generates_request_id_when_absent(client):
    resp = await client.get("/health")

    assert resp.status_code == 200
    assert REQUEST_ID_HEADER in resp.headers
    assert len(resp.headers[REQUEST_ID_HEADER]) == 36  # UUID4 string


@pytest.mark.asyncio
async def test_echoes_incoming_request_id(client):
    resp = await client.get(
        "/health", headers={REQUEST_ID_HEADER: "upstream-trace-123"}
    )

    assert resp.headers[REQUEST_ID_HEADER] == "upstream-trace-123"


@pytest.mark.asyncio
async def test_trace_id_in_error_body_matches_header(client):
    @app.get("/_test_boom_trace")
    async def _boom():
        raise MaxRecursionError()

    resp = await client.get(
        "/_test_boom_trace", headers={REQUEST_ID_HEADER: "trace-abc"}
    )

    assert resp.status_code == 422
    assert resp.json()["trace_id"] == "trace-abc"
    assert resp.headers[REQUEST_ID_HEADER] == "trace-abc"
