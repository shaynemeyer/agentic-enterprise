import pytest

RUN_BODY = {"task_description": "warm-up ping for auth", "agent_id": "probe"}


@pytest.mark.asyncio
async def test_run_requires_token(client):
    r = await client.post("/api/v1/run", json=RUN_BODY)
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_token_then_protected_route(client, auth_headers):
    r = await client.post(
        "/api/v1/run/stream",
        json={"task_description": "Say hello in five words.", "agent_id": "probe"},
        headers=auth_headers,
    )
    # 200 stream (or 500 if the LLM backend is down) - the point is it is not 401.
    assert r.status_code != 401


@pytest.mark.asyncio
async def test_bad_token_rejected(client):
    r = await client.post(
        "/api/v1/run",
        json=RUN_BODY,
        headers={"Authorization": "Bearer not.a.jwt"},
    )
    assert r.status_code == 401
