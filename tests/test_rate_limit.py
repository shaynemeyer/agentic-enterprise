"""Rate-limit tests, isolated from Postgres and the LLM."""

import pytest

VALID_BODY = {"task_description": "Say hello in five words.", "agent_id": "probe"}


@pytest.mark.asyncio
async def test_sixth_call_in_a_minute_is_429(client, auth_headers, fake_workflow):
    codes = [
        (
            await client.post("/api/v1/run", json=VALID_BODY, headers=auth_headers)
        ).status_code
        for _ in range(6)
    ]
    assert codes[:5] == [200] * 5
    assert codes[5] == 429


@pytest.mark.asyncio
async def test_429_body_and_retry_after(client, auth_headers, fake_workflow):
    for _ in range(5):
        await client.post("/api/v1/run", json=VALID_BODY, headers=auth_headers)
    r = await client.post("/api/v1/run", json=VALID_BODY, headers=auth_headers)
    assert r.status_code == 429
    assert r.json()["error_code"] == "RATE_LIMIT_EXCEEDED"
    assert "Retry-After" in r.headers


@pytest.mark.asyncio
async def test_rate_limit_does_not_preempt_auth(client):
    """No token -> 401 from get_current_user, not 429."""
    r = await client.post("/api/v1/run", json=VALID_BODY)
    assert r.status_code == 401
