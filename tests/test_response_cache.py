"""Response-cache behaviour: in-memory backend, stubbed graph, no Redis, no LLM."""

import pytest


@pytest.mark.asyncio
async def test_second_identical_query_hits_cache(
    client, auth_headers, fake_workflow, cache
):
    r1 = await client.get("/api/v1/ask", params={"q": "same"}, headers=auth_headers)
    r2 = await client.get("/api/v1/ask", params={"q": "same"}, headers=auth_headers)
    assert r1.status_code == r2.status_code == 200
    assert r1.json() == r2.json()
    assert fake_workflow.calls == 1  # LLM ran once; second call came from cache


@pytest.mark.asyncio
async def test_different_query_is_a_miss(client, auth_headers, fake_workflow, cache):
    await client.get("/api/v1/ask", params={"q": "one"}, headers=auth_headers)
    await client.get("/api/v1/ask", params={"q": "two"}, headers=auth_headers)
    assert fake_workflow.calls == 2


@pytest.mark.asyncio
async def test_ask_still_requires_auth(client):
    r = await client.get("/api/v1/ask", params={"q": "x"})
    assert r.status_code == 401
