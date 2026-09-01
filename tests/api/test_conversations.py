import pytest


@pytest.mark.asyncio
async def test_get_conversation_returns_prior_turns(client, auth_headers, fake_workflow):
    r = await client.get("/api/v1/conversations/conv-1", headers=auth_headers)
    assert r.status_code == 200
    body = r.json()
    assert body["conversation_id"] == "conv-1"
    assert [t["content"] for t in body["turns"]] == ["prior question", "prior answer"]
