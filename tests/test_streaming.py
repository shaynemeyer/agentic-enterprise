import json

import pytest

from app.core.context import REQUEST_ID_HEADER


@pytest.mark.asyncio
@pytest.mark.integration  # hits the real LLM via the graph
async def test_stream_emits_events_and_request_id(client, auth_headers):
    payload = {"task_description": "Say hello in five words.", "agent_id": "smoke"}

    async with client.stream(
        "POST",
        "/api/v1/run/stream",
        json=payload,
        headers={**auth_headers, REQUEST_ID_HEADER: "stream-trace-1"},
    ) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        assert resp.headers[REQUEST_ID_HEADER] == "stream-trace-1"

        phases = []
        async for line in resp.aiter_lines():
            if line.startswith("data:"):
                phases.append(json.loads(line[5:].strip())["phase"])

    assert "status" in phases  # node's own writer(...) line
    assert phases[-1] == "done"
