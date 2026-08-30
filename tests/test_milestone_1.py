import pytest


# httpx's ASGITransport calls the FastAPI app in-process, so no server
# process is needed here. This is still an end-to-end test, though: it makes
# a real call to the configured LLM backend, so it's slower and less
# isolated than a typical unit test.
@pytest.mark.integration
@pytest.mark.asyncio
async def test_end_to_end_ignition(client):
    response = await client.post(
        "/test/smoke",
        json={"test_id": "IGNITION-001", "payload": "Verify system integrity."},
    )
    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "PASS"
    assert "llm_response" in data

    print(f"\nSmoke Test Passed! Latency: {data['latency_ms']:.2f}ms")
