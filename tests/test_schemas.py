import pytest
from pydantic import ValidationError

from app.schemas.schemas import SmokeTestRequest, SmokeTestResponse


def test_smoke_test_request_requires_test_id():
    with pytest.raises(ValidationError):
        SmokeTestRequest()


def test_smoke_test_request_uses_default_payload():
    request = SmokeTestRequest(test_id="ST-2026-001")
    assert request.payload == "System Check: Respond with 'READY'"


def test_smoke_test_response_defaults_status_to_pass():
    response = SmokeTestResponse(
        test_id="ST-2026-001",
        graph_state="COMPLETED",
        llm_response="READY",
        latency_ms=42.0,
    )
    assert response.status == "PASS"


def test_smoke_test_response_rejects_non_numeric_latency():
    with pytest.raises(ValidationError):
        SmokeTestResponse(
            test_id="ST-2026-001",
            graph_state="COMPLETED",
            llm_response="READY",
            latency_ms="not-a-number",
        )
