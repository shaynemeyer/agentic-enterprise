import pytest
from pydantic import ValidationError

from app.graph.tools import calculate_corporate_risk


def test_rejects_industry_outside_the_enum():
    with pytest.raises(ValidationError):
        calculate_corporate_risk.invoke(
            {
                "company_name": "AI Corp",
                "industry": "Entertainment",
                "exposure_value": 50_000,
                "is_regulated": False,
            }
        )


def test_rejects_non_positive_exposure():
    with pytest.raises(ValidationError):
        calculate_corporate_risk.invoke(
            {
                "company_name": "AI Corp",
                "industry": "Tech",
                "exposure_value": 0,
                "is_regulated": False,
            }
        )


def test_valid_call_returns_deterministic_score():
    result = calculate_corporate_risk.invoke(
        {
            "company_name": "CyberBank",
            "industry": "Finance",
            "exposure_value": 200_000,
            "is_regulated": True,
        }
    )
    assert result == {
        "entity": "CyberBank",
        "risk_score": 45_000.0,
        "band": "Standard",
        "logic_version": "2026.09.0",
    }


def test_is_regulated_defaults_to_true():
    result = calculate_corporate_risk.invoke(
        {
            "company_name": "CyberBank",
            "industry": "Finance",
            "exposure_value": 200_000,
        }
    )
    assert result["risk_score"] == 45_000.0
