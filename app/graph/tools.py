"""Tools the graph's agent node can call.

Each is a plain function with the @tool decorator. The decorator reads the
signature and docstring into a JSON schema that gets sent to the model on
every call, so the docstring is a prompt - write it for the model to read.
"""

from typing import Literal

from langchain_core.tools import tool
from pydantic import BaseModel, Field


@tool
def get_deployment_status(service_name: str) -> str:
    """Return the current deployment status of one named service.

    Args:
        service_name: the service to look up, e.g. "agent-api" or "redis".
    """
    # A real implementation would query the orchestrator. Stubbed for now.
    known = {"agent-api": "healthy, v1.4.2", "redis": "healthy", "db": "healthy"}
    return known.get(service_name, f"unknown service: {service_name}")


tools = [get_deployment_status]


class RiskAssessmentInput(BaseModel):
    """Inputs for the corporate risk score tool."""

    company_name: str = Field(
        description="Full legal name of the entity being analyzed, as it "
        "appears on regulatory filings.",
    )
    industry: Literal["Finance", "Tech", "Healthcare", "Manufacturing"] = Field(
        description="Primary sector the company operates in. Pick the "
        "closest of the four; do not invent a value.",
    )
    exposure_value: float = Field(
        gt=0,
        le=1_000_000_000,
        description="Financial exposure in US dollars, greater than 0 and "
        "at most 1 billion.",
    )
    is_regulated: bool = Field(
        default=True,
        description="Whether the entity is under strict regulatory "
        "oversight (SEC, GDPR, HIPAA, and the like).",
    )


@tool(args_schema=RiskAssessmentInput)
def calculate_corporate_risk(
    company_name: str,
    industry: str,
    exposure_value: float,
    is_regulated: bool,
) -> dict:
    """Calculate a corporate risk score from industry base rate and exposure.

    Use this when asked to assess or quantify a company's risk. Returns the
    numeric score, a High/Standard band, and the ruleset version.
    """
    base_rates = {
        "Finance": 0.15,
        "Tech": 0.08,
        "Healthcare": 0.12,
        "Manufacturing": 0.05,
    }
    multiplier = 1.5 if is_regulated else 1.0
    score = round(exposure_value * base_rates[industry] * multiplier, 2)

    return {
        "entity": company_name,
        "risk_score": score,
        "band": "High" if score > 100_000 else "Standard",
        "logic_version": "2026.09.0",
    }


tools = [get_deployment_status, calculate_corporate_risk]
