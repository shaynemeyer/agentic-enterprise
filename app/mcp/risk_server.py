"""MCP server exposing the corporate-risk tool over streamable HTTP.

Runs as its own process, not inside the FastAPI app. The graph reaches
it through a MultiServerMCPClient (app/graph/tools.py). Start it with:

    uv run python -m app.mcp.risk_server
"""

from typing import Annotated, Literal

from mcp.server.fastmcp import FastMCP
from pydantic import Field

mcp = FastMCP("risk-tools", host="127.0.0.1", port=8100)

# Fields declared one per argument, not wrapped in a Pydantic model. A model
# param makes FastMCP nest the schema under "params", so the tool would only
# accept {"params": {...}} - the graph and the model call it with flat args.
# Annotated[..., Field(...)] keeps the gt=0 / enum constraints, so a bad
# argument still raises a ValidationError the client turns into a tool error.
CompanyName = Annotated[
    str,
    Field(
        description="Full legal name of the entity being analyzed, as it "
        "appears on regulatory filings.",
    ),
]
Industry = Annotated[
    Literal["Finance", "Tech", "Healthcare", "Manufacturing"],
    Field(
        description="Primary sector the company operates in. Pick the "
        "closest of the four; do not invent a value.",
    ),
]
ExposureValue = Annotated[
    float,
    Field(
        gt=0,
        le=1_000_000_000,
        description="Financial exposure in US dollars, greater than 0 and "
        "at most 1 billion.",
    ),
]
IsRegulated = Annotated[
    bool,
    Field(
        description="Whether the entity is under strict regulatory "
        "oversight (SEC, GDPR, HIPAA, and the like).",
    ),
]


@mcp.tool()
def calculate_corporate_risk(
    company_name: CompanyName,
    industry: Industry,
    exposure_value: ExposureValue,
    is_regulated: IsRegulated = True,
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


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
