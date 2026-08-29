from datetime import UTC, datetime

from pydantic import BaseModel, Field


class ErrorResponse(BaseModel):
    success: bool = False
    error_code: str = Field(description="Machine-readable error code")
    message: str = Field(description="Human-readable error message")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    details: dict | list | str | None = None
    trace_id: str | None = None  # populated in Lab 14 (request-id tracking)
