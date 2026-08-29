"""Payload carried in each SSE `data:` frame from the streaming run endpoint."""

from pydantic import BaseModel


class StreamEvent(BaseModel):
    phase: str  # "node" | "status" | "done" | "error"
    node: str | None = None
    content: str
