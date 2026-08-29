"""Async-safe request-id storage shared by middleware, logging, and the graph."""

import contextvars
from uuid import uuid4

_request_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "request_id", default=""
)

REQUEST_ID_HEADER = "X-Request-ID"


def get_request_id() -> str:
    """Current request id, or "" outside a request."""
    return _request_id.get()


def set_request_id(request_id: str) -> None:
    _request_id.set(request_id)


def new_request_id() -> str:
    return str(uuid4())
