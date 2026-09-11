import uuid

from sqlalchemy import Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base, TimestampMixin


class AgentExecution(Base, TimestampMixin):
    __tablename__ = "agent_executions"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    request_id: Mapped[uuid.UUID] = mapped_column(index=True)
    agent_id: Mapped[str] = mapped_column()
    status: Mapped[str] = mapped_column(default="pending")


class ThreadOwnership(Base, TimestampMixin):
    __tablename__ = "thread_ownership"

    thread_id: Mapped[str] = mapped_column(primary_key=True)
    owner_username: Mapped[str] = mapped_column(index=True)


class FileAuditEvent(Base, TimestampMixin):
    """One row per file write the agent made through the MCP filesystem
    server. `created_at` (from TimestampMixin) is the write time."""

    __tablename__ = "file_audit_events"

    id: Mapped[int] = mapped_column(primary_key=True)
    path: Mapped[str] = mapped_column(String(1024))
    action: Mapped[str] = mapped_column(String(32))  # "write" | "create"
    byte_count: Mapped[int] = mapped_column()
    preview: Mapped[str] = mapped_column(Text)  # first 200 chars written


class BalanceQueryEvent(Base, TimestampMixin):
    """One row per finance-API lookup the agent made. `created_at` is the
    query time. thread_id / username tie it back to the graph run so it
    joins to thread_ownership."""

    __tablename__ = "balance_query_events"

    id: Mapped[int] = mapped_column(primary_key=True)
    account_id: Mapped[str] = mapped_column(String(64))
    thread_id: Mapped[str | None] = mapped_column(String(128))
    username: Mapped[str | None] = mapped_column(String(128))
    outcome: Mapped[str] = mapped_column(String(16))  # "ok" | "error"
    detail: Mapped[str | None] = mapped_column(String(256))  # error text, if any


class CodeExecutionEvent(Base, TimestampMixin):
    """One row per sandboxed code execution. `created_at` is when the
    container was launched. thread_id ties it back to the graph run."""

    __tablename__ = "code_execution_events"

    id: Mapped[int] = mapped_column(primary_key=True)
    thread_id: Mapped[str | None] = mapped_column(String(128))
    code_length: Mapped[int] = mapped_column(Integer)
    outcome: Mapped[str] = mapped_column(String(16))  # "ok" | "error"
    detail: Mapped[str | None] = mapped_column(
        String(512)
    )  # stdout/stderr tail, or error text
