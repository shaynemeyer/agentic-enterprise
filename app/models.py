import uuid

from sqlalchemy import String, Text
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
