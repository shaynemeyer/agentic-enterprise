import uuid

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
