from datetime import datetime

from sqlalchemy import DateTime, func
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from app.core.config import settings

# Create the Engine: the central source of connections
engine = create_async_engine(
    settings.database_url, echo=False, pool_size=10, max_overflow=20
)

# Create SessionLocal: a factory for individual session objects
SessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


# Base class for our models
class Base(DeclarativeBase):
    pass


class TimestampMixin:
    """Adds created_at / updated_at to a model.

    Every table in this project should inherit this alongside Base so rows carry
    their own history. Both timestamps are filled by Postgres (server_default),
    so they are correct even for rows inserted outside the app. updated_at also
    bumps on any ORM-issued UPDATE via onupdate.
    """

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


# The dependency injected into endpoints
async def get_db():
    async with SessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
