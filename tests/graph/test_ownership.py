"""claim_or_check against a real Postgres session - no admin short-circuit.

  export CHECKPOINT_TEST_DSN=postgresql://agent:agent@localhost:5433/agent_db
Skipped when unset. Uses the same DSN var as tests/graph/test_thread_history.py
since both need the same running `db` service (CLAUDE.md: host port 5433).
"""

import os
import uuid

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.database import Base
from app.graph.ownership import claim_or_check, is_admin

DB_URL = os.getenv("CHECKPOINT_TEST_DSN")
pytestmark = pytest.mark.skipif(not DB_URL, reason="CHECKPOINT_TEST_DSN not set")

ASYNC_DB_URL = (DB_URL or "").replace("postgresql://", "postgresql+asyncpg://")


@pytest_asyncio.fixture
async def db():
    engine = create_async_engine(ASYNC_DB_URL)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    session = async_sessionmaker(engine, expire_on_commit=False)()
    yield session
    await session.close()
    await engine.dispose()


@pytest.mark.asyncio
async def test_first_caller_claims_an_unseen_thread(db):
    thread_id = f"claim-{uuid.uuid4()}"
    assert await claim_or_check(db, thread_id, "alice") is True
    await db.commit()


@pytest.mark.asyncio
async def test_second_non_owner_is_denied(db):
    thread_id = f"claim-{uuid.uuid4()}"
    assert await claim_or_check(db, thread_id, "alice") is True
    await db.commit()

    assert await claim_or_check(db, thread_id, "bob") is False
    await db.commit()


@pytest.mark.asyncio
async def test_owner_passes_on_every_later_call(db):
    thread_id = f"claim-{uuid.uuid4()}"
    assert await claim_or_check(db, thread_id, "alice") is True
    await db.commit()

    assert await claim_or_check(db, thread_id, "alice") is True
    await db.commit()


def test_is_admin_matches_only_the_demo_username():
    from app.core.config import settings

    assert is_admin(settings.demo_username) is True
    assert is_admin("not-the-demo-user") is False
