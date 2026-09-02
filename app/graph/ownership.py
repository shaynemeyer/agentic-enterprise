"""Thread ownership: who created a thread_id, and who may act on it.

One row per thread_id in `thread_ownership`, written the first time an
admin route sees that thread_id. Ownership is claimed, not assigned - the
first authenticated caller to reach any /admin/threads/{id} route becomes
the owner if no row exists yet. This matches how thread_id is already
created today: POST /run and GET /ask write a checkpoint under a thread_id
with no separate "create thread" step, so there is no earlier point to
record ownership at.
"""

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models import ThreadOwnership


def is_admin(username: str) -> bool:
    """True for the one demo admin account.

    Placeholder for a real role claim on the JWT - see Roadmap. Centralized
    here so the one call site that needs to change later is this function,
    not every route that currently checks user.username directly.
    """
    return username == settings.demo_username


async def claim_or_check(db: AsyncSession, thread_id: str, username: str) -> bool:
    """True if `username` may act on `thread_id`: owns it, or is admin.

    Inserts an ownership row on first sight of this thread_id, owned by
    whoever asked first - ON CONFLICT DO NOTHING so a race between two
    concurrent first-touches picks one winner instead of erroring. Every
    call after that first insert just reads the existing row.
    """
    if is_admin(username):
        return True

    stmt = (
        insert(ThreadOwnership)
        .values(thread_id=thread_id, owner_username=username)
        .on_conflict_do_nothing(index_elements=["thread_id"])
    )
    await db.execute(stmt)

    owner = await db.scalar(
        select(ThreadOwnership.owner_username).where(
            ThreadOwnership.thread_id == thread_id
        )
    )
    return owner == username
