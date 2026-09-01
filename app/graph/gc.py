"""Checkpoint retention: drop threads whose newest checkpoint is past the cutoff.

The checkpoint tables are owned by langgraph-checkpoint-postgres, not by the
SQLAlchemy models in app/models.py. Selection is one raw query through the
psycopg pool the lifespan already owns; deletion goes through the saver's public
adelete_thread so it stays correct across saver releases.
"""

import logging
from contextlib import asynccontextmanager

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool

logger = logging.getLogger("checkpoint_gc")

# Every checkpoint payload carries a 'ts' field (ISO-8601 UTC) - the checkpoint's
# own record of when it was written. MAX(ts) per thread_id is that thread's last
# activity; older than the cutoff means the thread is abandoned.
_STALE_QUERY = """
SELECT thread_id
FROM checkpoints
GROUP BY thread_id
HAVING MAX((checkpoint ->> 'ts')::timestamptz) < now() - make_interval(days => %s)
"""


@asynccontextmanager
async def _conn(saver: AsyncPostgresSaver):
    """Yield a connection whether the saver holds a pool or a single connection.

    AsyncPostgresSaver(pool) -> saver.conn is an AsyncConnectionPool.
    AsyncPostgresSaver.from_conn_string(...) -> saver.conn is an AsyncConnection.
    """
    if isinstance(saver.conn, AsyncConnectionPool):
        async with saver.conn.connection() as conn:
            yield conn
    else:
        yield saver.conn


async def find_stale_threads(
    saver: AsyncPostgresSaver, retention_days: int
) -> list[str]:
    async with _conn(saver) as conn:
        cur = await conn.execute(_STALE_QUERY, (retention_days,))
        rows = await cur.fetchall()
    # LangGraph opens its connection with row_factory=dict_row, so a row may be a
    # dict; a plain pooled connection gives tuples. Handle both.
    return [r["thread_id"] if isinstance(r, dict) else r[0] for r in rows]


async def sweep(saver: AsyncPostgresSaver, retention_days: int) -> dict:
    """Delete every thread past the cutoff. Returns a summary dict."""
    stale = await find_stale_threads(saver, retention_days)
    for thread_id in stale:
        await saver.adelete_thread(thread_id)

    summary = {"deleted": len(stale), "retention_days": retention_days}
    if stale:
        logger.info("checkpoint GC: deleted %d stale thread(s)", len(stale))
    return summary
