# Project conventions

## Database models

Every SQLAlchemy model must inherit `TimestampMixin` (from `app/database.py`) alongside
`Base`, so every table has `created_at` and `updated_at` (`timestamptz`, Postgres
`server_default=func.now()`, `updated_at` also `onupdate=func.now()`).

```python
from app.database import Base, TimestampMixin

class Thing(Base, TimestampMixin):
    __tablename__ = "things"
    ...
```

No table without those two columns.

## Container ports

The `db` service publishes on host port **5433**, not the default 5432, so the stack
coexists with a native Postgres a dev may run on 5432. From the host, connect to
`localhost:5433`; inside the compose network, services use `db:5432`. When adding any
future service whose default port a dev might already run natively, map it to a
non-default host port the same way.
