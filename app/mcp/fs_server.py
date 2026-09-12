"""MCP server for jailed local file access, over streamable HTTP.

Its own process, not inside the FastAPI app. Every path goes through
app/mcp/paths.resolve_in_root(); every write is logged to
file_audit_events. Behind the shared bearer token (app/mcp/auth.py).

Start it with:

    uv run python -m app.mcp.fs_server
"""

from typing import Annotated

import anyio
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from pydantic import Field
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.core.config import settings
from app.mcp.auth import require_bearer_token
from app.mcp.paths import ROOT, resolve_in_root
from app.models import FileAuditEvent

# See app/mcp/risk_server.py for why host.containers.internal is added here -
# a containerized agent-api (Lab 47) reaches this process by that hostname,
# which FastMCP's default DNS-rebinding guard would otherwise reject.
mcp = FastMCP(
    "filesystem-tools",
    host="127.0.0.1",
    port=8101,
    transport_security=TransportSecuritySettings(
        allowed_hosts=["127.0.0.1:*", "localhost:*", "[::1]:*", "host.containers.internal:*"],
        allowed_origins=[
            "http://127.0.0.1:*",
            "http://localhost:*",
            "http://[::1]:*",
            "http://host.containers.internal:*",
        ],
    ),
)

_engine = create_async_engine(settings.database_url)
_Session = async_sessionmaker(_engine, expire_on_commit=False)

RelPath = Annotated[
    str,
    Field(
        description="Path relative to the agent files directory. Must not "
        "be absolute and must not contain '..'. Example: 'notes/status.txt'.",
    ),
]


@mcp.tool()
async def read_text_file(path: RelPath) -> str:
    """Read a UTF-8 text file from the agent files directory.

    Use this to read a document the agent has been told to consult or a
    file it wrote earlier. Fails if the path escapes the directory or the
    file does not exist. Every read is recorded in the audit log, same as
    every write.
    """
    target = resolve_in_root(path)
    content = await anyio.to_thread.run_sync(lambda: target.read_text(encoding="utf-8"))

    async with _Session() as session:
        session.add(
            FileAuditEvent(
                path=path,
                action="read",
                byte_count=len(content.encode("utf-8")),
                preview=content[:200],
            )
        )
        await session.commit()

    return content


@mcp.tool()
async def write_text_file(path: RelPath, content: str) -> str:
    """Write UTF-8 text to a file in the agent files directory, creating
    parent folders as needed and overwriting any existing file.

    Use this to save a result for later. Every write is recorded in the
    audit log. Returns a one-line confirmation.
    """
    data = content.encode("utf-8")
    if len(data) > settings.agent_files_max_write_bytes:
        raise ValueError(
            f"write of {len(data)} bytes exceeds the "
            f"{settings.agent_files_max_write_bytes}-byte limit"
        )

    target = resolve_in_root(path)
    existed = target.exists()

    def _write() -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)

    await anyio.to_thread.run_sync(_write)

    async with _Session() as session:
        session.add(
            FileAuditEvent(
                path=path,
                action="write" if existed else "create",
                byte_count=len(data),
                preview=content[:200],
            )
        )
        await session.commit()

    return f"wrote {len(data)} bytes to {path}"


@mcp.tool()
async def list_directory(path: RelPath = ".") -> list[str]:
    """List the names in one directory of the agent files directory.

    Pass '.' (the default) for the top level. Fails if the path escapes
    the directory.
    """
    d = resolve_in_root(path)
    return sorted(p.name for p in d.iterdir())


if __name__ == "__main__":
    ROOT.mkdir(parents=True, exist_ok=True)
    app = require_bearer_token(mcp.streamable_http_app())
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8101)
