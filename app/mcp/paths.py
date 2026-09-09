"""Path-jail helper for the filesystem MCP server.

resolve_in_root() is the single choke point: every filesystem tool
turns a caller-supplied path into an absolute path with it, and the
call raises before any I/O if the path escapes the configured root.
"""

from pathlib import Path

from app.core.config import settings

ROOT = Path(settings.agent_files_dir).resolve()


def resolve_in_root(relative_path: str) -> Path:
    """Resolve `relative_path` under ROOT, or raise ValueError if it escapes.

    Rejects absolute paths, `..` traversal, and symlinks that point
    outside ROOT (resolve() follows them, is_relative_to() then fails).
    """
    candidate = (ROOT / relative_path).resolve()
    if not candidate.is_relative_to(ROOT):
        raise ValueError(f"path escapes the agent files directory: {relative_path!r}")
    return candidate
