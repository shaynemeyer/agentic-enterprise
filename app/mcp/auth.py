"""Bearer-token gate for the local MCP servers.

Both app/mcp/risk_server.py and app/mcp/fs_server.py wrap their ASGI app in require_bearer_token(). A request without a matching
`Authorization: Bearer <token>` header gets a 401 before it reaches the MCP handler. The token is settings.mcp_auth_token (MCP_AUTH_TOKEN in
.env).
"""

import hmac

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from app.core.config import settings

_PLACEHOLDER = "dev-only-change-me"

if settings.app_env != "dev" and settings.mcp_auth_token == _PLACEHOLDER:
    raise RuntimeError(
        "MCP_AUTH_TOKEN is still the placeholder; set a real secret "
        "(APP_ENV=dev bypasses this check for local work)."
    )

_EXPECTED = f"Bearer {settings.mcp_auth_token}".encode()


class _BearerTokenMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        presented = request.headers.get("Authorization", "").encode()
        if not hmac.compare_digest(presented, _EXPECTED):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        return await call_next(request)


def require_bearer_token(app):
    """Wrap an ASGI app so every request must carry the shared bearer token."""
    app.add_middleware(_BearerTokenMiddleware)
    return app
