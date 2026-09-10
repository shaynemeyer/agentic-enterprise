"""MCP server exposing one internal-API tool: get_account_balance.

Its own process. Two auth boundaries meet here:
  - inbound: the shared bearer token (app/mcp/auth.py), same as Lab 45.
  - outbound: an OAuth2 client-credentials token for the finance API
    (app/mcp/oauth.py), fetched and refreshed here.

Every lookup, success or failure, is a balance_query_events row.

Start it with:

    uv run python -m app.mcp.finance_server
"""

from typing import Annotated

import httpx
from mcp.server.fastmcp import Context, FastMCP
from pydantic import Field
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.core.config import settings
from app.mcp.auth import require_bearer_token
from app.mcp.oauth import ClientCredentialsToken
from app.models import BalanceQueryEvent

_PLACEHOLDER = "dev-only-change-me"
if settings.app_env != "dev" and settings.finance_oauth_client_secret == _PLACEHOLDER:
    raise RuntimeError(
        "FINANCE_OAUTH_CLIENT_SECRET is still the placeholder; set a real "
        "secret (APP_ENV=dev bypasses this for local work)."
    )

mcp = FastMCP("finance-tools", host="127.0.0.1", port=8103)

_engine = create_async_engine(settings.database_url)
_Session = async_sessionmaker(_engine, expire_on_commit=False)

_token = ClientCredentialsToken(
    token_url=settings.finance_oauth_token_url,
    client_id=settings.finance_oauth_client_id,
    client_secret=settings.finance_oauth_client_secret,
    scope=settings.finance_oauth_scope,
)

_API_TIMEOUT_S = 10.0

AccountId = Annotated[
    str,
    Field(
        description="Enterprise account identifier, e.g. 'ACC-1001'. "
        "Uppercase letters, a hyphen, then digits.",
    ),
]


async def _call_api(account_id: str) -> dict:
    """GET the balance, retrying once after a 401 (token revoked early)."""
    url = f"{settings.finance_api_url}/accounts/{account_id}/balance"
    async with httpx.AsyncClient(timeout=_API_TIMEOUT_S) as client:
        for attempt in (1, 2):
            headers = {"Authorization": f"Bearer {await _token.value()}"}
            resp = await client.get(url, headers=headers)
            if resp.status_code == 401 and attempt == 1:
                _token.force_refresh()
                continue
            resp.raise_for_status()
            return resp.json()


async def _audit(account_id, thread_id, username, outcome, detail) -> None:
    async with _Session() as session:
        session.add(
            BalanceQueryEvent(
                account_id=account_id,
                thread_id=thread_id,
                username=username,
                outcome=outcome,
                detail=detail,
            )
        )
        await session.commit()


@mcp.tool()
async def get_account_balance(account_id: AccountId, ctx: Context) -> dict:
    """Look up the current balance of one internal enterprise account.

    Use this when asked for an account's balance or currency. Returns the
    account id, numeric balance, and ISO currency code. Fails if the
    account does not exist or the finance service is unavailable.
    """
    thread_id = ctx.client_id  # may be None; see note below
    try:
        result = await _call_api(account_id)
    except httpx.HTTPError as exc:
        await _audit(account_id, thread_id, None, "error", str(exc)[:256])
        raise
    await _audit(account_id, thread_id, None, "ok", None)
    return result


if __name__ == "__main__":
    import uvicorn

    app = require_bearer_token(mcp.streamable_http_app())
    uvicorn.run(app, host="127.0.0.1", port=8103)
