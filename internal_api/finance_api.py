"""Mock internal finance API for Lab 46.

Stands in for a protected enterprise service (SAP, an internal ledger).
Issues short-lived access tokens via the OAuth2 client-credentials
grant and rejects any /accounts call without a valid one.

Run it:  uv run uvicorn internal_api.finance_api:app --port 8102
"""

import time
import uuid

from fastapi import FastAPI, Form, Header, HTTPException

from app.core.config import settings

app = FastAPI(title="mock-finance-api")

# In a real service these live in an identity provider. Here: one client.
_CLIENTS = {settings.finance_oauth_client_id: settings.finance_oauth_client_secret}
_TOKEN_TTL_S = 3600
# token -> unix expiry
_ISSUED: dict[str, float] = {}

_BALANCES = {
    "ACC-1001": {"balance": 1_250_500.75, "currency": "USD"},
    "ACC-2002": {"balance": 84_000.00, "currency": "EUR"},
}


@app.post("/oauth/token")
async def issue_token(
    grant_type: str = Form(...),
    client_id: str = Form(...),
    client_secret: str = Form(...),
    scope: str = Form(""),
):
    if grant_type != "client_credentials":
        raise HTTPException(400, "unsupported_grant_type")
    if _CLIENTS.get(client_id) != client_secret:
        raise HTTPException(401, "invalid_client")

    token = uuid.uuid4().hex
    _ISSUED[token] = time.time() + _TOKEN_TTL_S
    return {
        "access_token": token,
        "token_type": "Bearer",
        "expires_in": _TOKEN_TTL_S,
        "scope": scope,
    }


def _require_token(authorization: str | None) -> None:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "missing bearer token")
    token = authorization.removeprefix("Bearer ")
    expiry = _ISSUED.get(token)
    if expiry is None:
        raise HTTPException(401, "unknown token")
    if time.time() > expiry:
        del _ISSUED[token]
        raise HTTPException(401, "token expired")


@app.get("/accounts/{account_id}/balance")
async def get_balance(account_id: str, authorization: str | None = Header(None)):
    _require_token(authorization)
    record = _BALANCES.get(account_id)
    if record is None:
        raise HTTPException(404, f"no such account: {account_id}")
    return {"account_id": account_id, **record}
