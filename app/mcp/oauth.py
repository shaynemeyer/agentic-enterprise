"""OAuth2 client-credentials token cache.

ClientCredentialsToken.value() returns a currently-valid access token,
fetching one on first use and refreshing when the cached one is within
REFRESH_MARGIN_S of expiry. Call force_refresh() after a 401 to discard
a token the server rejected before its stated expiry.

One fetch at a time: concurrent callers that arrive while a refresh is
in flight wait on the same lock and reuse its result instead of each
firing their own token POST.
"""

import asyncio
import logging
import time

import httpx

logger = logging.getLogger("enterprise_agent.oauth")

REFRESH_MARGIN_S = 60
_FETCH_TIMEOUT_S = 10.0


class ClientCredentialsToken:
    def __init__(self, token_url: str, client_id: str, client_secret: str, scope: str):
        self._token_url = token_url
        self._client_id = client_id
        self._client_secret = client_secret
        self._scope = scope
        self._access_token: str | None = None
        self._expires_at = 0.0
        self._lock = asyncio.Lock()

    def _is_stale(self) -> bool:
        return (
            self._access_token is None
            or time.time() >= self._expires_at - REFRESH_MARGIN_S
        )

    async def value(self) -> str:
        """A valid access token, fetched or refreshed as needed."""
        if not self._is_stale():
            return self._access_token
        async with self._lock:
            # Another coroutine may have refreshed while we waited.
            if self._is_stale():
                await self._fetch()
            return self._access_token

    def force_refresh(self) -> None:
        """Drop the cached token so the next value() call re-fetches."""
        self._access_token = None

    async def _fetch(self) -> None:
        async with httpx.AsyncClient(timeout=_FETCH_TIMEOUT_S) as client:
            resp = await client.post(
                self._token_url,
                data={
                    "grant_type": "client_credentials",
                    "client_id": self._client_id,
                    "client_secret": self._client_secret,
                    "scope": self._scope,
                },
            )
            resp.raise_for_status()
            body = resp.json()
        self._access_token = body["access_token"]
        self._expires_at = time.time() + float(body["expires_in"])
        logger.info(
            "fetched client-credentials token, expires in %ss", body["expires_in"]
        )
