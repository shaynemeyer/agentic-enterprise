"""Password hashing and JWT mint/verify for the API's OAuth2 password flow."""

from datetime import UTC, datetime, timedelta

import jwt
from pwdlib import PasswordHash
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.core.config import settings

_hasher = PasswordHash.recommended()

# key_func here is only the fallback for routes that do not override it.
# The protected run routes override it with the JWT subject (see auth.py).
limiter = Limiter(key_func=get_remote_address)


def hash_password(plain: str) -> str:
    return _hasher.hash(plain)


def verify_password(plain: str, hashed: str) -> bool:
    return _hasher.verify(plain, hashed)


def create_access_token(subject: str) -> str:
    expire = datetime.now(UTC) + timedelta(minutes=settings.jwt_expire_minutes)
    payload = {"sub": subject, "exp": expire}
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def decode_subject(token: str) -> str:
    """Return the `sub` claim, or raise `jwt.InvalidTokenError` if the token is bad."""
    payload = jwt.decode(
        token, settings.jwt_secret, algorithms=[settings.jwt_algorithm]
    )
    subject = payload.get("sub")
    if not subject:
        raise jwt.InvalidTokenError("token missing 'sub'")
    return subject
