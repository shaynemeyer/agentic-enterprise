"""OAuth2 password-flow login and the `get_current_user` dependency."""

import jwt
from fastapi import APIRouter, Depends, HTTPException, Path, Request, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.security import (
    create_access_token,
    decode_subject,
    hash_password,
    verify_password,
)
from app.database import get_db
from app.graph.ownership import claim_or_check

router = APIRouter(tags=["Auth"])

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/token")


def _seed_users() -> dict[str, dict[str, str]]:
    """Demo user store, built from DEMO_PASSWORD/DEMO2_PASSWORD in .env.

    Each user is seeded only if its password is set; either or both may be
    empty. A real store is a SQLAlchemy model with TimestampMixin (see
    CLAUDE.md), queried through the request's db session.
    """
    users: dict[str, dict[str, str]] = {}
    if settings.demo_password:
        users[settings.demo_username] = {
            "username": settings.demo_username,
            "hashed_password": hash_password(settings.demo_password),
        }
    if settings.demo2_password:
        users[settings.demo2_username] = {
            "username": settings.demo2_username,
            "hashed_password": hash_password(settings.demo2_password),
        }
    return users


_USERS = _seed_users()


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class CurrentUser(BaseModel):
    username: str


_credentials_error = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Could not validate credentials",
    headers={"WWW-Authenticate": "Bearer"},
)


@router.post("/token", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = _USERS.get(form_data.username)
    if not user or not verify_password(form_data.password, user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return Token(access_token=create_access_token(user["username"]))


async def get_current_user(token: str = Depends(oauth2_scheme)) -> CurrentUser:
    try:
        username = decode_subject(token)
    except jwt.InvalidTokenError:
        raise _credentials_error from None
    if username not in _USERS:
        raise _credentials_error
    return CurrentUser(username=username)


def user_key(request: Request) -> str:
    """Rate-limit bucket key: the JWT subject, or the client IP if unauthenticated.

    slowapi calls this with the raw Request, before/around the dependency graph,
    so it cannot use Depends(get_current_user). An unauthenticated request will be
    rejected by get_current_user anyway; keying it by IP just keeps the buckets
    tidy until then.
    """
    auth = request.headers.get("Authorization", "")
    scheme, _, token = auth.partition(" ")
    if scheme.lower() == "bearer" and token:
        try:
            return decode_subject(token)
        except jwt.InvalidTokenError:
            pass
    return request.client.host if request.client else "anonymous"


async def require_thread_owner(
    thread_id: str = Path(...),
    user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> CurrentUser:
    """403 unless `user` owns `thread_id` or is admin. Claims an unowned thread."""
    if not await claim_or_check(db, thread_id, user.username):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not the owner of this thread_id.",
        )
    return user
