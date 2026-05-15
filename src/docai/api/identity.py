"""Cookie-derived anonymous user identity.

Single FastAPI dependency. No auth, no login. Each browser gets a
random UUID stored in an httpOnly cookie; the cookie *is* the user.
The whole point of this single helper is that switching to real auth
later (verified token, magic link, OAuth) only rewrites this file —
no schema migration, no API changes.
"""
from __future__ import annotations

from uuid import uuid4

from fastapi import Request, Response

from docai.config import settings

COOKIE_MAX_AGE_SECONDS = 31_536_000  # 1 year


def get_current_user_id(request: Request, response: Response) -> str:
    """Resolve the current user ID. Generates one + sets the cookie if missing.

    Never raises (no 401 path) — identity is always assigned. Callers
    that want to know "is this a brand-new user this request" can check
    the cookie themselves; this helper hides that distinction.
    """
    cookie_name = settings.SESSION_USER_ID_COOKIE
    user_id = request.cookies.get(cookie_name)
    if user_id is None:
        user_id = str(uuid4())
        response.set_cookie(
            key=cookie_name,
            value=user_id,
            httponly=True,
            samesite="lax",
            path="/",
            max_age=COOKIE_MAX_AGE_SECONDS,
        )
    return user_id
