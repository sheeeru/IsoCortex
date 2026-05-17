"""
IsoCortex — Authentication Middleware
======================================

JWT token verification dependency for FastAPI route protection.

SRS References:
  - Section 6: Authentication and User Management
  - NFR-05: All API endpoints require JWT (except login/setup)
  - NFR-06: bcrypt with cost factor 12

Author : Shaheer Qureshi
Project: IsoCortex
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

logger = logging.getLogger(__name__)

security = HTTPBearer(auto_error=False)


async def get_current_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> dict:
    """FastAPI dependency: Extract and validate JWT token.

    SRS Section 6.1: Authorization: Bearer <token>

    Returns the decoded token payload as a dict with:
    - sub (user_id)
    - username
    - role
    - exp

    Raises:
        401: Missing/invalid/expired token
        403: Account locked
    """
    from isocortex.auth import decode_access_token
    from isocortex.config import get_config

    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": "MISSING_TOKEN",
                "code": 401,
                "detail": "Authorization header with Bearer token is required",
                "request_id": _get_request_id(request),
            },
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = credentials.credentials
    try:
        config = get_config()
        payload = decode_access_token(
            token,
            config.server.jwt_secret_key,
            config.server.jwt_algorithm,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": "INVALID_TOKEN",
                "code": 401,
                "detail": str(exc),
                "request_id": _get_request_id(request),
            },
        )

    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": "EXPIRED_TOKEN",
                "code": 401,
                "detail": "Token has expired",
                "request_id": _get_request_id(request),
            },
        )

    # Check if account is locked
    username = payload.get("username", "")
    from isocortex.auth import get_user_manager
    user_mgr = get_user_manager()
    user = user_mgr.get_user_by_username(username)
    if user and user.locked_until:
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        lock_until = datetime.fromisoformat(user.locked_until.replace("Z", "+00:00"))
        if now < lock_until:
            remaining = (lock_until - now).total_seconds() / 60
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={
                    "error": "ACCOUNT_LOCKED",
                    "code": 403,
                    "detail": f"Account locked. Try again in {int(remaining)} minutes.",
                    "request_id": _get_request_id(request),
                },
            )

    return payload


async def require_admin(
    current_user: dict = Depends(get_current_user),
) -> dict:
    """FastAPI dependency: Require admin role.

    SRS Section 6.5: Admin endpoints require admin role.
    """
    if current_user.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "error": "FORBIDDEN",
                "code": 403,
                "detail": "Admin access required",
            },
        )
    return current_user


def _get_request_id(request: Request) -> str:
    """Extract request_id from request state (set by RequestIDMiddleware)."""
    return getattr(request.state, "request_id", "unknown")
