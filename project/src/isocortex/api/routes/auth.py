"""
IsoCortex — Auth Route Handlers
================================

FastAPI route handlers for authentication and user management.

All endpoints follow the SRS Section 6 specification:
  - First-run admin setup (only when 0 users exist)
  - JWT-based login with access + refresh tokens
  - Token refresh and logout
  - Admin-only user CRUD operations
  - Self-service password changes

SRS References:
  - Section 6:    Authentication and User Management
  - Section 6.1:  Login endpoint
  - Section 6.2:  Password requirements (min 12 chars, complexity)
  - Section 6.3:  Account lockout (5 attempts, 15 min)
  - Section 6.4:  First-run setup
  - Section 6.5:  Role-based access control
  - Section 10:   Error handling specification
  - NFR-05:       All API endpoints require JWT (except login/setup)
  - NFR-06:       bcrypt with cost factor 12

Author : Shaheer Qureshi
Project: IsoCortex
"""

from __future__ import annotations

import hashlib
import logging
import secrets
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from isocortex.api.middleware.auth import get_current_user, require_admin
from isocortex.api.schemas.auth import (
    ChangePasswordRequest,
    ChangeRoleRequest,
    CreateUserRequest,
    LoginRequest,
    LoginResponse,
    RefreshRequest,
    SetupRequest,
    SetupResponse,
    UserResponse,
    UserProfileResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Authentication"])

# ---------------------------------------------------------------------------
# Refresh-token SQLite store
# ---------------------------------------------------------------------------

REFRESH_TOKEN_EXPIRE_DAYS: int = 7

# SRS Section 6.3: Account lockout policy
MAX_FAILED_ATTEMPTS: int = 5
LOCKOUT_MINUTES: int = 15


def _get_refresh_db() -> sqlite3.Connection:
    """Return a SQLite connection for the refresh-tokens database.

    The database file is stored in the configured data directory (defaults to
    ``~/.isocortex``).  The ``refresh_tokens`` table is created automatically
    if it does not already exist.
    """
    try:
        from isocortex.config.settings import get_config
        cfg = get_config()
        data_dir = cfg.data_dir
    except Exception:
        data_dir = Path.home() / ".isocortex"

    db_path = Path(data_dir) / "refresh_tokens.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS refresh_tokens ("
        "  token TEXT PRIMARY KEY,"
        "  user_id TEXT NOT NULL,"
        "  role TEXT NOT NULL,"
        "  created_at REAL NOT NULL,"
        "  expires_at REAL NOT NULL"
        ")"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_refresh_expires "
        "ON refresh_tokens(expires_at)"
    )
    conn.commit()
    return conn

security = HTTPBearer(auto_error=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_request_id(request: Request) -> str:
    """Extract request_id from request state (set by RequestIDMiddleware)."""
    return getattr(request.state, "request_id", "unknown")


def _hash_refresh_token(token: str) -> str:
    """Hash a refresh token for secure storage (SHA-256).

    Only the hash is stored in the database.  This mirrors the pattern
    used for API keys and ensures a compromised DB does not leak
    usable refresh tokens.
    """
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _generate_refresh_token() -> str:
    """Generate a cryptographically random refresh token."""
    return f"iso_rt_{secrets.token_urlsafe(48)}"


def _store_refresh_token(
    token: str,
    user_id: str,
    role: str,
    expires_days: int = REFRESH_TOKEN_EXPIRE_DAYS,
) -> None:
    """Persist a refresh token (SHA-256 hashed) in the SQLite store."""
    now = datetime.now(timezone.utc)
    expires_at = now + timedelta(days=expires_days)
    token_hash = _hash_refresh_token(token)
    conn = _get_refresh_db()
    try:
        conn.execute(
            "INSERT INTO refresh_tokens (token, user_id, role, created_at, expires_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (token_hash, user_id, role, now.timestamp(), expires_at.timestamp()),
        )
        conn.commit()
    finally:
        conn.close()


def _invalidate_refresh_token(token: str) -> bool:
    """Remove a refresh token by its hash. Returns True if it existed."""
    token_hash = _hash_refresh_token(token)
    conn = _get_refresh_db()
    try:
        cursor = conn.execute(
            "DELETE FROM refresh_tokens WHERE token = ?", (token_hash,)
        )
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


def _get_refresh_token(token: str) -> Optional[dict[str, Any]]:
    """Look up a refresh token by hash and check expiry. Returns payload or None."""
    token_hash = _hash_refresh_token(token)
    conn = _get_refresh_db()
    try:
        row = conn.execute(
            "SELECT user_id, role, expires_at FROM refresh_tokens WHERE token = ?",
            (token_hash,),
        ).fetchone()
    finally:
        conn.close()

    if row is None:
        return None

    user_id, role, expires_at_ts = row
    if datetime.now(timezone.utc).timestamp() >= expires_at_ts:
        _invalidate_refresh_token(token)
        return None

    return {
        "user_id": user_id,
        "role": role,
    }


def _cleanup_expired_refresh_tokens() -> int:
    """Remove expired refresh tokens. Returns count removed."""
    conn = _get_refresh_db()
    try:
        cursor = conn.execute(
            "DELETE FROM refresh_tokens WHERE expires_at < ?",
            (datetime.now(timezone.utc).timestamp(),),
        )
        conn.commit()
        removed = cursor.rowcount
    finally:
        conn.close()
    if removed:
        logger.debug("[AUTH-ROUTES] Cleaned up %d expired refresh tokens", removed)
    return removed


def _user_to_response(user: Any) -> UserResponse:
    """Convert a User dataclass to UserResponse schema."""
    return UserResponse(
        user_id=user.user_id,
        username=user.username,
        email=user.email if hasattr(user, "email") else None,
        role=user.role,
        is_active=user.is_active,
        created_at=getattr(user, "created_at", ""),
        last_login=getattr(user, "last_login", None),
    )


# ---------------------------------------------------------------------------
# POST /api/v1/auth/setup — First-run admin setup
# ---------------------------------------------------------------------------

@router.post(
    "/setup",
    response_model=SetupResponse,
    status_code=status.HTTP_201_CREATED,
    summary="First-run admin setup",
    description=(
        "Create the initial admin user. Only allowed when no users exist "
        "(SRS Section 6.4)."
    ),
    responses={
        409: {"description": "Setup already completed (users exist)"},
        422: {"description": "Validation error"},
    },
)
async def setup_admin(
    body: SetupRequest,
    request: Request,
) -> SetupResponse:
    """First-run admin setup — only works when 0 users exist.

    SRS Section 6.4:
    - Returns 201 Created with access_token + refresh_token
    - Returns 409 Conflict if any users already exist
    """
    from isocortex.auth import create_access_token, get_user_manager
    from isocortex.config import get_config

    request_id = _get_request_id(request)
    config = get_config()
    user_mgr = get_user_manager()

    # Check that no users exist
    existing = user_mgr.list_users(offset=0, limit=1)
    if existing:
        logger.warning(
            "[AUTH-ROUTES] Setup attempted but %d users already exist  request_id=%s",
            len(existing),
            request_id,
        )
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "error": "SETUP_COMPLETED",
                "code": 409,
                "detail": "Initial setup has already been completed. Use /auth/login.",
                "request_id": request_id,
            },
        )

    # Create the admin user
    try:
        user = user_mgr.create_user(
            username=body.username,
            email=body.email or "",
            password=body.password,
            role="admin",
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "error": "USER_EXISTS",
                "code": 409,
                "detail": str(exc),
                "request_id": request_id,
            },
        )

    # Issue JWT access token
    access_token = create_access_token(
        user_id=user.user_id,
        role=user.role,
        secret_key=config.server.jwt_secret_key,
        algorithm=config.server.jwt_algorithm,
        expires_minutes=config.server.jwt_expire_minutes,
    )

    # Issue refresh token
    refresh_token = _generate_refresh_token()
    _store_refresh_token(
        token=refresh_token,
        user_id=user.user_id,
        role=user.role,
    )

    logger.info(
        "[AUTH-ROUTES] Admin setup completed  user_id=%s  username=%s  request_id=%s",
        user.user_id,
        user.username,
        request_id,
    )

    return SetupResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in=config.server.jwt_expire_minutes * 60,
    )


# ---------------------------------------------------------------------------
# POST /api/v1/auth/login — Login
# ---------------------------------------------------------------------------

@router.post(
    "/login",
    response_model=LoginResponse,
    summary="Authenticate and obtain tokens",
    description=(
        "Authenticate with username (or email) and password. "
        "Returns JWT access_token + refresh_token. "
        "SRS Section 6.1."
    ),
    responses={
        401: {"description": "Invalid credentials"},
        403: {"description": "Account locked"},
        422: {"description": "Validation error"},
    },
)
async def login(
    body: LoginRequest,
    request: Request,
) -> LoginResponse:
    """Authenticate a user and return JWT + refresh tokens.

    SRS Section 6.1:
    - Accepts username or email
    - Returns 401 for invalid credentials
    - Returns 403 for locked accounts

    SRS Section 6.3:
    - 5 consecutive failed login attempts triggers 15-minute lockout
    - Successful login resets the counter
    """
    from isocortex.auth import create_access_token, get_user_manager
    from isocortex.config import get_config

    request_id = _get_request_id(request)
    config = get_config()
    user_mgr = get_user_manager()

    # --- SRS 6.3: Check if account is locked BEFORE attempting auth ---
    user = user_mgr.get_user_by_username(body.username)
    if user is None:
        # User not found — don't reveal existence
        logger.debug("[AUTH-ROUTES] Login failed: user not found  request_id=%s", request_id)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": "INVALID_CREDENTIALS",
                "code": 401,
                "detail": "Username or password is incorrect",
                "request_id": request_id,
            },
            headers={"WWW-Authenticate": "Bearer"},
        )

    if user.locked_until:
        try:
            lock_until = datetime.fromisoformat(user.locked_until)
            if datetime.now(timezone.utc) < lock_until:
                remaining_minutes = (lock_until - datetime.now(timezone.utc)).seconds // 60 + 1
                logger.warning(
                    "[AUTH-ROUTES] Login rejected: account locked  user_id=%s  "
                    "remaining=%dmin  request_id=%s",
                    user.user_id, remaining_minutes, request_id,
                )
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail={
                        "error": "ACCOUNT_LOCKED",
                        "code": 403,
                        "detail": (
                            f"Account is locked due to too many failed login attempts. "
                            f"Try again in {remaining_minutes} minute(s)."
                        ),
                        "request_id": request_id,
                    },
                )
            else:
                # Lockout expired — reset counter
                user_mgr._reset_failed_attempts(user.user_id)
        except (ValueError, TypeError):
            pass  # Invalid locked_until value, ignore

    result = user_mgr.authenticate(
        username_or_email=body.username,
        password=body.password,
        secret_key=config.server.jwt_secret_key,
        algorithm=config.server.jwt_algorithm,
        expire_minutes=config.server.jwt_expire_minutes,
    )

    if result is None:
        # --- SRS 6.3: Increment failed login counter ---
        user_mgr._increment_failed_attempt(user.user_id)
        attempts = (user.failed_login_attempts or 0) + 1

        logger.warning(
            "[AUTH-ROUTES] Login failed for username=%r  attempts=%d/%d  request_id=%s",
            body.username, attempts, MAX_FAILED_ATTEMPTS, request_id,
        )

        if attempts >= MAX_FAILED_ATTEMPTS:
            # Lock the account
            lock_until = datetime.now(timezone.utc) + timedelta(minutes=LOCKOUT_MINUTES)
            user_mgr._lock_account(user.user_id, lock_until.isoformat())
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={
                    "error": "ACCOUNT_LOCKED",
                    "code": 403,
                    "detail": (
                        f"Account locked due to {MAX_FAILED_ATTEMPTS} failed login attempts. "
                        f"Try again in {LOCKOUT_MINUTES} minutes."
                    ),
                    "request_id": request_id,
                },
            )

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": "INVALID_CREDENTIALS",
                "code": 401,
                "detail": "Username or password is incorrect",
                "request_id": request_id,
            },
            headers={"WWW-Authenticate": "Bearer"},
        )

    # --- SRS 6.3: Successful login resets failed attempts ---
    token = result["token"]
    auth_user = result["user"]

    user_mgr._reset_failed_attempts(auth_user.user_id)

    # Issue refresh token
    refresh_token = _generate_refresh_token()
    _store_refresh_token(
        token=refresh_token,
        user_id=user.user_id,
        role=user.role,
    )

    logger.info(
        "[AUTH-ROUTES] Login successful  user_id=%s  username=%s  request_id=%s",
        auth_user.user_id,
        auth_user.username,
        request_id,
    )

    return LoginResponse(
        access_token=token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in=config.server.jwt_expire_minutes * 60,
        user=_user_to_response(auth_user),
    )


# ---------------------------------------------------------------------------
# POST /api/v1/auth/refresh — Refresh access token
# ---------------------------------------------------------------------------

@router.post(
    "/refresh",
    response_model=SetupResponse,
    summary="Refresh access token",
    description=(
        "Exchange a valid refresh token for a new access_token + refresh_token pair. "
        "The old refresh token is invalidated."
    ),
    responses={
        401: {"description": "Invalid or expired refresh token"},
    },
)
async def refresh_token(
    body: RefreshRequest,
    request: Request,
) -> SetupResponse:
    """Exchange a refresh token for new tokens.

    SRS Section 6.1 (Token lifecycle):
    - Invalidates the old refresh token
    - Returns a new access_token + refresh_token pair
    """
    from isocortex.auth import create_access_token, get_user_manager
    from isocortex.config import get_config

    request_id = _get_request_id(request)
    config = get_config()

    # Periodically clean up expired tokens
    _cleanup_expired_refresh_tokens()

    entry = _get_refresh_token(body.refresh_token)
    if entry is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": "INVALID_REFRESH_TOKEN",
                "code": 401,
                "detail": "Refresh token is invalid or has expired",
                "request_id": request_id,
            },
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Invalidate old token (rotation)
    _invalidate_refresh_token(body.refresh_token)

    user_id = entry["user_id"]
    role = entry["role"]

    # Verify the user still exists and is active
    user_mgr = get_user_manager()
    user = user_mgr.get_user(user_id)
    if user is None or not user.is_active:
        logger.warning(
            "[AUTH-ROUTES] Refresh token used for inactive/nonexistent user %s  request_id=%s",
            user_id,
            request_id,
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": "USER_INACTIVE",
                "code": 401,
                "detail": "User account is no longer active",
                "request_id": request_id,
            },
        )

    # Issue new tokens
    new_access_token = create_access_token(
        user_id=user_id,
        role=role,
        secret_key=config.server.jwt_secret_key,
        algorithm=config.server.jwt_algorithm,
        expires_minutes=config.server.jwt_expire_minutes,
    )
    new_refresh_token = _generate_refresh_token()
    _store_refresh_token(
        token=new_refresh_token,
        user_id=user_id,
        role=role,
    )

    logger.info(
        "[AUTH-ROUTES] Token refreshed  user_id=%s  request_id=%s",
        user_id,
        request_id,
    )

    return SetupResponse(
        access_token=new_access_token,
        refresh_token=new_refresh_token,
        token_type="bearer",
        expires_in=config.server.jwt_expire_minutes * 60,
    )


# ---------------------------------------------------------------------------
# POST /api/v1/auth/logout — Invalidate refresh token
# ---------------------------------------------------------------------------

@router.post(
    "/logout",
    summary="Invalidate refresh token",
    description="Invalidate the provided refresh token so it cannot be reused.",
    responses={
        401: {"description": "Invalid or expired refresh token"},
    },
)
async def logout(
    body: RefreshRequest,
    request: Request,
) -> dict[str, Any]:
    """Invalidate a refresh token.

    SRS Section 6.1 (Token lifecycle):
    - Removes the refresh token from the store
    - Always returns 200 even if token was not found
    """
    request_id = _get_request_id(request)

    invalidated = _invalidate_refresh_token(body.refresh_token)

    if invalidated:
        logger.info(
            "[AUTH-ROUTES] Refresh token invalidated  request_id=%s",
            request_id,
        )

    return {
        "message": "Logged out successfully",
        "request_id": request_id,
    }


# ---------------------------------------------------------------------------
# POST /api/v1/auth/users — Admin: create user
# ---------------------------------------------------------------------------

@router.post(
    "/users",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new user (admin only)",
    description="Admin-only endpoint to create a new user with specified role.",
    responses={
        403: {"description": "Admin access required"},
        409: {"description": "Username or email already exists"},
        422: {"description": "Validation error"},
    },
)
async def create_user(
    body: CreateUserRequest,
    request: Request,
    _admin: dict = Depends(require_admin),
) -> UserResponse:
    """Create a new user. Admin only.

    SRS Section 6.5:
    - Admin-only endpoint
    - Same password complexity requirements as setup
    - Returns 409 if username/email already exists
    """
    request_id = _get_request_id(request)

    from isocortex.auth import get_user_manager

    user_mgr = get_user_manager()

    try:
        user = user_mgr.create_user(
            username=body.username,
            email=body.email or "",
            password=body.password,
            role=body.role,
        )
    except ValueError as exc:
        logger.warning(
            "[AUTH-ROUTES] Create user failed: %s  request_id=%s",
            exc,
            request_id,
        )
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "error": "USER_EXISTS",
                "code": 409,
                "detail": str(exc),
                "request_id": request_id,
            },
        )

    logger.info(
        "[AUTH-ROUTES] User created by admin  user_id=%s  username=%s  role=%s  "
        "created_by=%s  request_id=%s",
        user.user_id,
        body.username,
        body.role,
        _admin.get("sub"),
        request_id,
    )

    return _user_to_response(user)


# ---------------------------------------------------------------------------
# GET /api/v1/auth/users — Admin: list users (paginated)
# ---------------------------------------------------------------------------

@router.get(
    "/users",
    response_model=dict[str, Any],
    summary="List all users (admin only)",
    description="Paginated list of all users. Admin only.",
    responses={
        403: {"description": "Admin access required"},
    },
)
async def list_users(
    request: Request,
    page: int = 1,
    page_size: int = 50,
    _admin: dict = Depends(require_admin),
) -> dict[str, Any]:
    """List all users with pagination. Admin only.

    SRS Section 6.5:
    - Offset-based pagination
    - Returns user metadata (never password hashes)
    """
    request_id = _get_request_id(request)

    from isocortex.auth import get_user_manager

    user_mgr = get_user_manager()

    # Clamp pagination params
    page = max(1, page)
    page_size = max(1, min(page_size, 100))
    offset = (page - 1) * page_size

    users = user_mgr.list_users(offset=offset, limit=page_size + 1)

    # Determine if there are more pages
    has_more = len(users) > page_size
    if has_more:
        users = users[:page_size]

    return {
        "request_id": request_id,
        "users": [_user_to_response(u) for u in users],
        "pagination": {
            "page": page,
            "page_size": page_size,
            "has_more": has_more,
        },
    }


# ---------------------------------------------------------------------------
# PUT /api/v1/auth/users/{user_id}/password — Change password
# ---------------------------------------------------------------------------

@router.put(
    "/users/{user_id}/password",
    summary="Change user password",
    description=(
        "Change password for a user. Users can change their own password; "
        "admins can change any user's password."
    ),
    responses={
        400: {"description": "Current password is incorrect"},
        403: {"description": "Cannot change another user's password"},
        404: {"description": "User not found"},
    },
)
async def change_password(
    user_id: str,
    body: ChangePasswordRequest,
    request: Request,
    current_user: dict = Depends(get_current_user),
) -> dict[str, Any]:
    """Change a user's password.

    SRS Section 6:
    - Users can change their own password (must provide current_password)
    - Admins can change any user's password (current_password optional)
    - Same password complexity rules apply
    """
    request_id = _get_request_id(request)

    from isocortex.auth import get_user_manager, verify_password

    user_mgr = get_user_manager()
    target_user = user_mgr.get_user(user_id)

    if target_user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": "USER_NOT_FOUND",
                "code": 404,
                "detail": f"User '{user_id}' not found",
                "request_id": request_id,
            },
        )

    # Authorization: user can change own password, admin can change any
    requester_id = current_user.get("sub")
    is_self = requester_id == user_id
    is_admin = current_user.get("role") == "admin"

    if not is_self and not is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "error": "FORBIDDEN",
                "code": 403,
                "detail": "You can only change your own password",
                "request_id": request_id,
            },
        )

    # If changing own password, verify current password
    if is_self:
        # Need to get the password hash from storage
        from isocortex.storage import get_database
        db = get_database()
        with db.transaction() as conn:
            row = conn.execute(
                "SELECT password_hash FROM users WHERE user_id = ?",
                (user_id,),
            ).fetchone()

        if row is None or not verify_password(
            body.current_password, row["password_hash"]
        ):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": "INVALID_PASSWORD",
                    "code": 400,
                    "detail": "Current password is incorrect",
                    "request_id": request_id,
                },
            )

    # Update password
    user_mgr.update_user(user_id, password=body.new_password)

    # Invalidate all refresh tokens for the user (security measure)
    conn = _get_refresh_db()
    try:
        conn.execute(
            "DELETE FROM refresh_tokens WHERE user_id = ?", (user_id,)
        )
        conn.commit()
    finally:
        conn.close()

    logger.info(
        "[AUTH-ROUTES] Password changed  user_id=%s  changed_by=%s  request_id=%s",
        user_id,
        requester_id,
        request_id,
    )

    return {
        "message": "Password updated successfully",
        "request_id": request_id,
    }


# ---------------------------------------------------------------------------
# PUT /api/v1/auth/users/{user_id}/role — Admin: change role
# ---------------------------------------------------------------------------

@router.put(
    "/users/{user_id}/role",
    response_model=UserResponse,
    summary="Change user role (admin only)",
    description="Admin-only endpoint to change a user's role.",
    responses={
        403: {"description": "Admin access required"},
        404: {"description": "User not found"},
    },
)
async def change_role(
    user_id: str,
    body: ChangeRoleRequest,
    request: Request,
    _admin: dict = Depends(require_admin),
) -> UserResponse:
    """Change a user's role. Admin only.

    SRS Section 6.5:
    - Admin-only endpoint
    - Cannot change own role (prevents lockout)
    """
    request_id = _get_request_id(request)

    from isocortex.auth import get_user_manager

    user_mgr = get_user_manager()

    # Prevent admin from changing own role
    if _admin.get("sub") == user_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "SELF_ROLE_CHANGE",
                "code": 400,
                "detail": "Cannot change your own role",
                "request_id": request_id,
            },
        )

    target = user_mgr.get_user(user_id)
    if target is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": "USER_NOT_FOUND",
                "code": 404,
                "detail": f"User '{user_id}' not found",
                "request_id": request_id,
            },
        )

    updated = user_mgr.update_user(user_id, role=body.role)
    if updated is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": "USER_NOT_FOUND",
                "code": 404,
                "detail": f"User '{user_id}' not found",
                "request_id": request_id,
            },
        )

    logger.info(
        "[AUTH-ROUTES] Role changed  user_id=%s  old_role=%s  new_role=%s  "
        "changed_by=%s  request_id=%s",
        user_id,
        target.role,
        body.role,
        _admin.get("sub"),
        request_id,
    )

    return _user_to_response(updated)


# ---------------------------------------------------------------------------
# DELETE /api/v1/auth/users/{user_id} — Admin: delete user
# ---------------------------------------------------------------------------

@router.delete(
    "/users/{user_id}",
    summary="Delete a user (admin only)",
    description="Soft-delete a user account. Admin only.",
    responses={
        403: {"description": "Admin access required"},
        404: {"description": "User not found"},
    },
)
async def delete_user(
    user_id: str,
    request: Request,
    _admin: dict = Depends(require_admin),
) -> dict[str, Any]:
    """Soft-delete a user. Admin only.

    SRS Section 6.5:
    - Admin-only endpoint
    - Cannot delete own account (prevents lockout)
    - Soft-deletes (sets is_active = 0)
    """
    request_id = _get_request_id(request)

    from isocortex.auth import get_user_manager

    # Prevent admin from deleting own account
    if _admin.get("sub") == user_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "SELF_DELETE",
                "code": 400,
                "detail": "Cannot delete your own account",
                "request_id": request_id,
            },
        )

    user_mgr = get_user_manager()
    deleted = user_mgr.delete_user(user_id)

    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": "USER_NOT_FOUND",
                "code": 404,
                "detail": f"User '{user_id}' not found or already deleted",
                "request_id": request_id,
            },
        )

    # Invalidate all refresh tokens for the user
    conn = _get_refresh_db()
    try:
        conn.execute(
            "DELETE FROM refresh_tokens WHERE user_id = ?", (user_id,)
        )
        conn.commit()
    finally:
        conn.close()

    logger.info(
        "[AUTH-ROUTES] User deleted  user_id=%s  deleted_by=%s  request_id=%s",
        user_id,
        _admin.get("sub"),
        request_id,
    )

    return {
        "message": f"User '{user_id}' deleted",
        "request_id": request_id,
    }


# ---------------------------------------------------------------------------
# POST /api/v1/auth/users/{user_id}/unlock — Admin: unlock locked account
# ---------------------------------------------------------------------------

@router.post(
    "/users/{user_id}/unlock",
    summary="Unlock a locked account (admin only)",
    description=(
        "Manually unlock a locked account. Resets failed login counter. "
        "SRS Section 6.3: Admins can manually unlock accounts."
    ),
    responses={
        403: {"description": "Admin access required"},
        404: {"description": "User not found or not locked"},
    },
)
async def unlock_user(
    user_id: str,
    request: Request,
    _admin: dict = Depends(require_admin),
) -> dict[str, Any]:
    """Unlock a locked account. Admin only.

    SRS Section 6.3:
    - Admin-only endpoint
    - Resets failed_login_attempts to 0
    - Clears locked_at timestamp
    """
    request_id = _get_request_id(request)

    from isocortex.auth import get_user_manager

    user_mgr = get_user_manager()
    unlocked = user_mgr.unlock_user(user_id)

    if not unlocked:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": "USER_NOT_LOCKED",
                "code": 404,
                "detail": f"User '{user_id}' is not locked",
                "request_id": request_id,
            },
        )

    logger.info(
        "[AUTH-ROUTES] Account unlocked  user_id=%s  unlocked_by=%s  request_id=%s",
        user_id,
        _admin.get("sub"),
        request_id,
    )

    return {
        "message": f"User '{user_id}' unlocked successfully",
        "request_id": request_id,
    }


# ---------------------------------------------------------------------------
# GET /api/v1/auth/me — Get current user profile
# ---------------------------------------------------------------------------

@router.get(
    "/me",
    response_model=UserProfileResponse,
    summary="Get current user profile",
    description="Return the authenticated user's profile information.",
    responses={
        401: {"description": "Authentication required"},
    },
)
async def get_me(
    request: Request,
    current_user: dict = Depends(get_current_user),
) -> UserProfileResponse:
    """Return the authenticated user's full profile.

    SRS Section 6:
    - Requires valid JWT
    - Returns user_id, username, email, role, created_at, etc.
    """
    request_id = _get_request_id(request)

    from isocortex.auth import get_user_manager

    user_mgr = get_user_manager()
    user = user_mgr.get_user(current_user.get("sub"))

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": "USER_NOT_FOUND",
                "code": 401,
                "detail": "Authenticated user not found in database",
                "request_id": request_id,
            },
        )

    # Count API keys for this user (best-effort)
    api_keys_count = 0
    try:
        from isocortex.storage import get_database
        db = get_database()
        with db.transaction() as conn:
            row = conn.execute(
                "SELECT COUNT(*) as cnt FROM api_keys WHERE user_id = ? AND is_active = 1",
                (user.user_id,),
            ).fetchone()
            if row:
                api_keys_count = row["cnt"]
    except Exception:
        pass

    return UserProfileResponse(
        user_id=user.user_id,
        username=user.username,
        email=user.email,
        role=user.role,
        is_active=user.is_active,
        created_at=user.created_at,
        last_login=user.last_login,
        api_keys_count=api_keys_count,
    )
