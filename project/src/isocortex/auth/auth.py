"""
IsoCortex — Authentication & User Management
==============================================

JWT-based authentication with bcrypt password hashing and
role-based access control (admin/user roles).

SRS References: Section 6 (Authentication & User Management)

Author : Shaheer Qureshi
Project: IsoCortex
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import secrets
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Password Hashing (bcrypt)
# =============================================================================

def hash_password(password: str) -> str:
    """
    Hash a password using bcrypt.

    Parameters
    ----------
    password : str — Plain-text password.

    Returns
    -------
    str — bcrypt hash string.
    """
    import bcrypt as _bcrypt
    password_bytes = password.encode("utf-8")
    salt = _bcrypt.gensalt(rounds=12)
    hashed = _bcrypt.hashpw(password_bytes, salt)
    return hashed.decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    """
    Verify a password against its bcrypt hash.

    Parameters
    ----------
    password      : str — Plain-text password to verify.
    password_hash : str — Stored bcrypt hash.

    Returns
    -------
    bool — True if password matches.
    """
    import bcrypt as _bcrypt
    try:
        return _bcrypt.checkpw(
            password.encode("utf-8"),
            password_hash.encode("utf-8"),
        )
    except Exception:
        return False


# =============================================================================
# JWT Token Management
# =============================================================================

def create_access_token(
    user_id: str,
    role: str,
    secret_key: str,
    algorithm: str = "HS256",
    expires_minutes: int = 1440,
) -> str:
    """
    Create a JWT access token.

    SRS Section 6: JWT-based authentication with configurable expiry.

    Parameters
    ----------
    user_id         : str — User's unique ID.
    role            : str — User's role (admin/user).
    secret_key      : str — HMAC secret key.
    algorithm       : str — Signing algorithm.
    expires_minutes : int — Token expiry in minutes.

    Returns
    -------
    str — Encoded JWT token string.
    """
    import base64
    import json

    now = datetime.now(timezone.utc)
    expire = now + timedelta(minutes=expires_minutes)

    # Header
    header = {
        "alg": algorithm,
        "typ": "JWT",
    }

    # Payload
    payload = {
        "sub": user_id,
        "role": role,
        "iat": int(now.timestamp()),
        "exp": int(expire.timestamp()),
        "jti": str(uuid.uuid4()),
    }

    def b64_encode(data: dict) -> str:
        return base64.urlsafe_b64encode(
            json.dumps(data, separators=(",", ":")).encode("utf-8")
        ).rstrip(b"=").decode("utf-8")

    header_b64 = b64_encode(header)
    payload_b64 = b64_encode(payload)

    # Signature
    signing_input = f"{header_b64}.{payload_b64}"
    signature = hmac.new(
        secret_key.encode("utf-8"),
        signing_input.encode("utf-8"),
        hashlib.sha256,
    ).digest()

    signature_b64 = base64.urlsafe_b64encode(signature).rstrip(b"=").decode("utf-8")

    token = f"{header_b64}.{payload_b64}.{signature_b64}"
    logger.debug("[AUTH] Token created  user_id=%s  role=%s  expires=%s",
                 user_id, role, expire.isoformat())
    return token


def decode_access_token(
    token: str,
    secret_key: str,
    algorithm: str = "HS256",
) -> Optional[dict[str, Any]]:
    """
    Decode and validate a JWT access token.

    Parameters
    ----------
    token      : str — Encoded JWT token.
    secret_key : str — HMAC secret key.
    algorithm  : str — Expected signing algorithm.

    Returns
    -------
    dict | None — Token payload if valid, None otherwise.
    """
    import base64
    import json

    try:
        parts = token.split(".")
        if len(parts) != 3:
            logger.warning("[AUTH] Token has invalid format (not 3 parts)")
            return None

        header_b64, payload_b64, signature_b64 = parts

        # Add padding
        header_b64_padded = header_b64 + "=" * (4 - len(header_b64) % 4)
        payload_b64_padded = payload_b64 + "=" * (4 - len(payload_b64) % 4)
        signature_b64_padded = signature_b64 + "=" * (4 - len(signature_b64) % 4)

        # Verify signature
        signing_input = f"{header_b64}.{payload_b64}"
        expected_sig = hmac.new(
            secret_key.encode("utf-8"),
            signing_input.encode("utf-8"),
            hashlib.sha256,
        ).digest()

        actual_sig = base64.urlsafe_b64decode(signature_b64_padded)

        if not hmac.compare_digest(expected_sig, actual_sig):
            logger.warning("[AUTH] Token signature verification failed")
            return None

        # Decode payload
        payload = json.loads(base64.urlsafe_b64decode(payload_b64_padded))

        # Check expiry
        exp = payload.get("exp")
        if exp is None:
            logger.warning("[AUTH] Token missing exp claim")
            return None

        now = datetime.now(timezone.utc).timestamp()
        if now > exp:
            logger.warning("[AUTH] Token expired  exp=%s  now=%s", exp, now)
            return None

        return payload

    except Exception as exc:
        logger.warning("[AUTH] Token decode failed: %s", exc)
        return None


# =============================================================================
# API Key Management
# =============================================================================

def generate_api_key() -> tuple[str, str]:
    """
    Generate a new API key and its hash.

    Returns
    -------
    (raw_key, key_hash)
      raw_key  : str — The plaintext API key (shown once to user).
      key_hash : str — SHA-256 hash for storage.
    """
    raw_key = f"iso_{secrets.token_urlsafe(32)}"
    key_hash = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()
    return raw_key, key_hash


def hash_api_key(api_key: str) -> str:
    """Hash an API key for database lookup."""
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()


# =============================================================================
# User Manager
# =============================================================================

@dataclass
class User:
    """User data object."""
    user_id: str
    username: str
    email: str
    role: str
    is_active: bool
    created_at: str
    updated_at: str
    last_login: Optional[str] = None
    locked_until: Optional[str] = None
    failed_login_attempts: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "username": self.username,
            "email": self.email,
            "role": self.role,
            "is_active": self.is_active,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "last_login": self.last_login,
            "locked_until": self.locked_until,
        }


class UserManager:
    """
    User management with SQLite persistence.

    SRS Section 6: Role-based user administration (admin/user roles).

    Operations:
      - create_user()    : Register a new user
      - authenticate()   : Verify credentials, return token
      - get_user()       : Retrieve user by ID
      - list_users()     : List all users (admin only)
      - update_user()    : Modify user fields
      - delete_user()    : Soft-delete a user
      - create_api_key() : Generate API key for user
      - validate_api_key(): Verify API key from request
    """

    def __init__(self, db: "Database") -> None:
        self._db = db

    def create_user(
        self,
        username: str,
        email: str,
        password: str,
        role: str = "user",
    ) -> User:
        """
        Create a new user.

        Parameters
        ----------
        username : str — Unique username.
        email    : str — Unique email.
        password : str — Plain-text password (will be hashed).
        role     : str — "admin" or "user".

        Returns
        -------
        User — The created user object.

        Raises
        ------
        ValueError — If username or email already exists.
        """
        user_id = str(uuid.uuid4())
        pwd_hash = hash_password(password)

        try:
            with self._db.transaction() as conn:
                conn.execute(
                    """INSERT INTO users (user_id, username, email, password_hash, role)
                       VALUES (?, ?, ?, ?, ?)""",
                    (user_id, username, email, pwd_hash, role),
                )
        except Exception as exc:
            if "UNIQUE constraint" in str(exc):
                raise ValueError(
                    f"User with username '{username}' or email '{email}' already exists"
                ) from exc
            raise

        logger.info("[AUTH] User created  user_id=%s  username=%s  role=%s", user_id, username, role)

        return User(
            user_id=user_id,
            username=username,
            email=email,
            role=role,
            is_active=True,
            created_at=datetime.now(timezone.utc).isoformat(),
            updated_at=datetime.now(timezone.utc).isoformat(),
        )

    def authenticate(
        self,
        username_or_email: str,
        password: str,
        secret_key: str,
        algorithm: str = "HS256",
        expire_minutes: int = 1440,
    ) -> Optional[dict[str, Any]]:
        """
        Authenticate a user and return a JWT token.

        Parameters
        ----------
        username_or_email : str — Username or email.
        password           : str — Plain-text password.
        secret_key         : str — JWT signing key.
        algorithm          : str — JWT algorithm.
        expire_minutes     : int — Token expiry.

        Returns
        -------
        dict | None — {"token": str, "user": User} or None on failure.
        """
        with self._db.transaction() as conn:
            row = conn.execute(
                """SELECT user_id, username, email, password_hash, role, is_active
                   FROM users
                   WHERE (username = ? OR email = ?) AND is_active = 1""",
                (username_or_email, username_or_email),
            ).fetchone()

        if row is None:
            logger.debug("[AUTH] Authentication failed: user not found")
            return None

        if not verify_password(password, row["password_hash"]):
            logger.debug("[AUTH] Authentication failed: wrong password for %s", username_or_email)
            return None

        # Update last login
        now = datetime.now(timezone.utc).isoformat()
        with self._db.transaction() as conn:
            conn.execute(
                "UPDATE users SET last_login = ?, updated_at = ? WHERE user_id = ?",
                (now, now, row["user_id"]),
            )

        # Generate token
        token = create_access_token(
            user_id=row["user_id"],
            role=row["role"],
            secret_key=secret_key,
            algorithm=algorithm,
            expires_minutes=expire_minutes,
        )

        user = User(
            user_id=row["user_id"],
            username=row["username"],
            email=row["email"],
            role=row["role"],
            is_active=bool(row["is_active"]),
            created_at="",
            updated_at=now,
            last_login=now,
        )

        logger.info("[AUTH] User authenticated  user_id=%s  username=%s",
                     row["user_id"], row["username"])

        return {"token": token, "user": user}

    def get_user(self, user_id: str) -> Optional[User]:
        """Retrieve a user by ID."""
        with self._db.transaction() as conn:
            row = conn.execute(
                "SELECT * FROM users WHERE user_id = ?", (user_id,)
            ).fetchone()

        if row is None:
            return None

        return User(
            user_id=row["user_id"],
            username=row["username"],
            email=row["email"],
            role=row["role"],
            is_active=bool(row["is_active"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            last_login=row["last_login"],
            locked_until=row["locked_at"],
            failed_login_attempts=row["failed_login_attempts"],
        )

    def get_user_by_username(self, username: str) -> Optional[User]:
        """Retrieve a user by username.

        SRS Section 6.3: Account lockout — needed for lockout checks in auth middleware.
        """
        with self._db.transaction() as conn:
            row = conn.execute(
                "SELECT * FROM users WHERE username = ?", (username,)
            ).fetchone()

        if row is None:
            return None

        return User(
            user_id=row["user_id"],
            username=row["username"],
            email=row["email"],
            role=row["role"],
            is_active=bool(row["is_active"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            last_login=row["last_login"],
            locked_until=row["locked_at"],
            failed_login_attempts=row["failed_login_attempts"],
        )

    def list_users(
        self,
        offset: int = 0,
        limit: int = 50,
    ) -> list[User]:
        """List all users with pagination."""
        with self._db.transaction() as conn:
            rows = conn.execute(
                "SELECT * FROM users ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()

        return [
            User(
                user_id=r["user_id"],
                username=r["username"],
                email=r["email"],
                role=r["role"],
                is_active=bool(r["is_active"]),
                created_at=r["created_at"],
                updated_at=r["updated_at"],
                last_login=r["last_login"],
            )
            for r in rows
        ]

    def update_user(
        self,
        user_id: str,
        **kwargs: Any,
    ) -> Optional[User]:
        """
        Update user fields.

        Allowed fields: username, email, role, is_active, password.
        """
        allowed = {"username", "email", "role", "is_active", "password"}
        updates = {k: v for k, v in kwargs.items() if k in allowed}

        if not updates:
            return self.get_user(user_id)

        if "password" in updates:
            updates["password_hash"] = hash_password(updates.pop("password"))
            updates.pop("password", None)

        if "is_active" in updates:
            updates["is_active"] = 1 if updates["is_active"] else 0

        updates["updated_at"] = datetime.now(timezone.utc).isoformat()

        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [user_id]

        try:
            with self._db.transaction() as conn:
                conn.execute(
                    f"UPDATE users SET {set_clause} WHERE user_id = ?",
                    values,
                )
        except Exception as exc:
            if "UNIQUE constraint" in str(exc):
                raise ValueError("Username or email already in use") from exc
            raise

        logger.info("[AUTH] User updated  user_id=%s  fields=%s", user_id, list(updates.keys()))
        return self.get_user(user_id)

    def delete_user(self, user_id: str) -> bool:
        """Soft-delete a user by setting is_active = 0."""
        with self._db.transaction() as conn:
            cursor = conn.execute(
                "UPDATE users SET is_active = 0 WHERE user_id = ? AND is_active = 1",
                (user_id,),
            )
            return cursor.rowcount > 0

    # -----------------------------------------------------------------
    # Account lockout helpers (SRS Section 6.3)
    # -----------------------------------------------------------------

    def _increment_failed_attempt(self, user_id: str) -> None:
        """Increment the failed login counter for a user."""
        now = datetime.now(timezone.utc).isoformat()
        with self._db.transaction() as conn:
            conn.execute(
                "UPDATE users SET failed_login_attempts = failed_login_attempts + 1, "
                "updated_at = ? WHERE user_id = ?",
                (now, user_id),
            )

    def _reset_failed_attempts(self, user_id: str) -> None:
        """Reset failed login counter and clear lockout."""
        now = datetime.now(timezone.utc).isoformat()
        with self._db.transaction() as conn:
            conn.execute(
                "UPDATE users SET failed_login_attempts = 0, locked_at = NULL, "
                "updated_at = ? WHERE user_id = ?",
                (now, user_id),
            )

    def _lock_account(self, user_id: str, locked_until: str) -> None:
        """Lock an account until the given ISO timestamp."""
        with self._db.transaction() as conn:
            conn.execute(
                "UPDATE users SET locked_at = ?, updated_at = ? WHERE user_id = ?",
                (locked_until, datetime.now(timezone.utc).isoformat(), user_id),
            )
        logger.warning("[AUTH] Account locked  user_id=%s  until=%s", user_id, locked_until)

    def unlock_user(self, user_id: str) -> bool:
        """Manually unlock a locked account. Returns True if user was locked.

        SRS Section 6.3: Admins can manually unlock accounts.
        """
        with self._db.transaction() as conn:
            cursor = conn.execute(
                "UPDATE users SET failed_login_attempts = 0, locked_at = NULL, "
                "updated_at = ? WHERE user_id = ? AND locked_at IS NOT NULL",
                (datetime.now(timezone.utc).isoformat(), user_id),
            )
            if cursor.rowcount > 0:
                logger.info("[AUTH] Account unlocked by admin  user_id=%s", user_id)
                return True
            return False

    def create_api_key(
        self,
        user_id: str,
        name: str = "",
        expires_hours: int = 0,
    ) -> tuple[str, str]:
        """
        Generate an API key for a user.

        Returns
        -------
        (raw_key, key_id)
          raw_key : str — Plaintext key (show once).
          key_id  : str — Key ID for reference.
        """
        raw_key, key_hash = generate_api_key()
        key_id = str(uuid.uuid4())

        expires_at = None
        if expires_hours > 0:
            expires_at = (
                datetime.now(timezone.utc) + timedelta(hours=expires_hours)
            ).isoformat()

        with self._db.transaction() as conn:
            conn.execute(
                """INSERT INTO api_keys (key_id, user_id, key_hash, name, expires_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (key_id, user_id, key_hash, name, expires_at),
            )

        logger.info("[AUTH] API key created  key_id=%s  user_id=%s  name=%s", key_id, user_id, name)
        return raw_key, key_id

    def validate_api_key(self, api_key: str) -> Optional[dict[str, Any]]:
        """
        Validate an API key and return user info.

        Returns
        -------
        dict | None — {"user_id": str, "role": str, "key_id": str} or None.
        """
        key_hash = hash_api_key(api_key)

        with self._db.transaction() as conn:
            row = conn.execute(
                """SELECT ak.key_id, ak.user_id, ak.expires_at, u.role
                   FROM api_keys ak
                   JOIN users u ON ak.user_id = u.user_id
                   WHERE ak.key_hash = ? AND ak.is_active = 1 AND u.is_active = 1""",
                (key_hash,),
            ).fetchone()

        if row is None:
            return None

        # Check expiry
        if row["expires_at"]:
            expires = datetime.fromisoformat(row["expires_at"])
            if datetime.now(timezone.utc) > expires:
                logger.debug("[AUTH] API key expired  key_id=%s", row["key_id"])
                return None

        return {
            "user_id": row["user_id"],
            "role": row["role"],
            "key_id": row["key_id"],
        }


# =============================================================================
# Singleton
# =============================================================================

_global_user_manager: Optional[UserManager] = None


def get_user_manager() -> UserManager:
    """Return the global UserManager singleton."""
    global _global_user_manager
    if _global_user_manager is None:
        from isocortex.storage import get_database
        _global_user_manager = UserManager(get_database())
    return _global_user_manager
