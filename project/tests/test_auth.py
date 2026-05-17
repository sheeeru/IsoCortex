"""
IsoCortex — Auth Tests
=======================
Tests for password hashing, JWT tokens, and UserManager.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest

from isocortex.auth.auth import (
    User,
    UserManager,
    create_access_token,
    decode_access_token,
    generate_api_key,
    hash_api_key,
    hash_password,
    verify_password,
)
from isocortex.storage.database import Database


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture()
def db(tmp_path: Path) -> Database:
    """Create a Database with initialized schema."""
    database = Database(db_path=tmp_path / "auth_test.db")
    database.initialize()
    return database


@pytest.fixture()
def user_manager(db: Database) -> UserManager:
    """Create a UserManager."""
    return UserManager(db)


@pytest.fixture()
def test_secret() -> str:
    return "test-secret-key-for-jwt"


# =============================================================================
# Password Hashing Tests
# =============================================================================

class TestPasswordHashing:

    def test_hash_and_verify_password(self):
        """Hash a password, verify correct/incorrect."""
        # Skip if bcrypt not installed
        pytest.importorskip("bcrypt")

        hashed = verify_password.__module__ and True  # just check import
        password = "secure_password_123"
        hashed = hash_password(password)
        assert hashed != password
        assert verify_password(password, hashed) is True
        assert verify_password("wrong_password", hashed) is False


# =============================================================================
# JWT Token Tests
# =============================================================================

class TestJWT:

    def test_create_access_token(self, test_secret: str):
        """Create JWT, verify claims."""
        token = create_access_token(
            user_id="user-123",
            role="admin",
            secret_key=test_secret,
            expires_minutes=60,
        )
        assert isinstance(token, str)
        parts = token.split(".")
        assert len(parts) == 3

    def test_decode_access_token_valid(self, test_secret: str):
        """Decode a valid token and verify claims."""
        token = create_access_token(
            user_id="user-456",
            role="user",
            secret_key=test_secret,
            expires_minutes=60,
        )
        payload = decode_access_token(token, test_secret)
        assert payload is not None
        assert payload["sub"] == "user-456"
        assert payload["role"] == "user"
        assert "exp" in payload
        assert "iat" in payload
        assert "jti" in payload

    def test_decode_access_token_expired(self, test_secret: str):
        """Expired token returns None."""
        token = create_access_token(
            user_id="user-789",
            role="user",
            secret_key=test_secret,
            expires_minutes=0,  # expires immediately
        )
        # Wait a moment for the token to expire
        time.sleep(1.5)
        payload = decode_access_token(token, test_secret)
        assert payload is None

    def test_decode_access_token_wrong_secret(self, test_secret: str):
        """Wrong secret returns None."""
        token = create_access_token(
            user_id="user-999",
            role="user",
            secret_key=test_secret,
        )
        payload = decode_access_token(token, "wrong-secret")
        assert payload is None


# =============================================================================
# UserManager Tests
# =============================================================================

class TestUserManager:

    def test_user_manager_create_user(self, user_manager: UserManager):
        """Create user, verify stored."""
        pytest.importorskip("bcrypt")

        user = user_manager.create_user(
            username="testuser",
            email="test@example.com",
            password="password123",
            role="user",
        )
        assert user.username == "testuser"
        assert user.email == "test@example.com"
        assert user.role == "user"
        assert user.is_active is True

    def test_user_manager_create_duplicate(self, user_manager: UserManager):
        """Duplicate user raises ValueError."""
        pytest.importorskip("bcrypt")

        user_manager.create_user("dup", "dup@test.com", "pass")
        with pytest.raises(ValueError, match="already exists"):
            user_manager.create_user("dup", "other@test.com", "pass")

    def test_user_manager_list_users(self, user_manager: UserManager):
        """List with pagination."""
        pytest.importorskip("bcrypt")

        user_manager.create_user("user1", "u1@test.com", "pass1")
        user_manager.create_user("user2", "u2@test.com", "pass2")
        users = user_manager.list_users()
        assert len(users) >= 2

    def test_user_manager_delete_user(self, user_manager: UserManager):
        """Soft delete."""
        pytest.importorskip("bcrypt")

        user = user_manager.create_user("delme", "del@test.com", "pass")
        assert user.is_active is True

        result = user_manager.delete_user(user.user_id)
        assert result is True

        # After delete, the user should be inactive
        updated = user_manager.get_user(user.user_id)
        assert updated is None or updated.is_active is False

    def test_user_manager_get_user_not_found(self, user_manager: UserManager):
        """Return None for non-existent user."""
        assert user_manager.get_user("nonexistent-id") is None


# =============================================================================
# API Key Tests
# =============================================================================

class TestAPIKey:

    def test_generate_api_key(self):
        """Generate API key and verify format."""
        raw_key, key_hash = generate_api_key()
        assert raw_key.startswith("iso_")
        assert len(key_hash) == 64  # SHA-256 hex
        assert hash_api_key(raw_key) == key_hash
