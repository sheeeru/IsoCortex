"""
IsoCortex — Auth Schema Models
================================

Pydantic models for authentication and user management endpoints.

SRS References:
  - Section 6: Authentication and User Management
  - SRS Section 6.2: Password Requirements (min 12 chars, complexity)
  - SRS Section 6.3: Account Lockout (5 attempts, 15 min)
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, EmailStr, Field, field_validator


# =============================================================================
# Enums
# =============================================================================

class UserRole(str):
    """User roles (SRS Section 6.5)."""
    ADMIN = "admin"
    USER = "user"


# =============================================================================
# Login / Setup
# =============================================================================

class SetupRequest(BaseModel):
    """First-run admin setup (SRS Section 6.4: POST /api/v1/auth/setup)."""
    username: str = Field(
        ..., min_length=3, max_length=50,
        description="Admin username",
        pattern=r"^[a-zA-Z0-9_-]+$",
    )
    password: str = Field(
        ..., min_length=12, max_length=128,
        description="Admin password (min 12 chars, complexity required)",
    )
    email: Optional[str] = Field(default=None, max_length=255)

    @field_validator("password")
    @classmethod
    def validate_password_complexity(cls, v: str) -> str:
        if not any(c.isupper() for c in v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not any(c.islower() for c in v):
            raise ValueError("Password must contain at least one lowercase letter")
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain at least one digit")
        if not any(c in "!@#$%^&*()_+-=[]{}|;':\",./<>?" for c in v):
            raise ValueError("Password must contain at least one special character")
        return v


class SetupResponse(BaseModel):
    """Setup response with tokens."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int = Field(description="Access token TTL in seconds")


class LoginRequest(BaseModel):
    """Login request (SRS Section 6.1: POST /api/v1/auth/login)."""
    username: str = Field(..., min_length=1, description="Username or email")
    password: str = Field(..., min_length=1, description="Password")


class LoginResponse(BaseModel):
    """Login response with tokens."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int = Field(description="Access token TTL in seconds")
    user: "UserResponse"


class RefreshRequest(BaseModel):
    """Token refresh request."""
    refresh_token: str = Field(..., description="Refresh token")


# =============================================================================
# User Management
# =============================================================================

class CreateUserRequest(BaseModel):
    """Create user request (admin only, SRS Section 6.5).

    Same password complexity as setup.
    """
    username: str = Field(
        ..., min_length=3, max_length=50,
        pattern=r"^[a-zA-Z0-9_-]+$",
    )
    password: str = Field(..., min_length=12, max_length=128)
    role: str = Field(default="user", pattern=r"^(admin|user)$")
    email: Optional[str] = Field(default=None, max_length=255)

    @field_validator("password")
    @classmethod
    def validate_password_complexity(cls, v: str) -> str:
        if not any(c.isupper() for c in v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not any(c.islower() for c in v):
            raise ValueError("Password must contain at least one lowercase letter")
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain at least one digit")
        if not any(c in "!@#$%^&*()_+-=[]{}|;':\",./<>?" for c in v):
            raise ValueError("Password must contain at least one special character")
        return v


class ChangePasswordRequest(BaseModel):
    """Change password request."""
    current_password: str = Field(..., min_length=1)
    new_password: str = Field(..., min_length=12, max_length=128)

    @field_validator("new_password")
    @classmethod
    def validate_password_complexity(cls, v: str) -> str:
        if not any(c.isupper() for c in v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not any(c.islower() for c in v):
            raise ValueError("Password must contain at least one lowercase letter")
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain at least one digit")
        if not any(c in "!@#$%^&*()_+-=[]{}|;':\",./<>?" for c in v):
            raise ValueError("Password must contain at least one special character")
        return v


class UserResponse(BaseModel):
    """User representation in API responses."""
    user_id: str
    username: str
    email: Optional[str] = None
    role: str
    is_active: bool = True
    created_at: str = ""
    last_login: Optional[str] = None


class UserProfileResponse(BaseModel):
    """Current user profile (SRS: GET /api/v1/auth/me)."""
    user_id: str
    username: str
    email: Optional[str] = None
    role: str
    is_active: bool = True
    created_at: str = ""
    last_login: Optional[str] = None
    api_keys_count: int = 0


class ChangeRoleRequest(BaseModel):
    """Change user role request (admin only)."""
    role: str = Field(..., pattern=r"^(admin|user)$")
