"""
IsoCortex api/middleware — Middleware Package
=============================================

Middleware pipeline for the FastAPI application:
  - Authentication (JWT verification)
  - Rate limiting (SQLite-backed sliding window)
  - Request ID generation (UUID v4)
  - Error handling (standardized error envelope)
  - Logging (structured request/response logging)

Author : Shaheer Qureshi
Project: IsoCortex
"""

from isocortex.api.middleware.auth import get_current_user, require_admin
from isocortex.api.middleware.rate_limit import RateLimitMiddleware
from isocortex.api.middleware.request_id import RequestIDMiddleware
from isocortex.api.middleware.error_handler import register_error_handlers

__all__ = [
    "get_current_user",
    "require_admin",
    "RateLimitMiddleware",
    "RequestIDMiddleware",
    "register_error_handlers",
]
