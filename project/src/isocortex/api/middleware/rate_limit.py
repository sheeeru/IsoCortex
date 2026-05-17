"""
IsoCortex — Rate Limiting Middleware
======================================

SQLite-backed sliding window rate limiting per user and endpoint.

SRS References:
  - Section 11: Rate Limiting
  - NFR-07: SQLite-backed sliding window, 100 req/min default
  - SRS 11.1: Response headers (X-RateLimit-Limit, X-RateLimit-Remaining, X-RateLimit-Reset)

Author : Shaheer Qureshi
Project: IsoCortex
"""

from __future__ import annotations

import logging
import time

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger(__name__)

# Paths excluded from rate limiting
EXEMPT_PATHS = {"/api/v1/auth/login", "/api/v1/auth/setup", "/health", "/docs", "/redoc", "/openapi.json"}


class RateLimitMiddleware(BaseHTTPMiddleware):
    """SQLite-backed sliding window rate limiter.

    SRS Section 11: Rate Limiting Design
    - Default: 100 requests per minute per user
    - Search-specific: 60 per minute
    - Write endpoints: 30 per minute
    - Response headers: X-RateLimit-Limit, X-RateLimit-Remaining, X-RateLimit-Reset
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        path = request.url.path

        # Skip rate limiting for exempt paths
        if path in EXEMPT_PATHS or not path.startswith("/api/v1/"):
            response = await call_next(request)
            return response

        # Determine rate limit key (user_id from JWT or IP fallback)
        key = self._get_rate_limit_key(request)

        # Determine limit based on endpoint type
        limit, window = self._get_limit_for_path(path, request.method)

        # Check rate limit
        from isocortex.storage import get_rate_limiter
        rate_limiter = get_rate_limiter()

        allowed, remaining, reset_at = rate_limiter.is_allowed(
            key_hash=key,
            endpoint=path,
            limit=limit,
            window_seconds=window,
        )

        if not allowed:
            logger.warning(
                "[RATE-LIMIT] Blocked key=%s endpoint=%s limit=%d",
                key, path, limit,
            )
            response = Response(
                content='{"error":"RATE_LIMITED","code":429,"detail":"Too many requests"}',
                status_code=429,
                media_type="application/json",
            )
            response.headers["X-RateLimit-Limit"] = str(limit)
            response.headers["X-RateLimit-Remaining"] = "0"
            response.headers["X-RateLimit-Reset"] = str(int(reset_at))
            response.headers["Retry-After"] = str(max(1, int(reset_at) - int(time.time())))
            return response

        # Process request
        response = await call_next(request)

        # Add rate limit headers
        response.headers["X-RateLimit-Limit"] = str(limit)
        response.headers["X-RateLimit-Remaining"] = str(max(0, remaining))
        response.headers["X-RateLimit-Reset"] = str(int(reset_at))

        return response

    @staticmethod
    def _get_rate_limit_key(request: Request) -> str:
        """Extract rate limit key from JWT payload or fallback to client IP."""
        # Try to get user_id from JWT (already decoded in auth middleware)
        user = getattr(request.state, "user", None)
        if user and isinstance(user, dict):
            return user.get("sub", "anonymous")

        # Fallback to IP address
        client_host = request.client.host if request.client else "unknown"
        return f"ip:{client_host}"

    def _get_limit_for_path(self, path: str, method: str) -> tuple[int, int]:
        """Determine rate limit based on endpoint type.

        SRS Section 11.2 Configuration:
        - Search endpoints: 60/min
        - Write endpoints: 30/min
        - Default: 100/min
        """
        if "/search" in path:
            return 60, 60
        if method in ("POST", "PUT", "DELETE"):
            return 30, 60
        return 100, 60
