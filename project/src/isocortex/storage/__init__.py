"""
IsoCortex — Storage Layer
==========================

Public API:
  - get_database()    → Global Database singleton
  - get_analytics()   → Global AnalyticsEngine singleton
  - get_rate_limiter() → Global RateLimiter singleton
  - close_database()  → Close all connections
  - Database          → SQLite manager with WAL mode
  - AnalyticsEngine   → Usage tracking and metrics
  - RateLimiter       → Sliding window rate limiting
"""

from isocortex.storage.database import (
    AnalyticsEngine,
    Database,
    RateLimiter,
    close_database,
    get_analytics,
    get_database,
    get_rate_limiter,
)

__all__ = [
    "AnalyticsEngine",
    "Database",
    "RateLimiter",
    "close_database",
    "get_analytics",
    "get_database",
    "get_rate_limiter",
]
