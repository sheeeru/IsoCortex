"""
IsoCortex — Admin Schema Models
=================================

Pydantic models for admin-only endpoints (rate limits, system management).

SRS References:
  - Section 11: Rate Limiting
  - NFR-07: SQLite-backed sliding window rate limiting
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class RateLimitEntry(BaseModel):
    """Single rate limit entry."""
    key: str
    endpoint: str
    limit: int
    remaining: int
    reset_at: str = ""


class RateLimitResponse(BaseModel):
    """Rate limit status for a specific key."""
    key: str
    endpoint: str
    limit: int
    remaining: int
    reset_at: str = ""


class RateLimitListResponse(BaseModel):
    """List of all rate limits."""
    limits: list[RateLimitEntry] = Field(default_factory=list)


class HealthResponse(BaseModel):
    """System health response."""
    status: str = Field(default="healthy")
    version: str = ""
    uptime_seconds: float = 0.0
    total_indexes: int = 0
    total_vectors: int = 0
    memory_usage_mb: float = 0.0
    active_jobs: int = 0


class StatsResponse(BaseModel):
    """System statistics."""
    total_queries: int = 0
    avg_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    top_queries: list[dict[str, Any]] = Field(default_factory=list)
    queries_per_minute: float = 0.0
    unique_queries: int = 0
    cache_hit_rate: float = 0.0
    indexes: list[dict[str, Any]] = Field(default_factory=list)
