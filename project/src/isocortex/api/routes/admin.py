"""
IsoCortex — Admin Route Handlers
=================================

FastAPI route handlers for system administration and monitoring.

Provides endpoints for rate limit inspection, system statistics,
and health checks. Health check does not require authentication.

SRS References:
  - Section 11:   Rate Limiting (view status)
  - NFR-07:      SQLite-backed sliding window rate limiting
  - Section 8:    System statistics and monitoring

Author : Shaheer Qureshi
Project: IsoCortex
"""

from __future__ import annotations

import logging
import time
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status

from isocortex.api.middleware.auth import require_admin
from isocortex.api.schemas.admin import (
    HealthResponse,
    RateLimitEntry,
    RateLimitListResponse,
    StatsResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Admin"])

# ---------------------------------------------------------------------------
# Application start time (set on module load)
# ---------------------------------------------------------------------------
_start_time: float = time.time()

# Version info (injected at build time or set here)
APP_VERSION: str = "0.1.0"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_request_id(request: Request) -> str:
    """Extract request_id from request state (set by RequestIDMiddleware)."""
    return getattr(request.state, "request_id", "unknown")


def _get_uptime_seconds() -> float:
    """Return application uptime in seconds."""
    return round(time.time() - _start_time, 1)


# ---------------------------------------------------------------------------
# GET /api/v1/admin/rate-limits — View rate limit status
# ---------------------------------------------------------------------------

@router.get(
    "/rate-limits",
    response_model=RateLimitListResponse,
    summary="View rate limit status",
    description=(
        "Return current rate limit counters for all tracked keys/endpoints. "
        "Admin only. SRS Section 11."
    ),
    responses={
        403: {"description": "Admin access required"},
    },
)
async def get_rate_limits(
    request: Request,
    _admin: dict = Depends(require_admin),
) -> RateLimitListResponse:
    """View current rate limit status across all users/endpoints.

    SRS Section 11: Rate Limiting
    - Shows all active rate limit entries
    - Includes remaining count and reset time
    """
    request_id = _get_request_id(request)

    limits: list[RateLimitEntry] = []

    try:
        from isocortex.storage import get_rate_limiter
        rate_limiter = get_rate_limiter()
        entries = rate_limiter.get_all_entries()

        for entry in entries:
            limits.append(
                RateLimitEntry(
                    key=entry.get("key", ""),
                    endpoint=entry.get("endpoint", ""),
                    limit=entry.get("limit", 0),
                    remaining=entry.get("remaining", 0),
                    reset_at=entry.get("reset_at", ""),
                )
            )
    except Exception as exc:
        logger.warning(
            "[ADMIN-ROUTES] Failed to retrieve rate limits: %s  request_id=%s",
            exc,
            request_id,
        )
        # Return empty list rather than erroring out
        limits = []

    logger.info(
        "[ADMIN-ROUTES] Rate limits viewed  entries=%d  user=%s  request_id=%s",
        len(limits),
        _admin.get("sub"),
        request_id,
    )

    return RateLimitListResponse(limits=limits)


# ---------------------------------------------------------------------------
# GET /api/v1/admin/stats — System statistics
# ---------------------------------------------------------------------------

@router.get(
    "/stats",
    response_model=StatsResponse,
    summary="Get system statistics",
    description=(
        "Return aggregated system statistics including search metrics, "
        "index info, and performance data. Admin only."
    ),
    responses={
        403: {"description": "Admin access required"},
    },
)
async def get_stats(
    request: Request,
    _admin: dict = Depends(require_admin),
) -> StatsResponse:
    """Get system statistics for admin dashboard.

    Returns search metrics (total queries, latency, top queries),
    per-index statistics, cache hit rate, and QPS.
    """
    request_id = _get_request_id(request)

    stats = StatsResponse()

    # Get search analytics
    try:
        from isocortex.engine.analytics.engine import AnalyticsEngine
        from isocortex.storage import get_analytics

        analytics = get_analytics()
        if analytics:
            search_stats = analytics.get_search_stats()

            stats.total_queries = search_stats.total_queries
            stats.avg_latency_ms = search_stats.avg_latency_ms
            stats.p95_latency_ms = search_stats.p95_latency_ms
            stats.queries_per_minute = search_stats.queries_per_minute
            stats.unique_queries = search_stats.unique_queries
            stats.top_queries = [
                {"query": q, "count": c}
                for q, c in search_stats.top_queries
            ]
    except Exception as exc:
        logger.warning(
            "[ADMIN-ROUTES] Failed to retrieve search stats: %s  request_id=%s",
            exc,
            request_id,
        )

    # Get per-index statistics
    try:
        from isocortex.engine.indexing.manager import IndexManager
        from isocortex.config import get_config

        config = get_config()
        mgr = IndexManager(config.storage.indices_dir)
        indexes = mgr.list_indexes()

        for idx in indexes:
            stats.indexes.append({
                "name": idx.name,
                "vector_count": idx.vector_count,
                "deleted_count": idx.deleted_count,
                "healthy": idx.healthy,
            })
    except Exception as exc:
        logger.warning(
            "[ADMIN-ROUTES] Failed to retrieve index stats: %s  request_id=%s",
            exc,
            request_id,
        )

    # Get cache hit rate from SearchEngine (if available)
    try:
        from isocortex.core.search.engine import SearchEngine
        # The cache stats are per-engine instance, so we report aggregate
        # In a production setup, this would come from a global registry
        stats.cache_hit_rate = 0.0  # Default — populated by active engines
    except Exception:
        pass

    logger.info(
        "[ADMIN-ROUTES] Stats viewed  total_queries=%d  indexes=%d  user=%s  "
        "request_id=%s",
        stats.total_queries,
        len(stats.indexes),
        _admin.get("sub"),
        request_id,
    )

    return stats


# ---------------------------------------------------------------------------
# GET /health — Health check (no auth required)
# ---------------------------------------------------------------------------

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description=(
        "Return system health status. No authentication required. "
        "Used by load balancers and monitoring systems."
    ),
    responses={
        200: {"description": "System is healthy"},
        503: {"description": "System is unhealthy"},
    },
)
async def health_check(request: Request) -> HealthResponse:
    """System health check endpoint.

    No authentication required (SRS: public endpoint).
    Used by load balancers and monitoring to determine service availability.

    Returns:
    - status: "healthy", "degraded", or "unhealthy"
    - version: Application version
    - uptime_seconds: Time since application start
    - total_indexes: Number of indexes on disk
    - total_vectors: Total active vectors across all indexes
    - memory_usage_mb: Current process memory usage
    - active_jobs: Number of running/queued jobs
    """
    request_id = _get_request_id(request)

    health_status = "healthy"
    total_indexes = 0
    total_vectors = 0
    memory_usage_mb = 0.0
    active_jobs = 0

    # Count indexes and vectors
    try:
        from isocortex.engine.indexing.manager import IndexManager
        from isocortex.config import get_config

        config = get_config()
        mgr = IndexManager(config.storage.indices_dir)
        indexes = mgr.list_indexes()
        total_indexes = len(indexes)
        for idx in indexes:
            total_vectors += idx.vector_count - idx.deleted_count
    except Exception as exc:
        logger.warning(
            "[ADMIN-ROUTES] Health check: failed to read indexes: %s  request_id=%s",
            exc,
            request_id,
        )
        health_status = "degraded"

    # Get memory usage
    try:
        import resource
        # RSS in bytes → MB
        memory_usage_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    except (ImportError, AttributeError):
        try:
            import psutil
            process = psutil.Process()
            memory_usage_mb = process.memory_info().rss / (1024 * 1024)
        except ImportError:
            memory_usage_mb = 0.0

    # Count active jobs
    try:
        from isocortex.engine.jobs.scheduler import JobScheduler, JobStatus
        from isocortex.config import get_config

        config = get_config()
        db_path = config.storage.data_dir / "jobs.db"
        scheduler = JobScheduler(db_path=str(db_path))

        running = scheduler.list_jobs(status=JobStatus.RUNNING, limit=1000)
        queued = scheduler.list_jobs(status=JobStatus.QUEUED, limit=1000)
        active_jobs = len(running) + len(queued)
    except Exception:
        pass

    response = HealthResponse(
        status=health_status,
        version=APP_VERSION,
        uptime_seconds=_get_uptime_seconds(),
        total_indexes=total_indexes,
        total_vectors=total_vectors,
        memory_usage_mb=round(memory_usage_mb, 1),
        active_jobs=active_jobs,
    )

    logger.debug(
        "[ADMIN-ROUTES] Health check  status=%s  indexes=%d  vectors=%d  "
        "memory=%.1fMB  jobs=%d  request_id=%s",
        health_status,
        total_indexes,
        total_vectors,
        memory_usage_mb,
        active_jobs,
        request_id,
    )

    return response
