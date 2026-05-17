"""
IsoCortex — Analytics Engine
=============================

Tracks search patterns, system health, usage metrics, and provides
admin-visible analytics data.

All data is stored in SQLite (via the storage layer's AnalyticsEngine)
for persistence across restarts and cross-worker visibility.

SRS References:
  - Analytics: Search pattern tracking, popular queries, system health
  - NFR-07: Rate limiting metrics
  - NFR-17: Embedding cache hit rate (> 80% target)

Author : Shaheer Qureshi
Project: IsoCortex
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass(frozen=True)
class SearchRecord:
    """A single search query record."""
    query: str
    index_name: str
    result_count: int
    latency_ms: float
    user_id: Optional[str] = None
    k: int = 5
    timestamp: str = ""


@dataclass(frozen=True)
class SearchStats:
    """Aggregated search statistics."""
    total_queries: int
    avg_latency_ms: float
    p95_latency_ms: float
    top_queries: list[tuple[str, int]]
    queries_per_minute: float
    unique_queries: int


@dataclass(frozen=True)
class SystemHealth:
    """System health snapshot."""
    status: str  # "healthy", "degraded", "unhealthy"
    uptime_seconds: float
    total_indexes: int
    total_vectors: int
    memory_usage_mb: float
    active_jobs: int
    search_qps: float
    cache_hit_rate: float


# =============================================================================
# Analytics Engine
# =============================================================================

class AnalyticsEngine:
    """High-level analytics engine for IsoCortex.

    Wraps the storage-layer AnalyticsEngine with convenience methods
    and provides pre-computed statistics for admin dashboards.

    Thread Safety:
    - All writes go through SQLite WAL mode
    - Read operations are non-blocking
    """

    def __init__(self, storage_analytics: Any = None) -> None:
        """
        Parameters
        ----------
        storage_analytics : AnalyticsEngine from storage.database
            The low-level storage analytics engine. If None, analytics
            are logged in-memory only (for testing).
        """
        self._storage = storage_analytics
        self._start_time = time.time()

        # In-memory fallback counters (when no storage engine)
        self._query_count: int = 0
        self._total_latency: float = 0.0
        self._query_history: list[dict[str, Any]] = []
        self._max_history: int = 10000

        logger.info("[ANALYTICS] Engine initialized")

    # -----------------------------------------------------------------
    # Record Events
    # -----------------------------------------------------------------

    def record_search(self, record: SearchRecord) -> None:
        """Record a search query event.

        SRS: All search queries are logged for analytics.
        """
        timestamp = record.timestamp or datetime.now(timezone.utc).isoformat()
        event_data = {
            "query": record.query[:200],  # Truncate long queries
            "index_name": record.index_name,
            "result_count": record.result_count,
            "latency_ms": record.latency_ms,
            "k": record.k,
        }
        if record.user_id:
            event_data["user_id"] = record.user_id

        # Write to storage
        if self._storage is not None:
            try:
                self._storage.record_event(
                    event_type="search",
                    metadata=event_data,
                )
            except Exception as exc:
                logger.warning("[ANALYTICS] Failed to record search: %s", exc)

        # In-memory tracking
        self._query_count += 1
        self._total_latency += record.latency_ms
        self._query_history.append({
            "timestamp": timestamp,
            **event_data,
        })
        if len(self._query_history) > self._max_history:
            self._query_history = self._query_history[-self._max_history:]

    def record_index_created(self, name: str, vector_count: int, duration_s: float) -> None:
        """Record an index creation event."""
        if self._storage is not None:
            try:
                self._storage.record_event("index_created", {
                    "name": name,
                    "vector_count": vector_count,
                    "duration_seconds": round(duration_s, 2),
                })
            except Exception as exc:
                logger.warning("[ANALYTICS] Failed to record index_created: %s", exc)

    def record_export(self, name: str, size_mb: float, duration_s: float) -> None:
        """Record an index export event."""
        if self._storage is not None:
            try:
                self._storage.record_event("export", {
                    "name": name,
                    "size_mb": round(size_mb, 2),
                    "duration_seconds": round(duration_s, 2),
                })
            except Exception as exc:
                logger.warning("[ANALYTICS] Failed to record export: %s", exc)

    def record_error(
        self,
        error_code: str,
        message: str,
        endpoint: str = "",
        user_id: Optional[str] = None,
    ) -> None:
        """Record an error event."""
        if self._storage is not None:
            try:
                self._storage.record_event("error", {
                    "code": error_code,
                    "message": message[:500],
                    "endpoint": endpoint,
                    "user_id": user_id,
                })
            except Exception as exc:
                logger.warning("[ANALYTICS] Failed to record error: %s", exc)

    # -----------------------------------------------------------------
    # Query Statistics
    # -----------------------------------------------------------------

    def get_search_stats(self) -> SearchStats:
        """Get aggregated search statistics.

        Returns avg/p95 latency, top queries, QPS, unique queries.
        """
        if self._query_count == 0:
            return SearchStats(
                total_queries=0,
                avg_latency_ms=0.0,
                p95_latency_ms=0.0,
                top_queries=[],
                queries_per_minute=0.0,
                unique_queries=0,
            )

        latencies = sorted(
            h["latency_ms"] for h in self._query_history if "latency_ms" in h
        )
        avg = self._total_latency / self._query_count

        # P95
        p95 = 0.0
        if latencies:
            p95_idx = int(len(latencies) * 0.95)
            p95 = latencies[min(p95_idx, len(latencies) - 1)]

        # QPS (queries in last 60 seconds)
        now = time.time()
        recent = sum(
            1 for h in self._query_history
            if now - _parse_ts(h.get("timestamp", "")).timestamp() < 60
        )

        # Top queries
        query_counts: dict[str, int] = {}
        for h in self._query_history:
            q = h.get("query", "").lower()
            if q:
                query_counts[q] = query_counts.get(q, 0) + 1
        top = sorted(query_counts.items(), key=lambda x: x[1], reverse=True)[:10]

        # Unique queries
        unique = len(set(
            h.get("query", "").lower()
            for h in self._query_history
            if h.get("query")
        ))

        uptime_min = (time.time() - self._start_time) / 60
        qpm = self._query_count / uptime_min if uptime_min > 0 else 0

        return SearchStats(
            total_queries=self._query_count,
            avg_latency_ms=round(avg, 2),
            p95_latency_ms=round(p95, 2),
            top_queries=top,
            queries_per_minute=round(qpm, 2),
            unique_queries=unique,
        )

    # -----------------------------------------------------------------
    # System Health
    # -----------------------------------------------------------------

    def get_system_health(
        self,
        total_indexes: int = 0,
        total_vectors: int = 0,
        memory_mb: float = 0.0,
        active_jobs: int = 0,
        cache_hit_rate: float = 0.0,
    ) -> SystemHealth:
        """Get system health snapshot.

        SRS: Dashboard health overview.
        """
        status = "healthy"
        if self._query_count > 0:
            stats = self.get_search_stats()
            if stats.p95_latency_ms > 500:
                status = "degraded"
            if stats.p95_latency_ms > 2000:
                status = "unhealthy"

        return SystemHealth(
            status=status,
            uptime_seconds=round(time.time() - self._start_time, 1),
            total_indexes=total_indexes,
            total_vectors=total_vectors,
            memory_usage_mb=round(memory_mb, 1),
            active_jobs=active_jobs,
            search_qps=self.get_search_stats().queries_per_minute / 60,
            cache_hit_rate=round(cache_hit_rate, 3),
        )


def _parse_ts(ts: str) -> datetime:
    """Parse an ISO timestamp string, returning epoch on failure."""
    if not ts:
        return datetime(1970, 1, 1, tzinfo=timezone.utc)
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return datetime(1970, 1, 1, tzinfo=timezone.utc)
