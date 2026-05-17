"""
IsoCortex — Database Tests
===========================
Tests for Database, AnalyticsEngine, and RateLimiter.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from isocortex.storage.database import (
    AnalyticsEngine,
    Database,
    RateLimiter,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture()
def database(tmp_path: Path) -> Database:
    """Create and initialize a Database."""
    db = Database(db_path=tmp_path / "test.db")
    db.initialize()
    return db


@pytest.fixture()
def analytics(database: Database) -> AnalyticsEngine:
    """Create an AnalyticsEngine backed by the test database."""
    return AnalyticsEngine(database)


@pytest.fixture()
def rate_limiter(database: Database) -> RateLimiter:
    """Create a RateLimiter with low limits for testing."""
    return RateLimiter(database, max_requests=3, window_seconds=60)


# =============================================================================
# Database Tests
# =============================================================================

class TestDatabase:

    def test_database_initialize(self, database: Database):
        """Schema creation."""
        with database.transaction() as conn:
            # Verify tables exist
            tables = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
            table_names = {t["name"] for t in tables}
            assert "users" in table_names
            assert "analytics" in table_names
            assert "rate_limits" in table_names
            assert "documents" in table_names
            assert "indexes" in table_names
            assert "jobs" in table_names

    def test_database_transaction_commit(self, database: Database):
        """Commit writes data."""
        with database.transaction() as conn:
            conn.execute(
                "INSERT INTO users (user_id, username, email, password_hash, role) "
                "VALUES (?, ?, ?, ?, ?)",
                ("u1", "testuser", "test@test.com", "hashed", "user"),
            )

        # Read back in a new transaction
        with database.transaction() as conn:
            row = conn.execute(
                "SELECT * FROM users WHERE user_id = ?", ("u1",)
            ).fetchone()
        assert row is not None
        assert row["username"] == "testuser"

    def test_database_transaction_rollback(self, database: Database):
        """Rollback on error."""
        try:
            with database.transaction() as conn:
                conn.execute(
                    "INSERT INTO users (user_id, username, email, password_hash, role) "
                    "VALUES (?, ?, ?, ?, ?)",
                    ("u2", "will_rollback", "rb@test.com", "hashed", "user"),
                )
                raise ValueError("Forced error")
        except ValueError:
            pass

        # Verify data was rolled back
        with database.transaction() as conn:
            row = conn.execute(
                "SELECT * FROM users WHERE user_id = ?", ("u2",)
            ).fetchone()
        assert row is None


# =============================================================================
# Rate Limiter Tests
# =============================================================================

class TestRateLimiter:

    def test_rate_limiter_allowed(self, rate_limiter: RateLimiter):
        """Under limit, return True."""
        allowed, remaining, reset_at = rate_limiter.is_allowed("key-1")
        assert allowed is True
        assert remaining > 0
        assert isinstance(reset_at, float)

    def test_rate_limiter_blocked(self, rate_limiter: RateLimiter):
        """Over limit, return False."""
        # max_requests=3, so 4th should be blocked
        for i in range(3):
            allowed, remaining, _ = rate_limiter.is_allowed("key-2")
            assert allowed is True

        allowed, remaining, _ = rate_limiter.is_allowed("key-2")
        assert allowed is False
        assert remaining == 0

    def test_rate_limiter_different_keys(self, rate_limiter: RateLimiter):
        """Different keys have independent limits."""
        for _ in range(3):
            rate_limiter.is_allowed("key-a")
        # key-a is blocked
        allowed, _, _ = rate_limiter.is_allowed("key-a")
        assert allowed is False
        # key-b is not
        allowed, _, _ = rate_limiter.is_allowed("key-b")
        assert allowed is True


# =============================================================================
# Analytics Tests
# =============================================================================

class TestAnalytics:

    def test_analytics_record_event(self, analytics: AnalyticsEngine):
        """Record and retrieve events."""
        event_id = analytics.record_event(
            "search",
            user_id="user-1",
            metadata={"query": "test query", "latency_ms": 42.5},
        )
        assert event_id

    def test_analytics_get_search_stats(self, analytics: AnalyticsEngine):
        """Get search stats returns expected structure."""
        analytics.record_event("search", metadata={"query": "hello", "latency_ms": 10})
        stats = analytics.get_search_stats(hours=24)
        assert "total_searches" in stats
        assert "avg_latency_ms" in stats
        assert "unique_users" in stats
        assert "top_queries" in stats
        assert stats["total_searches"] >= 1

    def test_analytics_get_system_health(self, analytics: AnalyticsEngine):
        """Get system health returns expected structure."""
        health = analytics.get_system_health()
        assert "active_indexes" in health
        assert "indexed_documents" in health
        assert "active_users" in health
        assert "errors_last_24h" in health
