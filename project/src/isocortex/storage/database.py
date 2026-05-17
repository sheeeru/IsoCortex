"""
IsoCortex — Storage Layer
==========================

SQLite-backed storage for metadata, analytics, user management,
rate limiting, and job tracking.

SRS References: Section 4.4 (SQLite WAL Mode), Section 6 (Auth),
                Section 9 (Async Jobs), Section 10 (NFRs)

All operations use WAL mode for concurrent read/write access
with connection pooling.

Author : Shaheer Qureshi
Project: IsoCortex
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Database Manager
# =============================================================================

class Database:
    """
    SQLite database manager with WAL mode and connection pooling.

    Provides thread-safe access to the IsoCortex database with:
      - WAL journal mode for concurrent reads
      - Connection pooling (default 5 connections)
      - Configurable busy timeout
      - Auto-initialization of all required tables

    SRS Section 4.4: SQLite WAL Mode
    """

    def __init__(
        self,
        db_path: str | Path,
        pool_size: int = 5,
        busy_timeout: int = 5000,
        wal_autocheckpoint: int = 1000,
    ) -> None:
        self._db_path = Path(db_path)
        self._pool_size = pool_size
        self._busy_timeout = busy_timeout
        self._wal_autocheckpoint = wal_autocheckpoint
        self._local = threading.local()
        logger.info(
            "[DB] Initializing  path=%s  pool=%d  busy_timeout=%dms",
            self._db_path, pool_size, busy_timeout,
        )

    @property
    def db_path(self) -> Path:
        return self._db_path

    def _create_connection(self) -> sqlite3.Connection:
        """Create a new connection with WAL mode configuration."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(
            str(self._db_path),
            check_same_thread=False,
            timeout=self._busy_timeout / 1000.0,
        )
        conn.row_factory = sqlite3.Row
        conn.execute(f"PRAGMA journal_mode=WAL")
        conn.execute(f"PRAGMA busy_timeout={self._busy_timeout}")
        conn.execute(f"PRAGMA synchronous=NORMAL")
        conn.execute(f"PRAGMA wal_autocheckpoint={self._wal_autocheckpoint}")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def get_connection(self) -> sqlite3.Connection:
        """Get a thread-local connection from the pool."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = self._create_connection()
            logger.debug("[DB] New connection for thread %s", threading.current_thread().name)
        return self._local.conn

    @contextmanager
    def transaction(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager for transactional database operations."""
        conn = self.get_connection()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def initialize(self) -> None:
        """Create all required tables if they don't exist."""
        with self.transaction() as conn:
            conn.executescript(SCHEMA_SQL)

            # Migration: add columns that may be missing from older DBs
            self._migrate_users_table(conn)

        logger.info("[DB] Schema initialized successfully")

    @staticmethod
    def _migrate_users_table(conn: sqlite3.Connection) -> None:
        """Add missing columns to the users table (idempotent migration).

        Handles upgrades from earlier schema versions that lacked
        locked_at and failed_login_attempts (SRS Section 6.3).
        """
        cols = {
            c[1].lower()
            for c in conn.execute("PRAGMA table_info(users)").fetchall()
        }

        migrations = [
            ("locked_at", "TEXT"),
            ("failed_login_attempts", "INTEGER NOT NULL DEFAULT 0"),
        ]

        for col_name, col_type in migrations:
            if col_name not in cols:
                conn.execute(
                    f"ALTER TABLE users ADD COLUMN {col_name} {col_type}"
                )
                logger.info("[DB] Migration: added column users.%s", col_name)

    def close(self) -> None:
        """Close the thread-local connection."""
        if hasattr(self._local, "conn") and self._local.conn is not None:
            self._local.conn.close()
            self._local.conn = None
            logger.debug("[DB] Connection closed for thread %s", threading.current_thread().name)


# Need threading for thread-local storage
import threading


# =============================================================================
# Schema Definition (SRS Section 6, 9, 10)
# =============================================================================

SCHEMA_SQL = """
-- =========================================================================
-- Users table (SRS Section 6: Authentication & User Management)
-- =========================================================================
CREATE TABLE IF NOT EXISTS users (
    user_id     TEXT PRIMARY KEY,
    username    TEXT    NOT NULL UNIQUE,
    email       TEXT    NOT NULL UNIQUE,
    password_hash TEXT  NOT NULL,
    role        TEXT    NOT NULL DEFAULT 'user' CHECK(role IN ('admin', 'user')),
    is_active   INTEGER NOT NULL DEFAULT 1,
    created_at  TEXT    NOT NULL DEFAULT (datetime('now')),
    updated_at  TEXT    NOT NULL DEFAULT (datetime('now')),
    last_login  TEXT,
    locked_at   TEXT,
    failed_login_attempts INTEGER NOT NULL DEFAULT 0
);

-- =========================================================================
-- API Keys table (SRS Section 6: Auth)
-- =========================================================================
CREATE TABLE IF NOT EXISTS api_keys (
    key_id      TEXT PRIMARY KEY,
    user_id     TEXT    NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    key_hash    TEXT    NOT NULL UNIQUE,
    name        TEXT,
    is_active   INTEGER NOT NULL DEFAULT 1,
    created_at  TEXT    NOT NULL DEFAULT (datetime('now')),
    expires_at  TEXT
);

-- =========================================================================
-- Indexes table (SRS Section 3: Index Manager)
-- =========================================================================
CREATE TABLE IF NOT EXISTS indexes (
    index_id      TEXT PRIMARY KEY,
    name          TEXT    NOT NULL UNIQUE,
    owner_id      TEXT    NOT NULL REFERENCES users(user_id),
    vector_count  INTEGER NOT NULL DEFAULT 0,
    dimension     INTEGER NOT NULL DEFAULT 384,
    metric        TEXT    NOT NULL DEFAULT 'cosine',
    hnsw_config   TEXT    NOT NULL DEFAULT '{}',
    format_version INTEGER NOT NULL DEFAULT 1,
    is_active     INTEGER NOT NULL DEFAULT 1,
    created_at    TEXT    NOT NULL DEFAULT (datetime('now')),
    updated_at    TEXT    NOT NULL DEFAULT (datetime('now')),
    file_count    INTEGER NOT NULL DEFAULT 0,
    total_size    INTEGER NOT NULL DEFAULT 0
);

-- =========================================================================
-- Documents table (SRS Section 5: Ingestion)
-- =========================================================================
CREATE TABLE IF NOT EXISTS documents (
    doc_id        TEXT PRIMARY KEY,
    index_id      TEXT    NOT NULL REFERENCES indexes(index_id) ON DELETE CASCADE,
    file_path     TEXT    NOT NULL,
    file_hash     TEXT    NOT NULL DEFAULT '',
    format_category TEXT NOT NULL DEFAULT '',
    chunk_count   INTEGER NOT NULL DEFAULT 0,
    word_count    INTEGER NOT NULL DEFAULT 0,
    token_count   INTEGER NOT NULL DEFAULT 0,
    status        TEXT    NOT NULL DEFAULT 'indexed'
                        CHECK(status IN ('pending', 'processing', 'indexed', 'failed', 'deleted')),
    error_message TEXT,
    created_at    TEXT    NOT NULL DEFAULT (datetime('now')),
    updated_at    TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_documents_index_id ON documents(index_id);
CREATE INDEX IF NOT EXISTS idx_documents_file_hash ON documents(file_hash);
CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status);

-- =========================================================================
-- Analytics table (SRS Section 10: NFRs)
-- =========================================================================
CREATE TABLE IF NOT EXISTS analytics (
    event_id      TEXT PRIMARY KEY,
    event_type    TEXT    NOT NULL CHECK(
        event_type IN ('search', 'index_created', 'index_deleted', 'document_ingested',
                       'user_login', 'user_created', 'error', 'job_started', 'job_completed')
    ),
    user_id       TEXT,
    index_id      TEXT,
    metadata      TEXT    NOT NULL DEFAULT '{}',
    created_at    TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_analytics_event_type ON analytics(event_type);
CREATE INDEX IF NOT EXISTS idx_analytics_created_at ON analytics(created_at);

-- =========================================================================
-- Rate Limiting table (SRS Section 10: NFR)
-- =========================================================================
CREATE TABLE IF NOT EXISTS rate_limits (
    key_hash      TEXT    NOT NULL,
    endpoint      TEXT    NOT NULL DEFAULT '',
    request_count INTEGER NOT NULL DEFAULT 0,
    window_start  TEXT    NOT NULL,
    PRIMARY KEY (key_hash, endpoint)
);

-- =========================================================================
-- Jobs table (SRS Section 9: Async Operations)
-- =========================================================================
CREATE TABLE IF NOT EXISTS jobs (
    job_id        TEXT PRIMARY KEY,
    job_type      TEXT    NOT NULL CHECK(
        job_type IN ('index_create', 'index_delete', 'compaction',
                     'export', 'import', 'batch_ingest')
    ),
    status        TEXT    NOT NULL DEFAULT 'pending'
                    CHECK(status IN ('pending', 'running', 'completed', 'failed', 'cancelled')),
    user_id       TEXT,
    index_id      TEXT,
    progress      REAL    NOT NULL DEFAULT 0.0,
    result        TEXT,
    error_message TEXT,
    created_at    TEXT    NOT NULL DEFAULT (datetime('now')),
    started_at    TEXT,
    completed_at  TEXT
);

CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_user_id ON jobs(user_id);
"""


# =============================================================================
# Analytics Engine (SRS Section 10)
# =============================================================================

class AnalyticsEngine:
    """
    SQLite-backed analytics for tracking system usage.

    Records search queries, indexing operations, user actions,
    and system errors. Provides aggregation queries for dashboards.
    """

    def __init__(self, db: Database) -> None:
        self._db = db

    def record_event(
        self,
        event_type: str,
        *,
        user_id: Optional[str] = None,
        index_id: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> str:
        """
        Record an analytics event.

        Parameters
        ----------
        event_type : str — Event category (search, index_created, etc.)
        user_id    : str | None — User who triggered the event.
        index_id   : str | None — Index involved (if applicable).
        metadata   : dict | None — Additional event data.

        Returns
        -------
        str — Event ID.
        """
        event_id = str(uuid.uuid4())
        meta_json = json.dumps(metadata or {}, ensure_ascii=False)

        with self._db.transaction() as conn:
            conn.execute(
                """INSERT INTO analytics (event_id, event_type, user_id, index_id, metadata)
                   VALUES (?, ?, ?, ?, ?)""",
                (event_id, event_type, user_id, index_id, meta_json),
            )

        logger.debug("[ANALYTICS] Recorded %s  event_id=%s", event_type, event_id)
        return event_id

    def get_search_stats(
        self,
        hours: int = 24,
    ) -> dict[str, Any]:
        """
        Get search statistics for the last N hours.

        Returns
        -------
        dict with total_searches, avg_latency_ms, unique_users, top_queries.
        """
        with self._db.transaction() as conn:
            row = conn.execute(
                """SELECT
                       COUNT(*) as total_searches,
                       COALESCE(AVG(CAST(json_extract(metadata, '$.latency_ms') AS REAL)), 0) as avg_latency,
                       COUNT(DISTINCT user_id) as unique_users
                   FROM analytics
                   WHERE event_type = 'search'
                     AND created_at >= datetime('now', ?)""",
                (f"-{hours} hours",),
            ).fetchone()

            top_queries = conn.execute(
                """SELECT json_extract(metadata, '$.query') as query_text, COUNT(*) as count
                   FROM analytics
                   WHERE event_type = 'search'
                     AND created_at >= datetime('now', ?)
                   GROUP BY query_text
                   ORDER BY count DESC
                   LIMIT 10""",
                (f"-{hours} hours",),
            ).fetchall()

        return {
            "total_searches": row["total_searches"] if row else 0,
            "avg_latency_ms": round(row["avg_latency"], 2) if row else 0.0,
            "unique_users": row["unique_users"] if row else 0,
            "top_queries": [
                {"query": r["query_text"], "count": r["count"]}
                for r in top_queries
            ],
        }

    def get_system_health(self) -> dict[str, Any]:
        """Get system health metrics."""
        with self._db.transaction() as conn:
            indexes = conn.execute(
                "SELECT COUNT(*) as count FROM indexes WHERE is_active = 1"
            ).fetchone()

            documents = conn.execute(
                "SELECT COUNT(*) as count FROM documents WHERE status = 'indexed'"
            ).fetchone()

            users = conn.execute(
                "SELECT COUNT(*) as count FROM users WHERE is_active = 1"
            ).fetchone()

            jobs = conn.execute(
                "SELECT status, COUNT(*) as count FROM jobs GROUP BY status"
            ).fetchall()

            errors = conn.execute(
                """SELECT COUNT(*) as count FROM analytics
                   WHERE event_type = 'error'
                     AND created_at >= datetime('now', '-24 hours')"""
            ).fetchone()

        return {
            "active_indexes": indexes["count"] if indexes else 0,
            "indexed_documents": documents["count"] if documents else 0,
            "active_users": users["count"] if users else 0,
            "jobs": {r["status"]: r["count"] for r in jobs},
            "errors_last_24h": errors["count"] if errors else 0,
        }


# =============================================================================
# Rate Limiter (SRS Section 10)
# =============================================================================

class RateLimiter:
    """
    SQLite-backed sliding window rate limiter.

    Limits requests per API key and endpoint combination.
    Uses a time-based sliding window for accurate rate tracking.

    SRS Section 10: Non-Functional Requirements
    """

    def __init__(
        self,
        db: Database,
        max_requests: int = 100,
        window_seconds: int = 60,
    ) -> None:
        self._db = db
        self._max_requests = max_requests
        self._window_seconds = window_seconds

    def is_allowed(
        self,
        key_hash: str,
        endpoint: str = "",
        limit: Optional[int] = None,
        window_seconds: Optional[int] = None,
    ) -> tuple[bool, int, float]:
        """
        Check if a request is allowed under the rate limit.

        Parameters
        ----------
        key_hash       : str — Hashed API key or user identifier.
        endpoint       : str — API endpoint being accessed.
        limit          : int | None — Override max requests (uses default if None).
        window_seconds : int | None — Override window size (uses default if None).

        Returns
        -------
        (allowed, remaining, reset_at)
          allowed   : bool — True if request is within limits.
          remaining : int — Number of requests remaining in this window.
          reset_at  : float — Unix timestamp when the window resets.
        """
        max_req = limit or self._max_requests
        window = window_seconds or self._window_seconds
        now_ts = datetime.now(timezone.utc).timestamp()
        window_start = datetime.now(timezone.utc).isoformat()

        with self._db.transaction() as conn:
            # Clean up old entries
            cutoff = now_ts - window
            conn.execute(
                """DELETE FROM rate_limits
                   WHERE key_hash = ? AND endpoint = ?
                     AND strftime('%s', window_start) < ?""",
                (key_hash, endpoint, cutoff),
            )

            # Get current count
            row = conn.execute(
                """SELECT request_count, window_start FROM rate_limits
                   WHERE key_hash = ? AND endpoint = ?""",
                (key_hash, endpoint),
            ).fetchone()

            current_count = row["request_count"] if row else 0
            current_start = row["window_start"] if row else None

            reset_at = now_ts + window

            # Check if window has expired
            if current_start:
                start_ts = datetime.fromisoformat(current_start).timestamp()
                if now_ts - start_ts > window:
                    # Reset window
                    current_count = 0
                    conn.execute(
                        """UPDATE rate_limits
                           SET request_count = 1, window_start = ?
                           WHERE key_hash = ? AND endpoint = ?""",
                        (window_start, key_hash, endpoint),
                    )
                    reset_at = now_ts + window
                else:
                    # Increment
                    conn.execute(
                        """UPDATE rate_limits SET request_count = request_count + 1
                           WHERE key_hash = ? AND endpoint = ?""",
                        (key_hash, endpoint),
                    )
                    current_count += 1
                    reset_at = start_ts + window
            else:
                # First request in window
                conn.execute(
                    """INSERT INTO rate_limits (key_hash, endpoint, request_count, window_start)
                       VALUES (?, ?, 1, ?)""",
                    (key_hash, endpoint, window_start),
                )
                current_count = 1

        remaining = max(0, max_req - current_count)
        allowed = current_count <= max_req

        return allowed, remaining, reset_at

    def get_all_entries(self) -> list[dict[str, Any]]:
        """Return all active rate limit entries.

        Used by the admin endpoint GET /api/v1/admin/rate-limits
        (SRS Section 11: Rate Limiting).
        """
        with self._db.transaction() as conn:
            rows = conn.execute(
                """SELECT key_hash, endpoint, request_count, window_start
                   FROM rate_limits"""
            ).fetchall()

        now_ts = datetime.now(timezone.utc).timestamp()
        entries = []
        for r in rows:
            start_ts = datetime.fromisoformat(r["window_start"]).timestamp()
            remaining = max(0, self._max_requests - r["request_count"])
            entries.append({
                "key": r["key_hash"],
                "endpoint": r["endpoint"],
                "limit": self._max_requests,
                "remaining": remaining,
                "reset_at": start_ts + self._window_seconds,
            })
        return entries

    def reset_key(self, key_hash: str) -> bool:
        """Remove all rate limit entries for a given key.

        Returns True if any entries were removed.
        """
        with self._db.transaction() as conn:
            cursor = conn.execute(
                "DELETE FROM rate_limits WHERE key_hash = ?", (key_hash,)
            )
            return cursor.rowcount > 0


# =============================================================================
# Global database singleton
# =============================================================================

_global_db: Optional[Database] = None
_global_analytics: Optional[AnalyticsEngine] = None
_global_rate_limiter: Optional[RateLimiter] = None


def get_database(db_path: Optional[Path] = None) -> Database:
    """Return the global database singleton."""
    global _global_db
    if _global_db is None:
        from isocortex.config import get_config
        config = get_config()
        path = db_path or config.storage.db_path
        _global_db = Database(
            db_path=path,
            busy_timeout=config.storage.busy_timeout,
            wal_autocheckpoint=config.storage.wal_auto_checkpoint,
        )
        _global_db.initialize()
    return _global_db


def get_analytics() -> AnalyticsEngine:
    """Return the global analytics engine singleton."""
    global _global_analytics
    if _global_analytics is None:
        _global_analytics = AnalyticsEngine(get_database())
    return _global_analytics


def get_rate_limiter() -> RateLimiter:
    """Return the global rate limiter singleton."""
    global _global_rate_limiter
    if _global_rate_limiter is None:
        from isocortex.config import get_config
        config = get_config()
        _global_rate_limiter = RateLimiter(
            db=get_database(),
            max_requests=config.rate_limit.requests,
            window_seconds=config.rate_limit.window_seconds,
        )
    return _global_rate_limiter


def close_database() -> None:
    """Close all global database connections."""
    global _global_db, _global_analytics, _global_rate_limiter
    if _global_db is not None:
        _global_db.close()
        _global_db = None
    _global_analytics = None
    _global_rate_limiter = None
    logger.info("[DB] All database connections closed")
