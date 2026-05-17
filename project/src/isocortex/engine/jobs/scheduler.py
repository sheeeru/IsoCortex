"""
IsoCortex — Job Scheduler
==========================

Background job execution framework for long-running operations.

Job Lifecycle (SRS Section 8.2):
  1. Client submits operation → server creates job record
  2. Server responds with 202 Accepted + job_id
  3. Job queued for execution by scheduler
  4. Client polls GET /api/v1/jobs/{job_id} for status
  5. Or subscribes to SSE for real-time progress
  6. On completion, result stored and available for retrieval
  7. Records cleaned up after retention period (default 7 days)

Job Types (SRS Section 8.4):
  - index_create, index_update, index_delete
  - index_compact, export, import

SRS References: Section 8, FR-API-001, FR-API-006, NFR-10

Author : Shaheer Qureshi
Project: IsoCortex
"""

from __future__ import annotations

import asyncio
import enum
import json
import logging
import os
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Coroutine, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

DEFAULT_RETENTION_DAYS: int = 7
DEFAULT_MAX_CONCURRENT: int = 2
DEFAULT_CLEANUP_INTERVAL_HRS: int = 24
SSE_KEEPALIVE_SECONDS: int = 15


# =============================================================================
# Enums
# =============================================================================

class JobStatus(str, enum.Enum):
    """Job execution status."""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobType(str, enum.Enum):
    """Job type identifiers."""
    INDEX_CREATE = "index_create"
    INDEX_UPDATE = "index_update"
    INDEX_DELETE = "index_delete"
    INDEX_COMPACT = "index_compact"
    EXPORT = "export"
    IMPORT = "import"


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class JobProgress:
    """Progress information for a running job."""
    percentage: float = 0.0
    message: str = ""
    files_processed: int = 0
    files_total: int = 0

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "percentage": round(self.percentage, 1),
            "message": self.message,
        }
        if self.files_total > 0:
            d["files_processed"] = self.files_processed
            d["files_total"] = self.files_total
        return d


@dataclass
class Job:
    """A background job with full lifecycle tracking.

    SRS Section 8.2/8.3: Job status endpoint response format.
    """
    job_id: str
    job_type: str
    status: JobStatus = JobStatus.PENDING
    progress: JobProgress = field(default_factory=JobProgress)
    result: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    started_at: str | None = None
    completed_at: str | None = None
    failed_at: str | None = None
    estimated_completion: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to the SRS-specified JSON format."""
        d: dict[str, Any] = {
            "job_id": self.job_id,
            "type": self.job_type,
            "status": self.status.value,
            "progress": self.progress.to_dict(),
            "created_at": self.created_at,
        }
        if self.started_at:
            d["started_at"] = self.started_at
        if self.estimated_completion:
            d["estimated_completion"] = self.estimated_completion
        if self.status == JobStatus.COMPLETED and self.result:
            d["completed_at"] = self.completed_at
            d["result"] = self.result
        elif self.status == JobStatus.FAILED and self.error:
            d["failed_at"] = self.failed_at
            d["error"] = self.error
        return d


# =============================================================================
# SQLite Job Store
# =============================================================================

class JobStore:
    """SQLite-backed persistent job storage.

    Jobs survive server restarts. Uses WAL mode for concurrent access.
    """

    def __init__(self, db_path: str | Path) -> None:
        db_path = str(db_path)
        self._local = threading.local()
        self._db_path = db_path
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            conn = sqlite3.connect(self._db_path, check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=5000")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
        return self._local.conn

    def _init_db(self) -> None:
        conn = self._get_conn()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                job_type TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                progress REAL DEFAULT 0,
                progress_message TEXT DEFAULT '',
                files_processed INTEGER DEFAULT 0,
                files_total INTEGER DEFAULT 0,
                result TEXT,
                error TEXT,
                payload TEXT,
                created_at TEXT NOT NULL,
                started_at TEXT,
                completed_at TEXT,
                failed_at TEXT,
                estimated_completion TEXT
            )
        """)
        conn.commit()
        logger.debug("[JOB-STORE] Initialized at %s", self._db_path)

    def create(self, job: Job) -> None:
        conn = self._get_conn()
        conn.execute("""
            INSERT INTO jobs (job_id, job_type, status, progress, progress_message,
                             files_processed, files_total, result, error, payload,
                             created_at, started_at, completed_at, failed_at,
                             estimated_completion)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            job.job_id, job.job_type, job.status.value,
            job.progress.percentage, job.progress.message,
            job.progress.files_processed, job.progress.files_total,
            json.dumps(job.result) if job.result else None,
            json.dumps(job.error) if job.error else None,
            json.dumps(job.payload) if job.payload else None,
            job.created_at, job.started_at, job.completed_at,
            job.failed_at, job.estimated_completion,
        ))
        conn.commit()

    def update_status(self, job_id: str, status: JobStatus) -> None:
        conn = self._get_conn()
        conn.execute(
            "UPDATE jobs SET status = ? WHERE job_id = ?",
            (status.value, job_id),
        )
        conn.commit()

    def update_progress(self, job_id: str, progress: JobProgress) -> None:
        conn = self._get_conn()
        conn.execute("""
            UPDATE jobs SET progress = ?, progress_message = ?,
                             files_processed = ?, files_total = ?
            WHERE job_id = ?
        """, (
            progress.percentage, progress.message,
            progress.files_processed, progress.files_total,
            job_id,
        ))
        conn.commit()

    def complete(self, job_id: str, result: dict[str, Any]) -> None:
        now = datetime.now(timezone.utc).isoformat()
        conn = self._get_conn()
        conn.execute("""
            UPDATE jobs SET status = 'completed', result = ?, completed_at = ?
            WHERE job_id = ?
        """, (json.dumps(result), now, job_id))
        conn.commit()

    def fail(self, job_id: str, error: dict[str, Any]) -> None:
        now = datetime.now(timezone.utc).isoformat()
        conn = self._get_conn()
        conn.execute("""
            UPDATE jobs SET status = 'failed', error = ?, failed_at = ?
            WHERE job_id = ?
        """, (json.dumps(error), now, job_id))
        conn.commit()

    def get(self, job_id: str) -> Job | None:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM jobs WHERE job_id = ?", (job_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_job(row)

    def list_jobs(
        self,
        status: JobStatus | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[Job]:
        conn = self._get_conn()
        if status:
            rows = conn.execute(
                "SELECT * FROM jobs WHERE status = ? ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (status.value, limit, offset),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
        return [self._row_to_job(r) for r in rows]

    def cleanup(self, retention_days: int = DEFAULT_RETENTION_DAYS) -> int:
        """Delete old completed/failed jobs. Returns count of deleted records."""
        conn = self._get_conn()
        cursor = conn.execute("""
            DELETE FROM jobs
            WHERE status IN ('completed', 'failed', 'cancelled')
              AND created_at < datetime('now', ? || ' days')
        """, (f"-{retention_days}",))
        conn.commit()
        deleted = cursor.rowcount
        if deleted > 0:
            logger.info("[JOB-STORE] Cleaned up %d old job records", deleted)
        return deleted

    @staticmethod
    def _row_to_job(row: sqlite3.Row) -> Job:
        return Job(
            job_id=row["job_id"],
            job_type=row["job_type"],
            status=JobStatus(row["status"]),
            progress=JobProgress(
                percentage=row["progress"],
                message=row["progress_message"] or "",
                files_processed=row["files_processed"] or 0,
                files_total=row["files_total"] or 0,
            ),
            result=json.loads(row["result"]) if row["result"] else None,
            error=json.loads(row["error"]) if row["error"] else None,
            created_at=row["created_at"],
            started_at=row["started_at"],
            completed_at=row["completed_at"],
            failed_at=row["failed_at"],
            estimated_completion=row["estimated_completion"],
            payload=json.loads(row["payload"]) if row["payload"] else {},
        )


# =============================================================================
# SSE Stream Manager
# =============================================================================

class SSEManager:
    """Manages Server-Sent Events connections for job progress.

    SRS Section 8.3: Real-time progress via SSE streaming.
    """

    def __init__(self) -> None:
        self._listeners: dict[str, list[asyncio.Queue]] = {}

    def add_listener(self, job_id: str) -> asyncio.Queue:
        """Register a new SSE listener for a job."""
        queue: asyncio.Queue = asyncio.Queue()
        if job_id not in self._listeners:
            self._listeners[job_id] = []
        self._listeners[job_id].append(queue)
        return queue

    def remove_listener(self, job_id: str, queue: asyncio.Queue) -> None:
        """Remove an SSE listener."""
        if job_id in self._listeners:
            try:
                self._listeners[job_id].remove(queue)
            except ValueError:
                pass
            if not self._listeners[job_id]:
                del self._listeners[job_id]

    async def broadcast(self, job_id: str, event: str, data: dict[str, Any]) -> None:
        """Send an SSE event to all listeners of a job."""
        if job_id not in self._listeners:
            return

        message = f"event: {event}\ndata: {json.dumps(data)}\n\n"
        dead_queues = []
        for queue in self._listeners[job_id]:
            try:
                await queue.put(message)
            except Exception:
                dead_queues.append(queue)

        for q in dead_queues:
            self.remove_listener(job_id, q)


# =============================================================================
# Job Scheduler
# =============================================================================

class JobScheduler:
    """Async job scheduler for long-running operations.

    Features:
    - SQLite-backed job persistence (survives restarts)
    - Configurable max concurrent jobs (default 2)
    - SSE real-time progress streaming
    - Automatic cleanup of old job records
    - Graceful shutdown with in-progress job tracking

    SRS References:
    - Section 8:    Async Operations and Long-Running Tasks
    - FR-API-001:   Create/delete index → 202 + job_id
    - FR-API-006:   Export/import → 202 + job_id
    - NFR-10:       Graceful shutdown with checkpointed jobs
    """

    def __init__(
        self,
        db_path: str | Path,
        max_concurrent: int = DEFAULT_MAX_CONCURRENT,
        retention_days: int = DEFAULT_RETENTION_DAYS,
    ) -> None:
        self._store = JobStore(db_path)
        self._sse = SSEManager()
        self._max_concurrent = max_concurrent
        self._retention_days = retention_days
        self._running_jobs: dict[str, asyncio.Task] = {}
        self._job_fns: dict[str, Callable[..., Awaitable[dict[str, Any]]]] = {}
        self._lock = asyncio.Lock()
        self._cleanup_task: asyncio.Task | None = None
        logger.info(
            "[JOB-SCHED] Initialized  max_concurrent=%d  retention=%d days",
            max_concurrent, retention_days,
        )

    # -----------------------------------------------------------------
    # Job Function Registration
    # -----------------------------------------------------------------

    def register_job_type(
        self,
        job_type: str,
        fn: Callable[..., Awaitable[dict[str, Any]]],
    ) -> None:
        """Register an async function for a job type.

        The function receives (job_id, payload, progress_callback) and must
        return a result dict.
        """
        self._job_fns[job_type] = fn
        logger.info("[JOB-SCHED] Registered job type: %s", job_type)

    # -----------------------------------------------------------------
    # Create Job
    # -----------------------------------------------------------------

    async def create_job(
        self,
        job_type: str,
        payload: dict[str, Any] | None = None,
        auto_start: bool = True,
    ) -> Job:
        """Create a new job and optionally start it.

        SRS Section 8.2: Returns job with 202 Accepted semantics.

        Parameters
        ----------
        job_type : str
            Type of job (e.g., "index_create", "export").
        payload : dict
            Job-specific parameters passed to the executor function.
        auto_start : bool
            If True, the job is queued for immediate execution.

        Returns
        -------
        Job
            The created job with its unique job_id.
        """
        job_id = str(uuid.uuid4())
        job = Job(
            job_id=job_id,
            job_type=job_type,
            status=JobStatus.QUEUED if auto_start else JobStatus.PENDING,
            payload=payload or {},
        )
        self._store.create(job)

        if auto_start:
            await self._queue_job(job)

        logger.info(
            "[JOB-SCHED] Created job %s type=%s auto_start=%s",
            job_id, job_type, auto_start,
        )
        return job

    async def _queue_job(self, job: Job) -> None:
        """Add a job to the execution queue."""
        async with self._lock:
            if job.job_type not in self._job_fns:
                logger.error("[JOB-SCHED] No executor for job type: %s", job.job_type)
                self._store.fail(job.job_id, {
                    "code": 500,
                    "message": f"No executor registered for job type '{job.job_type}'",
                })
                return

            running_count = sum(
                1 for t in self._running_jobs.values() if not t.done()
            )
            if running_count >= self._max_concurrent:
                logger.info(
                    "[JOB-SCHED] Max concurrent (%d) reached, job %s queued",
                    self._max_concurrent, job.job_id,
                )
                return  # Job stays in QUEUED status

            task = asyncio.create_task(self._execute_job(job))
            self._running_jobs[job.job_id] = task

    async def _execute_job(self, job: Job) -> None:
        """Execute a job in the background."""
        now = datetime.now(timezone.utc).isoformat()
        self._store.update_status(job.job_id, JobStatus.RUNNING)
        self._store._get_conn().execute(
            "UPDATE jobs SET started_at = ? WHERE job_id = ?",
            (now, job.job_id),
        )
        self._store._get_conn().commit()

        # Broadcast start event
        await self._sse.broadcast(job.job_id, "started", {"job_id": job.job_id})

        async def progress_cb(percentage: float, message: str) -> None:
            progress = JobProgress(percentage=percentage, message=message)
            self._store.update_progress(job.job_id, progress)
            await self._sse.broadcast(job.job_id, "progress", progress.to_dict())

        try:
            fn = self._job_fns[job.job_type]
            result = await fn(job.job_id, job.payload, progress_cb)
            self._store.complete(job.job_id, result)
            await self._sse.broadcast(job.job_id, "completed", result)
            logger.info("[JOB-SCHED] Job %s completed successfully", job.job_id)

        except asyncio.CancelledError:
            self._store.update_status(job.job_id, JobStatus.CANCELLED)
            await self._sse.broadcast(job.job_id, "cancelled", {})
            logger.info("[JOB-SCHED] Job %s cancelled", job.job_id)

        except Exception as exc:
            error = {
                "code": 500,
                "message": str(exc),
                "type": type(exc).__name__,
            }
            self._store.fail(job.job_id, error)
            await self._sse.broadcast(job.job_id, "error", error)
            logger.error(
                "[JOB-SCHED] Job %s failed: %s", job.job_id, exc, exc_info=True,
            )

        finally:
            self._running_jobs.pop(job.job_id, None)
            # Check for queued jobs
            await self._process_queue()

    async def _process_queue(self) -> None:
        """Process queued jobs if capacity is available."""
        async with self._lock:
            running_count = sum(
                1 for t in self._running_jobs.values() if not t.done()
            )
            if running_count >= self._max_concurrent:
                return

            queued = self._store.list_jobs(status=JobStatus.QUEUED, limit=1)
            for job in queued:
                if job.job_type in self._job_fns:
                    task = asyncio.create_task(self._execute_job(job))
                    self._running_jobs[job.job_id] = task
                    if sum(1 for t in self._running_jobs.values() if not t.done()) >= self._max_concurrent:
                        break

    # -----------------------------------------------------------------
    # Query Job
    # -----------------------------------------------------------------

    def get_job(self, job_id: str) -> Job | None:
        """Get a job by ID.

        SRS Section 8.3: GET /api/v1/jobs/{job_id}
        """
        return self._store.get(job_id)

    def list_jobs(
        self,
        status: JobStatus | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[Job]:
        """List jobs with optional status filter."""
        return self._store.list_jobs(status=status, limit=limit, offset=offset)

    # -----------------------------------------------------------------
    # SSE Streaming
    # -----------------------------------------------------------------

    def get_sse_queue(self, job_id: str) -> asyncio.Queue:
        """Get an SSE event queue for a job.

        SRS Section 8.3: GET /api/v1/jobs/{job_id}/stream
        """
        return self._sse.add_listener(job_id)

    def remove_sse_queue(self, job_id: str, queue: asyncio.Queue) -> None:
        """Remove an SSE listener."""
        self._sse.remove_listener(job_id, queue)

    # -----------------------------------------------------------------
    # Cleanup
    # -----------------------------------------------------------------

    def cleanup_old_jobs(self) -> int:
        """Delete old completed/failed jobs.

        SRS Section 8.5: Retention period (default 7 days).
        """
        return self._store.cleanup(self._retention_days)

    # -----------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------

    async def start_cleanup_task(self) -> None:
        """Start periodic cleanup of old job records."""
        async def _cleanup_loop() -> None:
            while True:
                await asyncio.sleep(DEFAULT_CLEANUP_INTERVAL_HRS * 3600)
                try:
                    self.cleanup_old_jobs()
                except Exception as exc:
                    logger.error("[JOB-SCHED] Cleanup failed: %s", exc)

        self._cleanup_task = asyncio.create_task(_cleanup_loop())
        logger.info("[JOB-SCHED] Cleanup task started (interval: %dh)", DEFAULT_CLEANUP_INTERVAL_HRS)

    async def shutdown(self) -> None:
        """Graceful shutdown: cancel running jobs.

        SRS NFR-10: In-progress jobs are cancelled cleanly.
        """
        for job_id, task in self._running_jobs.items():
            if not task.done():
                task.cancel()
                logger.info("[JOB-SCHED] Cancelled job %s for shutdown", job_id)

        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        # Wait for running tasks to finish
        if self._running_jobs:
            await asyncio.gather(
                *self._running_jobs.values(),
                return_exceptions=True,
            )

        logger.info("[JOB-SCHED] Shutdown complete")
