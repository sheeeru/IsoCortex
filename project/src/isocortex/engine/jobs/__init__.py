"""
IsoCortex engine/jobs — Async Job Scheduler
============================================

Manages long-running operations (index creation, export/import, compaction)
as background jobs with SQLite-backed state, progress tracking, SSE streaming,
and automatic cleanup.

SRS References:
  - Section 8:  Async Operations and Long-Running Tasks
  - FR-API-001: Create/delete index returns 202 + job_id
  - FR-API-006: Export/import returns 202 + job_id
  - NFR-10:    Graceful shutdown with checkpointed jobs

Author : Shaheer Qureshi
Project: IsoCortex
"""

from isocortex.engine.jobs.scheduler import JobScheduler, Job, JobStatus

__all__ = ["JobScheduler", "Job", "JobStatus"]
