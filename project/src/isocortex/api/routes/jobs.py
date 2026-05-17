"""
IsoCortex — Job Status Route Handlers
======================================

FastAPI route handlers for monitoring background job status.

Provides polling-based status checks and Server-Sent Events (SSE)
for real-time progress streaming.

SRS References:
  - Section 8:    Async Operations and Long-Running Tasks
  - Section 8.2:  Job lifecycle (202 → poll or SSE)
  - Section 8.3:  Job status endpoint response format
  - Section 8.4:  Job types (index_create, export, import, etc.)
  - Section 8.5:  Retention period (default 7 days)
  - NFR-10:      Graceful shutdown with checkpointed jobs

Author : Shaheer Qureshi
Project: IsoCortex
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse

from isocortex.api.middleware.auth import get_current_user
from isocortex.api.schemas.jobs import JobListResponse, JobProgressResponse, JobResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Jobs"])

# ---------------------------------------------------------------------------
# SSE keep-alive interval (seconds)
# ---------------------------------------------------------------------------
SSE_KEEPALIVE_INTERVAL: int = 15


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_request_id(request: Request) -> str:
    """Extract request_id from request state (set by RequestIDMiddleware)."""
    return getattr(request.state, "request_id", "unknown")


def _get_job_scheduler():
    """Lazy-load the JobScheduler singleton."""
    from isocortex.engine.jobs.scheduler import JobScheduler
    from isocortex.config import get_config
    config = get_config()
    db_path = config.storage.data_dir / "jobs.db"
    return JobScheduler(db_path=str(db_path))


def _job_to_response(job: Any) -> JobResponse:
    """Convert a Job dataclass to the JobResponse schema."""
    job_dict = job.to_dict() if hasattr(job, "to_dict") else {}
    return JobResponse(
        job_id=job_dict.get("job_id", ""),
        type=job_dict.get("type", ""),
        status=job_dict.get("status", ""),
        progress=JobProgressResponse(
            percentage=job_dict.get("progress", {}).get("percentage", 0.0),
            message=job_dict.get("progress", {}).get("message", ""),
        ),
        created_at=job_dict.get("created_at", ""),
        started_at=job_dict.get("started_at"),
        completed_at=job_dict.get("completed_at"),
        failed_at=job_dict.get("failed_at"),
        estimated_completion=job_dict.get("estimated_completion"),
        result=job_dict.get("result"),
        error=job_dict.get("error"),
    )


# ---------------------------------------------------------------------------
# GET /api/v1/jobs/{job_id} — Get job status
# ---------------------------------------------------------------------------

@router.get(
    "/{job_id}",
    response_model=JobResponse,
    summary="Get job status",
    description=(
        "Return the current status of a background job. "
        "SRS Section 8.3."
    ),
    responses={
        401: {"description": "Authentication required"},
        404: {"description": "Job not found"},
    },
)
async def get_job_status(
    job_id: str,
    request: Request,
    _user: dict = Depends(get_current_user),
) -> JobResponse:
    """Get the status of a specific job.

    SRS Section 8.3: GET /api/v1/jobs/{job_id}
    - Returns full job status including progress, timestamps, result/error
    """
    request_id = _get_request_id(request)
    scheduler = _get_job_scheduler()

    job = scheduler.get_job(job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": "JOB_NOT_FOUND",
                "code": 404,
                "detail": f"Job '{job_id}' not found",
                "request_id": request_id,
            },
        )

    logger.debug(
        "[JOB-ROUTES] Status poll  job_id=%s  status=%s  user=%s  request_id=%s",
        job_id,
        job.status.value,
        _user.get("sub"),
        request_id,
    )

    return _job_to_response(job)


# ---------------------------------------------------------------------------
# GET /api/v1/jobs/{job_id}/stream — SSE progress stream
# ---------------------------------------------------------------------------

@router.get(
    "/{job_id}/stream",
    summary="Stream job progress via SSE",
    description=(
        "Subscribe to real-time job progress via Server-Sent Events. "
        "Connection stays open until the job completes or fails. "
        "SRS Section 8.3."
    ),
    responses={
        401: {"description": "Authentication required"},
        404: {"description": "Job not found"},
        200: {
            "description": "SSE stream (text/event-stream)",
            "content": {"text/event-stream": {}},
        },
    },
)
async def stream_job_progress(
    job_id: str,
    request: Request,
    _user: dict = Depends(get_current_user),
) -> StreamingResponse:
    """Stream job progress via Server-Sent Events.

    SRS Section 8.3: GET /api/v1/jobs/{job_id}/stream
    - Real-time progress updates via SSE
    - Automatic keep-alive every 15 seconds
    - Connection closes when job reaches terminal state
    """
    request_id = _get_request_id(request)
    scheduler = _get_job_scheduler()

    # Verify job exists
    job = scheduler.get_job(job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": "JOB_NOT_FOUND",
                "code": 404,
                "detail": f"Job '{job_id}' not found",
                "request_id": request_id,
            },
        )

    # If job is already in a terminal state, return final status and close
    from isocortex.engine.jobs.scheduler import JobStatus
    if job.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
        logger.info(
            "[JOB-ROUTES] SSE requested for terminal job  job_id=%s  status=%s  "
            "request_id=%s",
            job_id,
            job.status.value,
            request_id,
        )

        async def _terminal_stream():
            job_data = job.to_dict()
            event_type = job.status.value
            yield f"event: {event_type}\ndata: {json.dumps(job_data)}\n\n"
            yield f"event: done\ndata: {{\"job_id\": \"{job_id}\"}}\n\n"

        return StreamingResponse(
            _terminal_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # Register SSE listener
    queue: asyncio.Queue = scheduler.get_sse_queue(job_id)

    # SRS Section 8.3: Handle Last-Event-ID for reconnection
    last_event_id = request.headers.get("Last-Event-ID")
    if last_event_id:
        logger.info(
            "[JOB-ROUTES] SSE reconnect  job_id=%s  last_event_id=%s  request_id=%s",
            job_id, last_event_id, request_id,
        )
        # Replay the current job state as a progress event
        current = scheduler.get_job(job_id)
        if current:
            async def _reconnect_stream():
                yield f"event: progress\ndata: {json.dumps(current.to_dict())}\nid: reconnect\n\n"
                if current.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
                    event_type = current.status.value
                    yield f"event: {event_type}\ndata: {json.dumps(current.to_dict())}\nid: reconnect\n\n"
                    yield f"event: done\ndata: {{\"job_id\": \"{job_id}\"}}\nid: reconnect\n\n"

            return StreamingResponse(
                _reconnect_stream(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
            )

    logger.info(
        "[JOB-ROUTES] SSE stream opened  job_id=%s  user=%s  request_id=%s",
        job_id,
        _user.get("sub"),
        request_id,
    )

    async def _event_generator():
        """Generate SSE events from the job queue."""
        try:
            while True:
                try:
                    # Wait for events with keep-alive timeout
                    msg = await asyncio.wait_for(queue.get(), timeout=SSE_KEEPALIVE_INTERVAL)
                    yield msg

                    # Check if job is done (done, completed, failed, cancelled events)
                    if ":done\n" in msg or ":completed\n" in msg or ":failed\n" in msg or ":cancelled\n" in msg:
                        logger.info(
                            "[JOB-ROUTES] SSE stream closing (job done)  job_id=%s",
                            job_id,
                        )
                        break
                except asyncio.TimeoutError:
                    # Send keep-alive comment to prevent connection timeout
                    yield f": keepalive {job_id}\n\n"

                    # Check if job reached terminal state while we were waiting
                    current_job = scheduler.get_job(job_id)
                    if current_job is None:
                        break
                    if current_job.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
                        final_data = current_job.to_dict()
                        event_type = current_job.status.value
                        yield f"event: {event_type}\ndata: {json.dumps(final_data)}\n\n"
                        yield f"event: done\ndata: {{\"job_id\": \"{job_id}\"}}\n\n"
                        break

        except asyncio.CancelledError:
            logger.debug(
                "[JOB-ROUTES] SSE stream cancelled by client  job_id=%s",
                job_id,
            )
        finally:
            scheduler.remove_sse_queue(job_id, queue)

    return StreamingResponse(
        _event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# GET /api/v1/jobs — List jobs
# ---------------------------------------------------------------------------

@router.get(
    "",
    response_model=JobListResponse,
    summary="List all jobs",
    description="Return a paginated list of all jobs. SRS Section 8.3.",
    responses={
        401: {"description": "Authentication required"},
    },
)
async def list_jobs(
    request: Request,
    status_filter: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    _user: dict = Depends(get_current_user),
) -> JobListResponse:
    """List all jobs with optional status filter.

    SRS Section 8.3: GET /api/v1/jobs
    - Supports filtering by status
    - Paginated with limit/offset
    """
    request_id = _get_request_id(request)
    scheduler = _get_job_scheduler()

    # Clamp pagination params
    limit = max(1, min(limit, 100))
    offset = max(0, offset)

    # Parse status filter
    parsed_status = None
    if status_filter:
        from isocortex.engine.jobs.scheduler import JobStatus
        try:
            parsed_status = JobStatus(status_filter)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": "BAD_REQUEST",
                    "code": 400,
                    "detail": f"Invalid job status filter: '{status_filter}'. "
                              f"Valid values: pending, queued, running, completed, "
                              f"failed, cancelled",
                    "request_id": request_id,
                },
            )

    # Fetch one extra to determine has_more
    jobs = scheduler.list_jobs(
        status=parsed_status,
        limit=limit + 1,
        offset=offset,
    )

    has_more = len(jobs) > limit
    if has_more:
        jobs = jobs[:limit]

    logger.debug(
        "[JOB-ROUTES] List jobs  status=%s  limit=%d  offset=%d  returned=%d  "
        "user=%s  request_id=%s",
        status_filter,
        limit,
        offset,
        len(jobs),
        _user.get("sub"),
        request_id,
    )

    return JobListResponse(
        jobs=[_job_to_response(j) for j in jobs],
        total=len(jobs),
    )
