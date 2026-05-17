"""
IsoCortex — Job Schema Models
==============================

Pydantic models for job status endpoints.

SRS References:
  - Section 8: Async Operations and Long-Running Tasks
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class JobProgressResponse(BaseModel):
    """Job progress information."""
    percentage: float = 0.0
    message: str = ""


class JobResponse(BaseModel):
    """Job status response.

    SRS Section 8.3: GET /api/v1/jobs/{job_id}
    """
    job_id: str
    type: str
    status: str
    progress: JobProgressResponse = Field(default_factory=JobProgressResponse)
    created_at: str = ""
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    failed_at: Optional[str] = None
    estimated_completion: Optional[str] = None
    result: Optional[dict[str, Any]] = None
    error: Optional[dict[str, Any]] = None


class JobListResponse(BaseModel):
    """List jobs response."""
    jobs: list[JobResponse] = Field(default_factory=list)
    total: int = 0
