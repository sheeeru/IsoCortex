"""
IsoCortex — Common Schema Models
=================================

Shared Pydantic models for error responses, success responses, and pagination.

SRS References:
  - Section 10: Error Handling Specification
  - FR-API-003: Pagination specification
"""

from __future__ import annotations

from typing import Any, Generic, Optional, TypeVar

from pydantic import BaseModel, Field


T = TypeVar("T")


# =============================================================================
# Error Response (SRS Section 10.2)
# =============================================================================

class ErrorResponse(BaseModel):
    """Standard error envelope for all API errors.

    SRS Section 10.2: error, code, detail, request_id.
    """
    error: str = Field(..., description="Human-readable error summary")
    code: int = Field(..., description="HTTP status code")
    detail: str = Field(default="", description="Detailed explanation")
    request_id: str = Field(default="", description="Unique request identifier")


# =============================================================================
# Success Response
# =============================================================================

class SuccessResponse(BaseModel):
    """Generic success response."""
    message: str = Field(default="OK")
    request_id: str = Field(default="")


# =============================================================================
# Pagination (SRS FR-API-003)
# =============================================================================

class PaginationMeta(BaseModel):
    """Offset-based pagination metadata."""
    total_results: int
    page: int
    page_size: int
    total_pages: int


class CursorPaginationMeta(BaseModel):
    """Cursor-based pagination metadata."""
    next_cursor: Optional[str] = None
    has_more: bool = False
    page_size: int


class PaginatedResponse(BaseModel, Generic[T]):
    """Generic paginated response wrapper."""
    request_id: str = ""
    results: list[T] = Field(default_factory=list)
    pagination: PaginationMeta | CursorPaginationMeta | None = None
