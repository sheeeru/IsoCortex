"""
IsoCortex — Search Schema Models
=================================

Pydantic models for search endpoints.

SRS References:
  - FR-API-002: Search Endpoints
  - FR-API-003: Pagination Specification
  - FR-API-005: Batch Search Specification
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


# =============================================================================
# Search Filters
# =============================================================================

class SearchFilter(BaseModel):
    """Metadata filters for search queries."""
    file_extension: Optional[list[str]] = Field(
        default=None,
        description="Filter by file extensions, e.g. ['.pdf', '.md']",
    )
    source: Optional[str] = Field(
        default=None,
        description="Filter by source file path substring",
    )
    min_score: Optional[float] = Field(
        default=None,
        ge=0.0, le=1.0,
        description="Minimum similarity score (0-1)",
    )


# =============================================================================
# Single Search (SRS FR-API-002)
# =============================================================================

class SearchRequest(BaseModel):
    """Single search request body.

    SRS FR-API-002: POST /api/v1/indexes/{name}/search
    """
    query: str = Field(
        ..., min_length=3, max_length=1000,
        description="Natural language query text",
    )
    k: int = Field(default=5, ge=1, le=100, description="Number of results to retrieve")
    page: int = Field(default=1, ge=1, description="Page number (offset-based)")
    page_size: int = Field(default=10, ge=1, le=100, description="Results per page")
    filters: Optional[SearchFilter] = Field(default=None)
    cursor: Optional[str] = Field(default=None, description="Cursor for cursor-based pagination")
    pagination: str = Field(
        default="offset",
        pattern=r"^(offset|cursor)$",
        description="Pagination mode",
    )


class SearchResultItem(BaseModel):
    """A single search result."""
    id: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    score: float
    rank: int


class SearchResponse(BaseModel):
    """Single search response.

    SRS FR-API-002: JSON response format.
    """
    request_id: str = ""
    results: list[SearchResultItem] = Field(default_factory=list)
    pagination: dict[str, Any] = Field(default_factory=dict)
    query: str = ""
    latency_ms: float = 0.0


# =============================================================================
# Batch Search (SRS FR-API-005)
# =============================================================================

class BatchQueryItem(BaseModel):
    """A single query within a batch search request."""
    query: str = Field(..., min_length=3, max_length=1000)
    k: int = Field(default=5, ge=1, le=100)
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=10, ge=1, le=100)
    filters: Optional[SearchFilter] = Field(default=None)


class BatchSearchRequest(BaseModel):
    """Batch search request body.

    SRS FR-API-005: POST /api/v1/indexes/{name}/search/batch
    Max 50 queries per batch.
    """
    queries: list[BatchQueryItem] = Field(
        ...,
        min_length=1,
        max_length=50,
        description="List of queries to execute",
    )
    timeout: int = Field(default=30, ge=1, le=120, description="Per-query timeout in seconds")


class BatchQueryResultItem(BaseModel):
    """Result of a single query in a batch."""
    query: str
    status: str = Field(default="success", pattern=r"^(success|error)$")
    results: Optional[list[SearchResultItem]] = None
    pagination: Optional[dict[str, Any]] = None
    error: Optional[dict[str, Any]] = None


class BatchSearchResponse(BaseModel):
    """Batch search response.

    SRS FR-API-005: 200 if all succeed, 207 Multi-Status if some fail.
    """
    request_id: str = ""
    results: list[BatchQueryResultItem] = Field(default_factory=list)
    total_queries: int = 0
    successful: int = 0
    failed: int = 0
    latency_ms: float = 0.0
