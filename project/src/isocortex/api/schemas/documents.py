"""
IsoCortex — Document Schema Models
===================================

Pydantic models for document management endpoints.

SRS References:
  - FR-API-004: Document Endpoints
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class AddDocumentsRequest(BaseModel):
    """Add documents to an existing index.

    SRS FR-API-004: POST /api/v1/indexes/{name}/documents
    """
    files: list[str] = Field(
        ...,
        min_length=1,
        description="File/directory paths to add",
    )
    exclude_patterns: list[str] = Field(
        default_factory=list,
        description="Glob patterns to exclude",
    )


class DocumentResponse(BaseModel):
    """Single document chunk."""
    id: str
    vector_index: int
    deleted: bool = False
    source_file: str = ""
    file_extension: str = ""
    page: Optional[int] = None
    section: Optional[str] = None
    chunk_index: int = 0
    text_preview: str = ""
    token_count: int = 0
    created_at: str = ""


class DocumentListResponse(BaseModel):
    """Paginated document list."""
    request_id: str = ""
    documents: list[DocumentResponse] = Field(default_factory=list)
    pagination: dict[str, Any] = Field(default_factory=dict)
