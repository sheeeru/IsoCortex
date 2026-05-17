"""
IsoCortex — Index Schema Models
================================

Pydantic models for index management endpoints.

SRS References:
  - FR-API-001: Index Management Endpoints
  - FR-API-006: Export/Import Endpoints
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


# =============================================================================
# Create Index (SRS FR-API-001: POST /api/v1/indexes)
# =============================================================================

class HNSWParams(BaseModel):
    """HNSW index parameters (SRS FR-IDX-001)."""
    M: int = Field(default=16, ge=4, le=128, description="Bidirectional links per layer")
    ef_construction: int = Field(default=200, ge=50, le=2000)
    ef_search: int = Field(default=50, ge=10, le=500)
    metric: str = Field(default="cosine", pattern=r"^(cosine|l2|ip)$")


class EmbeddingParams(BaseModel):
    """Embedding parameters."""
    model: str = Field(
        default="all-MiniLM-L6-v2",
        description="Embedding model identifier",
    )
    chunk_size: int = Field(default=512, ge=64, le=4096, description="Tokens per chunk")
    chunk_overlap: int = Field(default=50, ge=0, le=500, description="Token overlap")


class CreateIndexRequest(BaseModel):
    """Create index request body.

    SRS FR-API-001: POST /api/v1/indexes
    Returns 202 Accepted + job_id.
    """
    name: str = Field(
        ..., min_length=1, max_length=100,
        pattern=r"^[a-zA-Z0-9_-]+$",
        description="Index name (unique, alphanumeric + hyphens/underscores)",
    )
    description: str = Field(default="", max_length=500)
    files: list[str] = Field(
        default_factory=list,
        description="File/directory paths to index",
    )
    exclude_patterns: list[str] = Field(
        default_factory=list,
        description="Glob patterns to exclude",
    )
    embedding: EmbeddingParams = Field(default_factory=EmbeddingParams)
    hnsw: HNSWParams = Field(default_factory=HNSWParams)


class UpdateIndexRequest(BaseModel):
    """Update index configuration (SRS FR-API-001: PUT /api/v1/indexes/{name})."""
    description: Optional[str] = Field(default=None, max_length=500)
    hnsw: Optional[dict[str, Any]] = Field(
        default=None,
        description="Updatable HNSW params (ef_search only)",
    )


# =============================================================================
# Index Responses
# =============================================================================

class IndexInfoResponse(BaseModel):
    """Lightweight index info for listing."""
    name: str
    description: str = ""
    vector_count: int = 0
    deleted_count: int = 0
    created_at: str = ""
    updated_at: str = ""
    healthy: bool = True


class IndexDetailResponse(BaseModel):
    """Detailed index stats (SRS FR-API-001: GET /api/v1/indexes/{name})."""
    name: str
    vector_count: int = 0
    deleted_count: int = 0
    active_count: int = 0
    index_size_mb: float = 0.0
    created_at: str = ""
    updated_at: str = ""
    embedding_model: str = ""
    dimension: int = 384
    hnsw_params: dict[str, Any] = Field(default_factory=dict)
    chunk_config: dict[str, Any] = Field(default_factory=dict)
    format_version: int = 1
    healthy: bool = True


class IndexListResponse(BaseModel):
    """List all indexes response."""
    indexes: list[IndexInfoResponse] = Field(default_factory=list)
    total: int = 0


class CreateIndexResponse(BaseModel):
    """Response for index creation (202 Accepted)."""
    message: str = "Index creation started"
    job_id: str
    name: str


class DeleteIndexResponse(BaseModel):
    """Response for index deletion (202 Accepted)."""
    message: str = "Index deletion started"
    job_id: str
    name: str


# =============================================================================
# Export/Import (SRS FR-API-006)
# =============================================================================

class ExportRequest(BaseModel):
    """Export index request."""
    output_path: Optional[str] = Field(
        default=None,
        description="Custom output path (optional)",
    )


class ImportRequest(BaseModel):
    """Import index request."""
    archive_path: str = Field(..., description="Path to .isocortex archive")
    name: Optional[str] = Field(
        default=None,
        description="Custom index name (optional, uses archive name by default)",
    )


class ExportResponse(BaseModel):
    """Export response (202 Accepted)."""
    message: str = "Export started"
    job_id: str


class ImportResponse(BaseModel):
    """Import response (202 Accepted)."""
    message: str = "Import started"
    job_id: str
