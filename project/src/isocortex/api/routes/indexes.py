"""
IsoCortex — Index Management Route Handlers
============================================

FastAPI route handlers for index lifecycle management.

All endpoints require JWT authentication. Long-running operations
(create, delete, export, import) are asynchronous and return
202 Accepted with a job_id for progress tracking.

SRS References:
  - FR-API-001:  Index Management Endpoints
  - FR-API-006:  Export/Import Endpoints
  - Section 7:    Index Format Versioning
  - NFR-09:      Atomic writes
  - NFR-16:      Pre-indexing memory check
  - Section 8:    Async Operations (202 + job_id pattern)

Author : Shaheer Qureshi
Project: IsoCortex
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status

from isocortex.api.middleware.auth import get_current_user
from isocortex.api.schemas.indexes import (
    CreateIndexRequest,
    CreateIndexResponse,
    DeleteIndexResponse,
    ExportRequest,
    ExportResponse,
    ImportRequest,
    ImportResponse,
    IndexDetailResponse,
    IndexInfoResponse,
    IndexListResponse,
    UpdateIndexRequest,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Index Management"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_request_id(request: Request) -> str:
    """Extract request_id from request state (set by RequestIDMiddleware)."""
    return getattr(request.state, "request_id", "unknown")


def _get_index_manager():
    """Lazy-load the IndexManager singleton."""
    from isocortex.engine.indexing.manager import IndexManager
    from isocortex.config import get_config
    config = get_config()
    return IndexManager(config.storage.indices_dir)


def _get_job_scheduler():
    """Lazy-load the JobScheduler singleton."""
    from isocortex.engine.jobs.scheduler import JobScheduler
    from isocortex.config import get_config
    config = get_config()
    db_path = config.storage.data_dir / "jobs.db"
    return JobScheduler(db_path=str(db_path))


# ---------------------------------------------------------------------------
# POST /api/v1/indexes — Create index
# ---------------------------------------------------------------------------

@router.post(
    "",
    response_model=CreateIndexResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Create a new index",
    description=(
        "Create a new HNSW index from specified files/directories. "
        "Returns 202 Accepted with a job_id for tracking progress. "
        "SRS FR-API-001."
    ),
    responses={
        401: {"description": "Authentication required"},
        409: {"description": "Index already exists"},
        422: {"description": "Validation error"},
    },
)
async def create_index(
    body: CreateIndexRequest,
    request: Request,
    _user: dict = Depends(get_current_user),
) -> CreateIndexResponse:
    """Create a new index.

    SRS FR-API-001: POST /api/v1/indexes
    - Returns 202 Accepted + job_id
    - Index creation runs asynchronously via the job scheduler
    - Validates index name uniqueness
    """
    request_id = _get_request_id(request)
    mgr = _get_index_manager()
    scheduler = _get_job_scheduler()

    try:
        mgr.create_index(
            name=body.name,
            description=body.description,
            embedding_model=body.embedding.model,
            embedding_dimension=384,
            hnsw_params=body.hnsw.model_dump(),
            chunk_config=body.embedding.model_dump(),
        )
    except FileExistsError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "error": "INDEX_EXISTS",
                "code": 409,
                "detail": str(exc),
                "request_id": request_id,
            },
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "BAD_REQUEST",
                "code": 400,
                "detail": str(exc),
                "request_id": request_id,
            },
        )

    # Create async job for document ingestion
    job = await scheduler.create_job(
        job_type="index_create",
        payload={
            "index_name": body.name,
            "files": body.files,
            "exclude_patterns": body.exclude_patterns,
            "embedding_model": body.embedding.model,
            "hnsw_params": body.hnsw.model_dump(),
            "chunk_config": body.embedding.model_dump(),
            "user_id": _user.get("sub"),
        },
    )

    logger.info(
        "[INDEX-ROUTES] Index creation started  name=%s  job_id=%s  user=%s  "
        "request_id=%s",
        body.name,
        job.job_id,
        _user.get("sub"),
        request_id,
    )

    return CreateIndexResponse(
        message="Index creation started",
        job_id=job.job_id,
        name=body.name,
    )


# ---------------------------------------------------------------------------
# GET /api/v1/indexes — List all indexes
# ---------------------------------------------------------------------------

@router.get(
    "",
    response_model=IndexListResponse,
    summary="List all indexes",
    description="Return a list of all indexes with summary metadata. SRS FR-API-001.",
    responses={
        401: {"description": "Authentication required"},
    },
)
async def list_indexes(
    request: Request,
    _user: dict = Depends(get_current_user),
) -> IndexListResponse:
    """List all indexes.

    SRS FR-API-001: GET /api/v1/indexes
    """
    request_id = _get_request_id(request)
    mgr = _get_index_manager()

    try:
        indexes = mgr.list_indexes()
    except Exception as exc:
        logger.error(
            "[INDEX-ROUTES] Failed to list indexes: %s  request_id=%s",
            exc,
            request_id,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "INTERNAL_ERROR",
                "code": 500,
                "detail": "Failed to list indexes",
                "request_id": request_id,
            },
        )

    return IndexListResponse(
        indexes=[
            IndexInfoResponse(
                name=idx.name,
                description=idx.description,
                vector_count=idx.vector_count,
                deleted_count=idx.deleted_count,
                created_at=idx.created_at,
                updated_at=idx.updated_at,
                healthy=idx.healthy,
            )
            for idx in indexes
        ],
        total=len(indexes),
    )


# ---------------------------------------------------------------------------
# GET /api/v1/indexes/{name} — Get index details
# ---------------------------------------------------------------------------

@router.get(
    "/{name}",
    response_model=IndexDetailResponse,
    summary="Get index details",
    description="Return detailed statistics for a specific index. SRS FR-API-001.",
    responses={
        401: {"description": "Authentication required"},
        404: {"description": "Index not found"},
    },
)
async def get_index(
    name: str,
    request: Request,
    _user: dict = Depends(get_current_user),
) -> IndexDetailResponse:
    """Get detailed index statistics.

    SRS FR-API-001: GET /api/v1/indexes/{name}
    """
    request_id = _get_request_id(request)
    mgr = _get_index_manager()

    stats = mgr.get_index(name)
    if stats is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": "INDEX_NOT_FOUND",
                "code": 404,
                "detail": f"Index '{name}' not found",
                "request_id": request_id,
            },
        )

    return IndexDetailResponse(
        name=stats.name,
        vector_count=stats.vector_count,
        deleted_count=stats.deleted_count,
        active_count=stats.active_count,
        index_size_mb=round(stats.index_size_mb, 2),
        created_at=stats.created_at,
        updated_at=stats.updated_at,
        embedding_model=stats.embedding_model,
        dimension=stats.dimension,
        hnsw_params=stats.hnsw_params,
        chunk_config=stats.chunk_config,
        format_version=stats.format_version,
        healthy=stats.healthy,
    )


# ---------------------------------------------------------------------------
# PUT /api/v1/indexes/{name} — Update index config
# ---------------------------------------------------------------------------

@router.put(
    "/{name}",
    response_model=IndexDetailResponse,
    summary="Update index configuration",
    description=(
        "Update index metadata (description, HNSW ef_search). "
        "SRS FR-API-001."
    ),
    responses={
        401: {"description": "Authentication required"},
        404: {"description": "Index not found"},
    },
)
async def update_index(
    name: str,
    body: UpdateIndexRequest,
    request: Request,
    _user: dict = Depends(get_current_user),
) -> IndexDetailResponse:
    """Update index configuration.

    SRS FR-API-001: PUT /api/v1/indexes/{name}
    - Only description and hnsw.ef_search are updatable
    """
    request_id = _get_request_id(request)
    mgr = _get_index_manager()

    updates: dict[str, Any] = {}
    if body.description is not None:
        updates["description"] = body.description
    if body.hnsw is not None:
        updates["hnsw_params"] = body.hnsw

    try:
        mgr.update_index(name, updates)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": "INDEX_NOT_FOUND",
                "code": 404,
                "detail": str(exc),
                "request_id": request_id,
            },
        )

    # Re-fetch updated stats
    stats = mgr.get_index(name)
    if stats is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "INTERNAL_ERROR",
                "code": 500,
                "detail": "Index updated but could not be retrieved",
                "request_id": request_id,
            },
        )

    logger.info(
        "[INDEX-ROUTES] Index updated  name=%s  fields=%s  user=%s  request_id=%s",
        name,
        list(updates.keys()),
        _user.get("sub"),
        request_id,
    )

    return IndexDetailResponse(
        name=stats.name,
        vector_count=stats.vector_count,
        deleted_count=stats.deleted_count,
        active_count=stats.active_count,
        index_size_mb=round(stats.index_size_mb, 2),
        created_at=stats.created_at,
        updated_at=stats.updated_at,
        embedding_model=stats.embedding_model,
        dimension=stats.dimension,
        hnsw_params=stats.hnsw_params,
        chunk_config=stats.chunk_config,
        format_version=stats.format_version,
        healthy=stats.healthy,
    )


# ---------------------------------------------------------------------------
# DELETE /api/v1/indexes/{name} — Delete index
# ---------------------------------------------------------------------------

@router.delete(
    "/{name}",
    response_model=DeleteIndexResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Delete an index",
    description=(
        "Delete an index and all its data. Returns 202 Accepted with a job_id. "
        "SRS FR-API-001."
    ),
    responses={
        401: {"description": "Authentication required"},
        404: {"description": "Index not found"},
    },
)
async def delete_index(
    name: str,
    request: Request,
    _user: dict = Depends(get_current_user),
) -> DeleteIndexResponse:
    """Delete an index.

    SRS FR-API-001: DELETE /api/v1/indexes/{name}
    - Returns 202 Accepted + job_id
    - Actual deletion runs asynchronously
    """
    request_id = _get_request_id(request)
    mgr = _get_index_manager()
    scheduler = _get_job_scheduler()

    # Verify index exists before creating the job
    stats = mgr.get_index(name)
    if stats is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": "INDEX_NOT_FOUND",
                "code": 404,
                "detail": f"Index '{name}' not found",
                "request_id": request_id,
            },
        )

    job = await scheduler.create_job(
        job_type="index_delete",
        payload={
            "index_name": name,
            "user_id": _user.get("sub"),
        },
    )

    logger.info(
        "[INDEX-ROUTES] Index deletion started  name=%s  job_id=%s  user=%s  "
        "request_id=%s",
        name,
        job.job_id,
        _user.get("sub"),
        request_id,
    )

    return DeleteIndexResponse(
        message="Index deletion started",
        job_id=job.job_id,
        name=name,
    )


# ---------------------------------------------------------------------------
# POST /api/v1/indexes/{name}/export — Export index
# ---------------------------------------------------------------------------

@router.post(
    "/{name}/export",
    response_model=ExportResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Export an index",
    description=(
        "Export an index as a .isocortex archive. "
        "Returns 202 Accepted with a job_id. SRS FR-API-006."
    ),
    responses={
        401: {"description": "Authentication required"},
        404: {"description": "Index not found"},
    },
)
async def export_index(
    name: str,
    body: ExportRequest,
    request: Request,
    _user: dict = Depends(get_current_user),
) -> ExportResponse:
    """Export an index as a .isocortex archive.

    SRS FR-API-006: POST /api/v1/indexes/{name}/export
    - Returns 202 Accepted + job_id
    - Archive includes SHA-256 checksum
    """
    request_id = _get_request_id(request)
    mgr = _get_index_manager()
    scheduler = _get_job_scheduler()

    # Verify index exists
    stats = mgr.get_index(name)
    if stats is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": "INDEX_NOT_FOUND",
                "code": 404,
                "detail": f"Index '{name}' not found",
                "request_id": request_id,
            },
        )

    output_path = body.output_path or str(
        mgr.indices_dir / f"{name}{'.isocortex'}"
    )

    job = await scheduler.create_job(
        job_type="export",
        payload={
            "index_name": name,
            "output_path": output_path,
            "user_id": _user.get("sub"),
        },
    )

    logger.info(
        "[INDEX-ROUTES] Export started  name=%s  output=%s  job_id=%s  user=%s  "
        "request_id=%s",
        name,
        output_path,
        job.job_id,
        _user.get("sub"),
        request_id,
    )

    return ExportResponse(
        message="Export started",
        job_id=job.job_id,
    )


# ---------------------------------------------------------------------------
# POST /api/v1/indexes/import — Import index
# ---------------------------------------------------------------------------

@router.post(
    "/import",
    response_model=ImportResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Import an index",
    description=(
        "Import an index from a .isocortex archive. "
        "Returns 202 Accepted with a job_id. SRS FR-API-006."
    ),
    responses={
        401: {"description": "Authentication required"},
        404: {"description": "Archive file not found"},
        422: {"description": "Invalid archive format"},
    },
)
async def import_index(
    body: ImportRequest,
    request: Request,
    _user: dict = Depends(get_current_user),
) -> ImportResponse:
    """Import an index from a .isocortex archive.

    SRS FR-API-006: POST /api/v1/indexes/import
    - Returns 202 Accepted + job_id
    - Performs version negotiation (SRS Section 7)
    """
    request_id = _get_request_id(request)
    scheduler = _get_job_scheduler()

    # Validate archive path exists
    archive_path = Path(body.archive_path)
    if not archive_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": "ARCHIVE_NOT_FOUND",
                "code": 404,
                "detail": f"Archive file not found: {body.archive_path}",
                "request_id": request_id,
            },
        )

    job = await scheduler.create_job(
        job_type="import",
        payload={
            "archive_path": str(archive_path.resolve()),
            "index_name": body.name,
            "user_id": _user.get("sub"),
        },
    )

    logger.info(
        "[INDEX-ROUTES] Import started  archive=%s  name=%s  job_id=%s  user=%s  "
        "request_id=%s",
        body.archive_path,
        body.name or "(auto)",
        job.job_id,
        _user.get("sub"),
        request_id,
    )

    return ImportResponse(
        message="Import started",
        job_id=job.job_id,
    )
