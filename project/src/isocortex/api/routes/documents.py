"""
IsoCortex — Document Route Handlers
====================================

FastAPI route handlers for document management within indexes.

Provides endpoints for listing, retrieving, adding, and soft-deleting
documents (chunks) within a specific index.

SRS References:
  - FR-API-004:  Document Endpoints
  - FR-IDX-002:  Soft Delete (tombstone pattern)
  - FR-IDX-003:  Incremental updates
  - FR-API-003:  Pagination Specification

Author : Shaheer Qureshi
Project: IsoCortex
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status

from isocortex.api.middleware.auth import get_current_user
from isocortex.api.schemas.documents import (
    AddDocumentsRequest,
    DocumentListResponse,
    DocumentResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Documents"])


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


def _meta_to_document(meta: dict[str, Any], vector_index: int) -> DocumentResponse:
    """Convert raw metadata dict to DocumentResponse schema."""
    text_preview = meta.get("text_preview", "")
    if text_preview and len(text_preview) > 500:
        text_preview = text_preview[:500] + "..."
    return DocumentResponse(
        id=meta.get("id", f"vec-{vector_index}"),
        vector_index=vector_index,
        deleted=meta.get("deleted", False),
        source_file=meta.get("source_file", ""),
        file_extension=meta.get("file_extension", ""),
        page=meta.get("page"),
        section=meta.get("section"),
        chunk_index=meta.get("chunk_index", 0),
        text_preview=text_preview,
        token_count=meta.get("token_count", 0),
        created_at=meta.get("created_at", ""),
    )


# ---------------------------------------------------------------------------
# GET /api/v1/indexes/{name}/documents — List documents (paginated)
# ---------------------------------------------------------------------------

@router.get(
    "",
    response_model=DocumentListResponse,
    summary="List documents in an index",
    description=(
        "Return a paginated list of document chunks in the specified index. "
        "SRS FR-API-004."
    ),
    responses={
        401: {"description": "Authentication required"},
        404: {"description": "Index not found"},
    },
)
async def list_documents(
    name: str,
    request: Request,
    page: int = 1,
    page_size: int = 50,
    _user: dict = Depends(get_current_user),
) -> DocumentListResponse:
    """List documents (chunks) in an index.

    SRS FR-API-004: GET /api/v1/indexes/{name}/documents
    - Offset-based pagination
    - Returns document metadata (never raw vectors)
    """
    request_id = _get_request_id(request)
    mgr = _get_index_manager()

    # Verify index exists
    info = mgr._read_index_info(name)
    if info is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": "INDEX_NOT_FOUND",
                "code": 404,
                "detail": f"Index '{name}' not found",
                "request_id": request_id,
            },
        )

    # Load index metadata
    try:
        inmem = mgr.ensure_loaded(name)
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": "INDEX_NOT_FOUND",
                "code": 404,
                "detail": f"Index '{name}' not found",
                "request_id": request_id,
            },
        )

    # Clamp pagination params
    page = max(1, page)
    page_size = max(1, min(page_size, 100))
    offset = (page - 1) * page_size

    # Read metadata
    inmem.lock.acquire_read()
    try:
        all_metadata = list(inmem.metadata)
    finally:
        inmem.lock.release_read()

    total = len(all_metadata)
    page_metadata = all_metadata[offset:offset + page_size]
    total_pages = max(1, (total + page_size - 1) // page_size) if total > 0 else 1

    documents = [
        _meta_to_document(meta, i + offset)
        for i, meta in enumerate(page_metadata)
    ]

    logger.info(
        "[DOC-ROUTES] Listed documents  index=%s  page=%d  page_size=%d  "
        "total=%d  user=%s  request_id=%s",
        name,
        page,
        page_size,
        total,
        _user.get("sub"),
        request_id,
    )

    return DocumentListResponse(
        request_id=request_id,
        documents=documents,
        pagination={
            "total_results": total,
            "page": page,
            "page_size": page_size,
            "total_pages": total_pages,
        },
    )


# ---------------------------------------------------------------------------
# GET /api/v1/indexes/{name}/documents/{id} — Get specific document
# ---------------------------------------------------------------------------

@router.get(
    "/{id}",
    response_model=DocumentResponse,
    summary="Get a specific document",
    description="Return metadata for a specific document chunk. SRS FR-API-004.",
    responses={
        401: {"description": "Authentication required"},
        404: {"description": "Index or document not found"},
    },
)
async def get_document(
    name: str,
    id: str,
    request: Request,
    _user: dict = Depends(get_current_user),
) -> DocumentResponse:
    """Get a specific document by ID.

    SRS FR-API-004: GET /api/v1/indexes/{name}/documents/{id}
    """
    request_id = _get_request_id(request)
    mgr = _get_index_manager()

    # Verify index exists
    info = mgr._read_index_info(name)
    if info is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": "INDEX_NOT_FOUND",
                "code": 404,
                "detail": f"Index '{name}' not found",
                "request_id": request_id,
            },
        )

    # Load index and search for document by ID
    try:
        inmem = mgr.ensure_loaded(name)
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": "INDEX_NOT_FOUND",
                "code": 404,
                "detail": f"Index '{name}' not found",
                "request_id": request_id,
            },
        )

    inmem.lock.acquire_read()
    try:
        for idx, meta in enumerate(inmem.metadata):
            if meta.get("id") == id:
                return _meta_to_document(meta, idx)
    finally:
        inmem.lock.release_read()

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail={
            "error": "DOCUMENT_NOT_FOUND",
            "code": 404,
            "detail": f"Document '{id}' not found in index '{name}'",
            "request_id": request_id,
        },
    )


# ---------------------------------------------------------------------------
# POST /api/v1/indexes/{name}/documents — Add documents
# ---------------------------------------------------------------------------

@router.post(
    "",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Add documents to an index",
    description=(
        "Add documents from specified files/directories to an existing index. "
        "Returns 202 Accepted with a job_id. SRS FR-API-004."
    ),
    responses={
        401: {"description": "Authentication required"},
        404: {"description": "Index not found"},
        422: {"description": "Validation error"},
    },
)
async def add_documents(
    name: str,
    body: AddDocumentsRequest,
    request: Request,
    _user: dict = Depends(get_current_user),
) -> dict[str, Any]:
    """Add documents to an existing index.

    SRS FR-API-004: POST /api/v1/indexes/{name}/documents
    - Returns 202 Accepted + job_id
    - Incremental addition without full rebuild (FR-IDX-003)
    """
    request_id = _get_request_id(request)
    mgr = _get_index_manager()
    scheduler = _get_job_scheduler()

    # Verify index exists
    info = mgr._read_index_info(name)
    if info is None:
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
        job_type="index_update",
        payload={
            "index_name": name,
            "files": body.files,
            "exclude_patterns": body.exclude_patterns,
            "user_id": _user.get("sub"),
        },
    )

    logger.info(
        "[DOC-ROUTES] Documents add started  index=%s  files=%d  job_id=%s  "
        "user=%s  request_id=%s",
        name,
        len(body.files),
        job.job_id,
        _user.get("sub"),
        request_id,
    )

    return {
        "message": "Document ingestion started",
        "job_id": job.job_id,
        "index": name,
        "files_requested": len(body.files),
        "request_id": request_id,
    }


# ---------------------------------------------------------------------------
# DELETE /api/v1/indexes/{name}/documents/{id} — Soft-delete document
# ---------------------------------------------------------------------------

@router.delete(
    "/{id}",
    summary="Soft-delete a document",
    description=(
        "Mark a document chunk as deleted (tombstone pattern). "
        "The vector is not removed from the index graph but will be "
        "excluded from search results. SRS FR-IDX-002."
    ),
    responses={
        401: {"description": "Authentication required"},
        404: {"description": "Index or document not found"},
    },
)
async def delete_document(
    name: str,
    id: str,
    request: Request,
    _user: dict = Depends(get_current_user),
) -> dict[str, Any]:
    """Soft-delete a document chunk.

    SRS FR-IDX-002: Tombstone pattern
    - Instant deletion (no graph rebuild)
    - Compaction runs when tombstoned vectors exceed threshold (default 10%)
    """
    request_id = _get_request_id(request)
    mgr = _get_index_manager()

    # Verify index exists
    info = mgr._read_index_info(name)
    if info is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": "INDEX_NOT_FOUND",
                "code": 404,
                "detail": f"Index '{name}' not found",
                "request_id": request_id,
            },
        )

    # Find the document's vector index
    try:
        inmem = mgr.ensure_loaded(name)
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": "INDEX_NOT_FOUND",
                "code": 404,
                "detail": f"Index '{name}' not found",
                "request_id": request_id,
            },
        )

    vector_index = None
    inmem.lock.acquire_read()
    try:
        for idx, meta in enumerate(inmem.metadata):
            if meta.get("id") == id:
                vector_index = idx
                break
    finally:
        inmem.lock.release_read()

    if vector_index is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": "DOCUMENT_NOT_FOUND",
                "code": 404,
                "detail": f"Document '{id}' not found in index '{name}'",
                "request_id": request_id,
            },
        )

    # Perform soft delete via IndexManager
    deleted = mgr.soft_delete_vector(name, vector_index)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "DELETE_FAILED",
                "code": 500,
                "detail": "Failed to soft-delete document",
                "request_id": request_id,
            },
        )

    logger.info(
        "[DOC-ROUTES] Document soft-deleted  index=%s  doc_id=%s  vector_index=%d  "
        "user=%s  request_id=%s",
        name,
        id,
        vector_index,
        _user.get("sub"),
        request_id,
    )

    return {
        "message": f"Document '{id}' soft-deleted",
        "document_id": id,
        "index": name,
        "request_id": request_id,
    }
