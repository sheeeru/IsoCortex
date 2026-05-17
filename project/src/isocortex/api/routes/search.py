"""
IsoCortex — Search Route Handlers
==================================

FastAPI route handlers for semantic search operations.

Provides single and batch semantic search over HNSW indices with
metadata filtering, offset/cursor pagination, and latency tracking.

SRS References:
  - FR-API-002:  Search Endpoints (single + batch)
  - FR-API-003:  Pagination Specification (offset + cursor)
  - FR-API-005:  Batch Search Specification (max 50, 207 Multi-Status)
  - NFR-01:      p95 < 100ms, p99 < 500ms for 1M vectors
  - NFR-15:      10+ concurrent requests, 100+ QPS
  - NFR-17:      Embedding cache hit rate > 80%

Author : Shaheer Qureshi
Project: IsoCortex
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status

from isocortex.api.middleware.auth import get_current_user
from isocortex.api.schemas.search import (
    BatchSearchRequest,
    BatchSearchResponse,
    BatchQueryResultItem,
    SearchRequest,
    SearchResponse,
    SearchResultItem,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Search"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_request_id(request: Request) -> str:
    """Extract request_id from request state (set by RequestIDMiddleware)."""
    return getattr(request.state, "request_id", "unknown")


def _get_search_engine(index_name: str):
    """Build a SearchEngine instance wired to the given index.

    Lazily loads the index via IndexManager, then creates a SearchEngine
    with the appropriate callables.
    """
    from isocortex.engine.indexing.manager import IndexManager
    from isocortex.core.search.engine import SearchEngine
    from isocortex.config import get_config
    from isocortex.core.embedding.embedder import EmbeddingEngine

    config = get_config()
    mgr = IndexManager(config.storage.indices_dir)

    # Verify index exists
    info = mgr._read_index_info(index_name)
    if info is None:
        return None, "INDEX_NOT_FOUND"

    # Get search components
    search_fn, metadata_getter, vector_count_fn = mgr.get_search_components(
        index_name,
    )

    # Create embedding function
    embedder = EmbeddingEngine(model_name=info.get("embedding_model", config.embedding.model_name))

    engine = SearchEngine(
        embed_fn=embedder.embed,
        search_fn=search_fn,
        metadata_getter=metadata_getter,
        vector_count_fn=vector_count_fn,
    )
    return engine, None


# ---------------------------------------------------------------------------
# POST /api/v1/indexes/{name}/search — Single semantic search
# ---------------------------------------------------------------------------

@router.post(
    "",
    response_model=SearchResponse,
    summary="Single semantic search",
    description=(
        "Execute a semantic search against the specified index. "
        "Supports offset-based and cursor-based pagination, plus "
        "metadata filters. SRS FR-API-002."
    ),
    responses={
        401: {"description": "Authentication required"},
        404: {"description": "Index not found"},
        422: {"description": "Validation error"},
    },
)
async def search(
    name: str,
    body: SearchRequest,
    request: Request,
    _user: dict = Depends(get_current_user),
) -> SearchResponse:
    """Execute a single semantic search.

    SRS FR-API-002: POST /api/v1/indexes/{name}/search
    - Returns ranked results with similarity scores
    - Supports offset and cursor pagination (FR-API-003)
    - Supports metadata filtering (file_extension, source, min_score)
    """
    request_id = _get_request_id(request)

    engine, error = _get_search_engine(name)
    if error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": error,
                "code": 404,
                "detail": f"Index '{name}' not found",
                "request_id": request_id,
            },
        )

    try:
        filters = body.filters.model_dump(exclude_none=True) if body.filters else None

        result = await engine.async_search(
            query=body.query,
            k=body.k,
            page=body.page,
            page_size=body.page_size,
            filters=filters,
            cursor=body.cursor,
            pagination_mode=body.pagination,
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
    except Exception as exc:
        logger.error(
            "[SEARCH-ROUTES] Search failed  index=%s  query=%r  error=%s  request_id=%s",
            name,
            body.query[:80],
            exc,
            request_id,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "SEARCH_ERROR",
                "code": 500,
                "detail": "Search operation failed",
                "request_id": request_id,
            },
        )

    # Record analytics (best-effort)
    try:
        from isocortex.engine.analytics.engine import SearchRecord, AnalyticsEngine
        from isocortex.storage import get_analytics
        analytics = get_analytics()
        if analytics:
            record = SearchRecord(
                query=body.query,
                index_name=name,
                result_count=len(result.results),
                latency_ms=result.latency_ms,
                user_id=_user.get("sub"),
                k=body.k,
            )
            analytics.record_search(record)
    except Exception:
        pass  # Analytics should never block the response

    logger.info(
        "[SEARCH-ROUTES] Search  index=%s  query=%r  results=%d  latency=%.1fms  "
        "user=%s  request_id=%s",
        name,
        body.query[:60],
        len(result.results),
        result.latency_ms,
        _user.get("sub"),
        request_id,
    )

    return SearchResponse(
        request_id=request_id,
        results=[
            SearchResultItem(
                id=r.id,
                text=r.text,
                metadata=r.metadata,
                score=round(r.score, 4),
                rank=r.rank,
            )
            for r in result.results
        ],
        pagination=result.pagination.to_dict(),
        query=result.query,
        latency_ms=round(result.latency_ms, 2),
    )


# ---------------------------------------------------------------------------
# POST /api/v1/indexes/{name}/search/batch — Batch search
# ---------------------------------------------------------------------------

@router.post(
    "/batch",
    response_model=BatchSearchResponse,
    summary="Batch semantic search",
    description=(
        "Execute multiple search queries in a single request. "
        "Max 50 queries per batch. Returns 207 Multi-Status if "
        "some queries fail. SRS FR-API-005."
    ),
    responses={
        207: {"description": "Partial success (some queries failed)"},
        401: {"description": "Authentication required"},
        404: {"description": "Index not found"},
        422: {"description": "Validation error"},
    },
)
async def search_batch(
    name: str,
    body: BatchSearchRequest,
    request: Request,
    _user: dict = Depends(get_current_user),
) -> BatchSearchResponse:
    """Execute a batch of semantic searches.

    SRS FR-API-005: POST /api/v1/indexes/{name}/search/batch
    - Max 50 queries per batch
    - Returns 200 if all succeed, 207 Multi-Status if some fail
    - Each query has independent success/error status
    """
    request_id = _get_request_id(request)

    engine, error = _get_search_engine(name)
    if error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": error,
                "code": 404,
                "detail": f"Index '{name}' not found",
                "request_id": request_id,
            },
        )

    try:
        queries = [
            {
                "query": q.query,
                "k": q.k,
                "page": q.page,
                "page_size": q.page_size,
                "filters": q.filters.model_dump(exclude_none=True) if q.filters else None,
            }
            for q in body.queries
        ]

        result = await engine.async_search_batch(
            queries=queries,
            timeout=body.timeout,
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
    except Exception as exc:
        logger.error(
            "[SEARCH-ROUTES] Batch search failed  index=%s  error=%s  request_id=%s",
            name,
            exc,
            request_id,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "SEARCH_ERROR",
                "code": 500,
                "detail": "Batch search operation failed",
                "request_id": request_id,
            },
        )

    # Record analytics for each query (best-effort)
    try:
        from isocortex.engine.analytics.engine import SearchRecord
        from isocortex.storage import get_analytics
        analytics = get_analytics()
        if analytics:
            for qr in result.results:
                if qr.status == "success" and qr.results:
                    record = SearchRecord(
                        query=qr.query,
                        index_name=name,
                        result_count=len(qr.results),
                        latency_ms=0.0,  # batch doesn't track per-query latency separately
                        user_id=_user.get("sub"),
                    )
                    analytics.record_search(record)
    except Exception:
        pass

    logger.info(
        "[SEARCH-ROUTES] Batch search  index=%s  queries=%d  success=%d  failed=%d  "
        "latency=%.1fms  user=%s  request_id=%s",
        name,
        result.total_queries,
        result.successful,
        result.failed,
        result.latency_ms,
        _user.get("sub"),
        request_id,
    )

    response = BatchSearchResponse(
        request_id=request_id,
        results=[
            BatchQueryResultItem(
                query=qr.query,
                status=qr.status,
                results=[
                    SearchResultItem(
                        id=r.id,
                        text=r.text,
                        metadata=r.metadata,
                        score=round(r.score, 4),
                        rank=r.rank,
                    )
                    for r in (qr.results or [])
                ] if qr.results else None,
                pagination=qr.pagination.to_dict() if qr.pagination else None,
                error=qr.error,
            )
            for qr in result.results
        ],
        total_queries=result.total_queries,
        successful=result.successful,
        failed=result.failed,
        latency_ms=round(result.latency_ms, 2),
    )

    return response
