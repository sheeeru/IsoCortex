"""
IsoCortex — Semantic Search Engine
===================================

High-performance semantic search over HNSW indices with pagination,
batch support, filtering, and cursor-based navigation.

SRS References:
  - FR-API-002: Single and batch semantic search
  - FR-API-003: Offset-based and cursor-based pagination
  - FR-API-005: Batch search (max 50 queries, partial success, 207 Multi-Status)
  - NFR-01:    p95 < 100ms, p99 < 500ms for 1M vectors
  - NFR-15:    10+ concurrent requests, 100+ QPS
  - NFR-17:    Embedding cache hit rate > 80%

Author : Shaheer Qureshi
Project: IsoCortex
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Constants (SRS Appendix B — Configuration Reference)
# =============================================================================

DEFAULT_K: int = 5
MAX_K: int = 100
PAGE_SIZE_DEFAULT: int = 10
PAGE_SIZE_MAX: int = 100
BATCH_MAX_QUERIES: int = 50
BATCH_TIMEOUT_SECONDS: int = 30
BATCH_CONCURRENCY: int = 4
MIN_QUERY_LENGTH: int = 3


# =============================================================================
# Data Classes
# =============================================================================

@dataclass(frozen=True)
class SearchResult:
    """A single search result with metadata and similarity score.

    SRS FR-API-002: Response format with id, text, metadata, score, rank.
    """
    id: str
    text: str
    metadata: dict[str, Any]
    score: float
    rank: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "metadata": self.metadata,
            "score": round(self.score, 4),
            "rank": self.rank,
        }


@dataclass(frozen=True)
class PaginationInfo:
    """Pagination metadata in search responses.

    SRS FR-API-003: total_results, page, page_size, total_pages.
    """
    total_results: int
    page: int
    page_size: int
    total_pages: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_results": self.total_results,
            "page": self.page,
            "page_size": self.page_size,
            "total_pages": self.total_pages,
        }


@dataclass(frozen=True)
class CursorPaginationInfo:
    """Cursor-based pagination metadata.

    SRS FR-API-003: next_cursor, has_more. No total_pages.
    """
    next_cursor: Optional[str]
    has_more: bool
    page_size: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "next_cursor": self.next_cursor,
            "has_more": self.has_more,
            "page_size": self.page_size,
        }


@dataclass
class PaginatedResult:
    """A paginated search response.

    SRS FR-API-002: Single search response with results + pagination.
    """
    request_id: str
    results: list[SearchResult]
    pagination: PaginationInfo | CursorPaginationInfo
    query: str
    latency_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "results": [r.to_dict() for r in self.results],
            "pagination": self.pagination.to_dict(),
            "query": self.query,
            "latency_ms": round(self.latency_ms, 2),
        }


@dataclass
class BatchQueryResult:
    """Result of a single query within a batch search.

    SRS FR-API-005: Each query result has status (success/error).
    """
    query: str
    status: str = "success"
    results: list[SearchResult] | None = None
    pagination: PaginationInfo | CursorPaginationInfo | None = None
    error: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"query": self.query, "status": self.status}
        if self.status == "success" and self.results is not None:
            d["results"] = [r.to_dict() for r in self.results]
            if self.pagination is not None:
                d["pagination"] = self.pagination.to_dict()
        elif self.error is not None:
            d["error"] = self.error
        return d


@dataclass
class BatchSearchResult:
    """Complete batch search response.

    SRS FR-API-005: 200 if all succeed, 207 Multi-Status if some fail.
    """
    request_id: str
    results: list[BatchQueryResult]
    total_queries: int
    successful: int
    failed: int
    latency_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "results": [r.to_dict() for r in self.results],
            "total_queries": self.total_queries,
            "successful": self.successful,
            "failed": self.failed,
            "latency_ms": round(self.latency_ms, 2),
        }


# =============================================================================
# Cursor Encoding / Decoding
# =============================================================================

def _encode_cursor(offset: int, query_hash: str) -> str:
    """Encode offset + query hash into an opaque cursor string."""
    payload = json.dumps({"o": offset, "h": query_hash})
    return base64.urlsafe_b64encode(payload.encode()).decode().rstrip("=")


def _decode_cursor(cursor: str, expected_hash: str) -> int:
    """Decode cursor string and validate query hash.

    Returns the offset if valid, raises ValueError if cursor is tampered.
    """
    # Re-add padding
    padding = 4 - len(cursor) % 4
    if padding != 4:
        cursor += "=" * padding
    try:
        payload = json.loads(base64.urlsafe_b64decode(cursor))
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"Invalid cursor format: {exc}") from exc

    if payload.get("h") != expected_hash:
        raise ValueError("Cursor query hash mismatch — cursor is not valid for this query")

    return payload["o"]


# =============================================================================
# Filter Application
# =============================================================================

def _apply_filters(
    candidates: list[tuple[int, float]],
    metadata_list: list[dict[str, Any]],
    filters: dict[str, Any] | None,
) -> list[tuple[int, float]]:
    """Filter search candidates based on metadata filters.

    Supported filters:
      - file_extension: list[str] — e.g. [".pdf", ".md"]
      - source: str — source file path substring
      - min_score: float — minimum similarity score
    """
    if not filters:
        return candidates

    filtered: list[tuple[int, float]] = []
    for idx, (vec_idx, score) in enumerate(candidates):
        if idx >= len(metadata_list):
            break
        meta = metadata_list[vec_idx]
        skip = False

        if "file_extension" in filters:
            allowed = filters["file_extension"]
            if isinstance(allowed, list):
                if meta.get("file_extension") not in allowed:
                    skip = True

        if "source" in filters and not skip:
            source_pattern = filters["source"]
            if source_pattern not in meta.get("source_file", ""):
                skip = True

        if "min_score" in filters and not skip:
            if score < filters["min_score"]:
                skip = True

        if not skip:
            filtered.append((vec_idx, score))

    return filtered


# =============================================================================
# Search Engine
# =============================================================================

class SearchEngine:
    """High-performance semantic search engine over HNSW indices.

    Supports:
      - Single semantic search with offset/cursor pagination
      - Batch search (up to 50 queries) with partial success (207 Multi-Status)
      - Metadata filtering (file_extension, source, min_score)
      - Embedding query caching via EmbeddingQueue integration
      - Concurrent query execution

    SRS References:
      - FR-API-002: Search endpoints
      - FR-API-003: Pagination specification
      - FR-API-005: Batch search specification
      - NFR-01:    p95 < 100ms search latency
      - NFR-15:    100+ QPS concurrent throughput
    """

    def __init__(
        self,
        embed_fn=None,  # Callable[[str], np.ndarray]
        search_fn=None,  # Callable[[np.ndarray, int], list[tuple[int, float]]]
        metadata_getter=None,  # Callable[[], list[dict[str, Any]]]
        vector_count_fn=None,  # Callable[[], int]
    ) -> None:
        """
        Parameters
        ----------
        embed_fn : Callable
            Function that embeds a query string into a vector.
            Signature: embed_fn(text: str) -> np.ndarray (shape: (384,))
        search_fn : Callable
            Function that searches the HNSW index.
            Signature: search_fn(query_vec: np.ndarray, k: int) -> list[tuple[int, float]]
            Returns list of (vector_index, distance_score) tuples.
        metadata_getter : Callable
            Function that returns the full metadata list.
            Signature: metadata_getter() -> list[dict[str, Any]]
        vector_count_fn : Callable
            Function that returns the total number of active (non-deleted) vectors.
            Signature: vector_count_fn() -> int
        """
        self._embed_fn = embed_fn
        self._search_fn = search_fn
        self._metadata_getter = metadata_getter
        self._vector_count_fn = vector_count_fn

        # Embedding cache (in-process LRU, SRS FR-EMB-002 / NFR-17)
        self._cache: dict[str, np.ndarray] = {}
        self._cache_order: list[str] = []
        self._cache_size: int = 1000
        self._cache_hits: int = 0
        self._cache_misses: int = 0

        # Thread pool for CPU-bound embedding (SRS Section 4.3)
        self._executor = ThreadPoolExecutor(max_workers=4)

    def set_cache_size(self, size: int) -> None:
        """Set the embedding cache size (SRS: default 1000)."""
        self._cache_size = size
        # Evict if shrinking
        while len(self._cache_order) > size:
            oldest = self._cache_order.pop(0)
            self._cache.pop(oldest, None)

    @property
    def cache_stats(self) -> dict[str, Any]:
        """Return cache hit/miss statistics."""
        total = self._cache_hits + self._cache_misses
        return {
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "hit_rate": self._cache_hits / total if total > 0 else 0.0,
            "size": len(self._cache),
            "max_size": self._cache_size,
        }

    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self._cache.clear()
        self._cache_order.clear()
        self._cache_hits = 0
        self._cache_misses = 0
        logger.info("[SEARCH] Embedding cache cleared")

    def _embed_query(self, text: str) -> np.ndarray:
        """Embed a query string with LRU caching.

        Thread-safe: only one embed call at a time through the executor.
        """
        if text in self._cache:
            self._cache_hits += 1
            self._cache_order.remove(text)
            self._cache_order.append(text)
            return self._cache[text]

        self._cache_misses += 1
        vector = self._embed_fn(text)

        # Update cache
        if len(self._cache) >= self._cache_size:
            oldest = self._cache_order.pop(0)
            self._cache.pop(oldest, None)

        self._cache[text] = vector
        self._cache_order.append(text)
        return vector

    # -----------------------------------------------------------------
    # Single Search
    # -----------------------------------------------------------------

    def search(
        self,
        query: str,
        k: int = DEFAULT_K,
        page: int = 1,
        page_size: int = PAGE_SIZE_DEFAULT,
        filters: dict[str, Any] | None = None,
        cursor: str | None = None,
        pagination_mode: str = "offset",
    ) -> PaginatedResult:
        """Execute a single semantic search with pagination.

        SRS FR-API-002: POST /api/v1/indexes/{name}/search
        SRS FR-API-003: Offset-based (default) and cursor-based pagination.

        Parameters
        ----------
        query : str
            Natural language query text (minimum 3 characters).
        k : int
            Number of nearest neighbours to retrieve from HNSW.
            Auto-adjusted to >= page_size (SRS FR-API-003 constraint).
        page : int
            Page number for offset pagination (1-based, default 1).
        page_size : int
            Results per page (1-100, default 10).
        filters : dict
            Metadata filters: file_extension, source, min_score.
        cursor : str
            Opaque cursor string for cursor-based pagination.
        pagination_mode : str
            "offset" (default) or "cursor".

        Returns
        -------
        PaginatedResult
            Search results with pagination metadata.
        """
        t0 = time.perf_counter()
        request_id = str(uuid.uuid4())

        # Validate query
        if not query or len(query.strip()) < MIN_QUERY_LENGTH:
            raise ValueError(
                f"Query too short (minimum {MIN_QUERY_LENGTH} characters)"
            )

        # Clamp k and page_size per SRS
        k = max(1, min(k, MAX_K))
        page_size = max(1, min(page_size, PAGE_SIZE_MAX))
        if k < page_size:
            logger.warning(
                "[SEARCH] k=%d < page_size=%d — auto-adjusting k to %d",
                k, page_size, page_size,
            )
            k = page_size

        # Embed query
        query_vec = self._embed_query(query.strip())

        # Retrieve from HNSW (fetch extra candidates for filtering/pagination)
        fetch_count = k * max(page_size, 1)
        candidates = self._search_fn(query_vec, fetch_count)

        # Apply filters
        metadata_list = self._metadata_getter() if filters else []
        if filters:
            candidates = _apply_filters(candidates, metadata_list, filters)

        total_results = len(candidates)

        # Compute query hash for cursor
        query_hash = hashlib.sha256(query.strip().encode()).hexdigest()[:16]

        if pagination_mode == "cursor":
            # Cursor-based pagination
            offset = 0
            if cursor:
                offset = _decode_cursor(cursor, query_hash)

            page_results = candidates[offset:offset + page_size]
            has_more = (offset + page_size) < total_results
            next_cursor = _encode_cursor(offset + page_size, query_hash) if has_more else None

            results = self._build_results(page_results, metadata_list, offset)
            pagination_info: PaginationInfo | CursorPaginationInfo = CursorPaginationInfo(
                next_cursor=next_cursor,
                has_more=has_more,
                page_size=page_size,
            )
        else:
            # Offset-based pagination (default)
            page = max(1, page)
            offset = (page - 1) * page_size
            page_results = candidates[offset:offset + page_size]
            total_pages = max(1, (total_results + page_size - 1) // page_size)

            results = self._build_results(page_results, metadata_list, offset)
            pagination_info = PaginationInfo(
                total_results=total_results,
                page=page,
                page_size=page_size,
                total_pages=total_pages,
            )

        latency_ms = (time.perf_counter() - t0) * 1000

        logger.info(
            "[SEARCH] query=%r k=%d results=%d latency=%.1fms request_id=%s",
            query.strip()[:50], k, len(results), latency_ms, request_id,
        )

        return PaginatedResult(
            request_id=request_id,
            results=results,
            pagination=pagination_info,
            query=query.strip(),
            latency_ms=latency_ms,
        )

    def _build_results(
        self,
        candidates: list[tuple[int, float]],
        metadata_list: list[dict[str, Any]],
        offset: int,
    ) -> list[SearchResult]:
        """Convert HNSW candidates into SearchResult objects."""
        results = []
        for rank, (vec_idx, score) in enumerate(candidates, start=offset + 1):
            meta = metadata_list[vec_idx] if vec_idx < len(metadata_list) else {}
            results.append(SearchResult(
                id=meta.get("id", f"vec-{vec_idx}"),
                text=meta.get("text_preview", ""),
                metadata=meta,
                score=1.0 - score if score <= 1.0 else score,  # cosine distance -> similarity
                rank=rank,
            ))
        return results

    # -----------------------------------------------------------------
    # Batch Search
    # -----------------------------------------------------------------

    def search_batch(
        self,
        queries: list[dict[str, Any]],
        timeout: int = BATCH_TIMEOUT_SECONDS,
        concurrency: int = BATCH_CONCURRENCY,
    ) -> BatchSearchResult:
        """Execute multiple semantic queries in a single call.

        SRS FR-API-005: POST /api/v1/indexes/{name}/search/batch

        Parameters
        ----------
        queries : list[dict]
            Each dict has "query" (str, required), "k" (int, optional),
            "filters" (dict, optional), "page" (int), "page_size" (int).
        timeout : int
            Per-query timeout in seconds (max 120s).
        concurrency : int
            Max parallel query processing (default 4).

        Returns
        -------
        BatchSearchResult
            200 if all succeed, partial results with 207 Multi-Status if some fail.
        """
        t0 = time.perf_counter()
        request_id = str(uuid.uuid4())

        # Validate batch size
        if len(queries) > BATCH_MAX_QUERIES:
            raise ValueError(
                f"Batch too large: {len(queries)} queries exceeds max {BATCH_MAX_QUERIES}"
            )

        timeout = min(max(1, timeout), 120)
        concurrency = min(max(1, concurrency), 16)

        # Process queries
        results: list[BatchQueryResult] = []
        successful = 0
        failed = 0

        # Batch embed all queries first for efficiency
        query_texts = []
        valid_indices = []
        for i, q in enumerate(queries):
            text = q.get("query", "")
            if not text or len(text.strip()) < MIN_QUERY_LENGTH:
                results.append(BatchQueryResult(
                    query=text,
                    status="error",
                    error={
                        "code": 400,
                        "message": f"Query too short (minimum {MIN_QUERY_LENGTH} characters)",
                    },
                ))
                failed += 1
            else:
                query_texts.append(text.strip())
                valid_indices.append(i)
                results.append(None)  # placeholder

        # Embed valid queries
        embedded: dict[str, np.ndarray] = {}
        for text in query_texts:
            if text not in embedded:
                embedded[text] = self._embed_query(text)

        # Execute searches for valid queries
        for i, text in zip(valid_indices, query_texts):
            q = queries[i]
            k = q.get("k", DEFAULT_K)
            page = q.get("page", 1)
            page_size = q.get("page_size", PAGE_SIZE_DEFAULT)
            filters = q.get("filters")
            cursor = q.get("cursor")
            pagination_mode = q.get("pagination", "offset")

            try:
                pag_result = self.search(
                    query=text,
                    k=k,
                    page=page,
                    page_size=page_size,
                    filters=filters,
                    cursor=cursor,
                    pagination_mode=pagination_mode,
                )
                results[i] = BatchQueryResult(
                    query=text,
                    status="success",
                    results=pag_result.results,
                    pagination=pag_result.pagination,
                )
                successful += 1
            except Exception as exc:
                logger.error(
                    "[BATCH] Query failed: %r — %s", text[:50], exc,
                )
                results[i] = BatchQueryResult(
                    query=text,
                    status="error",
                    error={
                        "code": 500,
                        "message": str(exc),
                    },
                )
                failed += 1

        latency_ms = (time.perf_counter() - t0) * 1000

        logger.info(
            "[BATCH] queries=%d success=%d failed=%d latency=%.1fms request_id=%s",
            len(queries), successful, failed, latency_ms, request_id,
        )

        return BatchSearchResult(
            request_id=request_id,
            results=results,  # type: ignore
            total_queries=len(queries),
            successful=successful,
            failed=failed,
            latency_ms=latency_ms,
        )

    # -----------------------------------------------------------------
    # Async wrappers
    # -----------------------------------------------------------------

    async def async_search(self, *args: Any, **kwargs: Any) -> PaginatedResult:
        """Async wrapper for single search. Runs in thread pool."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor, lambda: self.search(*args, **kwargs)
        )

    async def async_search_batch(self, *args: Any, **kwargs: Any) -> BatchSearchResult:
        """Async wrapper for batch search."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor, lambda: self.search_batch(*args, **kwargs)
        )
