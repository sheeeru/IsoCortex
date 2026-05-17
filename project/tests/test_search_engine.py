"""
IsoCortex — Search Engine Tests
=================================
Tests for SearchEngine, SearchResult, PaginatedResult, BatchSearch,
and the embedding cache.
"""

from __future__ import annotations

import base64
import hashlib
import json
from typing import Any

import numpy as np
import pytest

from isocortex.core.search.engine import (
    BATCH_MAX_QUERIES,
    DEFAULT_K,
    MIN_QUERY_LENGTH,
    PAGE_SIZE_DEFAULT,
    BatchSearchResult,
    PaginatedResult,
    SearchEngine,
    SearchResult,
    _decode_cursor,
    _encode_cursor,
)
from isocortex.core.embedding.embedder import EmbeddingCache


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture()
def engine(mock_embed_fn, mock_search_fn, mock_metadata_getter, mock_vector_count_fn):
    """Create a SearchEngine with mocked dependencies."""
    return SearchEngine(
        embed_fn=mock_embed_fn,
        search_fn=mock_search_fn,
        metadata_getter=mock_metadata_getter,
        vector_count_fn=mock_vector_count_fn,
    )


@pytest.fixture()
def cache():
    """Create a fresh EmbeddingCache."""
    return EmbeddingCache(max_size=3)


# =============================================================================
# Single Search Tests
# =============================================================================

class TestSingleSearch:
    """Tests for SearchEngine.search()."""

    def test_single_search_basic(self, engine: SearchEngine):
        """Mock embed + search, verify results returned."""
        result = engine.search("hello world test query")

        assert isinstance(result, PaginatedResult)
        assert result.request_id
        assert result.query == "hello world test query"
        assert len(result.results) > 0
        assert isinstance(result.results[0], SearchResult)
        assert result.results[0].rank == 1
        assert result.results[0].score >= 0  # cosine similarity (may be > 1 due to FP)
        assert result.latency_ms >= 0

    def test_single_search_with_filters(self, engine: SearchEngine):
        """Test file_extension and min_score filters."""
        result = engine.search(
            "hello world test query",
            filters={"file_extension": [".pdf"]},
        )
        # All results should be from .pdf files (or empty if no matches)
        for r in result.results:
            assert r.metadata.get("file_extension") in [".pdf", None]

    def test_single_search_pagination_offset(self, engine: SearchEngine):
        """Test page/page_size parameters."""
        result = engine.search("hello world test query", page=1, page_size=3)
        assert len(result.results) <= 3
        assert result.pagination.page == 1
        assert result.pagination.page_size == 3

    def test_single_search_pagination_cursor(self, engine: SearchEngine):
        """Test cursor encode/decode."""
        result = engine.search(
            "hello world test query",
            pagination_mode="cursor",
            page_size=3,
        )
        # Should have cursor-based pagination info
        assert hasattr(result.pagination, "has_more")
        assert hasattr(result.pagination, "next_cursor")

    def test_single_search_empty_query(self, engine: SearchEngine):
        """Test ValueError for short query."""
        with pytest.raises(ValueError, match="too short"):
            engine.search("ab")


# =============================================================================
# Batch Search Tests
# =============================================================================

class TestBatchSearch:
    """Tests for SearchEngine.search_batch()."""

    def test_batch_search_basic(self, engine: SearchEngine):
        """Test multiple queries."""
        result = engine.search_batch([
            {"query": "first test query"},
            {"query": "second test query"},
        ])
        assert isinstance(result, BatchSearchResult)
        assert result.total_queries == 2
        assert result.successful == 2
        assert result.failed == 0

    def test_batch_search_partial_failure(self, engine: SearchEngine):
        """Mix valid and invalid queries."""
        result = engine.search_batch([
            {"query": "valid test query"},
            {"query": "ab"},  # too short
            {"query": "another valid query"},
        ])
        assert result.total_queries == 3
        assert result.successful == 2
        assert result.failed == 1
        # Verify error details
        for qr in result.results:
            if qr.status == "error":
                assert qr.error is not None

    def test_batch_search_max_exceeded(self, engine: SearchEngine):
        """Test ValueError for > 50 queries."""
        queries = [{"query": f"test query number {i}"} for i in range(51)]
        with pytest.raises(ValueError, match="too large"):
            engine.search_batch(queries)


# =============================================================================
# Embedding Cache Tests
# =============================================================================

class TestEmbeddingCache:
    """Test LRU cache behavior (hit/miss/eviction)."""

    def test_cache_hit_miss(self, cache: EmbeddingCache):
        """Verify hit on second access, miss on first."""
        vec = np.zeros(384, dtype=np.float32)
        assert cache.get("hello") is None  # miss
        cache.put("hello", vec)
        assert cache.get("hello") is not None  # hit
        assert cache.size == 1

    def test_cache_eviction(self, cache: EmbeddingCache):
        """Oldest entry evicted when cache is full."""
        for i in range(4):  # max_size=3
            cache.put(f"key-{i}", np.zeros(384, dtype=np.float32))
        assert cache.size == 3
        # "key-0" should be evicted
        assert cache.get("key-0") is None
        assert cache.get("key-3") is not None

    def test_cache_stats(self, cache: EmbeddingCache):
        """Verify cache stats tracking."""
        vec = np.zeros(384, dtype=np.float32)
        cache.get("a")  # miss
        cache.put("a", vec)
        cache.get("a")  # hit
        stats = cache.stats
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5

    def test_cache_clear(self, cache: EmbeddingCache):
        """Clear resets everything."""
        cache.put("a", np.zeros(384, dtype=np.float32))
        cache.clear()
        assert cache.size == 0
        assert cache.stats["hits"] == 0

    def test_cache_lru_ordering(self, cache: EmbeddingCache):
        """Accessing a key moves it to the end (most recent)."""
        vec = np.zeros(384, dtype=np.float32)
        cache.put("a", vec)
        cache.put("b", vec)
        cache.put("c", vec)
        # Access "a" to make it most recently used
        cache.get("a")
        # Add new entry -> "b" should be evicted (not "a")
        cache.put("d", vec)
        assert cache.get("b") is None  # evicted
        assert cache.get("a") is not None  # still there


# =============================================================================
# SearchResult Serialization Tests
# =============================================================================

class TestSearchResultSerialization:
    """Verify serialization of result objects."""

    def test_search_result_to_dict(self):
        """Verify SearchResult serializes correctly."""
        sr = SearchResult(
            id="vec-42",
            text="Sample text",
            metadata={"file_extension": ".pdf"},
            score=0.95,
            rank=1,
        )
        d = sr.to_dict()
        assert d["id"] == "vec-42"
        assert d["text"] == "Sample text"
        assert d["score"] == 0.95
        assert d["rank"] == 1
        assert d["metadata"]["file_extension"] == ".pdf"

    def test_paginated_result_to_dict(self, engine: SearchEngine):
        """Verify PaginatedResult serializes correctly."""
        result = engine.search("hello world test query", page=1, page_size=3)
        d = result.to_dict()
        assert "request_id" in d
        assert "results" in d
        assert "pagination" in d
        assert "query" in d
        assert isinstance(d["results"], list)
        assert d["pagination"]["page"] == 1


# =============================================================================
# Cursor Tests
# =============================================================================

class TestCursorEncoding:
    """Test cursor encode/decode round-trip."""

    def test_cursor_roundtrip(self):
        """Encode then decode should return original offset."""
        query_hash = hashlib.sha256("test query".encode()).hexdigest()[:16]
        cursor = _encode_cursor(42, query_hash)
        offset = _decode_cursor(cursor, query_hash)
        assert offset == 42

    def test_cursor_tampered_hash(self):
        """Tampered query hash should raise ValueError."""
        query_hash = hashlib.sha256("test query".encode()).hexdigest()[:16]
        cursor = _encode_cursor(42, query_hash)
        with pytest.raises(ValueError, match="hash mismatch"):
            _decode_cursor(cursor, "wronghash12345")
