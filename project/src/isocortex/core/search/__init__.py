"""
IsoCortex core/search — Semantic search engine.

Provides single search, batch search, pagination (offset + cursor),
and result filtering over HNSW indices.

SRS References: FR-API-002, FR-API-003, FR-API-005, NFR-01, NFR-15
"""

from isocortex.core.search.engine import SearchEngine, SearchResult, PaginatedResult

__all__ = ["SearchEngine", "SearchResult", "PaginatedResult"]
