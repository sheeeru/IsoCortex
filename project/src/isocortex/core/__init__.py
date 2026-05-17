"""
IsoCortex — Core Package
=========================

Core algorithms and data structures:
  - HNSW: Hierarchical Navigable Small World graph for ANN search
  - Embedding: Neural text embedding with MiniLM
  - Extractor: Multi-format document text extraction
  - Search: Semantic search engine with pagination and batching
"""

from isocortex.core.search import SearchEngine, SearchResult, PaginatedResult

__all__ = ["SearchEngine", "SearchResult", "PaginatedResult"]
