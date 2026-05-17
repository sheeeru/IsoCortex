"""
IsoCortex core/embedding — Embedding Provider
==============================================

Pluggable neural text embedding with:
  - EmbeddingProvider ABC for custom model integration
  - MiniLMProvider (all-MiniLM-L6-v2, 384 dimensions)
  - LRU embedding cache (SRS FR-EMB-002)
  - Batch embedding with async queue (SRS Section 4.3)

SRS References: FR-EMB-001, FR-EMB-002, Section 4.3
"""
