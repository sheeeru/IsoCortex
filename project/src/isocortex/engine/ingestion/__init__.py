"""
IsoCortex engine/ingestion — Ingestion Pipeline
===============================================

File scanning and text chunking pipeline:
  - Scanner: Recursive directory traversal with glob patterns
  - Chunker: Sentence-aware text chunking with overlap

SRS References:
  - FR-ING-002: Text chunking (512 tokens, 50 token overlap)
  - FR-ING-003: Directory ingestion (glob patterns, SHA-256 dedup)
"""
