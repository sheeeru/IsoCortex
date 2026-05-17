"""
IsoCortex engine/indexing — Index Manager
==========================================

Orchestrates multi-index CRUD, version negotiation, compaction,
incremental updates, and export/import for HNSW indices.

SRS References:
  - FR-IDX-001:  Index construction with configurable HNSW parameters
  - FR-IDX-002:  Soft delete pattern with tombstone compaction
  - FR-IDX-003:  Incremental updates (add/remove without full rebuild)
  - FR-API-001:  Index management endpoints (CRUD)
  - FR-API-006:  Export/Import endpoints
  - Section 7:   Index format versioning and migration
  - NFR-09:      Data integrity (atomic operations, checksums)
  - NFR-16:      Memory pre-check before index creation

Author : Shaheer Qureshi
Project: IsoCortex
"""

from isocortex.engine.indexing.manager import IndexManager

__all__ = ["IndexManager"]
