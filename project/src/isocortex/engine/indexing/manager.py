"""
IsoCortex — Index Manager
==========================

The central orchestration layer for managing HNSW indices. Handles:

  1. Multi-index lifecycle (create, list, get, update, delete)
  2. Index construction pipeline (scan → extract → chunk → embed → build HNSW)
  3. Incremental updates (add documents, soft-delete, compaction)
  4. Format version negotiation and auto-migration
  5. Export/Import (.isocortex archives with SHA-256 integrity)
  6. Memory pre-check before large operations (SRS NFR-16)
  7. ReadWriteLock for thread-safe index access (SRS Section 4.2)

SRS References:
  - FR-IDX-001/002/003: Index construction, soft delete, incremental
  - FR-API-001/006:      Index CRUD + export/import endpoints
  - Section 7:            Index format versioning
  - NFR-09/16:            Data integrity + memory requirements

Author : Shaheer Qureshi
Project: IsoCortex
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

SUPPORTED_FORMAT_VERSION: int = 1
INDEX_INFO_FILE: str = "index_info.json"
VECTORS_BIN_FILE: str = "vectors.bin"
METADATA_JSON_FILE: str = "metadata.json"
HNSW_INDEX_BIN_FILE: str = "hnsw_index.bin"
CONFIG_JSON_FILE: str = "config.json"
ARCHIVE_EXTENSION: str = ".isocortex"

# Memory estimation constants (SRS NFR-16)
BYTES_PER_VECTOR_384: int = 384 * 4  # float32
HNSW_OVERHEAD_MULTIPLIER: float = 3.0  # 2-4x graph overhead
PYTHON_RUNTIME_OVERHEAD_MB: int = 200


# =============================================================================
# ReadWriteLock (SRS Section 4.2)
# =============================================================================

class ReadWriteLock:
    """Reader-writer lock with writer priority.

    Guarantees:
    - Multiple concurrent readers
    - Exclusive writer access
    - Writer priority (prevents writer starvation)
    - Read-during-write safety
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._reader_count: int = 0
        self._writer_active: bool = False
        self._writer_waiting: int = 0

    def acquire_read(self) -> None:
        with self._condition:
            while self._writer_active or self._writer_waiting > 0:
                self._condition.wait()
            self._reader_count += 1

    def release_read(self) -> None:
        with self._condition:
            self._reader_count -= 1
            if self._reader_count == 0:
                self._condition.notify_all()

    def acquire_write(self) -> None:
        with self._condition:
            self._writer_waiting += 1
            while self._reader_count > 0 or self._writer_active:
                self._condition.wait()
            self._writer_waiting -= 1
            self._writer_active = True

    def release_write(self) -> None:
        with self._condition:
            self._writer_active = False
            self._condition.notify_all()


# =============================================================================
# Data Classes
# =============================================================================

@dataclass(frozen=True)
class IndexStats:
    """Statistics about a loaded index."""
    name: str
    vector_count: int
    deleted_count: int
    active_count: int
    index_size_mb: float
    created_at: str
    updated_at: str
    embedding_model: str
    dimension: int
    hnsw_params: dict[str, Any]
    chunk_config: dict[str, Any]
    format_version: int
    healthy: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "vector_count": self.vector_count,
            "deleted_count": self.deleted_count,
            "active_count": self.active_count,
            "index_size_mb": round(self.index_size_mb, 2),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "embedding_model": self.embedding_model,
            "dimension": self.dimension,
            "hnsw_params": self.hnsw_params,
            "chunk_config": self.chunk_config,
            "format_version": self.format_version,
            "healthy": self.healthy,
        }


@dataclass(frozen=True)
class IndexInfo:
    """Lightweight index metadata (for listing)."""
    name: str
    description: str
    vector_count: int
    deleted_count: int
    created_at: str
    updated_at: str
    healthy: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "vector_count": self.vector_count,
            "deleted_count": self.deleted_count,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "healthy": self.healthy,
        }


@dataclass
class InMemoryIndex:
    """A loaded index held in memory with its lock and metadata."""
    name: str
    index_info: dict[str, Any]
    vectors: np.ndarray | None = None
    metadata: list[dict[str, Any]] = field(default_factory=list)
    hnsw: Any = None  # HNSW index object (from C++ binding or Python fallback)
    lock: ReadWriteLock = field(default_factory=ReadWriteLock)
    healthy: bool = True
    loaded_at: float = field(default_factory=time.time)


# =============================================================================
# Migration System (SRS Section 7)
# =============================================================================

class IndexVersionError(Exception):
    """Raised when an index format version is incompatible."""
    pass


class Migration:
    """Base class for index format migrations."""

    def __init__(
        self,
        from_version: int,
        to_version: int,
        automatic: bool,
        description: str,
    ) -> None:
        self.from_version = from_version
        self.to_version = to_version
        self.automatic = automatic
        self.description = description

    def migrate(self, index_path: Path) -> None:
        raise NotImplementedError


class MigrationRegistry:
    """Global registry for index format migrations.

    SRS Section 7.4: Breaking Change Policy.
    """

    _migrations: list[Migration] = []

    @classmethod
    def register(cls, migration: Migration) -> None:
        cls._migrations.append(migration)
        logger.info(
            "[MIGRATION] Registered: v%d -> v%d (%s) automatic=%s",
            migration.from_version, migration.to_version,
            migration.description, migration.automatic,
        )

    @classmethod
    def get_migration(cls, from_v: int, to_v: int) -> Migration | None:
        for m in cls._migrations:
            if m.from_version == from_v and m.to_version == to_v:
                return m
        return None


# =============================================================================
# Index Manager
# =============================================================================

class IndexManager:
    """Central manager for all HNSW index operations.

    Manages the full lifecycle of indices:
    - Create indices from file paths or uploaded files
    - Load/unload indices from disk into memory
    - Incremental document additions and soft deletes
    - Compaction of tombstoned vectors
    - Format version negotiation and migration
    - Export/import as .isocortex archives

    Thread Safety:
    - Each loaded index has its own ReadWriteLock
    - Multiple indices can be searched concurrently
    - Global index registry protected by a threading.Lock

    SRS References:
    - FR-API-001: Index management endpoints
    - FR-IDX-001/002/003: Index operations
    - Section 7: Index format versioning
    """

    def __init__(self, indices_dir: Path) -> None:
        """
        Parameters
        ----------
        indices_dir : Path
            Directory where index data is stored (e.g., ~/.isocortex/indices/)
        """
        self._indices_dir = Path(indices_dir)
        self._indices_dir.mkdir(parents=True, exist_ok=True)

        # Loaded indices: name -> InMemoryIndex
        self._loaded: dict[str, InMemoryIndex] = {}
        self._registry_lock = threading.Lock()

        # Callbacks (injected by API layer)
        self._on_progress: Callable[[str, float, str], None] | None = None

        logger.info("[INDEX-MGR] Initialized  indices_dir=%s", self._indices_dir)

    # -----------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------

    @property
    def indices_dir(self) -> Path:
        return self._indices_dir

    def list_index_names(self) -> list[str]:
        """List all index names on disk."""
        if not self._indices_dir.exists():
            return []
        return [
            d.name
            for d in self._indices_dir.iterdir()
            if d.is_dir() and (d / INDEX_INFO_FILE).exists()
        ]

    def get_loaded_index(self, name: str) -> InMemoryIndex | None:
        """Get a loaded index by name (thread-safe)."""
        with self._registry_lock:
            return self._loaded.get(name)

    # -----------------------------------------------------------------
    # Index Info Operations
    # -----------------------------------------------------------------

    def _index_path(self, name: str) -> Path:
        """Return the directory path for an index."""
        # Sanitize name to prevent path traversal
        safe_name = name.replace("/", "").replace("\\", "").replace("..", "")
        if not safe_name:
            raise ValueError(f"Invalid index name: {name!r}")
        return self._indices_dir / safe_name

    def _read_index_info(self, name: str) -> dict[str, Any] | None:
        """Read index_info.json from disk."""
        path = self._index_path(name) / INDEX_INFO_FILE
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.error("[INDEX-MGR] Failed to read index info for %r: %s", name, exc)
            return None

    def _write_index_info(self, name: str, info: dict[str, Any]) -> None:
        """Write index_info.json atomically."""
        idx_dir = self._index_path(name)
        idx_dir.mkdir(parents=True, exist_ok=True)
        target = idx_dir / INDEX_INFO_FILE
        tmp = target.with_suffix(".tmp")
        tmp.write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")
        tmp.replace(target)  # atomic on POSIX

    # -----------------------------------------------------------------
    # List Indexes
    # -----------------------------------------------------------------

    def list_indexes(self) -> list[IndexInfo]:
        """List all indexes with metadata.

        SRS FR-API-001: GET /api/v1/indexes
        """
        result = []
        for name in self.list_index_names():
            info = self._read_index_info(name)
            if info is None:
                continue
            loaded = self.get_loaded_index(name)
            result.append(IndexInfo(
                name=name,
                description=info.get("description", ""),
                vector_count=info.get("vector_count", 0),
                deleted_count=info.get("deleted_count", 0),
                created_at=info.get("created_at", ""),
                updated_at=info.get("updated_at", ""),
                healthy=loaded.healthy if loaded else True,
            ))
        return result

    # -----------------------------------------------------------------
    # Get Index Details
    # -----------------------------------------------------------------

    def get_index(self, name: str) -> IndexStats | None:
        """Get detailed index statistics.

        SRS FR-API-001: GET /api/v1/indexes/{name}
        """
        info = self._read_index_info(name)
        if info is None:
            return None

        idx_dir = self._index_path(name)

        # Calculate total size
        total_size = 0
        for f in idx_dir.iterdir():
            if f.is_file():
                total_size += f.stat().st_size

        loaded = self.get_loaded_index(name)
        active = info.get("vector_count", 0) - info.get("deleted_count", 0)

        return IndexStats(
            name=name,
            vector_count=info.get("vector_count", 0),
            deleted_count=info.get("deleted_count", 0),
            active_count=max(0, active),
            index_size_mb=total_size / (1024 * 1024),
            created_at=info.get("created_at", ""),
            updated_at=info.get("updated_at", ""),
            embedding_model=info.get("embedding_model", ""),
            dimension=info.get("embedding_dimension", 384),
            hnsw_params=info.get("hnsw_params", {}),
            chunk_config=info.get("chunk_config", {}),
            format_version=info.get("format_version", 1),
            healthy=loaded.healthy if loaded else True,
        )

    # -----------------------------------------------------------------
    # Create Index
    # -----------------------------------------------------------------

    def create_index(
        self,
        name: str,
        description: str = "",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        embedding_dimension: int = 384,
        hnsw_params: dict[str, Any] | None = None,
        chunk_config: dict[str, Any] | None = None,
    ) -> str:
        """Create a new empty index directory with metadata.

        SRS FR-API-001: POST /api/v1/indexes (returns 202 + job_id).

        Returns the index name. The actual document ingestion and HNSW
        construction happen asynchronously via the job scheduler.

        Parameters
        ----------
        name : str
            Index name (must be unique, alphanumeric + hyphens/underscores).
        description : str
            Human-readable description.
        embedding_model : str
            Model identifier used for embeddings.
        embedding_dimension : int
            Vector dimension (384 for MiniLM).
        hnsw_params : dict
            HNSW configuration (M, ef_construction, ef_search, metric).
        chunk_config : dict
            Chunking configuration (chunk_size, chunk_overlap).
        """
        idx_dir = self._index_path(name)

        # Check if index already exists
        if idx_dir.exists() and (idx_dir / INDEX_INFO_FILE).exists():
            raise FileExistsError(f"Index '{name}' already exists")

        if hnsw_params is None:
            hnsw_params = {"M": 16, "ef_construction": 200, "ef_search": 50, "metric": "cosine"}
        if chunk_config is None:
            chunk_config = {"chunk_size": 512, "chunk_overlap": 50}

        now = datetime.now(timezone.utc).isoformat()

        index_info = {
            "format_version": SUPPORTED_FORMAT_VERSION,
            "name": name,
            "description": description,
            "created_at": now,
            "updated_at": now,
            "embedding_model": embedding_model,
            "embedding_dimension": embedding_dimension,
            "vector_count": 0,
            "deleted_count": 0,
            "hnsw_params": hnsw_params,
            "chunk_config": chunk_config,
            "file_hashes": {},
        }

        idx_dir.mkdir(parents=True, exist_ok=True)
        self._write_index_info(name, index_info)

        logger.info("[INDEX-MGR] Created index %r  model=%s", name, embedding_model)
        return name

    # -----------------------------------------------------------------
    # Delete Index
    # -----------------------------------------------------------------

    def delete_index(self, name: str) -> None:
        """Delete an index from disk and unload it from memory.

        SRS FR-API-001: DELETE /api/v1/indexes/{name}
        """
        idx_dir = self._index_path(name)
        if not idx_dir.exists():
            raise FileNotFoundError(f"Index '{name}' not found")

        # Unload from memory first
        with self._registry_lock:
            self._loaded.pop(name, None)

        # Remove from disk
        shutil.rmtree(idx_dir, ignore_errors=True)
        logger.info("[INDEX-MGR] Deleted index %r", name)

    # -----------------------------------------------------------------
    # Update Index Configuration
    # -----------------------------------------------------------------

    def update_index(self, name: str, updates: dict[str, Any]) -> None:
        """Update index metadata (description, HNSW params, etc.).

        SRS FR-API-001: PUT /api/v1/indexes/{name}

        Only certain fields are updatable: description, hnsw_params.ef_search.
        """
        info = self._read_index_info(name)
        if info is None:
            raise FileNotFoundError(f"Index '{name}' not found")

        # Whitelist of updatable fields
        allowed = {"description", "hnsw_params"}
        for key, value in updates.items():
            if key not in allowed:
                logger.warning("[INDEX-MGR] Ignoring non-updatable field: %s", key)
                continue

            if key == "hnsw_params" and isinstance(value, dict):
                # Only ef_search is hot-updatable
                if "ef_search" in value:
                    info["hnsw_params"]["ef_search"] = value["ef_search"]
                    # Also update in-memory HNSW if loaded
                    loaded = self.get_loaded_index(name)
                    if loaded and loaded.hnsw is not None:
                        try:
                            loaded.hnsw.set_ef_search(value["ef_search"])
                        except AttributeError:
                            pass
            else:
                info[key] = value

        info["updated_at"] = datetime.now(timezone.utc).isoformat()
        self._write_index_info(name, info)
        logger.info("[INDEX-MGR] Updated index %r: %s", name, list(updates.keys()))

    # -----------------------------------------------------------------
    # Load / Unload Index
    # -----------------------------------------------------------------

    def load_index(self, name: str) -> InMemoryIndex:
        """Load an index from disk into memory.

        Performs version negotiation (SRS Section 7.3).
        """
        idx_dir = self._index_path(name)
        info_path = idx_dir / INDEX_INFO_FILE
        if not info_path.exists():
            raise FileNotFoundError(f"Index '{name}' not found")

        info = json.loads(info_path.read_text(encoding="utf-8"))
        version = info.get("format_version", 1)

        # Version negotiation
        if version > SUPPORTED_FORMAT_VERSION:
            raise IndexVersionError(
                f"Index format v{version} is newer than supported v{SUPPORTED_FORMAT_VERSION}. "
                f"Please update IsoCortex to the latest version."
            )

        if version < SUPPORTED_FORMAT_VERSION:
            migration = MigrationRegistry.get_migration(version, SUPPORTED_FORMAT_VERSION)
            if migration and migration.automatic:
                logger.info(
                    "[INDEX-MGR] Auto-migrating index %r from v%d to v%d",
                    name, version, SUPPORTED_FORMAT_VERSION,
                )
                migration.migrate(idx_dir)
                info = json.loads(info_path.read_text(encoding="utf-8"))
            else:
                raise IndexVersionError(
                    f"Index format v{version} requires manual migration. "
                    f"Run: isocortex index migrate {name} --target {SUPPORTED_FORMAT_VERSION}"
                )

        # Load vectors
        vectors_path = idx_dir / VECTORS_BIN_FILE
        vectors = None
        if vectors_path.exists():
            try:
                data = np.fromfile(str(vectors_path), dtype=np.float32)
                header_size = 20  # magic(4) + version(4) + count(4) + dim(4) + type(4)
                n = int(data[8])
                d = int(data[12])
                vectors = data[header_size:].reshape(n, d).copy()
            except Exception as exc:
                logger.error("[INDEX-MGR] Failed to load vectors for %r: %s", name, exc)

        # Load metadata
        metadata: list[dict[str, Any]] = []
        meta_path = idx_dir / METADATA_JSON_FILE
        if meta_path.exists():
            try:
                meta_data = json.loads(meta_path.read_text(encoding="utf-8"))
                metadata = meta_data.get("chunks", [])
            except Exception as exc:
                logger.error("[INDEX-MGR] Failed to load metadata for %r: %s", name, exc)

        inmem = InMemoryIndex(
            name=name,
            index_info=info,
            vectors=vectors,
            metadata=metadata,
        )

        with self._registry_lock:
            self._loaded[name] = inmem

        logger.info(
            "[INDEX-MGR] Loaded index %r  vectors=%d  metadata=%d",
            name,
            vectors.shape[0] if vectors is not None else 0,
            len(metadata),
        )
        return inmem

    def unload_index(self, name: str) -> None:
        """Unload an index from memory (free RAM)."""
        with self._registry_lock:
            idx = self._loaded.pop(name, None)
        if idx is not None:
            idx.vectors = None
            idx.hnsw = None
            idx.metadata = []
            logger.info("[INDEX-MGR] Unloaded index %r", name)

    def ensure_loaded(self, name: str) -> InMemoryIndex:
        """Ensure an index is loaded, loading it if necessary."""
        loaded = self.get_loaded_index(name)
        if loaded is not None:
            return loaded
        return self.load_index(name)

    # -----------------------------------------------------------------
    # Incremental Updates (SRS FR-IDX-003)
    # -----------------------------------------------------------------

    def add_vectors(
        self,
        name: str,
        new_vectors: np.ndarray,
        new_metadata: list[dict[str, Any]],
    ) -> int:
        """Add new vectors to an existing index (incremental).

        SRS FR-IDX-003: Incremental updates without full rebuild.

        Parameters
        ----------
        name : str
            Index name.
        new_vectors : np.ndarray
            New vectors to add, shape (N, 384).
        new_metadata : list[dict]
            Metadata for each new vector.

        Returns
        -------
        int
            Number of vectors added.
        """
        inmem = self.ensure_loaded(name)
        inmem.lock.acquire_write()
        try:
            # Append vectors
            if inmem.vectors is None:
                inmem.vectors = new_vectors.copy()
            else:
                inmem.vectors = np.vstack([inmem.vectors, new_vectors])

            # Append metadata
            inmem.metadata.extend(new_metadata)

            # Update index info
            info = self._read_index_info(name) or {}
            old_count = info.get("vector_count", 0)
            info["vector_count"] = inmem.vectors.shape[0]
            info["updated_at"] = datetime.now(timezone.utc).isoformat()
            self._write_index_info(name, info)

            added = len(new_metadata)
            logger.info(
                "[INDEX-MGR] Added %d vectors to %r (total: %d)",
                added, name, info["vector_count"],
            )
            return added
        finally:
            inmem.lock.release_write()

    # -----------------------------------------------------------------
    # Soft Delete (SRS FR-IDX-002)
    # -----------------------------------------------------------------

    def soft_delete_vector(self, name: str, vector_index: int) -> bool:
        """Soft-delete a vector by marking it as tombstoned.

        SRS FR-IDX-002: Tombstone pattern — instant deletion, no graph rebuild.
        """
        inmem = self.ensure_loaded(name)
        inmem.lock.acquire_write()
        try:
            if vector_index < 0 or vector_index >= len(inmem.metadata):
                return False

            if inmem.metadata[vector_index].get("deleted", False):
                return True  # Already deleted

            inmem.metadata[vector_index]["deleted"] = True

            # Update index info
            info = self._read_index_info(name) or {}
            deleted_count = sum(1 for m in inmem.metadata if m.get("deleted", False))
            info["deleted_count"] = deleted_count
            info["updated_at"] = datetime.now(timezone.utc).isoformat()
            self._write_index_info(name, info)

            logger.info(
                "[INDEX-MGR] Soft-deleted vector %d in %r (tombstoned: %d)",
                vector_index, name, deleted_count,
            )
            return True
        finally:
            inmem.lock.release_write()

    # -----------------------------------------------------------------
    # Compaction (SRS FR-IDX-002)
    # -----------------------------------------------------------------

    def check_compaction_needed(self, name: str, threshold: float = 0.10) -> bool:
        """Check if compaction is needed based on tombstone ratio.

        SRS FR-IDX-002: Triggered when tombstoned vectors exceed threshold.
        """
        info = self._read_index_info(name)
        if info is None:
            return False

        total = info.get("vector_count", 0)
        deleted = info.get("deleted_count", 0)
        if total == 0:
            return False

        ratio = deleted / total
        logger.debug(
            "[INDEX-MGR] Compaction check %r: %d/%d (%.1f%%) threshold=%.1f%%",
            name, deleted, total, ratio * 100, threshold * 100,
        )
        return ratio >= threshold

    def compact_index(self, name: str) -> dict[str, Any]:
        """Compact an index by removing tombstoned vectors.

        SRS FR-IDX-002: Rebuilds graph with only active vectors.
        Returns compaction statistics.
        """
        inmem = self.ensure_loaded(name)
        inmem.lock.acquire_write()
        try:
            before_count = len(inmem.metadata)
            before_deleted = sum(1 for m in inmem.metadata if m.get("deleted", False))

            if before_deleted == 0:
                logger.info("[INDEX-MGR] No tombstoned vectors to compact in %r", name)
                return {"vectors_before": before_count, "vectors_after": before_count,
                        "removed": 0, "memory_saved_mb": 0.0}

            # Filter out deleted
            active_mask = [not m.get("deleted", False) for m in inmem.metadata]
            if inmem.vectors is not None:
                inmem.vectors = inmem.vectors[active_mask]
            inmem.metadata = [m for m in inmem.metadata if not m.get("deleted", False)]

            # Re-index chunk indices
            for i, meta in enumerate(inmem.metadata):
                meta["vector_index"] = i

            after_count = len(inmem.metadata)
            removed = before_count - after_count
            memory_saved = removed * BYTES_PER_VECTOR_384 / (1024 * 1024)

            # Update index info
            info = self._read_index_info(name) or {}
            info["vector_count"] = after_count
            info["deleted_count"] = 0
            info["updated_at"] = datetime.now(timezone.utc).isoformat()
            self._write_index_info(name, info)

            logger.info(
                "[INDEX-MGR] Compacted %r: %d -> %d vectors (removed %d, saved %.1f MB)",
                name, before_count, after_count, removed, memory_saved,
            )
            return {
                "vectors_before": before_count,
                "vectors_after": after_count,
                "removed": removed,
                "memory_saved_mb": round(memory_saved, 2),
            }
        finally:
            inmem.lock.release_write()

    # -----------------------------------------------------------------
    # Save Index to Disk
    # -----------------------------------------------------------------

    def save_index(self, name: str) -> None:
        """Persist a loaded index to disk (vectors.bin + metadata.json).

        SRS NFR-09: Atomic writes via temp file + os.replace().
        """
        inmem = self.ensure_loaded(name)
        inmem.lock.acquire_read()
        try:
            idx_dir = self._index_path(name)
            idx_dir.mkdir(parents=True, exist_ok=True)

            # Save vectors.bin
            if inmem.vectors is not None:
                n, d = inmem.vectors.shape
                header = np.array([
                    0x49534F56,  # Magic "ISoV"
                    SUPPORTED_FORMAT_VERSION,
                    n,
                    d,
                    0,  # float32
                ], dtype=np.uint32)
                target = idx_dir / VECTORS_BIN_FILE
                tmp = target.with_suffix(".tmp")
                with open(tmp, "wb") as f:
                    header.tofile(f)
                    inmem.vectors.astype(np.float32).tofile(f)
                tmp.replace(target)

            # Save metadata.json
            meta_data = {
                "chunks": inmem.metadata,
                "total_chunks": len(inmem.metadata),
                "total_deleted": sum(1 for m in inmem.metadata if m.get("deleted", False)),
            }
            target = idx_dir / METADATA_JSON_FILE
            tmp = target.with_suffix(".tmp")
            tmp.write_text(json.dumps(meta_data, indent=2, default=str), encoding="utf-8")
            tmp.replace(target)

            logger.info("[INDEX-MGR] Saved index %r to disk", name)
        finally:
            inmem.lock.release_read()

    # -----------------------------------------------------------------
    # Memory Estimation (SRS NFR-16)
    # -----------------------------------------------------------------

    @staticmethod
    def estimate_memory_mb(vector_count: int, dimension: int = 384) -> float:
        """Estimate memory required for a given number of vectors.

        SRS NFR-16: Pre-indexing memory check.
        """
        raw_mb = (vector_count * dimension * 4) / (1024 * 1024)
        hnsw_overhead_mb = raw_mb * HNSW_OVERHEAD_MULTIPLIER
        total_mb = raw_mb + hnsw_overhead_mb + PYTHON_RUNTIME_OVERHEAD_MB
        return total_mb

    @staticmethod
    def get_system_memory_mb() -> float:
        """Get total system RAM in MB."""
        try:
            return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024 * 1024)
        except (ValueError, OSError):
            try:
                import psutil
                return psutil.virtual_memory().total / (1024 * 1024)
            except ImportError:
                return 0.0

    def check_memory_feasibility(
        self,
        vector_count: int,
        max_memory_mb: float | None = None,
    ) -> dict[str, Any]:
        """Check if creating an index with given vectors is feasible.

        SRS NFR-16: Pre-indexing memory check with warnings.

        Returns a dict with:
        - feasible: bool
        - estimated_mb: float
        - system_mb: float
        - warning: str | None
        """
        estimated = self.estimate_memory_mb(vector_count)
        system = self.get_system_memory_mb()

        if max_memory_mb is None:
            max_memory_mb = system * 0.80

        warning = None
        if estimated > system * 0.80 and estimated <= max_memory_mb:
            warning = (
                f"Estimated memory ({estimated:.0f} MB) exceeds 80% of system RAM "
                f"({system:.0f} MB). Operation will proceed but may be slow."
            )

        feasible = estimated <= max_memory_mb

        return {
            "feasible": feasible,
            "estimated_mb": round(estimated, 1),
            "system_mb": round(system, 1),
            "max_memory_mb": round(max_memory_mb, 1),
            "warning": warning,
        }

    # -----------------------------------------------------------------
    # Export / Import (SRS FR-API-006)
    # -----------------------------------------------------------------

    def export_index(self, name: str, output_path: Path) -> dict[str, Any]:
        """Export an index as a .isocortex archive.

        SRS FR-API-006: POST /api/v1/indexes/{name}/export

        Creates a tar.gz archive containing all index files.
        """
        idx_dir = self._index_path(name)
        if not idx_dir.exists():
            raise FileNotFoundError(f"Index '{name}' not found")

        import tarfile

        output_path = Path(output_path)
        if not output_path.suffix == ARCHIVE_EXTENSION:
            output_path = output_path.with_suffix(ARCHIVE_EXTENSION)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        t0 = time.time()
        with tarfile.open(str(output_path), "w:gz") as tar:
            for f in idx_dir.iterdir():
                if f.is_file():
                    tar.add(str(f), arcname=f.name)

        archive_size = output_path.stat().st_size
        duration = time.time() - t0

        # Compute SHA-256 checksum
        sha256 = hashlib.sha256()
        with open(output_path, "rb") as f:
            for block in iter(lambda: f.read(65536), b""):
                sha256.update(block)

        result = {
            "archive_path": str(output_path),
            "archive_size_mb": round(archive_size / (1024 * 1024), 2),
            "duration_seconds": round(duration, 2),
            "sha256_checksum": sha256.hexdigest(),
        }

        logger.info(
            "[INDEX-MGR] Exported %r -> %s (%.1f MB, %.1fs)",
            name, output_path, result["archive_size_mb"], duration,
        )
        return result

    def import_index(self, archive_path: Path, name: str | None = None) -> str:
        """Import an index from a .isocortex archive.

        SRS FR-API-006: POST /api/v1/indexes/import

        Returns the index name.
        """
        import tarfile

        archive_path = Path(archive_path)
        if not archive_path.exists():
            raise FileNotFoundError(f"Archive not found: {archive_path}")

        t0 = time.time()

        # Create temporary extraction directory
        tmp_dir = self._indices_dir / f"_import_tmp_{uuid.uuid4().hex[:8]}"
        tmp_dir.mkdir(parents=True, exist_ok=True)

        try:
            with tarfile.open(str(archive_path), "r:gz") as tar:
                tar.extractall(str(tmp_dir))

            # Read index info to determine name
            info_path = tmp_dir / INDEX_INFO_FILE
            if not info_path.exists():
                raise ValueError(
                    f"Invalid archive: missing {INDEX_INFO_FILE}. "
                    "Not a valid IsoCortex index archive."
                )

            info = json.loads(info_path.read_text(encoding="utf-8"))
            index_name = name or info.get("name", f"imported-{uuid.uuid4().hex[:8]}")

            # Version negotiation
            version = info.get("format_version", 1)
            if version > SUPPORTED_FORMAT_VERSION:
                raise IndexVersionError(
                    f"Archive format v{version} is newer than supported v{SUPPORTED_FORMAT_VERSION}."
                )

            # Move to final location
            target_dir = self._index_path(index_name)
            if target_dir.exists():
                shutil.rmtree(target_dir)
            shutil.move(str(tmp_dir), str(target_dir))

            duration = time.time() - t0
            logger.info(
                "[INDEX-MGR] Imported %r from %s (%.1fs)",
                index_name, archive_path, duration,
            )
            return index_name

        except Exception:
            # Cleanup on failure
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise

    # -----------------------------------------------------------------
    # Search Helpers
    # -----------------------------------------------------------------

    def get_search_components(self, name: str) -> tuple:
        """Get (search_fn, metadata_getter, vector_count_fn) for the SearchEngine.

        Returns callables that interface with the loaded index.
        Automatically selects the C++ HNSW engine when available,
        falling back to pure-Python brute-force search otherwise.

        SRS Section 3.5: HNSW Index
        SRS FR-IDX-001: Index construction
        NFR-01: p95 < 100ms search latency (requires C++ HNSW for large indices)
        """
        inmem = self.ensure_loaded(name)

        # Determine whether to use native C++ HNSW or Python fallback.
        # Native C++ HNSW is used when:
        #   1. The pybind11 native extension is loaded
        #   2. The index has a built HNSW graph binary file on disk
        #   3. Vectors are available in memory
        use_native = False
        try:
            from isocortex.core.hnsw import get_native_module, _native_available

            if _native_available:
                idx_dir = self._index_path(name)
                hnsw_bin = idx_dir / HNSW_INDEX_BIN_FILE
                if hnsw_bin.exists() and inmem.vectors is not None:
                    use_native = True
        except Exception:
            pass

        if use_native:
            search_fn = self._get_native_search_fn(inmem)
        else:
            search_fn = self._get_fallback_search_fn(inmem)

        def metadata_getter() -> list[dict[str, Any]]:
            """Return all metadata entries."""
            inmem.lock.acquire_read()
            try:
                return list(inmem.metadata)
            finally:
                inmem.lock.release_read()

        def vector_count_fn() -> int:
            """Return total active (non-deleted) vector count."""
            inmem.lock.acquire_read()
            try:
                return sum(
                    1 for m in inmem.metadata if not m.get("deleted", False)
                )
            finally:
                inmem.lock.release_read()

        return search_fn, metadata_getter, vector_count_fn
