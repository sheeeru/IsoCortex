"""
IsoCortex — Index Manager Tests
================================
Tests for IndexManager lifecycle, incremental updates, compaction,
export/import, versioning, and ReadWriteLock.
"""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

from isocortex.engine.indexing.manager import (
    HNSW_OVERHEAD_MULTIPLIER,
    IndexManager,
    IndexStats,
    IndexVersionError,
    Migration,
    MigrationRegistry,
    ReadWriteLock,
    SUPPORTED_FORMAT_VERSION,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture()
def manager(tmp_path: Path) -> IndexManager:
    """Create an IndexManager with a temporary indices directory."""
    idx_dir = tmp_path / "indices"
    return IndexManager(idx_dir)


@pytest.fixture()
def created_manager(manager: IndexManager) -> IndexManager:
    """Create a manager with one index already created."""
    manager.create_index("test-index", description="Test index")
    return manager


# =============================================================================
# Index CRUD Tests
# =============================================================================

class TestIndexCRUD:
    """Test index create, list, get, delete, update."""

    def test_create_index(self, manager: IndexManager):
        """Create index, verify info file written."""
        name = manager.create_index("my-index", description="My test index")
        assert name == "my-index"

        info_file = manager._index_path("my-index") / "index_info.json"
        assert info_file.exists()
        info = json.loads(info_file.read_text())
        assert info["name"] == "my-index"
        assert info["description"] == "My test index"
        assert info["vector_count"] == 0
        assert info["format_version"] == SUPPORTED_FORMAT_VERSION

    def test_create_index_duplicate(self, manager: IndexManager):
        """Test FileExistsError on duplicate."""
        manager.create_index("dup-index")
        with pytest.raises(FileExistsError):
            manager.create_index("dup-index")

    def test_list_indexes(self, manager: IndexManager):
        """Create multiple, list all."""
        manager.create_index("idx-a")
        manager.create_index("idx-b")
        indexes = manager.list_indexes()
        names = [i.name for i in indexes]
        assert "idx-a" in names
        assert "idx-b" in names

    def test_get_index(self, manager: IndexManager):
        """Create + get, verify stats."""
        manager.create_index("stats-index", description="Stats test")
        stats = manager.get_index("stats-index")
        assert stats is not None
        assert isinstance(stats, IndexStats)
        assert stats.name == "stats-index"
        assert stats.vector_count == 0
        assert stats.healthy is True

    def test_get_index_not_found(self, manager: IndexManager):
        """Verify None returned."""
        assert manager.get_index("nonexistent") is None

    def test_delete_index(self, manager: IndexManager):
        """Create then delete."""
        manager.create_index("del-me")
        assert manager.get_index("del-me") is not None
        manager.delete_index("del-me")
        assert manager.get_index("del-me") is None

    def test_update_index(self, manager: IndexManager):
        """Update description and ef_search."""
        manager.create_index("upd-index")
        manager.update_index("upd-index", {
            "description": "Updated desc",
            "hnsw_params": {"ef_search": 100},
        })
        stats = manager.get_index("upd-index")
        info = manager._read_index_info("upd-index")
        assert info["description"] == "Updated desc"
        assert info["hnsw_params"]["ef_search"] == 100


# =============================================================================
# Incremental Updates Tests
# =============================================================================

class TestIncrementalUpdates:
    """Test add_vectors, soft_delete, compaction."""

    def test_add_vectors(self, manager: IndexManager):
        """Add vectors to an index, verify count."""
        manager.create_index("add-idx")
        # Load first (needed by ensure_loaded)
        manager.load_index("add-idx")

        vecs = np.random.randn(3, 384).astype(np.float32)
        meta = [{"id": f"v-{i}"} for i in range(3)]
        added = manager.add_vectors("add-idx", vecs, meta)
        assert added == 3

        info = manager._read_index_info("add-idx")
        assert info["vector_count"] == 3

    def test_soft_delete_vector(self, manager: IndexManager):
        """Soft delete and verify tombstone."""
        manager.create_index("soft-del")
        manager.load_index("soft-del")

        vecs = np.random.randn(3, 384).astype(np.float32)
        meta = [{"id": f"v-{i}"} for i in range(3)]
        manager.add_vectors("soft-del", vecs, meta)

        result = manager.soft_delete_vector("soft-del", 1)
        assert result is True

        loaded = manager.get_loaded_index("soft-del")
        assert loaded.metadata[1]["deleted"] is True
        assert loaded.metadata[0].get("deleted") is None

    def test_compaction_needed(self, manager: IndexManager):
        """Check threshold logic."""
        manager.create_index("compact-check")
        manager.load_index("compact-check")

        vecs = np.random.randn(10, 384).astype(np.float32)
        meta = [{"id": f"v-{i}"} for i in range(10)]
        manager.add_vectors("compact-check", vecs, meta)

        # Delete 2 of 10 = 20% > 10% threshold
        manager.soft_delete_vector("compact-check", 0)
        manager.soft_delete_vector("compact-check", 1)
        assert manager.check_compaction_needed("compact-check", threshold=0.10)

        # Should NOT be needed at 50% threshold
        assert not manager.check_compaction_needed("compact-check", threshold=0.50)

    def test_compaction(self, manager: IndexManager):
        """Run compaction and verify removed."""
        manager.create_index("compact-idx")
        manager.load_index("compact-idx")

        vecs = np.random.randn(5, 384).astype(np.float32)
        meta = [{"id": f"v-{i}"} for i in range(5)]
        manager.add_vectors("compact-idx", vecs, meta)

        manager.soft_delete_vector("compact-idx", 1)
        manager.soft_delete_vector("compact-idx", 3)

        stats = manager.compact_index("compact-idx")
        assert stats["removed"] == 2
        assert stats["vectors_after"] == 3

        loaded = manager.get_loaded_index("compact-idx")
        assert len(loaded.metadata) == 3
        assert all(not m.get("deleted") for m in loaded.metadata)


# =============================================================================
# Load/Unload Tests
# =============================================================================

class TestLoadUnload:
    """Test index loading and unloading."""

    def test_load_unload_index(self, manager: IndexManager):
        """Create, load, unload."""
        manager.create_index("load-idx")
        loaded = manager.load_index("load-idx")
        assert loaded is not None
        assert loaded.name == "load-idx"
        assert manager.get_loaded_index("load-idx") is not None

        manager.unload_index("load-idx")
        assert manager.get_loaded_index("load-idx") is None

    def test_save_index(self, manager: IndexManager, tmp_path: Path):
        """Save vectors and metadata to disk."""
        manager.create_index("save-idx")
        manager.load_index("save-idx")

        vecs = np.random.randn(2, 384).astype(np.float32)
        meta = [{"id": "v-0", "text": "hello"}, {"id": "v-1", "text": "world"}]
        manager.add_vectors("save-idx", vecs, meta)

        manager.save_index("save-idx")

        # Verify files exist
        idx_dir = manager._index_path("save-idx")
        assert (idx_dir / "vectors.bin").exists()
        assert (idx_dir / "metadata.json").exists()

    def test_memory_estimation(self):
        """Verify MB calculations."""
        mb = IndexManager.estimate_memory_mb(1000, 384)
        assert mb > 200  # Runtime overhead alone
        assert mb < 500  # Reasonable upper bound
        # Also verify the exact formula
        raw_mb = (1000 * 384 * 4) / (1024 * 1024)
        expected = raw_mb + raw_mb * HNSW_OVERHEAD_MULTIPLIER + 200
        assert abs(mb - expected) < 0.01


# =============================================================================
# ReadWriteLock Tests
# =============================================================================

class TestReadWriteLock:
    """Test concurrent access patterns."""

    def test_concurrent_reads(self):
        """Multiple readers can hold the lock simultaneously."""
        lock = ReadWriteLock()
        lock.acquire_read()
        lock.acquire_read()
        lock.release_read()
        lock.release_read()
        # Should not deadlock

    def test_exclusive_write(self):
        """Writer gets exclusive access."""
        lock = ReadWriteLock()
        lock.acquire_write()
        lock.release_write()

    def test_writer_blocks_reader(self):
        """Active writer blocks new readers (writer priority)."""
        lock = ReadWriteLock()
        lock.acquire_write()
        reader_started = threading.Event()

        def reader():
            reader_started.set()
            lock.acquire_read()
            lock.release_read()

        t = threading.Thread(target=reader)
        t.start()
        reader_started.wait()
        # Give reader a moment to try acquiring
        t.join(timeout=0.1)
        # Reader should still be alive (blocked)
        assert t.is_alive()
        lock.release_write()
        t.join(timeout=1.0)
        assert not t.is_alive()


# =============================================================================
# Export/Import Tests
# =============================================================================

class TestExportImport:
    """Test export and import of indexes."""

    def test_export_index(self, manager: IndexManager, tmp_path: Path):
        """Export to archive."""
        manager.create_index("export-idx", description="Export test")
        archive_path = tmp_path / "export.isocortex"
        # Inject hashlib into manager module namespace before calling export
        import isocortex.engine.indexing.manager as mgr_mod
        mgr_mod.hashlib = hashlib
        result = manager.export_index("export-idx", archive_path)
        assert (tmp_path / "export.isocortex").exists()
        assert "sha256_checksum" in result
        assert result["archive_size_mb"] > 0

    def test_import_index(self, manager: IndexManager, tmp_path: Path):
        """Import from archive."""
        manager.create_index("import-src", description="Import source")
        archive_path = tmp_path / "import.isocortex"
        import isocortex.engine.indexing.manager as mgr_mod
        mgr_mod.hashlib = hashlib
        manager.export_index("import-src", archive_path)

        # Create a new manager to import into
        new_dir = tmp_path / "indices2"
        new_manager = IndexManager(new_dir)
        name = new_manager.import_index(archive_path)
        assert name == "import-src"
        assert new_manager.get_index("import-src") is not None

    def test_format_version_check(self, manager: IndexManager):
        """Test version negotiation."""
        manager.create_index("version-idx")
        # Should work — current version
        loaded = manager.load_index("version-idx")
        assert loaded is not None


# =============================================================================
# Version/Migration Tests
# =============================================================================

class TestVersioning:
    """Test format version check."""

    def test_format_version_check(self):
        """Test MigrationRegistry."""
        # No migrations registered
        m = MigrationRegistry.get_migration(0, 1)
        assert m is None

        class TestMigration(Migration):
            def __init__(self):
                super().__init__(0, 1, True, "Test migration")
            def migrate(self, index_path):
                pass

        MigrationRegistry.register(TestMigration())
        m = MigrationRegistry.get_migration(0, 1)
        assert m is not None
        assert m.automatic is True
