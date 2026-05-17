"""
IsoCortex — Config Tests
=========================
Tests for configuration defaults, validation, env loading,
and directory creation.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from isocortex.config.settings import (
    DEFAULT_API_HOST,
    DEFAULT_API_PORT,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_COMPACTION_THRESHOLD,
    DEFAULT_HNSW_EF_CONSTRUCTION,
    DEFAULT_HNSW_EF_SEARCH,
    DEFAULT_HNSW_M,
    DEFAULT_TOKEN_LIMIT,
    DEFAULT_VECTOR_DIM,
    AppConfig,
    ChunkingConfig,
    EmbeddingConfig,
    HnswConfig,
    RateLimitConfig,
    ServerConfig,
    StorageConfig,
)


# =============================================================================
# Default Config Tests
# =============================================================================

class TestDefaultConfig:

    def test_default_config_values(self):
        """Verify all defaults."""
        hnsw = HnswConfig()
        assert hnsw.M == DEFAULT_HNSW_M
        assert hnsw.ef_construction == DEFAULT_HNSW_EF_CONSTRUCTION
        assert hnsw.ef_search == DEFAULT_HNSW_EF_SEARCH
        assert hnsw.dim == DEFAULT_VECTOR_DIM
        assert hnsw.metric == "cosine"
        assert hnsw.max_elements == 100000

        embedding = EmbeddingConfig()
        assert embedding.batch_size == 64
        assert embedding.cache_size == 1000
        assert embedding.vector_dim == 384

        chunking = ChunkingConfig()
        assert chunking.chunk_size == DEFAULT_CHUNK_SIZE
        assert chunking.overlap == DEFAULT_CHUNK_OVERLAP
        assert chunking.token_limit == DEFAULT_TOKEN_LIMIT

        server = ServerConfig()
        assert server.host == DEFAULT_API_HOST
        assert server.port == DEFAULT_API_PORT
        assert server.workers == 1

        storage = StorageConfig()
        assert storage.compaction_threshold == DEFAULT_COMPACTION_THRESHOLD

        rate_limit = RateLimitConfig()
        assert rate_limit.requests == 100
        assert rate_limit.window_seconds == 60


# =============================================================================
# HNSW Config Validation
# =============================================================================

class TestHnswConfigValidation:

    def test_hnsw_valid_ranges(self):
        """M, ef_construction, ef_search in valid range."""
        hnsw = HnswConfig(M=32, ef_construction=500, ef_search=200)
        assert hnsw.M == 32
        assert hnsw.ef_construction == 500
        assert hnsw.ef_search == 200

    def test_hnsw_m_out_of_range(self):
        """M outside 1-128 raises ValueError."""
        with pytest.raises(ValueError, match="M"):
            HnswConfig(M=0)
        with pytest.raises(ValueError, match="M"):
            HnswConfig(M=200)

    def test_hnsw_ef_construction_out_of_range(self):
        """ef_construction outside 1-2000 raises ValueError."""
        with pytest.raises(ValueError, match="ef_construction"):
            HnswConfig(ef_construction=0)
        with pytest.raises(ValueError, match="ef_construction"):
            HnswConfig(ef_construction=3000)

    def test_hnsw_ef_search_out_of_range(self):
        """ef_search outside 1-500 raises ValueError."""
        with pytest.raises(ValueError, match="ef_search"):
            HnswConfig(ef_search=0)
        with pytest.raises(ValueError, match="ef_search"):
            HnswConfig(ef_search=600)

    def test_hnsw_invalid_dim(self):
        """dim != 384 raises ValueError."""
        with pytest.raises(ValueError, match="dim"):
            HnswConfig(dim=128)

    def test_hnsw_invalid_metric(self):
        """Invalid metric raises ValueError."""
        with pytest.raises(ValueError, match="metric"):
            HnswConfig(metric="hamming")

    def test_hnsw_to_dict(self):
        """Verify serialization."""
        hnsw = HnswConfig()
        d = hnsw.to_dict()
        assert d["M"] == 16
        assert d["efConstruction"] == 200
        assert d["efSearch"] == 50
        assert d["space"] == "cosine"


# =============================================================================
# Config from Environment
# =============================================================================

class TestConfigFromEnv:

    def test_config_from_env(self, monkeypatch):
        """Environment variable loading."""
        monkeypatch.setenv("ISOCORTEX_HOST", "127.0.0.1")
        monkeypatch.setenv("ISOCORTEX_PORT", "9999")
        monkeypatch.setenv("ISOCORTEX_WORKERS", "4")
        monkeypatch.setenv("ISOCORTEX_JWT_SECRET", "test-secret")
        monkeypatch.setenv("ISOCORTEX_JWT_ALGORITHM", "HS512")
        monkeypatch.setenv("ISOCORTEX_JWT_EXPIRE", "60")

        config = ServerConfig.from_env()
        assert config.host == "127.0.0.1"
        assert config.port == 9999
        assert config.workers == 4
        assert config.jwt_secret_key == "test-secret"
        assert config.jwt_algorithm == "HS512"
        assert config.jwt_expire_minutes == 60

    def test_storage_config_from_env(self, monkeypatch, tmp_path):
        """Storage config from env vars."""
        monkeypatch.setenv("ISOCORTEX_DATA_DIR", str(tmp_path / "custom_data"))
        monkeypatch.setenv("ISOCORTEX_DB_NAME", "custom.db")
        monkeypatch.setenv("ISOCORTEX_MAX_FILE_SIZE_MB", "100")
        monkeypatch.setenv("ISOCORTEX_COMPACTION_THRESHOLD", "0.20")

        config = StorageConfig.from_env()
        assert config.data_dir == tmp_path / "custom_data"
        assert config.db_name == "custom.db"
        assert config.max_file_size_mb == 100.0
        assert config.compaction_threshold == 0.20


# =============================================================================
# Config Directory Creation
# =============================================================================

class TestConfigDirectories:

    def test_config_ensure_directories(self, tmp_path: Path):
        """Directory creation."""
        config = AppConfig(
            storage=StorageConfig(data_dir=tmp_path / "isocortex_test"),
        )
        config.ensure_directories()

        assert (tmp_path / "isocortex_test").is_dir()
        assert (tmp_path / "isocortex_test" / "models").is_dir()
        assert (tmp_path / "isocortex_test" / "indices").is_dir()


# =============================================================================
# AppConfig Tests
# =============================================================================

class TestAppConfig:

    def test_app_config_to_dict(self):
        """Full config serialization."""
        config = AppConfig()
        d = config.to_dict()
        assert "hnsw" in d
        assert "embedding" in d
        assert "chunking" in d
        assert "server" in d
        assert "storage" in d
        assert "rate_limit" in d

    def test_app_config_from_file(self, tmp_path: Path):
        """Load config from JSON file."""
        config_data = {
            "hnsw": {"M": 32, "efConstruction": 400, "efSearch": 100,
                     "max_elements": 50000, "dim": 384, "space": "cosine"},
            "embedding": {"model_name": "custom-model", "batch_size": 128,
                          "cache_size": 500, "queue_size": 200, "vector_dim": 384},
            "chunking": {"chunk_size": 200, "overlap": 20, "token_limit": 256},
            "storage": {"data_dir": str(tmp_path / "file_data"),
                        "max_file_size_mb": 100, "compaction_threshold": 0.15},
            "rate_limit": {"requests": 50, "window_seconds": 30},
        }
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(config_data), encoding="utf-8")

        config = AppConfig.from_file(config_path)
        assert config.hnsw.M == 32
        assert config.embedding.batch_size == 128
        assert config.chunking.chunk_size == 200

    def test_app_config_from_missing_file(self, tmp_path: Path):
        """Missing config file returns defaults."""
        config = AppConfig.from_file(tmp_path / "nonexistent.json")
        assert config.hnsw.M == DEFAULT_HNSW_M
