"""
IsoCortex — Test Fixtures
==========================
Shared pytest fixtures for the test suite.
"""

from __future__ import annotations

import os
import sqlite3
import threading
from pathlib import Path
from typing import Any

import numpy as np
import pytest


# =============================================================================
# Paths
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
TESTS_DIR = PROJECT_ROOT / "tests"


# =============================================================================
# Temporary directory fixture (built-in, re-exported for clarity)
# =============================================================================


# =============================================================================
# In-memory SQLite database fixture
# =============================================================================

@pytest.fixture()
def db_path(tmp_path: Path) -> Path:
    """Return a path to a temporary SQLite database file."""
    return tmp_path / "test.db"


@pytest.fixture()
def sqlite_conn(db_path: Path) -> sqlite3.Connection:
    """Create a fresh SQLite connection with WAL mode."""
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.row_factory = sqlite3.Row
    yield conn
    conn.close()


@pytest.fixture()
def sqlite_conn_inmemory() -> sqlite3.Connection:
    """In-memory SQLite connection (no file I/O)."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    yield conn
    conn.close()


# =============================================================================
# Mock Configuration
# =============================================================================

@pytest.fixture()
def mock_config(tmp_path: Path) -> dict[str, Any]:
    """Return a mock AppConfig-like dict for testing."""
    return {
        "hnsw": {
            "M": 16,
            "efConstruction": 200,
            "efSearch": 50,
            "dim": 384,
            "metric": "cosine",
        },
        "embedding": {
            "model_name": "test-model",
            "batch_size": 32,
            "cache_size": 100,
            "vector_dim": 384,
        },
        "chunking": {
            "chunk_size": 350,
            "overlap": 38,
            "token_limit": 512,
        },
        "storage": {
            "data_dir": str(tmp_path / "data"),
            "db_name": "test.db",
        },
    }


# =============================================================================
# Mock Embeddings
# =============================================================================

@pytest.fixture()
def mock_embed_fn():
    """Return a deterministic mock embedding function (384-dim)."""
    rng = np.random.default_rng(42)

    def _embed(text: str) -> np.ndarray:
        # Deterministic embedding based on text hash
        vec = rng.standard_normal(384).astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    return _embed


@pytest.fixture()
def sample_vectors() -> np.ndarray:
    """Return 10 normalized 384-dim vectors."""
    rng = np.random.default_rng(42)
    vecs = rng.standard_normal((10, 384)).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return vecs / norms


@pytest.fixture()
def sample_metadata() -> list[dict[str, Any]]:
    """Return 10 metadata records matching sample_vectors."""
    return [
        {
            "id": f"vec-{i}",
            "text_preview": f"Sample text chunk {i}",
            "file_extension": ".txt",
            "source_file": f"/docs/file_{i}.txt",
            "deleted": False,
        }
        for i in range(10)
    ]


@pytest.fixture()
def sample_query_vector() -> np.ndarray:
    """Return a single normalized 384-dim query vector."""
    rng = np.random.default_rng(123)
    vec = rng.standard_normal(384).astype(np.float32)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec


# =============================================================================
# Mock Index Manager (with temp directory)
# =============================================================================

@pytest.fixture()
def indices_dir(tmp_path: Path) -> Path:
    """Return a temporary directory for index data."""
    d = tmp_path / "indices"
    d.mkdir(parents=True, exist_ok=True)
    return d


# =============================================================================
# Mock search function
# =============================================================================

@pytest.fixture()
def mock_search_fn(sample_vectors: np.ndarray, sample_metadata: list[dict]):
    """Return a mock search function that returns cosine-sorted results."""
    def _search(query_vec: np.ndarray, k: int) -> list[tuple[int, float]]:
        # Compute cosine distances
        norms = np.linalg.norm(sample_vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1
        normed = sample_vectors / norms
        q_norm = np.linalg.norm(query_vec)
        if q_norm == 0:
            return [(i, 1.0) for i in range(min(k, len(sample_vectors)))]
        q_normed = query_vec / q_norm
        similarities = normed @ q_normed
        distances = 1.0 - similarities
        k = min(k, len(distances))
        sorted_idx = np.argsort(distances)[:k]
        return [(int(i), float(distances[i])) for i in sorted_idx]
    return _search


@pytest.fixture()
def mock_metadata_getter(sample_metadata: list[dict]):
    """Return a mock metadata getter function."""
    return lambda: list(sample_metadata)


@pytest.fixture()
def mock_vector_count_fn(sample_vectors: np.ndarray):
    """Return a mock vector count function."""
    return lambda: sample_vectors.shape[0]


# =============================================================================
# Mock ExtractionResult (for chunker tests)
# =============================================================================

@pytest.fixture()
def mock_extraction_result():
    """Return a mock ExtractionResult-like object for chunker tests."""
    class MockExtractionResult:
        def __init__(self):
            self.absolute_path = Path("/test/document.txt")
            self.format_category = "plain_text"
            self.success = True
            self.chunks = [
                MockChunk(
                    text="This is the first sentence. It introduces the topic. "
                         "The second sentence provides more context. "
                         "Third sentence adds details about the subject. "
                         "Fourth sentence concludes the first paragraph.",
                    source_label="File: document.txt",
                ),
                MockChunk(
                    text="Another paragraph starts here. It discusses a related topic. "
                         "This sentence elaborates further on the idea. "
                         "The final sentence wraps up the discussion.",
                    source_label="File: document.txt",
                ),
            ]

    class MockChunk:
        def __init__(self, text: str, source_label: str):
            self.text = text
            self.source_label = source_label

    return MockExtractionResult()


# =============================================================================
# Environment variable fixtures
# =============================================================================

@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Clear IsoCortex env vars between tests."""
    env_vars = [v for v in os.environ if v.startswith("ISOCORTEX_")]
    saved = {v: os.environ.pop(v) for v in env_vars}
    yield
    for v, val in saved.items():
        os.environ[v] = val
