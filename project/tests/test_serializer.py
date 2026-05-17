"""
IsoCortex — Serializer Tests
=============================
Tests for vector/metadata/config export and load,
atomic writes, and integrity validation.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from isocortex.storage.serializer import (
    VECTOR_DIM,
    LoadedIndex,
    PipelineOutput,
    export_config,
    export_metadata_json,
    export_pipeline_result,
    export_vectors_bin,
    load_config,
    load_metadata_json,
    load_pipeline_result,
    load_vectors_bin,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture()
def sample_vectors() -> np.ndarray:
    """Generate 5 normalized 384-dim vectors."""
    rng = np.random.default_rng(42)
    vecs = rng.standard_normal((5, VECTOR_DIM)).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return vecs / norms


@pytest.fixture()
def sample_metadata() -> list[dict]:
    """Generate metadata matching sample_vectors."""
    return [
        {
            "chunk_id": i,
            "text": f"Sample text chunk {i}",
            "source_file": f"/test/file_{i}.txt",
            "source_label": f"File: file_{i}.txt",
            "format_category": "plain_text",
            "word_count": 4,
            "token_count": 6,
            "source_chunk_index": 0,
        }
        for i in range(5)
    ]


# =============================================================================
# Vector Export/Load Tests
# =============================================================================

class TestVectorExportLoad:

    def test_export_vectors_bin_and_load(self, sample_vectors: np.ndarray, tmp_path: Path):
        """Round-trip vectors through export/load."""
        out_path = tmp_path / "vectors.bin"
        export_vectors_bin(sample_vectors, out_path)
        assert out_path.exists()

        loaded = load_vectors_bin(out_path)
        assert loaded.shape == sample_vectors.shape
        assert np.allclose(loaded, sample_vectors, atol=1e-6)

    def test_export_vectors_empty_raises(self, tmp_path: Path):
        """Empty vector matrix raises ValueError."""
        empty = np.zeros((0, VECTOR_DIM), dtype=np.float32)
        with pytest.raises(ValueError, match="zero vectors"):
            export_vectors_bin(empty, tmp_path / "empty.bin")

    def test_export_vectors_wrong_dim_raises(self, tmp_path: Path):
        """Wrong dimension raises ValueError."""
        wrong = np.zeros((5, 128), dtype=np.float32)
        with pytest.raises(ValueError, match="dim"):
            export_vectors_bin(wrong, tmp_path / "wrong.bin")


# =============================================================================
# Metadata Export/Load Tests
# =============================================================================

class TestMetadataExportLoad:

    def test_export_metadata_json_and_load(self, sample_metadata: list[dict], tmp_path: Path):
        """Round-trip metadata through export/load."""
        out_path = tmp_path / "metadata.json"
        export_metadata_json(sample_metadata, out_path)
        assert out_path.exists()

        loaded = load_metadata_json(out_path)
        assert len(loaded) == len(sample_metadata)
        assert loaded[0]["chunk_id"] == 0
        assert loaded[0]["text"] == "Sample text chunk 0"

    def test_export_metadata_empty_raises(self, tmp_path: Path):
        """Empty metadata raises ValueError."""
        with pytest.raises(ValueError, match="empty"):
            export_metadata_json([], tmp_path / "empty.json")


# =============================================================================
# Corrupted Data Tests
# =============================================================================

class TestCorruptedData:

    def test_load_vectors_corrupted_header(self, tmp_path: Path):
        """Handle bad data (file too small)."""
        bad_path = tmp_path / "bad.bin"
        bad_path.write_bytes(b"\x00\x00")  # Only 2 bytes
        with pytest.raises(ValueError, match="too small"):
            load_vectors_bin(bad_path)

    def test_load_vectors_missing_file(self, tmp_path: Path):
        """Missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_vectors_bin(tmp_path / "nonexistent.bin")

    def test_load_metadata_missing_file(self, tmp_path: Path):
        """Missing metadata file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_metadata_json(tmp_path / "nonexistent.json")

    def test_load_metadata_not_a_list(self, tmp_path: Path):
        """Non-list metadata raises ValueError."""
        bad_path = tmp_path / "bad_meta.json"
        bad_path.write_text('{"not": "a list"}', encoding="utf-8")
        with pytest.raises(ValueError, match="must be a list"):
            load_metadata_json(bad_path)


# =============================================================================
# Config Export/Load Tests
# =============================================================================

class TestConfigExportLoad:

    def test_export_and_load_config(self, tmp_path: Path):
        """Round-trip config."""
        config = {
            "M": 16,
            "efConstruction": 200,
            "efSearch": 50,
            "dim": VECTOR_DIM,
            "space": "cosine",
        }
        config_path = tmp_path / "config.json"
        export_config(config, config_path)
        loaded = load_config(config_path)
        assert loaded["M"] == 16
        assert loaded["efConstruction"] == 200
        assert loaded["space"] == "cosine"

    def test_load_config_missing_keys(self, tmp_path: Path):
        """Missing required keys raises ValueError."""
        bad_path = tmp_path / "bad_config.json"
        bad_path.write_text('{"M": 16}', encoding="utf-8")
        with pytest.raises(ValueError, match="missing required"):
            load_config(bad_path)


# =============================================================================
# Pipeline Export/Load Tests
# =============================================================================

class TestPipelineExportLoad:

    def test_export_and_load_pipeline(self, sample_vectors: np.ndarray, sample_metadata: list[dict], tmp_path: Path):
        """Full pipeline round-trip."""
        output = export_pipeline_result(
            sample_vectors,
            sample_metadata,
            tmp_path / "pipeline_out",
            total_tokens=30,
            elapsed_seconds=1.5,
        )
        assert isinstance(output, PipelineOutput)
        assert output.vector_count == 5
        assert output.total_chunks == 5

        loaded = load_pipeline_result(tmp_path / "pipeline_out")
        assert isinstance(loaded, LoadedIndex)
        assert loaded.vector_count == 5
        assert loaded.vector_dim == VECTOR_DIM
