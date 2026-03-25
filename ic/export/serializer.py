"""
IsoCortex — export/serializer.py
=================================
Serialization engine for vectors, metadata, and HNSW indices.

Responsibilities (FR-4, FR-7):
  - Export EmbeddingResult to vectors.bin + metadata.json.
  - Load vectors.bin and metadata.json back for HNSW ingestion.
  - Manage HNSW index file persistence (save/load).
  - Support incremental index updates (UC-3/FR-7).
  - Export and load configuration (config.json).
  - Provide atomic write operations to prevent data corruption.
  - Validate file integrity at every load point.
  - Emit structured log messages at every decision point.

SRS References: FR-4, FR-7, UC-3, NFR-3, CON-5

Author : Shaheer Qureshi
Project: IsoCortex
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
VECTOR_DIM: int             = 384
DEFAULT_VECTORS_FILENAME:   str = "vectors.bin"
DEFAULT_METADATA_FILENAME:  str = "metadata.json"
DEFAULT_INDEX_FILENAME:     str = "isocortex.index"
DEFAULT_CONFIG_FILENAME:    str = "config.json"

# Default HNSW tuning parameters (from SRS Section 3)
DEFAULT_CONFIG: Dict[str, Any] = {
    "M":              16,
    "efConstruction": 200,
    "efSearch":       50,
    "dim":            384,
    "space":          "cosine",
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PipelineOutput:
    """
    Complete output from the ingestion pipeline (scan -> embed -> export).

    Attributes
    ----------
    vectors_path    : Path  — Path to the vectors.bin file.
    metadata_path   : Path  — Path to the metadata.json file.
    output_dir      : Path  — Directory containing both files.
    vector_count    : int   — Number of vectors exported.
    vector_dim      : int   — Dimensionality of each vector.
    total_chunks    : int   — Total chunks across all documents.
    total_tokens    : int   — Total tokens across all chunks.
    elapsed_seconds : float — Total wall-clock pipeline time.
    """
    vectors_path:    Path
    metadata_path:   Path
    output_dir:      Path
    vector_count:    int
    vector_dim:      int
    total_chunks:    int
    total_tokens:    int
    elapsed_seconds: float


@dataclass
class LoadedIndex:
    """
    Loaded pipeline data ready for HNSW ingestion or search.

    Attributes
    ----------
    vector_matrix : np.ndarray    — (N, 384) float32 matrix.
    metadata      : list[dict]    — Per-chunk metadata records.
    vectors_path  : Path          — Source vectors file path.
    metadata_path : Path          — Source metadata file path.
    config        : dict          — HNSW configuration (if loaded).
    """
    vector_matrix: np.ndarray
    metadata:      List[Dict[str, Any]]
    vectors_path:  Path
    metadata_path: Path
    config:        Dict[str, Any] = field(default_factory=dict)

    @property
    def vector_count(self) -> int:
        return self.vector_matrix.shape[0]

    @property
    def vector_dim(self) -> int:
        return self.vector_matrix.shape[1]


# ---------------------------------------------------------------------------
# Atomic write helpers
# ---------------------------------------------------------------------------

def _atomic_write_bytes(data: bytes, target_path: Path) -> None:
    """
    Write *data* to *target_path* atomically.

    Writes to a temporary file in the same directory, then renames.
    This prevents partial writes if the process crashes mid-write.

    Parameters
    ----------
    data        : bytes  — Raw bytes to write.
    target_path : Path   — Final destination file.

    Raises
    ------
    OSError
        If write or rename fails.
    """
    target_path.parent.mkdir(parents=True, exist_ok=True)

    # Write to temp file in same directory (required for atomic rename)
    tmp_fd, tmp_path = tempfile.mkstemp(
        dir    = target_path.parent,
        prefix = f".{target_path.name}.",
        suffix = ".tmp",
    )

    try:
        with os.fdopen(tmp_fd, "wb") as tmp_fh:
            tmp_fh.write(data)
            tmp_fh.flush()
            os.fsync(tmp_fh.fileno())

        # Atomic rename
        os.replace(tmp_path, str(target_path))

    except Exception:
        # Clean up temp file on failure
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise



def _atomic_write_text(
    text: str,
    target_path: Path,
    encoding: str = "utf-8",
) -> None:
    """
    Write *text* to *target_path* atomically.

    Parameters
    ----------
    text        : str    — Text content to write.
    target_path : Path   — Final destination file.
    encoding    : str    — Character encoding. Default utf-8.
    """
    _atomic_write_bytes(text.encode(encoding), target_path)


# ---------------------------------------------------------------------------
# Export: vectors.bin
# ---------------------------------------------------------------------------

def export_vectors_bin(
    vector_matrix: np.ndarray,
    output_path: str | Path,
    atomic: bool = True,
) -> Path:
    """
    Write vector matrix to a flat binary file.

    Format (FR-4):
      [n_vectors:uint32][dim:uint32][vectors:float32 * n * dim]

    Parameters
    ----------
    vector_matrix : np.ndarray — (N, 384) float32 array.
    output_path   : str | Path — Destination .bin file path.
    atomic        : bool       — Use atomic write. Default True.

    Returns
    -------
    Path — The written file path.

    Raises
    ------
    ValueError
        If vector_matrix is not 2D or has wrong dimensionality.
    OSError
        If file cannot be written.
    """
    if vector_matrix.ndim != 2:
        raise ValueError(
            f"vector_matrix must be 2D, got shape {vector_matrix.shape}"
        )

    n_vectors, dim = vector_matrix.shape

    if dim != VECTOR_DIM:
        raise ValueError(
            f"Vector dim {dim} != expected {VECTOR_DIM}"
        )

    if n_vectors == 0:
        raise ValueError("vector_matrix has zero vectors")

    output = Path(output_path)

    # Build binary data
    header = np.array([n_vectors, dim], dtype=np.uint32).tobytes()
    body   = vector_matrix.astype(np.float32).tobytes()
    data   = header + body

    expected_size = 8 + n_vectors * dim * 4
    if len(data) != expected_size:
        raise OSError(
            f"Binary data size {len(data)} != expected {expected_size}"
        )

    if atomic:
        _atomic_write_bytes(data, output)
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())

    file_size = output.stat().st_size
    logger.info(
        "[SERIALIZER] Vectors exported  path=%s  vectors=%d  dim=%d  size=%s",
        output, n_vectors, dim, _human_bytes(file_size),
    )

    return output


# ---------------------------------------------------------------------------
# Export: metadata.json
# ---------------------------------------------------------------------------

def export_metadata_json(
    metadata: List[Dict[str, Any]],
    output_path: str | Path,
    atomic: bool = True,
) -> Path:
    """
    Write chunk metadata to a JSON sidecar file.

    Parameters
    ----------
    metadata    : list[dict]  — Metadata records.
    output_path : str | Path  — Destination .json file path.
    atomic      : bool        — Use atomic write. Default True.

    Returns
    -------
    Path — The written file path.

    Raises
    ------
    ValueError
        If metadata is empty.
    OSError
        If file cannot be written.
    """
    if not metadata:
        raise ValueError("Cannot export empty metadata list")

    output = Path(output_path)
    text   = json.dumps(metadata, indent=2, ensure_ascii=False)

    if atomic:
        _atomic_write_text(text, output)
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())

    file_size = output.stat().st_size
    logger.info(
        "[SERIALIZER] Metadata exported  path=%s  records=%d  size=%s",
        output, len(metadata), _human_bytes(file_size),
    )

    return output


# ---------------------------------------------------------------------------
# Export: config.json
# ---------------------------------------------------------------------------

def export_config(
    config: Dict[str, Any],
    output_path: str | Path,
    atomic: bool = True,
) -> Path:
    """
    Write HNSW configuration to config.json.

    Parameters
    ----------
    config      : dict       — Configuration dictionary.
    output_path : str | Path — Destination .json file path.
    atomic      : bool       — Use atomic write. Default True.

    Returns
    -------
    Path — The written file path.
    """
    output = Path(output_path)
    text   = json.dumps(config, indent=2, ensure_ascii=False)

    if atomic:
        _atomic_write_text(text, output)
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())

    logger.info(
        "[SERIALIZER] Config exported  path=%s  keys=%s",
        output, list(config.keys()),
    )

    return output


# ---------------------------------------------------------------------------
# Export: full pipeline result
# ---------------------------------------------------------------------------

def export_pipeline_result(
    vector_matrix: np.ndarray,
    metadata: List[Dict[str, Any]],
    output_dir: str | Path,
    config: Optional[Dict[str, Any]] = None,
    vectors_filename: str = DEFAULT_VECTORS_FILENAME,
    metadata_filename: str = DEFAULT_METADATA_FILENAME,
    config_filename: str = DEFAULT_CONFIG_FILENAME,
    total_tokens: int = 0,
    elapsed_seconds: float = 0.0,
) -> PipelineOutput:
    """
    Export complete pipeline output to vectors.bin, metadata.json,
    and optionally config.json.

    Parameters
    ----------
    vector_matrix     : np.ndarray        — (N, 384) float32 matrix.
    metadata          : list[dict]        — Per-chunk metadata.
    output_dir        : str | Path        — Output directory.
    config            : dict | None       — H_filename  : str               — Binary vectors filename.
    metadata_filename : str               — Metadata JSON filename.
    config_filename   : str               — Config JSON filename.
    total_tokens      : int               — Total token count for report.
    elapsed_seconds   : float             — Pipeline elapsed time.

    Returns
    -------
    PipelineOutput — Paths and statistics for the exported files.

    Raises
    ------
    ValueError
        If vector_matrix is None or metadata is empty.
    """
    if vector_matrix is None:
        raise ValueError("vector_matrix is None — cannot export")

    if not metadata:
        raise ValueError("metadata is empty — cannot export")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Export vectors
    vectors_path = export_vectors_bin(
        vector_matrix,
        out_dir / vectors_filename,
    )

    # Export metadata
    metadata_path = export_metadata_json(
        metadata,
        out_dir / metadata_filename,
    )

    # Export config (if provided or use defaults)
    if config is None:
        config = dict(DEFAULT_CONFIG)
        config["dim"] = vector_matrix.shape[1]

    export_config(
        config,
        out_dir / config_filename,
    )

    output = PipelineOutput(
        vectors_path    = vectors_path,
        metadata_path   = metadata_path,
        output_dir      = out_dir,
        vector_count    = vector_matrix.shape[0],
        vector_dim      = vector_matrix.shape[1],
        total_chunks    = len(metadata),
        total_tokens    = total_tokens,
        elapsed_seconds = elapsed_seconds,
    )

    logger.info(
        "[SERIALIZER] Pipeline export complete  dir=%s  vectors=%d  "
        "metadata=%d  elapsed=%.3fs",
        out_dir, output.vector_count, output.total_chunks, elapsed_seconds,
    )

    return output


# ---------------------------------------------------------------------------
# Load: vectors.bin
# ---------------------------------------------------------------------------

def load_vectors_bin(path: str | Path) -> np.ndarray:
    """
    Read a vectors.bin file back into a numpy array.

    Parameters
    ----------
    path : str | Path — Path to the .bin file.

    Returns
    -------
    np.ndarray — (N, dim) float32 matrix.

    Raises
    ------
    FileNotFoundError
        If file does not exist.
    ValueError
        If file is malformed or dimension != 384.
    """
    bin_path = Path(path)

    if not bin_path.exists():
        raise FileNotFoundError(f"Vectors file not found: {bin_path}")

    file_size = bin_path.stat().st_size
    if file_size < 8:
        raise ValueError(
            f"Vectors file too small ({file_size} bytes): {bin_path}"
        )

    with open(bin_path, "rb") as fh:
        header = fh.read(8)
        n_vectors = np.frombuffer(header[:4], dtype=np.uint32)[0]
        dim       = np.frombuffer(header[4:8], dtype=np.uint32)[0]

        if dim != VECTOR_DIM:
            raise ValueError(
                f"Vector dim {dim} != expected {VECTOR_DIM} in {bin_path}"
            )

        expected_bytes = n_vectors * dim * 4
        actual_bytes   = file_size - 8

        if actual_bytes < expected_bytes:
            raise ValueError(
                f"Vectors file truncated: expected {expected_bytes} bytes "
                f"of vector data, got {actual_bytes}"
            )

        if actual_bytes > expected_bytes:
            logger.warning(
                "[SERIALIZER] Vectors file has %d extra bytes — "
                "may be corrupted",
                actual_bytes - expected_bytes,
            )

        data = fh.read(expected_bytes)

    matrix = np.frombuffer(data, dtype=np.float32).reshape(n_vectors, dim).copy()

    # Validate no NaN or Inf
    if not np.isfinite(matrix).all():
        bad_count = int((~np.isfinite(matrix)).sum())
        raise ValueError(
            f"Vectors contain {bad_count} NaN/Inf values in {bin_path}"
        )

    logger.info(
        "[SERIALIZER] Vectors loaded  path=%s  shape=%s  dtype=%s",
        bin_path, matrix.shape, matrix.dtype,
    )

    return matrix


# ---------------------------------------------------------------------------
# Load: metadata.json
# ---------------------------------------------------------------------------

def load_metadata_json(path: str | Path) -> List[Dict[str, Any]]:
    """
    Read metadata.json and return the metadata list.

    Parameters
    ----------
    path : str | Path — Path to the metadata file.

    Returns
    -------
    list[dict] — Metadata records.

    Raises
    ------
    FileNotFoundError
        If file does not exist.
    ValueError
        If file is malformed or missing required fields.
    """
    json_path = Path(path)

    if not json_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as fh:
        metadata = json.load(fh)

    if not isinstance(metadata, list):
        raise ValueError(
            f"Metadata must be a list, got {type(metadata).__name__}"
        )

    if not metadata:
        raise ValueError("Metadata file contains an empty list")

    # Validate required fields on first record
    required = {"chunk_id", "text", "source_file", "source_label",
                "format_category", "word_count", "token_count",
                "source_chunk_index"}
    first = metadata[0]
    missing = required - set(first.keys())
    if missing:
        raise ValueError(
            f"Metadata missing required fields: {missing}"
        )

    logger.info(
        "[SERIALIZER] Metadata loaded  path=%s  records=%d",
        json_path, len(metadata),
    )

    return metadata


# ---------------------------------------------------------------------------
# Load: config.json
# ---------------------------------------------------------------------------

def load_config(path: str | Path) -> Dict[str, Any]:
    """
    Read config.json and return the configuration dictionary.

    Parameters
    ----------
    path : str | Path — Path to the config file.

    Returns
    -------
    dict — Configuration dictionary with HNSW tuning parameters.

    Raises
    ------
    FileNotFoundError
        If file does not exist.
    ValueError
        If file is malformed.
    """
    config_path = Path(path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as fh:
        config = json.load(fh)

    if not isinstance(config, dict):
        raise ValueError(
            f"Config must be a dict, got {type(config).__name__}"
        )

    # Validate required keys
    required = {"M", "efConstruction", "efSearch", "dim", "space"}
    missing = required - set(config.keys())
    if missing:
        raise ValueError(f"Config missing required keys: {missing}")

    # Validate types
    if not isinstance(config["M"], int) or config["M"] < 1:
        raise ValueError(
            f"Config 'M' must be a positive int, got {config['M']}"
        )

    if not isinstance(config["efConstruction"], int) or config["efConstruction"] < 1:
        raise ValueError(
            f"Config 'efConstruction' must be a positive int, got "
            f"{config['efConstruction']}"
        )

    if not isinstance(config["efSearch"], int) or config["efSearch"] < 1:
        raise ValueError(
            f"Config 'efSearch' must be a positive int, got {config['efSearch']}"
        )

    if not isinstance(config["dim"], int) or config["dim"] != VECTOR_DIM:
        raise ValueError(
            f"Config 'dim' must be {VECTOR_DIM}, got {config['dim']}"
        )

    if config["space"] not in ("cosine", "l2"):
        raise ValueError(
            f"Config 'space' must be 'cosine' or 'l2', got {config['space']}"
        )

    logger.info(
        "[SERIALIZER] Config loaded  path=%s  M=%d  efC=%d  efS=%d  space=%s",
        config_path, config["M"], config["efConstruction"],
        config["efSearch"], config["space"],
    )

    return config


# ---------------------------------------------------------------------------
# Load: full pipeline result
# ---------------------------------------------------------------------------

def load_pipeline_result(
    input_dir: str | Path,
    vectors_filename: str = DEFAULT_VECTORS_FILENAME,
    metadata_filename: str = DEFAULT_METADATA_FILENAME,
    config_filename: str = DEFAULT_CONFIG_FILENAME,
) -> LoadedIndex:
    """
    Load vectors.bin, metadata.json, and optionally config.json
    from a directory.

    Parameters
    ----------
    input_dir         : str | Path — Directory containing the files.
    vectors_filename  : str        — Binary vectors filename.
    metadata_filename : str        — Metadata JSON filename.
    config_filename   : str        — Config JSON filename.

    Returns
    -------
    LoadedIndex — Vectors, metadata, and configuration.
    """
    in_dir = Path(input_dir)

    if not in_dir.is_dir():
        raise NotADirectoryError(f"Input directory not found: {in_dir}")

    vectors_path  = in_dir / vectors_filename
    metadata_path = in_dir / metadata_filename
    config_path   = in_dir / config_filename

    # Load vectors
    vector_matrix = load_vectors_bin(vectors_path)

    # Load metadata
    metadata = load_metadata_json(metadata_path)

    # Cross-validate counts
    if vector_matrix.shape[0] != len(metadata):
        raise ValueError(
            f"Vector count ({vector_matrix.shape[0]}) != "
            f"metadata count ({len(metadata)}) in {in_dir}"
        )

    # Load config (optional — use defaults if missing)
    config: Dict[str, Any] = {}
    if config_path.exists():
        try:
            config = load_config(config_path)
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning(
                "[SERIALIZER] Config load failed (%s) — using defaults",
                exc,
            )
            config = dict(DEFAULT_CONFIG)
    else:
        logger.debug(
            "[SERIALIZER] No config.json found — using defaults"
        )
        config = dict(DEFAULT_CONFIG)

    # Validate config dim matches vectors
    if "dim" in config and config["dim"] != vector_matrix.shape[1]:
        raise ValueError(
            f"Config dim ({config['dim']}) != vector dim "
            f"({vector_matrix.shape[1]})"
        )

    loaded = LoadedIndex(
        vector_matrix = vector_matrix,
        metadata      = metadata,
        vectors_path  = vectors_path,
        metadata_path = metadata_path,
        config        = config,
    )

    logger.info(
        "[SERIALIZER] Pipeline loaded  dir=%s  vectors=%d  dim=%d",
        in_dir, loaded.vector_count, loaded.vector_dim,
    )

    return loaded


# ---------------------------------------------------------------------------
# Incremental index support (UC-3 / FR-7)
# ---------------------------------------------------------------------------

def build_incremental_update(
    existing_metadata: List[Dict[str, Any]],
    new_metadata: List[Dict[str, Any]],
) -> Tuple[List[int], List[Dict[str, Any]]]:
    """
    Compare existing metadata with new metadata and identify
    which chunks are new or modified.

    A chunk is considered "new" if its (source_file, source_chunk_index)
    pair does not exist in the existing metadata, or if its text has
    changed.

    Parameters
    ----------
    existing_metadata : list[dict] — Current metadata from metadata.json.
    new_metadata      : list[dict] — Fresh metadata from re-scan.

    Returns
    -------
    (new_indices, updated_metadata)
      new_indices       : list[int]   — Indices into new_metadata that are
                                        genuinely new or modified.
      updated_metadata  : list[dict]  — Merged metadata (existing + new).
    """
    # Build lookup: (source_file, source_chunk_index) -> text
    existing_lookup: Dict[Tuple[str, int], str] = {}
    for rec in existing_metadata:
        key = (rec["source_file"], rec.get("source_chunk_index", 0))
        existing_lookup[key] = rec["text"]

    new_indices: List[int] = []
    for idx, rec in enumerate(new_metadata):
        key = (rec["source_file"], rec.get("source_chunk_index", 0))
        existing_text = existing_lookup.get(key)

        if existing_text is None:
            # Completely new chunk
            new_indices.append(idx)
        elif existing_text != rec["text"]:
            # Source file modified — text changed
            new_indices.append(idx)

    # Build merged metadata: existing + new chunks
    # Existing chunks keep their original chunk_id
    # New chunks get chunk_ids starting after the last existing one
    last_id = max((r.get("chunk_id", 0) for r in existing_metadata), default=-1)
    updated_metadata = list(existing_metadata)

    for new_idx in new_indices:
        rec = dict(new_metadata[new_idx])  # shallow copy
        last_id += 1
        rec["chunk_id"] = last_id
        updated_metadata.append(rec)

    logger.info(
        "[SERIALIZER] Incremental update  existing=%d  new=%d  "
        "merged=%d",
        len(existing_metadata), len(new_indices), len(updated_metadata),
    )

    return new_indices, updated_metadata


# ---------------------------------------------------------------------------
# Index file management (FR-7)
# ---------------------------------------------------------------------------

def index_file_exists(
    output_dir: str | Path,
    index_filename: str = DEFAULT_INDEX_FILENAME,
) -> bool:
    """Check if the HNSW index file exists in the output directory."""
    return (Path(output_dir) / index_filename).is_file()


def get_index_path(
    output_dir: str | Path,
    index_filename: str = DEFAULT_INDEX_FILENAME,
) -> Path:
    """Return the full path to the HNSW index file."""
    return Path(output_dir) / index_filename


# ---------------------------------------------------------------------------
# File integrity utilities
# ---------------------------------------------------------------------------

def validate_export_directory(output_dir: str | Path) -> bool:
    """
    Validate that an output directory contains a complete, consistent
    export (vectors.bin + metadata.json with matching counts).

    Parameters
    ----------
    output_dir : str | Path — Directory to validate.

    Returns
    -------
    bool — True if valid and consistent.
    """
    out_dir = Path(output_dir)

    if not out_dir.is_dir():
        logger.warning("[SERIALIZER] Output dir does not exist: %s", out_dir)
        return False

    vectors_path  = out_dir / DEFAULT_VECTORS_FILENAME
    metadata_path = out_dir / DEFAULT_METADATA_FILENAME

    if not vectors_path.exists():
        logger.warning("[SERIALIZER] Missing: %s", vectors_path)
        return False

    if not metadata_path.exists():
        logger.warning("[SERIALIZER] Missing: %s", metadata_path)
        return False

    try:
        vector_matrix = load_vectors_bin(vectors_path)
        metadata      = load_metadata_json(metadata_path)
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("[SERIALIZER] Validation failed: %s", exc)
        return False

    if vector_matrix.shape[0] != len(metadata):
        logger.warning(
            "[SERIALIZER] Count mismatch: vectors=%d  metadata=%d",
            vector_matrix.shape[0], len(metadata),
        )
        return False

    logger.info(
        "[SERIALIZER] Export directory valid  dir=%s  vectors=%d",
        out_dir, vector_matrix.shape[0],
    )

    return True


def get_export_stats(output_dir: str | Path) -> Optional[Dict[str, Any]]:
    """
    Return statistics about an existing export directory without
    loading the full vector matrix into memory.

    Parameters
    ----------
    output_dir : str | Path — Export directory.

    Returns
    -------
    dict | None — Statistics dict, or None if directory is invalid.
    """
    out_dir = Path(output_dir)

    if not out_dir.is_dir():
        return None

    vectors_path  = out_dir / DEFAULT_VECTORS_FILENAME
    metadata_path = out_dir / DEFAULT_METADATA_FILENAME
    config_path   = out_dir / DEFAULT_CONFIG_FILENAME
    index_path    = out_dir / DEFAULT_INDEX_FILENAME

    stats: Dict[str, Any] = {
        "output_dir":      str(out_dir),
        "vectors_exists":  vectors_path.exists(),
        "metadata_exists": metadata_path.exists(),
        "config_exists":   config_path.exists(),
        "index_exists":    index_path.exists(),
        "vector_count":    0,
        "vector_dim":      0,
        "metadata_count":  0,
        "valid":           False,
    }

    # Read vector header without loading full matrix
    if vectors_path.exists():
        try:
            with open(vectors_path, "rb") as fh:
                header = fh.read(8)
                if len(header) == 8:
                    stats["vector_count"] = int(
                        np.frombuffer(header[:4], dtype=np.uint32)[0]
                    )
                    stats["vector_dim"] = int(
                        np.frombuffer(header[4:8], dtype=np.uint32)[0]
                    )
        except OSError:
            pass

    # Count metadata records without loading full JSON
    if metadata_path.exists():
        try:
            with open(metadata_path, "r", encoding="utf-8") as fh:
                meta = json.load(fh)
                if isinstance(meta, list):
                    stats["metadata_count"] = len(meta)
        except (json.JSONDecodeError, OSError):
            pass

    # File sizes
    for name, path in [
        ("vectors_size", vectors_path),
        ("metadata_size", metadata_path),
        ("config_size", config_path),
        ("index_size", index_path),
    ]:
        if path.exists():
            stats[name] = _human_bytes(path.stat().st_size)
        else:
            stats[name] = None

    # Validate consistency
    if (
        stats["vectors_exists"]
        and stats["metadata_exists"]
        and stats["vector_count"] == stats["metadata_count"]
        and stats["vector_dim"] == VECTOR_DIM
    ):
        stats["valid"] = True

    return stats


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _human_bytes(num_bytes: int | float) -> str:
    """Convert byte count to human-readable string."""
    for unit in ("B", "KB", "MB", "GB"):
        if abs(num_bytes) < 1024:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} TB"


# ---------------------------------------------------------------------------
# CLI entry point — standalone debug mode
# ---------------------------------------------------------------------------

def _configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level   = level,
        format  = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt = "%Y-%m-%d %H:%M:%S",
    )

if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        prog        = "serializer",
        description = "IsoCortex serializer — standalone debug mode.",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable DEBUG-level logging.",
    )
    subparsers = parser.add_subparsers(dest="command", help="Command")

    # -- validate -------------------------------------------------------
    val_parser = subparsers.add_parser(
        "validate", help="Validate an export directory.",
    )
    val_parser.add_argument("dir", help="Export directory to validate.")

    # -- stats ----------------------------------------------------------
    stats_parser = subparsers.add_parser(
        "stats", help="Show export directory statistics.",
    )
    stats_parser.add_argument("dir", help="Export directory to inspect.")

    # -- load -----------------------------------------------------------
    load_parser = subparsers.add_parser(
        "load", help="Load and verify vectors + metadata.",
    )
    load_parser.add_argument("dir", help="Export directory to load.")

    # -- export ---------------------------------------------------------
    export_parser = subparsers.add_parser(
        "export", help="Export test vectors (debug).",
    )
    export_parser.add_argument(
        "dir", help="Output directory.",
    )
    export_parser.add_argument(
        "--count", type=int, default=10,
        help="Number of random vectors to generate.",
    )

    args = parser.parse_args()
    _configure_logging(args.verbose)

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    # -- validate -------------------------------------------------------
    if args.command == "validate":
        if validate_export_directory(args.dir):
            print(f"[OK] {args.dir} is valid and consistent.")
            sys.exit(0)
        else:
            print(
                f"[FAIL] {args.dir} is invalid or inconsistent.",
                file=sys.stderr,
            )
            sys.exit(1)

    # -- stats ----------------------------------------------------------
    elif args.command == "stats":
        stats = get_export_stats(args.dir)
        if stats is None:
            print(
                f"[ERROR] Invalid directory: {args.dir}",
                file=sys.stderr,
            )
            sys.exit(1)

        print()
        print("=" * 50)
        print(f"  EXPORT STATS  —  {args.dir}")
        print("=" * 50)
        for key, val in stats.items():
            print(f"  {key:<20}: {val}")
        print("=" * 50)

    # -- load -----------------------------------------------------------
    elif args.command == "load":
        try:
            loaded = load_pipeline_result(args.dir)
            print(f"[OK] Vectors  : {loaded.vectors_path}")
            print(f"[OK] Metadata : {loaded.metadata_path}")
            print(f"[OK] Count    : {loaded.vector_count}")
            print(f"[OK] Dim      : {loaded.vector_dim}")
            print(f"[OK] Config   : {loaded.config}")
        except Exception as exc:  # pylint: {exc}", file=sys.stderr)
            sys.exit(1)

    # -- export ---------------------------------------------------------
    elif args.command == "export":
        rng = np.random.default_rng(42)
        test_vectors = rng.standard_normal(
            (args.count, VECTOR_DIM)
        ).astype(np.float32)
        # Normalize for cosine similarity
        norms = np.linalg.norm(test_vectors, axis=1, keepdims=True)
        test_vectors = test_vectors / norms

        test_metadata = [
            {
                "chunk_id":           i,
                "text":               f"Test chunk {i}",
                "source_file":        f"/test/file_{i}.txt",
                "source_label":       f"File: file_{i}.txt",
                "format_category":    "plain_text",
                "word_count":         3,
                "token_count":        5,
                "source_chunk_index": 0,
            }
            for i in range(args.count)
        ]

        try:
            output = export_pipeline_result(
                test_vectors,
                test_metadata,
                args.dir,
                total_tokens=args.count * 5,
            )
            print(f"\n[OK] Vectors : {output.vectors_path}")
            print(f"[OK] Metadata: {output.metadata_path}")
            print(f"[OK] Config  : {output.output_dir / DEFAULT_CONFIG_FILENAME}")
        except Exception as exc:  # pylint: disable=broad-except
            print(f"[ERROR] {exc}", file=sys.stderr)
            sys.exit(1)
