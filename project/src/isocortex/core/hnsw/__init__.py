"""
IsoCortex core/hnsw — HNSW Graph Implementation
================================================

Custom C++ HNSW (Hierarchical Navigable Small World) implementation
with pybind11 bindings for Python.

SRS References:
  - FR-IDX-001: Index construction with configurable M, ef_construction, ef_search
  - FR-IDX-002: Soft delete with tombstone pattern
  - Section 4.2: ReadWriteLock for thread safety
  - Section 7: Index format versioning and binary persistence
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

logger = logging.getLogger(__name__)


def _try_import_native() -> bool:
    """Attempt to import the pybind11 native module.

    Returns True if the native C++ extension is available.
    """
    try:
        from isocortex.core.hnsw._hnsw_native import (  # type: ignore[import]
            HnswIndex as NativeHnswIndex,
            cosine_distance,
            l2_distance,
            has_simd_support,
            VECTOR_DIM,
        )
        _native_cache = {
            "HnswIndex": NativeHnswIndex,
            "cosine_distance": cosine_distance,
            "l2_distance": l2_distance,
            "has_simd_support": has_simd_support,
            "VECTOR_DIM": VECTOR_DIM,
        }
        simd = has_simd_support()
        logger.info(
            "[HNSW] Native C++ extension loaded successfully  "
            "SIMD=%s  dim=%d",
            simd, VECTOR_DIM,
        )
        return True
    except ImportError:
        logger.warning(
            "[HNSW] Native C++ extension not available — "
            "falling back to pure-Python search. "
            "Build with: pip install -e '.[dev]'"
        )
        return False


# Module-level state
_native_available: bool = _try_import_native()


def get_native_module() -> dict | None:
    """Return the native module cache, or None if native is unavailable."""
    if _native_available:
        import isocortex.core.hnsw._hnsw_native as _native  # type: ignore[import]
        return _native_cache
    return None


def distance_fn(metric: str = "cosine"):
    """Return the appropriate distance function for the given metric.

    Parameters
    ----------
    metric : str
        One of 'cosine', 'l2', or 'ip'.

    Returns
    -------
    Callable[[np.ndarray, np.ndarray], float]
    """
    mod = get_native_module()
    if mod is None:
        raise RuntimeError(
            "Native HNSW extension not available. "
            "Build the project with: pip install -e '.[dev]'"
        )

    if metric == "cosine":
        return mod["cosine_distance"]
    elif metric == "l2":
        return mod["l2_distance"]
    else:
        raise ValueError(f"Unknown metric: {metric!r}. Use 'cosine' or 'l2'.")


def get_simd_support() -> str:
    """Return the SIMD instruction set available ('avx2', 'sse4_1', 'neon', 'scalar')."""
    mod = get_native_module()
    if mod is None:
        return "scalar"
    return mod["has_simd_support"]()
