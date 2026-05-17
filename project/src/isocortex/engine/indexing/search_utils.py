"""
IsoCortex — Search Utilities
=============================

Brute-force search fallback and distance computation utilities.
Used by IndexManager when the C++ HNSW binding is not available.

Author : Shaheer Qureshi
Project: IsoCortex
"""

from __future__ import annotations

import numpy as np


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine distance between two vectors.

    Returns 0 for identical vectors, 2 for opposite vectors.
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 1.0
    similarity = np.dot(a, b) / (norm_a * norm_b)
    return 1.0 - float(similarity)


def l2_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute L2 (Euclidean) distance between two vectors."""
    return float(np.linalg.norm(a - b))


def brute_force_search(
    vectors: np.ndarray,
    query: np.ndarray,
    k: int,
    metric: str = "cosine",
    skip_deleted: bool = True,
    deleted_flags: list[bool] | None = None,
) -> list[tuple[int, float]]:
    """Brute-force k-nearest-neighbour search.

    Used as a fallback when the HNSW C++ binding is not compiled.
    Also used for testing and recall evaluation.

    Parameters
    ----------
    vectors : np.ndarray
        Vector matrix of shape (N, D).
    query : np.ndarray
        Query vector of shape (D,).
    k : int
        Number of nearest neighbours to return.
    metric : str
        Distance metric: "cosine" (default) or "l2".
    skip_deleted : bool
        Skip vectors marked as deleted.
    deleted_flags : list[bool]
        Per-vector deletion flags (same length as vectors).

    Returns
    -------
    list[tuple[int, float]]
        List of (vector_index, distance) tuples, sorted by distance ascending.
    """
    n = vectors.shape[0]
    k = min(k, n)

    if metric == "cosine":
        # Normalize for cosine similarity
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        normed = vectors / norms

        q_norm = np.linalg.norm(query)
        if q_norm == 0:
            return [(i, 1.0) for i in range(k)]
        q_normed = query / q_norm

        # Compute similarities
        similarities = normed @ q_normed
        distances = 1.0 - similarities
    elif metric == "l2":
        diff = vectors - query
        distances = np.linalg.norm(diff, axis=1)
    else:
        raise ValueError(f"Unknown metric: {metric}")

    # Build result array
    indices = np.arange(n)

    # Filter deleted
    if skip_deleted and deleted_flags is not None:
        mask = np.array([not d for d in deleted_flags])
        indices = indices[mask]
        distances = distances[mask]

    # Sort by distance
    sorted_idx = np.argsort(distances)[:k]

    return [(int(indices[i]), float(distances[indices[i]])) for i in range(len(sorted_idx))]
