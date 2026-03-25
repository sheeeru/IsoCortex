"""
IsoCortex — ingestion/embedder.py
====================================
Local neural embedding engine.

Responsibilities (FR-3, FR-4, NFR-3, NFR-4):
  - Accept ChunkedDocument objects from chunker.py.
  - Load the all-MiniLM-L6-v2 sentence-transformer model locally.
  - Generate 384-dimensional dense vectors for every chunk.
  - Process chunks in configurable batches for throughput.
  - Populate token_count on every chunk (creating new frozen objects).
  - Return structured EmbeddedDocument objects — never raw arrays.
  - Emit structured log messages at every decision point.

SRS References: FR-3, FR-4, NFR-2, NFR-3, NFR-4, CON-1, CON-5

Author : Shaheer Qureshi
Project: IsoCortex
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

import numpy as np

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
VECTOR_DIM: int          = 384    # all-MiniLM-L6-v2 output dimension
DEFAULT_BATCH_SIZE: int  = 64     # FR-3 default batch size
MODEL_NAME: str          = "sentence-transformers/all-MiniLM-L6-v2"

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EmbeddedChunk:
    """
    A single text chunk with its embedding vector and token count.

    Attributes
    ----------
    chunk_index        : int          — Sequential index (from Chunk).
    text               : str          — Chunk text content.
    vector             : np.ndarray   — 384-dim float32 embedding vector.
    token_count        : int          — Sub-word token count from the model.
    source_file        : str          — Absolute path of the source file.
    source_label       : str          — Sub-document label.
    format_category    : str          — e.g. "pdf", "spreadsheet".
    word_count         : int          — Word count of this chunk.
    source_chunk_index : int          — Index of the parent ExtractionChunk.
    """
    chunk_index:        int
    text:               str
    vector:             np.ndarray
    token_count:        int
    source_file:        str
    source_label:       str
    format_category:    str
    word_count:         int
    source_chunk_index: int

    def __post_init__(self) -> None:
        # Validate vector dimension
        if self.vector.shape != (VECTOR_DIM,):
            raise ValueError(
                f"Vector shape {self.vector.shape} != expected ({VECTOR_DIM},)"
            )
        # Validate dtype
        if self.vector.dtype != np.float32:
            object.__setattr__(self, "vector", self.vector.astype(np.float32))


@dataclass
class EmbeddedDocument:
    """
    All embedded chunks produced from a single ChunkedDocument.

    Attributes
    ----------
    source_path     : Path                  — Source file path.
    format_category : str                   — Format category.
    embedded_chunks : list[EmbeddedChunk]   — Ordered list with vectors.
    success         : bool                  — False if embedding failed.
    error_message   : str | None            — Failure reason.
    elapsed_seconds : float                 — Wall-clock embedding time.
    """
    source_path:     Path
    format_category: str
    embedded_chunks: List[EmbeddedChunk] = field(default_factory=list)
    success:         bool                 = True
    error_message:   Optional[str]        = None
    elapsed_seconds: float                = 0.0

    @property
    def total_chunks(self) -> int:
        return len(self.embedded_chunks)

    @property
    def total_words(self) -> int:
        return sum(c.word_count for c in self.embedded_chunks)

    @property
    def total_tokens(self) -> int:
        return sum(c.token_count for c in self.embedded_chunks)


@dataclass
class EmbeddingResult:
    """
    Top-level return value of :func:`embed_batch`.

    Attributes
    ----------
    documents        : list[EmbeddedDocument] — Per-document results.
    vector_matrix    : np.ndarray             — (N, 384) float32 matrix.
    metadata         : list[dict]             — Per-chunk metadata records.
    success          : bool                   — True if all documents embedded.
    elapsed_seconds  : float                  — Total wall-clock time.
    """
    documents:       List[EmbeddedDocument] = field(default_factory=list)
    vector_matrix:   Optional[np.ndarray]   = None
    metadata:        List[dict]             = field(default_factory=list)
    success:         bool                    = True
    elapsed_seconds: float                   = 0.0

    @property
    def total_chunks(self) -> int:
        return sum(d.total_chunks for d in self.documents if d.success)

    @property
    def total_tokens(self) -> int:
        return sum(d.total_tokens for d in self.documents if d.success)


# ---------------------------------------------------------------------------
# Model management
# ---------------------------------------------------------------------------

_model_cache: dict = {}


def load_model(
    model_name: str = MODEL_NAME,
    device: str = "",
) -> Optional[object]:
    """
    Load the sentence-transformer model (singleton).

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier.
        Default: sentence-transformers/all-MiniLM-L6-v2.

    device : str
        Target device: "cpu", "cuda", or "" for auto-detect.
        Default: "" (auto).

    Returns
    -------
    SentenceTransformer model instance, or None on failure.
    """
    cache_key = f"{model_name}::{device or 'auto'}"

    if cache_key in _model_cache:
        return _model_cache[cache_key]

    try:
        from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]
    except ImportError:
        logger.error(
            "[EMBEDDER] sentence-transformers not installed — "
            "run: pip install sentence-transformers",
        )
        _model_cache[cache_key] = None
        return None

    try:
        if device:
            model = SentenceTransformer(
                model_name,
                device=device,
            )
        else:
            model = SentenceTransformer(model_name)

        # Verify output dimension
        test_vec = model.encode(
            "test",
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        actual_dim = test_vec.shape[0] if test_vec.ndim == 1 else test_vec.shape[-1]

        if actual_dim != VECTOR_DIM:
            logger.warning(
                "[EMBEDDER] Model '%s' outputs %d-dim vectors, "
                "expected %d. This may cause HNSW build failures.",
                model_name, actual_dim, VECTOR_DIM,
            )

        _model_cache[cache_key] = model
        logger.info(
            "[EMBEDDER] Model loaded  name=%s  device=%s  dim=%d",
            model_name,
            model.device if hasattr(model, "device") else "unknown",
            actual_dim,
        )
        return model

    except Exception as exc:  # pylint: disable=broad-except
        logger.error(
            "[EMBEDDER] Model load failed: %s", exc,
        )
        _model_cache[cache_key] = None
        return None


def clear_model_cache() -> None:
    """Release all cached models and free GPU memory."""
    _model_cache.clear()
    try:
        import torch  # type: ignore[import-untyped]
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    logger.debug("[EMBEDDER] Model cache cleared")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def embed_document(
    chunked_doc: object,
    *,
    model:       Optional[object] = None,
    batch_size:  int = DEFAULT_BATCH_SIZE,
    model_name:  str = MODEL_NAME,
    device:      str = "",
    show_progress: bool = False,
) -> EmbeddedDocument:
    """
    Embed all chunks in a single ChunkedDocument.

    Parameters
    ----------
    chunked_doc : ChunkedDocument
        Output from chunker.chunk_document().

    model : SentenceTransformer | None
        Pre-loaded model. If None, loads one automatically.

    batch_size : int
        Chunks per encode() call. Default 64 (FR-3).

    model_name : str
        Model identifier if auto-loading.

    device : str
        "cpu", "cuda", or "" for auto-detect.

    show_progress : bool
        Show tqdm progress bar during encoding.

    Returns
    -------
    EmbeddedDocument
        Always returns an object. On failure success=False.
    """
    # ---- Unpack ChunkedDocument (duck-typed) --------------------------
    abs_path: Path = chunked_doc.source_path          # type: ignore[attr-defined]
    category: str  = chunked_doc.format_category      # type: ignore[attr-defined]
    chunks:   list = chunked_doc.chunks               # type: ignore[attr-defined]

    emb_doc = EmbeddedDocument(
        source_path     = abs_path,
        format_category = category,
    )

    if not chunked_doc.success:  # type: ignore[attr-defined]
        _fail_emb(emb_doc, "ChunkedDocument indicates failure")
        return emb_doc

    if not chunks:
        _fail_emb(emb_doc, "ChunkedDocument contains zero chunks")
        return emb_doc

    # ---- Validate parameters ------------------------------------------
    if batch_size < 1:
        _fail_emb(emb_doc, f"batch_size must be >= 1, got {batch_size}")
        return emb_doc

    # ---- Load model if needed -----------------------------------------
    if model is None:
        model = load_model(model_name=model_name, device=device)
        if model is None:
            _fail_emb(emb_doc, "Model load failed — cannot embed")
            return emb_doc

    t_start = time.perf_counter()

    try:
        # ---- Extract texts and batch encode ---------------------------
        texts: list[str] = [c.text for c in chunks]  # type: ignore[attr-defined]

        logger.debug(
            "[EMBEDDER] Encoding %d chunks  batch_size=%d  file=%s",
            len(texts), batch_size, abs_path.name,
        )

        # Encode in batches to manage memory
        all_vectors: list[np.ndarray] = []
        all_token_counts: list[int] = []

        for batch_start in range(0, len(texts), batch_size):
            batch_end = min(batch_start + batch_size, len(texts))
            batch_texts = texts[batch_start:batch_end]

            batch_num = batch_start // batch_size + 1
            total_batches = (len(texts) + batch_size - 1) // batch_size

            if total_batches > 1:
                logger.debug(
                    "[EMBEDDER] Batch %d/%d  chunks=%d-%d  file=%s",
                    batch_num, total_batches,
                    batch_start + 1, batch_end,
                    abs_path.name,
                )

            # Encode batch
            vectors = model.encode(  # type: ignore[union-attr]
                batch_texts,
                batch_size=len(batch_texts),
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )

            # Ensure 2D: (batch, dim)
            if vectors.ndim == 1:
                vectors = vectors.reshape(1, -1)

            # Validate dimensions
            if vectors.shape[1] != VECTOR_DIM:
                _fail_emb(
                    emb_doc,
                    f"Vector dim mismatch: got {vectors.shape[1]}, "
                    f"expected {VECTOR_DIM}",
                )
                return emb_doc

            # Ensure float32
            if vectors.dtype != np.float32:
                vectors = vectors.astype(np.float32)

            all_vectors.append(vectors)

            # Get token counts from the tokenizer
            tokenizer = model.tokenizer  # type: ignore[union-attr]
            for text in batch_texts:
                try:
                    encoded = tokenizer(
                        text,
                        add_special_tokens=False,
                        truncation=False,
                        return_tensors=None,
                    )
                    token_count = len(encoded["input_ids"])
                    all_token_counts.append(token_count)
                except Exception:  # pylint: disable=broad-except
                    all_token_counts.append(0)

        # ---- Stack into single matrix ---------------------------------
        vector_matrix = np.vstack(all_vectors)

        # ---- Build EmbeddedChunk objects -------------------------------
        embedded_chunks: list[EmbeddedChunk] = []

        for idx, chunk in enumerate(chunks):
            emb_chunk = EmbeddedChunk(
                chunk_index        = idx,
                text               = chunk.text,             # type: ignore[attr-defined]
                vector             = vector_matrix[idx],
                token_count        = all_token_counts[idx],
                source_file        = chunk.source_file,       # type: ignore[attr-defined]
                source_label       = chunk.source_label,      # type: ignore[attr-defined]
                format_category    = chunk.format_category,   # type: ignore[attr-defined]
                word_count         = chunk.word_count,        # type: ignore[attr-defined]
                source_chunk_index = chunk.source_chunk_index, # type: ignore[attr-defined]
            )
            embedded_chunks.append(emb_chunk)

        emb_doc.embedded_chunks = embedded_chunks

    except MemoryError:
        _fail_emb(emb_doc, "MemoryError during embedding")
    except Exception as exc:  # pylint: disable=broad-except
        _fail_emb(emb_doc, f"{type(exc).__name__}: {exc}")

    emb_doc.elapsed_seconds = time.perf_counter() - t_start

    if emb_doc.success and emb_doc.total_chunks == 0:
        _fail_emb(emb_doc, "Embedding produced zero chunks")

    if emb_doc.success:
        logger.info(
            "[EMBEDDER] OK      %-40s  chunks=%d  tokens=%d  elapsed=%.3fs",
            abs_path.name,
            emb_doc.total_chunks,
            emb_doc.total_tokens,
            emb_doc.elapsed_seconds,
        )
    else:
        logger.warning(
            "[EMBEDDER] FAILED  %-40s  reason=%s",
            abs_path.name, emb_doc.error_message,
        )

    return emb_doc


def embed_batch(
    chunked_docs: list,
    *,
    model:         Optional[object] = None,
    batch_size:    int = DEFAULT_BATCH_SIZE,
    model_name:    str = MODEL_NAME,
    device:        str = "",
    show_progress: bool = False,
) -> EmbeddingResult:
    """
    Embed all chunks across a list of ChunkedDocument objects.

    Produces a single (N, 384) float32 vector matrix and a parallel
    metadata list, ready for export (FR-4) and HNSW ingestion (FR-5).

    Parameters
    ----------
    chunked_docs : list[ChunkedDocument]
        Output from chunker.chunk_batch().

    model : SentenceTransformer | None
        Pre-loaded model. If None, loads one automatically.

    batch_size : int
        Chunks per encode() call. Default 64 (FR-3).

    model_name : str
        Model identifier if auto-loading.

    device : str
        "cpu", "cuda", or "" for auto-detect.

    show_progress : bool
        Show tqdm progress bar during encoding.

    Returns
    -------
    EmbeddingResult
        Contains vector_matrix, metadata, and per-document results.
    """
    result = EmbeddingResult()

    if not chunked_docs:
        logger.warning("[EMBEDDER] Empty chunked_docs list — nothing to embed")
        return result

    # ---- Load model once for entire batch -----------------------------
    if model is None:
        model = load_model(model_name=model_name, device=device)
        if model is None:
            result.success = False
            logger.error("[EMBEDDER] Model load failed — aborting batch")
            return result

    t_start = time.perf_counter()

    all_vectors: list[np.ndarray] = []
    all_metadata: list[dict]      = []
    global_offset: int             = 0

    total_docs = len(chunked_docs)

    for doc_idx, chunked_doc in enumerate(chunked_docs, start=1):
        if doc_idx % 100 == 0 or doc_idx == total_docs:
            logger.info(
                "[EMBEDDER] Batch progress: %d/%d", doc_idx, total_docs,
            )

        emb_doc = embed_document(
            chunked_doc,
            model        = model,
            batch_size   = batch_size,
            show_progress= show_progress,
        )

        result.documents.append(emb_doc)

        if not emb_doc.success:
            result.success = False
            continue

        # Collect vectors and metadata
        for emb_chunk in emb_doc.embedded_chunks:
            all_vectors.append(emb_chunk.vector)

            all_metadata.append({
                "chunk_id":             global_offset,
                "text":                 emb_chunk.text,
                "source_file":          emb_chunk.source_file,
                "source_label":         emb_chunk.source_label,
                "format_category":      emb_chunk.format_category,
                "word_count":           emb_chunk.word_count,
                "token_count":          emb_chunk.token_count,
                "source_chunk_index":   emb_chunk.source_chunk_index,
            })
            global_offset += 1

    # ---- Build vector matrix ------------------------------------------
    if all_vectors:
        result.vector_matrix = np.vstack(all_vectors).astype(np.float32)
        logger.debug(
            "[EMBEDDER] Vector matrix shape: %s  dtype: %s",
            result.vector_matrix.shape,
            result.vector_matrix.dtype,
        )
    else:
        logger.warning("[EMBEDDER] No vectors produced — vector_matrix is None")

    result.metadata = all_metadata
    result.elapsed_seconds = time.perf_counter() - t_start

    succeeded = sum(1 for d in result.documents if d.success)
    failed    = total_docs - succeeded

    logger.info(
        "[EMBEDDER] Batch complete  docs=%d  OK=%d  FAIL=%d  "
        "vectors=%d  elapsed=%.3fs",
        total_docs, succeeded, failed,
        len(all_vectors), result.elapsed_seconds,
    )

    return result


# ---------------------------------------------------------------------------
# Export helpers (FR-4)
# ---------------------------------------------------------------------------

def export_vectors_bin(
    vector_matrix: np.ndarray,
    output_path:   str | Path,
) -> Path:
    """
    Write vector matrix to a flat binary file.

    Format: [n_vectors:uint32][dim:uint32][vectors:float32 * n * dim]

    Parameters
    ----------
    vector_matrix : np.ndarray — (N, 384) float32 array.
    output_path   : str | Path — Destination .bin file path.

    Returns
    -------
    Path — The written file path.

    Raises
    ------
    ValueError
        If vector_matrix is not 2D float32.
    OSError
        If file cannot be written.
    """
    if vector_matrix.ndim != 2:
        raise ValueError(
            f"vector_matrix must be 2D, got shape {vector_matrix.shape}"
        )

    n_vectors, dim = vector_matrix.shape
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with open(output, "wb") as fh:
        fh.write(np.uint32(n_vectors).tobytes())
        fh.write(np.uint32(dim).tobytes())
        fh.write(vector_matrix.astype(np.float32).tobytes())

    file_size = output.stat().st_size
    expected_size = 8 + n_vectors * dim * 4  # 2 * uint32 header + float32 data

    if file_size != expected_size:
        raise OSError(
            f"Written file size {file_size} != expected {expected_size}"
        )

    logger.info(
        "[EMBEDDER] Vectors exported  path=%s  vectors=%d  dim=%d  size=%s",
        output, n_vectors, dim, _human_bytes(file_size),
    )

    return output


def export_metadata_json(
    metadata:    list[dict],
    output_path: str | Path,
) -> Path:
    """
    Write chunk metadata to a JSON sidecar file.

    Parameters
    ----------
    metadata    : list[dict]  — Metadata records from EmbeddingResult.
    output_path : str | Path  — Destination .json file path.

    Returns
    -------
    Path — The written file path.
    """
    import json as json_mod

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with open(output, "w", encoding="utf-8") as fh:
        json_mod.dump(metadata, fh, indent=2, ensure_ascii=False)

    file_size = output.stat().st_size

    logger.info(
        "[EMBEDDER] Metadata exported  path=%s  records=%d  size=%s",
        output, len(metadata), _human_bytes(file_size),
    )

    return output


def export_embedding_result(
    result:      EmbeddingResult,
    output_dir:  str | Path,
    vectors_filename: str = "vectors.bin",
    metadata_filename: str = "metadata.json",
) -> tuple[Path, Path]:
    """
    Export an EmbeddingResult to vectors.bin + metadata.json.

    Convenience wrapper combining export_vectors_bin and
    export_metadata_json.

    Parameters
    ----------
    result       : EmbeddingResult — Output from embed_batch().
    output_dir   : str | Path     — Directory for output files.
    vectors_filename : str        — Binary vectors filename.
    metadata_filename : str       — Metadata JSON filename.

    Returns
    -------
    (vectors_path, metadata_path) — Paths to the written files.

    Raises
    ------
    ValueError
        If result.vector_matrix is None (no vectors produced).
    """
    if result.vector_matrix is None:
        raise ValueError(
            "Cannot export: vector_matrix is None. "
            "Check result.success and per-document error messages."
        )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    vectors_path = export_vectors_bin(
        result.vector_matrix,
        out_dir / vectors_filename,
    )

    metadata_path = export_metadata_json(
        result.metadata,
        out_dir / metadata_filename,
    )

    return vectors_path, metadata_path


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

    with open(bin_path, "rb") as fh:
        header = fh.read(8)
        if len(header) < 8:
            raise ValueError(f"Vectors file too small: {bin_path}")

        n_vectors = np.frombuffer(header[:4], dtype=np.uint32)[0]
        dim       = np.frombuffer(header[4:8], dtype=np.uint32)[0]

        if dim != VECTOR_DIM:
            raise ValueError(
                f"Vector dim {dim} != expected {VECTOR_DIM} in {bin_path}"
            )

        expected_bytes = n_vectors * dim * 4
        data = fh.read(expected_bytes)

        if len(data) < expected_bytes:
            raise ValueError(
                f"Vectors file truncated: expected {expected_bytes} bytes, "
                f"got {len(data)}"
            )

    matrix = np.frombuffer(data, dtype=np.float32).reshape(n_vectors, dim).copy()
    logger.info(
        "[EMBEDDER] Vectors loaded  path=%s  shape=%s",
        bin_path, matrix.shape,
    )
    return matrix


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fail_emb(doc: EmbeddedDocument, message: str) -> None:
    """Mark EmbeddedDocument as failed."""
    doc.success         = False
    doc.error_message   = message
    doc.embedded_chunks = []


def _human_bytes(num_bytes: Union[int, float]) -> str:
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

    # Add project root so ingestion modules resolve
    _project_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(_project_root))

    from ingestion.scanner import scan_directory    # type: ignore[import]
    from ingestion.extractor import extract_batch   # type: ignore[import]
    from ingestion.chunker import chunk_batch       # type: ignore[import]

    parser = argparse.ArgumentParser(
        prog        = "embedder",
        description = "IsoCortex embedder — standalone debug mode.",
    )
    parser.add_argument("root", help="Directory to scan, extract, chunk, and embed.")
    parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
        help=f"Embedding batch size (default: {DEFAULT_BATCH_SIZE}).",
    )
    parser.add_argument(
        "--device", type=str, default="",
        help='Device: "cpu", "cuda", or "" for auto-detect.',
    )
    parser.add_argument(
        "--output-dir", type=str, default="./isocortex_output",
        help="Directory for vectors.bin and metadata.json.",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=120,
        help="Target words per chunk (default: 120).",
    )
    parser.add_argument(
        "--overlap", type=int, default=30,
        help="Word overlap between chunks (default: 30).",
    )
    parser.add_argument(
        "--token-limit", type=int, default=256,
        help="Hard token limit per chunk (default: 256).",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable DEBUG-level logging.",
    )
    args = parser.parse_args()

    _configure_logging(args.verbose)

    # ---- Scan ---------------------------------------------------------
    try:
        scan_result = scan_directory(args.root)
    except (FileNotFoundError, NotADirectoryError, PermissionError) as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)

    if not scan_result.files:
        print("[ERROR] No supported files found.", file=sys.stderr)
        sys.exit(1)

    # ---- Extract ------------------------------------------------------
    extraction_results = extract_batch(scan_result.files)

    # ---- Chunk --------------------------------------------------------
    chunked_docs = chunk_batch(
        extraction_results,
        chunk_size  = args.chunk_size,
        overlap     = args.overlap,
        token_limit = args.token_limit,
    )

    # ---- Embed --------------------------------------------------------
    emb_result = embed_batch(
        chunked_docs,
        batch_size = args.batch_size,
        device     = args.device,
    )

    # ---- Export -------------------------------------------------------
    if emb_result.vector_matrix is not None and emb_result.vector_matrix.shape[0] > 0:
        try:
            v_path, m_path = export_embedding_result(
                emb_result, args.output_dir,
            )
            print(f"\n[OK] Vectors : {v_path}")
            print(f"[OK] Metadata: {m_path}")
        except Exception as exc:  # pylint: disable=broad-except
            print(f"[ERROR] Export failed: {exc}", file=sys.stderr)
            sys.exit(1)
    else:
        print("[ERROR] No vectors produced — nothing to export.", file=sys.stderr)
        sys.exit(1)

    # Report partial failures as warnings (not fatal)
    failed_docs = [d for d in emb_result.documents if not d.success]
    if failed_docs:
        print(f"\n[WARNING] {len(failed_docs)} document(s) failed:", file=sys.stderr)
        for doc in failed_docs:
            print(
                f"  FAIL  {doc.source_path.name}: {doc.error_message}",
                file=sys.stderr,
            )

    # ---- Report -------------------------------------------------------
    print()
    print("=" * 70)
    print(f"  EMBEDDING REPORT  —  {args.root}")
    print("=" * 70)

    for doc in emb_result.documents:
        status = "OK  " if doc.success else "FAIL"
        print(
            f"  [{status}]  [{doc.format_category:<14}]"
            f"  chunks={doc.total_chunks:<4}"
            f"  tokens={doc.total_tokens:<6}"
            f"  elapsed={doc.elapsed_seconds:.3f}s"
            f"  {doc.source_path.name}"
        )
        if not doc.success:
            print(f"           error: {doc.error_message}")

    total_chunks = emb_result.total_chunks
    total_tokens = emb_result.total_tokens
    succeeded    = sum(1 for d in emb_result.documents if d.success)
    failed       = len(emb_result.documents) - succeeded

    print("=" * 70)
    print(f"  Documents : {len(emb_result.documents)}  (OK: {succeeded}  FAIL: {failed})")
    print(f"  Chunks    : {total_chunks}")
    print(f"  Tokens    : {total_tokens}")
    if emb_result.vector_matrix is not None:
        print(f"  Matrix    : {emb_result.vector_matrix.shape}")
    print(f"  Elapsed   : {emb_result.elapsed_seconds:.3f}s")
    print("=" * 70)
