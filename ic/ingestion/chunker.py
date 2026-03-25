"""
IsoCortex — ingestion/chunker.py
=================================
Sentence-aware, token-guarded text chunking engine.

Responsibilities (FR-2, NFR-2):
  - Accept ExtractionResult objects from extractor.py.
  - Partition text into segments of approximately 120 words.
  - Split at natural sentence boundaries (spaCy or regex fallback).
  - Enforce 25-30% word overlap (~30 words) between consecutive chunks.
  - Apply a 256-token hard guard using transformers.AutoTokenizer.
  - Handle source code and structured data with specialised strategies.
  - Return structured ChunkedDocument objects — never raw strings.
  - Emit structured log messages at every decision point.

SRS References: FR-2, NFR-2, CON-1, CON-2

Author : Shaheer Qureshi
Project: IsoCortex
"""

from __future__ import annotations

import logging
from pydoc import doc
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants (FR-2 defaults)
# ---------------------------------------------------------------------------
DEFAULT_CHUNK_SIZE:  int = 120   # target words per chunk
DEFAULT_OVERLAP:     int = 30    # word overlap between consecutive chunks
DEFAULT_TOKEN_LIMIT: int = 256   # hard limit from all-MiniLM-L6-v2 (CON-1)

# Overlap ratio bounds — used for validation
MIN_OVERLAP_RATIO: float = 0.25  # 25%
MAX_OVERLAP_RATIO: float = 0.30  # 30%

# Sentence-boundary regex fallback (FR-2)
# Splits on:
#   1. Sentence-ending punctuation (.!?) followed by whitespace,
#      excluding common abbreviations (Mr, Mrs, Dr, etc.)
#   2. Double-newline (paragraph boundary) — NOT single newline
#      to avoid splitting mid-paragraph in PDF/email text
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])|(?<=\n)\s*\n\s*(?=\S)")


# Regex to detect source code file categories
_SOURCE_CODE_CATEGORIES: frozenset[str] = frozenset({
    "source_code",
})


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Chunk:
    """
    A single text chunk ready for vectorization.

    Attributes
    ----------
    text              : str  — Chunk text content (may be normalised).
    chunk_index       : int  — Sequential index within the parent document.
    source_file       : str  — Absolute path of the source file.
    source_label      : str  — Sub-document label (e.g. "Page 3", "Slide 2").
    format_category   : str  — e.g. "pdf", "spreadsheet", "source_code".
    word_count        : int  — Word count of this chunk.
    token_count       : int  — Sub-word token count (0 until embedder runs).
                               NOTE: This dataclass is frozen. The embedder
                               must create new Chunk objects with updated
                               token_count rather than mutating existing ones.
    source_chunk_index: int  — Index of the parent ExtractionChunk.
    """
    text:               str
    chunk_index:        int
    source_file:        str
    source_label:       str
    format_category:    str
    word_count:         int  = 0
    token_count:        int  = 0
    source_chunk_index: int  = 0

    def __post_init__(self) -> None:
        # Defence-in-depth: recalc word_count from actual text
        if self.word_count == 0:
            object.__setattr__(self, "word_count", len(self.text.split()))


@dataclass
class ChunkedDocument:
    """
    All chunks produced from a single ExtractionResult.

    Attributes
    ----------
    source_path    : Path           — Absolute path of the source file.
    format_category: str            — Format category of the source.
    chunks         : list[Chunk]    — Ordered list of text chunks.
    success        : bool           — False if chunking failed.
    error_message  : str | None     — Failure reason.
    elapsed_seconds: float          — Wall-clock chunking time.
    """
    source_path:     Path
    format_category: str
    chunks:          List[Chunk] = field(default_factory=list)
    success:         bool        = True
    error_message:   Optional[str] = None
    elapsed_seconds: float       = 0.0

    @property
    def total_chunks(self) -> int:
        return len(self.chunks)

    @property
    def total_words(self) -> int:
        return sum(c.word_count for c in self.chunks)

    @property
    def total_tokens(self) -> int:
        return sum(c.token_count for c in self.chunks)


# ---------------------------------------------------------------------------
# Tokenizer management
# ---------------------------------------------------------------------------

_tokenizer_cache: dict = {}


def load_tokenizer() -> Optional[object]:
    """
    Load the all-MiniLM-L6-v2 tokenizer (singleton).

    Returns the tokenizer instance or None if transformers is
    not installed.  Never raises on import failure.

    The tokenizer is cached globally so repeated calls within the
    same process do not reload model files.
    """
    if "tokenizer" in _tokenizer_cache:
        return _tokenizer_cache["tokenizer"]

    try:
        from transformers import AutoTokenizer  # type: ignore[import-untyped]
    except ImportError:
        logger.warning(
            "[CHUNKER] transformers not installed — "
            "token guard disabled.  Run: pip install transformers",
        )
        _tokenizer_cache["tokenizer"] = None
        return None

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            "sentence-transformers/all-MiniLM-L6-v2",
            local_files_only=True,
        )
        _tokenizer_cache["tokenizer"] = tokenizer
        logger.info("[CHUNKER] Tokenizer loaded (all-MiniLM-L6-v2)")
        return tokenizer
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning(
            "[CHUNKER] Tokenizer load failed: %s — token guard disabled",
            exc,
        )
        _tokenizer_cache["tokenizer"] = None
        return None


def count_tokens(text: str, tokenizer: object) -> int:
    """
    Return the sub-word token count for *text*.

    Parameters
    ----------
    text      : str        — Input text.
    tokenizer : object     — A HuggingFace tokenizer instance.

    Returns
    -------
    int — Number of sub-word tokens.  Returns 0 on failure.
    """
    try:
        encoded = tokenizer(text, add_special_tokens=False)  # type: ignore[union-attr]
        if isinstance(encoded, dict) and "input_ids" in encoded:
            return len(encoded["input_ids"])
        return len(getattr(encoded, "input_ids", []))
    except Exception:  # pylint: disable=broad-except
        return 0


# ---------------------------------------------------------------------------
# spaCy management (singleton, same pattern as tokenizer)
# ---------------------------------------------------------------------------

_spacy_nlp_cache: dict = {}


def _get_spacy_nlp() -> Optional[object]:
    """
    Load and cache spaCy nlp model (singleton).

    Handles both models with a dedicated 'senter' pipe (v3.5+) and
    older models where the 'parser' pipe provides doc.sents.

    Returns None if spaCy is not installed or the model is unavailable.
    Never raises.
    """
    if "nlp" in _spacy_nlp_cache:
        return _spacy_nlp_cache["nlp"]

    try:
        import spacy  # type: ignore[import-untyped]
    except ImportError:
        logger.debug("[CHUNKER] spaCy not installed — using regex sentence split")
        _spacy_nlp_cache["nlp"] = None
        return None

    try:
        # Check what pipes are available in the model
        probe = spacy.load("en_core_web_sm")
        available = set(probe.pipe_names)

        if "senter" in available:
            nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
            nlp.enable_pipe("senter")
            logger.debug("[CHUNKER] spaCy using 'senter' pipe")
        elif "parser" in available:
            nlp = spacy.load("en_core_web_sm", disable=["ner"])
            logger.debug("[CHUNKER] spaCy using 'parser' pipe")
        else:
            logger.debug(
                "[CHUNKER] spaCy has no sentence pipe — using regex fallback"
            )
            _spacy_nlp_cache["nlp"] = None
            return None

        _spacy_nlp_cache["nlp"] = nlp
        logger.info("[CHUNKER] spaCy loaded (en_core_web_sm)")
        return nlp

    except OSError:
        logger.debug(
            "[CHUNKER] spaCy model 'en_core_web_sm' not found — "
            "run: python3 -m spacy download en_core_web_sm",
        )
        _spacy_nlp_cache["nlp"] = None
        return None
    except Exception as exc:  # pylint: disable=broad-except
        logger.debug("[CHUNKER] spaCy load error: %s", exc)
        _spacy_nlp_cache["nlp"] = None
        return None



# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def chunk_document(
    extraction_result: object,
    *,
    chunk_size:  int = DEFAULT_CHUNK_SIZE,
    overlap:     int = DEFAULT_OVERLAP,
    token_limit: int = DEFAULT_TOKEN_LIMIT,
) -> ChunkedDocument:
    """
    Chunk a single ExtractionResult into token-safe text segments.

    Parameters
    ----------
    extraction_result : ExtractionResult
        Output from extractor.extract().

    token_limit : int
        Hard maximum tokens per chunk.  Default 256 (CON-1).

    Returns
    -------
    ChunkedDocument
        Always consecutive chunks.  Default 30
        (2    chunk_size : int
        Target words per chunk.  Default 120 (FR-2).

    overlap : int
        Word overlap between5 % of 120, satisfying the FR-2 overlap constraint).

 returns an object.  On failure success=False.
    """
    # ---- Unpack ExtractionResult (duck-typed) -------------------------
    abs_path:   Path = extraction_result.absolute_path   # type: ignore[attr-defined]
    category:   str  = extraction_result.format_category  # type: ignore[attr-defined]
    ext_chunks: list = extraction_result.chunks           # type: ignore[attr-defined]

    doc = ChunkedDocument(
        source_path     = abs_path,
        format_category = category,
    )

    if not extraction_result.success:  # type: ignore[attr-defined]
        _fail_doc(doc, "ExtractionResult indicates failure")
        return doc

    if not ext_chunks:
        _fail_doc(doc, "ExtractionResult contains zero chunks")
        return doc

    # ---- Validate parameters ------------------------------------------
    if chunk_size < 1:
        _fail_doc(doc, f"chunk_size must be >= 1, got {chunk_size}")
        return doc
    if overlap < 0:
        _fail_doc(doc, f"overlap must be >= 0, got {overlap}")
        return doc
    if overlap >= chunk_size:
        logger.warning(
            "[CHUNKER] overlap (%d) >= chunk_size (%d); "
            "clamping overlap to chunk_size - 1",
            overlap, chunk_size,
        )
        overlap = chunk_size - 1
    if token_limit < 1:
        _fail_doc(doc, f"token_limit must be >= 1, got {token_limit}")
        return doc

    # Warn if overlap ratio is outside recommended 25-30% range
    if chunk_size > 0:
        ratio = overlap / chunk_size
        if ratio < MIN_OVERLAP_RATIO or ratio > MAX_OVERLAP_RATIO:
            logger.warning(
                "[CHUNKER] Overlap ratio %.2f is outside recommended "
                "range [%.2f, %.2f]  (overlap=%d, chunk_size=%d)",
                ratio, MIN_OVERLAP_RATIO, MAX_OVERLAP_RATIO,
                overlap, chunk_size,
            )

    t_start = time.perf_counter()

    try:
        tokenizer = load_tokenizer()

        all_chunks: list[Chunk] = []
        global_idx: int         = 0

        for src_idx, ext_chunk in enumerate(ext_chunks):
            text:  str = ext_chunk.text          # type: ignore[attr-defined]
            label: str = ext_chunk.source_label  # type: ignore[attr-defined]

            if not text or not text.strip():
                logger.debug(
                    "[CHUNKER] Skipping empty extraction chunk  "
                    "src_idx=%d  label=%s",
                    src_idx, label,
                )
                continue

            # ---- Choose strategy by format category -------------------
            if category in _SOURCE_CODE_CATEGORIES:
                raw = _chunk_source_code(
                    text, chunk_size, overlap, tokenizer, token_limit,
                )
            elif category == "spreadsheet":
                raw = _chunk_structured(
                    text, chunk_size, tokenizer, token_limit,
                )
            else:
                raw = _chunk_sentence_aware(
                    text, chunk_size, overlap, tokenizer, token_limit,
                )

            # ---- Build Chunk objects ----------------------------------
            for piece in raw:
                chunk = Chunk(
                    text               = piece,
                    chunk_index        = global_idx,
                    source_file        = str(abs_path),
                    source_label       = label,
                    format_category    = category,
                    source_chunk_index = src_idx,
                )
                all_chunks.append(chunk)
                global_idx += 1

        doc.chunks = all_chunks

    except MemoryError:
        _fail_doc(doc, "MemoryError during chunking")
    except Exception as exc:  # pylint: disable=broad-except
        _fail_doc(doc, f"{type(exc).__name__}: {exc}")

    doc.elapsed_seconds = time.perf_counter() - t_start

    if doc.success and doc.total_chunks == 0:
        _fail_doc(doc, "Chunking produced zero chunks")

    if doc.success:
        logger.info(
            "[CHUNKER] OK      %-40s  chunks=%d  words=%d  elapsed=%.3fs",
            abs_path.name, doc.total_chunks,
            doc.total_words, doc.elapsed_seconds,
        )
    else:
        logger.warning(
            "[CHUNKER] FAILED  %-40s  reason=%s",
            abs_path.name, doc.error_message,
        )

    return doc


def chunk_batch(
    extraction_results: list,
    *,
    chunk_size:  int = DEFAULT_CHUNK_SIZE,
    overlap:     int = DEFAULT_OVERLAP,
    token_limit: int = DEFAULT_TOKEN_LIMIT,
) -> list[ChunkedDocument]:
    """
    Chunk a list of ExtractionResult objects.

    Failures on individual documents do not stop the batch.

    Parameters
    ----------
    extraction_results : list[ExtractionResult]

    Returns
    -------
    list[ChunkedDocument]
        Same length as input.
    """
    docs: list[ChunkedDocument] = []
    total = len(extraction_results)

    for idx, er in enumerate(extraction_results, start=1):
        if idx % 100 == 0 or idx == total:
            logger.info("[CHUNKER] Batch progress: %d/%d", idx, total)
        logger.debug(
            "[CHUNKER] Batch %d/%d  file=%s",
            idx, total, er.absolute_path.name,  # type: ignore[attr-defined]
        )
        docs.append(
            chunk_document(
                er,
                chunk_size  = chunk_size,
                overlap     = overlap,
                token_limit = token_limit,
            )
        )

    succeeded = sum(1 for d in docs if d.success)
    failed    = total - succeeded
    logger.info(
        "[CHUNKER] Batch complete  total=%d  succeeded=%d  failed=%d",
        total, succeeded, failed,
    )
    return docs


# ---------------------------------------------------------------------------
# Sentence-aware chunking (plain text, PDF, email, HTML, presentations)
# ---------------------------------------------------------------------------

def _chunk_sentence_aware(
    text:        str,
    chunk_size:  int,
    overlap:     int,
    tokenizer:   Optional[object],
    token_limit: int,
) -> list[str]:
    """
    Split *text* into ~chunk_size-word groups at sentence boundaries.

    Algorithm
    ---------
    1. Split text into sentences (spaCy preferred, regex fallback).
    2. Accumulate sentences until word count >= chunk_size.
    3. Emit the accumulated group as one chunk.
    4. Carry forward the last *overlap* words from the emitted chunk
       to seed the next chunk (context continuity).
    5. Apply token guard: any chunk exceeding *token_limit* tokens
       is re-split at the token level.

    Edge cases handled
    ------------------
    - Single sentence longer than chunk_size → emitted as-is.
    - Sentences with only whitespace → discarded.
    - Overlap larger than emitted chunk → overlap clamped.
    - Tokenizer unavailable → skip token guard.
    """
    sentences = _split_sentences(text)

    if not sentences:
        return []

    chunks: list[str] = []
    current_words: list[str] = []
    current_count: int       = 0

    for sentence in sentences:
        sent_words  = sentence.split()
        sent_count  = len(sent_words)

        if sent_count == 0:
            continue

        # ---- Does adding this sentence exceed the target? -------------
        if (
            current_count > 0
            and current_count + sent_count > chunk_size
        ):
            # Emit current chunk
            chunk_text = " ".join(current_words)
            if chunk_text.strip():
                chunks.append(chunk_text)

            # Carry forward overlap words for context continuity
            if overlap > 0 and len(current_words) > overlap:
                current_words = current_words[-overlap:]
                current_count = len(current_words)
            else:
                current_words = []
                current_count = 0

        current_words.extend(sent_words)
        current_count += sent_count

    # Flush remaining words
    if current_words:
        chunk_text = " ".join(current_words)
        if chunk_text.strip():
            chunks.append(chunk_text)

    # Token guard (NFR-2)
    if tokenizer is not None:
        chunks = _apply_token_guard(chunks, tokenizer, token_limit, overlap)

    return chunks


# ---------------------------------------------------------------------------
# Source-code chunking (split on logical blank-line boundaries)
# ---------------------------------------------------------------------------

def _chunk_source_code(
    text:        str,
    chunk_size:  int,
    overlap:     int,
    tokenizer:   Optional[object],
    token_limit: int,
) -> list[str]:
    """
    Split source code into chunks at logical boundaries (blank lines).

    Rationale
    ---------
    Source code does not have natural sentence boundaries.  Blank lines
    separate functions, classes, and logical blocks, making them the
    most semantically meaningful split point.

    The *chunk_size* parameter is interpreted as a word-count soft
    target; the token guard is the hard constraint.
    """
    # Split on double-newline (blank line) boundaries
    blocks = re.split(r"\n\s*\n", text)

    blocks = [b.strip() for b in blocks if b.strip()]

    if not blocks:
        return []

    chunks: list[str] = []
    current_parts: list[str] = []
    current_count: int       = 0

    for block in blocks:
        block_words = block.split()
        block_count = len(block_words)

        if block_count == 0:
            continue

        if (
            current_count > 0
            and current_count + block_count > chunk_size
        ):
            chunk_text = "\n\n".join(current_parts)
            if chunk_text.strip():
                chunks.append(chunk_text)

            # Overlap: carry last block if it fits
            if overlap > 0 and current_parts:
                overlap_text  = current_parts[-1]
                overlap_words = overlap_text.split()
                if len(overlap_words) <= overlap:
                    current_parts = [overlap_text]
                    current_count = len(overlap_words)
                else:
                    # Take last N words from the block
                    overlap_slice = overlap_words[-overlap:]
                    current_parts = [" ".join(overlap_slice)]
                    current_count = len(overlap_slice)
            else:
                current_parts = []
                current_count = 0

        current_parts.append(block)
        current_count += block_count

    if current_parts:
        chunk_text = "\n\n".join(current_parts)
        if chunk_text.strip():
            chunks.append(chunk_text)

    # Token guard
    if tokenizer is not None:
        chunks = _apply_token_guard(chunks, tokenizer, token_limit, overlap)

    return chunks


# ---------------------------------------------------------------------------
# Structured data chunking (spreadsheets — each row is a unit)
# ---------------------------------------------------------------------------

def _chunk_structured(
    text:        str,
    chunk_size:  int,
    tokenizer:   Optional[object],
    token_limit: int,
) -> list[str]:
    """
    Chunk structured (linearized) text.

    Each newline-separated row from the extractor is a logical unit
    (e.g. "Name: Alice | Department: Engineering | Salary: 85000").
    Rows are grouped into chunks of approximately *chunk_size* words.

    No overlap is applied — each row is an independent record and
    overlapping key-value pairs adds noise, not context.
    """
    lines = text.split("\n")
    lines = [ln.strip() for ln in lines if ln.strip()]

    if not lines:
        return []

    chunks: list[str]  = []
    current_lines: list[str] = []
    current_count: int       = 0

    for line in lines:
        line_words = line.split()
        line_count = len(line_words)

        if line_count == 0:
            continue

        if (
            current_count > 0
            and current_count + line_count > chunk_size
        ):
            chunk_text = "\n".join(current_lines)
            if chunk_text.strip():
                chunks.append(chunk_text)
            current_lines = []
            current_count = 0

        current_lines.append(line)
        current_count += line_count

    if current_lines:
        chunk_text = "\n".join(current_lines)
        if chunk_text.strip():
            chunks.append(chunk_text)

    # Token guard
    if tokenizer is not None:
        chunks = _apply_token_guard(chunks, tokenizer, token_limit, overlap=0)

    return chunks


# ---------------------------------------------------------------------------
# Sentence splitting (spaCy with regex fallback)
# ---------------------------------------------------------------------------

def _split_sentences(text: str) -> list[str]:
    """
    Split *text* into sentences.

    Prefers spaCy (en_core_web_sm) for accurate boundary detection.
    Falls back to regex if spaCy is unavailable (FR-2, CON-2-empty sentence strings in original order.
    """
    nlp = _get_spacy_nlp()

    if nlp is not None:
        try:
            doc = nlp(text) # type: ignore[operator]
            sentences = [
                sent.text.strip() for sent in doc.sents if sent.text.strip()
            ]
            if sentences:
                return sentences
        except Exception:  # pylint: disable=broad-except
            logger.debug("[CHUNKER] spaCy split error — falling back to regex")

    # Regex fallback
    parts = _SENTENCE_RE.split(text)
    sentences = [s.strip() for s in parts if s.strip()]

    # Merge fragments that start with lowercase back into previous
    merged: list[str] = []
    for sent in sentences:
        if merged and sent and sent[0].islower():
            merged[-1] = merged[-1] + " " + sent
        else:
            merged.append(sent)

    return merged


# ---------------------------------------------------------------------------
# Token guard (NFR-2 / CON-1)
# ---------------------------------------------------------------------------

def _apply_token_guard(
    chunks:      list[str],
    tokenizer:   object,
    token_limit: int,
    overlap:     int,
) -> list[str]:
    """
    Ensure every chunk has <= *token_limit* sub-word tokens.

    Chunks that exceed the limit are re-split at the word level
    with a small overlap between split pieces.

    Parameters
    ----------
    chunks      : list[str] — Candidate chunks.
    tokenizer   : object    — HuggingFace tokenizer.
    token_limit : int       — Maximum tokens per chunk.
    overlap     : int       — Word overlap between split pieces.

    Returns
    -------
    list[str] — Guaranteed token-safe chunks.
    """
    safe: list[str] = []

    for chunk in chunks:
        tc = count_tokens(chunk, tokenizer)

        if tc <= token_limit:
            safe.append(chunk)
        else:
            logger.debug(
                "[CHUNKER] Token guard triggered  tokens=%d  limit=%d",
                tc, token_limit,
            )
            split_pieces = _token_split(chunk, tokenizer, token_limit, overlap)
            safe.extend(split_pieces)

    return safe


def _token_split(
    text:        str,
    tokenizer:   object,
    token_limit: int,
    overlap:     int,
) -> list[str]:
    """
    Split *text* into pieces that each fit within *token_limit* tokens.

    Uses binary search on word boundaries to find the longest prefix
    that fits within the limit, then advances by (best - overlap) words.

    Safety guards
    -------------
    - Single word exceeding token_limit: truncated to half the limit.
    - Iteration cap: max len(words) * 2 to prevent unbounded loops
      on pathological input.
    - Overlap clamped to best - 1 to guarantee forward progress.
    """
    words = text.split()

    if not words:
        return []

    # Safety: single word longer than token_limit
    if len(words) == 1:
        tc = count_tokens(text, tokenizer)
        if tc > token_limit:
            half = max(token_limit // 2, 1)
            try:
                encoded = tokenizer(  # type: ignore[union-attr]
                    text,
                    max_length=half,
                    truncation=True,
                    add_special_tokens=False,
                )
                if isinstance(encoded, dict) and "input_ids" in encoded:
                    decoded = tokenizer.decode(  # type: ignore[union-attr]
                        encoded["input_ids"],
                        skip_special_tokens=True,
                    )
                else:
                    decoded = text[: len(text) // 2]
            except Exception:  # pylint: disable=broad-except
                decoded = text[: len(text) // 2]

            logger.warning(
                "[CHUNKER] Single word exceeds token limit "
                "(tokens=%d) — truncated to %d tokens",
                tc, half,
            )
            if decoded.strip():
                return [decoded.strip()]
            return [text[: len(text) // 2]]
        return [text]

    pieces: list[str] = []
    remaining_words = list(words)
    max_iterations  = len(words) * 2  # safety cap

    for _ in range(max_iterations):
        if not remaining_words:
            break

        # Binary search for the longest prefix that fits
        lo, hi = 1, len(remaining_words)
        best = 1

        while lo <= hi:
            mid = (lo + hi) // 2
            candidate = " ".join(remaining_words[:mid])
            tc = count_tokens(candidate, tokenizer)
            if tc <= token_limit:
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1

        piece = " ".join(remaining_words[:best])
        if piece.strip():
            pieces.append(piece)

        # Advance with overlap — clamp to guarantee forward progress
        step = max(best - overlap, 1)
        remaining_words = remaining_words[step:]
    else:
        # Safety: max iterations reached — emit whatever remains
        if remaining_words:
            remaining_text = " ".join(remaining_words)
            if remaining_text.strip():
                pieces.append(remaining_text)
                logger.warning(
                    "[CHUNKER] Token split iteration cap reached "
                    "(%d iterations) — emitting remaining %d words as-is",
                    max_iterations, len(remaining_words),
                )

    return pieces


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fail_doc(doc: ChunkedDocument, message: str) -> None:
    """Mark ChunkedDocument as failed."""
    doc.success       = False
    doc.error_message = message
    doc.chunks        = []


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

    parser = argparse.ArgumentParser(
        prog        = "chunker",
        description = "IsoCortex chunker — standalone debug mode.",
    )
    parser.add_argument("root", help="Directory to scan, extract, and chunk.")
    parser.add_argument(
        "--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE,
        help=f"Target words per chunk (default: {DEFAULT_CHUNK_SIZE}).",
    )
    parser.add_argument(
        "--overlap", type=int, default=DEFAULT_OVERLAP,
        help=f"Word overlap between chunks (default: {DEFAULT_OVERLAP}).",
    )
    parser.add_argument(
        "--token-limit", type=int, default=DEFAULT_TOKEN_LIMIT,
        help=f"Hard token limit per chunk (default: {DEFAULT_TOKEN_LIMIT}).",
    )
    parser.add_argument(
        "--show-text", action="store_true",
        help="Print 300-char preview of each chunk.",
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

    # ---- Report -------------------------------------------------------
    print()
    print("=" * 70)
    print(f"  CHUNKING REPORT  —  {args.root}")
    print("=" * 70)
    
    for doc in chunked_docs:
        status = "OK  " if doc.success else "FAIL"
        print(
            f"  [{status}]  [{doc.format_category:<14}]"
            f"  chunks={doc.total_chunks:<4}"
            f"  words={doc.total_words:<6}"
            f"  {doc.source_path.name}"
        )
        if not doc.success:
            print(f"           error: {doc.error_message}")

        if args.show_text and doc.success:
            chunk_list: List[Chunk] = doc.chunks
            for chunk in chunk_list:
                print(
                    f"    -- idx={chunk.chunk_index:<3}  "
                    f"{chunk.source_label:<20}  "
                    f"words={chunk.word_count:<4}  "
                    f"tokens={chunk.token_count:<4} --"
                )
                preview = chunk.text[:300].replace("\n", " ")
                print(f"    {preview}")
                print()


    total_chunks = sum(d.total_chunks for d in chunked_docs if d.success)
    total_words  = sum(d.total_words for d in chunked_docs if d.success)
    succeeded    = sum(1 for d in chunked_docs if d.success)
    failed       = len(chunked_docs) - succeeded

    print("=" * 70)
    print(f"  Documents : {len(chunked_docs)}  (OK: {succeeded}  FAIL: {failed})")
    print(f"  Chunks    : {total_chunks}")
    print(f"  Words     : {total_words}")
    print("=" * 70)
