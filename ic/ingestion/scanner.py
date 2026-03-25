"""
IsoCortex — ingestion/scanner.py
=================================
Recursive, production-grade directory scanner.

Responsibilities (FR-1):
  - Accept a root directory path from the CLI.
  - Recursively traverse all subdirectories.
  - Identify every file whose extension is in the supported format registry.
  - Detect and skip circular symlinks.
  - Handle permission errors, broken symlinks, and unreadable paths gracefully.
  - Respect a configurable ignore-list (hidden dirs, __pycache__, .git, etc.).
  - Return structured ScanResult objects — never raw strings.
  - Emit structured log messages at every decision point.

SRS References: FR-1, FR-7 (modified_ts), UC-1, UC-3, DR-1, DR-2,
                NFR-3, NFR-4, CON-5, OOS-2

Author : Shaheer Qureshi
Project: IsoCortex
"""

from __future__ import annotations

import logging
import os
import stat
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator, Optional, Tuple
from typing import Generator, Optional, Protocol, Tuple

# ---------------------------------------------------------------------------
# Protocol for summary objects (ScanSummary and _NullSummary)
# ---------------------------------------------------------------------------
class _SummaryProtocol(Protocol):
    """Interface that both ScanSummary and _NullSummary satisfy."""
    def record_skip(self, path: Path, reason: str) -> None: ...
    def record_expected_skip(self, path: Path, reason: str) -> None: ...

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Supported format registry  (FR-1 / Section 1.3 Scope table)
# ---------------------------------------------------------------------------
# Maps each file extension (lowercase, with leading dot) to a
# human-readable format category label used downstream by extractor.py.
# ---------------------------------------------------------------------------
SUPPORTED_EXTENSIONS: dict[str, str] = {
    # Plain Text
    ".txt":  "plain_text",
    ".md":   "plain_text",
    ".log":  "plain_text",
    # Source Code
    ".py":   "source_code",
    ".cpp":  "source_code",
    ".c":    "source_code",
    ".h":    "source_code",
    ".js":   "source_code",
    ".ts":   "source_code",
    ".java": "source_code",
    # PDF
    ".pdf":  "pdf",
    # Word Documents
    ".docx": "word",
    ".odt":  "word",
    # Presentations
    ".pptx": "presentation",
    ".odp":  "presentation",
    # Spreadsheets
    ".xlsx": "spreadsheet",
    ".xls":  "spreadsheet",
    ".ods":  "spreadsheet",
    ".csv":  "spreadsheet",
    # Email & Web
    ".eml":  "email",
    ".html": "web",
    ".htm":  "web",
}

# ---------------------------------------------------------------------------
# Default directories to skip entirely (DR-2 + NFR-3 compliance).
# Keeps the scanner out of VCS internals, caches, and virtual envs.
# ---------------------------------------------------------------------------
DEFAULT_IGNORE_DIRS: frozenset[str] = frozenset({
    ".git",
    ".hg",
    ".svn",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".tox",
    "node_modules",
    "venv",
    ".venv",
    "env",
    ".env",
    "dist",
    "build",
    ".idea",
    ".vscode",
})

# Maximum filesystem depth to prevent RecursionError on pathological trees.
MAX_RECURSION_DEPTH: int = 200


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ScannedFile:
    """
    Immutable record describing a single file accepted by the scanner.

    Attributes
    ----------
    absolute_path    : Path  — Fully resolved absolute path to the file.
    relative_path    : Path  — Path relative to the root directory.
    extension        : str   — Lowercase extension e.g. '.pdf'.
    format_category  : str   — Category label from SUPPORTED_EXTENSIONS.
    size_bytes       : int   — File size in bytes at scan time.
    modified_ts      : float — Last-modified Unix timestamp.
                               Used for incremental re-indexing (UC-3/FR-7).
    """
    absolute_path:   Path
    relative_path:   Path
    extension:       str
    format_category: str
    size_bytes:      int
    modified_ts:     float


@dataclass
class ScanSummary:
    """
    Aggregated statistics returned alongside the scanned file list.

    Attributes
    ----------
    root_directory  : Path                  — The directory that was scanned.
    total_visited   : int                   — Every filesystem node visited.
    total_accepted  : int                   — Files with a supported extension.
    total_skipped   : int                   — Files/dirs skipped for any reason.
    skipped_details : list[dict[str, str]]  — Per-skip reason records.
    elapsed_seconds : float                 — Wall-clock time for the scan.
    """
    root_directory:  Path
    total_visited:   int                    = 0
    total_accepted:  int                    = 0
    total_skipped:   int                    = 0
    skipped_details: list[dict[str, str]]   = field(default_factory=list)
    elapsed_seconds: float                  = 0.0

    def record_skip(self, path: Path, reason: str) -> None:
        """
        Increment skip counter and store the human-readable reason.
        Logged at WARNING — use for genuinely unexpected skips.
        """
        self.total_skipped += 1
        self.skipped_details.append({"path": str(path), "reason": reason})
        logger.warning(
            "[SCANNER] Skipped %-60s  reason=%s", path, reason,
        )

    def record_expected_skip(self, path: Path, reason: str) -> None:
        """
        Increment skip counter for expected/normal filtering.
        Logged at DEBUG to avoid flooding output on large trees.
        Examples: unsupported extension, hidden directory, ignored dir.
        """
        self.total_skipped += 1
        self.skipped_details.append({"path": str(path), "reason": reason})
        logger.debug(
            "[SCANNER] Skipped (expected) %-60s  reason=%s", path, reason,
        )


@dataclass
class ScanResult:
    """
    Top-level return value of :func:`scan_directory`.

    Attributes
    ----------
    files   : list[ScannedFile] — All accepted files in traversal order.
    summary : ScanSummary       — Aggregate statistics for the scan run.
    """
    files:   list[ScannedFile]
    summary: ScanSummary


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _check_file(
    path: Path,
    max_file_size_bytes: int,
) -> Tuple[bool, str, int, float]:
    """
    Single stat call that checks readability, size, and returns metadata.

    Returns
    -------
    (ok, reason, size_bytes, modified_ts)
      ok=True  means the file passed all checks.
      ok=False means the file should be skipped; reason is human-readable.
    """
    try:
        st = path.stat()
    except PermissionError:
        return False, "permission_denied", 0, 0.0
    except OSError as exc:
        return False, f"os_error:{exc.strerror}", 0, 0.0

    if stat.S_ISDIR(st.st_mode):
        return False, "is_directory", 0, 0.0

    if not stat.S_ISREG(st.st_mode):
        return False, "not_regular_file", 0, 0.0

    if not os.access(path, os.R_OK):
        return False, "not_readable", 0, 0.0

    if st.st_size > max_file_size_bytes:
        mb = st.st_size // (1024 * 1024)
        limit_mb = max_file_size_bytes // (1024 * 1024)
        return False, f"exceeds_size_limit:{mb}MB>{limit_mb}MB", 0, 0.0

    return True, "", st.st_size, st.st_mtime


def _resolve_safe(path: Path) -> Optional[Path]:
    """
    Resolve path to its canonical absolute form.
    Returns None if resolution fails (circular symlink, broken link).
    """
    try:
        return path.resolve(strict=True)
    except (OSError, RuntimeError):
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def scan_directory(
    root: str | Path,
    *,
    extra_ignore_dirs: Optional[set[str]] = None,
    max_file_size_mb: float = 500.0,
    max_files: int = 0,
    follow_symlinks: bool = False,
    include_hidden: bool = False,
) -> ScanResult:
    """
    Recursively scan *root* and return all files with supported extensions.

    Parameters
    ----------
    root : str | Path
        The root directory to scan. Must exist and be a directory.

    extra_ignore_dirs : set[str] | None
        Additional directory *names* to ignore on top of
        DEFAULT_IGNORE_DIRS. Example: ``{"scratch", "tmp"}``.

    max_file_size_mb : float
        Files larger than this threshold (MB) are skipped.
        Default: 500 MB. Set to ``float("inf")`` to disable.

    max_files : int
        Stop scanning after accepting this many files.
        0 means unlimited. Useful as a safety cap.

    follow_symlinks : bool
        If True, symbolic links to directories are followed and
        symbolic links to files are accepted.
        Circular links are always detected and skipped regardless.

    include_hidden : bool
        If True, directories starting with '.' are no longer
        auto-skipped. Only DEFAULT_IGNORE_DIRS and extra_ignore_dirs
        apply. Default: False.

    Returns
    -------
    ScanResult
        Contains the ordered list of ScannedFile objects and a
        ScanSummary with aggregate statistics.

    Raises
    ------
    FileNotFoundError
        If *root* does not exist.
    NotADirectoryError
        If *root* exists but is not a directory.
    PermissionError
        If *root* itself cannot be read.

    Examples
    --------
    >>> result = scan_directory("./my_documents")
    >>> for f in result.files:
    ...     print(f.relative_path, f.format_category)
    >>> print(result.summary.total_accepted, "files accepted")
    """
    root_path = Path(root).expanduser()

    # ------------------------------------------------------------------
    # Pre-flight validation
    # ------------------------------------------------------------------
    if not root_path.exists():
        raise FileNotFoundError(
            f"[IsoCortex] Root directory does not exist: '{root_path}'"
        )
    if not root_path.is_dir():
        raise NotADirectoryError(
            f"[IsoCortex] Path is not a directory: '{root_path}'"
        )
    if not os.access(root_path, os.R_OK):
        raise PermissionError(
            f"[IsoCortex] Cannot read root directory: '{root_path}'"
        )

    root_path = root_path.resolve()

    # ------------------------------------------------------------------
    # Build the ignore set
    # ------------------------------------------------------------------
    ignore_dirs: frozenset[str] = DEFAULT_IGNORE_DIRS
    if extra_ignore_dirs:
        ignore_dirs = ignore_dirs | frozenset(extra_ignore_dirs)

    max_file_size_bytes = int(max_file_size_mb * 1024 * 1024)

    logger.info(
        "[SCANNER] Starting scan  root=%s  max_size=%.1f MB  "
        "follow_symlinks=%s  include_hidden=%s  ignore_dirs=%d  max_files=%d",
        root_path, max_file_size_mb, follow_symlinks,
        include_hidden, len(ignore_dirs), max_files,
    )

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------
    summary    = ScanSummary(root_directory=root_path)
    files:     list[ScannedFile] = []
    seen_real: set[Path]         = set()
    t_start    = time.perf_counter()

    # ------------------------------------------------------------------
    # Walk
    # ------------------------------------------------------------------
    for entry in _walk(
        root_path, ignore_dirs, follow_symlinks,
        include_hidden, summary, depth=0,
    ):
        summary.total_visited += 1

        # ---- Circular / duplicate symlink guard ----------------------
        resolved = _resolve_safe(entry)
        if resolved is None:
            summary.record_skip(entry, "unresolvable_symlink")
            continue
        if resolved in seen_real:
            summary.record_expected_skip(
                entry, "circular_symlink_or_duplicate",
            )
            continue
        seen_real.add(resolved)

        # ---- Extension filter ----------------------------------------
        ext = entry.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            summary.record_expected_skip(
                entry, f"unsupported_extension:{ext or 'none'}",
            )
            continue

        # ---- Single stat call for all file checks --------------------
        ok, reason, size_bytes, modified_ts = _check_file(
            entry, max_file_size_bytes,
        )
        if not ok:
            summary.record_skip(entry, reason)
            continue

        # ---- Accept --------------------------------------------------
        scanned = ScannedFile(
            absolute_path   = resolved,
            relative_path   = entry.relative_to(root_path),
            extension       = ext,
            format_category = SUPPORTED_EXTENSIONS[ext],
            size_bytes      = size_bytes,
            modified_ts     = modified_ts,
        )
        files.append(scanned)
        summary.total_accepted += 1

        logger.debug(
            "[SCANNER] Accepted  %-55s  category=%-14s  size=%d B",
            scanned.relative_path,
            scanned.format_category,
            scanned.size_bytes,
        )

        # ---- Max files cap -------------------------------------------
        if max_files > 0 and summary.total_accepted >= max_files:
            logger.info(
                "[SCANNER] Reached max_files=%d — stopping scan",
                max_files,
            )
            break

    # ------------------------------------------------------------------
    # Finalise summary
    # ------------------------------------------------------------------
    summary.elapsed_seconds = time.perf_counter() - t_start

    logger.info(
        "[SCANNER] Scan complete  visited=%d  accepted=%d  "
        "skipped=%d  elapsed=%.3fs",
        summary.total_visited,
        summary.total_accepted,
        summary.total_skipped,
        summary.elapsed_seconds,
    )

    return ScanResult(files=files, summary=summary)


def iter_scan_directory(
    root: str | Path,
    *,
    extra_ignore_dirs: Optional[set[str]] = None,
    max_file_size_mb: float = 500.0,
    max_files: int = 0,
    follow_symlinks: bool = False,
    include_hidden: bool = False,
) -> Generator[ScannedFile, None, None]:
    """
    Memory-efficient generator variant of directory scanning.

    Yields ScannedFile objects one at a time as they are discovered
    during traversal — suitable for very large directory trees where
    holding all results in memory is undesirable.

    Unlike scan_directory(), this does NOT build a complete list first.
    Each file is yielded immediately after acceptance.

    Parameters
    ----------
    root : str | Path
        The root directory to scan.

    extra_ignore_dirs : set[str] | None
        Additional directory names to ignore.

    max_file_size_mb : float
        Skip files larger than this (MB). Default 500.

    max_files : int
        Stop after yielding this many files. 0 = unlimited.

    follow_symlinks : bool
        Follow symlinks to directories. Default False.

    include_hidden : bool
        Scan dot-prefixed directories. Default False.

    Yields
    ------
    ScannedFile
        One per accepted file, in traversal order.

    Examples
    --------
    >>> for scanned_file in iter_scan_directory("./corpus"):
    ...     process(scanned_file)
    """
    root_path = Path(root).expanduser()

    # Pre-flight validation (same as scan_directory)
    if not root_path.exists():
        raise FileNotFoundError(
            f"[IsoCortex] Root directory does not exist: '{root_path}'"
        )
    if not root_path.is_dir():
        raise NotADirectoryError(
            f"[IsoCortex] Path is not a directory: '{root_path}'"
        )
    if not os.access(root_path, os.R_OK):
        raise PermissionError(
            f"[IsoCortex] Cannot read root directory: '{root_path}'"
        )

    root_path = root_path.resolve()

    # Build ignore set
    ignore_dirs: frozenset[str] = DEFAULT_IGNORE_DIRS
    if extra_ignore_dirs:
        ignore_dirs = ignore_dirs | frozenset(extra_ignore_dirs)

    max_file_size_bytes = int(max_file_size_mb * 1024 * 1024)

    logger.info(
        "[SCANNER] Starting iter scan  root=%s  max_size=%.1f MB  "
        "follow_symlinks=%s  include_hidden=%s",
        root_path, max_file_size_mb, follow_symlinks, include_hidden,
    )

    # Lightweight summary for logging only (not returned)
    seen_real: set[Path] = set()
    accepted_count: int  = 0
    t_start              = time.perf_counter()

    for entry in _walk(
        root_path, ignore_dirs, follow_symlinks,
        include_hidden, _NullSummary(), depth=0,
    ):
        # Circular / duplicate symlink guard
        resolved = _resolve_safe(entry)
        if resolved is None:
            logger.debug("[SCANNER] Iter skip  %s  unresolvable", entry.name)
            continue
        if resolved in seen_real:
            logger.debug("[SCANNER] Iter skip  %s  duplicate", entry.name)
            continue
        seen_real.add(resolved)

        # Extension filter
        ext = entry.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            continue

        # File checks
        ok, reason, size_bytes, modified_ts = _check_file(
            entry, max_file_size_bytes,
        )
        if not ok:
            logger.debug("[SCANNER] Iter skip  %s  %s", entry.name, reason)
            continue

        scanned = ScannedFile(
            absolute_path   = resolved,
            relative_path   = entry.relative_to(root_path),
            extension       = ext,
            format_category = SUPPORTED_EXTENSIONS[ext],
            size_bytes      = size_bytes,
            modified_ts     = modified_ts,
        )
        accepted_count += 1
        yield scanned

        if max_files > 0 and accepted_count >= max_files:
            logger.info(
                "[SCANNER] Iter reached max_files=%d — stopping",
                max_files,
            )
            break

    elapsed = time.perf_counter() - t_start
    logger.info(
        "[SCANNER] Iter scan complete  yielded=%d  elapsed=%.3fs",
        accepted_count, elapsed,
    )


# ---------------------------------------------------------------------------
# Null summary for iter_scan_directory (avoids storing skip details)
# ---------------------------------------------------------------------------

class _NullSummary:
    """Drop-in replacement for ScanSummary that discards skip records."""

    def record_skip(self, path: Path, reason: str) -> None:
        logger.debug("[SCANNER] Skip  %s  %s", path, reason)

    def record_expected_skip(self, path: Path, reason: str) -> None:
        logger.debug("[SCANNER] Skip (expected)  %s  %s", path, reason)



# ---------------------------------------------------------------------------
# Internal walk helper
# ---------------------------------------------------------------------------

def _walk(
    root:            Path,
    ignore_dirs:     frozenset[str],
    follow_symlinks: bool,
    include_hidden:  bool,
    summary:         _SummaryProtocol,
    depth:           int,
) -> Generator[Path, None, None]:
    """
    Yield every filesystem file entry under *root*, pruning ignored dirs.

    Uses os.scandir for maximum performance — one syscall per directory
    vs. multiple calls in pathlib.rglob.

    Skipped directories are recorded in *summary* but never raise.

    Parameters
    ----------
    depth : int
        Current recursion depth. Stops at MAX_RECURSION_DEPTH.
    """
    if depth > MAX_RECURSION_DEPTH:
        summary.record_skip(root, f"max_depth_exceeded:{depth}")
        return

    try:
        entries = list(os.scandir(root))
    except PermissionError:
        summary.record_skip(root, "directory_permission_denied")
        return
    except OSError as exc:
        summary.record_skip(root, f"directory_os_error:{exc.strerror}")
        return

    for entry in entries:
        entry_path = Path(entry.path)

        # ---- Symlink handling (checked FIRST) ------------------------
        try:
            is_symlink = entry.is_symlink()
        except OSError:
            summary.record_skip(entry_path, "stat_error")
            continue

        if is_symlink:
            if not follow_symlinks:
                summary.record_expected_skip(
                    entry_path, "symlink_not_followed",
                )
                continue

            # follow_symlinks=True: resolve target type
            try:
                target_is_dir = entry.is_dir(follow_symlinks=True)
            except OSError:
                summary.record_skip(
                    entry_path, "broken_symlink",
                )
                continue

            if target_is_dir:
                dir_name = entry.name

                if not include_hidden and dir_name.startswith("."):
                    summary.record_expected_skip(
                        entry_path, "hidden_directory",
                    )
                    continue

                if dir_name in ignore_dirs:
                    summary.record_expected_skip(
                        entry_path, f"ignored_dir:{dir_name}",
                    )
                    continue

                yield from _walk(
                    entry_path, ignore_dirs, follow_symlinks,
                    include_hidden, summary, depth + 1,
                )
                continue

            # Symlink target is a file — yield it, extension check
            # happens in the caller
            yield entry_path
            continue

        # ---- Non-symlink handling ------------------------------------
        try:
            is_dir  = entry.is_dir()
            is_file = entry.is_file()
        except OSError:
            summary.record_skip(entry_path, "stat_error")
            continue

        if is_dir:
            dir_name = entry.name

            if not include_hidden and dir_name.startswith("."):
                summary.record_expected_skip(
                    entry_path, "hidden_directory",
                )
                continue

            if dir_name in ignore_dirs:
                summary.record_expected_skip(
                    entry_path, f"ignored_dir:{dir_name}",
                )
                continue

            # Recurse
            yield from _walk(
                entry_path, ignore_dirs, follow_symlinks,
                include_hidden, summary, depth + 1,
            )

        elif is_file:
            yield entry_path

        else:
            # Special file: socket, device, FIFO, or unexpected type
            summary.record_expected_skip(
                entry_path, "special_file",
            )


# ---------------------------------------------------------------------------
# Utility helpers — used by extractor.py and main.py
# ---------------------------------------------------------------------------

def get_supported_extensions() -> list[str]:
    """Return a sorted list of all supported file extensions."""
    return sorted(SUPPORTED_EXTENSIONS.keys())


def get_format_category(path: str | Path) -> Optional[str]:
    """
    Return the format category for *path* based on its extension.
    Returns None if the extension is not supported.

    Examples
    --------
    >>> get_format_category("report.pdf")
    'pdf'
    >>> get_format_category("data.parquet")
    None
    """
    ext = Path(path).suffix.lower()
    return SUPPORTED_EXTENSIONS.get(ext)


def is_supported(path: str | Path) -> bool:
    """Return True if the file extension is in the supported registry."""
    return Path(path).suffix.lower() in SUPPORTED_EXTENSIONS


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
    import json as json_mod

    parser = argparse.ArgumentParser(
        prog        = "scanner",
        description = "IsoCortex directory scanner — standalone debug mode.",
    )
    parser.add_argument(
        "root",
        help = "Root directory to scan.",
    )
    parser.add_argument(
        "--max-size",
        type    = float,
        default = 500.0,
        metavar = "MB",
        help    = "Skip files larger than this many megabytes (default: 500).",
    )
    parser.add_argument(
        "--max-files",
        type    = int,
        default = 0,
        metavar = "N",
        help    = "Stop after accepting N files. 0 = unlimited (default).",
    )
    parser.add_argument(
        "--follow-symlinks",
        action = "store_true",
        help   = "Follow symbolic links to directories.",
    )
    parser.add_argument(
        "--include-hidden",
        action = "store_true",
        help   = "Scan directories that start with '.'",
    )
    parser.add_argument(
        "--ignore",
        nargs   = "*",
        metavar = "DIR",
        default = [],
        help    = "Additional directory names to ignore.",
    )
    parser.add_argument(
        "--iter",
        action = "store_true",
        dest   = "use_iter",
        help   = "Use memory-efficient streaming mode.",
    )
    parser.add_argument(
        "--json",
        action = "store_true",
        help   = "Output accepted files as a JSON array.",
    )
    parser.add_argument(
        "-v", "--verbose",
        action = "store_true",
        help   = "Enable DEBUG-level logging.",
    )
    args = parser.parse_args()

    _configure_logging(args.verbose)

    extra_ignore = set(args.ignore) if args.ignore else None

    try:
        if args.use_iter:
            # Streaming mode — true memory efficiency
            files: list[ScannedFile] = []
            t_start = time.perf_counter()
            for scanned in iter_scan_directory(
                args.root,
                extra_ignore_dirs = extra_ignore,
                max_file_size_mb  = args.max_size,
                max_files         = args.max_files,
                follow_symlinks   = args.follow_symlinks,
                include_hidden    = args.include_hidden,
            ):
                files.append(scanned)
            elapsed = time.perf_counter() - t_start
            # Build a minimal summary for display
            result = ScanResult(
                files   = files,
                summary = ScanSummary(
                    root_directory  = Path(args.root).resolve(),
                    total_accepted  = len(files),
                    elapsed_seconds = elapsed,
                ),
            )
        else:
            result = scan_directory(
                args.root,
                extra_ignore_dirs = extra_ignore,
                max_file_size_mb  = args.max_size,
                max_files         = args.max_files,
                follow_symlinks   = args.follow_symlinks,
                include_hidden    = args.include_hidden,
            )
    except (FileNotFoundError, NotADirectoryError, PermissionError) as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)

    if args.json:
        output = [
            {
                "absolute_path":   str(f.absolute_path),
                "relative_path":   str(f.relative_path),
                "extension":       f.extension,
                "format_category": f.format_category,
                "size_bytes":      f.size_bytes,
                "modified_ts":     f.modified_ts,
            }
            for f in result.files
        ]
        print(json_mod.dumps(output, indent=2))
    else:
        for f in result.files:
            print(f"  [{f.format_category:<14}]  {f.relative_path}")

        print()
        print("=" * 60)
        print(f"  Root      : {result.summary.root_directory}")
        print(f"  Visited   : {result.summary.total_visited}")
        print(f"  Accepted  : {result.summary.total_accepted}")
        print(f"  Skipped   : {result.summary.total_skipped}")
        print(f"  Elapsed   : {result.summary.elapsed_seconds:.3f}s")
        print("=" * 60)
