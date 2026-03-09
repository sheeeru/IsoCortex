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
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator, Optional

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


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
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
    root_directory  : Path             — The directory that was scanned.
    total_visited   : int              — Every filesystem node visited.
    total_accepted  : int              — Files with a supported extension.
    total_skipped   : int              — Files/dirs skipped for any reason.
    skipped_details : list             — Per-skip reason records.
    elapsed_seconds : float            — Wall-clock time for the scan.
    """
    root_directory:  Path
    total_visited:   int                    = 0
    total_accepted:  int                    = 0
    total_skipped:   int                    = 0
    skipped_details: list[dict[str, str]]   = field(default_factory=list)
    elapsed_seconds: float                  = 0.0

    def record_skip(self, path: Path, reason: str) -> None:
        """Increment skip counter and store the human-readable reason."""
        self.total_skipped += 1
        self.skipped_details.append({"path": str(path), "reason": reason})
        logger.warning(
            "[SCANNER] Skipped %-60s  reason=%s", path, reason
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
) -> tuple[bool, str, int, float]:
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

    if st.st_size == 0:
        return False, "empty_file", 0, 0.0

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
    follow_symlinks: bool = False,
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

    follow_symlinks : bool
        If True, symbolic links to directories are followed.
        Circular links are always detected and skipped regardless.

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
        "follow_symlinks=%s  ignore_dirs=%d",
        root_path, max_file_size_mb, follow_symlinks, len(ignore_dirs),
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
    for entry in _walk(root_path, ignore_dirs, follow_symlinks, summary):

        summary.total_visited += 1

        # ---- Circular / duplicate symlink guard ----------------------
        resolved = _resolve_safe(entry)
        if resolved is None:
            summary.record_skip(entry, "unresolvable_symlink")
            continue
        if resolved in seen_real:
            summary.record_skip(entry, "circular_symlink_or_duplicate")
            continue
        seen_real.add(resolved)

        # ---- Extension filter ----------------------------------------
        ext = entry.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            summary.record_skip(
                entry,
                f"unsupported_extension:{ext or 'none'}",
            )
            continue

        # ---- Single stat call for all file checks --------------------
        ok, reason, size_bytes, modified_ts = _check_file(
            entry, max_file_size_bytes
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
    follow_symlinks: bool = False,
) -> Generator[ScannedFile, None, None]:
    """
    Memory-efficient generator variant of :func:`scan_directory`.

    Yields ScannedFile objects one at a time — suitable for very large
    directory trees where holding all results in memory is undesirable.

    Parameters and semantics are identical to :func:`scan_directory`.

    Examples
    --------
    >>> for scanned_file in iter_scan_directory("./corpus"):
    ...     process(scanned_file)
    """
    result = scan_directory(
        root,
        extra_ignore_dirs = extra_ignore_dirs,
        max_file_size_mb  = max_file_size_mb,
        follow_symlinks   = follow_symlinks,
    )
    yield from result.files


# ---------------------------------------------------------------------------
# Internal walk helper
# ---------------------------------------------------------------------------

def _walk(
    root:            Path,
    ignore_dirs:     frozenset[str],
    follow_symlinks: bool,
    summary:         ScanSummary,
) -> Generator[Path, None, None]:
    """
    Yield every filesystem file entry under *root*, pruning ignored dirs.

    Uses os.scandir for maximum performance — one syscall per directory
    vs. multiple calls in pathlib.rglob.

    Skipped directories are recorded in *summary* but never raise.
    """
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

        try:
            is_dir  = entry.is_dir(follow_symlinks=follow_symlinks)
            is_file = entry.is_file(follow_symlinks=follow_symlinks)
        except OSError:
            summary.record_skip(entry_path, "stat_error")
            continue

        if is_dir:
            dir_name = entry.name

            # FIX: removed redundant `and dir_name not in {"."}` check
            # os.scandir never yields "." or ".." entries
            if dir_name.startswith("."):
                summary.record_skip(entry_path, "hidden_directory")
                continue

            if dir_name in ignore_dirs:
                summary.record_skip(
                    entry_path, f"ignored_dir:{dir_name}"
                )
                continue

            # Recurse
            yield from _walk(
                entry_path, ignore_dirs, follow_symlinks, summary
            )

        elif is_file:
            yield entry_path

        else:
            # Broken symlink or special file (socket, device, FIFO)
            summary.record_skip(entry_path, "not_file_or_dir")


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
    import json
    import sys

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
        "--follow-symlinks",
        action = "store_true",
        help   = "Follow symbolic links to directories.",
    )
    parser.add_argument(
        "--ignore",
        nargs   = "*",
        metavar = "DIR",
        default = [],
        help    = "Additional directory names to ignore.",
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

    try:
        result = scan_directory(
            args.root,
            extra_ignore_dirs = set(args.ignore) if args.ignore else None,
            max_file_size_mb  = args.max_size,
            follow_symlinks   = args.follow_symlinks,
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
        print(json.dumps(output, indent=2))
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