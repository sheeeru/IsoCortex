"""
IsoCortex Desktop App — Folder Watcher
=======================================
Monitors specified folders for file-system changes and auto-ingests
new/modified files into the configured index via the engine pipeline.

Uses `watchdog` for cross-platform file-system events and runs in a
background daemon thread with a 2-second debounce to coalesce rapid
events (e.g. multi-file copies, editor saves).
"""

from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileCreatedEvent, FileModifiedEvent, FileDeletedEvent

if TYPE_CHECKING:
    from desktop_app.engine import IsoCortexEngine

logger = logging.getLogger("IsoCortex.watcher")

# Re-export the supported-extensions set so the watcher module is
# self-contained without duplicating the list.
try:
    from desktop_app.engine import SUPPORTED_EXTENSIONS
except ImportError:
    SUPPORTED_EXTENSIONS = {
        ".txt", ".md", ".log", ".rst", ".cfg", ".ini", ".py", ".cpp", ".c",
        ".h", ".js", ".ts", ".java", ".go", ".rs", ".rb", ".pdf", ".docx",
        ".odt", ".pptx", ".odp", ".xlsx", ".xls", ".ods", ".csv", ".json",
        ".tsv", ".eml", ".html", ".htm",
    }

# Files matching these patterns are silently ignored.
_IGNORE_PREFIXES = (".", "~")
_IGNORE_SUFFIXES = (".tmp",)


def _should_ignore(filename: str) -> bool:
    """Return *True* if *filename* is a temporary / hidden file."""
    if not filename:
        return True
    base = os.path.basename(filename)
    if any(base.startswith(p) for p in _IGNORE_PREFIXES):
        return True
    if any(base.endswith(s) for s in _IGNORE_SUFFIXES):
        return True
    return False


def _is_supported(filename: str) -> bool:
    """Return *True* if *filename* has a supported extension."""
    return Path(filename).suffix.lower() in SUPPORTED_EXTENSIONS


class _DebouncedHandler(FileSystemEventHandler):
    """Internal event handler that debounces file-system events.

    Events are collected for *debounce_seconds* after the last event
    before the batch is flushed to the engine for ingestion.
    """

    def __init__(self, engine: IsoCortexEngine, debounce_seconds: float = 2.0) -> None:
        super().__init__()
        self._engine = engine
        self._debounce_seconds = debounce_seconds
        self._lock = threading.Lock()
        self._pending: dict[str, str] = {}  # path -> index_name
        self._timer: threading.Timer | None = None
        # Mapping from watched folder path -> index_name.
        # Populated by FolderWatcher.add_watch() so that _enqueue can
        # route new files to the correct index.
        self._folder_index_map: dict[str, str] = {}

    # -- watchdog callbacks ------------------------------------------------

    def on_created(self, event) -> None:
        if event.is_directory:
            return
        self._enqueue(event.src_path)

    def on_modified(self, event) -> None:
        if event.is_directory:
            return
        self._enqueue(event.src_path)

    def on_deleted(self, event) -> None:
        if event.is_directory:
            return
        if _should_ignore(event.src_path) or not _is_supported(event.src_path):
            return
        logger.info("File deleted (noted): %s", event.src_path)

    # -- internal ----------------------------------------------------------

    def _enqueue(self, path: str) -> None:
        """Add *path* to the pending set and (re)schedule the debounce timer."""
        if _should_ignore(path) or not _is_supported(path):
            return

        # Check engine exclusion patterns
        try:
            if self._engine.is_excluded(path):
                return
        except Exception:
            pass

        # Determine which index this path belongs to by matching the
        # file path against the watched folder prefixes.
        index_name = "default"
        abs_path = os.path.abspath(path)
        for folder, idx in self._folder_index_map.items():
            if abs_path.startswith(folder + os.sep) or abs_path == folder:
                index_name = idx
                break
        with self._lock:
            self._pending[path] = index_name
        self._schedule_flush()

    def _schedule_flush(self) -> None:
        """(Re)schedule the debounce timer."""
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
            self._timer = threading.Timer(self._debounce_seconds, self._flush)
            self._timer.daemon = True
            self._timer.start()

    def _flush(self) -> None:
        """Ingest all pending files via the engine."""
        with self._lock:
            files = dict(self._pending)
            self._pending.clear()
            self._timer = None

        if not files:
            return

        # Group by index_name so we can batch per-index.
        by_index: dict[str, list[str]] = {}
        for fpath, idx in files.items():
            by_index.setdefault(idx, []).append(fpath)

        for index_name, paths in by_index.items():
            logger.info(
                "Auto-ingesting %d file(s) into index %r",
                len(paths), index_name,
            )
            try:
                self._engine.ingest_files(index_name, paths)
                logger.info("Auto-ingest complete for %d file(s)", len(paths))
            except Exception:
                logger.exception(
                    "Auto-ingest failed for index %r", index_name,
                )


class FolderWatcher:
    """High-level folder watcher that wraps a ``watchdog`` observer.

    Typical usage::

        watcher = FolderWatcher(engine)
        watcher.add_watch("/path/to/folder", "default")
        watcher.start()
        # ... later ...
        watcher.stop()
    """

    def __init__(
        self,
        engine: IsoCortexEngine,
        debounce_seconds: float = 2.0,
    ) -> None:
        self._engine = engine
        self._debounce_seconds = debounce_seconds
        self._handler = _DebouncedHandler(engine, debounce_seconds)
        self._observer = Observer()
        self._watched: dict[str, str] = {}  # folder_path -> index_name
        self._running = False

    # -- public API --------------------------------------------------------

    def start(self) -> None:
        """Start the background observer thread."""
        if self._running:
            logger.warning("FolderWatcher is already running")
            return
        self._observer.start()
        self._running = True
        logger.info("FolderWatcher started")

    def stop(self) -> None:
        """Stop the observer and wait for the thread to finish."""
        if not self._running:
            return
        self._observer.stop()
        self._observer.join(timeout=5)
        self._running = False
        logger.info("FolderWatcher stopped")

    def add_watch(self, folder_path: str, index_name: str = "default") -> None:
        """Begin watching *folder_path* and auto-ingest into *index_name*."""
        folder_path = os.path.abspath(folder_path)
        if not os.path.isdir(folder_path):
            raise ValueError(f"Not a directory: {folder_path}")
        if folder_path in self._watched:
            logger.debug("Already watching %s", folder_path)
            return
        self._observer.schedule(
            self._handler,
            folder_path,
            recursive=True,
        )
        self._watched[folder_path] = index_name
        self._handler._folder_index_map[folder_path] = index_name
        logger.info("Now watching %s (index: %s)", folder_path, index_name)

    def remove_watch(self, folder_path: str) -> None:
        """Stop watching *folder_path*."""
        folder_path = os.path.abspath(folder_path)
        if folder_path not in self._watched:
            logger.debug("Not watching %s — nothing to remove", folder_path)
            return
        # watchdog does not expose an unschedule-by-path that matches our
        # use-case perfectly, so we iterate the internal watches.
        for watch in list(self._observer._watches.values()):
            if watch.path == folder_path:
                self._observer.unschedule(watch)
                break
        del self._watched[folder_path]
        self._handler._folder_index_map.pop(folder_path, None)
        logger.info("Stopped watching %s", folder_path)

    def get_watched_folders(self) -> list[dict]:
        """Return a list of dicts describing each watched folder."""
        return [
            {"folder_path": path, "index_name": idx}
            for path, idx in self._watched.items()
        ]

    @property
    def is_running(self) -> bool:
        """Whether the observer thread is currently active."""
        return self._running