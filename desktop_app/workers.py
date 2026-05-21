"""
IsoCortex Desktop App — Background Workers
==========================================
Thread workers for heavy operations (indexing, searching, model loading)
so the GUI remains responsive.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable, Optional

logger = logging.getLogger("IsoCortex.workers")


class WorkerThread(threading.Thread):
    """
    A reusable background worker thread.
    Runs a callable in a daemon thread and reports results back
    to the GUI via after() callbacks.
    """

    def __init__(
        self,
        target: Callable,
        on_success: Optional[Callable] = None,
        on_error: Optional[Callable] = None,
        on_complete: Optional[Callable] = None,
        name: str = "WorkerThread",
    ):
        super().__init__(daemon=True, name=name)
        self._target = target
        self._on_success = on_success
        self._on_error = on_error
        self._on_complete = on_complete
        self._result = None
        self._error = None
        self._app_ref = None  # Set to the CTk instance for after() calls

    def set_app_ref(self, app):
        """Set the CTk app reference for scheduling GUI callbacks."""
        self._app_ref = app

    def run(self):
        """Execute the target function and schedule callbacks."""
        try:
            self._result = self._target()
            if self._on_success and self._app_ref:
                self._app_ref.after(0, lambda: self._on_success(self._result))
        except Exception as exc:
            self._error = exc
            logger.error("Worker %s failed: %s", self.name, exc, exc_info=True)
            if self._on_error and self._app_ref:
                self._app_ref.after(0, lambda: self._on_error(exc))
        finally:
            if self._on_complete and self._app_ref:
                self._app_ref.after(0, self._on_complete)


class ModelLoader(WorkerThread):
    """Background worker that pre-loads the embedding model."""

    def __init__(self, engine, on_done: Optional[Callable] = None):
        super().__init__(
            target=self._load_model,
            on_success=on_done,
            name="ModelLoader",
        )
        self._engine = engine

    def _load_model(self) -> dict:
        """Load the model and return status."""
        success = self._engine.ensure_model()
        return self._engine.get_model_status()


class IngestionWorker(WorkerThread):
    """Background worker for file ingestion pipeline."""

    def __init__(
        self,
        engine,
        index_name: str,
        file_paths: list[str],
        on_progress: Optional[Callable] = None,
        on_done: Optional[Callable] = None,
        on_error: Optional[Callable] = None,
    ):
        super().__init__(
            target=self._run_ingestion,
            on_success=on_done,
            on_error=on_error,
            name="IngestionWorker",
        )
        self._engine = engine
        self._index_name = index_name
        self._file_paths = file_paths
        self._on_progress = on_progress

    def _run_ingestion(self):
        """Run the ingestion pipeline."""
        return self._engine.ingest_files(
            self._index_name,
            self._file_paths,
            progress_callback=self._progress_cb,
        )

    def _progress_cb(self, processed: int, total: int, filename: str):
        """Forward progress to the GUI thread."""
        if self._on_progress and self._app_ref:
            self._app_ref.after(
                0,
                lambda: self._on_progress(processed, total, filename),
            )


class SearchWorker(WorkerThread):
    """Background worker for semantic search."""

    def __init__(
        self,
        engine,
        index_name: str,
        query: str,
        top_k: int = 10,
        on_done: Optional[Callable] = None,
        on_error: Optional[Callable] = None,
    ):
        super().__init__(
            target=self._run_search,
            on_success=on_done,
            on_error=on_error,
            name="SearchWorker",
        )
        self._engine = engine
        self._index_name = index_name
        self._query = query
        self._top_k = top_k
        self._start_time = 0.0

    def _run_search(self):
        """Run the search and return (results, elapsed_seconds)."""
        import time
        self._start_time = time.perf_counter()
        results = self._engine.search(
            self._index_name,
            self._query,
            self._top_k,
        )
        elapsed = time.perf_counter() - self._start_time
        return results, elapsed
