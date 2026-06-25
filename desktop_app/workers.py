"""
IsoCortex Desktop App — Background Workers
==========================================
Thread workers for heavy operations (indexing, searching, model loading,
LLM generation) so the GUI remains responsive.
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
                on_success = self._on_success
                result = self._result
                self._app_ref.after(0, lambda: on_success(result))
        except Exception as exc:
            self._error = exc
            captured_exc = exc  # Python 3.14 deletes 'exc' after except block
            error_msg = str(captured_exc)
            logger.error("Worker %s failed: %s", self.name, captured_exc, exc_info=True)
            if self._on_error and self._app_ref:
                on_error = self._on_error
                # Always pass str to error callbacks — never raw Exception
                self._app_ref.after(0, lambda: on_error(error_msg))
        finally:
            if self._on_complete and self._app_ref:
                on_complete = self._on_complete
                self._app_ref.after(0, on_complete)


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
            on_progress = self._on_progress
            self._app_ref.after(
                0,
                lambda: on_progress(processed, total, filename),
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


class LLMWorker(threading.Thread):
    """
    Background worker that streams LLM tokens and reports each token
    back to the GUI via after() callbacks.

    Unlike WorkerThread, this uses a streaming approach — each token
    is sent individually to the GUI as it's generated.
    """

    def __init__(
        self,
        llm,  # LLM instance from llm.py
        messages: list[dict],
        on_token: Optional[Callable] = None,
        on_complete: Optional[Callable] = None,
        on_error: Optional[Callable] = None,
        name: str = "LLMWorker",
    ):
        super().__init__(daemon=True, name=name)
        self._llm = llm
        self._messages = messages
        self._on_token = on_token
        self._on_complete = on_complete
        self._on_error = on_error
        self._app_ref = None
        self._stop_event = threading.Event()
        self._full_response = ""
        self._token_count = 0
        self._start_time = 0.0
        self._elapsed = 0.0

    def set_app_ref(self, app):
        self._app_ref = app

    def stop(self):
        """Signal the worker to stop generating."""
        self._stop_event.set()

    @property
    def is_generating(self) -> bool:
        return self.is_alive() and not self._stop_event.is_set()

    @property
    def tokens_per_second(self) -> float:
        if self._elapsed <= 0:
            return 0.0
        return self._token_count / self._elapsed

    def run(self):
        """Stream tokens from the LLM and report each to the GUI."""
        import time
        self._start_time = time.perf_counter()
        self._full_response = ""
        self._token_count = 0

        try:
            for token_text in self._llm.stream_chat(self._messages):
                if self._stop_event.is_set():
                    break

                self._full_response += token_text
                self._token_count += 1
                self._elapsed = time.perf_counter() - self._start_time

                if self._on_token and self._app_ref:
                    on_token = self._on_token
                    self._app_ref.after(
                        0,
                        lambda t=token_text, tc=self._token_count, el=self._elapsed:
                            on_token(t, tc, el),
                    )

            # Generation complete
            self._elapsed = time.perf_counter() - self._start_time
            result = {
                "response": self._full_response,
                "token_count": self._token_count,
                "elapsed": self._elapsed,
                "tokens_per_second": self._token_count / self._elapsed if self._elapsed > 0 else 0,
            }

            if self._on_complete and self._app_ref:
                on_complete = self._on_complete
                self._app_ref.after(0, lambda: on_complete(result))

        except Exception as exc:
            captured_exc = exc  # Python 3.14: capture before lambda
            logger.error("LLMWorker failed: %s", captured_exc, exc_info=True)
            if self._on_error and self._app_ref:
                on_error = self._on_error
                self._app_ref.after(0, lambda: on_error(str(captured_exc)))


class RAGWorker(threading.Thread):
    """
    Two-phase worker:
      Phase 1: Retrieve RAG context from engine (blocking, background)
      Phase 2: Stream LLM tokens (same thread, sequential)

    This keeps the GUI responsive during both retrieval and generation.
    """

    def __init__(
        self,
        engine,
        llm,  # LLM instance
        query: str,
        conversation_messages: list[dict],
        system_prompt: str,
        top_k: int = 5,
        index_name: Optional[str] = None,
        on_context_ready: Optional[Callable] = None,
        on_token: Optional[Callable] = None,
        on_complete: Optional[Callable] = None,
        on_error: Optional[Callable] = None,
        name: str = "RAGWorker",
    ):
        super().__init__(daemon=True, name=name)
        self._engine = engine
        self._llm = llm
        self._query = query
        self._conversation_messages = conversation_messages
        self._system_prompt = system_prompt
        self._top_k = top_k
        self._index_name = index_name
        self._on_context_ready = on_context_ready
        self._on_token = on_token
        self._on_complete = on_complete
        self._on_error = on_error
        self._app_ref = None
        self._stop_event = threading.Event()
        self._full_response = ""
        self._token_count = 0
        self._sources = []
        self._start_time = 0.0
        self._elapsed = 0.0

    def set_app_ref(self, app):
        self._app_ref = app

    def stop(self):
        self._stop_event.set()

    @property
    def is_generating(self) -> bool:
        return self.is_alive() and not self._stop_event.is_set()

    def run(self):
        import time
        self._start_time = time.perf_counter()

        try:
            # ── Phase 1: RAG retrieval ──────────────────────────────
            if self._index_name:
                context_parts = []
                source_parts = []
                for src in self._engine.search(self._index_name, self._query, top_k=self._top_k):
                    sp = {
                        "index": len(source_parts) + 1,
                        "file": src.source_file or "unknown",
                        "score": round(src.score, 3),
                        "keyword_score": 0,
                        "page": src.page_number or 0,
                        "format": src.format_category or "",
                        "full_text": src.full_text or src.text or "",
                        "chunk_text": src.full_text or src.text or "",
                    }
                    source_parts.append(sp)
                doc_context = "\n\n---\n\n".join(
                    f"[Source {s['index']}] ({s['file']}" + (f", page {s['page']}" if s['page'] else "") + f"):\n{s['full_text']}" for s in source_parts
                ) if source_parts else ""
                # Always include library stats so AI can answer meta questions
                stats = self._engine._build_stats_summary() or ""
                context_str = stats + ("\n\n" + doc_context if doc_context else "")
                sources = source_parts
            else:
                context_str, sources = self._engine.build_rag_context(self._query, top_k=self._top_k)
            self._sources = sources

            # Notify GUI that context is ready (sources can be shown)
            if self._on_context_ready and self._app_ref:
                on_context_ready = self._on_context_ready
                self._app_ref.after(0, lambda: on_context_ready(sources))

            if self._stop_event.is_set():
                result = {
                    "response": self._full_response,
                    "token_count": self._token_count,
                    "elapsed": time.perf_counter() - self._start_time,
                    "tokens_per_second": 0,
                    "sources": self._sources,
                    "cancelled": True,
                }
                if self._on_complete and self._app_ref:
                    try:
                        on_complete = self._on_complete
                        self._app_ref.after(0, lambda: on_complete(result))
                    except Exception:
                        pass
                return

            # ── Phase 2: Build messages and stream LLM ──────────────
            messages = [{"role": "system", "content": self._system_prompt}]

            # Add RAG context as a system-level knowledge injection
            if context_str:
                messages.append({
                    "role": "system",
                    "content": (
                        "DOCUMENT CONTEXT:\n"
                        + context_str
                        + "\n\nINSTRUCTIONS:\n"
                        "- Answer the user's question using the context above.\n"
                        "- Reference sources as [Source N].\n"
                        "- For library stats, use [Library Stats] directly.\n"
                        "- If context is insufficient, say so, then answer from "
                        "general knowledge labeled [General Knowledge]."
                    ),
                })

            # Add conversation history
            for msg in self._conversation_messages:
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"],
                })

            # Add current user query
            messages.append({"role": "user", "content": self._query})

            # Stream tokens
            self._full_response = ""
            self._token_count = 0

            for token_text in self._llm.stream_chat(messages):
                if self._stop_event.is_set():
                    break

                self._full_response += token_text
                self._token_count += 1
                self._elapsed = time.perf_counter() - self._start_time

                # ── Repetition loop detector ──
                # After 30+ tokens, check if the last ~80 chars appear
                # twice in the trailing 200 chars — sign of a degenerate loop.
                _repetition_detected = False
                if self._token_count > 30 and len(self._full_response) > 200:
                    tail = self._full_response[-200:]
                    for seg_len in (80, 60, 40):
                        if len(tail) >= seg_len * 2:
                            seg = tail[-seg_len:]
                            before = tail[-(seg_len * 2):-seg_len]
                            if seg == before:
                                logger.warning(
                                    "Repetition loop detected at token %d — stopping",
                                    self._token_count,
                                )
                                _repetition_detected = True
                                break

                if self._on_token and self._app_ref:
                    on_token = self._on_token
                    self._app_ref.after(
                        0,
                        lambda t=token_text, tc=self._token_count, el=self._elapsed:
                            on_token(t, tc, el),
                    )

                if _repetition_detected:
                    break

            # Done
            self._elapsed = time.perf_counter() - self._start_time
            result = {
                "response": self._full_response,
                "token_count": self._token_count,
                "elapsed": self._elapsed,
                "tokens_per_second": self._token_count / self._elapsed if self._elapsed > 0 else 0,
                "sources": self._sources,
            }

            if self._on_complete and self._app_ref:
                on_complete = self._on_complete
                self._app_ref.after(0, lambda: on_complete(result))

        except Exception as exc:
            captured_exc = exc  # Python 3.14: capture before lambda
            logger.error("RAGWorker failed: %s", captured_exc, exc_info=True)
            if self._on_error and self._app_ref:
                on_error = self._on_error
                self._app_ref.after(0, lambda: on_error(str(captured_exc)))
