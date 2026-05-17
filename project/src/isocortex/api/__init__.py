"""
IsoCortex — FastAPI Application Factory
=========================================

Creates and configures the FastAPI application with:
  - Middleware pipeline (RequestID, RateLimit, Error handlers)
  - All route modules (auth, indexes, search, documents, jobs, admin)
  - Startup/shutdown lifecycle (DB init, job scheduler, embedding model)
  - CORS, OpenAPI docs, static file serving

SRS References:
  - Section 3: System Architecture (API Layer)
  - Section 4: Concurrency Model
  - Section 5: Functional Requirements (API)
  - NFR-12: Auto-generated OpenAPI 3.0 docs at /docs and /redoc

Author : Shaheer Qureshi
Project: IsoCortex
"""

from __future__ import annotations

import logging
import os
import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from isocortex import __version__

logger = logging.getLogger(__name__)


# =============================================================================
# Application Lifespan
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application startup and shutdown lifecycle.

    Startup:
      1. Initialize configuration
      2. Ensure data directories exist
      3. Initialize SQLite database (WAL mode)
      4. Start job scheduler
      5. Pre-load embedding model (lazy, on first request)

    Shutdown:
      1. Cancel running jobs
      2. Close database connections
      3. Clean up resources
    """
    logger.info("=" * 60)
    logger.info("IsoCortex v%s starting up...", __version__)
    logger.info("=" * 60)

    # --- Startup ---
    from isocortex.config import load_config
    config = load_config()
    app.state.config = config

    # Ensure directories
    config.ensure_directories()

    # Initialize database
    from isocortex.storage import get_database, close_database
    db = get_database()
    db.initialize()
    app.state.db = db

    # Initialize user manager
    from isocortex.auth import get_user_manager
    user_mgr = get_user_manager()
    app.state.user_manager = user_mgr

    # Initialize index manager
    from isocortex.engine.indexing.manager import IndexManager
    index_mgr = IndexManager(config.storage.indices_dir)
    app.state.index_manager = index_mgr

    # Initialize job scheduler
    from isocortex.engine.jobs.scheduler import JobScheduler
    scheduler = JobScheduler(
        db_path=config.storage.db_path,
        max_concurrent=2,
        retention_days=7,
    )
    app.state.job_scheduler = scheduler

    # Register job executors
    _register_job_executors(app)

    # Start cleanup task
    await scheduler.start_cleanup_task()

    # Initialize analytics
    from isocortex.engine.analytics.engine import AnalyticsEngine
    analytics = AnalyticsEngine()
    app.state.analytics = analytics

    logger.info(
        "[STARTUP] Data dir: %s  Port: %d  Model: %s",
        config.storage.data_dir, config.server.port, config.embedding.model_name,
    )
    logger.info("[STARTUP] IsoCortex ready to accept connections")

    yield

    # --- Shutdown ---
    logger.info("[SHUTDOWN] IsoCortex shutting down...")
    await scheduler.shutdown()
    close_database()
    logger.info("[SHUTDOWN] Shutdown complete")


def _get_extension(source_file: str) -> str:
    """Extract file extension from a source file path."""
    return os.path.splitext(source_file)[1].lower()


def _register_job_executors(app: FastAPI) -> None:
    """Register async job executor functions with the scheduler."""
    from isocortex.engine.jobs.scheduler import JobType

    async def index_create_executor(
        job_id: str,
        payload: dict,
        progress_cb,
    ) -> dict:
        """Execute index creation pipeline."""
        name = payload["name"]
        files = payload.get("files", [])
        config_data = payload.get("config", {})

        index_mgr = app.state.index_manager

        # Create empty index
        index_mgr.create_index(
            name=name,
            description=payload.get("description", ""),
            embedding_model=config_data.get("embedding", {}).get(
                "model", "sentence-transformers/all-MiniLM-L6-v2"
            ),
            hnsw_params=config_data.get("hnsw"),
            chunk_config={
                "chunk_size": config_data.get("embedding", {}).get("chunk_size", 512),
                "chunk_overlap": config_data.get("embedding", {}).get("chunk_overlap", 50),
            },
        )

        # Ingest files if provided
        total = len(files)
        processed = 0
        if files:
            from isocortex.engine.ingestion.scanner import scan_directory
            from isocortex.core.extractor.extractor import extract_batch
            from isocortex.engine.ingestion.chunker import chunk_batch, load_tokenizer
            from isocortex.core.embedding.embedder import embed_batch

            # Scan
            await progress_cb(5, "Scanning files...")
            all_files = []
            for fpath in files:
                if os.path.isdir(fpath):
                    result = scan_directory(fpath)
                    all_files.extend(sf.absolute_path for sf in result.files)
                elif os.path.isfile(fpath):
                    all_files.append(fpath)

            # Extract
            await progress_cb(15, f"Extracting text from {len(all_files)} files...")
            extraction_results = extract_batch(all_files)

            # Chunk
            await progress_cb(30, "Chunking documents...")
            chunked = chunk_batch(extraction_results)

            # Embed
            total_chunks = sum(len(cd.chunks) for cd in chunked)
            vector_count = 0

            for i, cd in enumerate(chunked):
                texts = [c.text for c in cd.chunks]
                if not texts:
                    continue

                from isocortex.core.embedding.embedder import embed_batch as do_embed
                result = do_embed(cd)
                if result.vector_matrix is not None:
                    metadata = []
                    for j, chunk in enumerate(cd.chunks):
                        metadata.append({
                            "id": str(uuid.uuid4()),
                            "vector_index": vector_count + j,
                            "deleted": False,
                            "source_file": chunk.source_file,
                            "file_extension": _get_extension(chunk.source_file),
                            "text_preview": chunk.text[:200],
                            "token_count": chunk.token_count,
                        })
                    index_mgr.add_vectors(name, result.vector_matrix, metadata)
                    vector_count += len(metadata)

                processed += 1
                pct = 30 + (60 * processed / max(len(chunked), 1))
                await progress_cb(pct, f"Embedding... ({processed}/{len(chunked)})")

            # Save
            await progress_cb(95, "Saving index to disk...")
            index_mgr.save_index(name)

        stats = index_mgr.get_index(name)
        await progress_cb(100, "Complete")
        return {
            "index_name": name,
            "vector_count": stats.vector_count if stats else 0,
            "file_count": total,
        }

    app.state.job_scheduler.register_job_type("index_create", index_create_executor)

    async def index_delete_executor(job_id, payload, progress_cb) -> dict:
        name = payload["name"]
        index_mgr = app.state.index_manager
        stats = index_mgr.get_index(name)
        size_mb = stats.index_size_mb if stats else 0
        index_mgr.delete_index(name)
        await progress_cb(100, "Index deleted")
        return {"index_name": name, "disk_freed_mb": round(size_mb, 2)}

    app.state.job_scheduler.register_job_type("index_delete", index_delete_executor)

    async def export_executor(job_id, payload, progress_cb) -> dict:
        name = payload["name"]
        output = payload.get("output_path", f"/tmp/{name}.isocortex")
        from pathlib import Path
        await progress_cb(20, "Preparing export...")
        result = app.state.index_manager.export_index(name, Path(output))
        await progress_cb(100, "Export complete")
        return result

    app.state.job_scheduler.register_job_type("export", export_executor)

    async def import_executor(job_id, payload, progress_cb) -> dict:
        archive = payload["archive_path"]
        name = payload.get("name")
        from pathlib import Path
        await progress_cb(20, "Importing archive...")
        index_name = app.state.index_manager.import_index(Path(archive), name)
        await progress_cb(100, "Import complete")
        return {"index_name": index_name}

    app.state.job_scheduler.register_job_type("import", import_executor)

    async def index_update_executor(job_id, payload, progress_cb) -> dict:
        """Execute incremental document ingestion into an existing index.

        SRS Section 8.4: index_update job type — triggered by document uploads.
        """
        name = payload["name"]
        files = payload.get("files", [])

        index_mgr = app.state.index_manager
        total = len(files)
        processed = 0
        vector_count = 0

        if not files:
            await progress_cb(100, "No files to process")
            return {"index_name": name, "vectors_added": 0}

        from isocortex.core.extractor.extractor import extract_batch
        from isocortex.engine.ingestion.chunker import chunk_batch
        from isocortex.core.embedding.embedder import embed_batch as do_embed

        # Extract
        await progress_cb(10, f"Extracting text from {total} files...")
        extraction_results = extract_batch(files)

        # Chunk
        await progress_cb(30, "Chunking documents...")
        chunked = chunk_batch(extraction_results)

        # Embed and add vectors
        total_chunks = sum(len(cd.chunks) for cd in chunked)
        for i, cd in enumerate(chunked):
            texts = [c.text for c in cd.chunks]
            if not texts:
                continue

            result = do_embed(cd)
            if result.vector_matrix is not None:
                metadata = []
                for j, chunk in enumerate(cd.chunks):
                    metadata.append({
                        "id": str(uuid.uuid4()),
                        "vector_index": vector_count + j,
                        "deleted": False,
                        "source_file": getattr(chunk, "source_file", ""),
                        "file_extension": _get_extension(getattr(chunk, "source_file", "")),
                        "text_preview": chunk.text[:200],
                    })
                index_mgr.add_vectors(name, result.vector_matrix, metadata)
                vector_count += len(metadata)

            processed += 1
            pct = 30 + (60 * processed / max(len(chunked), 1))
            await progress_cb(pct, f"Embedding... ({processed}/{len(chunked)})")

        # Save
        await progress_cb(95, "Saving index...")
        index_mgr.save_index(name)

        await progress_cb(100, "Update complete")
        return {"index_name": name, "vectors_added": vector_count, "files_processed": total}

    app.state.job_scheduler.register_job_type("index_update", index_update_executor)

    async def index_compact_executor(job_id, payload, progress_cb) -> dict:
        """Compact an HNSW index by removing soft-deleted vectors.

        SRS FR-IDX-002: Soft delete with tombstone pattern — compaction
        removes tombstoned entries to reclaim memory and disk.
        """
        name = payload["name"]
        index_mgr = app.state.index_manager

        await progress_cb(10, "Loading index...")
        idx_info = index_mgr.get_index(name)

        if idx_info is None:
            await progress_cb(100, f"Index '{name}' not found")
            return {"error": f"Index '{name}' not found"}

        deleted_count = idx_info.deleted_count
        total_count = idx_info.vector_count

        if deleted_count == 0:
            await progress_cb(100, "No deleted vectors to compact")
            return {"index_name": name, "vectors_removed": 0}

        await progress_cb(30, f"Compacting {deleted_count}/{total_count} deleted vectors...")

        # Compact the index (rebuilds without deleted entries)
        index_mgr.compact_index(name)

        await progress_cb(100, "Compaction complete")
        return {
            "index_name": name,
            "vectors_removed": deleted_count,
            "remaining_vectors": total_count - deleted_count,
        }

    app.state.job_scheduler.register_job_type("index_compact", index_compact_executor)


# =============================================================================
# App Factory
# =============================================================================

def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    SRS Section 3.1: Layered architecture — API Layer.
    SRS NFR-12: Auto-generated OpenAPI docs.
    """
    app = FastAPI(
        title="IsoCortex API",
        description=(
            "Production-grade local semantic search engine. "
            "Transform files into vector embeddings, build HNSW indices, "
            "and query with natural language."
        ),
        version=__version__,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
        contact={
            "name": "Shaheer Qureshi",
            "email": "shaheer@isocortex.dev",
        },
        license_info={
            "name": "MIT",
        },
    )

    # --- CORS ---
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configurable in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # --- Custom Middleware ---
    from isocortex.api.middleware.request_id import RequestIDMiddleware
    app.add_middleware(RequestIDMiddleware)

    # Rate limiting middleware (after request ID, before auth)
    from isocortex.api.middleware.rate_limit import RateLimitMiddleware
    app.add_middleware(RateLimitMiddleware)

    # --- Error Handlers ---
    from isocortex.api.middleware.error_handler import register_error_handlers
    register_error_handlers(app)

    # --- Include Routers ---
    from isocortex.api.routes import (
        auth_router,
        indexes_router,
        search_router,
        documents_router,
        jobs_router,
        admin_router,
    )

    app.include_router(auth_router, prefix="/api/v1/auth", tags=["Authentication"])
    app.include_router(indexes_router, prefix="/api/v1/indexes", tags=["Indexes"])
    app.include_router(search_router, prefix="/api/v1/indexes", tags=["Search"])
    app.include_router(documents_router, prefix="/api/v1/indexes", tags=["Documents"])
    app.include_router(jobs_router, prefix="/api/v1/jobs", tags=["Jobs"])
    app.include_router(admin_router, prefix="/api/v1/admin", tags=["Admin"])

    # Health check (no auth)
    from isocortex.api.routes.admin import router as health_router
    app.include_router(health_router, tags=["Health"])

    return app


# Module-level app instance for `uvicorn isocortex.api:app`
app = create_app()
