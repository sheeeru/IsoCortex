"""
IsoCortex — Configuration Management
=====================================

Centralized configuration for all IsoCortex components.
Supports environment variables, config file loading, and validation.

SRS References: NFR-3, NFR-4, CON-1, CON-5
Default data directory: ~/.isocortex/

Author : Shaheer Qureshi
Project: IsoCortex
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

DEFAULT_HNSW_M: int = 16
DEFAULT_HNSW_EF_CONSTRUCTION: int = 200
DEFAULT_HNSW_EF_SEARCH: int = 50
DEFAULT_VECTOR_DIM: int = 384
DEFAULT_METRIC: str = "cosine"
DEFAULT_MAX_ELEMENTS: int = 100000

DEFAULT_EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_EMBEDDING_BATCH_SIZE: int = 64
DEFAULT_EMBEDDING_CACHE_SIZE: int = 1000
DEFAULT_EMBEDDING_QUEUE_SIZE: int = 500

DEFAULT_CHUNK_SIZE: int = 350  # ~512 tokens for MiniLM
DEFAULT_CHUNK_OVERLAP: int = 38  # ~50 tokens
DEFAULT_TOKEN_LIMIT: int = 512

DEFAULT_API_HOST: str = "0.0.0.0"
DEFAULT_API_PORT: int = 8900
DEFAULT_API_WORKERS: int = 1

DEFAULT_DB_NAME: str = "isocortex.db"
DEFAULT_WAL_AUTO_CHECKPOINT: int = 1000
DEFAULT_BUSY_TIMEOUT: int = 5000

DEFAULT_MAX_FILE_SIZE_MB: float = 50.0
DEFAULT_COMPACTION_THRESHOLD: float = 0.10

DEFAULT_JWT_SECRET_KEY: str = "isocortex-default-change-in-production"
DEFAULT_JWT_ALGORITHM: str = "HS256"
DEFAULT_JWT_EXPIRE_MINUTES: int = 1440  # 24 hours

DEFAULT_RATE_LIMIT_REQUESTS: int = 100
DEFAULT_RATE_LIMIT_WINDOW_SECONDS: int = 60

APP_NAME: str = "IsoCortex"


# =============================================================================
# Data Classes
# =============================================================================

@dataclass(frozen=True)
class HnswConfig:
    """HNSW index tuning parameters (SRS FR-IDX-001)."""
    M: int = DEFAULT_HNSW_M
    ef_construction: int = DEFAULT_HNSW_EF_CONSTRUCTION
    ef_search: int = DEFAULT_HNSW_EF_SEARCH
    max_elements: int = DEFAULT_MAX_ELEMENTS
    dim: int = DEFAULT_VECTOR_DIM
    metric: str = DEFAULT_METRIC

    def __post_init__(self) -> None:
        if not (4 <= self.M <= 128):
            raise ValueError(f"HnswConfig: M must be 4-128, got {self.M}")
        if not (50 <= self.ef_construction <= 2000):
            raise ValueError(f"HnswConfig: ef_construction must be 50-2000, got {self.ef_construction}")
        if not (10 <= self.ef_search <= 500):
            raise ValueError(f"HnswConfig: ef_search must be 10-500, got {self.ef_search}")
        if self.dim != 384:
            raise ValueError(f"HnswConfig: dim must be 384, got {self.dim}")
        if self.metric not in ("cosine", "l2", "ip"):
            raise ValueError(f"HnswConfig: metric must be cosine/l2/ip, got {self.metric}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "M": self.M,
            "efConstruction": self.ef_construction,
            "efSearch": self.ef_search,
            "max_elements": self.max_elements,
            "dim": self.dim,
            "space": self.metric,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> HnswConfig:
        return cls(
            M=d.get("M", DEFAULT_HNSW_M),
            ef_construction=d.get("efConstruction", DEFAULT_HNSW_EF_CONSTRUCTION),
            ef_search=d.get("efSearch", DEFAULT_HNSW_EF_SEARCH),
            max_elements=d.get("max_elements", DEFAULT_MAX_ELEMENTS),
            dim=d.get("dim", DEFAULT_VECTOR_DIM),
            metric=d.get("space", DEFAULT_METRIC),
        )


@dataclass(frozen=True)
class EmbeddingConfig:
    """Embedding model configuration (SRS FR-EMB-001, FR-EMB-002)."""
    model_name: str = DEFAULT_EMBEDDING_MODEL
    batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE
    cache_size: int = DEFAULT_EMBEDDING_CACHE_SIZE
    queue_size: int = DEFAULT_EMBEDDING_QUEUE_SIZE
    vector_dim: int = DEFAULT_VECTOR_DIM

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "batch_size": self.batch_size,
            "cache_size": self.cache_size,
            "queue_size": self.queue_size,
            "vector_dim": self.vector_dim,
        }


@dataclass(frozen=True)
class ChunkingConfig:
    """Text chunking configuration (SRS FR-ING-002)."""
    chunk_size: int = DEFAULT_CHUNK_SIZE
    overlap: int = DEFAULT_CHUNK_OVERLAP
    token_limit: int = DEFAULT_TOKEN_LIMIT

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_size": self.chunk_size,
            "overlap": self.overlap,
            "token_limit": self.token_limit,
        }


@dataclass(frozen=True)
class ServerConfig:
    """API server configuration."""
    host: str = DEFAULT_API_HOST
    port: int = DEFAULT_API_PORT
    workers: int = DEFAULT_API_WORKERS
    jwt_secret_key: str = DEFAULT_JWT_SECRET_KEY
    jwt_algorithm: str = DEFAULT_JWT_ALGORITHM
    jwt_expire_minutes: int = DEFAULT_JWT_EXPIRE_MINUTES

    @classmethod
    def from_env(cls) -> ServerConfig:
        return cls(
            host=os.environ.get("ISOCORTEX_HOST", DEFAULT_API_HOST),
            port=int(os.environ.get("ISOCORTEX_PORT", str(DEFAULT_API_PORT))),
            workers=int(os.environ.get("ISOCORTEX_WORKERS", str(DEFAULT_API_WORKERS))),
            jwt_secret_key=os.environ.get(
                "ISOCORTEX_JWT_SECRET", DEFAULT_JWT_SECRET_KEY
            ),
            jwt_algorithm=os.environ.get(
                "ISOCORTEX_JWT_ALGORITHM", DEFAULT_JWT_ALGORITHM
            ),
            jwt_expire_minutes=int(os.environ.get(
                "ISOCORTEX_JWT_EXPIRE", str(DEFAULT_JWT_EXPIRE_MINUTES)
            )),
        )


@dataclass(frozen=True)
class StorageConfig:
    """Storage and SQLite configuration."""
    data_dir: Path = field(default_factory=lambda: Path.home() / ".isocortex")
    db_name: str = DEFAULT_DB_NAME
    wal_auto_checkpoint: int = DEFAULT_WAL_AUTO_CHECKPOINT
    busy_timeout: int = DEFAULT_BUSY_TIMEOUT
    max_file_size_mb: float = DEFAULT_MAX_FILE_SIZE_MB
    compaction_threshold: float = DEFAULT_COMPACTION_THRESHOLD

    def __post_init__(self) -> None:
        if not (0 < self.compaction_threshold <= 1.0):
            raise ValueError(
                f"StorageConfig: compaction_threshold must be 0-1, got {self.compaction_threshold}"
            )

    @property
    def db_path(self) -> Path:
        return self.data_dir / self.db_name

    @property
    def models_dir(self) -> Path:
        return self.data_dir / "models"

    @property
    def indices_dir(self) -> Path:
        return self.data_dir / "indices"

    @classmethod
    def from_env(cls) -> StorageConfig:
        data_dir = os.environ.get("ISOCORTEX_DATA_DIR", "")
        return cls(
            data_dir=Path(data_dir) if data_dir else Path.home() / ".isocortex",
            db_name=os.environ.get("ISOCORTEX_DB_NAME", DEFAULT_DB_NAME),
            max_file_size_mb=float(os.environ.get(
                "ISOCORTEX_MAX_FILE_SIZE_MB", str(DEFAULT_MAX_FILE_SIZE_MB)
            )),
            compaction_threshold=float(os.environ.get(
                "ISOCORTEX_COMPACTION_THRESHOLD", str(DEFAULT_COMPACTION_THRESHOLD)
            )),
        )


@dataclass(frozen=True)
class RateLimitConfig:
    """Rate limiting configuration."""
    requests: int = DEFAULT_RATE_LIMIT_REQUESTS
    window_seconds: int = DEFAULT_RATE_LIMIT_WINDOW_SECONDS


@dataclass
class AppConfig:
    """
    Master application configuration.

    Combines all sub-configurations. Supports loading from environment
    variables and from a JSON config file.

    SRS References: CON-1, CON-5, NFR-3, NFR-4
    """
    hnsw: HnswConfig = field(default_factory=HnswConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    server: ServerConfig = field(default_factory=ServerConfig.from_env)
    storage: StorageConfig = field(default_factory=StorageConfig.from_env)
    rate_limit: RateLimitConfig = field(default_factory=RateLimitConfig)

    def ensure_directories(self) -> None:
        """Create required directories if they don't exist."""
        dirs = [
            self.storage.data_dir,
            self.storage.models_dir,
            self.storage.indices_dir,
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
            logger.debug("[CONFIG] Ensured directory: %s", d)

    def to_dict(self) -> dict[str, Any]:
        return {
            "hnsw": self.hnsw.to_dict(),
            "embedding": self.embedding.to_dict(),
            "chunking": self.chunking.to_dict(),
            "server": {
                "host": self.server.host,
                "port": self.server.port,
                "workers": self.server.workers,
            },
            "storage": {
                "data_dir": str(self.storage.data_dir),
                "db_name": self.storage.db_name,
                "max_file_size_mb": self.storage.max_file_size_mb,
                "compaction_threshold": self.storage.compaction_threshold,
            },
            "rate_limit": {
                "requests": self.rate_limit.requests,
                "window_seconds": self.rate_limit.window_seconds,
            },
        }

    @classmethod
    def from_file(cls, path: Path) -> AppConfig:
        """Load configuration from a JSON file."""
        if not path.exists():
            logger.warning("[CONFIG] Config file not found: %s — using defaults", path)
            return cls()

        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("[CONFIG] Failed to parse config file: %s — using defaults", exc)
            return cls()

        return cls(
            hnsw=HnswConfig.from_dict(data.get("hnsw", {})),
            embedding=EmbeddingConfig(**data.get("embedding", {})),
            chunking=ChunkingConfig(**data.get("chunking", {})),
            storage=StorageConfig(
                data_dir=Path(data["storage"]["data_dir"])
                if "storage" in data and "data_dir" in data["storage"]
                else Path.home() / ".isocortex",
                max_file_size_mb=data.get("storage", {}).get(
                    "max_file_size_mb", DEFAULT_MAX_FILE_SIZE_MB
                ),
                compaction_threshold=data.get("storage", {}).get(
                    "compaction_threshold", DEFAULT_COMPACTION_THRESHOLD
                ),
            ),
            rate_limit=RateLimitConfig(**data.get("rate_limit", {})),
        )

    def save_to_file(self, path: Path) -> None:
        """Save current configuration to a JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, ensure_ascii=False)
        logger.info("[CONFIG] Configuration saved to %s", path)


# =============================================================================
# Global configuration singleton
# =============================================================================

_global_config: Optional[AppConfig] = None


def get_config() -> AppConfig:
    """Return the global configuration singleton, creating it if needed."""
    global _global_config
    if _global_config is None:
        _global_config = AppConfig()
        _global_config.ensure_directories()
        logger.info(
            "[CONFIG] Initialized  data_dir=%s  port=%d  model=%s",
            _global_config.storage.data_dir,
            _global_config.server.port,
            _global_config.embedding.model_name,
        )
    return _global_config


def reset_config() -> None:
    """Reset the global configuration (used for testing)."""
    global _global_config
    _global_config = None


def load_config(path: Optional[Path] = None) -> AppConfig:
    """
    Load configuration from file or environment.

    Priority: config file > environment variables > defaults.

    Parameters
    ----------
    path : Path | None
        Path to a JSON configuration file. If None, checks
        ~/.isocortex/config.json, then uses defaults.

    Returns
    -------
    AppConfig
    """
    global _global_config

    if path is None:
        default_path = Path.home() / ".isocortex" / "config.json"
        if default_path.exists():
            path = default_path

    if path is not None and path.exists():
        _global_config = AppConfig.from_file(path)
    else:
        _global_config = AppConfig()

    _global_config.ensure_directories()

    logger.info(
        "[CONFIG] Loaded  data_dir=%s  port=%d",
        _global_config.storage.data_dir,
        _global_config.server.port,
    )

    return _global_config
