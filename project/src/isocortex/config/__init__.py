"""
IsoCortex — Configuration Management
=====================================

Public API:
  - get_config()     → Returns the global AppConfig singleton
  - load_config()    → Loads from file/env, returns AppConfig
  - reset_config()   → Resets singleton (for testing)
  - AppConfig        → Master configuration dataclass
  - HnswConfig       → HNSW tuning parameters
  - EmbeddingConfig  → Embedding model settings
  - ChunkingConfig   → Text chunking settings
  - ServerConfig     → API server settings
  - StorageConfig    → Storage/SQLite settings
  - RateLimitConfig  → Rate limiting settings
"""

from isocortex.config.settings import (
    APP_NAME,
    AppConfig,
    ChunkingConfig,
    EmbeddingConfig,
    HnswConfig,
    RateLimitConfig,
    ServerConfig,
    StorageConfig,
    get_config,
    load_config,
    reset_config,
)

__all__ = [
    "APP_NAME",
    "AppConfig",
    "ChunkingConfig",
    "EmbeddingConfig",
    "HnswConfig",
    "RateLimitConfig",
    "ServerConfig",
    "StorageConfig",
    "get_config",
    "load_config",
    "reset_config",
]
