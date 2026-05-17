"""
IsoCortex engine/analytics — Analytics Engine
==============================================

High-level analytics wrapper built on top of the SQLite-backed
storage layer. Provides search logging, usage metrics, system
health monitoring, and admin dashboards.

SRS References:
  - FR-API: Analytics tracking
  - NFR-07: Rate limiting metrics
  - NFR-17: Embedding cache hit rate tracking

Author : Shaheer Qureshi
Project: IsoCortex
"""

from isocortex.engine.analytics.engine import AnalyticsEngine

__all__ = ["AnalyticsEngine"]
