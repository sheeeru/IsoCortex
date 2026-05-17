"""
IsoCortex api/routes — Route Handlers
=====================================

All FastAPI route handlers organized by domain.

Each module exposes a single ``router`` (``APIRouter``) instance so that the
application factory can include them with a consistent pattern:

    from isocortex.api.routes import auth_router, indexes_router
    app.include_router(auth_router)
    app.include_router(indexes_router)

SRS References:
  - Section 5.3: REST API Overview
  - Section 6:   Authentication and User Management
  - FR-API-001:  Index Management Endpoints
  - FR-API-002:  Search Endpoints
  - FR-API-004:  Document Endpoints
  - Section 8:    Async Operations / Job Status

Author : Shaheer Qureshi
Project: IsoCortex
"""

from isocortex.api.routes.auth import router as auth_router
from isocortex.api.routes.indexes import router as indexes_router
from isocortex.api.routes.search import router as search_router
from isocortex.api.routes.documents import router as documents_router
from isocortex.api.routes.jobs import router as jobs_router
from isocortex.api.routes.admin import router as admin_router

__all__ = [
    "auth_router",
    "indexes_router",
    "search_router",
    "documents_router",
    "jobs_router",
    "admin_router",
]
