"""
IsoCortex api/schemas — Pydantic Request/Response Models
========================================================

All Pydantic models for API request validation and response serialization.
Covers every endpoint specified in SRS Sections 5.3-5.9 and Section 6.

Author : Shaheer Qureshi
Project: IsoCortex
"""

from isocortex.api.schemas.auth import (
    LoginRequest,
    LoginResponse,
    RefreshRequest,
    SetupRequest,
    SetupResponse,
    ChangePasswordRequest,
    CreateUserRequest,
    UserResponse,
    UserProfileResponse,
    UserRole,
)
from isocortex.api.schemas.indexes import (
    CreateIndexRequest,
    UpdateIndexRequest,
    IndexListResponse,
    IndexDetailResponse,
    IndexInfoResponse,
)
from isocortex.api.schemas.search import (
    SearchRequest,
    SearchResponse,
    BatchSearchRequest,
    BatchSearchResponse,
    SearchFilter,
)
from isocortex.api.schemas.documents import (
    AddDocumentsRequest,
    DocumentListResponse,
    DocumentResponse,
)
from isocortex.api.schemas.jobs import (
    JobResponse,
    JobListResponse,
)
from isocortex.api.schemas.common import (
    ErrorResponse,
    SuccessResponse,
    PaginatedResponse,
)
from isocortex.api.schemas.admin import (
    RateLimitResponse,
    RateLimitListResponse,
)

__all__ = [
    # Auth
    "LoginRequest", "LoginResponse", "RefreshRequest",
    "SetupRequest", "SetupResponse",
    "ChangePasswordRequest", "CreateUserRequest",
    "UserResponse", "UserProfileResponse", "UserRole",
    # Indexes
    "CreateIndexRequest", "UpdateIndexRequest",
    "IndexListResponse", "IndexDetailResponse", "IndexInfoResponse",
    # Search
    "SearchRequest", "SearchResponse",
    "BatchSearchRequest", "BatchSearchResponse", "SearchFilter",
    # Documents
    "AddDocumentsRequest", "DocumentListResponse", "DocumentResponse",
    # Jobs
    "JobResponse", "JobListResponse",
    # Common
    "ErrorResponse", "SuccessResponse", "PaginatedResponse",
    # Admin
    "RateLimitResponse", "RateLimitListResponse",
]
