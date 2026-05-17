"""
IsoCortex — Error Handler Middleware
======================================

Global exception handlers that produce standardized error responses.

SRS References:
  - Section 10: Error Handling Specification
  - Section 10.2: Standard error format
  - Section 10.4: Error codes reference
  - Appendix C: Complete API error codes

Author : Shaheer Qureshi
Project: IsoCortex
"""

from __future__ import annotations

import logging
import traceback
import uuid

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


def register_error_handlers(app: FastAPI) -> None:
    """Register all global error handlers on the FastAPI app.

    SRS Section 10: Every error response includes error, code, detail, request_id.
    """

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        """Handle Pydantic validation errors (422)."""
        raw_errors = exc.errors()
        errors = []
        for e in raw_errors:
            safe = dict(e)
            if "ctx" in safe and isinstance(safe["ctx"], dict):
                safe["ctx"] = {
                    k: str(v) if not isinstance(v, (str, int, float, bool, list, dict, type(None))) else v
                    for k, v in safe["ctx"].items()
                }
            errors.append(safe)

        detail = "; ".join(
            f"{e.get('loc', ['?'])[-1]}: {e.get('msg', '')}" for e in errors
        )
        request_id = getattr(request.state, "request_id", str(uuid.uuid4()))

        logger.warning(
            "[VALIDATION] %s %s — %s  request_id=%s",
            request.method, request.url.path, detail, request_id,
        )

        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "error": "VALIDATION_ERROR",
                "code": 422,
                "detail": detail,
                "request_id": request_id,
                "fields": errors,
            },
        )

    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError):
        """Handle ValueError (400 Bad Request)."""
        request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"error": "BAD_REQUEST", "code": 400, "detail": str(exc), "request_id": request_id},
        )

    @app.exception_handler(FileNotFoundError)
    async def not_found_handler(request: Request, exc: FileNotFoundError):
        """Handle FileNotFoundError (404 Not Found)."""
        request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"error": "NOT_FOUND", "code": 404, "detail": str(exc), "request_id": request_id},
        )

    @app.exception_handler(FileExistsError)
    async def conflict_handler(request: Request, exc: FileExistsError):
        """Handle FileExistsError (409 Conflict)."""
        request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
        return JSONResponse(
            status_code=status.HTTP_409_CONFLICT,
            content={"error": "CONFLICT", "code": 409, "detail": str(exc), "request_id": request_id},
        )

    @app.exception_handler(PermissionError)
    async def forbidden_handler(request: Request, exc: PermissionError):
        """Handle PermissionError (403 Forbidden)."""
        request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
        return JSONResponse(
            status_code=status.HTTP_403_FORBIDDEN,
            content={"error": "FORBIDDEN", "code": 403, "detail": str(exc), "request_id": request_id},
        )

    @app.exception_handler(Exception)
    async def generic_error_handler(request: Request, exc: Exception):
        """Handle all uncaught exceptions (500 Internal Error)."""
        request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
        logger.error(
            "[INTERNAL] %s %s — %s  request_id=%s\n%s",
            request.method, request.url.path, exc, request_id,
            traceback.format_exc(),
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "INTERNAL_ERROR",
                "code": 500,
                "detail": "An unexpected error occurred. Please check the request_id in server logs.",
                "request_id": request_id,
            },
        )
