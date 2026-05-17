"""
IsoCortex — Request ID Middleware
===================================

Generates a unique UUID v4 for every incoming request and attaches it
to the request state. Also sets X-Request-ID response header.

SRS References:
  - Section 10.3: Request ID Generation
  - Section 10.2: Standard Error Format (request_id field)

Author : Shaheer Qureshi
Project: IsoCortex
"""

from __future__ import annotations

import logging
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger(__name__)


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Attach a unique request_id to every request.

    - Generates UUID v4 for each incoming request
    - Stores in request.state.request_id
    - Adds X-Request-ID response header
    - Logs request method, path, and duration
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id

        start = time.perf_counter()

        response = await call_next(request)

        duration_ms = (time.perf_counter() - start) * 1000

        response.headers["X-Request-ID"] = request_id

        logger.debug(
            "[REQ] %s %s -> %d (%.1fms) request_id=%s",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
            request_id,
        )

        return response
