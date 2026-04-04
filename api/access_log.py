"""
Optional JSON access logs (one line per request) for operators and log aggregators.

Enable with ``SAR_ACCESS_LOG_JSON=1``. Sets response header ``X-Request-ID`` (generated or
echoed from the incoming ``X-Request-ID`` when logging is enabled).
"""
from __future__ import annotations

import json
import logging
import os
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

_LOG = logging.getLogger("sar.api.access")


def _access_log_enabled() -> bool:
    v = os.environ.get("SAR_ACCESS_LOG_JSON", "").strip().lower()
    return v in ("1", "true", "yes", "on")


class JsonAccessLogMiddleware(BaseHTTPMiddleware):
    """Log ``method``, ``path``, ``status_code``, ``elapsed_ms``, ``request_id`` as JSON."""

    async def dispatch(self, request: Request, call_next):
        if not _access_log_enabled():
            return await call_next(request)

        rid = request.headers.get("x-request-id") or str(uuid.uuid4())
        t0 = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        response.headers["X-Request-ID"] = rid
        line = json.dumps(
            {
                "event": "http_request",
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "elapsed_ms": elapsed_ms,
                "request_id": rid,
            },
            separators=(",", ":"),
        )
        _LOG.info(line)
        return response
