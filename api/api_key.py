"""
Optional shared-secret gate for mutating ``/v1/*`` routes.

When ``SAR_API_KEY`` is non-empty, clients must send either:

- ``Authorization: Bearer <key>``, or
- ``X-API-Key: <key>``

``GET /health``, OpenAPI (``/docs``, ``/openapi.json``), etc. stay unauthenticated.
"""
from __future__ import annotations

import hmac
import os

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

_PROTECTED_PREFIXES = ("/v1/denoise", "/v1/jobs")


def _extract_api_key(request: Request) -> str:
    auth = request.headers.get("authorization") or ""
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return (request.headers.get("x-api-key") or "").strip()


def _keys_match(got: str, want: str) -> bool:
    ga, wa = got.encode("utf-8"), want.encode("utf-8")
    return len(ga) == len(wa) and hmac.compare_digest(ga, wa)


class OptionalApiKeyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        expected = os.environ.get("SAR_API_KEY", "").strip()
        if not expected:
            return await call_next(request)

        path = request.url.path
        if not any(path.startswith(p) for p in _PROTECTED_PREFIXES):
            return await call_next(request)

        token = _extract_api_key(request)
        if not _keys_match(token, expected):
            return JSONResponse(
                status_code=401,
                content={"detail": "Unauthorized: invalid or missing API key"},
            )
        return await call_next(request)
