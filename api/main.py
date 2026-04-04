"""
FastAPI app: health + synchronous PNG denoise.

Run from repository root::

    export SAR_CHECKPOINT=checkpoints_improved/best_model.pth   # optional; or set in configs/infer/
    uvicorn api.main:app --host 0.0.0.0 --port 8000

OpenAPI: ``/docs``. Optional auth: set ``SAR_API_KEY`` to require **Bearer** or **X-API-Key** on ``/v1/*``.
"""
from __future__ import annotations

import base64
import io
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import Response
from PIL import Image
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from api.access_log import JsonAccessLogMiddleware
from api.api_key import OptionalApiKeyMiddleware
from api.constants import MAX_UPLOAD_BYTES, DenoiseMethodEnum, default_model_type_label
from api.deps import get_service
from api.infer_config import effective_checkpoint_str, service_options_from_merged
from api.meta import git_sha_short
from inference.service import SARDenoiseService
from inference.types import DenoiseMethod


@asynccontextmanager
async def lifespan(app: FastAPI):
    import logging

    lvl_name = os.environ.get("SAR_LOG_LEVEL", "INFO").strip().upper()
    lvl = getattr(logging, lvl_name, logging.INFO)
    logging.getLogger("sar.api.access").setLevel(lvl)

    if os.environ.get("SAR_USE_QUEUE") == "1":
        url = os.environ.get("REDIS_URL")
        if url:
            try:
                from redis import Redis

                Redis.from_url(url).ping()
            except Exception:
                pass  # worker may start later; avoid hard-fail API
    yield


class LimitUploadSizeMiddleware(BaseHTTPMiddleware):
    """Reject oversized bodies on ``POST /v1/denoise`` and ``POST /v1/jobs``."""

    def __init__(self, app, max_bytes: int = MAX_UPLOAD_BYTES):
        super().__init__(app)
        self.max_bytes = max_bytes

    async def dispatch(self, request: Request, call_next):
        if request.method == "POST" and (
            request.url.path.startswith("/v1/denoise")
            or request.url.path.rstrip("/") == "/v1/jobs"
        ):
            cl = request.headers.get("content-length")
            if cl is not None:
                try:
                    n = int(cl)
                except ValueError:
                    n = -1
                if n > self.max_bytes:
                    return JSONResponse(
                        status_code=413,
                        content={"detail": f"Upload exceeds {self.max_bytes} bytes"},
                    )
        return await call_next(request)


API_VERSION = "0.1.0"
app = FastAPI(title="SAR Denoise API", version=API_VERSION, lifespan=lifespan)
app.add_middleware(LimitUploadSizeMiddleware, max_bytes=MAX_UPLOAD_BYTES)
app.add_middleware(JsonAccessLogMiddleware)
app.add_middleware(OptionalApiKeyMiddleware)

if os.environ.get("SAR_USE_QUEUE") == "1":
    from api.jobs import router as jobs_router

    app.include_router(jobs_router)


@app.get("/health")
def health():
    """Liveness: always 200 when the process is up. Includes optional deploy metadata."""
    body: dict = {"status": "ok", "version": API_VERSION}
    g = git_sha_short()
    if g:
        body["git_sha"] = g
    opts = service_options_from_merged()
    bk = opts.get("infer_backend")
    body["direct_infer_backend"] = bk if bk else "pytorch"
    body["onnx_path_set"] = bool(opts.get("onnx_path"))
    return body


@app.get("/ready")
def ready():
    """Readiness for orchestrators: when ``SAR_USE_QUEUE=1``, Redis must answer ``PING``."""
    if os.environ.get("SAR_USE_QUEUE") != "1":
        return {"status": "ready", "queue": "disabled"}

    url = os.environ.get("REDIS_URL", "").strip()
    if not url:
        return JSONResponse(
            status_code=503,
            content={
                "status": "not_ready",
                "detail": "SAR_USE_QUEUE=1 but REDIS_URL is unset",
            },
        )

    try:
        from redis import Redis

        Redis.from_url(url, socket_connect_timeout=2).ping()
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "detail": str(e)},
        )

    return {"status": "ready", "queue": "redis"}


@app.post("/v1/denoise", responses={413: {"description": "Payload too large"}})
async def denoise_v1(
    file: UploadFile = File(..., description="Grayscale or RGB image (converted to L)"),
    method: DenoiseMethodEnum = Query(DenoiseMethodEnum.admm),
    max_iter: int = Query(15, ge=1, le=100),
    rho_init: float = Query(1.0),
    alpha: float = Query(0.1),
    theta: float = Query(0.05),
    use_log_transform: bool = Query(False),
    quality_enhancement: bool = Query(False),
    speckle_factor: float = Query(0.3),
    model_type: str = Query(
        default_model_type_label(),
        description="U-Net, DnCNN, or Res-U-Net",
    ),
    response_format: str = Query(
        "png",
        description="png (default) or json (base64 PNG fields)",
    ),
    include_uncertainty: bool = Query(
        False,
        description="Direct denoising only: TTA pixelwise std map (requires response_format=json)",
    ),
    uncertainty_tta_passes: int = Query(4, ge=1, le=4),
    include_blind_qa: bool = Query(
        False,
        description="If true (with response_format=json), meta includes blind no-reference QA metrics.",
    ),
    svc: SARDenoiseService = Depends(get_service),
):
    """
    Synchronous denoise; default returns PNG (uint8). Use ``response_format=json`` for
    base64 payloads; ``include_uncertainty=1`` adds a TTA uncertainty heatmap (Direct only).
    """
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, detail=f"File exceeds {MAX_UPLOAD_BYTES} bytes")

    try:
        img = np.array(Image.open(io.BytesIO(data)).convert("L"), dtype=np.float32) / 255.0
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}") from e

    method_str: DenoiseMethod = method.value  # type: ignore[assignment]

    rf = response_format.lower()
    if rf not in ("png", "json"):
        raise HTTPException(status_code=400, detail="response_format must be png or json")
    if include_uncertainty and rf != "json":
        raise HTTPException(
            status_code=400,
            detail="include_uncertainty requires response_format=json",
        )
    if include_uncertainty and method_str != "Direct Denoising":
        raise HTTPException(
            status_code=400,
            detail="include_uncertainty is only supported for Direct Denoising",
        )

    ckpt = effective_checkpoint_str()
    direct_ck = Path(ckpt).resolve() if ckpt and method_str == "Direct Denoising" else None

    t0 = time.perf_counter()
    try:
        out = svc.denoise_numpy(
            img,
            method_str,
            model_type=model_type,
            max_iter=max_iter,
            rho_init=rho_init,
            alpha=alpha,
            theta=theta,
            use_log_transform=use_log_transform,
            quality_enhancement=quality_enhancement,
            speckle_factor=speckle_factor,
            direct_checkpoint=direct_ck,
            return_uncertainty=include_uncertainty,
            uncertainty_tta_passes=uncertainty_tta_passes,
            include_blind_qa=include_blind_qa and rf == "json",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)

    den = (np.clip(out["denoised"], 0, 1) * 255).astype(np.uint8)
    out_buf = io.BytesIO()
    Image.fromarray(den).save(out_buf, format="PNG")
    den_png = out_buf.getvalue()

    if rf == "json":
        from inference.uncertainty import uncertainty_to_vis_u8

        meta = dict(out.get("meta") or {})
        meta["inference_ms"] = elapsed_ms
        payload: dict = {
            "meta": meta,
            "denoised_png": base64.standard_b64encode(den_png).decode("ascii"),
        }
        unc = out.get("uncertainty")
        if unc is not None:
            u8 = uncertainty_to_vis_u8(np.asarray(unc))
            ub = io.BytesIO()
            Image.fromarray(u8).save(ub, format="PNG")
            payload["uncertainty_png"] = base64.standard_b64encode(ub.getvalue()).decode("ascii")
        return JSONResponse(content=payload)

    return Response(content=den_png, media_type="image/png")
