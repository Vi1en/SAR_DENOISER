"""
Async denoise jobs via Redis + RQ.

Enable with ``SAR_USE_QUEUE=1`` and ``REDIS_URL`` (see README).
"""
from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image
from starlette.status import HTTP_425_TOO_EARLY
from redis import Redis
from rq import Queue
from rq.job import Job
from rq.exceptions import NoSuchJobError

from api import storage as job_storage
from api.constants import MAX_UPLOAD_BYTES, DenoiseMethodEnum, default_model_type_label

router = APIRouter(prefix="/v1/jobs", tags=["jobs"])

QUEUE_NAME = "sar_denoise"


def _require_redis_url() -> str:
    url = os.environ.get("REDIS_URL")
    if not url:
        raise HTTPException(
            status_code=503,
            detail="Job queue disabled: set REDIS_URL (and SAR_USE_QUEUE=1).",
        )
    return url


def get_queue() -> Queue:
    return Queue(QUEUE_NAME, connection=Redis.from_url(_require_redis_url()))


def get_redis() -> Redis:
    return Redis.from_url(_require_redis_url())


@router.post("", response_model=None)
async def create_job(
    file: UploadFile = File(...),
    method: DenoiseMethodEnum = Query(DenoiseMethodEnum.admm),
    max_iter: int = Query(15, ge=1, le=100),
    rho_init: float = Query(1.0),
    alpha: float = Query(0.1),
    theta: float = Query(0.05),
    use_log_transform: bool = Query(False),
    quality_enhancement: bool = Query(False),
    speckle_factor: float = Query(0.3),
    model_type: str = Query(default_model_type_label()),
    include_uncertainty: bool = Query(False),
    uncertainty_tta_passes: int = Query(4, ge=1, le=4),
):
    """Submit image; returns ``job_id`` to poll with ``GET /v1/jobs/{id}``."""
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, detail=f"File exceeds {MAX_UPLOAD_BYTES} bytes")

    try:
        Image.open(io.BytesIO(data)).convert("L")
    except Exception as e:
        raise HTTPException(400, detail=f"Invalid image: {e}") from e

    if include_uncertainty and method.value != "Direct Denoising":
        raise HTTPException(
            status_code=400,
            detail="include_uncertainty is only valid for Direct Denoising",
        )

    job_id = job_storage.new_job_id()
    try:
        d = job_storage.ensure_job_dir(job_id)
    except FileExistsError:
        raise HTTPException(500, detail="Job id collision") from None

    (d / "input.png").write_bytes(data)

    from api.infer_config import effective_checkpoint_str, get_merged

    ckpt = effective_checkpoint_str()
    merged = get_merged()
    meta: Dict[str, Any] = {
        "method": method.value,
        "model_type": model_type,
        "max_iter": max_iter,
        "rho_init": rho_init,
        "alpha": alpha,
        "theta": theta,
        "use_log_transform": use_log_transform,
        "quality_enhancement": quality_enhancement,
        "speckle_factor": speckle_factor,
        "checkpoint": str(Path(ckpt).resolve()) if ckpt else None,
        "device": str(merged.get("device", "auto")),
        "include_uncertainty": include_uncertainty,
        "uncertainty_tta_passes": uncertainty_tta_passes,
    }
    job_storage.write_meta(job_id, meta)
    job_storage.write_status(job_id, "queued")

    q = get_queue()
    rq_job = q.enqueue(
        "workers.tasks.run_denoise_job",
        str(d.resolve()),
        job_timeout="30m",
        result_ttl=3600,
        failure_ttl=3600,
    )
    meta["rq_job_id"] = rq_job.id
    job_storage.write_meta(job_id, meta)

    return JSONResponse({"job_id": job_id, "rq_job_id": rq_job.id})


@router.get("/{job_id}")
def get_job_status(job_id: str) -> Dict[str, Any]:
    d = job_storage.job_dir(job_id)
    if not d.is_dir():
        raise HTTPException(404, detail="Unknown job_id")

    meta = job_storage.read_meta(job_id)
    disk = job_storage.read_status(job_id)
    disk_status = disk.get("status") if disk else None
    disk_err = disk.get("error") if disk else None

    status = disk_status or "queued"
    error = disk_err

    rq_id = meta.get("rq_job_id")
    if rq_id:
        try:
            job = Job.fetch(rq_id, connection=get_redis())
            if job.is_failed:
                status = "failed"
                err_rq = job.exc_info or (str(job.result) if job.result is not None else "")
                error = (error or err_rq)[:2000]
            elif disk_status not in ("done", "failed"):
                if job.is_started:
                    status = "running"
                elif job.is_finished and not job.is_failed:
                    status = "done" if (d / "output.png").is_file() else "running"
        except NoSuchJobError:
            if disk_status is None:
                status = "failed"
                error = error or "RQ job not found"

    if disk_status in ("done", "failed"):
        status = disk_status
        error = disk_err

    return {"job_id": job_id, "status": status, "error": error}


@router.get("/{job_id}/result")
def get_job_result(job_id: str):
    st = job_storage.read_status(job_id)
    if not st or st.get("status") != "done":
        raise HTTPException(
            HTTP_425_TOO_EARLY,
            detail="Result not ready; poll GET /v1/jobs/{id} until status is done.",
        )
    outp = job_storage.job_dir(job_id) / "output.png"
    if not outp.is_file():
        raise HTTPException(404, detail="Output missing")
    return FileResponse(outp, media_type="image/png", filename="denoised.png")


@router.get("/{job_id}/uncertainty")
def get_job_uncertainty(job_id: str):
    st = job_storage.read_status(job_id)
    if not st or st.get("status") != "done":
        raise HTTPException(
            HTTP_425_TOO_EARLY,
            detail="Result not ready; poll GET /v1/jobs/{id} until status is done.",
        )
    up = job_storage.job_dir(job_id) / "uncertainty.png"
    if not up.is_file():
        raise HTTPException(404, detail="Uncertainty output not available for this job")
    return FileResponse(up, media_type="image/png", filename="uncertainty.png")
