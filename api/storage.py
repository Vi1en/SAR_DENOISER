"""
On-disk layout for async denoise jobs.

``data/jobs/<job_id>/``
  - ``input.png`` — uploaded image
  - ``output.png`` — written by worker when successful
  - ``meta.json`` — request parameters + ``rq_job_id``
  - ``status.json`` — ``queued`` | ``running`` | ``done`` | ``failed``

**Disk growth:** old job directories are not auto-deleted; add a cron job or periodic
cleanup (e.g. delete dirs older than N days). See README (Step 08).
"""
from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any, Dict


def jobs_root() -> Path:
    return Path(os.environ.get("SAR_JOBS_DIR", "data/jobs"))


def job_dir(job_id: str) -> Path:
    return jobs_root() / job_id


def new_job_id() -> str:
    return str(uuid.uuid4())


def ensure_job_dir(job_id: str) -> Path:
    d = job_dir(job_id)
    d.mkdir(parents=True, exist_ok=False)
    return d


def write_meta(job_id: str, meta: Dict[str, Any]) -> None:
    p = job_dir(job_id) / "meta.json"
    p.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def read_meta(job_id: str) -> Dict[str, Any]:
    p = job_dir(job_id) / "meta.json"
    return json.loads(p.read_text(encoding="utf-8"))


def write_status(job_id: str, status: str, error: str | None = None) -> None:
    payload: Dict[str, Any] = {"status": status}
    if error is not None:
        payload["error"] = error
    p = job_dir(job_id) / "status.json"
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_status(job_id: str) -> Dict[str, Any] | None:
    p = job_dir(job_id) / "status.json"
    if not p.is_file():
        return None
    return json.loads(p.read_text(encoding="utf-8"))
