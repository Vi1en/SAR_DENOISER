"""
RQ task: load job dir, run :meth:`~inference.service.SARDenoiseService.denoise_numpy`, save PNG.

Run worker from **repository root**::

    export REDIS_URL=redis://localhost:6379/0
    rq worker sar_denoise --url $REDIS_URL
"""
from __future__ import annotations

import io
import json
import os
import sys
import traceback
from pathlib import Path

# RQ worker must be started with cwd = repo root; support PYTHONPATH if needed
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def run_denoise_job(job_dir: str) -> None:
    job_path = Path(job_dir)
    job_id = job_path.name

    from api import storage as job_storage
    from api.infer_config import service_options_from_merged
    from inference.service import SARDenoiseService
    from inference.types import DenoiseMethod
    from PIL import Image
    import numpy as np

    job_storage.write_status(job_id, "running")
    try:
        meta = json.loads((job_path / "meta.json").read_text(encoding="utf-8"))
        method = meta["method"]
        if method not in ("ADMM-PnP-DL", "Direct Denoising", "TV Denoising"):
            raise ValueError(f"Invalid method: {method}")
        method_typed: DenoiseMethod = method  # type: ignore[assignment]

        data = (job_path / "input.png").read_bytes()
        img = np.array(Image.open(io.BytesIO(data)).convert("L"), dtype=np.float32) / 255.0

        device = meta.get("device") or os.environ.get("SAR_DEVICE", "auto")
        ckpt = meta.get("checkpoint")
        p = Path(ckpt).resolve() if ckpt else None
        opts = service_options_from_merged()
        svc = SARDenoiseService(
            device=device,
            improved_checkpoint=p,
            simple_checkpoint=p,
            infer_backend=opts["infer_backend"],
            onnx_path=opts["onnx_path"],
        )
        direct_ck = p if ckpt and method == "Direct Denoising" else None
        include_unc = bool(meta.get("include_uncertainty", False))

        out = svc.denoise_numpy(
            img,
            method_typed,
            model_type=meta.get("model_type", "U-Net"),
            max_iter=int(meta.get("max_iter", 15)),
            rho_init=float(meta.get("rho_init", 1.0)),
            alpha=float(meta.get("alpha", 0.1)),
            theta=float(meta.get("theta", 0.05)),
            use_log_transform=bool(meta.get("use_log_transform", False)),
            quality_enhancement=bool(meta.get("quality_enhancement", False)),
            speckle_factor=float(meta.get("speckle_factor", 0.3)),
            direct_checkpoint=direct_ck,
            return_uncertainty=include_unc and method == "Direct Denoising",
            uncertainty_tta_passes=int(meta.get("uncertainty_tta_passes", 4)),
        )

        den = (out["denoised"].clip(0, 1) * 255).astype("uint8")
        buf = io.BytesIO()
        Image.fromarray(den).save(buf, format="PNG")
        (job_path / "output.png").write_bytes(buf.getvalue())

        if include_unc and out.get("uncertainty") is not None:
            from inference.uncertainty import uncertainty_to_vis_u8

            u8 = uncertainty_to_vis_u8(out["uncertainty"])
            ub = io.BytesIO()
            Image.fromarray(u8).save(ub, format="PNG")
            (job_path / "uncertainty.png").write_bytes(ub.getvalue())
        job_storage.write_status(job_id, "done")
    except Exception as e:
        tb = traceback.format_exc()
        job_storage.write_status(job_id, "failed", error=f"{e!s}\n{tb}"[:8000])
        raise
