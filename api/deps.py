"""
Lazily constructed :class:`~inference.service.SARDenoiseService` singleton.

Environment
-----------
SAR_CHECKPOINT
    Optional path to ``.pth``. When set, used as both improved and simple checkpoint
    paths (same behavior as pointing Streamlit at one weights file). Omit for TV-only
    or to rely on default ``checkpoints_*`` paths in the repo.
SAR_DEVICE
    ``cpu``, ``cuda``, or ``auto`` (default).
SAR_MODEL_TYPE
    Slug ``unet``, ``dncnn``, or ``res_unet`` — default label for API ``model_type`` query (see ``api.main``).
SAR_BACKEND
    ``pytorch`` (default) or ``onnx`` — Direct denoising only; ADMM still uses PyTorch.
SAR_ONNX_PATH
    Path to ``.onnx`` when ``SAR_BACKEND=onnx``.
SAR_ONNX_EP
    Optional comma-separated ORT providers, e.g. ``CUDAExecutionProvider,CPUExecutionProvider``.
SAR_INFER_CONFIG
    Optional path to infer YAML (default ``configs/infer/default.yaml``). See ``api.infer_config``.
    YAML may set ``backend`` / ``onnx_path`` for Direct denoising; ``SAR_BACKEND`` / ``SAR_ONNX_PATH`` override.
SAR_GIT_SHA
    Optional short git revision for ``GET /health`` (set at image build time if ``git`` is unavailable).
SAR_ACCESS_LOG_JSON
    Set to ``1`` / ``true`` to emit one JSON log line per HTTP request (logger ``sar.api.access``).
SAR_LOG_LEVEL
    Logging level for ``sar.api.access`` (default ``INFO``).
SAR_API_KEY
    When set, ``POST /v1/denoise`` and ``/v1/jobs*`` require ``Authorization: Bearer <key>`` or ``X-API-Key: <key>``. ``/health``, ``/ready``, and ``/docs`` stay open.
SAR_USE_QUEUE / REDIS_URL
    When ``SAR_USE_QUEUE=1``, ``GET /ready`` returns **503** if ``REDIS_URL`` is unset or Redis does not answer ``PING`` (Kubernetes-style readiness).
"""
from __future__ import annotations

from pathlib import Path

from inference.service import SARDenoiseService

_service: SARDenoiseService | None = None


def get_service() -> SARDenoiseService:
    global _service
    if _service is None:
        from api.infer_config import get_merged, service_options_from_merged

        m = get_merged()
        opts = service_options_from_merged()
        device = str(m.get("device", "auto"))
        ck = m.get("checkpoint")
        ckpt = str(ck).strip() if ck not in (None, "") else None
        p = Path(ckpt).resolve() if ckpt else None
        _service = SARDenoiseService(
            device=device,
            improved_checkpoint=p,
            simple_checkpoint=p,
            infer_backend=opts["infer_backend"],
            onnx_path=opts["onnx_path"],
        )
    return _service


def reset_service_for_tests() -> None:
    """Clear singleton (tests only)."""
    global _service
    _service = None
