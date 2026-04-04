"""
Optional YAML defaults for inference (``configs/infer/default.yaml``).

Precedence: **environment variables** override file values. Set ``SAR_INFER_CONFIG`` to
a different YAML path (repo-relative or absolute).

Supported keys include ``device``, ``model_type``, ``checkpoint``, ``backend`` (``pytorch`` /
``onnx`` for Direct denoising), and ``onnx_path``. ``SAR_BACKEND`` and ``SAR_ONNX_PATH``
override ``backend`` and ``onnx_path`` when set.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def infer_config_path() -> Path:
    env = os.environ.get("SAR_INFER_CONFIG", "").strip()
    root = _repo_root()
    if env:
        p = Path(env)
        return p if p.is_absolute() else (root / p)
    return root / "configs" / "infer" / "default.yaml"


def _load_yaml_file(path: Path) -> dict[str, Any]:
    """Load YAML if PyYAML is installed; otherwise skip file (env vars still apply)."""
    if not path.is_file():
        return {}
    try:
        import yaml  # PyPI: PyYAML — lazy so import never fails at module load
    except ImportError:  # pragma: no cover
        return {}
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return raw if isinstance(raw, dict) else {}


def get_merged() -> dict[str, Any]:
    """File defaults overlaid with ``SAR_*`` env vars (device, checkpoint, model, backend, ONNX path)."""
    base = _load_yaml_file(infer_config_path())
    out: dict[str, Any] = dict(base)

    b = out.get("backend")
    if b is not None and str(b).strip():
        out["backend"] = str(b).strip().lower()
    else:
        out.pop("backend", None)
    op = out.get("onnx_path")
    if op is None or str(op).strip() == "":
        out.pop("onnx_path", None)
    elif not isinstance(op, str):
        out["onnx_path"] = str(op).strip()

    if os.environ.get("SAR_DEVICE", "").strip():
        out["device"] = os.environ["SAR_DEVICE"].strip()
    if os.environ.get("SAR_CHECKPOINT", "").strip():
        out["checkpoint"] = os.environ["SAR_CHECKPOINT"].strip()
    if os.environ.get("SAR_MODEL_TYPE", "").strip():
        out["model_type"] = (
            os.environ["SAR_MODEL_TYPE"].strip().lower().replace("-", "").replace("_", "")
        )
    if os.environ.get("SAR_BACKEND", "").strip():
        out["backend"] = os.environ["SAR_BACKEND"].strip().lower()
    if os.environ.get("SAR_ONNX_PATH", "").strip():
        out["onnx_path"] = os.environ["SAR_ONNX_PATH"].strip()

    return out


def service_options_from_merged() -> dict[str, str | None]:
    """``infer_backend`` / ``onnx_path`` for :class:`~inference.service.SARDenoiseService` (merged YAML + env)."""
    m = get_merged()
    ib = m.get("backend")
    op = m.get("onnx_path")
    return {
        "infer_backend": str(ib).strip() if ib not in (None, "") else None,
        "onnx_path": str(op).strip() if op not in (None, "") else None,
    }


def _normalize_model_slug(s: str) -> str:
    return s.strip().lower().replace("-", "").replace("_", "")


def model_type_label() -> str:
    """OpenAPI / query default: U-Net, DnCNN, or Res-U-Net."""
    slug = _normalize_model_slug(str(get_merged().get("model_type", "unet")))
    if slug == "dncnn":
        return "DnCNN"
    if slug == "resunet":
        return "Res-U-Net"
    return "U-Net"


def effective_checkpoint_str() -> str | None:
    c = get_merged().get("checkpoint")
    if c is None or c == "":
        return None
    return str(c)
