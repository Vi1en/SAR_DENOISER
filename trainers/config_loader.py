"""
Load YAML training configs into ``TrainingConfig`` (safe_load only).
Paths in YAML are relative to the repository root unless absolute.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import yaml

from trainers.config_dataclass import TrainingConfig

# Keys accepted from YAML; others are ignored (e.g. use_sample_dataset, comments).
_TRAINING_KEYS = frozenset(
    {
        "seed",
        "device",
        "data_dir",
        "patch_size",
        "batch_size",
        "num_workers",
        "epochs",
        "lr",
        "checkpoint_dir",
        "model_type",
        "noise_conditioning",
        "results_png",
    }
)


def load_training_yaml(path: Path) -> Dict[str, Any]:
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping, got {type(data).__name__}")
    return dict(data)


def resolve_training_device(cfg: TrainingConfig) -> TrainingConfig:
    """Map ``device: auto`` (or legacy ``auto`` string) to ``cuda`` or ``cpu``."""
    if cfg.device != "auto":
        return cfg
    import torch

    picked = "cuda" if torch.cuda.is_available() else "cpu"
    return replace(cfg, device=picked)


def dict_to_training_config(d: Mapping[str, Any], base: Optional[TrainingConfig] = None) -> TrainingConfig:
    """
    Overlay YAML mapping onto ``TrainingConfig``.
    Missing keys keep ``base`` values (default: dataclass defaults).
    """
    base = base or TrainingConfig()
    if not d:
        return base

    kwargs: Dict[str, Any] = {}

    for key in _TRAINING_KEYS:
        if key not in d:
            continue
        val = d[key]
        if val is None:
            continue
        if key == "data_dir":
            kwargs["data_dir"] = Path(val)
        elif key == "checkpoint_dir":
            kwargs["checkpoint_dir"] = Path(val)
        elif key == "results_png":
            kwargs["results_png"] = Path(val)
        elif key == "device":
            kwargs["device"] = str(val)
        elif key == "model_type":
            kwargs["model_type"] = str(val)
        elif key == "noise_conditioning":
            kwargs["noise_conditioning"] = bool(val)
        elif key in ("patch_size", "batch_size", "num_workers", "epochs", "seed"):
            kwargs[key] = int(val)
        elif key == "lr":
            kwargs["lr"] = float(val)

    return replace(base, **kwargs)


def training_config_from_yaml_path(path: Path, base: Optional[TrainingConfig] = None) -> TrainingConfig:
    """Load YAML file and return ``TrainingConfig`` (device may still be ``auto``)."""
    raw = load_training_yaml(path)
    return dict_to_training_config(raw, base=base)
