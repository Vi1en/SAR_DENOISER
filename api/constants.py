"""Shared limits and enums for HTTP API."""
from __future__ import annotations

from enum import Enum

MAX_UPLOAD_BYTES = 20 * 1024 * 1024


class DenoiseMethodEnum(str, Enum):
    admm = "ADMM-PnP-DL"
    direct = "Direct Denoising"
    tv = "TV Denoising"


def default_model_type_label() -> str:
    from api.infer_config import model_type_label

    return model_type_label()
