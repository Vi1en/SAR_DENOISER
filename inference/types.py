"""Types for SAR denoising inference (no Streamlit)."""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Tuple

import numpy as np

DenoiseMethod = Literal["ADMM-PnP-DL", "Direct Denoising", "TV Denoising"]
MessageLevel = Literal["info", "success", "warning", "error"]

# (level, text) for UI layers (Streamlit, CLI, FastAPI) to replay
Message = Tuple[MessageLevel, str]


def denoise_result_dict(
    denoised: np.ndarray,
    energies: List[float],
    residuals: List[float],
    messages: List[Message],
    meta: Dict[str, Any] | None = None,
    *,
    uncertainty: np.ndarray | None = None,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "denoised": denoised,
        "energies": energies,
        "residuals": residuals,
        "messages": messages,
        "meta": meta or {},
    }
    if uncertainty is not None:
        out["uncertainty"] = uncertainty
    return out
