"""
No-reference (blind) quality proxies for denoised SAR-like images.

These do **not** require a clean reference. They are indicative only: scene-dependent
and not comparable across different sensors without calibration.
"""
from __future__ import annotations

import numpy as np
from skimage import filters


def _clip01(x: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(x, dtype=np.float64), 0.0, 1.0)


def gradient_magnitude(image: np.ndarray) -> np.ndarray:
    """Sobel gradient magnitude on ``[0, 1]`` float image."""
    g = _clip01(image)
    gx = filters.sobel_h(g)
    gy = filters.sobel_v(g)
    return np.hypot(gx, gy)


def blind_edge_preservation_index(
    noisy: np.ndarray, denoised: np.ndarray, eps: float = 1e-8
) -> float:
    """
    Edge preservation vs **input** (no clean reference): EPI-style ratio
    ``sum(Gn * Gd) / (sum(Gn^2) + eps)`` with ``G`` = gradient magnitude.
    Values near **1** → denoised edges align with the noisy input structure;
    very low values → heavy smoothing vs input edges.
    """
    gn = gradient_magnitude(noisy)
    gd = gradient_magnitude(denoised)
    num = float(np.sum(gn * gd))
    den = float(np.sum(gn * gn) + eps)
    return num / den


def blind_enl_median_homogeneous(
    denoised: np.ndarray,
    block: int = 16,
    homo_percentile: float = 25.0,
) -> float:
    """
    **ENL-like** estimate: median over spatial blocks of ``mean^2 / variance``,
    keeping only blocks whose mean gradient lies below a percentile threshold
    (homogeneous speckle **proxy**).

    If no block qualifies (e.g. very textured scene), falls back to a **global**
    ``mean^2 / var`` on the full image.
    """
    d = np.clip(np.asarray(denoised, dtype=np.float64), 1e-8, 1.0)
    h, w = d.shape
    bh = max(1, min(int(block), h))
    bw = max(1, min(int(block), w))
    g = gradient_magnitude(d)
    g_flat = g.ravel()
    thresh = float(np.percentile(g_flat, homo_percentile)) if g_flat.size else 0.0

    enl_list: list[float] = []
    for r in range(0, h - bh + 1, bh):
        for c in range(0, w - bw + 1, bw):
            patch = d[r : r + bh, c : c + bw]
            gp = g[r : r + bh, c : c + bw]
            if float(np.mean(gp)) > thresh:
                continue
            v = float(np.var(patch))
            m = float(np.mean(patch))
            if v < 1e-12:
                continue
            enl_list.append((m * m) / v)

    if not enl_list:
        v = float(np.var(d))
        m = float(np.mean(d))
        return float((m * m) / (v + 1e-12)) if v > 1e-12 else 1.0
    return float(np.median(enl_list))


def variance_metrics(denoised: np.ndarray) -> dict[str, float]:
    """Simple dispersion stats on the denoised image (``[0,1]`` domain)."""
    d = np.asarray(denoised, dtype=np.float64)
    var_d = float(np.var(d))
    std_d = float(np.std(d))
    dl = np.log(np.clip(d, 1e-8, 1.0))
    var_log = float(np.var(dl))
    return {
        "variance": var_d,
        "std": std_d,
        "variance_log": var_log,
    }


def compute_blind_qa(noisy: np.ndarray, denoised: np.ndarray) -> dict[str, float]:
    """
    Aggregate blind metrics for one pair ``(noisy, denoised)`` in ``[0, 1]``.

    Returns JSON-serializable floats.
    """
    vm = variance_metrics(denoised)
    return {
        "enl_homogeneous_median": blind_enl_median_homogeneous(denoised),
        "edge_preservation_vs_input": blind_edge_preservation_index(noisy, denoised),
        **vm,
    }
