"""
Task-oriented and structure-aware metrics beyond PSNR/SSIM for denoising evaluation.

These proxies measure edge/gradient alignment with the reference (clean) patch, which
correlates better with perceived structure preservation than intensity MSE alone.
"""
from __future__ import annotations

import numpy as np
from skimage import filters


def _to_float_gray(x: np.ndarray) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    if a.ndim > 2:
        a = np.mean(a, axis=0)
    return np.clip(a, 0.0, 1.0)


def gradient_magnitude(image: np.ndarray) -> np.ndarray:
    """Sobel gradient magnitude (structure edge map)."""
    g = _to_float_gray(image)
    gx = filters.sobel_h(g)
    gy = filters.sobel_v(g)
    return np.hypot(gx, gy)


def gradient_magnitude_correlation(clean: np.ndarray, denoised: np.ndarray) -> float:
    """
    Pearson correlation between flattened gradient magnitudes of clean vs denoised.
    Higher → structure (edge energy layout) more aligned with reference.
    """
    gc = gradient_magnitude(clean).ravel()
    gd = gradient_magnitude(denoised).ravel()
    if gc.size < 2 or np.std(gc) < 1e-12 or np.std(gd) < 1e-12:
        return 0.0
    c = np.corrcoef(gc, gd)[0, 1]
    return float(c) if not np.isnan(c) else 0.0


def edge_preservation_index(clean: np.ndarray, denoised: np.ndarray, eps: float = 1e-8) -> float:
    """
    EPI-style ratio: sum(Gc * Gd) / (sum(Gc^2) + eps), with G = gradient magnitude.
    Ideal structure-preserving denoising approaches 1 when gradients align in phase/magnitude.
    """
    gc = gradient_magnitude(clean)
    gd = gradient_magnitude(denoised)
    num = float(np.sum(gc * gd))
    den = float(np.sum(gc * gc) + eps)
    return num / den


def laplacian_mse_to_clean(clean: np.ndarray, denoised: np.ndarray) -> float:
    """
    Mean squared error between Laplacian(clean) and Laplacian(denoised).
    Lower → high-frequency structure closer to reference (penalizes blur vs clean).
    """
    g = _to_float_gray(clean)
    d = _to_float_gray(denoised)
    lc = filters.laplace(g)
    ld = filters.laplace(d)
    return float(np.mean((lc - ld) ** 2))


def gradient_ssim_proxy(clean: np.ndarray, denoised: np.ndarray) -> float:
    """
    SSIM on normalized gradient-magnitude maps (data_range=1.0 after min-max per patch).
    Complements pixel SSIM with a structure-focused view.
    """
    from skimage.metrics import structural_similarity

    gc = gradient_magnitude(clean)
    gd = gradient_magnitude(denoised)
    gc = (gc - gc.min()) / (gc.max() - gc.min() + 1e-8)
    gd = (gd - gd.min()) / (gd.max() - gd.min() + 1e-8)
    return float(structural_similarity(gc, gd, data_range=1.0))


def compute_task_metrics(
    clean: np.ndarray,
    denoised: np.ndarray,
    *,
    noisy: np.ndarray | None = None,
) -> dict[str, float]:
    """
    Aggregate task/structure metrics for one patch pair.

    ``noisy`` reserved for future extensions (e.g. speckle-normalized gradients).
    """
    _ = noisy
    return {
        "gsm_corr": gradient_magnitude_correlation(clean, denoised),
        "epi": edge_preservation_index(clean, denoised),
        "laplacian_mse": laplacian_mse_to_clean(clean, denoised),
        "grad_ssim": gradient_ssim_proxy(clean, denoised),
    }
