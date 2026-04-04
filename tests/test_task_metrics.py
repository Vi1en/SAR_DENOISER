"""Tests for evaluators.task_metrics structure/edge proxies."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluators.task_metrics import compute_task_metrics, edge_preservation_index, gradient_magnitude_correlation


def test_identical_patch_high_correlation():
    rng = np.random.default_rng(0)
    clean = rng.random((64, 64)).astype(np.float32)
    tm = compute_task_metrics(clean, clean.copy())
    assert tm["gsm_corr"] > 0.99
    assert tm["grad_ssim"] > 0.99
    assert abs(tm["epi"] - 1.0) < 0.05


def test_blur_lowers_gradient_correlation():
    rng = np.random.default_rng(1)
    clean = rng.random((48, 48)).astype(np.float32)
    blurred = np.asarray(
        __import__("scipy.ndimage", fromlist=["gaussian_filter"]).gaussian_filter(clean, sigma=3.0),
        dtype=np.float32,
    )
    c_same = gradient_magnitude_correlation(clean, clean)
    c_blur = gradient_magnitude_correlation(clean, blurred)
    assert c_same > c_blur


def test_epi_finite_on_noise():
    rng = np.random.default_rng(2)
    clean = rng.random((32, 32)).astype(np.float32)
    noisy = np.clip(clean + 0.2 * rng.standard_normal(clean.shape).astype(np.float32), 0, 1)
    e = edge_preservation_index(clean, noisy)
    assert np.isfinite(e)
