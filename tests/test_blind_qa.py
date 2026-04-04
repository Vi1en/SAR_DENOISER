"""No-reference blind QA metrics."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

pytest.importorskip("skimage")

from evaluators.blind_qa import compute_blind_qa


def test_compute_blind_qa_shapes_and_keys():
    rng = np.random.RandomState(0)
    noisy = rng.rand(64, 64).astype(np.float32)
    denoised = np.clip(noisy * 0.95 + 0.02, 0.0, 1.0).astype(np.float32)
    out = compute_blind_qa(noisy, denoised)
    assert set(out.keys()) == {
        "enl_homogeneous_median",
        "edge_preservation_vs_input",
        "variance",
        "std",
        "variance_log",
    }
    for v in out.values():
        assert isinstance(v, float)
        assert np.isfinite(v)


def test_blind_qa_small_image():
    rng = np.random.RandomState(1)
    noisy = rng.rand(8, 8).astype(np.float32)
    denoised = np.clip(noisy * 0.9 + 0.05, 0.0, 1.0).astype(np.float32)
    out = compute_blind_qa(noisy, denoised)
    assert out["edge_preservation_vs_input"] >= 0.0
    assert np.isfinite(out["enl_homogeneous_median"])
