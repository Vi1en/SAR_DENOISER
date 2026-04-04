"""Smoke tests for inference.service (no Streamlit, no checkpoint required for TV)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_service_import():
    from inference.service import SARDenoiseService

    SARDenoiseService(device="cpu")


def test_preprocess_noisy_array():
    from inference.service import preprocess_noisy_array

    x = np.random.rand(32, 32).astype(np.float64) * 0.5
    y = preprocess_noisy_array(x)
    assert y.dtype == np.float32
    assert y.shape == x.shape
    assert float(y.max()) <= 1.0 + 1e-5


def test_tv_denoise_smoke():
    from inference.service import SARDenoiseService

    svc = SARDenoiseService(device="cpu")
    noisy = np.random.rand(48, 48).astype(np.float32)
    out = svc.denoise_numpy(noisy, "TV Denoising", model_type="U-Net")
    assert out["denoised"].shape == noisy.shape
    assert "messages" in out
    assert "energies" in out


def test_replay_messages_streamlit_no_crash():
    from inference.service import replay_messages_streamlit

    class FakeSt:
        def info(self, x):
            pass

        def success(self, x):
            pass

        def warning(self, x):
            pass

        def error(self, x):
            pass

    replay_messages_streamlit([("info", "a"), ("success", "b")], FakeSt())
