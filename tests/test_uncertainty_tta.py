"""TTA uncertainty (Direct path)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_tta_pytorch_shapes():
    from inference.uncertainty import tta_direct_pytorch
    from models.unet import create_model

    m = create_model("unet", n_channels=1, noise_conditioning=True)
    m.eval()
    device = torch.device("cpu")
    noisy = np.random.rand(16, 20).astype(np.float32)
    mean, std = tta_direct_pytorch(m, noisy, 0.2, device, passes=4)
    assert mean.shape == noisy.shape
    assert std.shape == noisy.shape
    assert np.all(std >= 0)


def test_service_direct_uncertainty():
    from inference.service import SARDenoiseService

    svc = SARDenoiseService(device="cpu")
    # U-Net needs H,W large enough for 4× maxpool (≥16; use 64 for margin).
    noisy = np.random.rand(64, 64).astype(np.float32)
    out = svc.denoise_numpy(
        noisy,
        "Direct Denoising",
        model_type="U-Net",
        speckle_factor=0.2,
        return_uncertainty=True,
        uncertainty_tta_passes=2,
    )
    assert "uncertainty" in out
    assert out["uncertainty"].shape == noisy.shape
    assert "uncertainty_mean" in out["meta"]


def test_uncertainty_to_vis_u8():
    from inference.uncertainty import uncertainty_to_vis_u8

    s = np.array([[0.0, 0.5], [0.1, 0.2]], dtype=np.float32)
    u8 = uncertainty_to_vis_u8(s)
    assert u8.shape == s.shape
    assert u8.dtype == np.uint8
