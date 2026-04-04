"""Smoke tests for ResUNet backbone."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_create_res_unet_forward():
    from models.unet import create_model

    m = create_model("res_unet", n_channels=1, noise_conditioning=False)
    m.eval()
    x = torch.randn(2, 1, 64, 64)
    y = m(x)
    assert y.shape == x.shape


def test_res_unet_noise_conditioning():
    from models.unet import create_model

    m = create_model("res_unet", n_channels=1, noise_conditioning=True)
    m.eval()
    x = torch.randn(1, 1, 32, 32)
    nl = torch.tensor([0.2])
    y = m(x, noise_level=nl)
    assert y.shape == x.shape


def test_state_keys_detect_as_res_unet():
    from inference.service import _detect_arch_from_state_keys
    from models.unet import create_model

    m = create_model("res_unet", n_channels=1, noise_conditioning=False)
    keys = list(m.state_dict().keys())
    assert _detect_arch_from_state_keys(keys) == "res_unet"
