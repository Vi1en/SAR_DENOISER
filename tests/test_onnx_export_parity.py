"""ONNX export + ORT parity (random weights; no real checkpoint required)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

from inference.onnx_backend import ONNXDirectDenoiser
from inference.onnx_export import export_denoiser_to_onnx
from models.unet import create_model


def test_unet_onnx_matches_pytorch(tmp_path: Path) -> None:
    m = create_model("unet", n_channels=1, noise_conditioning=True)
    m.eval()
    onx_path = tmp_path / "d.onnx"
    export_denoiser_to_onnx(m, onx_path, noise_conditioning=True, height=96, width=96)

    ort = ONNXDirectDenoiser(onx_path)
    torch.manual_seed(42)
    x = torch.randn(1, 1, 96, 96)
    nl = torch.tensor([0.18], dtype=torch.float32)
    with torch.no_grad():
        pt = m(x, nl).numpy()
    o = ort.run(x.numpy(), nl.numpy())
    assert np.allclose(pt, o, atol=1e-4, rtol=1e-3), float(np.max(np.abs(pt - o)))


def test_service_direct_onnx_env(tmp_path: Path, monkeypatch) -> None:
    m = create_model("unet", n_channels=1, noise_conditioning=True)
    m.eval()
    onx_path = tmp_path / "d.onnx"
    export_denoiser_to_onnx(m, onx_path, noise_conditioning=True, height=64, width=64)

    monkeypatch.setenv("SAR_BACKEND", "onnx")
    monkeypatch.setenv("SAR_ONNX_PATH", str(onx_path))

    from inference.service import SARDenoiseService

    svc = SARDenoiseService(device="cpu")
    noisy = np.random.rand(48, 48).astype(np.float32)
    out = svc.denoise_numpy(noisy, "Direct Denoising", model_type="U-Net", speckle_factor=0.2)
    assert out["denoised"].shape == noisy.shape


def test_service_direct_onnx_infer_kwargs_no_env(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("SAR_BACKEND", raising=False)
    monkeypatch.delenv("SAR_ONNX_PATH", raising=False)
    m = create_model("unet", n_channels=1, noise_conditioning=True)
    m.eval()
    onx_path = tmp_path / "d.onnx"
    export_denoiser_to_onnx(m, onx_path, noise_conditioning=True, height=64, width=64)

    from inference.service import SARDenoiseService

    svc = SARDenoiseService(
        device="cpu",
        infer_backend="onnx",
        onnx_path=onx_path,
    )
    noisy = np.random.rand(48, 48).astype(np.float32)
    out = svc.denoise_numpy(noisy, "Direct Denoising", model_type="U-Net", speckle_factor=0.2)
    assert out["denoised"].shape == noisy.shape
