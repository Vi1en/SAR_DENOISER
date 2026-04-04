"""
Test-time augmentation (TTA) stack for cheap pixelwise uncertainty on **Direct** denoising.

Uses multiple flips/rotations of the input; **std across transformed predictions** (after
inverse geometry) is a proxy for epistemic spread. No dropout or retraining required.
"""
from __future__ import annotations

from typing import Callable, List, Tuple

import numpy as np
import torch


def _tta_transforms_4() -> List[Tuple[str, Callable[[torch.Tensor], torch.Tensor], Callable[[torch.Tensor], torch.Tensor]]]:
    """Identity, hflip, vflip, rot90 (and inverses on output [1,1,H,W])."""

    def id_(x: torch.Tensor) -> torch.Tensor:
        return x

    def inv_id(y: torch.Tensor) -> torch.Tensor:
        return y

    def hf(x: torch.Tensor) -> torch.Tensor:
        return torch.flip(x, [3])

    def inv_hf(y: torch.Tensor) -> torch.Tensor:
        return torch.flip(y, [3])

    def vf(x: torch.Tensor) -> torch.Tensor:
        return torch.flip(x, [2])

    def inv_vf(y: torch.Tensor) -> torch.Tensor:
        return torch.flip(y, [2])

    def r90(x: torch.Tensor) -> torch.Tensor:
        return torch.rot90(x, 1, [2, 3])

    def inv_r90(y: torch.Tensor) -> torch.Tensor:
        return torch.rot90(y, 3, [2, 3])

    return [
        ("id", id_, inv_id),
        ("h", hf, inv_hf),
        ("v", vf, inv_vf),
        ("r90", r90, inv_r90),
    ]


def _forward_pytorch(
    denoiser: torch.nn.Module,
    x: torch.Tensor,
    noise_level: torch.Tensor,
) -> torch.Tensor:
    denoiser.eval()
    with torch.no_grad():
        if hasattr(denoiser, "noise_conditioning") and denoiser.noise_conditioning:
            return denoiser(x, noise_level)
        return denoiser(x)


def tta_direct_pytorch(
    denoiser: torch.nn.Module,
    noisy_hw: np.ndarray,
    speckle_factor: float,
    device: torch.device,
    *,
    passes: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (mean_denoised_HW, std_HW) float32 in same spatial shape as ``noisy_hw``.
    """
    t0 = (
        torch.from_numpy(noisy_hw.astype(np.float32))
        .unsqueeze(0)
        .unsqueeze(0)
        .to(device)
    )
    nl = torch.tensor(speckle_factor, device=device, dtype=t0.dtype)
    transforms = _tta_transforms_4()
    if passes < 1:
        passes = 1
    transforms = transforms[: min(passes, len(transforms))]

    acc: list[np.ndarray] = []
    for _, t_in, t_out in transforms:
        xt = t_in(t0)
        y = _forward_pytorch(denoiser, xt, nl)
        y = t_out(y)
        acc.append(y.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32))

    stack = np.stack(acc, axis=0)
    mean = np.mean(stack, axis=0)
    std = np.std(stack, axis=0)
    return mean.astype(np.float32), std.astype(np.float32)


def _tta_numpy_4():
    return [
        ("id", lambda x: x, lambda y: y),
        ("h", np.fliplr, np.fliplr),
        ("v", np.flipud, np.flipud),
        ("r90", lambda x: np.rot90(x, k=1), lambda y: np.rot90(y, k=3)),
    ]


def tta_direct_onnx(
    ort_d: object,
    noisy_hw: np.ndarray,
    speckle_factor: float,
    *,
    passes: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """TTA for :class:`~inference.onnx_backend.ONNXDirectDenoiser` (numpy HW paths)."""
    x0 = np.asarray(noisy_hw, dtype=np.float32)
    transforms = _tta_numpy_4()
    if passes < 1:
        passes = 1
    transforms = transforms[: min(passes, len(transforms))]

    acc: list[np.ndarray] = []
    for _, t_in, t_out in transforms:
        xn = np.ascontiguousarray(t_in(x0).astype(np.float32))
        y = ort_d.run(xn, speckle_factor)
        y = np.asarray(y, dtype=np.float32).squeeze()
        if y.ndim != 2:
            y = y.reshape(y.shape[-2], y.shape[-1])
        y_inv = np.ascontiguousarray(t_out(y).astype(np.float32))
        acc.append(y_inv)

    stack = np.stack(acc, axis=0)
    return np.mean(stack, axis=0).astype(np.float32), np.std(stack, axis=0).astype(np.float32)


def uncertainty_to_vis_u8(std_hw: np.ndarray, percentile: float = 99.0) -> np.ndarray:
    """Map std to uint8 heatmap (robust to outliers)."""
    s = np.asarray(std_hw, dtype=np.float64)
    flat = s[np.isfinite(s)]
    if flat.size == 0:
        return np.zeros_like(s, dtype=np.uint8)
    hi = float(np.percentile(flat, percentile))
    if hi <= 1e-12:
        hi = 1e-12
    g = np.clip(s / hi, 0.0, 1.0)
    return (g * 255.0).astype(np.uint8)
