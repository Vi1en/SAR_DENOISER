"""TV path in SARDenoisingEvaluator must run under torch.no_grad() wrapper."""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algos.admm_pnp import TVDenoiser
from algos.evaluation import SARDenoisingEvaluator


class _PatchDictDataset(Dataset):
    def __init__(self, clean: torch.Tensor, noisy: torch.Tensor, noise_level: torch.Tensor):
        self.clean = clean
        self.noisy = noisy
        self.noise_level = noise_level

    def __len__(self) -> int:
        return self.clean.shape[0]

    def __getitem__(self, i: int) -> dict:
        return {
            "clean": self.clean[i],
            "noisy": self.noisy[i],
            "noise_level": self.noise_level[i],
        }


def test_evaluate_method_tv_denoising_runs():
    torch.manual_seed(0)
    b, c, h, w = 2, 1, 64, 64
    clean = torch.rand(b, c, h, w)
    noisy = torch.clamp(clean + 0.05 * torch.randn(b, c, h, w), 0, 1)
    nl = torch.full((b,), 0.1)
    ds = _PatchDictDataset(clean, noisy, nl)
    # batch_size=1 so clean/denoised squeeze to 2D [H,W] for skimage SSIM
    loader = DataLoader(ds, batch_size=1, pin_memory=False)

    ev = SARDenoisingEvaluator(device=torch.device("cpu"))
    tv = TVDenoiser(device="cpu", max_iter=3, lambda_tv=0.1)
    ev.evaluate_method("TV Denoising", tv, loader, include_task_metrics=False)

    row = ev.results["TV Denoising"]
    assert "psnr_mean" in row and row["psnr_mean"] == row["psnr_mean"]
    assert row["psnr_std"] >= 0
    assert len(row["psnr_values"]) == b
