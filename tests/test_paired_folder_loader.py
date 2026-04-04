"""Tests for paired-folder OOD loader (no full SAMPLE tree)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _write_gray_png(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    g = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(g).save(path)


def test_paired_loader_one_pair(tmp_path: Path):
    from data.paired_folder_loader import PairedFolderPatchDataset, create_paired_eval_dataloader

    clean = np.random.rand(64, 64).astype(np.float32)
    noisy = np.clip(clean + 0.05 * np.random.randn(64, 64).astype(np.float32), 0, 1)
    _write_gray_png(tmp_path / "clean" / "a.png", clean)
    _write_gray_png(tmp_path / "noisy" / "a.png", noisy)

    ds = PairedFolderPatchDataset(tmp_path, patch_size=32, augment=False)
    assert len(ds) == 1
    s = ds[0]
    assert s["clean"].shape == (1, 32, 32)
    assert s["noisy"].shape == (1, 32, 32)

    loader = create_paired_eval_dataloader(tmp_path, batch_size=1, patch_size=32, num_workers=0)
    batch = next(iter(loader))
    assert batch["clean"].shape[0] == 1
