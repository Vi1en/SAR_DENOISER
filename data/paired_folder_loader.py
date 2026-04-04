"""
Paired-folder dataset for cross-domain / OOD evaluation (Upgrade 2).

Layout::

    data_dir/
      clean/   # grayscale PNG/JPEG, basenames match noisy/
      noisy/

Same tensor contract as ``SAMPLESARDataset``: ``clean``, ``noisy``, ``noise_level`` per batch item.
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


def _list_images(folder: Path) -> list[Path]:
    exts = {".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"}
    files = [p for p in folder.iterdir() if p.suffix in exts]
    files.sort()
    return files


class PairedFolderPatchDataset(Dataset):
    """Aligned clean/noisy images under ``clean/`` and ``noisy/`` subfolders."""

    def __init__(
        self,
        data_dir: str | Path,
        *,
        patch_size: int = 128,
        augment: bool = False,
    ):
        self.data_dir = Path(data_dir)
        self.patch_size = patch_size
        self.augment = augment

        self.clean_dir = self.data_dir / "clean"
        self.noisy_dir = self.data_dir / "noisy"
        if not self.clean_dir.is_dir():
            raise FileNotFoundError(f"Missing clean/: {self.clean_dir}")
        if not self.noisy_dir.is_dir():
            raise FileNotFoundError(f"Missing noisy/: {self.noisy_dir}")

        clean_files = _list_images(self.clean_dir)
        self.pairs: list[tuple[Path, Path]] = []
        noisy_by_name = {p.name: p for p in _list_images(self.noisy_dir)}
        for c in clean_files:
            n = noisy_by_name.get(c.name)
            if n is not None:
                self.pairs.append((c, n))

        if not self.pairs:
            raise FileNotFoundError(
                f"No aligned clean/noisy pairs under {self.data_dir} (matching filenames)."
            )

        print(f"📁 Paired-folder: {len(self.pairs)} pairs from {self.data_dir}")

    def __len__(self) -> int:
        return len(self.pairs)

    def load_image(self, path: Path) -> np.ndarray:
        image = Image.open(path).convert("L")
        return np.array(image, dtype=np.float32) / 255.0

    def extract_patch(
        self, clean_img: np.ndarray, noisy_img: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        h, w = clean_img.shape
        ps = self.patch_size
        if h > ps:
            top = random.randint(0, h - ps)
        else:
            top = 0
        if w > ps:
            left = random.randint(0, w - ps)
        else:
            left = 0
        clean_patch = clean_img[top : top + ps, left : left + ps]
        noisy_patch = noisy_img[top : top + ps, left : left + ps]
        return clean_patch, noisy_patch

    def augment_pair(
        self, clean: np.ndarray, noisy: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        clean = clean.copy()
        noisy = noisy.copy()
        if random.random() > 0.5:
            clean = np.fliplr(clean)
            noisy = np.fliplr(noisy)
        if random.random() > 0.5:
            clean = np.flipud(clean)
            noisy = np.flipud(noisy)
        if random.random() > 0.5:
            k = random.randint(1, 3)
            clean = np.rot90(clean, k)
            noisy = np.rot90(noisy, k)
        return clean, noisy

    def __getitem__(self, idx: int) -> dict:
        clean_file, noisy_file = self.pairs[idx]
        clean_img = self.load_image(clean_file)
        noisy_img = self.load_image(noisy_file)

        if clean_img.shape != noisy_img.shape:
            min_h = min(clean_img.shape[0], noisy_img.shape[0])
            min_w = min(clean_img.shape[1], noisy_img.shape[1])
            clean_img = clean_img[:min_h, :min_w]
            noisy_img = noisy_img[:min_h, :min_w]

        if clean_img.shape[0] > self.patch_size or clean_img.shape[1] > self.patch_size:
            clean_patch, noisy_patch = self.extract_patch(clean_img, noisy_img)
        else:
            clean_patch, noisy_patch = clean_img, noisy_img

        if self.augment:
            clean_patch, noisy_patch = self.augment_pair(clean_patch, noisy_patch)

        clean_tensor = torch.from_numpy(clean_patch.copy()).float().unsqueeze(0)
        noisy_tensor = torch.from_numpy(noisy_patch.copy()).float().unsqueeze(0)
        noise_level = torch.tensor(np.std(noisy_patch - clean_patch), dtype=torch.float32)

        return {
            "clean": clean_tensor,
            "noisy": noisy_tensor,
            "noise_level": noise_level,
            "clean_path": str(clean_file),
            "noisy_path": str(noisy_file),
        }


def create_paired_eval_dataloader(
    data_dir: str | Path,
    *,
    batch_size: int = 16,
    patch_size: int = 128,
    num_workers: int = 4,
) -> DataLoader:
    """Single eval DataLoader (no train/val splits)."""
    ds = PairedFolderPatchDataset(data_dir, patch_size=patch_size, augment=False)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
