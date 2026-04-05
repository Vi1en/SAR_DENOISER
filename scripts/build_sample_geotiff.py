#!/usr/bin/env python3
"""
Write a tiny single-band GeoTIFF with CRS for Streamlit / presentation demos.

Requires rasterio (pip install rasterio or requirements-full.txt).

Prefers the first noisy SAMPLE patch PNG if present; otherwise writes a synthetic
speckled patch so CI and fresh clones can regenerate the asset.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
_OUT_DIR = _REPO_ROOT / "data" / "sample_geotiff"
_OUT_PATH = _OUT_DIR / "presentation_sample.tif"


def _load_from_sample_png() -> np.ndarray | None:
    noisy = (
        _REPO_ROOT / "data" / "sample_sar" / "processed" / "test_patches" / "noisy"
    )
    if not noisy.is_dir():
        return None
    pngs = sorted(p for p in noisy.iterdir() if p.suffix.lower() == ".png")
    if not pngs:
        return None
    try:
        import cv2
    except ImportError:
        return None
    arr = cv2.imread(str(pngs[0]), cv2.IMREAD_GRAYSCALE)
    if arr is None:
        return None
    return np.clip(arr.astype(np.float32) / 255.0, 0.0, 1.0)


def _synthetic_speckle(h: int = 256, w: int = 256, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = np.outer(
        np.linspace(0.25, 0.85, h, dtype=np.float32),
        np.linspace(0.2, 0.9, w, dtype=np.float32),
    )
    noise = rng.lognormal(0.0, 0.2, (h, w)).astype(np.float32)
    noise /= float(np.mean(noise)) + 1e-8
    return np.clip(base * noise, 0.0, 1.0).astype(np.float32)


def main() -> int:
    try:
        import rasterio
        from rasterio.crs import CRS
        from rasterio.transform import from_origin
    except ImportError:
        print("rasterio is required: pip install rasterio", file=sys.stderr)
        return 1

    img = _load_from_sample_png()
    source = "SAMPLE noisy PNG"
    if img is None:
        img = _synthetic_speckle()
        source = "synthetic speckle"

    h, w = img.shape
    # WGS84 top-left ~ representative patch; 0.00005° ≈ few metres at mid-latitudes
    transform = from_origin(77.2090, 28.6139, 0.00005, 0.00005)

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff",
        "height": h,
        "width": w,
        "count": 1,
        "dtype": rasterio.float32,
        "crs": CRS.from_epsg(4326),
        "transform": transform,
        "compress": "deflate",
    }

    with rasterio.open(_OUT_PATH, "w", **profile) as dst:
        dst.write(img.astype(np.float32), 1)

    print(f"Wrote {_OUT_PATH} ({w}×{h}, EPSG:4326) from {source}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
