"""GeoTIFF windowed denoise smoke test (requires rasterio)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

pytest.importorskip("rasterio")

import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin

from inference.geotiff import denoise_geotiff, make_tile_denoise_fn
from inference.service import SARDenoiseService


def _write_tiny_geotiff(path: Path, h: int = 64, w: int = 80) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    transform = from_origin(1000.0, 2000.0, 10.0, 10.0)
    data = (np.random.RandomState(0).rand(h, w).astype(np.float32) * 50 + 20.0)
    data[0, :] = -9999.0
    profile = {
        "driver": "GTiff",
        "height": h,
        "width": w,
        "count": 1,
        "dtype": "float32",
        "crs": CRS.from_epsg(32633),
        "transform": transform,
        "nodata": -9999.0,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data, 1)


def test_denoise_geotiff_tv_preserve_geo(tmp_path: Path) -> None:
    src_path = tmp_path / "in.tif"
    out_path = tmp_path / "out.tif"
    _write_tiny_geotiff(src_path)

    svc = SARDenoiseService(device="cpu")
    fn = make_tile_denoise_fn(svc, "TV Denoising", model_type="U-Net")
    denoise_geotiff(src_path, out_path, fn, tile_size=32, overlap=0, compress=None)

    with rasterio.open(src_path) as a, rasterio.open(out_path) as b:
        assert a.crs == b.crs
        assert a.transform == b.transform
        assert a.nodata == b.nodata
        assert b.count == 1


def test_overlap_rejected(tmp_path: Path) -> None:
    src_path = tmp_path / "in.tif"
    out_path = tmp_path / "out.tif"
    _write_tiny_geotiff(src_path)
    svc = SARDenoiseService(device="cpu")
    fn = make_tile_denoise_fn(svc, "TV Denoising", model_type="U-Net")
    with pytest.raises(ValueError, match="overlap=0"):
        denoise_geotiff(src_path, out_path, fn, tile_size=32, overlap=8)
