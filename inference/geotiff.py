"""
Windowed GeoTIFF denoising with preserved CRS and geotransform.

Per-tile min–max maps values to [0, 1] before ``denoise_fn``; outputs are mapped
back to the tile’s dynamic range. This does **not** auto-detect amplitude vs dB;
match preprocessing to your training pipeline.

Multiband / PolSAR: not supported (single band only). Overlap blending: use
``overlap=0`` (default); ``overlap > 0`` is rejected until Hann blending (Step 06b).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np

from inference.service import SARDenoiseService
from inference.types import DenoiseMethod


def make_tile_denoise_fn(
    service: SARDenoiseService,
    method: DenoiseMethod,
    **kwargs: Any,
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Build ``denoise_fn(tile) -> denoised_tile`` for :func:`denoise_geotiff`.

    ``kwargs`` are passed to :meth:`SARDenoiseService.denoise_numpy` (e.g.
    ``model_type``, ``max_iter``, ``direct_checkpoint``).
    """

    def fn(norm_tile: np.ndarray) -> np.ndarray:
        out = service.denoise_numpy(norm_tile, method, **kwargs)
        return np.asarray(out["denoised"], dtype=np.float32)

    return fn


def make_tile_denoise_fn_with_uncertainty(
    service: SARDenoiseService,
    method: DenoiseMethod,
    **kwargs: Any,
) -> Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """
    Like :func:`make_tile_denoise_fn` but returns ``(denoised_norm, uncertainty_norm)``.
    Requires **Direct Denoising** and a service call with ``return_uncertainty=True``.
    """

    kw = {**kwargs, "return_uncertainty": True}

    def fn(norm_tile: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        out = service.denoise_numpy(norm_tile, method, **kw)
        u = out.get("uncertainty")
        if u is None:
            raise ValueError(
                "No uncertainty map (use Direct Denoising and compatible checkpoint path)."
            )
        return (
            np.asarray(out["denoised"], dtype=np.float32),
            np.asarray(u, dtype=np.float32),
        )

    return fn


def _valid_mask(tile: np.ndarray, nodata: float | None) -> np.ndarray:
    v = np.isfinite(tile)
    if nodata is not None:
        if np.isnan(nodata):
            v = v & ~np.isnan(tile)
        else:
            v = v & (tile != nodata)
    return v


def denoise_geotiff(
    in_path: Path | str,
    out_path: Path | str,
    denoise_fn: Callable[[np.ndarray], np.ndarray],
    *,
    tile_size: int = 512,
    overlap: int = 0,
    compress: str | None = "deflate",
) -> None:
    """
    Read a single-band georeferenced GeoTIFF in windows, denoise each tile, write output.

    Parameters
    ----------
    denoise_fn
        Accepts float32 ``(H, W)`` with values roughly in ``[0, 1]`` (caller normalizes
        per tile); returns float32 ``(H, W)`` in the same normalized space before
        physical rescaling is applied here.
    overlap
        Must be ``0`` in this version (non-overlapping tiles). Positive overlap
        would require blending to avoid seams.
    """
    try:
        import rasterio
        from rasterio.windows import Window
    except ImportError as e:  # pragma: no cover - optional extra (requirements-full.txt)
        raise ImportError(
            "GeoTIFF support requires rasterio. Install locally with "
            "`pip install rasterio` or `pip install -r requirements-full.txt`."
        ) from e

    if overlap != 0:
        raise ValueError(
            "denoise_geotiff currently requires overlap=0 (non-overlapping tiles). "
            "Use tile_size to control window size; Hann blending is deferred to Step 06b."
        )

    in_path = Path(in_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(in_path) as src:
        if src.crs is None:
            raise ValueError("Input raster has no CRS; expected a georeferenced GeoTIFF.")
        if src.count != 1:
            raise ValueError(f"Expected a single-band raster, got count={src.count}.")

        nodata = src.nodata
        profile = src.profile.copy()
        profile.update(dtype=rasterio.float32, count=1)
        if compress:
            profile["compress"] = compress

        h, w = int(src.height), int(src.width)
        step = tile_size - overlap

        with rasterio.open(out_path, "w", **profile) as dst:
            for row in range(0, h, step):
                for col in range(0, w, step):
                    tw = min(tile_size, w - col)
                    th = min(tile_size, h - row)
                    if tw <= 0 or th <= 0:
                        continue
                    win = Window(col, row, tw, th)
                    tile = src.read(1, window=win).astype(np.float32)

                    valid = _valid_mask(tile, nodata)
                    if not np.any(valid):
                        dst.write(tile.astype(np.float32), 1, window=win)
                        continue

                    tmin = float(np.min(tile[valid]))
                    tmax = float(np.max(tile[valid]))
                    out = tile.copy()

                    if tmax > tmin:
                        norm = np.zeros_like(tile, dtype=np.float32)
                        norm[valid] = (tile[valid] - tmin) / (tmax - tmin)
                        denoised_norm = np.asarray(denoise_fn(norm), dtype=np.float32)
                        if denoised_norm.shape != tile.shape:
                            raise ValueError(
                                f"denoise_fn returned shape {denoised_norm.shape}, expected {tile.shape}"
                            )
                        out[valid] = denoised_norm[valid] * (tmax - tmin) + tmin
                    # constant or degenerate valid region: pass through
                    invalid = ~valid
                    if np.any(invalid):
                        out[invalid] = tile[invalid]

                    dst.write(out.astype(np.float32), 1, window=win)


def denoise_geotiff_two_band(
    in_path: Path | str,
    out_path: Path | str,
    denoise_fn_unc: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]],
    *,
    tile_size: int = 512,
    overlap: int = 0,
    compress: str | None = "deflate",
) -> None:
    """
    Two-band float32 GeoTIFF: band 1 = denoised (physical units), band 2 = TTA uncertainty
    scaled by the tile dynamic range ``(tmax - tmin)`` (same per-tile normalization as band 1).
    """
    try:
        import rasterio
        from rasterio.windows import Window
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "GeoTIFF support requires rasterio. Install locally with "
            "`pip install rasterio` or `pip install -r requirements-full.txt`."
        ) from e

    if overlap != 0:
        raise ValueError("denoise_geotiff_two_band requires overlap=0 (same as denoise_geotiff).")

    in_path = Path(in_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(in_path) as src:
        if src.crs is None:
            raise ValueError("Input raster has no CRS; expected a georeferenced GeoTIFF.")
        if src.count != 1:
            raise ValueError(f"Expected a single-band raster, got count={src.count}.")

        nodata = src.nodata
        profile = src.profile.copy()
        profile.update(dtype=rasterio.float32, count=2)
        if compress:
            profile["compress"] = compress

        h, w = int(src.height), int(src.width)
        step = tile_size - overlap

        with rasterio.open(out_path, "w", **profile) as dst:
            for row in range(0, h, step):
                for col in range(0, w, step):
                    tw = min(tile_size, w - col)
                    th = min(tile_size, h - row)
                    if tw <= 0 or th <= 0:
                        continue
                    win = Window(col, row, tw, th)
                    tile = src.read(1, window=win).astype(np.float32)

                    valid = _valid_mask(tile, nodata)
                    out_d = tile.copy()
                    out_u = np.zeros_like(tile, dtype=np.float32)

                    if not np.any(valid):
                        dst.write(out_d.astype(np.float32), 1, window=win)
                        dst.write(out_u.astype(np.float32), 2, window=win)
                        continue

                    tmin = float(np.min(tile[valid]))
                    tmax = float(np.max(tile[valid]))

                    if tmax > tmin:
                        scale = tmax - tmin
                        norm = np.zeros_like(tile, dtype=np.float32)
                        norm[valid] = (tile[valid] - tmin) / scale
                        denoised_norm, unc_norm = denoise_fn_unc(norm)
                        denoised_norm = np.asarray(denoised_norm, dtype=np.float32)
                        unc_norm = np.asarray(unc_norm, dtype=np.float32)
                        if denoised_norm.shape != tile.shape or unc_norm.shape != tile.shape:
                            raise ValueError("denoise_fn_unc returned wrong spatial shape")
                        out_d[valid] = denoised_norm[valid] * scale + tmin
                        out_u[valid] = unc_norm[valid] * scale
                    invalid = ~valid
                    if np.any(invalid):
                        out_d[invalid] = tile[invalid]
                        out_u[invalid] = 0.0

                    dst.write(out_d.astype(np.float32), 1, window=win)
                    dst.write(out_u.astype(np.float32), 2, window=win)
