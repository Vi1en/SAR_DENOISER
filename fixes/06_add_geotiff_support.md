# Step 06: Add GeoTIFF / Geospatial I/O Support

## 1. Objective

- Add an **optional** code path to read a **single-band GeoTIFF**, denoise **windowed tiles** (with overlap), and write a **GeoTIFF** output with **preserved CRS and geotransform**.
- **Why:** Production SAR workflows use GeoTIFF/COG, not PNG uploads.

## 2. Current Problem

- Pipeline assumes numpy arrays or PNG from PIL/OpenCV; **georeferencing is lost** on save.
- Large rasters do not fit in GPU memory at once.

## 3. Scope of Changes

### New dependencies

Add to `requirements.txt`:

```
rasterio>=1.3.0
```

Optional (for future):

```
rioxarray>=0.15.0
```

### New files

| Path | Purpose |
|------|---------|
| `inference/geotiff.py` | `denoise_geotiff(input_path, output_path, service: SARDenoiseService, tile=512, overlap=32, ...)` |
| `scripts/denoise_geotiff.py` | CLI: `python scripts/denoise_geotiff.py --in a.tif --out b.tif --checkpoint ...` |

### Modified files

| Path | Change |
|------|--------|
| `README.md` | New subsection: GeoTIFF usage (optional feature flag) |

### Not in scope

- Multiband polSAR (document as future).
- COG overviews (`rio cogeo`) — optional note.

## 4. Detailed Implementation Steps

1. **Install**

   ```bash
   pip install rasterio
   ```

2. **Implement windowed read/write in `inference/geotiff.py`**

   ```python
   from __future__ import annotations
   import numpy as np
   import rasterio
   from rasterio.windows import Window
   from rasterio.transform import Affine
   from pathlib import Path
   from typing import Callable

   def denoise_geotiff(
       in_path: Path,
       out_path: Path,
       denoise_fn: Callable[[np.ndarray], np.ndarray],
       tile_size: int = 512,
       overlap: int = 32,
   ) -> None:
       """denoise_fn: float32 HxW [0,1] or raw — document contract."""
       with rasterio.open(in_path) as src:
           if src.count != 1:
               raise ValueError(f"Expected 1 band, got {src.count}")
           profile = src.profile.copy()
           profile.update(dtype=rasterio.float32, count=1, compress="deflate")

           h, w = src.height, src.width
           with rasterio.open(out_path, "w", **profile) as dst:
               step = tile_size - overlap
               for row in range(0, h, step):
                   for col in range(0, w, step):
                       win = Window(col, row, min(tile_size, w - col), min(tile_size, h - row))
                       if win.width <= 0 or win.height <= 0:
                           continue
                       tile = src.read(1, window=win).astype(np.float32)
                       # Normalize to model range — MUST match training (document)
                       tmin, tmax = tile.min(), tile.max()
                       if tmax > tmin:
                           norm = (tile - tmin) / (tmax - tmin)
                       else:
                           norm = tile
                       out = denoise_fn(norm)
                       # Rescale back if you scaled — match streamlit behavior
                       if tmax > tmin:
                           out = out * (tmax - tmin) + tmin
                       dst.write(out.astype(np.float32), 1, window=win)
   ```

3. **Blend overlaps (recommended follow-up)**  
   The snippet above **overwrites** overlaps naively. **Step 06 minimal:** use **non-overlapping tiles** first (`overlap=0`) to avoid seams — document limitation. **Step 06b:** add Hann window blending in overlap region.

4. **Wire `SARDenoiseService` from Step 05**

   ```python
   def make_fn(service, method, **kw):
       def fn(norm_tile):
           r = service.denoise_numpy(norm_tile, method=method, **kw)
           return r["denoised"]
       return fn
   ```

5. **CLI `scripts/denoise_geotiff.py`**

   - Parse `--in`, `--out`, `--checkpoint`, `--model_type`, `--method`, ADMM args.
   - Construct `SARDenoiseService`, `load_weights`, `denoise_geotiff`.

6. **Normalization contract**

   - Add a short docstring: SAR GeoTIFFs may be amplitude or dB; **user must choose** preprocessing consistent with training. This step **does not** auto-detect product type.

## 5. Code-Level Guidance

### BEFORE

- Only PNG/Streamlit upload path.

### AFTER

```bash
python scripts/denoise_geotiff.py \
  --in /path/to/s1_amplitude.tif \
  --out /path/to/denoised.tif \
  --checkpoint checkpoints_improved/best_model.pth \
  --model_type unet \
  --method "ADMM-PnP-DL"
```

### CRS check

```python
assert src.crs is not None, "Input must be georeferenced"
```

## 6. Safety Constraints (VERY IMPORTANT)

- **MUST NOT** break existing PNG/Streamlit flow; GeoTIFF is **additive**.
- **MUST** default dtype and nodata: if `src.nodata` is set, mask and **do not** corrupt nodata regions (pass through or skip denoise for nodata pixels).
- Large files: process **streaming** — never `read()` full raster if dimensions exceed threshold.

## 7. Testing & Verification

```bash
pip install rasterio
# Create tiny GeoTIFF with rasterio (one-time fixture script) or use real chip
python scripts/denoise_geotiff.py --in tests/fixtures/tiny.tif --out /tmp/out.tif ...
python - <<'PY'
import rasterio
with rasterio.open("tests/fixtures/tiny.tif") as a, rasterio.open("/tmp/out.tif") as b:
    assert a.crs == b.crs
    assert a.transform == b.transform
PY
```

## 8. Rollback Plan

- Remove `scripts/denoise_geotiff.py` and `inference/geotiff.py`; remove rasterio from requirements if unused.

## 9. Result After This Step

- First **production-shaped** I/O path for real EO tiles.
- FastAPI (Step 07) can call the same `denoise_geotiff` or tile fn.
