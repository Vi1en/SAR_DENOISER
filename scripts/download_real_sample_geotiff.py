#!/usr/bin/env python3
"""
Download a small, real Sentinel-1 SAR chip as a single-band GeoTIFF (with CRS).

Data source: **Microsoft Planetary Computer** — collection **sentinel-1-rtc**
(radiometric terrain corrected VV backscatter). STAC item is pinned below for
reproducibility; SAS URLs are obtained from Planetary Computer’s public sign API.

License: Copernicus Sentinel data (ESA); see data/sample_geotiff/ATTRIBUTION.txt

Requires: rasterio, network access.

Usage (from repo root):
    python scripts/download_real_sample_geotiff.py
    python scripts/download_real_sample_geotiff.py --chip 512 --out data/sample_geotiff/sentinel1_rtc_sample.tif
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]

# IW GRD RTC over land (Europe); VV gamma0 float32, projected CRS in product.
_DEFAULT_STAC_ITEM = (
    "https://planetarycomputer.microsoft.com/api/stac/v1/collections/"
    "sentinel-1-rtc/items/"
    "S1C_IW_GRDH_1SDV_20260331T171425_20260331T171450_007011_00E324_rtc"
)
_SIGN_API = "https://planetarycomputer.microsoft.com/api/sas/v1/sign"
_DEFAULT_OUT = _REPO_ROOT / "data" / "sample_geotiff" / "sentinel1_rtc_sample.tif"
_USER_AGENT = "SAR_DENOISER-download_real_sample_geotiff/1.0"


def _fetch_json(url: str, timeout: int = 120) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def _sign_blob_href(href: str) -> str:
    q = urllib.parse.urlencode({"href": href})
    data = _fetch_json(f"{_SIGN_API}?{q}")
    return str(data["href"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--stac-item",
        default=_DEFAULT_STAC_ITEM,
        help="STAC Item JSON URL (default: pinned Sentinel-1 RTC scene)",
    )
    parser.add_argument(
        "--asset",
        default="vv",
        help="Asset key (default: vv)",
    )
    parser.add_argument(
        "--chip",
        type=int,
        default=1024,
        help="Square chip size in pixels (default: 1024)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=_DEFAULT_OUT,
        help="Output GeoTIFF path",
    )
    args = parser.parse_args()

    try:
        import rasterio
        from rasterio.windows import Window, transform as win_transform
    except ImportError:
        print("rasterio is required: pip install rasterio", file=sys.stderr)
        return 1

    chip = int(args.chip)
    if chip < 64 or chip > 8192:
        print("--chip must be between 64 and 8192", file=sys.stderr)
        return 1

    try:
        feat = _fetch_json(args.stac_item)
        assets = feat.get("assets") or {}
        if args.asset not in assets:
            print(f"Asset {args.asset!r} not in item; have: {list(assets)}", file=sys.stderr)
            return 1
        href = assets[args.asset]["href"]
        signed = _sign_blob_href(href)
    except urllib.error.URLError as e:
        print(
            f"Network error ({e}). Check internet / firewall / SSL certificates.\n"
            "On macOS, try installing certificates for Python or use a normal shell.",
            file=sys.stderr,
        )
        return 1
    except (KeyError, json.JSONDecodeError) as e:
        print(f"Unexpected STAC response: {e}", file=sys.stderr)
        return 1

    out: Path = args.out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(signed) as src:
        if src.crs is None:
            print("Remote asset has no CRS; pick another STAC item.", file=sys.stderr)
            return 1
        h, w = src.height, src.width
        if h < chip or w < chip:
            print(f"Scene smaller than chip ({h}x{w} < {chip})", file=sys.stderr)
            return 1
        row = (h - chip) // 2
        col = (w - chip) // 2
        win = Window(col, row, chip, chip)
        data = src.read(1, window=win).astype("float32")
        tfm = win_transform(win, src.transform)
        profile = {
            "driver": "GTiff",
            "height": chip,
            "width": chip,
            "count": 1,
            "dtype": rasterio.float32,
            "crs": src.crs,
            "transform": tfm,
            "compress": "deflate",
            "predictor": 3,
        }
        item_id = feat.get("id", "unknown")
        with rasterio.open(out, "w", **profile) as dst:
            dst.write(data, 1)
            dst.update_tags(
                SOURCE="Microsoft Planetary Computer / Sentinel-1 RTC",
                STAC_ITEM=item_id,
                ASSET=str(args.asset),
                BAND="VV gamma0 backscatter (RTC) when asset=vv",
            )

    print(f"Wrote {out} ({chip}×{chip}, {out.stat().st_size // 1024} KiB)")
    print("Attribution: see data/sample_geotiff/ATTRIBUTION.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
