#!/usr/bin/env python3
"""
CLI: windowed GeoTIFF denoising (CRS + transform preserved).

Example:
  python scripts/denoise_geotiff.py \\
    --in chip.tif --out chip_denoised.tif \\
    --checkpoint checkpoints_improved/best_model.pth \\
    --model_type unet --method "ADMM-PnP-DL"

TV denoising does not need --checkpoint.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Denoise a single-band GeoTIFF (windowed).")
    parser.add_argument("--in", dest="in_path", type=Path, required=True, help="Input GeoTIFF path")
    parser.add_argument("--out", dest="out_path", type=Path, required=True, help="Output GeoTIFF path")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Weights file (.pth). Required for ADMM-PnP-DL and Direct Denoising.",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="unet",
        choices=["unet", "dncnn", "res_unet", "U-Net", "DnCNN", "Res-U-Net"],
    )
    parser.add_argument(
        "--method",
        type=str,
        default="ADMM-PnP-DL",
        choices=["ADMM-PnP-DL", "Direct Denoising", "TV Denoising"],
    )
    parser.add_argument("--device", type=str, default="auto", help="cpu, cuda, or auto")
    parser.add_argument("--tile_size", type=int, default=512, help="Window size in pixels")
    parser.add_argument("--overlap", type=int, default=0, help="Must be 0 (non-overlapping tiles)")
    parser.add_argument("--max_iter", type=int, default=15)
    parser.add_argument("--rho_init", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--theta", type=float, default=0.05)
    parser.add_argument("--use_log_transform", action="store_true")
    parser.add_argument("--quality_enhancement", action="store_true")
    parser.add_argument("--speckle_factor", type=float, default=0.3)
    args = parser.parse_args()

    if args.method in ("ADMM-PnP-DL", "Direct Denoising") and args.checkpoint is None:
        parser.error(f"--checkpoint is required for method {args.method!r}")

    from api.infer_config import service_options_from_merged
    from inference.geotiff import denoise_geotiff, make_tile_denoise_fn
    from inference.service import SARDenoiseService

    device = args.device
    improved = args.checkpoint if args.method == "ADMM-PnP-DL" else None
    simple = args.checkpoint if args.method == "Direct Denoising" else None
    opts = service_options_from_merged()

    svc = SARDenoiseService(
        device=device,
        improved_checkpoint=improved,
        simple_checkpoint=simple,
        infer_backend=opts["infer_backend"],
        onnx_path=opts["onnx_path"],
    )

    mt = args.model_type.lower().replace("-", "").replace("_", "")
    if mt == "dncnn":
        model_type = "DnCNN"
    elif mt == "resunet":
        model_type = "Res-U-Net"
    else:
        model_type = "U-Net"

    kw = dict(
        model_type=model_type,
        max_iter=args.max_iter,
        rho_init=args.rho_init,
        alpha=args.alpha,
        theta=args.theta,
        use_log_transform=args.use_log_transform,
        quality_enhancement=args.quality_enhancement,
        speckle_factor=args.speckle_factor,
    )
    if args.method == "Direct Denoising" and args.checkpoint is not None:
        kw["direct_checkpoint"] = args.checkpoint

    denoise_fn = make_tile_denoise_fn(svc, args.method, **kw)

    denoise_geotiff(
        args.in_path,
        args.out_path,
        denoise_fn,
        tile_size=args.tile_size,
        overlap=args.overlap,
    )
    print(f"Wrote {args.out_path}")


if __name__ == "__main__":
    main()
