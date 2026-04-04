#!/usr/bin/env python3
"""
Export trained denoiser to ONNX for Direct denoising (``SAR_BACKEND=onnx``).

Example::

    export SAR_CHECKPOINT=checkpoints_simple/best_model.pth
    python scripts/export_onnx.py --out models_artifacts/denoiser.onnx

ADMM-PnP still uses PyTorch; only the Direct path uses ONNX at runtime.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Export U-Net/DnCNN to ONNX")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help=".pth with model_state_dict (default: SAR_CHECKPOINT env)",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="unet",
        choices=["unet", "dncnn", "res_unet"],
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("models_artifacts/denoiser.onnx"),
    )
    parser.add_argument(
        "--no-noise-conditioning",
        action="store_true",
        help="Single-input ONNX (must match how you call ONNXDirectDenoiser)",
    )
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    ckpt = args.checkpoint or Path(os.environ.get("SAR_CHECKPOINT", ""))
    if not ckpt or not ckpt.is_file():
        raise SystemExit("Set --checkpoint or SAR_CHECKPOINT to a valid .pth file")

    import torch
    from models.unet import create_model

    from inference.onnx_export import export_denoiser_to_onnx

    nc = not args.no_noise_conditioning
    model = create_model(args.model_type, n_channels=1, noise_conditioning=nc)
    state = torch.load(ckpt, map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"], strict=False)
    else:
        model.load_state_dict(state, strict=False)
    model.eval()

    export_denoiser_to_onnx(
        model,
        args.out,
        noise_conditioning=nc,
        opset_version=args.opset,
    )
    print(f"Wrote {args.out.resolve()}")


if __name__ == "__main__":
    main()
