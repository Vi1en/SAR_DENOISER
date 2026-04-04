#!/usr/bin/env python3
"""Compare PyTorch vs ONNX outputs on random noise (parity check)."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--onnx", type=Path, default=Path("models_artifacts/denoiser.onnx"))
    parser.add_argument(
        "--model_type",
        type=str,
        default="unet",
        choices=["unet", "dncnn", "res_unet"],
    )
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-3)
    args = parser.parse_args()

    ckpt = args.checkpoint or Path(os.environ.get("SAR_CHECKPOINT", ""))
    if not ckpt.is_file():
        raise SystemExit("Need --checkpoint or SAR_CHECKPOINT")

    import torch
    from models.unet import create_model

    from inference.onnx_backend import ONNXDirectDenoiser

    model = create_model(args.model_type, n_channels=1, noise_conditioning=True)
    state = torch.load(ckpt, map_location="cpu")
    sd = state["model_state_dict"] if isinstance(state, dict) and "model_state_dict" in state else state
    model.load_state_dict(sd, strict=False)
    model.eval()

    if not args.onnx.is_file():
        raise SystemExit(f"Missing ONNX: {args.onnx} (run scripts/export_onnx.py first)")

    ort = ONNXDirectDenoiser(args.onnx)
    torch.manual_seed(0)
    x = torch.randn(1, 1, 128, 128)
    nl = torch.tensor([0.25], dtype=torch.float32)
    with torch.no_grad():
        pt = model(x, nl).numpy()
    onx = ort.run(x.numpy(), nl.numpy())
    ok = np.allclose(pt, onx, atol=args.atol, rtol=args.rtol)
    diff = float(np.max(np.abs(pt - onx)))
    print(f"max_abs_diff={diff:.6e} allclose={ok}")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
