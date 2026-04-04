"""
Export U-Net / DnCNN denoiser to ONNX (Direct denoising path; ADMM still uses PyTorch).
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn


def export_denoiser_to_onnx(
    model: nn.Module,
    out_path: Path,
    *,
    noise_conditioning: bool = True,
    opset_version: int = 17,
    height: int = 128,
    width: int = 128,
) -> None:
    """Write ONNX model; validates with ``onnx.checker`` when available."""
    import onnx

    model.eval()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dummy_x = torch.randn(1, 1, height, width)
    if noise_conditioning:
        dummy_nl = torch.tensor([0.3], dtype=torch.float32)
        args: Tuple[torch.Tensor, ...] = (dummy_x, dummy_nl)
        input_names = ["input", "noise_level"]
        dynamic_axes = {
            "input": {0: "batch", 2: "h", 3: "w"},
            "noise_level": {0: "batch"},
            "output": {0: "batch", 2: "h", 3: "w"},
        }
    else:
        args = (dummy_x,)
        input_names = ["input"]
        dynamic_axes = {
            "input": {0: "batch", 2: "h", 3: "w"},
            "output": {0: "batch", 2: "h", 3: "w"},
        }

    # dynamo=False: classic exporter (PyTorch 2.5+ defaults to dynamo/onnxscript otherwise)
    torch.onnx.export(
        model,
        args,
        str(out_path),
        input_names=input_names,
        output_names=["output"],
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        do_constant_folding=True,
        dynamo=False,
    )
    onnx_model = onnx.load(str(out_path))
    onnx.checker.check_model(onnx_model)
