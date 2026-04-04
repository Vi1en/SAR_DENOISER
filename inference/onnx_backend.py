"""
ONNX Runtime wrapper for exported denoiser (Direct denoising only).

Environment
-----------
``SAR_ONNX_PATH``
    Path to ``.onnx`` file (also accepted: ``direct_checkpoint`` ending in ``.onnx``).
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

import numpy as np


class ONNXDirectDenoiser:
    """Runs a single forward; supports 1 or 2 inputs (``noise_level`` optional)."""

    def __init__(self, path: Path | str, providers: Sequence[str] | None = None):
        import onnxruntime as ort

        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"ONNX model not found: {path}")
        prov: List[str] = list(providers) if providers else ["CPUExecutionProvider"]
        self._sess = ort.InferenceSession(str(path), providers=prov)
        self._inputs = [i.name for i in self._sess.get_inputs()]

    def run(self, x: np.ndarray, noise_level: float | np.ndarray | None = None) -> np.ndarray:
        """
        Parameters
        ----------
        x
            Float32 array shaped ``(1, 1, H, W)`` or ``(H, W)`` (adds batch/channel).
        noise_level
            Scalar or shape ``(batch,)``; required if the model was exported with noise conditioning.
        """
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 2:
            x = x[np.newaxis, np.newaxis, ...]
        elif x.ndim == 3:
            x = x[np.newaxis, ...]

        feeds = {self._inputs[0]: x}
        if len(self._inputs) > 1:
            if noise_level is None:
                raise ValueError("Model expects noise_level input")
            nl = np.asarray(noise_level, dtype=np.float32).reshape(-1)
            b = x.shape[0]
            if nl.size == 1 and b > 1:
                nl = np.full((b,), nl[0], dtype=np.float32)
            elif nl.size != b:
                raise ValueError(f"noise_level batch {nl.size} != input batch {b}")
            feeds[self._inputs[1]] = nl

        out = self._sess.run(None, feeds)[0]
        return np.asarray(out, dtype=np.float32)
