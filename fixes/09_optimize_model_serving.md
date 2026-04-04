# Step 09: Improve Model Serving (ONNX Export + Optional Triton)

## 1. Objective

- Export the **denoiser** (U-Net/DnCNN) to **ONNX** for faster CPU inference or deployment to **NVIDIA Triton** — **without** changing training code.
- ADMM loop may stay in PyTorch/Python initially; **only the `z`-update network** is exported first (most common PnP pattern).
- **Why:** Batching, TensorRT optimization, and reduced Python GIL contention in multi-worker setups.

## 2. Current Problem

- Inference requires full PyTorch stack per process; duplicate models across Streamlit, API, and RQ workers increase RAM footprint.
- No standardized artifact for edge deployment.

## 3. Scope of Changes

### New dependencies (dev / optional group)

```
onnx>=1.15.0
onnxruntime>=1.16.0  # or onnxruntime-gpu
```

### New files

| Path | Purpose |
|------|---------|
| `scripts/export_onnx.py` | Load `.pth`, `torch.onnx.export`, save `models_artifacts/denoiser.onnx` |
| `inference/onnx_backend.py` | `class ONNXDenoiser`: `__call__(x)` returns denoised tensor |
| `inference/service.py` | Add env `SAR_BACKEND=pytorch|onnx` to switch implementation |

### Modified files

| Path | Change |
|------|--------|
| `inference/service.py` | If `onnx`, wrap ONNX runtime in same interface used by ADMM |

### Not in scope (document as Phase B)

- Full ADMM in ONNX (hard); Triton **ensemble** mixing Python backend + ONNX — advanced.

## 4. Detailed Implementation Steps

1. **Create `scripts/export_onnx.py`**

   - Load `create_model` + `state_dict` same as training.
   - Build **dummy input** matching `NCHW` with `height=width=128` (or dynamic axes).
   - Export:

   ```python
   import torch
   from pathlib import Path
   from models.unet import create_model

   def main():
       ckpt = Path(os.environ["SAR_CHECKPOINT"])
       m = create_model("unet", n_channels=1, noise_conditioning=True)
       m.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=False)
       m.eval()
       dummy = torch.randn(1, 1, 128, 128)
       torch.onnx.export(
           m,
           dummy,
           "denoiser.onnx",
           input_names=["input"],
           output_names=["output"],
           dynamic_axes={"input": {0: "batch", 2: "h", 3: "w"}, "output": {0: "batch", 2: "h", 3: "w"}},
           opset_version=17,
       )
   ```

2. **Validate ONNX**

   ```bash
   python -m onnxruntime.tools.check_onnx_model_mobile_usability denoiser.onnx  # or onnx.checker
   ```

3. **Implement `inference/onnx_backend.py`**

   ```python
   import numpy as np
   import onnxruntime as ort

   class ONNXDenoiserWrapper:
       def __init__(self, path: str, providers=None):
           self.sess = ort.InferenceSession(path, providers=providers or ["CPUExecutionProvider"])

       def run(self, x: np.ndarray) -> np.ndarray:
           # x: 1x1xHxW float32
           out = self.sess.run(None, {"input": x})[0]
           return out
   ```

4. **Integrate into `SARDenoiseService`**

   - When `SAR_BACKEND=onnx`, `load_weights` loads ONNX session instead of `torch.load`.
   - **ADMMPnP** currently expects `torch.nn.Module` — two paths:
     - **Path A (recommended for Step 09):** Implement a **thin `torch.nn.Module`** that calls ORT inside `forward` (still PyTorch ADMM wrapping) — may not speed ADMM much.
     - **Path B:** Refactor `ADMMPnP` `z`-update to call `denoiser_fn(x_numpy)` — **interface injection** (larger change; do only if needed).

   **Minimal safe Step 09:** Export ONNX + verify numerical parity on **direct denoising path only**; document ADMM-PnP still uses PyTorch until interface refactor.

5. **Parity test script `scripts/compare_pytorch_onnx.py`**

   - Random tensor → PyTorch model vs ONNX → `np.allclose(atol=1e-4)`.

## 5. Code-Level Guidance

### BEFORE

```python
pred = self._denoiser(noisy_tensor)
```

### AFTER (direct path only)

```python
if self._backend == "onnx":
    arr = noisy_tensor.cpu().numpy()
    pred = torch.from_numpy(self._onnx.run(arr))
else:
    pred = self._denoiser(noisy_tensor)
```

## 6. Safety Constraints (VERY IMPORTANT)

- **MUST NOT** delete PyTorch weights; ONNX is **additive**.
- **MUST** default `SAR_BACKEND=pytorch` so all existing behavior unchanged.
- **Numerical parity:** document acceptable tolerance; speckle + ADMM may diverge slightly — capture in tests.

## 7. Testing & Verification

```bash
export SAR_CHECKPOINT=...
python scripts/export_onnx.py
python scripts/compare_pytorch_onnx.py
SAR_BACKEND=onnx SAR_ONNX_PATH=denoiser.onnx python -c "from inference.service import SARDenoiseService; ..."
```

**Triton (optional)**

- Place `denoiser.onnx` in Triton model repository with `config.pbtxt`; point a microservice client — only after ONNX validated.

## 8. Rollback Plan

- Unset `SAR_BACKEND`; remove ONNX files; remove optional deps.

## 9. Result After This Step

- Portable denoiser artifact for optimized serving and future full pipeline hardening.
