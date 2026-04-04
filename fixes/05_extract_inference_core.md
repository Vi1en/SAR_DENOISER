# Step 05: Extract Core Inference Logic from Streamlit

## 1. Objective

- Move **model load + ADMM/direct/TV denoise + tensor/numpy I/O** into a **pure Python module** with **no Streamlit import**.
- `demo/streamlit_app.py` becomes a **thin UI** that calls this module.
- **Why:** Enables FastAPI (Step 07), batch CLIs, and unit tests without launching a browser.

## 2. Current Problem

- `demo/streamlit_app.py` is large and mixes: file upload, widgets, matplotlib, and inference.
- Cannot import “denoise this numpy array” from tests or workers without executing Streamlit.

## 3. Scope of Changes

### New files

| Path | Purpose |
|------|---------|
| `inference/__init__.py` | Package marker |
| `inference/service.py` | `SARDenoiseService` class or functions: `load_model`, `denoise_array`, `denoise_file` |
| `inference/types.py` | Optional `DenoiseRequest` dataclass (numpy path, method, admm kwargs) |

### Modified files

| Path | Change |
|------|--------|
| `demo/streamlit_app.py` | Replace inline load/denoise blocks with calls to `inference.service` |

### Not in scope

- Changing ADMM mathematics (`algos/admm_pnp.py`) except imports.
- GeoTIFF (Step 06).

## 4. Detailed Implementation Steps

1. **Identify extraction boundaries in `streamlit_app.py`**

   Search for:
   - `create_model`
   - `ADMMPnP`
   - `torch.load`
   - `denoise`
   - Checkpoint path resolution (`checkpoints_improved`, `best_model.pth`, etc.)

2. **Create `inference/service.py`**

   ```python
   from __future__ import annotations
   from pathlib import Path
   from typing import Any, Dict, Literal, Optional, Union
   import numpy as np
   import torch

   MethodName = Literal["ADMM-PnP-DL", "Direct Denoising", "TV Denoising"]

   class SARDenoiseService:
       def __init__(
           self,
           device: str = "cpu",
           checkpoint_dir: Optional[Path] = None,
       ):
           self.device = device
           self._denoiser: Optional[torch.nn.Module] = None
           self._model_type: Optional[str] = None

       def load_weights(
           self,
           model_type: str,
           checkpoint_path: Path,
       ) -> None:
           from models.unet import create_model
           net = create_model(model_type.lower(), n_channels=1, noise_conditioning=True)
           state = torch.load(checkpoint_path, map_location=self.device)
           net.load_state_dict(state, strict=False)  # match your actual checkpoint format
           net.to(self.device)
           net.eval()
           self._denoiser = net
           self._model_type = model_type

       def denoise_numpy(
           self,
           noisy: np.ndarray,
           method: MethodName,
           admm_max_iter: int = 15,
           rho_init: float = 1.0,
           alpha: float = 0.1,
           theta: float = 0.05,
           use_log_transform: bool = False,
       ) -> Dict[str, Any]:
           assert self._denoiser is not None, "Call load_weights first"
           # Port logic from streamlit_app.py verbatim
           ...
           return {"denoised": denoised_np, "meta": {...}}
   ```

3. **Copy-paste the denoising branch** from Streamlit into `denoise_numpy` **without** `st.` calls. Use `torch.no_grad()` where appropriate.

4. **Refactor `streamlit_app.py`**

   - After sidebar builds parameters, call:

   ```python
   from inference.service import SARDenoiseService

   @st.cache_resource
   def get_service(device_str):
       svc = SARDenoiseService(device=device_str)
       return svc
   ```

   - **Checkpoint path:** keep existing `st.sidebar` logic to pick path, then `service.load_weights(...)`.
   - **Important:** `@st.cache_resource` and mutating `load_weights` can conflict if users switch models in-session — either use `cache_resource` with a key that includes model path, or drop caching for `load_weights` and only cache device. Safer pattern:

   ```python
   service = SARDenoiseService(device=device)
   service.load_weights(model_type, checkpoint_path)
   ```

5. **Add unit test `tests/test_inference_service_smoke.py`** (optional in this step; Step 10 can formalize)

   ```python
   def test_service_import():
       from inference.service import SARDenoiseService
       SARDenoiseService(device="cpu")
   ```

## 5. Code-Level Guidance

### BEFORE (in `streamlit_app.py`)

```python
denoiser = create_model(...)
denoiser.load_state_dict(torch.load(...))
admm = ADMMPnP(denoiser, device=device, max_iter=max_iter, ...)
result = admm.denoise(...)
```

### AFTER (`streamlit_app.py`)

```python
from inference.service import SARDenoiseService

svc = SARDenoiseService(device=str(device))
svc.load_weights("unet", Path("checkpoints_improved/best_model.pth"))
out = svc.denoise_numpy(arr, method=method, admm_max_iter=max_iter, ...)
```

**Note:** Match your actual checkpoint loading (sometimes `state_dict` is nested under `'model'`).

## 6. Safety Constraints (VERY IMPORTANT)

- **Pixel output** for the same inputs and weights must match **pre-refactor** within floating tolerance (use `np.allclose` on a saved fixture in tests).
- **MUST NOT** remove Streamlit features (upload, sliders) — only **relocate** logic.
- Handle missing checkpoint gracefully: same error messages as before (or better).

## 7. Testing & Verification

```bash
python -c "from inference.service import SARDenoiseService; print('ok')"
streamlit run demo/streamlit_app.py
```

Manual: run one denoise with same settings before/after diff on output array (save PNG hash or `np.save`).

```bash
python - <<'PY'
import numpy as np
from inference.service import SARDenoiseService
# load small random array, skip if no checkpoint
PY
```

## 8. Rollback Plan

- Restore `streamlit_app.py` from git; delete `inference/` package.

## 9. Result After This Step

- Importable inference API for Step 06 (GeoTIFF wrapper) and Step 07 (FastAPI).
