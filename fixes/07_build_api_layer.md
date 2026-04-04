# Step 07: Build API Layer (FastAPI)

## 1. Objective

- Expose **stateless HTTP endpoints** for health check and **synchronous** denoise of an uploaded image (numpy-friendly) using `SARDenoiseService` (Step 05).
- **Why:** Decouples UI from inference; enables mobile clients, automation, and later job queue (Step 08).

## 2. Current Problem

- Only Streamlit can trigger inference; no standard JSON/multipart contract for integrations.

## 3. Scope of Changes

### New dependencies

```
fastapi>=0.109.0
uvicorn[standard]>=0.27.0
python-multipart>=0.0.6
```

### New files

| Path | Purpose |
|------|---------|
| `api/__init__.py` | Package |
| `api/main.py` | FastAPI app: `GET /health`, `POST /v1/denoise` |
| `api/deps.py` | Singleton or lifespan-loaded `SARDenoiseService` |

### Modified files

| Path | Change |
|------|--------|
| `README.md` | Section: “Run API: `uvicorn api.main:app --reload`” |

### Not in scope

- Redis queue (Step 08).
- Authentication (document as TODO or add simple API key env in Step 10).

## 4. Detailed Implementation Steps

1. **Add dependencies** to `requirements.txt` and install.

2. **Create `api/deps.py`**

   ```python
   from pathlib import Path
   import os
   from inference.service import SARDenoiseService

   _service: SARDenoiseService | None = None

   def get_service() -> SARDenoiseService:
       global _service
       if _service is None:
           device = os.environ.get("SAR_DEVICE", "cpu")
           ckpt = Path(os.environ["SAR_CHECKPOINT"])
           model = os.environ.get("SAR_MODEL_TYPE", "unet")
           _service = SARDenoiseService(device=device)
           _service.load_weights(model, ckpt)
       return _service
   ```

3. **Create `api/main.py`**

   ```python
   from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
   from fastapi.responses import Response
   import numpy as np
   from PIL import Image
   import io
   from api.deps import get_service

   app = FastAPI(title="SAR Denoise API", version="0.1.0")

   @app.get("/health")
   def health():
       return {"status": "ok"}

   @app.post("/v1/denoise")
   async def denoise(
       file: UploadFile = File(...),
       method: str = "ADMM-PnP-DL",
       max_iter: int = 15,
   ):
       data = await file.read()
       img = np.array(Image.open(io.BytesIO(data)).convert("L")).astype(np.float32) / 255.0
       try:
           svc = get_service()
           out = svc.denoise_numpy(
               img,
               method=method,  # type: ignore
               admm_max_iter=max_iter,
           )
       except Exception as e:
           raise HTTPException(500, detail=str(e))
       den = (np.clip(out["denoised"], 0, 1) * 255).astype(np.uint8)
       buf = io.BytesIO()
       Image.fromarray(den).save(buf, format="PNG")
       return Response(content=buf.getvalue(), media_type="image/png")
   ```

4. **Environment variables**

   | Variable | Required | Description |
   |----------|----------|-------------|
   | `SAR_CHECKPOINT` | Yes | Path to `.pth` |
   | `SAR_DEVICE` | No | `cpu` or `cuda` |
   | `SAR_MODEL_TYPE` | No | `unet` / `dncnn` |

5. **Run locally**

   ```bash
   export SAR_CHECKPOINT=checkpoints_improved/best_model.pth
   uvicorn api.main:app --host 0.0.0.0 --port 8000
   ```

6. **Optional:** OpenAPI docs at `/docs` (FastAPI default).

## 5. Code-Level Guidance

### BEFORE

- `streamlit run demo/streamlit_app.py` only.

### AFTER

- Streamlit unchanged; API is **parallel** entry.

### Method parameter validation

Use `Literal` or `Enum` in FastAPI:

```python
from enum import Enum
class Method(str, Enum):
    admm = "ADMM-PnP-DL"
    direct = "Direct Denoising"
    tv = "TV Denoising"
```

## 6. Safety Constraints (VERY IMPORTANT)

- **MUST NOT** remove or alter Streamlit app behavior.
- **MUST** cap upload size (middleware or `Starlette` limit) to avoid OOM — e.g. max 20MB in first version.
- **Synchronous** endpoint: timeout risk for large images — document max dimensions; Step 08 moves heavy work async.

## 7. Testing & Verification

```bash
export SAR_CHECKPOINT=...
uvicorn api.main:app --port 8000 &
curl -s http://127.0.0.1:8000/health
curl -s -F "file=@tests/fixtures/small.png" -o /tmp/out.png http://127.0.0.1:8000/v1/denoise
```

**Expected:** `{"status":"ok"}`; PNG written.

## 8. Rollback Plan

- Delete `api/` directory; remove FastAPI deps if unused.

## 9. Result After This Step

- Machine-readable integration surface.
- Ready for Redis job IDs in Step 08.
