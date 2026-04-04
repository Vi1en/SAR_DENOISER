# Step 10: Add CI/CD Pipeline (GitHub Actions)

## 1. Objective

- Add **continuous integration** that runs on every push/PR: **lint**, **import smoke tests**, **`verify_system.py`**, and a **fast training/eval smoke** using configs from Step 03/04.
- **Why:** Prevents regressions (especially doc paths, broken imports) and validates the upgrade path is real.

## 2. Current Problem

- No automated gate; documentation or path regressions could recur without CI; refactors break silently.

## 3. Scope of Changes

### New files

| Path | Purpose |
|------|---------|
| `.github/workflows/ci.yml` | Main CI workflow |
| `pyproject.toml` or keep `requirements.txt` | Optional: ruff config — or use `ruff.toml` |
| `tests/conftest.py` | Shared fixtures (optional) |

### Modified files

| Path | Change |
|------|--------|
| `requirements.txt` | Add **dev** extras in separate `requirements-dev.txt` **or** optional `[dev]` in `pyproject.toml` — recommended: `requirements-dev.txt` with `ruff`, `pytest` |

### New configs (if not exists)

| Path | Purpose |
|------|---------|
| `configs/train/smoke.yaml` | `epochs: 1`, tiny batch |
| `tests/test_imports.py` | Import `algos.admm_pnp`, `inference.service` (if Step 05 done), `api.main` (if Step 07 done) |

## 4. Detailed Implementation Steps

1. **Create `requirements-dev.txt`**

   ```
   -r requirements.txt
   ruff>=0.2.0
   pytest>=8.0.0
   ```

2. **Create `.github/workflows/ci.yml`**

   ```yaml
   name: CI

   on:
     push:
       branches: [main, master]
     pull_request:
       branches: [main, master]

   jobs:
     test:
       runs-on: ubuntu-latest
       steps:
         - uses: actions/checkout@v4

         - uses: actions/setup-python@v5
           with:
             python-version: "3.11"

         - name: Install dependencies
           run: |
             python -m pip install --upgrade pip
             pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
             pip install -r requirements-dev.txt

         - name: Ruff (lint)
           run: ruff check . --exclude .venv --exclude fixes

         - name: Verify system paths
           run: python verify_system.py

         - name: Pytest
           run: pytest -q --ignore=.venv
   ```

3. **Add `ruff.toml`** (minimal)

   ```toml
   line-length = 120
   target-version = "py311"
   exclude = [".venv", "fixes"]
   ```

4. **Add `tests/test_smoke_imports.py`**

   ```python
   def test_import_admm():
       from algos.admm_pnp import ADMMPnP
       assert ADMMPnP is not None

   def test_import_unet():
       from models.unet import create_model
       assert callable(create_model)
   ```

5. **Optional job: training smoke** (CPU, slow)

   - Only if `configs/train/smoke.yaml` exists and dataset is **not** required (use **synthetic single batch** test instead to avoid large downloads):

   ```python
   # tests/test_train_smoke_synthetic.py
   def test_one_batch_forward():
       import torch
       from models.unet import create_model
       m = create_model("unet", n_channels=1, noise_conditioning=True)
       x = torch.randn(1, 1, 64, 64)
       y = m(x)
       assert y.shape == x.shape
   ```

6. **CD (optional)**  
   - Deploy Streamlit / Docker image on tag — **out of scope** unless you add `Dockerfile` in a separate step; document placeholder job `deploy:` with `if: startsWith(github.ref, 'refs/tags/')`.

## 5. Code-Level Guidance

### BEFORE

- Manual `python verify_system.py` only.

### AFTER

- CI fails if:
  - Ruff errors (start with **warnings-only** or limit to `algos/`, `models/` if legacy debt is large).
  - pytest fails.
  - `verify_system.py` fails.

### Gradual ruff adoption

If `ruff check .` produces thousands of issues:

```yaml
- run: ruff check algos models data inference api --exit-zero
```

Then tighten to `--exit-nonzero` over time.

## 6. Safety Constraints (VERY IMPORTANT)

- **MUST NOT** run long GPU training in default CI (cost + flakiness).
- **MUST NOT** commit secrets; use GitHub **encrypted secrets** for future deploy.
- **Exclude** `.venv` and large `checkpoints/` from test discovery.
- **Dataset download** must not run on every PR unless cached — prefer **synthetic** tests.

## 7. Testing & Verification

Local:

```bash
pip install -r requirements-dev.txt
ruff check algos models
pytest -q
python verify_system.py
```

Push branch → confirm GitHub Actions green.

## 8. Rollback Plan

- Delete `.github/workflows/ci.yml`; remove dev requirements.

## 9. Result After This Step

- Automated safety net for the full refactor chain (Steps 01–09).
- Professional open-source / hiring signal for the repository.
