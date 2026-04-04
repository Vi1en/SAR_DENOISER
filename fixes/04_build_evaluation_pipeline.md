# Step 04: Standardize Evaluation and Metrics Logging

## 1. Objective

- Every evaluation run must produce a **deterministic artifact bundle**: `metrics.json`, optional `summary.txt`, and a **run manifest** (git SHA, config, checkpoint path, timestamp).
- **Why:** Bridges research demos to production MLOps; enables regression detection in CI (Step 10).

## 2. Current Problem

- `evaluate_sample.py` writes to `results_sample` (or `save_dir`) but likely **without a single JSON schema** or run ID folder.
- `SARDenoisingEvaluator` in `algos/evaluation.py` stores results in memory (`self.results`) — persistence format may be ad hoc or plot-only.
- Hard to answer: “What exact command produced the README table?”

## 3. Scope of Changes

### New files

| Path | Purpose |
|------|---------|
| `evaluators/run_logger.py` | `EvaluationRunContext`: creates `results/runs/<run_id>/`, writes `manifest.json`, `metrics.json` |
| `configs/eval/sample_default.yaml` | Optional mirror of `evaluate_sample.py` argparse defaults (can be Step 04b) |

### Modified files

| Path | Change |
|------|--------|
| `evaluate_sample.py` | After evaluation, call `save_run_artifacts(...)` |
| `algos/evaluation.py` | Add optional helper `serialize_results(results: dict) -> dict` (JSON-safe floats/lists) — **additive** only |

### New directory convention

```
results/runs/<YYYYMMDD_HHMMSS>_<short_git_sha>/
  manifest.json
  metrics.json
  plots/   # optional, if existing code saves figures
```

## 4. Detailed Implementation Steps

1. **Implement `evaluators/run_logger.py`**

   ```python
   from __future__ import annotations
   import json
   import subprocess
   from datetime import datetime, timezone
   from pathlib import Path
   from typing import Any, Dict, Optional

   def get_git_sha_short(fallback: str = "nogit") -> str:
       try:
           return subprocess.check_output(
               ["git", "rev-parse", "--short", "HEAD"], text=True
           ).strip()
       except Exception:
           return fallback

   class EvaluationRunContext:
       def __init__(self, base_dir: Path = Path("results/runs")):
           sha = get_git_sha_short()
           ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
           self.run_id = f"{ts}_{sha}"
           self.run_dir = base_dir / self.run_id
           self.run_dir.mkdir(parents=True, exist_ok=True)

       def write_manifest(self, extra: Dict[str, Any]) -> None:
           manifest = {
               "run_id": self.run_id,
               "git_sha": get_git_sha_short(),
               **extra,
           }
           (self.run_dir / "manifest.json").write_text(
               json.dumps(manifest, indent=2), encoding="utf-8"
           )

       def write_metrics(self, metrics: Dict[str, Any]) -> None:
           (self.run_dir / "metrics.json").write_text(
               json.dumps(metrics, indent=2), encoding="utf-8"
           )
   ```

2. **Add JSON-safe conversion** in `algos/evaluation.py` (bottom, new function):

   ```python
   def results_to_jsonable(results: dict) -> dict:
       import numpy as np
       def convert(x):
           if isinstance(x, (np.floating, float)):
               return float(x)
           if isinstance(x, (np.integer, int)):
               return int(x)
           if isinstance(x, dict):
               return {k: convert(v) for k, v in x.items()}
           if isinstance(x, (list, tuple)):
               return [convert(i) for i in x]
           return x
       return convert(results)
   ```

3. **Patch `evaluate_sample.py` `main()`** near the end (after evaluation completes):

   ```python
   from evaluators.run_logger import EvaluationRunContext

   ctx = EvaluationRunContext()
   ctx.write_manifest({
       "script": "evaluate_sample.py",
       "data_dir": args.data_dir,
       "model_type": args.model_type,
       "methods": args.methods,
       "device": str(device),
   })
   # evaluator.results or aggregated dict — adapt to actual attribute names
   ctx.write_metrics(results_to_jsonable(evaluator.results))
   print(f"Saved run to {ctx.run_dir}")
   ```

4. **Inspect `SARDenoisingEvaluator`** for where aggregated metrics live after `compare_methods` or per-method loops; ensure you capture **mean PSNR/SSIM** etc. in one dict.

5. **Create empty `evaluators/__init__.py`** if needed.

6. **Document** in README: “Evaluation outputs under `results/runs/`.”

## 5. Code-Level Guidance

### BEFORE

```python
# Prints only or saves ad hoc plots
print("Final results:", evaluator.results)
```

### AFTER

```python
ctx = EvaluationRunContext()
ctx.write_manifest({...})
ctx.write_metrics(results_to_jsonable(evaluator.results))
```

### `metrics.json` shape (example)

```json
{
  "ADMM-PnP-DL": {"mean_psnr": 28.4, "mean_ssim": 0.82},
  "Direct-UNet": {"mean_psnr": 27.1, "mean_ssim": 0.79}
}
```

Adapt keys to match what `evaluator.results` actually contains after your code review.

## 6. Safety Constraints (VERY IMPORTANT)

- **MUST NOT** change metric **formulas** (PSNR/SSIM/ENL) in this step — only **logging**.
- **MUST** keep existing CLI flags on `evaluate_sample.py` working; new flags are optional (`--no-run-log` to disable logging if you need backward silent mode).
- If `git` is unavailable, manifest must still write with `git_sha: "nogit"`.

## 7. Testing & Verification

```bash
python evaluate_sample.py --methods unet --model_type unet --data_dir data/sample_sar/processed
# Use real path that exists on your machine
ls -la results/runs/
cat results/runs/*/manifest.json
cat results/runs/*/metrics.json
```

**Expected:** New directory per run; valid JSON; keys match evaluator output.

## 8. Rollback Plan

- Remove calls to `EvaluationRunContext` from `evaluate_sample.py`.
- Delete `evaluators/run_logger.py` if unused.

## 9. Result After This Step

- Reproducible evaluation artifacts for reports and CI baselines.
- Clean handoff to Step 05 (inference core) and Step 10 (CI smoke eval).
