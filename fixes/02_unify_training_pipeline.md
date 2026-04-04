# Step 02: Unify Training Pipeline (Single Entry, Legacy Wrappers)

## 1. Objective

- Provide **one canonical Python module** that performs training (the logic currently concentrated in `train_improved.py` / `train_sample.py`), callable from a **single entry script** without removing existing scripts.
- **Why:** Multiple top-level trainers (`train.py`, `train_simple.py`, `train_improved.py`, `train_sample.py`) confuse reproducibility; a unified core lets Step 03 (configs) attach cleanly while **old commands keep working** as thin wrappers.

## 2. Current Problem

- Training logic is **duplicated or forked** across files with different defaults, paths, and loss recipes.
- There is no single `if __name__ == "__main__"` story for “how we train for the report.”
- Import paths use `sys.path.append` inconsistently.

## 3. Scope of Changes

### New files

| Path | Purpose |
|------|---------|
| `trainers/pipeline.py` | `run_training(config: TrainingConfig)` — pure function / class using existing `ImprovedTrainer` or extracted loops |
| `trainers/config_dataclass.py` | `TrainingConfig` dataclass with defaults mirroring **current** `train_improved.py` / `train_sample.py` (no YAML yet) |

### Modified files

| Path | Change |
|------|--------|
| `train_improved.py` | Reduce to: build `TrainingConfig` from current argparse or constants → call `run_training()` |
| `train_sample.py` | Same pattern: set `data_dir`, `split` → `TrainingConfig` → `run_training()` |
| `train_simple.py` | Optional: delegate to `run_training` with `trainer_type="simple"` or leave untouched in this step if risk is high — **prefer** at least import shared utilities from `trainers/pipeline.py` for checkpoint path constants |

**Minimum viable Step 02:** Extract **only** the loop from `train_improved.py` into `trainers/pipeline.py` and call it from `train_improved.py`. Then make `train_sample.py` call the same `run_training` with SAMPLE-specific `data_dir`.

### Not in scope for this step

- YAML files (Step 03).
- Deleting `train.py` / `train_simple.py`.

## 4. Detailed Implementation Steps

1. **Read `train_improved.py` end-to-end** — identify: model creation, dataloaders, `ImprovedTrainer`, checkpoint dir, epochs, device.

2. **Create `trainers/config_dataclass.py`**

   ```python
   from dataclasses import dataclass, field
   from pathlib import Path
   from typing import Optional

   @dataclass
   class TrainingConfig:
       data_dir: Optional[Path] = None  # SAMPLE organized root
       use_sample_dataset: bool = False
       patch_size: int = 128
       batch_size: int = 8
       epochs: int = 100
       lr: float = 1e-4
       device: str = "cpu"
       checkpoint_dir: Path = Path("checkpoints_improved")
       model_type: str = "unet"  # or enum
       seed: int = 42
   ```

   - Adjust field names to match **actual** `train_improved.py` parameters.

3. **Create `trainers/pipeline.py`**

   - Function signature:

   ```python
   def run_training(config: TrainingConfig) -> None:
       ...
   ```

   - Move **verbatim** (copy-paste first, refactor second) the body from `train_improved.py`’s `main` or script level into this function.
   - Use `config.checkpoint_dir`, `config.device`, etc.
   - Preserve existing imports: `create_sample_dataloaders`, `create_model`, `ImprovedTrainer`.

4. **Refactor `train_improved.py`**

   ```python
   from trainers.config_dataclass import TrainingConfig
   from trainers.pipeline import run_training

   if __name__ == "__main__":
       cfg = TrainingConfig(
           use_sample_dataset=False,  # or whatever improved script used
           ...
       )
       run_training(cfg)
   ```

5. **Refactor `train_sample.py`**

   - Build `TrainingConfig(use_sample_dataset=True, data_dir=Path("..."))` from existing argparse in that file.
   - Call `run_training(cfg)`.

6. **Ensure `sys.path` / package imports**

   - If scripts run from repo root, `from trainers.pipeline import run_training` works if `trainers/` has `__init__.py` (add empty `trainers/__init__.py` if missing).

7. **Run both scripts** and confirm checkpoints still write to the same directories as before.

## 5. Code-Level Guidance

### BEFORE (`train_improved.py` conceptual)

```python
# 200 lines of setup + trainer loop inline
```

### AFTER

```python
from pathlib import Path
from trainers.config_dataclass import TrainingConfig
from trainers.pipeline import run_training

def main():
    cfg = TrainingConfig(
        checkpoint_dir=Path("checkpoints_improved"),
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    run_training(cfg)

if __name__ == "__main__":
    main()
```

### `trainers/pipeline.py` skeleton

```python
def run_training(config: TrainingConfig) -> None:
    import torch
    from data.sample_dataset_loader import create_sample_dataloaders
    from models.unet import create_model
    # from train_improved import ImprovedTrainer  # avoid circular import — move ImprovedTrainer
```

**Circular import avoidance:** Move class `ImprovedTrainer` from `train_improved.py` to `trainers/improved_trainer.py` and import it from both `train_improved.py` and `pipeline.py`. If that is too large for one step, **keep `ImprovedTrainer` in `train_improved.py`** and import it inside `run_training()`:

```python
def run_training(config: TrainingConfig) -> None:
    from train_improved import ImprovedTrainer  # temporary; clean in Step 03
```

Prefer moving `ImprovedTrainer` to `trainers/improved_trainer.py` in the same PR if you can test quickly.

## 6. Safety Constraints (VERY IMPORTANT)

- **MUST NOT** change default hyperparameters or checkpoint filenames compared to pre-refactor behavior (byte-for-byte optional; metric parity expected within floating noise).
- **MUST** keep `python train_improved.py` and `python train_sample.py` working.
- **DO NOT** delete `train_simple.py` or `train.py` in this step.
- If `ImprovedTrainer` is moved, update **all** imports that reference it.

## 7. Testing & Verification

```bash
# CPU smoke (short run): temporarily set epochs=1 in config for local test, then revert
python -c "from trainers.config_dataclass import TrainingConfig; from trainers.pipeline import run_training; ..."

# Or run full script after setting EPOCHS=1 via env if you add support
python train_improved.py
python train_sample.py   # requires dataset on disk
```

**Expected:** Same checkpoint paths; training loss prints; no new exceptions.

## 8. Rollback Plan

- Revert commit; restore `train_improved.py` / `train_sample.py` from git.
- Remove new `trainers/pipeline.py` if rollback.

## 9. Result After This Step

- Single implementation path for “serious” training flows.
- Step 03 can load YAML into `TrainingConfig` without touching multiple scripts again.
