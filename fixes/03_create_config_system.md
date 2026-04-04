# Step 03: Create Config-Based Training (YAML)

## 1. Objective

- Move training hyperparameters and paths from **hardcoded Python** into **version-controlled YAML** files loaded at runtime into `TrainingConfig` (from Step 02).
- **Why:** Reproducible experiments, diffable changes, and CI can run `train --config configs/train/smoke.yaml` with tiny epochs.

## 2. Current Problem

- Hyperparameters live inside scripts; two runs are hard to compare without reading code.
- No single file to cite in a paper or report as “the config for Table 3.”

## 3. Scope of Changes

### New files

| Path | Purpose |
|------|---------|
| `configs/train/default.yaml` | Mirrors current `train_improved` defaults |
| `configs/train/sample.yaml` | SAMPLE dataset paths + patch settings |
| `configs/train/smoke.yaml` | `epochs: 1`, tiny batch for CI (Step 10) |

### New dependency

- Add to `requirements.txt`: `pyyaml>=6.0`

### New files (optional but recommended)

| Path | Purpose |
|------|---------|
| `trainers/config_loader.py` | `load_training_config(path: Path) -> TrainingConfig` |

### Modified files

| Path | Change |
|------|--------|
| `train_improved.py` | If `--config path.yaml` present, load YAML; else use `TrainingConfig()` defaults (backward compatible) |
| `train_sample.py` | Same optional `--config` |
| `trainers/config_dataclass.py` | Add `from_dict` / merge helper if needed |

## 4. Detailed Implementation Steps

1. **Install dependency**

   ```
   echo "pyyaml>=6.0" >> requirements.txt
   pip install pyyaml
   ```

2. **Create directory**

   ```bash
   mkdir -p configs/train
   ```

3. **Author `configs/train/default.yaml`** (example — tune to match real defaults)

   ```yaml
   seed: 42
   device: auto   # resolved to cuda/cpu in loader
   patch_size: 128
   batch_size: 8
   epochs: 100
   lr: 0.0001
   model_type: unet
   checkpoint_dir: checkpoints_improved
   use_sample_dataset: false
   data_dir: null
   ```

4. **Author `configs/train/sample.yaml`** (duplicate keys from `default.yaml` or use `!include` only if you add a loader for it; simplest is explicit keys)

   ```yaml
   seed: 42
   device: auto
   patch_size: 128
   batch_size: 8
   epochs: 100
   lr: 0.0001
   model_type: unet
   use_sample_dataset: true
   data_dir: data/sample_sar/processed
   checkpoint_dir: checkpoints_sample
   ```

5. **Implement `trainers/config_loader.py`**

   ```python
   from pathlib import Path
   import yaml
   from trainers.config_dataclass import TrainingConfig

   def load_training_yaml(path: Path) -> dict:
       with open(path) as f:
           return yaml.safe_load(f)

   def dict_to_training_config(d: dict) -> TrainingConfig:
       # map keys; handle device: auto -> torch.cuda.is_available()
       ...
   ```

6. **Wire CLI in `train_improved.py`**

   ```python
   import argparse
   from pathlib import Path
   from trainers.config_loader import load_training_yaml, dict_to_training_config

   parser = argparse.ArgumentParser()
   parser.add_argument("--config", type=Path, default=None)
   args = parser.parse_args()
   if args.config:
       raw = load_training_yaml(args.config)
       cfg = dict_to_training_config(raw)
   else:
       cfg = TrainingConfig()
   run_training(cfg)
   ```

7. **Document in README** (one subsection): “Training with config: `python train_improved.py --config configs/train/default.yaml`”

## 5. Code-Level Guidance

### BEFORE

```python
cfg = TrainingConfig(epochs=100, lr=1e-4)
```

### AFTER

```yaml
# configs/train/default.yaml
epochs: 100
lr: 0.0001
```

```python
cfg = dict_to_training_config(load_training_yaml(path))
```

### Device: auto

```python
if d.get("device") == "auto":
    import torch
    d["device"] = "cuda" if torch.cuda.is_available() else "cpu"
```

## 6. Safety Constraints (VERY IMPORTANT)

- **Without `--config`, behavior MUST match Step 02** (same defaults as prior hardcoded `TrainingConfig()`).
- **DO NOT** remove ability to run bare `python train_improved.py`.
- YAML must not contain executable code; use `yaml.safe_load` only.
- Paths in YAML should be **relative to repo root**; document that convention.

## 7. Testing & Verification

```bash
pip install pyyaml
python train_improved.py --config configs/train/smoke.yaml
```

**Expected:** Training starts, completes 1 epoch, writes checkpoint if logic allows.

```bash
python train_improved.py
```

**Expected:** Identical behavior to pre-YAML run (spot-check epochs and lr printed).

## 8. Rollback Plan

- Remove `--config` branches; delete `configs/` and `config_loader.py`; remove PyYAML from requirements if unused elsewhere.
- Git revert.

## 9. Result After This Step

- Every training run can be pinned to a YAML file (git SHA + config path).
- Ready for evaluation pipeline to log `config_path` in metrics JSON (Step 04).
