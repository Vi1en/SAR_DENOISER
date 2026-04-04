# Frozen baseline (SAMPLE, TV reference)

This folder holds a **small, reproducible** evaluation snapshot for reports and regression checks.

## What is pinned

- **Data layout:** SAMPLE `data/sample_sar/processed` (official test split from `create_sample_dataloaders`).
- **Method:** **TV Denoising** only (no learned checkpoint; cheap and stable across clones).
- **Metrics file:** `metrics.json` — summary statistics plus **provenance** (`git_sha`, capture time, source path).

For full method sweeps (U-Net, ADMM, etc.), use `evaluate_sample.py` and `results_sample/` or ablation manifests (`fixes/updates.md` Update 22).

## Regenerate `metrics.json`

From the repository root (after SAMPLE data is present):

```bash
export PYTHONHASHSEED=0
python evaluate_sample.py \
  --data_dir data/sample_sar/processed \
  --data_layout sample \
  --dataset_tag baseline_tv \
  --methods tv \
  --save_dir results/baseline/staging \
  --no-run-log \
  --no-task-metrics \
  --batch_size 32 \
  --device cpu
python scripts/capture_baseline.py \
  --baseline-id sample_tv_v1 \
  --command 'PYTHONHASHSEED=0 python evaluate_sample.py --data_dir data/sample_sar/processed --data_layout sample --dataset_tag baseline_tv --methods tv --save_dir results/baseline/staging --no-run-log --no-task-metrics --batch_size 32 --device cpu'
```

Then commit `results/baseline/metrics.json` (and this README if the procedure changes).

`results/baseline/staging/` is regenerable intermediate output; it is listed in `.gitignore`.

## Note on batching

The SAMPLE **test** split is small (order of tens of patches). Metrics are aggregated **per evaluation batch** as implemented in `algos/evaluation.py`; for a single batch you may see one aggregate row. The baseline is still useful as a **process and numerics** anchor; expand `--methods` or data for paper-scale tables.
