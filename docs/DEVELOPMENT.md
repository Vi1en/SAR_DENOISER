# Development quickstart

Single reference for **training**, **evaluation**, and **checks** on a fresh clone.

## Unified CLI (`sar.py`)

Thin wrapper around existing scripts (no extra dependencies):

```bash
python sar.py train --config configs/train/default.yaml
python sar.py eval --no-run-log
python sar.py verify
python sar.py train-sample --help
python sar.py capture-baseline --help
python sar.py export-onnx --help
python sar.py compare-onnx --help
python sar.py denoise-geotiff --help
python sar.py ablation-grid --help
python sar.py ablation-md --help
python sar.py api -- --host 127.0.0.1 --port 8000
python sar.py worker -- --url redis://localhost:6379/0
python sar.py streamlit
python sar.py streamlit -- --server.port 8502
```

Run `python sar.py` with no arguments to list commands. **`api`** runs **`uvicorn api.main:app`**; **`worker`** runs **`rq worker sar_denoise`** (set **`REDIS_URL`** or pass **`--url`**); **`streamlit`** runs **`streamlit run demo/streamlit_app.py`** (or **`sar.py streamlit version`** for the Streamlit CLI).

## Environment

```bash
python -m pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu   # or your CUDA wheel
pip install -r requirements.txt
# Optional (tests + lint, matches CI):
pip install -r requirements-dev.txt
```

## Recommended training (SAMPLE / improved pipeline)

From the repo root, with SAMPLE data under `data/sample_sar/processed/` (see `download_sample_dataset.py`):

```bash
python scripts/train_improved.py --config configs/train/default.yaml
```

Fast smoke (1 epoch, tiny batch):

```bash
python scripts/train_improved.py --config configs/train/smoke.yaml
```

ResUNet variant:

```bash
python scripts/train_improved.py --config configs/train/res_unet_default.yaml
```

## Recommended evaluation

```bash
python scripts/evaluate_sample.py --no-run-log
```

Use `--methods`, `--save_dir`, `--data_layout sample|paired`, and `--no-task-metrics` as needed; see `python scripts/evaluate_sample.py --help`.

## Frozen baseline (reports / regression anchor)

TV-only SAMPLE snapshot and provenance live under **`results/baseline/`**. Regenerate **`metrics.json`** with the commands in **`results/baseline/README.md`** (then `python scripts/capture_baseline.py`).

## Demo and API

```bash
streamlit run demo/streamlit_app.py
uvicorn api.main:app --host 127.0.0.1 --port 8000
```

## Verification and CI-style checks

```bash
python scripts/verify_system.py
pytest -q
ruff check tests inference api workers evaluators scripts
```

## Inference defaults (`configs/infer/`)

Optional **`configs/infer/default.yaml`** supplies **`device`**, **`model_type`** (slug), **`checkpoint`**, **`backend`** (**`pytorch`** / **`onnx`** for Direct denoising), and **`onnx_path`** when the matching **`SAR_*`** vars are unset. Override the file path with **`SAR_INFER_CONFIG`**. **`api.infer_config.service_options_from_merged()`** feeds **`SARDenoiseService`** in the **API**, **RQ worker**, **`scripts/denoise_geotiff.py`**, and **`demo/streamlit_app.py`** (job metadata still supplies per-job device/checkpoint).

## Docker (optional)

CPU **FastAPI** image and Compose **`api`** profile (starts **Redis** too):

```bash
docker compose --profile api up --build
```

Mount trained weights if needed (defaults expect `checkpoints_*` under `/app`). See **README** HTTP API section.

For **JSON access logs** in production aggregators: **`SAR_ACCESS_LOG_JSON=1`** (and optional **`SAR_LOG_LEVEL`**); see **`api/access_log.py`** / **`api.deps`** env list. **`GET /health`** also returns **`direct_infer_backend`** and **`onnx_path_set`** (from **`api.infer_config.service_options_from_merged()`**).

Shared-secret gate for **`/v1/*`**: set **`SAR_API_KEY`** and send **Bearer** or **`X-API-Key`** (see **`api/api_key.py`**).

**`GET /ready`**: when **`SAR_USE_QUEUE=1`**, requires a working **`REDIS_URL`** (**503** if Redis is down or unset); see **`api.deps`** env list.

Change log for larger refactors: `fixes/updates.md`.
