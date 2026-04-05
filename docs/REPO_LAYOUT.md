# Repository layout (navigation)

| Location | What lives here |
|----------|-----------------|
| **`assets/images/`** | PNG figures (comparisons, demos, reports) — not raw datasets |
| **`scripts/`** | Runnable tools: training, evaluation, plotting, GeoTIFF/ONNX helpers, presentation build |
| **`tests/`** | `pytest` tests (including former root `test_*.py` files) |
| **`api/`**, **`inference/`**, **`models/`**, **`trainers/`**, **`algos/`**, **`demo/`**, **`data/`**, **`evaluators/`**, **`workers/`** | Application and library code (Streamlit entrypoint: **`demo/streamlit_app.py`**; Cloud uses root **`requirements.txt`**) |
| **`data/`** | Datasets, loaders (image patches stay next to their splits) |
| **`results/`** | Evaluation outputs, run logs, baseline JSON |
| **`presentations/`** | PowerPoint decks |
| **`configs/`** | YAML training / eval configs |
| **`.streamlit/`** | Streamlit Cloud / local UI config |
| **`runtime.txt`** | Suggested Python version for hosts that read it (`python-3.11`); **Streamlit Community Cloud** uses **Advanced settings** at deploy time |
| **`docs/DEPLOY_STREAMLIT.md`** | Free hosting steps |

## Common commands (from repo root)

```bash
python sar.py eval --help
python scripts/evaluate_sample.py --help
python scripts/train_improved.py --help
streamlit run demo/streamlit_app.py
python scripts/download_real_sample_geotiff.py   # real Sentinel-1 GeoTIFF chip into data/sample_geotiff/
pytest -q
```

`python sar.py …` remains the main CLI; it forwards into `scripts/` where needed.
