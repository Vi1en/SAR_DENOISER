# Repository layout (navigation)

| Location | What lives here |
|----------|-----------------|
| **`assets/images/`** | PNG figures (comparisons, demos, reports) — not raw datasets |
| **`scripts/`** | Runnable tools: training, evaluation, plotting, GeoTIFF/ONNX helpers, presentation build |
| **`tests/`** | `pytest` tests (including former root `test_*.py` files) |
| **`api/`**, **`inference/`**, **`models/`**, **`trainers/`**, **`algos/`**, **`demo/`**, **`data/`**, **`evaluators/`**, **`workers/`** | Application and library code (importable packages) |
| **`data/`** | Datasets, loaders (image patches stay next to their splits) |
| **`results/`** | Evaluation outputs, run logs, baseline JSON |
| **`presentations/`** | PowerPoint decks |
| **`configs/`** | YAML training / eval configs |
| **`.streamlit/`** | Streamlit Cloud / local UI config |
| **`packages.txt`** | Apt deps for Streamlit Cloud (**one package per line, no comments**) |
| **`runtime.txt`** | Streamlit Cloud Python version (`python-3.11`) |
| **`docs/DEPLOY_STREAMLIT.md`** | Free hosting steps |

## Common commands (from repo root)

```bash
python sar.py eval --help
python scripts/evaluate_sample.py --help
python scripts/train_improved.py --help
streamlit run demo/streamlit_app.py
pytest -q
```

`python sar.py …` remains the main CLI; it forwards into `scripts/` where needed.
