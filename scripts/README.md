# Scripts (runnable entrypoints)

Run these from the **repository root** (`PYTHONPATH` is adjusted inside each script).

| Script | Purpose |
|--------|---------|
| `sar.py` | Unified CLI (`train`, `eval`, `verify`, `streamlit`, …) |
| `train_improved.py`, `train_sample.py`, `train_simple.py`, `train.py` | Training |
| `evaluate_sample.py`, `evaluate.py` | Evaluation |
| `verify_system.py`, `test_setup.py` | Environment checks |
| `download_sample_dataset.py` | SAMPLE dataset download |
| `run_demo.py`, `run_complete_workflow.py` | Demos / full pipeline |
| `plot_*.py` | Generate figures → `assets/images/` |
| `build_improvements_presentation.py` | Builds `presentations/project_improvements_presentation.pptx` |
| `capture_baseline.py`, `export_onnx.py`, `denoise_geotiff.py`, … | Tooling |

**Tip:** `python sar.py` at the repo root forwards here automatically.
