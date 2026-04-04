# Project Improvement & Refactoring Report

**Scope:** SAR image denoising using **ADMM-PnP-DL** (Alternating Direction Method of Multipliers with plug-and-play deep priors), with **U-Net / DnCNN** denoisers, **SAMPLE** dataset tooling, **simulated speckle** (`data/sar_simulation.py`), training scripts (`train.py`, `train_simple.py`, `train_improved.py`, `train_sample.py`), evaluation (`evaluate.py`, `evaluate_sample.py`, `algos/evaluation.py`), and a **Streamlit** demo (`demo/streamlit_app.py`).  
**Audited against:** repository layout, `README.md`, `requirements.txt`, `verify_system.py`, and primary modules under `algos/`, `models/`, `data/`, `demo/`, `trainers/`.

---

## 1. Current State of the Project

### Existing system

The project is a **research and demonstration codebase** for denoising SAR-like imagery. The core algorithm lives in `algos/admm_pnp.py` (`ADMMPnP`): an iterative ADMM loop with an **FFT-based data fidelity (x) step** and a **deep network prior (z step)** (U-Net or DnCNN from `models/unet.py`). Optional **log-domain** handling, **PSF / blur** modeling, and **convergence-style monitoring** are embedded in the same module.

**Data:**  
- **SAMPLE** SAR patches via `data/sample_dataset_loader.py` and `download_sample_dataset.py` (paired `clean` / `noisy` patch folders).  
- **Synthetic degradations** via `SARSimulator` in `data/sar_simulation.py` (Gaussian PSF blur, multiplicative speckle using Rayleigh draws, additive Gaussian noise).

**Training:** Multiple entry points (`train_simple.py`, `train_improved.py`, `train_sample.py`, plus `trainers/train_denoiser.py`, `trainers/train_unrolled.py`) reflect evolution of the project rather than a single canonical pipeline.

**Demo:** `demo/streamlit_app.py` provides a browser UI: model and method selection, ADMM hyperparameters, optional synthetic noise injection, presets, and visualization. Auxiliary demo helpers include `demo/smart_display.py`, `demo/bulletproof_display.py`, `demo/emergency_denoiser.py`.

**Ops / quality:** `verify_system.py`, numerous `test_*.py` scripts, plotting utilities (`plot_*.py`), and several narrative `*_SUMMARY.md` files alongside `PROJECT_REPORT.md`.

### Architecture (how things work today)

```
User / developer
    │
    ├─► Training: scripts → DataLoader (SAMPLE or synthetic) → PyTorch model → checkpoints_* /
    │
    ├─► Evaluation: evaluate*.py → metrics in algos/evaluation.py → results/ plots
    │
    ├─► HTTP API (optional): FastAPI api/ + inference/ → /v1/denoise; optional Redis/RQ /v1/jobs
    │
    └─► Demo: streamlit run demo/streamlit_app.py → loads weights → ADMMPnP / direct / TV path → matplotlib/Streamlit display
```

**Inference** also runs via **`inference/`** + **`api/`** (sync PNG/JSON and optional async jobs). Streamlit remains a **separate** interactive UI, not the only entrypoint.

### Tech stack and design approach

| Layer | Technology |
|--------|------------|
| Language | Python 3.10–3.12 (per README) |
| DL | PyTorch, torchvision |
| Numerics | NumPy, SciPy (FFT utilities in ADMM path) |
| Vision I/O | OpenCV (headless), Pillow, scikit-image |
| UI | Streamlit |
| Packaging | `requirements.txt`, `setup.py` |

Design is **modular at folder level** (`algos`, `models`, `data`, `trainers`, `demo`) but **operational entry points are fragmented** (many top-level scripts and overlapping docs).

### What the project does well

- **Clear separation of algorithm (`algos/admm_pnp.py`) vs architectures (`models/unet.py`) vs data helpers (`data/`).**
- **Hybrid classical + DL narrative** (ADMM + PnP denoiser) is technically coherent and defensible for academic work.
- **Practical training upgrades** in `train_improved.py` (e.g., AdamW, cosine restarts, gradient-aware loss) show attention to optimization hygiene.
- **SAR-aware simulation** (speckle + PSF + optional log transform) supports controlled experiments.
- **Evaluation module** targets SAR-relevant ideas (e.g., ENL alongside PSNR/SSIM in documentation and `algos/evaluation.py`).
- **Streamlit demo** lowers the barrier for reviewers and non-developers to try the system.

---

## 2. Key Problems & Limitations

### Major technical flaws

1. ~~**Documentation vs repository mismatch**~~ **(resolved):** All docs and scripts now reference **`demo/streamlit_app.py`**, and **`verify_system.py`** checks that path. *(If you deploy on Streamlit Cloud, set the main file to `demo/streamlit_app.py`.)*
2. **Fragmented training surface:** Competing scripts (`train.py`, `train_simple.py`, `train_improved.py`, `train_sample.py`) without a single **CLI** (e.g., Hydra/Typer) or **config schema** make reproduction and comparison error-prone.
3. **Mixed inference semantics in the UI:** The demo allows **user uploads**, **synthetic noise sliders**, and **SAMPLE loading** in one flow. That blurs “denoise real SAR” vs “run the simulator,” which undermines interpretability of metrics shown in-session.
4. **Performance hotspots:** ADMM combines **Python-side SciPy/PyTorch FFT paths** and repeated denoiser forward passes; large tiles will hit **latency and memory** limits without tiling, mixed precision, or compiled kernels.

### Architectural issues

- **No bounded context for “inference service” vs “experimentation”** — everything tends to rerun in Streamlit’s process model.
- **No artifact registry:** Checkpoints (`checkpoints_improved/`, `checkpoints_simple/`) are not tied to **versioned configs**, **dataset hashes**, or **evaluation reports** in code.
- **Root-level sprawl:** Many `*_SUMMARY.md`, `test_*.py`, and one-off fix narratives belong in `docs/archive/` or issues/PRs to reduce cognitive load.

### ML / model weaknesses

- **Generalization claims are fragile** if evaluation is dominated by **in-distribution** pairs (same simulation statistics or same patch generation pipeline as training).
- **U-Net / DnCNN** are strong baselines but **not automatically SOTA**; README-style **wide metric bands** (ranges without pinned experiment IDs) do not meet industry or peer-review standards for reproducibility.
- **PSNR/SSIM** on intensity/amplitude SAR are **weak proxies** for downstream utility (detection, segmentation, change detection).
- **Risk of oversmoothing or structure hallucination** is not systematically surfaced to users (no uncertainty maps, no residual diagnostics in product form).

### UI / UX and product-level problems

- Streamlit is adequate for **demos**, not for **geospatial workflows** (no CRS-aware preview, no GeoTIFF-native pipeline in the described architecture).
- **High-dimensional sidebar** (many sliders + presets + “disable denoising” + enhancement modes) increases odds of **misconfiguration** without guided “safe defaults” or locked expert modes.

### Scalability and real-world limitations

- **Single-process Streamlit** does not scale horizontally; no async job processing for large uploads or batch tiles.
- **No authentication, quotas, or audit logs** — unacceptable for many enterprise or government-adjacent SAR use cases.
- **Production SAR pipelines** require **geospatial metadata preservation**, **tiling**, and **deterministic preprocessing** — largely absent as first-class features.

---

## 3. Gap Between Current State and Industry Standards

| Area | Industry / production expectation | This project |
|------|-----------------------------------|--------------|
| **Serving** | REST/gRPC API or dedicated inference worker (Triton/TorchServe), health checks, autoscaling | Streamlit monolith |
| **Jobs** | Queue (Redis/RabbitMQ/SQS) + workers for batch denoising | Interactive only |
| **Data** | GeoTIFF/COG, SRS/CRS, multiband/polSAR, STAC catalogs | PNG/JPEG upload; SAMPLE patches on disk |
| **Reproducibility** | Locked configs, MLflow/W&B/Bento, container images, CI gates | Multiple scripts; demo entrypoint docs aligned with repo |
| **Quality** | Unit + integration tests in CI, lint/format/typecheck | Ad hoc `test_*.py`; no visible CI config in repo root audit |
| **Security** | Secrets management, virus scanning on uploads, PII/geo policy | Not addressed |
| **Observability** | Structured logs, metrics, tracing | Minimal |

**Why this is not production-ready:** Operational SAR software must combine **correct radiometry**, **metadata fidelity**, **scalable inference**, **reproducible models**, and **governance**. The current system is optimized for **local experimentation and demonstration**, not for **SLA-backed services** or **operational EO pipelines**.

---

## 4. Required Changes (What Needs to Be Fixed)

### Immediate fixes (quick wins)

1. ~~**Unify the Streamlit entrypoint**~~ **(done):** References updated to `demo/streamlit_app.py` across README, reports, summaries, and helper scripts.
2. ~~**Fix `verify_system.py`**~~ **(done for path check):** Optional follow-up: add `python -c "import algos.admm_pnp"` smoke test in `verify_system.py`.
3. **Add a `CONTRIBUTING.md` or short `docs/DEVELOPMENT.md`** listing the **one** recommended training command and **one** evaluation command for the SAMPLE pipeline.
4. **Pin critical dependencies** in a `requirements.lock` or Poetry/`uv.lock` (or document exact torch+cuda index URL per platform).

### Medium-level improvements

1. ~~**Single CLI:**~~ **(largely done)** — Root **`sar.py`** forwards **`train`**, **`eval`**, **`api`**, **`worker`**, ONNX/GeoTIFF/ablation scripts (stdlib **argparse** + **subprocess**; no Typer required).
2. ~~**Config package:**~~ **(largely done)** — **`configs/train/*.yaml`** and **`configs/infer/default.yaml`** with env overrides; training and API read these paths.
3. **Consolidate documentation:** Move historical `*_SUMMARY.md` into `docs/history/`; keep root README focused; link to one architecture diagram.
4. **Evaluation bundle:** One script writes `results/<run_id>/metrics.json`, `preds/`, and plots — always logging **git SHA**, **config hash**, and **dataset version**.
5. ~~**Geospatial MVP:**~~ **(done for windowed GeoTIFF)** — **`inference/geotiff.py`**, **`scripts/denoise_geotiff.py`** (see **`fixes/updates.md`** Step 06).

### Major system redesigns

1. ~~**Inference service:**~~ **(MVP done)** — **`api/`** FastAPI sync **`/v1/denoise`** + optional **Redis/RQ** **`/v1/jobs`**; inference core in **`inference/`** (not yet a separate installable **`sar_denoise`** package name).
2. **Model serving:** **ONNX** export + ORT Direct path in-repo; **Triton**, load tests, and p95 SLOs remain future work.
3. **Data pipeline:** Object storage (S3/GCS/MinIO) for inputs/outputs; **Apache Beam** / **Dask** / **Prefect** for large-area batch denoising.
4. **Frontend split:** Optional React/Leaflet or integration as a **QGIS/SNAP**-adjacent plugin rather than Streamlit for serious analysts.
5. ~~**MLOps:**~~ **(partial)** — **`Dockerfile`**, **`docker-compose`** **`api`** profile, **GitHub Actions** (**ruff**, **`verify_system.py`**, **pytest**); model registry / smoke train in CI not added.

---

## 5. Improved Architecture & Approach

### Target system design

```
                    ┌─────────────────────────────────────────┐
                    │           Client / Integrator          │
                    │  (Web UI, QGIS plugin, batch CLI)       │
                    └────────────────────┬────────────────────┘
                                         │ HTTPS / internal
                    ┌────────────────────▼────────────────────┐
                    │            API Gateway / FastAPI          │
                    │  POST /v1/jobs  GET /v1/jobs/{id}        │
                    └────────────────────┬────────────────────┘
                                         │
              ┌──────────────────────────┼──────────────────────────┐
              │                          │                          │
    ┌─────────▼─────────┐      ┌─────────▼─────────┐      ┌────────▼────────┐
    │   Job queue       │      │  Metadata DB      │      │  Object storage  │
    │ (Redis/RabbitMQ)  │      │ (Postgres/SQLite) │      │ (S3/MinIO/local) │
    └─────────┬─────────┘      └───────────────────┘      └─────────────────┘
              │
    ┌─────────▼─────────┐
    │  GPU worker pool  │
    │  Triton / Torch   │
    │  ADMM-PnP + model │
    └─────────┬─────────┘
              │
    ┌─────────▼─────────┐
    │  ML training      │  ← separate pipeline: experiment tracking,       │
    │  (offline)        │    registry, scheduled retrains on new data        │
    └───────────────────┘
```

### Backend (API, serving)

- **FastAPI** application: submit job (file URL or upload id), receive `job_id`, poll status, fetch result URL and **provenance JSON** (model version, ADMM params, preprocessing).
- **Workers** pull jobs, stream tiles through denoiser, stitch outputs, upload to storage.
- **Why better:** Decouples UI from compute, enables **horizontal scaling**, **retries**, and **monitoring**.

### ML pipeline improvements

- **Canonical training path** with **frozen datasets** and **ablation configs**: e.g. `direct_unet` vs `admm_pnp_unet` vs classical baselines on the **same** test list.
- **Cross-sensor or cross-scene holdout** to report **domain shift**.
- **Downstream probe:** fine-tune a tiny segmentation/detector on denoised vs raw — aligns with real user value.
- Optional **mixed precision** and **torch.compile** for throughput experiments.

### Data handling (real-world formats)

- First-class **GeoTIFF/COG** I/O with **windowed reads**, **nodata** masks, and **consistent radiometric scaling** (document linear vs log pipeline per product).
- Later: **polSAR** stacks, **multitemporal** stacks for true SAR workflows.

### Deployment strategy

- **Docker Compose** locally: API + Redis + worker + MinIO.
- **Kubernetes** (optional) for autoscaling workers; **GPU node pools**.
- **Streamlit** retained only as a **thin client** calling the API, or dropped for production.

### Why this approach is better

It matches how **EO and defense-adjacent** systems are built: **asynchronous jobs**, **versioned models**, **preserved geospatial context**, and **separation of concerns**. It also makes security, rate limiting, and auditing tractable.

---

## 6. Before vs After Comparison

| Aspect | Current Approach | Improved Approach |
|--------|------------------|-------------------|
| **Architecture** | Streamlit app + loose scripts | API + queue + GPU workers + optional thin UI |
| **Scalability** | Single process; poor for batch/large images | Horizontal workers; tiled processing; batched inference |
| **ML Model** | U-Net/DnCNN + ADMM; multiple training scripts | Versioned configs; ablations; export to ONNX/Triton; drift-aware eval |
| **Usability** | Power-user sliders | Guided modes; GeoTIFF in/out |
| **Production Readiness** | Demo / research grade | Containerized, observable, reproducible, governable |

---

## 7. Step-by-Step Upgrade Roadmap

1. ~~**Correct documentation and verification**~~ **(done)** — `demo/streamlit_app.py` is the single documented entrypoint; `verify_system.py` matches and smoke-tests **`GET /health`** + **`GET /ready`** via **TestClient** when **FastAPI** imports.
2. ~~**Introduce `configs/` + one CLI**~~ **(done)** — **`configs/train/*.yaml`**, **`configs/infer/default.yaml`**, **`sar.py`** (`train`, `eval`, `api`, `worker`, tooling).
3. ~~**Freeze one baseline experiment**~~ **(done)** — **`results/baseline/metrics.json`** + **`results/baseline/README.md`** (TV SAMPLE snapshot + provenance).
4. ~~**Add CI**~~ **(done)** — **`.github/workflows/ci.yml`** (CPU torch, **ruff**, **`verify_system.py`**, **pytest**).
5. ~~**GeoTIFF MVP**~~ **(done)** — **`scripts/denoise_geotiff.py`** + **`inference/geotiff.py`**.
6. ~~**Extract inference core**~~ **(done as `inference/` package)** — importable without Streamlit; **renaming** to **`sar_denoise/`** on PyPI is optional polish.
7. ~~**FastAPI + job queue**~~ **(MVP done)** — **`api/main.py`**, optional **Redis/RQ** jobs; Streamlit still standalone (thin client optional).
8. **Model serving hardening** — ONNX + ORT in-repo; **`configs/infer`** can set **`backend`** / **`onnx_path`** (env overrides); **Triton**, load tests, p95 targets **TBD**.
9. **Cross-domain evaluation** — **paired-folder** eval + **task metrics** in-repo; **downstream probe** (e.g. tiny detector) **TBD**.
10. **Hardening** — **upload size** cap on API; optional **JSON access logs**; optional **`SAR_API_KEY`** on **`/v1/*`**; **`GET /ready`** when queue mode needs Redis; malware policy, retention **TBD** (`GET /health`: **`version`**, optional **`git_sha`**, **`direct_infer_backend`**, **`onnx_path_set`**).

---

## 8. Final Evaluation

| Criterion | Score ( /10) | Notes |
|-----------|--------------|--------|
| **Current project** | **7.0** | Strong academic/engineering learning artifact; clear core idea; hampered by doc drift, fragmented entry points, and demo-centric architecture. |
| **Improved version potential** | **8.5** | With API + geospatial I/O + reproducible eval + CI, this becomes credibly **internship- to early-production-grade** for a narrow EO tool; **startup-level** only if paired with a sharp workflow wedge and validated ROI on real customer tiles. |

**Classification**

- **Resume-level:** **Already** — if described accurately (hybrid ADMM–DL, PyTorch, SAMPLE + simulation, metrics, demo).
- **Internship-level:** Achievable after **roadmap steps 1–4** (docs, CLI, frozen baseline, CI).
- **Production / startup-level:** Requires **steps 5–10** and **domain validation** on real operational data and integrations.

---

*This report is based on the repository state at audit time and should be refreshed after major refactors.*
