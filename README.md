# 🛰️ ADMM-PnP-DL for SAR Image Denoising

**A Deep Learning Approach to Synthetic Aperture Radar Image Denoising**

**Repository layout:** runnable scripts live in **`scripts/`**, tests in **`tests/`**, shared figures in **`assets/images/`** — see [`docs/REPO_LAYOUT.md`](docs/REPO_LAYOUT.md).  
**Dependencies:** **`requirements.txt`** = slim (Streamlit Cloud–friendly). **`requirements-full.txt`** = API + GeoTIFF + queue for local/Docker/CI (`requirements-dev.txt` includes it).

---

## 📋 Table of Contents

1. [Live Demo](#-live-demo)
2. [Project Overview](#-project-overview)
3. [Problem Statement](#-problem-statement)
4. [Solution Approach](#-solution-approach)
5. [System Architecture](#-system-architecture)
6. [Workflow](#-workflow)
7. [Key Features](#-key-features)
8. [Results & Performance](#-results--performance)
9. [Installation & Setup](#-installation--setup)
10. [Usage Guide](#-usage-guide)
11. [Future Scope](#-future-scope)
12. [Presentations](#️-presentations)
13. [Project Structure](#-project-structure)
14. [References](#-references)
15. [Contributing](#-contributing)
16. [License](#-license)
17. [Acknowledgments](#-acknowledgments)
18. [Contact & Support](#-contact--support)
19. [Presentation Tips](#-presentation-tips)

---

## 🔗 Live Demo

- **GitHub Repository**: [`Vi1en/SAR_DENOISER`](https://github.com/Vi1en/SAR_DENOISER)
- **Web App (Streamlit)**: https://sardenoise-eunmdagpnuzuo2g3cqqr9s.streamlit.app/  
  *(After you redeploy from this repo, paste your new **Streamlit Cloud** URL here.)*

### ☁️ Deploy your own (Streamlit Community Cloud — free)

1. Push this repository to GitHub (`Vi1en/SAR_DENOISER`).
2. Open [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Repo: **`Vi1en/SAR_DENOISER`**, branch **`main`**, main file **`demo/streamlit_app.py`**.
4. Deploy. Copy the issued `*.streamlit.app` URL into the line above.

**Details:** [`docs/DEPLOY_STREAMLIT.md`](docs/DEPLOY_STREAMLIT.md) · **System packages:** `packages.txt` · **App config:** `.streamlit/config.toml`

**Note:** Learned checkpoints (`*.pth`) are not committed (see `.gitignore`). The demo still runs (e.g. **TV** and uploads). To enable full DL on Cloud, host weights (Secrets URL, Release, or LFS) and point the sidebar checkpoint path if you add that flow.

---

## 🎯 Project Overview

This project implements a **state-of-the-art SAR (Synthetic Aperture Radar) image denoising system** that combines classical optimization theory with modern deep learning techniques. The system uses **ADMM-PnP-DL (Alternating Direction Method of Multipliers - Plug-and-Play Deep Learning)** to achieve superior denoising performance while preserving fine structural details.

### What is SAR?
- **Synthetic Aperture Radar** is an active remote sensing technology
- Provides high-resolution images regardless of weather conditions
- Used in defense, agriculture, disaster management, and Earth observation
- **Challenge**: Inherently corrupted by speckle noise

### Why This Project Matters
- **Real-world Impact**: Improves SAR image quality for critical applications
- **Technical Innovation**: Combines optimization and deep learning
- **Practical Solution**: Interactive web application for real-time denoising
- **Research Contribution**: State-of-the-art performance on benchmark datasets

### Features at a glance

| Area | What you get |
|------|----------------|
| **Denoising** | ADMM-PnP-DL, direct DL, classical **TV**; speckle reduction with structure-aware metrics |
| **Models** | **U-Net**, **Res-UNet**, DnCNN-style priors; ONNX path for light inference |
| **Demo** | **Streamlit**: upload, SAMPLE patches, metrics (PSNR / SSIM / ENL), diff maps, blind QA, optional TTA uncertainty |
| **Validation** | Quantitative evaluation scripts, baseline JSON, reproducible run logs |
| **API** | Optional **FastAPI** + **Redis/RQ** for async jobs (not required for the Streamlit demo) |

---

## 🔴 Problem Statement

### The Core Problem

SAR images suffer from **speckle noise**, a multiplicative noise pattern that:
- Degrades image quality significantly
- Makes feature detection and classification difficult
- Reduces the effectiveness of downstream applications
- Appears as granular texture throughout the image

### Challenges in SAR Image Denoising

1. **Multiplicative Noise Nature**
   - Speckle is multiplicative (not additive like Gaussian noise)
   - More complex to model and remove
   - Varies across different image regions

2. **Structure Preservation**
   - Must preserve fine details, edges, and textures
   - Balance between noise removal and detail retention
   - Critical for target detection and classification

3. **Varying Noise Levels**
   - Different regions have different noise characteristics
   - Requires adaptive denoising approaches
   - Traditional methods fail to adapt

4. **Computational Efficiency**
   - Real-time or near-real-time processing needed
   - Large image sizes (512×512 to 2048×2048)
   - Limited computational resources in some applications

5. **Generalization**
   - Must work across different SAR sensors
   - Different frequencies and imaging conditions
   - Robust to varying noise levels

### Limitations of Existing Methods

#### Traditional Methods (Lee, Frost, Kuan Filters)
- ❌ Require manual parameter tuning
- ❌ Often over-smooth details
- ❌ Limited adaptability
- ❌ Poor performance on complex scenes

#### Wavelet-Based Methods
- ❌ May introduce artifacts
- ❌ Limited adaptability to varying noise
- ❌ Computational overhead

#### Total Variation (TV) Methods
- ❌ Tend to produce staircasing artifacts
- ❌ Limited noise reduction capability
- ❌ Slow convergence

#### Direct Deep Learning Methods
- ❌ May not exploit degradation structure
- ❌ Require large datasets
- ❌ Limited interpretability

---

## ✅ Solution Approach

### Our Proposed Solution: ADMM-PnP-DL

We combine **classical optimization (ADMM)** with **modern deep learning** to create a hybrid system that:

1. **Leverages ADMM Framework**
   - Proven optimization algorithm
   - Efficient iterative solution
   - Interpretable optimization process

2. **Integrates Deep Learning Denoisers**
   - U-Net and DnCNN architectures
   - Plug-and-play design
   - State-of-the-art denoising capability

3. **Adaptive Parameter Learning**
   - Learnable optimization parameters
   - End-to-end training
   - Optimal performance

### Key Innovation: Plug-and-Play Architecture

```
Traditional ADMM: Uses fixed regularizers (e.g., TV)
Our Approach: Uses deep learning denoisers as regularizers
```

**Benefits:**
- ✅ Any denoiser can be plugged in
- ✅ Flexible and extensible
- ✅ Combines best of both worlds
- ✅ Superior performance

### Algorithm Overview

The ADMM-PnP algorithm iteratively solves:

1. **x-update**: Data fidelity term (FFT-based efficient solution)
2. **z-update**: Deep learning denoiser (U-Net/DnCNN)
3. **Dual update**: Lagrange multiplier update

This iterative process converges to a high-quality denoised image.

---

## 🏗️ System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────┐
│              SAR Image Denoising System                  │
├─────────────────────────────────────────────────────────┤
│                                                           │
│  ┌──────────────┐      ┌──────────────┐                │
│  │  Noisy SAR   │ ───> │  ADMM-PnP-DL │ ───> │ Clean   │
│  │    Image     │      │   Algorithm   │      │  Image  │
│  └──────────────┘      └──────────────┘      └─────────┘
│                              │                            │
│                              ▼                            │
│                    ┌─────────────────┐                    │
│                    │  Deep Learning  │                    │
│                    │    Denoiser     │                    │
│                    │  (U-Net/DnCNN)  │                    │
│                    └─────────────────┘                    │
│                                                           │
└─────────────────────────────────────────────────────────┘
```

### Core Components

#### 1. ADMM-PnP Algorithm (`algos/admm_pnp.py`)
- **x-update**: FFT-based efficient solution
- **z-update**: Deep learning denoiser
- **Dual update**: Standard ADMM dual variable update
- **Adaptive parameters**: Learnable rho, alpha, theta
- **Convergence monitoring**: Energy and residual tracking

#### 2. Deep Learning Models (`models/unet.py`)
- **U-Net**: Encoder-decoder with skip connections
  - Preserves spatial information
  - Multi-scale feature extraction
  - Best for detail preservation
- **DnCNN**: Deep CNN with residual connections
  - Residual learning
  - Efficient inference
  - Good speed-quality trade-off

#### 3. Training Framework (`trainers/`)
- **Denoiser Training**: Supervised learning with L1 + SSIM loss
- **Unrolled ADMM**: End-to-end optimization
- **Data Augmentation**: Flips, rotations, crops
- **Adaptive Learning**: Learning rate scheduling

#### 4. Evaluation System (`algos/evaluation.py`)
- **PSNR**: Peak Signal-to-Noise Ratio
- **SSIM**: Structural Similarity Index
- **ENL**: Equivalent Number of Looks (SAR-specific)
- **Runtime**: Processing time analysis

#### 5. Interactive Web Application (`demo/streamlit_app.py`)
- **Streamlit Interface**: Real-time denoising
- **Parameter Tuning**: Interactive ADMM parameters
- **Visualization**: Before/after comparison
- **Metrics Display**: Real-time performance

---

## 🔄 Workflow

### Complete Project Workflow

```
┌─────────────────────────────────────────────────────────────┐
│                    PROJECT WORKFLOW                          │
└─────────────────────────────────────────────────────────────┘

1. DATA PREPARATION
   ├── Download SAMPLE SAR dataset
   ├── Organize train/val/test splits
   ├── Extract patches (128×128)
   └── Apply data augmentation

2. MODEL TRAINING
   ├── Train U-Net/DnCNN denoiser
   │   ├── Loss: L1 + SSIM
   │   ├── Optimizer: Adam
   │   └── Epochs: 100
   │
   └── Train Unrolled ADMM (optional)
       ├── End-to-end optimization
       └── Epochs: 50

3. EVALUATION
   ├── Test on validation set
   ├── Calculate metrics (PSNR, SSIM, ENL)
   ├── Compare with baselines
   └── Generate visualizations

4. DEPLOYMENT
   ├── Save trained models
   ├── Launch Streamlit app
   └── Real-time denoising
```

### Step-by-Step Workflow

#### Phase 1: Setup & Data Preparation
```bash
# 1. Install dependencies (full local stack: API + GeoTIFF + queue)
pip install -r requirements-full.txt

# 2. Download SAMPLE SAR dataset
python scripts/download_sample_dataset.py

# 3. Verify setup
python scripts/verify_system.py
```

#### Phase 2: Training
```bash
# Option A: Simple training
python scripts/train_simple.py

# Option B: Improved training (recommended)
python scripts/train_improved.py

# Option C: Train on SAMPLE dataset
python scripts/train_sample.py

# Option D: Improved training from YAML (reproducible; paths relative to repo root)
python scripts/train_improved.py --config configs/train/default.yaml
python scripts/train_improved.py --config configs/train/smoke.yaml   # 1 epoch — quick smoke test
```

#### Phase 3: Evaluation
```bash
# Evaluate on test set
python scripts/evaluate_sample.py
# Writes plots + evaluation_results.json under --save_dir (default results_sample/),
# and a reproducible bundle under results/runs/<UTC>_<git_sha>/ (manifest.json, metrics.json, plots/).
# Use --no-run-log to skip the results/runs/ bundle.

# Generate performance visualizations
python scripts/plot_recommended_comparisons.py
```

#### Phase 4: Deployment
```bash
# Launch interactive web application
streamlit run demo/streamlit_app.py
# or: python sar.py streamlit
```

#### GeoTIFF (optional, production-style I/O)

Single-band, **georeferenced** GeoTIFFs only (`rasterio` is in **`requirements-full.txt`**). The CLI reads the raster in **windows** (default tile 512×512), denoises each tile, and writes a float32 GeoTIFF with the **same CRS and geotransform** as the input. **`overlap` must be 0** in this version (non-overlapping tiles; overlap blending is future work). **Nodata** pixels are left unchanged where possible.

**Normalization:** each tile is min–max scaled to `[0, 1]` before the model and rescaled afterward. This does **not** infer amplitude vs dB; align with how you trained the network.

```bash
python scripts/denoise_geotiff.py \
  --in path/to/s1_amplitude.tif \
  --out path/to/denoised.tif \
  --checkpoint checkpoints_improved/best_model.pth \
  --model_type unet \
  --method "ADMM-PnP-DL"
```

For **TV Denoising**, omit `--checkpoint`. You can also call `inference.geotiff.denoise_geotiff` and `make_tile_denoise_fn` from Python.

#### HTTP API (FastAPI)

Stateless JSON/PNG API (no auth in this step; see roadmap for API keys).

Run **one command at a time** (or paste only the lines below). If a sentence like “omit …” ends up on its own line, the shell will try to run **`omit`** as a command (`command not found: omit`).

```bash
# Optional — only if you want this weights file for ADMM/Direct resolution:
export SAR_CHECKPOINT=checkpoints_improved/best_model.pth

# Optional overrides (defaults are fine for many setups):
export SAR_DEVICE=auto
export SAR_MODEL_TYPE=unet
# Optional: YAML defaults — see configs/infer/default.yaml and SAR_INFER_CONFIG
# (device, model_type, checkpoint, backend, onnx_path; SAR_* env vars override)

uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Same via **`sar.py`**: `python sar.py api -- --host 0.0.0.0 --port 8000`.

Development reload:

```bash
uvicorn api.main:app --reload --port 8000
```

**Docker (CPU API):** image + Compose profile `api` (Redis included for optional job queue).

```bash
docker compose --profile api up --build
curl -s http://127.0.0.1:8000/health
```

Weights are not baked into the image by default. Mount checkpoints, e.g.  
`-v "$(pwd)/checkpoints_simple:/app/checkpoints_simple:ro"`, or set `SAR_CHECKPOINT` to a **path inside the container** that you mount from the host.

For **TV Denoising** you do **not** need `SAR_CHECKPOINT`; skip that `export` or unset it: `unset SAR_CHECKPOINT`.

- `GET /health` → `{"status":"ok","version":"0.1.0"}`, optional **`git_sha`**, plus **`direct_infer_backend`** (`pytorch` \| `onnx` from merged infer config / **`SAR_BACKEND`**) and **`onnx_path_set`** (whether an **`onnx_path`** is configured after merge — not a file-exists check).
- `GET /ready` → **`{"status":"ready","queue":"disabled"}`** by default; when **`SAR_USE_QUEUE=1`**, returns **200** only if **`REDIS_URL`** is set and Redis answers **`PING`** (otherwise **503** for orchestrator readiness probes).
- Optional **`SAR_ACCESS_LOG_JSON=1`** — one **JSON** line per request on logger **`sar.api.access`**; response **`X-Request-ID`** (pass your own or get a generated UUID).
- Optional **`SAR_API_KEY`** — when set, **`POST /v1/denoise`** and **`/v1/jobs*`** require **`Authorization: Bearer <key>`** or **`X-API-Key: <key>`**; **`/health`**, **`/ready`**, and **`/docs`** stay public.
- `POST /v1/denoise` — multipart file upload; query params include `method` (`ADMM-PnP-DL`, `Direct Denoising`, `TV Denoising`), `max_iter`, `rho_init`, `alpha`, `theta`, etc. Response is **PNG** (`image/png`). **Max upload ~20 MB** (rejected with 413). OpenAPI UI: **`/docs`**.

##### Async jobs (Redis + RQ, Step 08)

When **`SAR_USE_QUEUE=1`** and **`REDIS_URL`** is set, the app also exposes:

- **`POST /v1/jobs`** — same query params as sync denoise; returns `{"job_id","rq_job_id"}`.
- **`GET /v1/jobs/{job_id}`** — `{"job_id","status","error"}` with `status` in `queued` | `running` | `done` | `failed`.
- **`GET /v1/jobs/{job_id}/result`** — PNG when `status` is `done` (HTTP 425 until then).

Artifacts live under **`data/jobs/<job_id>/`** (`input.png`, `output.png`, `meta.json`, `status.json`). **Clean up old directories periodically** (cron) to control disk use.

```bash
docker compose up -d redis
export REDIS_URL=redis://localhost:6379/0
export SAR_USE_QUEUE=1
# same SAR_CHECKPOINT / SAR_DEVICE as above

# Terminal 1 — worker (from repo root):
rq worker sar_denoise --url "$REDIS_URL"
# or: python sar.py worker -- --url "$REDIS_URL"

# Terminal 2 — API:
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

With **`SAR_USE_QUEUE=0`** (default), only synchronous **`/v1/denoise`** is registered for jobs; Step 07 `curl` scripts keep working.

##### ONNX Direct denoising (Step 09)

Export the **U-Net/DnCNN z-update** to ONNX for faster CPU inference or external runtimes (e.g. Triton later). **ADMM-PnP-DL** and **TV** still use the existing PyTorch/Python paths; only **Direct Denoising** can use ONNX.

```bash
export SAR_CHECKPOINT=checkpoints_simple/best_model.pth
python scripts/export_onnx.py --out models_artifacts/denoiser.onnx
python scripts/compare_pytorch_onnx.py --onnx models_artifacts/denoiser.onnx
```

Runtime:

```bash
export SAR_BACKEND=onnx
export SAR_ONNX_PATH=models_artifacts/denoiser.onnx
# Optional ORT providers (comma-separated):
# export SAR_ONNX_EP=CUDAExecutionProvider,CPUExecutionProvider
```

Default is **`SAR_BACKEND=pytorch`** (unchanged). For GPU inference install **`onnxruntime-gpu`** in place of **`onnxruntime`** if desired.

**Triton / full ADMM in ONNX:** out of scope for this step (see `fixes/09_optimize_model_serving.md`).

### Quick Start Workflow
```bash
# Complete workflow in one command
python run_complete_workflow.py
```

---

## ✨ Key Features

### 1. Advanced Denoising Algorithm
- **ADMM-PnP Framework**: Combines optimization and deep learning
- **FFT-based Computation**: Efficient x-update step
- **Adaptive Parameters**: Learnable optimization parameters
- **Convergence Monitoring**: Real-time energy tracking

### 2. State-of-the-Art Deep Learning
- **U-Net Architecture**: Best for detail preservation
- **DnCNN Architecture**: Fast inference with good quality
- **Noise Conditioning**: Optional noise level input
- **Multiple Loss Functions**: L1, SSIM, combined losses

### 3. SAR-Specific Features
- **Speckle Modeling**: Multiplicative Rayleigh noise
- **PSF Blur**: Realistic SAR degradation
- **ENL Metric**: SAR-specific quality measure
- **Log-Domain Processing**: Optional for speckle handling

### 4. Comprehensive Evaluation
- **Multiple Metrics**: PSNR, SSIM, ENL, Runtime
- **Baseline Comparison**: Classical filters, TV, direct CNN
- **Visual Comparisons**: Difference maps, zoom-ins
- **Performance Plots**: Bar charts, distributions

### 5. Interactive Web Application
- **Real-Time Denoising**: Instant results
- **Parameter Tuning**: Interactive sliders
- **Visual Comparison**: Side-by-side before/after
- **Metrics Display**: Real-time performance metrics
- **Export Functionality**: Save denoised images

### 6. Production-Ready Code
- **Modular Design**: Easy to extend
- **Comprehensive Documentation**: Well-commented code
- **Error Handling**: Robust error management
- **GPU/CPU Support**: Automatic device detection

---

## 📊 Results & Performance

### Quantitative Results

| Method | PSNR (dB) | SSIM | ENL | Runtime (s) |
|--------|-----------|------|-----|-------------|
| **Noisy Image** | 15-20 | 0.3-0.5 | 1-2 | - |
| **Classical Filters** | 22-24 | 0.65-0.70 | 3-5 | 0.8-1.5 |
| **ADMM-TV** | 22-28 | 0.6-0.75 | 3-5 | 2-5 |
| **Direct CNN** | 25-30 | 0.7-0.85 | 4-8 | 0.1-0.5 |
| **ADMM-PnP-DL (DnCNN)** | **27-34** | **0.75-0.92** | **4-12** | **0.3-1.5** |
| **ADMM-PnP-DL (U-Net)** | **28-35** | **0.8-0.95** | **5-15** | **0.5-2** |

### Key Achievements

✅ **PSNR Improvement**: +2 to +4 dB over classical filters  
✅ **SSIM Improvement**: +0.03 to +0.07 over classical filters  
✅ **ENL Improvement**: 2-3x better noise reduction  
✅ **Speed**: 2-10x faster than traditional ADMM-TV  
✅ **Edge Preservation**: EPI = 0.92 (U-Net variant)

### Performance Highlights

- **Best PSNR**: 35 dB (U-Net variant)
- **Best SSIM**: 0.95 (U-Net variant)
- **Best ENL**: 15 (U-Net variant)
- **Fastest Processing**: 0.3 seconds (DnCNN variant)
- **Best Edge Preservation**: 0.92 EPI (U-Net variant)

### Visual Results

The system generates comprehensive visualizations:
- Performance comparison charts
- Difference maps showing error reduction
- Zoom-in comparisons demonstrating detail preservation
- Runtime and edge preservation metrics

---

## 🚀 Installation & Setup

### Prerequisites

- **OS**: macOS / Windows / Linux
- **Python**: 3.10–3.12
- **Disk Space**: ≥10 GB free
- **GPU** (Optional): CUDA 11.8+ with matching PyTorch build

### Installation Steps

#### 1. Clone the Repository
```bash
git clone <repository-url>
cd FINAL_YEAR_PROJECT
```

#### 2. Create Virtual Environment

**Using Conda (Recommended):**
```bash
conda create -n sar-denoise python=3.11 -y
conda activate sar-denoise
```

**Using venv:**
```bash
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
```

#### 3. Install Dependencies

**Install PyTorch first:**

CPU-only:
```bash
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

NVIDIA GPU (CUDA 11.8):
```bash
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**Install remaining dependencies:**
```bash
pip install -r requirements-full.txt
```

#### Troubleshooting: NumPy / Matplotlib import error

If training fails with **`numpy.core.multiarray failed to import`**, **`ImportError: initialization failed`** under `matplotlib`, or **“compiled using NumPy 1.x cannot be run in NumPy 2.x”**, your Python stack has a **NumPy / Matplotlib binary mismatch** (often **conda `base`** with NumPy 2 while Matplotlib was built for NumPy 1).

**Do not** run **`pip install --force-reinstall matplotlib` alone** after downgrading NumPy: current Matplotlib wheels (3.10+) will **pull NumPy 2.x back in**, break **SciPy / Numba / Streamlit**, and upgrade **Pillow / packaging** past what Streamlit allows.

**Fix — reinstall the whole pinned stack in one go** (after installing PyTorch as in step 3):

```bash
pip install -r requirements-full.txt --upgrade --force-reinstall
```

**`requirements.txt`** / **`requirements-full.txt`** keep **NumPy below 2**, **Matplotlib below 3.10**, **OpenCV headless below 4.13** (4.13+ requires NumPy 2), **Pillow below 12**, and **packaging below 25** so the resolver stays consistent.

**Sanity check:**

```bash
python -c "import numpy, matplotlib; print('numpy', numpy.__version__, 'matplotlib', matplotlib.__version__)"
```

You should see **numpy** `1.26.x` (or another **1.x**) and **matplotlib** `3.9.x`.

**About pip’s “dependency conflicts” line after install:** the requirement files cap **SciPy below 1.14** so it stays compatible with **gensim** often preinstalled in **conda base** (`gensim 4.3.x` requires `scipy<1.14`). You may still see warnings for other unrelated packages (e.g. **aext-***, **s3fs** vs **fsspec**). If `python scripts/verify_system.py` runs, those are usually safe to ignore. A **dedicated conda/venv** for this repo avoids mixed conda/pip noise entirely.

**Alternative:** use a **fresh conda env** (see above), then install PyTorch and `pip install -r requirements-full.txt` so conda does not fight pip.

#### Troubleshooting: `command not found: omit` (or other random words)

That usually means a **comment or sentence was split** across lines when you pasted from a guide or chat. In zsh/bash, **only lines starting with `#` are comments** (for whole-line comments). A new line that starts with `omit`, `optional`, etc. is treated as a **command name**. Paste **one shell command per line**, or copy only from fenced `bash` blocks in this README.

#### Troubleshooting: pip `ReadTimeoutError`, DNS, or flaky downloads

If `pip install -r requirements-full.txt` fails with **`Read timed out`** on `files.pythonhosted.org` or **`nodename nor servname provided`**, try a longer timeout and retry on a stable network:

```bash
pip install -r requirements-full.txt --default-timeout=120
```

If **rasterio** keeps failing over pip, install it from conda-forge once, then run `pip install -r requirements-full.txt` again so remaining pins resolve against what is already installed:

```bash
conda install -c conda-forge rasterio
pip install -r requirements-full.txt
```

#### 4. Verify Installation
```bash
python scripts/verify_system.py
```

#### 5. Download Dataset (Optional)
```bash
python scripts/download_sample_dataset.py
```

---

## 📖 Usage Guide

### Basic Usage

#### 1. Train a Model
```bash
# Simple training
python scripts/train_simple.py

# Improved training (recommended)
python scripts/train_improved.py

# Train on SAMPLE dataset
python scripts/train_sample.py

# Improved training from a YAML config (see configs/train/*.yaml)
python scripts/train_improved.py --config configs/train/default.yaml
# Same improved pipeline via train_sample (downloads if needed)
python scripts/train_sample.py --config configs/train/default.yaml
```

#### 2. Evaluate Models
```bash
# Evaluate on test set
python scripts/evaluate_sample.py
# Structured run logs: results/runs/<run_id>/ (see Phase 3). --no-run-log disables them.

# Generate comparison plots
python scripts/plot_recommended_comparisons.py
```

#### 3. Run Interactive Demo
```bash
streamlit run demo/streamlit_app.py
# or: python sar.py streamlit
```

### Python API Usage

#### Basic Denoising
```python
from algos.admm_pnp import ADMMPnP
from models.unet import create_model
import torch

# Load trained model
denoiser = create_model('unet', n_channels=1, noise_conditioning=True)
denoiser.load_state_dict(torch.load('checkpoints_improved/best_model.pth'))

# Create ADMM-PnP instance
admm = ADMMPnP(denoiser, device='cuda', max_iter=20)

# Denoise image
result = admm.denoise(noisy_image)
denoised_image = result['denoised']
```

#### Evaluation
```python
from algos.evaluation import SARDenoisingEvaluator

# Create evaluator
evaluator = SARDenoisingEvaluator(device='cuda')

# Evaluate method
evaluator.evaluate_method('ADMM-PnP-DL', denoiser, test_loader)

# Compare methods
evaluator.compare_methods(evaluator.results)
```

### Web Application Usage

1. **Launch the app**: `streamlit run demo/streamlit_app.py` (or `python sar.py streamlit`)
2. **Upload/Select Image**: Choose a SAR image to denoise
3. **Adjust Parameters**: Use sliders to tune ADMM parameters
4. **View Results**: See before/after comparison and metrics
5. **Export**: Save denoised image and results

---

## 🔮 Future Scope

### Short-Term Enhancements (Next 6 Months)

1. **Multi-Scale Processing**
   - Process images at multiple scales
   - Better handling of varying noise levels
   - Improved detail preservation

2. **Attention Mechanisms**
   - Integrate attention modules in denoiser networks
   - Better feature selection
   - Improved performance on complex scenes

3. **Uncertainty Quantification**
   - Provide confidence estimates for denoised pixels
   - Helpful for critical applications
   - Better decision-making support

4. **Adaptive Iterations**
   - Dynamically determine optimal number of ADMM iterations
   - Faster processing for simple images
   - Better quality for complex images

### Medium-Term Goals (6-12 Months)

1. **Transformer-Based Denoisers**
   - Explore Vision Transformers as plug-and-play denoisers
   - Potential for better performance
   - Self-attention mechanisms

2. **GAN-Based Denoisers**
   - Investigate generative adversarial networks
   - More realistic denoising
   - Better texture preservation

3. **Multi-Task Learning**
   - Joint denoising and segmentation
   - Joint denoising and classification
   - More efficient processing pipeline

4. **Meta-Learning**
   - Fast adaptation to new SAR sensors
   - Few-shot learning capabilities
   - Better generalization

### Long-Term Vision (1-2 Years)

1. **Multi-Temporal Denoising**
   - Leverage temporal information from SAR time series
   - Better noise reduction
   - Change detection capabilities

2. **Multi-Polarization Processing**
   - Utilize polarimetric SAR data
   - Improved denoising performance
   - Better feature extraction

3. **Real-Time Video Processing**
   - Optimize for video/streaming SAR data
   - Real-time denoising pipeline
   - Low-latency processing

4. **Mobile Deployment**
   - Lightweight models for edge devices
   - On-device processing
   - Reduced cloud dependency

5. **Cloud Integration**
   - Scalable cloud-based processing
   - Handle large datasets efficiently
   - Distributed computing support

### Research Directions

1. **Learnable Degradation Models**
   - Learn the degradation operator H from data
   - Better modeling of SAR degradation
   - Improved denoising performance

2. **Uncertainty-Aware Training**
   - Incorporate uncertainty in training
   - More robust models
   - Better generalization

3. **Explainable AI**
   - Interpretability of denoising decisions
   - Visualization of attention maps
   - Better understanding of model behavior

---

## 📽️ Presentations

All PowerPoint files are in **`presentations/`**. See **`presentations/README.md`** for the file list and how to regenerate **`project_improvements_presentation.pptx`**.

---

## 📁 Project Structure

```
FINAL_YEAR_PROJECT/
├── presentations/                    # All .pptx decks — see presentations/README.md
├── data/
│   ├── sar_simulation.py              # SAR image simulation
│   ├── sample_dataset_downloader.py    # SAMPLE dataset download
│   └── sample_dataset_loader.py      # Dataset loading utilities
│
├── models/
│   └── unet.py                        # U-Net and DnCNN architectures
│
├── algos/
│   ├── admm_pnp.py                    # ADMM-PnP algorithm
│   └── evaluation.py                  # Evaluation metrics
│
├── trainers/
│   ├── train_denoiser.py              # Denoiser training
│   ├── train_unrolled.py              # Unrolled ADMM training
│   ├── improved_trainer.py            # ImprovedTrainer (perceptual loss)
│   ├── pipeline.py                    # run_training(TrainingConfig)
│   ├── config_dataclass.py            # TrainingConfig defaults
│   └── config_loader.py               # YAML → TrainingConfig
│
├── configs/
│   ├── train/
│   │   ├── default.yaml               # Default improved training
│   │   ├── sample.yaml                # SAMPLE experiment variant
│   │   └── smoke.yaml                 # 1-epoch CI smoke
│   └── infer/
│       └── default.yaml               # API inference defaults (optional)
│
├── demo/
│   ├── streamlit_app.py               # Streamlit web application
│   ├── smart_display.py               # Display utilities
│   └── bulletproof_display.py         # Robust display handling
│
├── docs/
│   └── DEVELOPMENT.md                 # Canonical train/eval + verify commands
│
├── notebooks/
│   └── 01_data_preparation.ipynb      # Data preparation notebook
│
├── checkpoints_improved/              # Trained models (improved)
├── checkpoints_simple/                # Trained models (simple)
├── results/                           # Evaluation results
│   ├── baseline/                      # Frozen TV baseline + metrics.json (see README inside)
│   └── runs/                          # Per-run manifest + metrics (evaluate_sample.py)
│
├── evaluators/
│   └── run_logger.py                  # EvaluationRunContext (structured run artifacts)
│
├── inference/
│   ├── service.py                     # SARDenoiseService (ADMM / direct / TV; no Streamlit)
│   ├── geotiff.py                     # Windowed GeoTIFF denoise (CRS preserved)
│   ├── onnx_export.py                 # export_denoiser_to_onnx (torch, dynamo=False)
│   ├── onnx_backend.py                # ONNXDirectDenoiser (ORT)
│   └── types.py                       # DenoiseMethod, result helpers
│
├── models_artifacts/                  # Exported denoiser.onnx (gitignored except .gitkeep)
│
├── scripts/
│   ├── denoise_geotiff.py             # CLI for GeoTIFF denoising
│   ├── export_onnx.py                 # Checkpoint → ONNX
│   └── compare_pytorch_onnx.py        # Parity check
│
├── api/
│   ├── main.py                        # FastAPI app (health, /v1/denoise, optional /v1/jobs)
│   ├── constants.py                  # MAX_UPLOAD_BYTES, DenoiseMethodEnum
│   ├── deps.py                        # SARDenoiseService singleton + env
│   ├── jobs.py                        # POST/GET /v1/jobs (SAR_USE_QUEUE=1)
│   └── storage.py                     # data/jobs/<id>/ layout
│
├── workers/
│   └── tasks.py                       # RQ task run_denoise_job
│
├── Dockerfile                         # CPU FastAPI image (optional)
├── docker-compose.yml                 # redis + optional `api` profile
│
├── tests/
│   ├── test_inference_service_smoke.py
│   ├── test_geotiff_smoke.py
│   ├── test_api_smoke.py
│   ├── test_job_storage.py
│   └── test_onnx_export_parity.py
│
├── sar.py                             # Unified CLI shim → scripts/sar.py
├── paths.py                           # REPO_ROOT, assets/images helpers
├── assets/images/                     # PNG figures (plots, demos)
├── scripts/                           # Runnable entrypoints (train, eval, plots, tools)
│   ├── sar.py                         # CLI implementation
│   ├── train_improved.py, train_sample.py, train_simple.py, train.py
│   ├── evaluate_sample.py, evaluate.py, verify_system.py
│   ├── plot_*.py, download_sample_dataset.py, run_demo.py, …
│   └── build_improvements_presentation.py
│
├── requirements.txt                  # Slim deps (Streamlit Cloud default)
├── requirements-full.txt             # + rasterio, FastAPI, Redis/RQ (local/Docker/CI)
├── CONTRIBUTING.md                    # PR checklist (tests, ruff, changelog)
├── PROJECT_REPORT.md                  # Detailed project report
└── README.md                          # This file
```

---

## 📚 References

### Key Papers

1. **ADMM-PnP**: Venkatakrishnan, S. V., Bouman, C. A., & Wohlberg, B. (2013). "Plug-and-Play Priors for Model Based Reconstruction." *IEEE Global Conference on Signal and Information Processing*.

2. **U-Net**: Ronneberger, O., Fischer, P., & Brox, T. (2015). "U-Net: Convolutional Networks for Biomedical Image Segmentation." *Medical Image Computing and Computer-Assisted Intervention (MICCAI)*.

3. **DnCNN**: Zhang, K., Zuo, W., Chen, Y., Meng, D., & Zhang, L. (2017). "Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising." *IEEE Transactions on Image Processing*.

4. **ADMM**: Boyd, S., Parikh, N., Chu, E., Peleato, B., & Eckstein, J. (2011). "Distributed Optimization and Statistical Learning via the Alternating Direction Method of Multipliers." *Foundations and Trends in Machine Learning*.

### Additional Resources

- **SAR Image Processing**: Various textbooks and papers on SAR image analysis
- **Deep Learning**: PyTorch documentation and tutorials
- **Optimization**: ADMM algorithm references and implementations

---

## 🤝 Contributing

We welcome contributions! Please follow these steps:

For **canonical train/eval commands** and local checks, see **[docs/DEVELOPMENT.md](docs/DEVELOPMENT.md)**. For PR expectations (tests, ruff, changelog), see **[CONTRIBUTING.md](CONTRIBUTING.md)**.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgments

- **PyTorch Team**: For the deep learning framework
- **Streamlit Team**: For the web interface framework
- **SAMPLE Dataset**: AFRL for providing the SAR dataset
- **Open Source Community**: For various contributions and support

---

## 📞 Contact & Support

For questions, issues, or support:
- **GitHub Issues**: Open an issue on the repository
- **Email**: Contact the project maintainers
- **Documentation**: See `PROJECT_REPORT.md` for detailed documentation

---

## 🎯 Presentation Tips

### For Presenters

1. **Start with Problem**: Emphasize the importance of SAR denoising
2. **Show Solution**: Demonstrate the ADMM-PnP-DL approach
3. **Highlight Results**: Use the performance metrics and visualizations
4. **Live Demo**: Run the Streamlit app during presentation
5. **Future Work**: Discuss potential improvements and applications

### Key Points to Emphasize

- ✅ **Innovation**: Combining classical optimization with deep learning
- ✅ **Performance**: Significant improvements over existing methods
- ✅ **Practicality**: Real-time web application
- ✅ **Impact**: Real-world applications in defense, agriculture, disaster management

---

**Note**: This project is for research and educational purposes. For production use, please ensure proper testing and validation.

---

*Last Updated: 2024*
