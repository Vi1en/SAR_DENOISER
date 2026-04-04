# Project updates log

Structured changelog: newest entries first. Each update documents what changed, why, and how to verify.

## Update 53 - Professor presentation pack (`presentations/project_improvements_presentation.pptx`)
**Date:** 2026-04-04  
**Type:** Documentation / Presentation

### What Changed
- **`presentations/`** — single folder for **all** `.pptx` files + **`presentations/README.md`** index; **README.md** TOC + **Presentations** section point here.
- **`scripts/build_improvements_presentation.py`** — writes **`presentations/project_improvements_presentation.pptx`** (**~15 slides**, 16:9): **matplotlib** figures (denoising noisy/denoised/diff; **PSNR & SSIM bar charts**; A/B diff map); **4-column method table** (TV from **`results/baseline/metrics.json`**, DL/ADMM from **README** mid-bands with footnote); **flow** Problem→…→Result; proof checklist; **IMPROVEMENTS_SUMMARY** PSNR line; MAD **0.09** note.
- **`PROJECT_PRESENTATION_SUMMARY.md`** — top section links to **`presentations/`** paths.
- Content tied to **fixes/updates.md** (especially **46–52**, **50**, **49**, **45**).

### Why
- Easy to find decks; professor-facing **debugging + mathematical validation** story.

### How to Verify
- `pip install python-pptx` → `python scripts/build_improvements_presentation.py` → open **`presentations/project_improvements_presentation.pptx`**.

### Breaking Changes
- None

---

## Update 52 - Advanced Image Similarity & Identity Validation
**Date:** 2026-04-04  
**Type:** UI / Evaluation

### What Changed
- **`demo/streamlit_app.py`** — **Per-patch statistics**: **SHA256** over **contiguous float32** `tobytes()`, first **16** hex chars via **`st.write("Hash:", …)`** per file.
- **🔬 Image Similarity Analysis** (replaces the old short “compare two patches” block): **A / B / normalized difference map** where **`diff_norm = |A−B| / (max(|A−B|) + ε)`**, caption **“Normalized Difference Map”**.
- **Metrics:** **`st.metric`** for **MAD**, **Max Diff**, and **SSIM** via **`calculate_ssim`** from **`algos.evaluation`** (with safe fallback if computation fails).
- **Verdict:** **Nearly identical** / **Similar** / **Clearly different** from MAD thresholds with **`st.warning` / `st.info` / `st.success`**; hash prefix line for A and B plus warning if hashes match.
- **Optional** overlaid **matplotlib** histograms for A vs B.

### Why This Change Was Needed
- SAR speckle makes patches look alike; **hashes + MAD + SSIM + verdict + diff map** give **quantitative proof** and clearer demo narrative.

### Old vs New
- **Before:** Mostly visual compare + informal **`st.write`** of mean/max diff.
- **After:** **Identity hashes**, **metrics row**, **interpretation**, **max-normalized** diff display, **optional** dual histograms.

### Impact
- Stronger **interpretability**, **research explanation**, and **demo clarity**.

### Breaking Changes
- None

### How to Verify
- SAMPLE grid → **Per-patch statistics** → confirm **Hash** differs across files.
- **Image Similarity Analysis** → pick two patches → **MAD** & **Max Diff** > 0, **SSIM** < 1, **verdict** matches thresholds, **Normalized Difference Map** visible; enable **Histogram comparison**.

---

## Update 51 - Enhanced Visualization for SAR Differences
**Date:** 2026-04-04  
**Type:** UI Improvement

### What Changed
- **`demo/streamlit_app.py`** (inside **SAMPLE dataset grid viewer**): helpers **`contrast_stretch_display_float01`**, **`sample_patch_stats`**, **`sample_pair_diff_display`** (display / stats only — **no inference changes**).
- Optional **contrast-stretch** for the thumbnail row; **📈 Per-patch statistics** expander with **`st.write`** mean / std / min / max per file.
- **🔬 Compare two patches**: dual **`st.selectbox`**, side-by-side A & B plus **|A − B|** with **p99** display stretch; **`st.write`** of mean and max absolute difference.
- **🔍 Zoom crop & histogram**: ROI sliders **`y1:y2`**, **`x1:x2`**, full patch + contrast-stretched crop, crop stats, optional **matplotlib** histogram (**32** bins, **[0, 1]**).

### Why
- SAR speckle makes raw thumbnails look alike; **stats, difference maps, ROI zoom, and stretching** make numerical differences interpretable in the demo.

### Old vs New
- **Before:** Grid thumbnails only.
- **After:** Stats, pairwise diff map, zoom, optional histogram, optional stretched thumbnails.

### Impact
- Clearer **interpretability** and stronger **demo** quality.

### Breaking Changes
- None

### How to Verify
- Open SAMPLE grid expander → enable **Contrast-stretch thumbnails** → open **Compare two patches** and confirm **difference map** lights up for different files → **Zoom crop** and optional histogram.

---

## Update 50 - SAMPLE grid / batch thumbnails (Streamlit display fix)
**Date:** 2026-04-04  
**Type:** Bugfix (Streamlit UI)

### What Changed
- **`demo/streamlit_app.py`** — SAMPLE **grid preview** no longer uses **`st.columns` → `st.image`** in a nested loop. Each row uses **`st.image([...], caption=[...])`** (official multi-image API).
- Added **`load_sample_noisy_patch_float01`** — **`os.path.join(noisy_dir, fn)`** + **`cv2.imread`** + **`np.ascontiguousarray`** so each patch is an independent buffer.
- **Batch denoise** thumbnails: each cell wrapped in **`st.container(key=...)`** so Streamlit does not reuse the wrong media slot; **`batch_images`** session list uses **`np.ascontiguousarray`** per entry.

### Why
- Nested **`st.image`** inside **`st.columns`** loops can produce **identical thumbnails** for different arrays (delta / layout identity issue), even when filenames and numpy data differ.

### Breaking Changes
- None

### How to Verify
- Open SAMPLE grid with **N ≥ 2** distinct PNGs — thumbnails must differ; run **batch denoising** and confirm output tiles match distinct inputs.

---

## Update 49 - Metrics for Dataset Images
**Date:** 2026-04-04  
**Type:** Evaluation Fix

### What Changed
- **`demo/streamlit_app.py`** — PSNR / SSIM / ENL and the comparison **Ground Truth** column no longer require **`image_source == 'synthetic'`**. They run whenever **`clean_image`** is present in **`st.session_state`** and is **not `None`** (covers **SAMPLE** pairs and any future synthetic path that stores a reference).

### Why
- SAMPLE patches ship with **paired clean + noisy**; the reference was stored but **reference metrics were gated** on a synthetic-only flag.

### Old vs New
- **Before:** Metrics and GT column only if **`image_source == 'synthetic'`** (and **`clean_image`** was never set for SAMPLE).
- **After:** Metrics and GT whenever a **usable clean reference** exists in session state.

### Impact
- **Stronger evaluation** in the demo for bundled dataset images.

### Breaking Changes
- None

### How to Verify
- **Load selected sample from SAMPLE dataset** → **Run Denoising** → confirm **Performance Metrics** (PSNR, SSIM, ENL) and **Image Comparison** ground-truth panel appear.
- Upload-only flow (no **`clean_image`**) still shows **No ground truth** and no reference metrics.

---

## Update 48 - Batch Dataset Processing
**Date:** 2026-04-04  
**Type:** Feature

### What Changed
- In **`📊 SAMPLE dataset grid viewer`** (`demo/streamlit_app.py`): after the preview grid, added **Batch denoising (SAMPLE)** — **`st.slider("Batch denoise count", 1, min(8, #loaded), …)`** taking the **first N** of the current grid selection.
- Each rerun syncs **`st.session_state["batch_images"]`** (list of noisy **`float32`** patches in **[0, 1]**) and **`batch_image_names`**; changing the selection clears stale **`batch_denoise_results`** via a filename signature.
- **`Run Batch Denoising`** runs **`SARDenoiseService.denoise_numpy`** per image with the sidebar **method** and **`build_streamlit_denoise_kw(with_direct_uncertainty=False)`**, **`include_blind_qa=False`** for lighter compute; results stored in **`batch_denoise_results`** and shown in a **3-column** grid (errors per tile if any).

### Why
- Supports **multi-patch inference** on the bundled SAMPLE set without replacing the single-image **Run Denoising** column.

### Old vs New
- **Before:** Grid preview only; denoising was effectively **one image** at a time in the main workflow (plus unrelated **ZIP** batch upload elsewhere).
- **After:** Optional **batch denoise** for up to **8** SAMPLE patches with a results grid in the same expander.

### Impact
- Closer to **small-batch operational** checks while keeping a hard cap on work per click.

### Breaking Changes
- None

### How to Verify
- Open the SAMPLE grid expander → set **Number of images** and **Batch denoise count** → **Run Batch Denoising** → confirm denoised grid matches count and respects sidebar method; change selection and confirm old results clear until you run again.

---

## Update 47 - Dataset Grid Viewer
**Date:** 2026-04-04  
**Type:** UI Feature

### What Changed
- Added **`📊 SAMPLE dataset grid viewer`** expander in `demo/streamlit_app.py`: **`st.slider("Number of images", 1, …, default)`** with max **`min(12, #PNG)`**, default **`min(4, max)`**.
- Renders a **3-column** grid of **noisy** patches (`test_patches/noisy`) with filenames as captions.
- **First N (sorted)** is the default selection; **Load N random samples** stores a random subset in **`session_state`**; **Show first N (sorted)** clears it. Changing the slider clears a stale random selection.

### Why
- Users can **scan the SAMPLE dataset** without relying on a single preview image.

### Old vs New
- **Before:** One image at a time in the main workspace (plus optional single-patch load).
- **After:** Optional **multi-image grid** in a separate expander; single-image **selectbox + Load selected sample** unchanged.

### Impact
- Better **dataset understanding** for demos and qualitative checks.

### Breaking Changes
- None

### How to Verify
- Run Streamlit → open **SAMPLE dataset grid viewer** → move **Number of images** and confirm the grid updates (first N sorted).
- Click **Load N random samples** and confirm thumbnails change; **Show first N (sorted)** restores deterministic order.
- Confirm **Select Sample Image** + **Load selected sample** still drives denoising as before.

---

## Update 46 - Dataset Selector for SAMPLE
**Date:** 2026-04-04  
**Type:** UI Improvement

### What Changed
- Replaced random dataset loading with a **sorted** file list from `data/sample_sar/processed/test_patches/clean` and a **`st.selectbox("Select Sample Image", …)`** control.
- **Load selected sample from SAMPLE dataset** button loads the **matching** clean + noisy PNG pair and sets **`session_state`**: `clean_image`, `noisy_image`, `image_source = 'sample'`.
- Modified `demo/streamlit_app.py` only; inference pipeline and upload flow unchanged.

### Why
- Allows **controlled** dataset browsing instead of **random** sampling; improves **reproducibility** and usability.

### Old vs New
- **Before:** `random.choice(clean_files)` — one unpredictable patch per click.
- **After:** User picks a filename from the dropdown, then loads that patch.

### Impact
- Better usability and reproducibility for demos and thesis evaluation.

### Breaking Changes
- None

### How to Verify
- Run Streamlit (`streamlit run demo/streamlit_app.py` from project root as documented in README).
- Open the sidebar SAMPLE section: confirm **Select Sample Image** lists all `*.png` files from the clean folder (sorted).
- Choose different names and click **Load selected sample from SAMPLE dataset**; confirm input/clean update and denoising still runs.
- Upload an image: confirm upload still overrides display and `image_source` as before.

---

## Update 45 - **Blind (no-reference) QA** — `evaluators/blind_qa.py` + service / API / Streamlit
**Date:** 2026-04-04  
**Type:** Evaluation / API / Demo

### What Changed (blind QA metrics)
- **Added** `evaluators/blind_qa.py` — **`compute_blind_qa(noisy, denoised)`** returns JSON-serializable floats:
  - **`enl_homogeneous_median`** — median **mean²/var** over low-gradient blocks (homogeneous speckle proxy; fallback global).
  - **`edge_preservation_vs_input`** — EPI-style **Σ(GₙGₐ)/(ΣGₙ²+ε)** (Sobel magnitudes; **no clean reference**).
  - **`variance`**, **`std`**, **`variance_log`** on the denoised image.
- **Updated** `inference/service.py` — optional **`include_blind_qa`** on **`denoise_numpy`** (default **False**); merges **`meta["blind_qa"]`** or **`meta["blind_qa_error"]`** on failure. **Training / `evaluate_sample.py` unchanged.**
- **Updated** `api/main.py` — query **`include_blind_qa`** (default **False**); honored only when **`response_format=json`**; **`meta`** in JSON gains **`blind_qa`** when requested.
- **Updated** `demo/streamlit_app.py` — sidebar **“Blind QA metrics (no-reference)”** (default on for **Run Denoising**); metrics row + raw JSON expander.
- **Added** `tests/test_blind_qa.py`, **`tests/test_api_smoke.py`** — **`test_denoise_json_include_blind_qa`**.

### Why This Change Was Needed
- **No-reference evaluation** for uploaded / operational imagery where **no ground truth** exists.

### Old vs New
- **Before:** **PSNR/SSIM/ENL** in the demo only with **synthetic clean** reference; API JSON had no blind diagnostics.
- **After:** Optional **blind_qa** block for qualitative research and **API** introspection.

### Impact
- **Research + usability** for thesis discussion of smoothing vs structure; metrics are **indicative**, not calibrated across sensors.

### How to Verify
- `pytest tests/test_blind_qa.py tests/test_api_smoke.py -q`
- `curl` **`POST /v1/denoise`** with **`response_format=json&include_blind_qa=true`**
- Streamlit → enable checkbox → **Run Denoising** → **Blind QA** section

---

## Update 43 - Streamlit **GeoTIFF** upload → **`denoise_geotiff`** → download
**Date:** 2026-04-04  
**Type:** Web UI / Geospatial

### What Changed (GeoTIFF UI)
- **Updated** `demo/streamlit_app.py` — expander **“GeoTIFF — georeferenced tile denoise”** in the input column:
  - **`st.file_uploader`** for **`.tif` / `.tiff`** (single file).
  - **`st.number_input`** for **tile size** (passed to **`denoise_geotiff`** as **`tile_size`**, **`overlap=0`** unchanged).
  - Writes upload to a **`tempfile.TemporaryDirectory`**, validates with **`rasterio.open`** (**CRS** + **single band**) before calling **`inference.geotiff.denoise_geotiff`** with **`make_tile_denoise_fn`** and **`SARDenoiseService`** (same checkpoint / infer-config pattern as **`scripts/denoise_geotiff.py`**).
  - Output read from temp **`denoised.tif`** into session state; **`st.download_button`** (**`image/tiff`**) for the result.
  - **500 MB** upload cap for the demo; clear errors if **rasterio** is missing.

### Why This Change Was Needed
- **Real SAR usage** is often **GeoTIFF** with georeferencing; PNG-only demos miss CRS-aware workflows.

### Old vs New
- **Before:** Streamlit path was **PNG/JPEG**-only; GeoTIFF was **CLI** (**`scripts/denoise_geotiff.py`**).
- **After:** Optional **browser upload** and **download** of a denoised **GeoTIFF** without changing **`inference/geotiff.py`**.

### Impact
- **Real-world usability** for small chips; **core GeoTIFF module untouched** (UI integration only).

### How to Verify
- `streamlit run demo/streamlit_app.py` → **GeoTIFF** expander → upload a **single-band, CRS** GeoTIFF → **Run** → **Download denoised GeoTIFF** → open in QGIS/rasterio and confirm CRS/transform.

---

## Update 42 - Streamlit **async job history** (read-only `data/jobs/`)
**Date:** 2026-04-04  
**Type:** Web UI / Ops / Traceability

### What Changed (history UI)
- **Updated** `demo/streamlit_app.py`:
  - **`load_job_history_entries()`** — scans **`api.storage.jobs_root()`** (default **`data/jobs`**, env **`SAR_JOBS_DIR`**) for subdirectories containing **`meta.json`**; loads **`meta.json`** via **`job_storage.read_meta`** and **`status.json`** via **`job_storage.read_status`** (read-only); sorts by directory **mtime**, newest first (cap **100**).
  - Section **“Async job history”** above the footer: metric **Jobs shown**, per-job **expander** (short id, **status**, **method** from meta), folder mtime, **`st.json`** of meta, optional **`st.error`** for **`status.json`** **`error`** field.
  - **`st.download_button`** for **`output.png`**, **`meta.json`**, **`status.json`**, and **`uncertainty.png`** when present (unique widget keys per **`job_id`**).

### Why This Change Was Needed
- **Traceability** and **debugging** for **FastAPI + Redis/RQ** jobs without shell access to **`data/jobs/<id>/`**.

### Old vs New
- **Before:** Job artifacts existed on disk but the Streamlit demo gave **no visibility** into past async runs.
- **After:** A simple **dashboard** to inspect status/meta and **download** artifacts.

### Impact
- **Better UX** for operators and thesis demos; **queue/worker code unchanged** (UI only reads files).

### How to Verify
- Create jobs via **`POST /v1/jobs`** with queue enabled → open Streamlit → **Async job history** → expand a job → download files.

---

## Update 41 - Streamlit **batch processing** (multi-upload → temp dir → ZIP + `manifest.csv`)
**Date:** 2026-04-04  
**Type:** Web UI / Demo / Scalability

### What Changed (batch processing)
- **Updated** `demo/streamlit_app.py`:
  - **`st.expander`** **“Batch processing — multiple images → ZIP”** with a **second** **`st.file_uploader(..., accept_multiple_files=True)`** (single-image uploader unchanged).
  - **Process batch & build ZIP**: one **`SARDenoiseService`**, **`build_streamlit_denoise_kw(False)`** (no Direct TTA), **one file at a time** (read bytes → numpy → denoise → write PNG → **`del`** large arrays); **`tempfile.TemporaryDirectory()`** for outputs.
  - **`manifest.csv`**: columns **`filename`**, **`method`**, **`time_seconds`** (original upload name, sidebar method, per-image wall time).
  - **`zipfile.ZipFile`** (**`ZIP_DEFLATED`**) bundles PNGs + manifest; **`st.download_button`** serves the archive; bytes stored in **`st.session_state`** until a new successful batch overwrites them.
  - Limits: **40** files max, **20 MB** per file; **`safe_batch_png_name`** for safe arcnames.

### Why This Change Was Needed
- **Scalability** for folders of chips: operators and thesis experiments need bulk export without scripting.

### Old vs New
- **Before:** Only single-file upload into session state / one-off denoise.
- **After:** Optional batch lane with **ZIP** artifact and **CSV** manifest; single flow untouched.

### Impact
- **Real-world usability** for small batches; bounded memory via sequential processing and temp disk.

### How to Verify
- `streamlit run demo/streamlit_app.py` → expand **Batch processing** → upload several PNGs → **Process batch & build ZIP** → **Download batch ZIP** → inspect **`manifest.csv`** and denoised PNGs.

---

## Update 40 - Streamlit **Run All Methods** (TV + Direct + ADMM, one run)
**Date:** 2026-04-04  
**Type:** Web UI / Demo / Evaluation

### What Changed (multi-method run)
- **Updated** `demo/streamlit_app.py`:
  - **`build_streamlit_denoise_kw(with_direct_uncertainty)`** — single builder for **`SARDenoiseService.denoise_numpy`** kwargs from the sidebar (used by both single **Run Denoising** and multi-method run).
  - **`normalize_images_shared_range()`** — linear **[0, 1]** stretch using **one global min/max** across all passed arrays (fair side-by-side brightness).
  - **`MULTI_METHOD_ORDER`** — **TV Denoising** → **Direct Denoising** → **ADMM-PnP-DL**.
  - Second button **“Run All Methods”** next to **Run Denoising**; one **`SARDenoiseService`** instance; sequential **`denoise_numpy`** calls with **`time.perf_counter()`** per method.
  - Session state **`multi_method_comparison`**: list of **`{name, display, seconds}`**; full-width **three columns** with **`st.metric`** inference time each.
  - Multi-method path disables Direct TTA uncertainty (**`with_direct_uncertainty=False`**) for speed and comparable three-way output.

### Why This Change Was Needed
- Research and demos need **aligned** classical vs DL vs ADMM results on the **same chip** without manually switching the sidebar method three times.

### Old vs New
- **Before:** Only one method per **Run Denoising** click; no built-in **timed** trio comparison or **shared** display normalization.
- **After:** One click runs all three; **shared min–max** visualization; **per-method wall time** shown.

### Impact
- **Better evaluation** workflow for reports and parameter tuning; **no duplicated inference logic** (same service + shared kwargs builder).

### How to Verify
- `streamlit run demo/streamlit_app.py` → load input → **Run All Methods** → confirm three columns, times, and consistent-looking intensity scale.

---

## Update 39 - Streamlit **Comparison Dashboard** (blend, diff, view toggles)
**Date:** 2026-04-04  
**Type:** Web UI / Demo

### What Changed (UI additions)
- **Updated** `demo/streamlit_app.py` — new section **“Comparison Dashboard”** (full width, after denoising results when both **`noisy_image`** and **`denoised_image`** exist in session state):
  - **`st.radio`** view modes: **Interactive blend (slider)**, **Original (noisy)**, **Denoised**, **Absolute difference**.
  - **`st.slider`** for blend **α** in **[0, 1]** when blend mode is selected; display **`α×denoised + (1−α)×noisy`** via **`numpy`** (**`comparison_blend_noisy_denoised`**).
  - **Difference** view: **`|noisy − denoised|`** with **99th-percentile** stretch for visibility (**`comparison_abs_diff_map`**).
  - Shape mismatch guard (warning, no crash).

### Why This Change Was Needed
- Easier qualitative analysis of denoising: cross-fade noisy→denoised and highlight change regions without re-running inference.

### Old vs New
- **Before:** Static side-by-side strips only (e.g. **“Image Comparison”** row); no cross-fade, no dedicated diff visualization in one place.
- **After:** One dashboard with **interactive opacity/blend**, **explicit toggles** for noisy / denoised / diff, and short captions explaining the math.

### Impact
- **Better analysis** for demos, debugging ADMM parameters, and thesis figures; **inference path unchanged** (UI-only helpers).

### How to Verify
- `streamlit run demo/streamlit_app.py` → run denoising → open **Comparison Dashboard** → exercise all four views and the blend slider.

---

## Update 38 - `GET /health`: `direct_infer_backend` + `onnx_path_set`
**Date:** 2026-04-04  
**Type:** Observability (Step 10 / ops)

### What Changed
- **Updated** `api/main.py` — **`GET /health`** adds **`direct_infer_backend`** (**`pytorch`** \| **`onnx`**, default **`pytorch`** when unset) and **`onnx_path_set`** (**bool**: merged **`onnx_path`** string is non-empty; does not check the file on disk). Values come from **`service_options_from_merged()`** (same as the API service).
- **Updated** `tests/test_api_smoke.py` — asserts new fields; **`test_health_direct_backend_reflects_sar_backend`** with **`SAR_BACKEND`** / **`SAR_ONNX_PATH`**.
- **Updated** `verify_system.py` — FastAPI smoke validates **`direct_infer_backend`** and **`onnx_path_set`**.
- **Updated** **`README.md`**, **`docs/DEVELOPMENT.md`**, **`PROJECT_IMPROVEMENT_REPORT.md`**.

### Why This Change Was Needed
- Operators probing liveness can see which Direct path the process will prefer without reading **`configs/infer/`** or env on the host.

### Breaking Changes
- None; new JSON keys only (clients should already treat unknown **`/health`** keys as optional per prior docs).

### How to Verify
- `pytest tests/test_api_smoke.py -q`
- `python verify_system.py`

---

## Update 37 - `service_options_from_merged()` for worker, GeoTIFF CLI, Streamlit
**Date:** 2026-04-04  
**Type:** Config consistency (follow-up to Update 36)

### What Changed
- **Added** `api/infer_config.service_options_from_merged()` — returns **`infer_backend`** / **`onnx_path`** from merged infer YAML + env (same rules as **`get_merged()`**).
- **Updated** `api/deps.py` — uses **`service_options_from_merged()`** instead of duplicating key reads.
- **Updated** `workers/tasks.py`, **`scripts/denoise_geotiff.py`**, **`demo/streamlit_app.py`** — pass those options into **`SARDenoiseService`** so **`backend` / `onnx_path`** in **`configs/infer/`** apply to async jobs, GeoTIFF tiling, and the demo (still overridden by **`SAR_BACKEND`** / **`SAR_ONNX_PATH`** when set).
- **Added** `tests/test_infer_config.py` — **`test_service_options_from_merged`**.
- **Updated** **`docs/DEVELOPMENT.md`** (infer defaults sentence).

### Why This Change Was Needed
- FastAPI was the only entrypoint reading infer **`backend`** / **`onnx_path`**; workers and CLIs should behave the same when operators set **`SAR_INFER_CONFIG`** or repo defaults.

### Breaking Changes
- None.

### How to Verify
- `pytest tests/test_infer_config.py -q`
- `ruff check api workers scripts/denoise_geotiff.py` (Streamlit demo is not in CI ruff scope)

---

## Update 36 - Infer YAML: `backend` + `onnx_path` (API / `SARDenoiseService`)
**Date:** 2026-04-04  
**Type:** Config / serving (Step 8 hygiene)

### What Changed
- **Updated** `api/infer_config.py` — **`get_merged()`** reads **`backend`** and **`onnx_path`** from the infer YAML (normalized), with **`SAR_BACKEND`** and **`SAR_ONNX_PATH`** overriding when set.
- **Updated** `configs/infer/default.yaml` — explicit **`backend: pytorch`** and commented **`onnx_path`** example; header lists **`SAR_BACKEND`** / **`SAR_ONNX_PATH`**.
- **Updated** `inference/service.py` — **`SARDenoiseService(..., infer_backend=..., onnx_path=...)`** (optional); **`_effective_inference_backend`** / **`_effective_onnx_path`** (env wins, then config, then checkpoint **`.onnx`**).
- **Updated** `api/deps.py` — passes merged **`backend`** / **`onnx_path`** into the service singleton.
- **Updated** `tests/test_infer_config.py`, **`tests/test_onnx_export_parity.py`** — merge + kwargs smoke.
- **Updated** **`README.md`**, **`docs/DEVELOPMENT.md`**.

### Why This Change Was Needed
- ONNX Direct mode could only be selected via environment variables; **`configs/infer/`** is now the single place for defaults consistent with **`device`** / **`checkpoint`**.

### Breaking Changes
- None. Default **`backend: pytorch`** matches prior implicit default.

### How to Verify
- `pytest tests/test_infer_config.py tests/test_onnx_export_parity.py -q`

---

## Update 35 - `verify_system.py`: `/health` + `/ready` TestClient smoke
**Date:** 2026-04-04  
**Type:** Ops / QA

### What Changed
- **Updated** `verify_system.py` — after a successful **`import api.main`**, runs **`TestClient`** **`GET /health`** and **`GET /ready`** (default env: queue off → **200** + **`status`** **`ok`** / **`ready`**). On failure prints **`❌`** and **`return False`** so **CI** fails; FastAPI **import** failure remains a non-fatal **⚠️** skip (optional stack).

### Why This Change Was Needed
- Catches regressions in **liveness** / **readiness** wiring without starting **uvicorn** or **pytest**-only coverage gaps.

### Breaking Changes
- Environments that import **`api.main`** but where **TestClient** smoke fails now fail **`verify_system.py`** (expected in normal dev/CI with **FastAPI** installed).

### How to Verify
- `python verify_system.py` → expect **`✅ FastAPI /health + /ready smoke (TestClient)`** after the import line.

---

## Update 34 - `GET /ready` (queue readiness)
**Date:** 2026-04-04  
**Type:** Ops / API

### What Changed
- **Updated** `api/main.py` — **`GET /ready`**: if **`SAR_USE_QUEUE`** is not **`1`**, returns **`200`** with **`{"status":"ready","queue":"disabled"}`**; if queue mode is on, requires **`REDIS_URL`** and **`redis.Redis.from_url(...).ping()`** (2s connect timeout), else **503** with **`not_ready`** and a **`detail`** string. Not gated by **`SAR_API_KEY`** (same as **`/health`**).
- **Updated** `tests/test_api_smoke.py` — queue off, missing **`REDIS_URL`**, mocked Redis OK / failure, **`/ready`** without API key when key configured.
- **Updated** `api/deps.py` — documents **`/ready`** and **`SAR_USE_QUEUE` / `REDIS_URL`**.
- **Updated** **`README.md`**, **`docs/DEVELOPMENT.md`**, **`PROJECT_IMPROVEMENT_REPORT.md`**.

### Why This Change Was Needed
- Kubernetes-style **readiness** separate from **liveness** (**`/health`** always **200** when the process is up).

### Breaking Changes
- None.

### How to Verify
- `pytest tests/test_api_smoke.py -q`
- `curl -s http://127.0.0.1:8000/ready` with and without **`SAR_USE_QUEUE=1`** + Redis.

---

## Update 33 - `sar.py streamlit` (unified CLI)
**Date:** 2026-04-04  
**Type:** Tooling

### What Changed
- **Updated** `sar.py` — subcommand **`streamlit`**: default **`python -m streamlit run demo/streamlit_app.py`** + forwarded args; if the first forwarded token is **`version`**, **`help`**, **`--version`**, or **`-h`**, runs **`python -m streamlit <those args>`** instead (e.g. **`sar.py streamlit version`**).
- **Added** `tests/test_sar_cli.py` — **`test_sar_streamlit_version`** (**`pytest.importorskip("streamlit")`**).
- **Updated** `docs/DEVELOPMENT.md`, **`README.md`**, **`CONTRIBUTING.md`** — **`python sar.py streamlit`** beside **`streamlit run`**.

### Why This Change Was Needed
- One entrypoint for the primary **demo** alongside **train**, **eval**, **api**, and **worker**.

### Breaking Changes
- None.

### How to Verify
- `pytest tests/test_sar_cli.py -q`
- `python sar.py streamlit version`

---

## Update 32 - Optional `SAR_API_KEY` for `/v1/*` (Step 10 partial)
**Date:** 2026-04-04  
**Type:** Security / Hardening

### What Changed
- **Added** `api/api_key.py` — **`OptionalApiKeyMiddleware`**: when **`SAR_API_KEY`** is set, **`/v1/denoise`** and **`/v1/jobs*`** require **`Authorization: Bearer <key>`** or **`X-API-Key: <key>`**; comparison uses **`hmac.compare_digest`** (length mismatch → reject). **`/health`**, **`/docs`**, **`/openapi.json`** unchanged.
- **Updated** `api/main.py` — register middleware (outer stack); restored **lifespan** **`SAR_LOG_LEVEL`** handling for **`sar.api.access`**.
- **Updated** `api/deps.py` — documents **`SAR_API_KEY`**.
- **Added** `tests/test_api_smoke.py` — **401** without key, **200** with Bearer, **`/health`** without key when API key configured.
- **Updated** `README.md`, **`docs/DEVELOPMENT.md`**.

### Why This Change Was Needed
- Roadmap **Step 10** minimal **authn** for exposed APIs without forcing keys in local dev (env unset → prior behavior).

### Breaking Changes
- **None** unless **`SAR_API_KEY`** is set in your environment; then clients must send the header.

### How to Verify
- `pytest tests/test_api_smoke.py -q`

---

## Update 31 - JSON access logging + `X-Request-ID` (Step 10 partial)
**Date:** 2026-04-04  
**Type:** Observability

### What Changed
- **Added** `api/access_log.py` — **`JsonAccessLogMiddleware`**; when **`SAR_ACCESS_LOG_JSON`** is truthy, logs one **JSON** object per request (`event`, `method`, `path`, `status_code`, `elapsed_ms`, `request_id`) on **`sar.api.access`**; sets response **`X-Request-ID`** (incoming header or new UUID).
- **Updated** `api/main.py` — register middleware; **lifespan** sets **`sar.api.access`** level from **`SAR_LOG_LEVEL`** (default **INFO**).
- **Updated** `api/deps.py` — documents **`SAR_ACCESS_LOG_JSON`**, **`SAR_LOG_LEVEL`**.
- **Updated** `tests/test_api_smoke.py` — **`test_json_access_log_and_request_id`**.
- **Updated** `README.md`, **`docs/DEVELOPMENT.md`**.

### Why This Change Was Needed
- Roadmap **hardening**: structured request traces without new dependencies; complements **`GET /health`** metadata.

### Breaking Changes
- None when **`SAR_ACCESS_LOG_JSON`** is unset (default quiet).

### How to Verify
- `pytest tests/test_api_smoke.py -q`
- `SAR_ACCESS_LOG_JSON=1 uvicorn api.main:app` then `curl -v http://127.0.0.1:8000/health`

---

## Update 30 - `/health` metadata + roadmap refresh in `PROJECT_IMPROVEMENT_REPORT.md`
**Date:** 2026-04-04  
**Type:** Observability / Docs

### What Changed
- **Added** `api/meta.py` — **`git_sha_short()`** via **`SAR_GIT_SHA`** or **`git rev-parse --short HEAD`** (best-effort).
- **Updated** `api/main.py` — **`GET /health`** returns **`status`**, **`version`** (`API_VERSION`), and optional **`git_sha`**.
- **Updated** `api/deps.py` — documents **`SAR_GIT_SHA`**.
- **Updated** `tests/test_api_smoke.py` — health asserts **version**; **`git_sha`** optional.
- **Updated** `Dockerfile` — **`ARG`/`ENV SAR_GIT_SHA`** for **`docker build --build-arg SAR_GIT_SHA=…`**.
- **Updated** `README.md` — health response shape.
- **Updated** `PROJECT_IMPROVEMENT_REPORT.md` — **section 4** quick wins / medium / major items and **section 7** step-by-step roadmap aligned with current repo (done vs TBD).

### Why This Change Was Needed
- Load balancers and deploy scripts benefit from a stable **version** field; **git SHA** supports audit without adding a metrics stack. The report was drifting from delivered work.

### Breaking Changes
- **`GET /health`** JSON is no longer exactly `{"status":"ok"}`; clients should treat unknown keys as optional.

### How to Verify
- `pytest tests/test_api_smoke.py::test_health -q`
- `curl -s http://127.0.0.1:8000/health` after `uvicorn api.main:app`

---

## Update 29 - `sar.py api` + `sar.py worker` (uvicorn + RQ)
**Date:** 2026-04-04  
**Type:** Tooling

### What Changed
- **Updated** `sar.py` — subcommands **`api`** (`python -m uvicorn api.main:app …`) and **`worker`** (`python -m rq.cli worker sar_denoise …`); special dispatch alongside existing script forwards.
- **Updated** `tests/test_sar_cli.py` — **`--help`** smoke for **`api`** and **`worker`**.
- **Updated** `docs/DEVELOPMENT.md`, **`README.md`**, **`CONTRIBUTING.md`** — examples for HTTP API and Redis worker.

### Why This Change Was Needed
- Completes the **single entrypoint** story for **running** the stack, not only train/eval scripts.

### Breaking Changes
- None.

### How to Verify
- `pytest tests/test_sar_cli.py -q`
- `python sar.py api -- --help` and `python sar.py worker -- --help`

---

## Update 28 - `configs/infer/default.yaml` + `api/infer_config` (env overrides)
**Date:** 2026-04-04  
**Type:** Feature / Config

### What Changed
- **Added** `configs/infer/default.yaml` — optional defaults for **`device`**, **`model_type`** (slug), and commented **`checkpoint`**.
- **Added** `api/infer_config.py` — **`get_merged()`**, **`model_type_label()`**, **`effective_checkpoint_str()`**; load path from **`SAR_INFER_CONFIG`** or repo **`configs/infer/default.yaml`**; **env** (`SAR_DEVICE`, `SAR_CHECKPOINT`, `SAR_MODEL_TYPE`) overrides YAML.
- **Updated** `api/deps.py` — **`get_service()`** reads merged **`device`** / **`checkpoint`**.
- **Updated** `api/constants.py` — **`default_model_type_label()`** delegates to **`infer_config.model_type_label()`**.
- **Updated** `api/main.py` — Direct path uses **`effective_checkpoint_str()`** for **`direct_checkpoint`**.
- **Updated** `api/jobs.py` — job **`meta`** **`checkpoint`** / **`device`** from infer merge.
- **Added** `tests/test_infer_config.py`.
- **Updated** `docs/DEVELOPMENT.md`, **`README.md`** (HTTP snippet + tree).

### Why This Change Was Needed
- Improvement report **config package** thread: versionable inference defaults beside **`configs/train/`**, without breaking env-based deploys.

### Breaking Changes
- None when the YAML file is absent or leaves **`checkpoint`** unset; behavior matches prior **env-only** defaults.

### How to Verify
- `pytest tests/test_infer_config.py tests/test_api_smoke.py -q`

---

## Update 27 - Dockerfile + Compose `api` profile (deployment hygiene)
**Date:** 2026-04-04  
**Type:** DevOps / Docs

### What Changed
- **Added** `Dockerfile` — **Python 3.11-slim**; **`pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu`** so **`requirements.txt`** NumPy/stack pins and **CPU** torch resolve in one step; **`uvicorn api.main:app`** on **8000** (`SAR_DEVICE=cpu`).
- **Added** `.dockerignore` — drops **`data/`**, **`notebooks/`**, **`results/`**, **`fixes/`**, **`.git`**, **`*.pth`**, caches, and bulky docs so the build context stays small.
- **Updated** `docker-compose.yml` — optional service **`api`** with **`profiles: [api]`** (does not run on plain `docker compose up redis`); **`REDIS_URL`** points at compose **Redis**; **`depends_on: redis`**.
- **Updated** `README.md` — Docker run snippet + checkpoint mount note; project tree lists **Dockerfile**.
- **Updated** `docs/DEVELOPMENT.md` — short **Docker** section.

### Why This Change Was Needed
- `PROJECT_IMPROVEMENT_REPORT.md` / MLOps thread: a **reproducible** way to run the API without a local Python venv; aligns with existing **Redis** compose service.

### Breaking Changes
- None. Default workflow remains `uvicorn` on the host.

### How to Verify
- `docker compose --profile api build` (or `up --build`) when Docker is available.
- `curl http://127.0.0.1:8000/health` after **`--profile api up`**.

---

## Update 26 - CONTRIBUTING + extended `sar.py` (train-sample, ablation, ONNX compare)
**Date:** 2026-04-04  
**Type:** Docs / Tooling

### What Changed
- **Added** `CONTRIBUTING.md` — PR expectations: **docs/DEVELOPMENT.md**, **`sar.py`**, **pytest** / **ruff** (CI-aligned), **`fixes/updates.md`** for notable changes.
- **Updated** `sar.py` — forwards **`train-sample`** → **`train_sample.py`**, **`compare-onnx`** → **`scripts/compare_pytorch_onnx.py`**, **`ablation-grid`** → **`scripts/run_ablation_grid.py`**, **`ablation-md`** → **`scripts/ablation_to_markdown.py`**.
- **Updated** `README.md` — Contributing links **CONTRIBUTING.md**; project tree lists it.
- **Updated** `docs/DEVELOPMENT.md` — new **`sar.py`** examples.
- **Updated** `tests/test_sar_cli.py` — **`--help`** smoke for the new commands.

### Why This Change Was Needed
- `PROJECT_IMPROVEMENT_REPORT.md` **immediate fix**: a visible **contributing** guide; **`sar.py`** should cover the other common scripts so one entrypoint stays useful.

### Breaking Changes
- None.

### How to Verify
- `pytest tests/test_sar_cli.py -q`
- `python sar.py train-sample --help`

---

## Update 25 - Unified `sar.py` CLI (roadmap step 2)
**Date:** 2026-04-04  
**Type:** Tooling

### What Changed
- **Added** `sar.py` (repo root) — **argparse** subcommands that **forward** argv to **`train_improved.py`**, **`evaluate_sample.py`**, **`verify_system.py`**, **`scripts/capture_baseline.py`**, **`scripts/export_onnx.py`**, **`scripts/denoise_geotiff.py`** (`train`, `eval`, `verify`, `capture-baseline`, `export-onnx`, `denoise-geotiff`).
- **Added** `tests/test_sar_cli.py` — smoke tests for missing command, unknown command, and `--help` passthrough.
- **Updated** `docs/DEVELOPMENT.md` — **Unified CLI** section.
- **Updated** `README.md` — project tree lists **`sar.py`**.
- **Updated** `.github/workflows/ci.yml` — **ruff** includes **`sar.py`**.

### Why This Change Was Needed
- `PROJECT_IMPROVEMENT_REPORT.md` **step 2**: one discoverable entrypoint alongside existing **`configs/train/*.yaml`**; avoids new dependencies (stdlib **subprocess** only).

### Breaking Changes
- None. All underlying scripts keep their CLIs.

### How to Verify
- `pytest tests/test_sar_cli.py -q`
- `python sar.py` → nonzero exit + command map on stderr
- `python sar.py eval --help`

---

## Update 24 - Frozen baseline metrics + TV evaluation fix (roadmap step 3)
**Date:** 2026-04-04  
**Type:** Docs / Tooling / Bug Fix

### What Changed
- **Added** `results/baseline/README.md` — how to regenerate the pinned **TV-only** SAMPLE evaluation and provenance.
- **Added** `results/baseline/metrics.json` — committed snapshot (`baseline_id: sample_tv_v1`) with **trimmed** aggregates + `provenance` (git SHA, UTC time, command, `data_dir`).
- **Added** `scripts/capture_baseline.py` — reads `evaluation_results.json`, writes `metrics.json` via `results_to_jsonable(..., include_per_patch_lists=False)`.
- **Updated** `.gitignore` — **`results/baseline/staging/`** (intermediate plots/JSON from regen commands).
- **Updated** `algos/evaluation.py` — **TV Denoising** uses `tv_denoise` inside **`torch.enable_grad()`** (outer loop uses `no_grad`); fixes `evaluate_sample.py --methods tv`.
- **Added** `tests/test_evaluation_tv_smoke.py` — synthetic dict-style loader, **batch_size 1**, short TV iterations.
- **Updated** `docs/DEVELOPMENT.md`, **`README.md`** project tree — point to **`results/baseline/`**.
- **Updated** `.github/workflows/ci.yml` — **ruff** includes **`algos/evaluation.py`** (TV path change).

### Why This Change Was Needed
- `PROJECT_IMPROVEMENT_REPORT.md` **step 3**: publish a **frozen baseline** with exact command and revision context; TV path was **broken** in batch evaluation (callable + autograd).

### Breaking Changes
- None.

### How to Verify
- `pytest tests/test_evaluation_tv_smoke.py -q`
- `python scripts/capture_baseline.py --help`
- Re-run full regen per `results/baseline/README.md` if you change TV defaults or the test split.

---

## Update 23 - Developer quickstart + inference/FastAPI checks in verify_system (roadmap hygiene)
**Date:** 2026-04-04  
**Type:** Docs / Tooling

### What Changed
- **Added** `docs/DEVELOPMENT.md` — one-page **environment**, **recommended training** (`train_improved.py` + YAML configs), **evaluation** (`evaluate_sample.py`), demo/API, and **verify / pytest / ruff** commands.
- **Updated** `README.md` — **Contributing** section links to `docs/DEVELOPMENT.md`; **Project structure** tree lists `docs/DEVELOPMENT.md`.
- **Updated** `verify_system.py` — after ADMM smoke, runs **`SARDenoiseService.denoise_numpy`** on a random **64×64** patch with **TV Denoising** (no checkpoint); **try**-imports **`api.main`** and prints success or a **non-fatal** skip message.

### Why This Change Was Needed
- `PROJECT_IMPROVEMENT_REPORT.md` calls for a single **recommended train + eval** entry point for reproducibility and onboarding; CI already guards tests, but **`verify_system.py`** did not exercise the extracted **inference** layer.

### Breaking Changes
- None. `verify_system.py` can still fail only inside the existing **basic functionality** `try` (now including inference TV smoke).

### How to Verify
- Open `docs/DEVELOPMENT.md` and run the listed commands that apply to your machine.
- `python verify_system.py` → expect **Inference service (TV path…)** and either **FastAPI application import** or a **skipped** line if FastAPI deps are missing.

---

## Update 21 - Streamlit: TTA uncertainty for Direct denoising (Upgrade 5)
**Date:** 2026-04-04  
**Type:** Feature / UX

### What Changed
- **Updated** `demo/streamlit_app.py` — when **Denoising Method** is **Direct Denoising**, sidebar adds **Pixelwise uncertainty (TTA)** and **TTA passes** (1–4). On run, calls `denoise_numpy(..., return_uncertainty=True, uncertainty_tta_passes=…)` and shows **TTA-mean denoised** next to an **uncertainty heatmap** (`uncertainty_to_vis_u8`). Comparison row adds an uncertainty column when present; **meta** summary line for mean/max/passes.

### Why This Change Was Needed
- **Upgrade 5:** surface the same TTA uncertainty path already in the API (`Update 20`) inside the primary demo so reviewers can see spread maps without `curl`.

### Breaking Changes
- None.

### How to Verify
- `streamlit run demo/streamlit_app.py` → Direct Denoising → enable **Pixelwise uncertainty (TTA)** → run on an image **≥ ~32×32** (e.g. SAMPLE patch or upload).

---

## Update 19 - Residual U-Net backbone (Upgrade 4)
**Date:** 2026-04-04  
**Type:** Feature / Research

### What Changed
- **Added** `models/res_unet.py` — **`ResUNet`** with **residual double-conv** blocks (distinct checkpoint keys: `maxpool_conv.1.conv1.*` vs vanilla U-Net’s `double_conv`).
- **Updated** `models/unet.py` — **`create_model`** accepts **`res_unet`**, **`resunet`**, **`Res-U-Net`** (via slug).
- **Updated** `inference/service.py` — **`_detect_arch_from_state_keys`** detects **ResUNet** checkpoints; **`_sidebar_model_slug`** maps **Res-U-Net** → **`res_unet`**.
- **Updated** `evaluate_sample.py` — **`--model_type res_unet`** and checkpoints under **`checkpoints_sample_res_unet/`** (and matching unrolled path).
- **Added** `configs/train/res_unet_default.yaml` — train into **`checkpoints_res_unet/`** for ablations vs **`configs/train/default.yaml`**.
- **Updated** `demo/streamlit_app.py` — **Res-U-Net** in model selectbox.
- **Updated** `api/constants.py`, `api/main.py` — default / docs for **Res-U-Net** via **`SAR_MODEL_TYPE=res_unet`**.
- **Updated** `scripts/export_onnx.py`, `scripts/compare_pytorch_onnx.py`, `scripts/denoise_geotiff.py` — **`res_unet`** / **Res-U-Net** choices.
- **Updated** `configs/ablation/example_manifest.yaml` — commented **res_unet** eval template.
- **Added** `tests/test_res_unet_smoke.py`; **extended** `tests/test_train_smoke_synthetic.py`.

### Why This Change Was Needed
- **Upgrade 4:** stronger inductive bias / gradient flow than vanilla U-Net while keeping the **same ADMM / training / ONNX** interfaces.

### Old vs New Comparison
| Aspect | Before | After |
|--------|--------|-------|
| Denoisers | `unet`, `dncnn` | + **`res_unet`** |
| Auto-detect load | U-Net vs DnCNN only | + **ResUNet** from state dict |

### Breaking Changes
- None; existing checkpoints and defaults unchanged.

### How to Verify
- `pytest tests/test_res_unet_smoke.py tests/test_train_smoke_synthetic.py -q`
- `python -c "from models.unet import create_model; import torch; m=create_model('res_unet',n_channels=1,noise_conditioning=False); print(m(torch.randn(1,1,64,64)).shape)"`
- Train (with SAMPLE data): `python train_improved.py --config configs/train/res_unet_default.yaml`

---

## Update 20 - TTA uncertainty + API JSON + GeoTIFF two-band (Upgrade 3)
**Date:** 2026-04-04  
**Type:** Feature / Research

### What Changed
- **Added** `inference/uncertainty.py` — **test-time augmentation (TTA)** stack (identity, hflip, vflip, rot90): **mean prediction** as denoised output when uncertainty is requested; **pixelwise std** across transforms as uncertainty map. PyTorch and **ONNX Direct** paths.
- **Updated** `inference/types.py` — optional **`uncertainty`** key in **`denoise_result_dict`**.
- **Updated** `inference/service.py` — **`denoise_numpy(..., return_uncertainty=False, uncertainty_tta_passes=4)`**; **Direct denoising only**; ADMM/TV with flag → warning. **`denoiser.to(device)`** in Direct PyTorch path (correct device placement). **`meta`**: `uncertainty_mean`, `uncertainty_max`, `uncertainty_tta_passes`.
- **Updated** `api/main.py` — **`response_format=png|json`**; **`include_uncertainty`** (requires **`json`** + **Direct**); JSON body: **`denoised_png`**, **`uncertainty_png`** (base64), **`meta.inference_ms`**.
- **Updated** `inference/geotiff.py` — **`make_tile_denoise_fn_with_uncertainty`**, **`denoise_geotiff_two_band`** (band1 denoised, band2 uncertainty × tile scale).
- **Updated** `workers/tasks.py`, **`api/jobs.py`** — **`include_uncertainty`**, **`uncertainty_tta_passes`**; optional **`GET /v1/jobs/{id}/uncertainty`**.
- **Updated** `inference/__init__.py` exports.
- **Added** `tests/test_uncertainty_tta.py`; **extended** `tests/test_api_smoke.py`.

### Why This Change Was Needed
- **Upgrade 3:** risk-aware outputs without retraining; TTA is a standard **cheap** proxy for spread under input symmetry.

### Old vs New Comparison
| Aspect | Before | After |
|--------|--------|-------|
| Direct output | Point estimate only | Optional **std map** + summary stats |
| API | PNG only | **`response_format=json`** + uncertainty |
| GeoTIFF | Single band | Optional **two-band** writer |

### Breaking Changes
- None for default **`return_uncertainty=False`** / **`response_format=png`**.

### How to Verify
- `pytest tests/test_uncertainty_tta.py tests/test_api_smoke.py -q`
- `curl` example: `POST /v1/denoise?method=Direct%20Denoising&response_format=json&include_uncertainty=true` with image upload.

---

## Update 22 - Ablation manifest + grid runner + Markdown summary (Upgrade 6)
**Date:** 2026-04-04  
**Type:** Feature / Research

### What Changed
- **Added** `configs/ablation/example_manifest.yaml` — template listing multiple **`evaluate_sample.py`** runs (methods, `save_dir`, `data_dir`, task-metrics toggles). Copy to **`configs/ablation/manifest.yaml`** for local use (not committed as mandatory).
- **Added** `scripts/run_ablation_grid.py` — reads manifest, runs each configuration, writes **`results/ablation/aggregate_ablation.json`** (manifest path + per-run **`evaluation_results.json`** payloads). Supports **`--dry-run`** and **`--device`**.
- **Added** `evaluators/ablation_report.py` — **`render_markdown()`** for aggregate → table (PSNR/SSIM/ENL + task metrics when present).
- **Added** `scripts/ablation_to_markdown.py` — CLI wrapper; optional **`--out`** file.
- **Added** `tests/test_ablation_tools.py` — table rendering smoke test (no dataset).

### Why This Change Was Needed
- **Upgrade 6** in the roadmap: systematic, reproducible evaluation sweeps and paper-style tables without ad hoc shell loops.

### Implementation Details
- Each manifest **run** is one subprocess to **`evaluate_sample.py`** with an isolated **`--save_dir`** under e.g. **`results/ablation/<id>/`**.
- Aggregate JSON shape: `{ "manifest", "schema_version", "runs": { "<ablation_id>": { "<Method Name>": { metrics... } } } }`.

### Old vs New Comparison
| Aspect | Before | After |
|--------|--------|-------|
| Multi-eval | Manual commands | YAML-driven grid + merged JSON |
| Reporting | Ad hoc | `ablation_to_markdown.py` table |

### Breaking Changes
- None.

### How to Verify
- `pytest tests/test_ablation_tools.py -q`
- `python scripts/run_ablation_grid.py --manifest configs/ablation/example_manifest.yaml --dry-run`
- With SAMPLE data: copy example → `manifest.yaml`, run without `--dry-run`, then `python scripts/ablation_to_markdown.py --aggregate results/ablation/aggregate_ablation.json`

---

## Update 18 - Cross-dataset / OOD paired-folder evaluation (Upgrade 2)
**Date:** 2026-04-04  
**Type:** Feature / Research

### What Changed
- **Added** `data/paired_folder_loader.py` — **`PairedFolderPatchDataset`** + **`create_paired_eval_dataloader`**: expects **`data_dir/clean/`** and **`data_dir/noisy/`** with **matching filenames** (PNG/JPEG); same batch keys as SAMPLE (`clean`, `noisy`, `noise_level`).
- **Added** `data/ood_paired/.gitkeep` — placeholder tree for an optional second corpus (user adds images).
- **Updated** `evaluate_sample.py` — **`--data_layout sample|paired`** (default `sample`); **`--dataset_tag`** for run logs; manifest fields **`data_layout`**, **`dataset_tag`**; paired path skips SAMPLE stats and uses only the paired eval loader.
- **Updated** `scripts/run_ablation_grid.py` — manifest keys **`data_layout`**, **`dataset_tag`** forwarded to CLI.
- **Updated** `configs/ablation/example_manifest.yaml` — commented **OOD paired** run template.
- **Added** `tests/test_paired_folder_loader.py`.

### Why This Change Was Needed
- **Train / calibrate on A, report metrics on B** without changing checkpoints — supports **generalization** and **ablation** tables (Upgrade 2).

### Implementation Details
- **No automatic intensity normalization** — document radiometry per dataset; mismatched scaling affects absolute PSNR.
- Checkpoints remain **`checkpoints_sample_*`**; OOD eval reuses the same weights (frozen transfer).

### Old vs New Comparison
| Aspect | Before | After |
|--------|--------|-------|
| Eval data | SAMPLE tree only | + **paired** folder layout for second corpus |
| Run metadata | data_dir only | + `data_layout`, `dataset_tag` |

### Breaking Changes
- None; default **`--data_layout sample`** preserves prior behavior.

### How to Verify
- `pytest tests/test_paired_folder_loader.py -q`
- With paired data: `python evaluate_sample.py --data_layout paired --data_dir data/ood_paired --methods unet --save_dir results/ood_smoke --no-run-log`

---

## Update 17 - Task-based evaluation (structure / edge metrics)
**Date:** 2026-04-04  
**Type:** Feature / Research

### What Changed
- **Added** `evaluators/task_metrics.py` — gradient-magnitude **Pearson correlation** (GSM-corr), **edge preservation index** (EPI), **Laplacian MSE** vs clean, **gradient-map SSIM** (structure-focused proxy).
- **Updated** `algos/evaluation.py` — `SARDenoisingEvaluator.evaluate_method(..., include_task_metrics=True)` aggregates task metrics per method; extended **`compare_methods`** table; new plot **`task_metric_comparison.png`**; **`denoiser.eval()`** only if `hasattr(denoiser, "eval")` (safer for non-`nn.Module` baselines).
- **Updated** `evaluate_sample.py` — flag **`--no-task-metrics`** to skip task metrics; run log **`manifest.json`** includes **`task_metrics_enabled`**; copies **`task_metric_comparison.png`** into `results/runs/<id>/plots/` when present; minor lint cleanups (unused imports / f-string).
- **Added** `tests/test_task_metrics.py` — sanity checks on identical patches, blur vs correlation, finite EPI.

### Why This Change Was Needed
- PSNR/SSIM alone are weak proxies for **perceived structure** and downstream SAR utility; task-style **edge/gradient alignment** metrics support **research-grade** evaluation narratives (Upgrade 1 in the multi-semester roadmap).

### Implementation Details
- Metrics use **scikit-image** Sobel / Laplacian filters on **per-patch** clean vs denoised float arrays in `[0,1]`.
- **`results_to_jsonable`** already JSON-sanitizes new keys; **`metrics.json`** in run logs automatically includes task aggregates when enabled.

### Old vs New Comparison
| Aspect | Before | After |
|--------|--------|-------|
| Evaluation | PSNR, SSIM, ENL only | + GSM-corr, EPI, Laplacian MSE, Grad SSIM |
| Run artifacts | `method_comparison.png`, distributions | + `task_metric_comparison.png` (when task metrics on) |
| `evaluate_sample.py` | No task toggle | `--no-task-metrics` for faster runs |

### Impact
- Stronger **structure preservation** signal for comparing ADMM vs direct denoising; better alignment with **academic** reporting expectations.

### Breaking Changes
- None. Default remains **task metrics enabled**; JSON/plots gain **additional fields** (consumers that assumed only PSNR/SSIM/ENL keys should ignore unknown keys).

### How to Verify
- `pytest tests/test_task_metrics.py -q`
- `python evaluate_sample.py --methods tv --no-run-log` (smoke; TV path may still need a callable denoiser—unchanged legacy behavior)
- Full SAMPLE eval (with data): `python evaluate_sample.py --no-run-log` and inspect `results_sample/task_metric_comparison.png` and `evaluation_results.json` for `gsm_corr_mean`, etc.

---

## Update 16 - ONNX export + Direct path ORT backend (Step 09)
**Date:** 2026-04-04  
**Type:** Feature

### What Changed
- **`requirements.txt`:** `onnx>=1.15.0`, `onnxruntime>=1.16.0` (comment: optional serving).
- **Added** `inference/onnx_export.py` — `export_denoiser_to_onnx` (legacy exporter: **`dynamo=False`** for PyTorch 2.9+ without `onnxscript`); validates with `onnx.checker`.
- **Added** `inference/onnx_backend.py` — `ONNXDirectDenoiser` (1 or 2 ONNX inputs for noise conditioning).
- **Added** `scripts/export_onnx.py`, `scripts/compare_pytorch_onnx.py`.
- **Added** `models_artifacts/.gitkeep`.
- **Updated** `inference/service.py` — **`SAR_BACKEND=pytorch|onnx`** (default pytorch); **`SAR_ONNX_PATH`** / **`.onnx` direct_checkpoint**; **`SAR_ONNX_EP`** ORT providers; **Direct Denoising** uses ORT when enabled; ADMM/TV unchanged.
- **Updated** `api/deps.py` — documents `SAR_BACKEND`, `SAR_ONNX_PATH`, `SAR_ONNX_EP`.
- **Added** `tests/test_onnx_export_parity.py`.
- **Updated** `README.md` — ONNX subsection + tree; **`inference/__init__.py`** exports.

### Why This Change Was Needed
- Portable denoiser artifact and optional faster Direct inference (`fixes/09_optimize_model_serving.md`).

### Breaking Changes
- None; default backend remains PyTorch.

### How to Verify
- `pytest tests/test_onnx_export_parity.py -q`
- `python scripts/export_onnx.py --help`

## Update 15 - Async job queue Redis + RQ (Step 08)
**Date:** 2026-04-04  
**Type:** Feature

### What Changed
- **`requirements.txt`:** `redis>=5.0.0`, `rq>=1.16.0`.
- **Added** `docker-compose.yml` — `redis:7-alpine` on port 6379.
- **Added** `api/constants.py` — `MAX_UPLOAD_BYTES`, `DenoiseMethodEnum`, `default_model_type_label` (shared with jobs router).
- **Added** `api/storage.py` — `data/jobs/<job_id>/` (`input.png`, `output.png`, `meta.json`, `status.json`); `SAR_JOBS_DIR` override.
- **Added** `api/jobs.py` — `POST /v1/jobs`, `GET /v1/jobs/{id}`, `GET /v1/jobs/{id}/result` (PNG); RQ queue **`sar_denoise`**; 20 MB upload limit; HTTP 425 until result ready.
- **Added** `workers/tasks.py` — `run_denoise_job(job_dir)` for RQ.
- **Updated** `api/main.py` — lifespan Redis ping when `SAR_USE_QUEUE=1`; include `jobs` router when **`SAR_USE_QUEUE=1`**; middleware also limits **`POST /v1/jobs`** body size.
- **Added** `tests/test_job_storage.py`.
- **Updated** `README.md` — async jobs subsection + tree.

### Why This Change Was Needed
- Non-blocking API for long ADMM runs and worker scaling (`fixes/08_add_job_queue.md`).

### Environment
- **`SAR_USE_QUEUE=1`** — enable job routes.
- **`REDIS_URL`** — e.g. `redis://localhost:6379/0`.
- Worker: **`rq worker sar_denoise --url $REDIS_URL`** from repo root.

### Breaking Changes
- None; default **`SAR_USE_QUEUE=0`** keeps Step 07 sync-only surface.

### How to Verify
- `pytest tests/test_job_storage.py tests/test_api_smoke.py -q`
- With Redis + worker: `curl -F file=@img.png "http://127.0.0.1:8000/v1/jobs?method=TV%20Denoising"` then poll status and fetch `/result`.

## Update 14 - Cap SciPy <1.14 (gensim / conda base)
**Date:** 2026-04-04  
**Type:** Bug Fix / Env

### What Changed
- **`requirements.txt`:** `scipy>=1.8.0,<1.14` (comment: conda **gensim 4.3.x** requires `scipy<1.14`).
- **`README.md`:** pip conflict note updated to mention the SciPy cap.

### Why This Change Was Needed
- After `pip install -r requirements.txt`, pip reported **gensim** vs **scipy 1.17** incompatibility on typical Anaconda **base** installs.

### Breaking Changes
- Environments that **require** SciPy **≥1.14** for other tools must relax the pin locally.

### How to Verify
- `pip install -r requirements.txt` then `python -c "import scipy; print(scipy.__version__)"` — expect **1.13.x** (or any **1.x** with minor **< 14**).

## Update 13 - README: shell paste + pip timeout troubleshooting
**Date:** 2026-04-04  
**Type:** Docs

### What Changed
- **`README.md`:** HTTP API section — safer copy-paste (no prose that can break into a bogus `omit` command); split `uvicorn --reload` into its own block; note `unset SAR_CHECKPOINT` for TV-only.
- **`README.md`:** Installation troubleshooting — **`command not found: omit`** (split comments / pasted prose); **pip `ReadTimeoutError` / DNS** — `--default-timeout=120`, optional `conda-forge rasterio` then `pip install -r requirements.txt`.

### Why This Change Was Needed
- Users pasting multi-line snippets hit `zsh: command not found: omit`; pip installs failed on slow/flaky networks when downloading large wheels.

## Update 12 - FastAPI HTTP layer (Step 07)
**Date:** 2026-04-04  
**Type:** Feature

### What Changed
- **`requirements.txt`:** `fastapi>=0.109.0`, `uvicorn[standard]>=0.27.0`, `python-multipart>=0.0.6`.
- **Added** `api/deps.py` — lazy singleton `SARDenoiseService` from `SAR_DEVICE`, optional `SAR_CHECKPOINT` (improved + simple paths), `SAR_MODEL_TYPE` default label; `reset_service_for_tests()`.
- **Added** `api/main.py` — `GET /health`, `POST /v1/denoise` (multipart PNG/JPEG in → PNG out), `DenoiseMethodEnum` query validation, ADMM/direct query params, **20 MB** upload cap (Content-Length middleware + post-read check), `Depends(get_service)`.
- **Added** `api/__init__.py`.
- **Added** `tests/test_api_smoke.py` — `TestClient` health, TV denoise PNG round-trip, invalid image 400.
- **Updated** `README.md` — HTTP API subsection + tree.

### Why This Change Was Needed
- Machine-readable integration surface for automation and future Redis queue (Step 08) (`fixes/07_build_api_layer.md`).

### Breaking Changes
- None; Streamlit unchanged.

### How to Verify
- `pip install -r requirements.txt` then `pytest tests/test_api_smoke.py -q`
- `uvicorn api.main:app --port 8000` and `curl -s http://127.0.0.1:8000/health`

## Update 11 - GeoTIFF windowed I/O (Step 06)
**Date:** 2026-04-04  
**Type:** Feature

### What Changed
- **`requirements.txt`:** `rasterio>=1.3.0`.
- **Added** `inference/geotiff.py` — `denoise_geotiff` (single band, `assert` CRS, windowed read/write, per-tile min–max, nodata preserved, `overlap != 0` rejected), `make_tile_denoise_fn` wiring `SARDenoiseService`.
- **Added** `scripts/denoise_geotiff.py` — CLI (`--in`, `--out`, `--checkpoint`, `--method`, ADMM/direct args, `--tile_size`, `--overlap`).
- **Updated** `inference/service.py` — `denoise_numpy(..., direct_checkpoint=...)` for Direct path override (CLI / GeoTIFF).
- **Updated** `inference/__init__.py` — exports `denoise_geotiff`, `make_tile_denoise_fn`.
- **Added** `tests/test_geotiff_smoke.py` — tiny GeoTIFF fixture, TV denoise, CRS/transform/nodata equality, overlap rejection.
- **Updated** `README.md` — GeoTIFF subsection + tree.

### Why This Change Was Needed
- Production SAR stacks are often GeoTIFF/COG with georeferencing; windowing avoids loading full rasters into memory (`fixes/06_add_geotiff_support.md`).

### How to Verify
- `pip install -r requirements.txt` then `pytest tests/test_geotiff_smoke.py -q`
- `python scripts/denoise_geotiff.py --help`

## Update 10 - Inference core extracted from Streamlit (Step 05)
**Date:** 2026-04-03  
**Type:** Feature / Refactor

### What Changed
- **Added** `inference/` — `types.py` (`DenoiseMethod`, `denoise_result_dict`), `service.py` (`SARDenoiseService`, `preprocess_noisy_array`, `load_admm_denoiser`, `replay_messages_streamlit`, optional `denoise_file`).
- **Updated** `demo/streamlit_app.py` — denoise button path calls `SARDenoiseService.denoise_numpy` and replays `(level, text)` messages via Streamlit (same behavior as inlined logic).
- **Added** `tests/test_inference_service_smoke.py` — import, preprocess, TV denoising smoke (no checkpoint), fake Streamlit replay.
- **Updated** `README.md` project tree (`inference/`, `tests/`).
- **Updated** `algos/admm_pnp.py` — `TVDenoiser.compute_tv_loss` supports 2D `[H,W]` (Streamlit / inference) and 4D `[B,C,H,W]`; `tv_denoise` uses `x.clamp_(0, 1)` in place so the optimized leaf keeps `requires_grad` across iterations (TV path was effectively broken after the first step).

### Why This Change Was Needed
- Importable inference for FastAPI (Step 07), GeoTIFF (Step 06), batch CLIs, and tests without running Streamlit (`fixes/05_extract_inference_core.md`).

### Breaking Changes
- None for Streamlit UX; checkpoint paths and defaults match the previous demo.

### How to Verify
- `python -c "from inference.service import SARDenoiseService; print('ok')"`
- `pytest tests/test_inference_service_smoke.py -q`
- `streamlit run demo/streamlit_app.py` — run denoising once per method as before.

## Update 9 - U-Net / DnCNN noise conditioning: 4D expand fix
**Date:** 2026-04-03  
**Type:** Bug Fix

### What Changed
- **`models/unet.py`** (`UNet` and `DnCNN` `forward`): batched `noise_level` `[B]` was turned into 3D `[B, 1, 1]` then `.expand(B, 1, H, W)`, which PyTorch rejects (ndim / broadcast rules). Noise is now normalized to length `B`, then **`reshape(B, 1, 1, 1).expand(B, 1, H, W)`**; extra handling for `[B, 1]` and `(1, B)` layouts.

### Why This Change Was Needed
- **`evaluate_sample.py`** with full test batches (e.g. 13 samples) crashed before saving `results/runs/` (`RuntimeError` at `expand`).

### How to Verify
- `python evaluate_sample.py --methods unet --model_type unet --data_dir data/sample_sar/processed` completes and writes `results/runs/<run_id>/`.

## Update 8 - Evaluation run logging (Step 04)
**Date:** 2026-04-03  
**Type:** Feature

### What Changed
- **Added** `evaluators/run_logger.py` — `EvaluationRunContext` creates `results/runs/<YYYYMMDD_HHMMSS>_<git_sha>/` with `manifest.json`, `metrics.json`, and `plots/`; `get_git_sha_short()` falls back to `nogit` when git is unavailable.
- **Added** `evaluators/__init__.py`.
- **Updated** `algos/evaluation.py` — **`results_to_jsonable`** (JSON-safe floats/ints/nested structures; `inf`/`nan` → `null`; optional **`include_per_patch_lists`** to omit `*_values` keys).
- **Updated** `evaluate_sample.py` — after saving to `--save_dir`, writes run bundle unless **`--no-run-log`**; copies comparison PNGs into `runs/.../plots/` when present; manifest includes argv, paths, device, checkpoint paths if files exist.
- **Updated** `README.md` — Phase 3 / Evaluate sections and project tree document `results/runs/` and `evaluators/`.

### Why This Change Was Needed
- Reproducible evaluation artifacts for reports and future CI baselines (`fixes/04_build_evaluation_pipeline.md`).

### Breaking Changes
- None. Run logging is on by default; use `--no-run-log` for previous behavior (no `results/runs/` writes).

### How to Verify
- `python evaluate_sample.py --methods unet --model_type unet --data_dir data/sample_sar/processed` (with data present), then `ls results/runs/` and `cat results/runs/*/manifest.json`.

## Update 7 - README: explain conda-base pip conflict warnings
**Date:** 2026-04-03  
**Type:** Docs

### What Changed
- **`README.md`** (NumPy/Matplotlib troubleshooting): note that **successful** `pip install -r requirements.txt` may still print **dependency conflicts** for **aext-***, **gensim**, **s3fs**—unrelated to this repo; use **`verify_system.py`** as ground truth; dedicated env avoids noise.

### Why This Change Was Needed
- Users saw `ERROR: pip's dependency resolver... gensim... s3fs...` after a **successful** install and assumed failure (terminal log 6).

### Breaking Changes
- None.

## Update 6 - Pin matplotlib/opencv/pillow for NumPy 1.x stack
**Date:** 2026-04-03  
**Type:** Bug Fix / Docs

### What Changed
- **`requirements.txt`:** `opencv-python-headless>=4.6.0,<4.13` (4.13+ requires NumPy 2); `matplotlib>=3.5.0,<3.10` (avoids resolver pulling NumPy 2 with Matplotlib 3.10+); explicit **`Pillow>=9.0.0,<12`** and **`packaging>=20.0,<25`** for Streamlit 1.45.x compatibility.
- **`README.md`:** troubleshooting rewritten — **warn against** `pip install --force-reinstall matplotlib` alone; recommend **`pip install -r requirements.txt --upgrade --force-reinstall`** as one atomic recovery.

### Why This Change Was Needed
- User followed “downgrade numpy then reinstall matplotlib”; bare `matplotlib` upgraded **NumPy to 2.4.4**, **Pillow 12**, **packaging 26**, conflicting with **scipy**, **numba**, **gensim**, and **streamlit** (terminal log).

### Breaking Changes
- None for code; environments that relied on OpenCV 4.13+ or Matplotlib 3.10+ with NumPy 2 must adjust pins locally.

### How to Verify
- `pip install -r requirements.txt` then `python -c "import numpy; assert numpy.__version__.startswith('1.')"`

## Update 5 - NumPy / Matplotlib compatibility and lazy plotting imports
**Date:** 2026-04-03  
**Type:** Bug Fix / Docs

### What Changed
- **`requirements.txt`:** pinned **`numpy>=1.21.0,<2.0`** with a short comment (avoids NumPy 2.x vs wheels compiled for NumPy 1.x, e.g. conda `base` + older Matplotlib binaries).
- **`trainers/improved_trainer.py`:** removed module-level `matplotlib` import; **`plot_training_curves`** now imports `matplotlib` / `pyplot` lazily and sets **`matplotlib.use("Agg")`** before plotting (headless-safe).
- **`trainers/pipeline.py`:** moved `matplotlib` import to just before the results figure (same **Agg** pattern) so importing the pipeline module does not load Matplotlib.
- **`README.md`:** new **Troubleshooting: NumPy / Matplotlib import error** under Installation with `pip install "numpy>=1.21.0,<2.0" --force-reinstall` and `matplotlib` reinstall commands.

### Why This Change Was Needed
- Users hitting **`ImportError`** when running `train_improved.py` / `train_sample.py` because **NumPy 2.2.x** was active while **Matplotlib** (or extensions) expected **NumPy 1.x** ABI (terminal log: `numpy.core.multiarray failed to import`).

### Implementation Details
- Lazy imports do **not** remove the need for a consistent NumPy/Matplotlib pair when training runs (plotting still loads Matplotlib); the pin + reinstall is the real fix for broken envs.

### Impact
- Fresh **`pip install -r requirements.txt`** installs resolve to NumPy 1.26.x line; existing broken conda installs have a documented recovery path.

### Breaking Changes
- Environments that **require** NumPy 2-only stacks must relax the pin locally or reinstall Matplotlib/scipy stacks built for NumPy 2.

### How to Verify
- `pip install -r requirements.txt` then `python -c "import numpy; import matplotlib.pyplot; print(numpy.__version__)"`
- `python train_improved.py --help` (imports without triggering `improved_trainer` plotting path).

## Update 4 - YAML training configs (Step 03)
**Date:** 2026-04-03  
**Type:** Feature / Improvement

### What Changed
- **Added** `pyyaml>=6.0` to `requirements.txt`.
- **Added** `trainers/config_loader.py` — `load_training_yaml`, `dict_to_training_config`, `training_config_from_yaml_path`, `resolve_training_device` (unknown YAML keys ignored; only `_TRAINING_KEYS` merged).
- **Added** `configs/train/default.yaml`, `configs/train/sample.yaml` (`checkpoint_dir: checkpoints_sample`), `configs/train/smoke.yaml` (1 epoch, `batch_size: 2`, `num_workers: 0`).
- **Updated** `trainers/config_dataclass.py` — `device` default is **`auto`** (resolved to `cuda`/`cpu` in `run_training` via `resolve_training_device`).
- **Updated** `trainers/pipeline.py` — calls `resolve_training_device` at start so `TrainingConfig(device="auto")` always becomes a concrete device.
- **Updated** `train_improved.py` — `argparse` with optional `--config PATH`; loads YAML or uses bare `TrainingConfig()` (same behavior as before when omitted).
- **Updated** `train_sample.py` — optional `--config`; runs improved pipeline then returns (dataset download + `data_dir` check); processed **before** device/mode prints to avoid misleading CLI output.
- **Updated** `README.md` — Phase 2 / Usage training examples and project tree (`configs/`, new `trainers/` files).

### Why This Change Was Needed
- Reproducible, diffable training runs and CI-friendly smoke config (`fixes/03_create_config_system.md`).

### Implementation Details
- YAML loaded with `yaml.safe_load` only. Path fields accept strings relative to repo root.

### Impact
- Runs can be cited as e.g. `git rev-parse HEAD` + `configs/train/default.yaml`.

### Breaking Changes
- **None** — `python train_improved.py` without flags still runs the same defaults; device resolution now goes through `auto` → `cuda`/`cpu` inside `run_training`.

### How to Verify
- `pip install pyyaml` (or `pip install -r requirements.txt`)
- `python -c "from pathlib import Path; from trainers.config_loader import training_config_from_yaml_path; print(training_config_from_yaml_path(Path('configs/train/default.yaml')))"`
- `python train_improved.py --help` / `python train_improved.py --config configs/train/smoke.yaml` (requires SAMPLE data at configured `data_dir`)

## Update 3 - Unify improved training pipeline (Step 02)
**Date:** 2026-04-03  
**Type:** Refactor / Improvement

### What Changed
- **Added** `trainers/improved_trainer.py` — `ImprovedTrainer` moved out of `train_improved.py` (same loss, optimizer, checkpoint format).
- **Added** `trainers/config_dataclass.py` — `TrainingConfig` with defaults matching historical `train_improved.py` (batch 8, patch 128, workers 2, epochs 20, lr `2e-4`, `checkpoints_improved`, U-Net `noise_conditioning=False`).
- **Added** `trainers/pipeline.py` — `run_training(config)` performs dataloaders → train → quick PSNR test → `improved_training_results.png`.
- **Added** `trainers/__init__.py` (package marker).
- **Slimmed** `train_improved.py` — builds `TrainingConfig(device=…)` and calls `run_training`; re-exports `ImprovedTrainer`, `TrainingConfig`, `run_training` for compatibility.
- **Extended** `train_sample.py` — new flag **`--use-improved-pipeline`** calls the same `run_training` after dataset checks; **default behavior unchanged** (DenoiserTrainer / UnrolledTrainer, `checkpoints_sample_*`).

### Why This Change Was Needed
- Single canonical implementation for the “improved” SAMPLE-patch trainer, ready for YAML config (Step 03) and less duplication (`fixes/02_unify_training_pipeline.md`).

### Implementation Details
- `trainers.pipeline` prepends repo root to `sys.path` so imports match script-style execution from project root.
- Legacy `train_sample.py` modes do not import the new pipeline unless `--use-improved-pipeline` is set.

### Impact
- One code path for improved training; `python train_improved.py` should match prior behavior (same defaults and artifact paths).

### Breaking Changes
- **None** for default commands. Code that imported `ImprovedTrainer` from `train_improved` still works (`train_improved` re-exports the class).

### How to Verify
- `python -c "from trainers.pipeline import run_training; from train_improved import ImprovedTrainer"`
- `python train_improved.py` (optional: interrupt after one epoch if you only smoke-test imports)
- `python train_sample.py --help` → see `--use-improved-pipeline`
- `python train_sample.py` (no flag) → same as before

## Update 2 - Align Streamlit demo path with repository (Step 01)
**Date:** 2026-04-03  
**Type:** Docs / Bug Fix

### What Changed
- Replaced all **incorrect** `demo/app.py` references with **`demo/streamlit_app.py`** across docs and tooling.
- **Modified:** `README.md`, `PROJECT_REPORT.md`, `PROJECT_SUMMARY.md`, `PROJECT_IMPROVEMENT_REPORT.md`, `FINAL_SUMMARY.md`, `IMPROVEMENTS_SUMMARY.md`, `BULLETPROOF_DISPLAY_SUMMARY.md`, `SMART_DISPLAY_SUMMARY.md`, `verify_system.py`, `setup.py`, `test_setup.py`, `run_complete_workflow.py`, `run_demo.py`, `download_sample_dataset.py`, `train_sample.py`, `train.py`, `demo/streamlit_app.py` (inline comment), `fixes/10_add_ci_cd_pipeline.md`.
- **Project structure trees** in `README.md`, `PROJECT_REPORT.md`, and `PROJECT_SUMMARY.md` now list `streamlit_app.py` instead of `app.py`.

### Why This Change Was Needed
- The app only exists as `demo/streamlit_app.py`; docs and `verify_system.py` pointed at a missing `demo/app.py`, causing confusion and false verification failures (`fixes/01_fix_documentation.md`).

### Implementation Details
- String/path updates only; no rename of `streamlit_app.py`, no change to ADMM, models, or Streamlit UI logic.
- `PROJECT_IMPROVEMENT_REPORT.md` updated to mark the doc/entrypoint issue and related roadmap items as **resolved/done**, and adjusted the reproducibility/usability table rows that mentioned the mismatch.

### Impact
- Clone-and-run and printed instructions match the real file; `python verify_system.py` reports **`✅ Streamlit demo: demo/streamlit_app.py`**.

### Breaking Changes
- Anyone who still ran `streamlit run demo/app.py` must use **`streamlit run demo/streamlit_app.py`** (or update Streamlit Cloud “Main file” to `demo/streamlit_app.py`). No Python API breakage.

### How to Verify
- `python verify_system.py` → expect Streamlit demo check passes.
- `rg "demo/app\.py" --glob '!fixes/01_fix_documentation.md'` → should find no stale **usage** in project sources (fix guide `fixes/01_*.md` may still mention `demo/app.py` in examples).

## Update 1 - Initialize structured update log
**Date:** 2026-04-03  
**Type:** Docs

### What Changed
- Added `fixes/updates.md` as the canonical append-only changelog for project changes.
- Established the required section layout (What Changed, Why, Implementation Details, Impact, Breaking Changes, How to Verify) and “latest first” ordering.

### Why This Change Was Needed
- Ongoing refactors and improvements (per `fixes/*.md` and `PROJECT_IMPROVEMENT_REPORT.md`) need a single, auditable trail of what was modified, when, and how to validate it.

### Implementation Details
- New file only; no application code, dependencies, or build steps altered.

### Impact
- Improves traceability, onboarding, and release/debug context for future changes.

### Breaking Changes
- No breaking changes.

### How to Verify
- Confirm the file exists: `ls fixes/updates.md`
- Open `fixes/updates.md` and confirm **Update 1** is present (newer updates are listed above it).
