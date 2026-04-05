"""
Streamlit demo application for ADMM-PnP-DL SAR image denoising
"""
import streamlit as st

# First Streamlit call must run before other heavy imports (Streamlit + Community Cloud).
st.set_page_config(
    page_title="ADMM-PnP-DL SAR Image Denoising",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

import csv
import hashlib
import io
import math
import os
import random
import re
import sys
import tempfile
import time
import zipfile
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Repo root on path before any `api` / `algos` imports.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def _bundled_geotiff_path() -> Path:
    """Prefer real Sentinel-1 RTC chip; fall back to tiny PNG/synthetic sample."""
    d = PROJECT_ROOT / "data" / "sample_geotiff"
    for name in ("sentinel1_rtc_sample.tif", "presentation_sample.tif"):
        p = d / name
        if p.is_file():
            return p
    return d / "sentinel1_rtc_sample.tif"


def _ensure_pyyaml() -> None:
    """Ensure `import yaml` works (Cloud `uv` occasionally omits PyYAML from the env)."""
    for _ in range(2):
        try:
            import yaml  # noqa: F401

            return
        except ImportError:
            import subprocess

            subprocess.run(
                [sys.executable, "-m", "pip", "install", "--no-input", "PyYAML>=6.0"],
                check=False,
                timeout=180,
                capture_output=True,
            )


_ensure_pyyaml()


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


from algos.evaluation import (
    calculate_enl,
    calculate_metrics,
    calculate_psnr,
    calculate_ssim,
)
from evaluators.blind_qa import compute_blind_qa
from api import storage as job_storage
from api.infer_config import get_merged, service_options_from_merged
from inference.geotiff import denoise_geotiff, make_tile_denoise_fn
from inference.service import SARDenoiseService, replay_messages_streamlit
from inference.uncertainty import uncertainty_to_vis_u8
from data.sar_simulation import SARSimulator


def comparison_blend_noisy_denoised(
    noisy: np.ndarray, denoised: np.ndarray, alpha: float
) -> np.ndarray:
    """Blend: ``alpha * denoised + (1 - alpha) * noisy``, clipped to [0, 1]."""
    n = noisy.astype(np.float32)
    d = denoised.astype(np.float32)
    out = float(alpha) * d + (1.0 - float(alpha)) * n
    return np.clip(out, 0.0, 1.0)


def comparison_abs_diff_map(noisy: np.ndarray, denoised: np.ndarray) -> np.ndarray:
    """|noisy − denoised| with 99th-percentile stretch for display in [0, 1]."""
    diff = np.abs(noisy.astype(np.float64) - denoised.astype(np.float64))
    p99 = float(np.percentile(diff, 99)) if diff.size else 1.0
    return np.clip(diff / (p99 + 1e-8), 0.0, 1.0).astype(np.float32)


_GEOTIFF_VIZ_STATE_KEYS = (
    "streamlit_geotiff_in_vis",
    "streamlit_geotiff_out_vis",
    "streamlit_geotiff_diff_vis",
    "streamlit_geotiff_blind_qa",
    "streamlit_geotiff_enl_input",
    "streamlit_geotiff_enl_output",
    "streamlit_geotiff_norm_rmse",
    "streamlit_geotiff_ssim_vs_input",
    "streamlit_geotiff_psnr_vs_input",
    "streamlit_geotiff_method_label",
)


def _clear_geotiff_viz_state() -> None:
    for k in _GEOTIFF_VIZ_STATE_KEYS:
        st.session_state.pop(k, None)


def _format_metric_float(x: float, *, nd: int = 4) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "—"
    return f"{float(x):.{nd}f}"


def _markdown_metric_table(rows: list[tuple[str, str]]) -> None:
    """Full-width table so long labels are not truncated like narrow ``st.metric`` tiles."""
    lines = ["| Metric | Value |", "| :--- | :--- |"]
    for k, v in rows:
        k_esc = str(k).replace("|", "·")
        v_esc = str(v).replace("|", "·")
        lines.append(f"| {k_esc} | {v_esc} |")
    st.markdown("\n".join(lines))


def load_sample_noisy_patch_float01(noisy_dir: str, filename: str):
    """
    Load one noisy SAMPLE patch as float32 in [0, 1].

    Returns a new contiguous array per call (never a shared view), so grids and
    batch lists cannot accidentally alias the same buffer.
    """
    path = os.path.join(noisy_dir, filename)
    if not os.path.isfile(path):
        return None
    arr = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if arr is None:
        return None
    return np.ascontiguousarray(arr.astype(np.float32) / 255.0)


def contrast_stretch_display_float01(img: np.ndarray) -> np.ndarray:
    """Per-image min–max normalize to [0, 1] for display only (inference unchanged)."""
    a = np.asarray(img, dtype=np.float32)
    lo = float(np.min(a))
    hi = float(np.max(a))
    if hi <= lo + 1e-8:
        return np.clip(a, 0.0, 1.0)
    return np.clip((a - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def contrast_stretch_percentile_float01(
    img: np.ndarray, p_low: float = 2.0, p_high: float = 98.0
) -> np.ndarray:
    """
    Display-only stretch for wide-dynamic-range SAR GeoTIFFs.

    Min–max can map almost the whole scene to black when a few pixels are much
    brighter (common for amplitude / gamma0). Percentile limits ignore those tails.
    """
    a = np.asarray(img, dtype=np.float64)
    v = a[np.isfinite(a)]
    if v.size == 0:
        return np.zeros_like(a, dtype=np.float32)
    lo = float(np.percentile(v, p_low))
    hi = float(np.percentile(v, p_high))
    if hi <= lo + 1e-12:
        lo = float(np.min(v))
        hi = float(np.max(v))
    if hi <= lo + 1e-12:
        return np.clip(a.astype(np.float32), 0.0, 1.0)
    out = np.clip((a - lo) / (hi - lo), 0.0, 1.0)
    return np.ascontiguousarray(out.astype(np.float32))


def sample_patch_stats(img: np.ndarray) -> dict:
    """Scalar stats for a single-channel float patch."""
    a = np.asarray(img, dtype=np.float64)
    return {
        "mean": float(np.mean(a)),
        "std": float(np.std(a)),
        "min": float(np.min(a)),
        "max": float(np.max(a)),
    }


def sample_pair_diff_display(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    """|img1 − img2| stretched with 99th-percentile for visible difference maps."""
    d = np.abs(img1.astype(np.float64) - img2.astype(np.float64))
    p99 = float(np.percentile(d, 99)) if d.size else 1.0
    return np.clip(d / (p99 + 1e-8), 0.0, 1.0).astype(np.float32)


def sample_array_hash_prefix(arr: np.ndarray, prefix_len: int = 16) -> str:
    """SHA256 over raw float32 bytes (contiguous); prefix for compact UI."""
    a = np.ascontiguousarray(np.asarray(arr, dtype=np.float32))
    return hashlib.sha256(a.tobytes()).hexdigest()[:prefix_len]


def sample_pair_diff_normalized_by_max(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    """|img1 − img2| normalized by max(|diff|) + ε for display in [0, 1]."""
    d = np.abs(img1.astype(np.float64) - img2.astype(np.float64))
    mx = float(np.max(d)) if d.size else 0.0
    return np.clip(d / (mx + 1e-8), 0.0, 1.0).astype(np.float32)


def similarity_verdict_mad(mad: float) -> tuple[str, str]:
    """Returns (label, level) where level is success | warning | info."""
    if mad < 0.002:
        return "Nearly identical", "warning"
    if mad < 0.02:
        return "Similar", "info"
    return "Clearly different", "success"


def load_job_history_entries(limit: int = 100):
    """
    List async denoise jobs under :func:`~api.storage.jobs_root` (read-only).
    One row per directory that contains ``meta.json``; newest ``mtime`` first.
    """
    root = job_storage.jobs_root()
    if not root.is_dir():
        return root, []
    entries = []
    for p in root.iterdir():
        if not p.is_dir():
            continue
        if not (p / "meta.json").is_file():
            continue
        job_id = p.name
        try:
            meta = job_storage.read_meta(job_id)
        except Exception:
            meta = {"_error": "could not parse meta.json", "job_id": job_id}
        st_obj = job_storage.read_status(job_id) or {}
        status = st_obj.get("status", "unknown")
        err = st_obj.get("error")
        try:
            mtime = p.stat().st_mtime
        except OSError:
            mtime = 0.0
        entries.append(
            {
                "job_id": job_id,
                "path": p,
                "meta": meta,
                "status": status,
                "error": err,
                "mtime": mtime,
            }
        )
    entries.sort(key=lambda e: e["mtime"], reverse=True)
    return root, entries[:limit]


def safe_batch_png_name(original_name: str, index: int) -> str:
    """Filesystem-safe PNG name; preserves order via numeric prefix."""
    stem = Path(original_name).stem
    stem = re.sub(r"[^a-zA-Z0-9._-]+", "_", stem).strip("._-") or "image"
    stem = stem[:60]
    return f"{index:04d}_{stem}.png"


def normalize_multi_method_display(images: list[np.ndarray]) -> list[np.ndarray]:
    """
    Per-panel display stretch for TV / Direct / ADMM side-by-side.

    A single global min–max across methods makes **Direct** look black when its
    amplitude is much smaller than TV/ADMM (e.g. random weights with no checkpoint).
    Percentile stretch per panel matches GeoTIFF preview behavior.
    """
    return [
        contrast_stretch_percentile_float01(np.asarray(x, dtype=np.float32))
        for x in images
    ]


# Order for multi-method run: TV → Direct → ADMM (matches typical classical → DL progression)
MULTI_METHOD_ORDER = ("TV Denoising", "Direct Denoising", "ADMM-PnP-DL")


# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #1f77b4;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stButton > button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        border: none;
        border-radius: 0.5rem;
        padding: 0.5rem 1rem;
        font-size: 1rem;
    }
    .stButton > button:hover {
        background-color: #0d5aa7;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🛰️ ADMM-PnP-DL SAR Image Denoising</h1>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar
st.sidebar.title("Configuration")

# Model selection
model_type = st.sidebar.selectbox(
    "Select Denoiser Model",
    ["U-Net", "DnCNN", "Res-U-Net"],
    help="Choose the deep learning denoiser architecture",
)

# ADMM parameters
st.sidebar.subheader("ADMM Parameters")
max_iter = st.sidebar.slider("Max Iterations", 5, 50, 15, help="Maximum ADMM iterations")
rho_init = st.sidebar.slider("Initial Rho", 0.1, 5.0, 1.0, help="Initial penalty parameter (lower = less over-smoothing)")
alpha = st.sidebar.slider("Alpha", 0.0, 1.0, 0.1, help="Relaxation parameter (lower = less denoiser dominance)")
theta = st.sidebar.slider("Theta", 0.0, 1.0, 0.05, help="Denoising strength parameter (lower = less smoothing)")

# Add log-transform option
use_log_transform = st.sidebar.checkbox("Use Log Transform", False, help="Apply log-transform for better SAR speckle handling")

# Add option to disable denoising entirely
disable_denoising = st.sidebar.checkbox("Disable Denoising (Return Original)", False, help="Skip ADMM processing and return the original noisy image")

# Quality Enhancement Mode
quality_enhancement = st.sidebar.checkbox("Quality Enhancement Mode", False, help="Enable 2-pass denoising with refinement for better results")

# Preset parameter configurations
st.sidebar.subheader("Parameter Presets")
preset = st.sidebar.selectbox(
    "Choose Preset Configuration",
    ["Custom", "Balanced (Recommended)", "Sharp Edges", "Smooth Output", "Conservative"],
    help="Quick parameter presets for different denoising styles"
)

if preset != "Custom":
        if preset == "Balanced (Recommended)":
            max_iter, rho_init, alpha, theta = 15, 1.0, 0.1, 0.05
            st.sidebar.success("✅ Balanced: Anti-over-smoothing with minimal processing")
        elif preset == "Sharp Edges":
            max_iter, rho_init, alpha, theta = 12, 0.8, 0.05, 0.02
            st.sidebar.success("🔪 Sharp Edges: Maximum detail preservation, minimal smoothing")
        elif preset == "Smooth Output":
            max_iter, rho_init, alpha, theta = 20, 1.5, 0.2, 0.1
            st.sidebar.success("🌊 Smooth: Moderate denoising with edge preservation")
        elif preset == "Conservative":
            max_iter, rho_init, alpha, theta = 10, 0.5, 0.05, 0.01
            st.sidebar.success("🛡️ Conservative: Very light processing, preserve original quality")

# Noise parameters
st.sidebar.subheader("Noise Parameters")
speckle_factor = st.sidebar.slider("Speckle Factor", 0.0, 1.0, 0.3, help="Multiplicative speckle noise level")
gaussian_sigma = st.sidebar.slider("Gaussian Noise", 0.0, 0.2, 0.05, help="Additive Gaussian noise level")
psf_sigma = st.sidebar.slider("PSF Sigma", 0.5, 3.0, 1.0, help="Point spread function blur level")

# Method selection
method = st.sidebar.selectbox(
    "Denoising Method",
    ["TV Denoising", "Direct Denoising", "ADMM-PnP-DL"],
    help="Choose the denoising approach (TV is lightweight and needs no checkpoint)",
)

direct_show_uncertainty = False
uncertainty_tta_passes = 4
if method == "Direct Denoising":
    st.sidebar.subheader("Direct: uncertainty (TTA)")
    direct_show_uncertainty = st.sidebar.checkbox(
        "Pixelwise uncertainty (TTA)",
        value=False,
        help="Test-time flips/rotations: std across predictions as a cheap spread map. "
        "Use images at least ~32×32 px (larger is safer for U-Net).",
    )
    uncertainty_tta_passes = st.sidebar.slider(
        "TTA passes",
        min_value=1,
        max_value=4,
        value=4,
        help="1 = identity only; up to 4 = id + hflip + vflip + rot90.",
    )

st.sidebar.subheader("Quality diagnostics")
include_blind_qa_sidebar = st.sidebar.checkbox(
    "Blind QA metrics (no-reference)",
    value=True,
    help="Optional: ENL-like estimate (homogeneous blocks), edge preservation vs input, variance stats — in **Run Denoising** results.",
)


def build_streamlit_denoise_kw(with_direct_uncertainty: bool) -> dict:
    """Sidebar → kwargs for :meth:`~SARDenoiseService.denoise_numpy` (single- and multi-method)."""
    kw = dict(
        model_type=model_type,
        max_iter=max_iter,
        rho_init=rho_init,
        alpha=alpha,
        theta=theta,
        use_log_transform=use_log_transform,
        disable_denoising=disable_denoising,
        quality_enhancement=quality_enhancement,
        speckle_factor=speckle_factor,
    )
    if (
        with_direct_uncertainty
        and method == "Direct Denoising"
        and direct_show_uncertainty
    ):
        kw["return_uncertainty"] = True
        kw["uncertainty_tta_passes"] = int(uncertainty_tta_passes)
    return kw


# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 Input Image")
    
    # Image upload
    uploaded_file = st.file_uploader(
        "Upload a SAR image",
        type=['png', 'jpg', 'jpeg', 'tiff'],
        help="Upload a single-channel SAR image for denoising"
    )

    BATCH_MAX_FILES = 40
    BATCH_MAX_BYTES_PER_FILE = 20 * 1024 * 1024

    with st.expander("Batch processing — multiple images → ZIP", expanded=False):
        st.caption(
            "Uses the sidebar **Denoising Method** and parameters. "
            f"Up to **{BATCH_MAX_FILES}** files, **{BATCH_MAX_BYTES_PER_FILE // (1024 * 1024)} MB** each; "
            "one image loaded at a time to limit memory."
        )
        batch_files = st.file_uploader(
            "Upload multiple SAR images",
            type=["png", "jpg", "jpeg", "tiff"],
            accept_multiple_files=True,
            key="sar_batch_uploader",
        )
        if batch_files and len(batch_files) > BATCH_MAX_FILES:
            st.warning(f"Only the first **{BATCH_MAX_FILES}** files will be processed.")

        if st.button("Process batch & build ZIP", key="sar_batch_process_btn"):
            todo = list(batch_files or [])[:BATCH_MAX_FILES]
            if not todo:
                st.error("Upload at least one image in the batch uploader above.")
            else:
                with st.spinner(f"Denoising {len(todo)} image(s)…"):
                    device_str = "cuda" if _cuda_available() else "cpu"
                    opts = service_options_from_merged()
                    svc = SARDenoiseService(
                        device=device_str,
                        infer_backend=opts["infer_backend"],
                        onnx_path=opts["onnx_path"],
                    )
                    kw = build_streamlit_denoise_kw(with_direct_uncertainty=False)
                    manifest_rows = []
                    zip_buffer = io.BytesIO()
                    try:
                        with tempfile.TemporaryDirectory() as tmpdir:
                            tmp_path = Path(tmpdir)
                            out_idx = 0
                            for uf in todo:
                                raw = uf.getvalue()
                                if len(raw) > BATCH_MAX_BYTES_PER_FILE:
                                    st.warning(
                                        f"Skipped **{uf.name}** (exceeds {BATCH_MAX_BYTES_PER_FILE // (1024 * 1024)} MB)."
                                    )
                                    continue
                                try:
                                    pil = Image.open(io.BytesIO(raw)).convert("L")
                                except Exception as e:
                                    st.warning(f"Skipped **{uf.name}** (invalid image: {e})")
                                    continue
                                arr = np.asarray(pil, dtype=np.float32) / 255.0
                                del pil, raw
                                t0 = time.perf_counter()
                                out = svc.denoise_numpy(arr, method, **kw)
                                elapsed = time.perf_counter() - t0
                                den = np.clip(np.asarray(out["denoised"], dtype=np.float32), 0.0, 1.0)
                                del out, arr
                                png_name = safe_batch_png_name(uf.name, out_idx)
                                out_idx += 1
                                out_png = tmp_path / png_name
                                Image.fromarray((den * 255.0).astype(np.uint8)).save(
                                    out_png, format="PNG"
                                )
                                del den
                                manifest_rows.append(
                                    [uf.name, method, f"{elapsed:.6f}"]
                                )

                            man_path = tmp_path / "manifest.csv"
                            with man_path.open("w", newline="", encoding="utf-8") as mf:
                                w = csv.writer(mf)
                                w.writerow(["filename", "method", "time_seconds"])
                                w.writerows(manifest_rows)

                            with zipfile.ZipFile(
                                zip_buffer, "w", compression=zipfile.ZIP_DEFLATED
                            ) as zf:
                                for p in sorted(tmp_path.iterdir()):
                                    zf.write(p, arcname=p.name)
                    except Exception as e:
                        st.error(f"Batch failed: {e}")
                    else:
                        if not manifest_rows:
                            st.error("No outputs written (all files skipped or failed).")
                        else:
                            st.session_state["batch_zip_bytes"] = zip_buffer.getvalue()
                            st.session_state["batch_zip_filename"] = (
                                f"sar_denoised_batch_{int(time.time())}.zip"
                            )
                            st.success(
                                f"Ready: **{len(manifest_rows)}** PNG(s) + **manifest.csv**."
                            )

    if st.session_state.get("batch_zip_bytes"):
        st.download_button(
            label="Download batch ZIP",
            data=st.session_state["batch_zip_bytes"],
            file_name=st.session_state.get(
                "batch_zip_filename", "sar_denoised_batch.zip"
            ),
            mime="application/zip",
            key="sar_batch_zip_download",
        )

    with st.expander("GeoTIFF — georeferenced tile denoise", expanded=False):
        st.caption(
            "**Single-band** GeoTIFF with **CRS** (transform + metadata preserved via "
            "**`inference.geotiff.denoise_geotiff`** and **rasterio**). Uses the sidebar "
            "**Denoising Method** and parameters; weights follow **`configs/infer`** / **`SAR_CHECKPOINT`** "
            "when required (same as the **`scripts/denoise_geotiff.py`** CLI)."
        )
        geotiff_file = st.file_uploader(
            "Upload GeoTIFF (.tif / .tiff)",
            type=["tif", "tiff"],
            key="streamlit_geotiff_uploader",
        )
        col_gt_a, col_gt_b = st.columns(2)
        with col_gt_a:
            if st.button(
                "Load bundled presentation sample",
                help="Loads `data/sample_geotiff/sentinel1_rtc_sample.tif` (real Sentinel-1) "
                "or `presentation_sample.tif` if the RTC file is missing.",
                key="streamlit_geotiff_load_sample",
            ):
                sample_path = _bundled_geotiff_path()
                if not sample_path.is_file():
                    st.error(
                        "Bundled GeoTIFF not found. From the repo root run:\n\n"
                        "**Real Sentinel-1 chip:** `python scripts/download_real_sample_geotiff.py`\n\n"
                        "**Tiny fallback:** `python scripts/build_sample_geotiff.py`\n\n"
                        "(both need **rasterio**)."
                    )
                else:
                    st.session_state["streamlit_geotiff_sample_bytes"] = (
                        sample_path.read_bytes()
                    )
                    st.session_state["streamlit_geotiff_sample_name"] = sample_path.name
                    st.success(
                        f"Loaded **{sample_path.name}** — set tile size if needed "
                        "(e.g. **1024** for the RTC sample), then **Run GeoTIFF denoising**."
                    )
        with col_gt_b:
            if st.session_state.get("streamlit_geotiff_sample_bytes") and st.button(
                "Clear bundled sample",
                key="streamlit_geotiff_clear_sample",
            ):
                st.session_state.pop("streamlit_geotiff_sample_bytes", None)
                st.session_state.pop("streamlit_geotiff_sample_name", None)
                st.rerun()

        if st.session_state.get("streamlit_geotiff_sample_bytes") and not geotiff_file:
            st.info(
                f"Ready to run: **{st.session_state.get('streamlit_geotiff_sample_name', 'sample')}** "
                "(uploading a file below will use the upload instead)."
            )

        geotiff_tile = st.number_input(
            "Tile size (pixels)",
            min_value=64,
            max_value=4096,
            value=512,
            step=64,
            key="streamlit_geotiff_tile",
        )
        if st.button("Run GeoTIFF denoising", type="secondary", key="streamlit_geotiff_run"):
            gt_raw = None
            input_label = ""
            if geotiff_file is not None:
                gt_raw = geotiff_file.getvalue()
                input_label = geotiff_file.name
            elif st.session_state.get("streamlit_geotiff_sample_bytes"):
                gt_raw = st.session_state["streamlit_geotiff_sample_bytes"]
                input_label = str(
                    st.session_state.get(
                        "streamlit_geotiff_sample_name", "sentinel1_rtc_sample.tif"
                    )
                )

            if gt_raw is None:
                st.error(
                    "Upload a GeoTIFF or click **Load bundled presentation sample** first."
                )
            else:
                try:
                    import rasterio
                except ImportError:
                    st.error(
                        "**rasterio** is not installed. Install with "
                        "`pip install rasterio` or `pip install -r requirements.txt`, "
                        "then restart the app."
                    )
                else:
                    if len(gt_raw) > 500 * 1024 * 1024:
                        st.error("File exceeds **500 MB** demo limit.")
                    else:
                        _clear_geotiff_viz_state()
                        with st.spinner("Denoising GeoTIFF (windowed)…"):
                            try:
                                merged = get_merged()
                                ck = merged.get("checkpoint")
                                ckpt = (
                                    Path(str(ck)).resolve()
                                    if ck not in (None, "")
                                    else None
                                )
                                imp_ck = ckpt if method == "ADMM-PnP-DL" else None
                                sim_ck = ckpt if method == "Direct Denoising" else None

                                device_str = "cuda" if _cuda_available() else "cpu"
                                opts = service_options_from_merged()
                                gt_svc = SARDenoiseService(
                                    device=device_str,
                                    improved_checkpoint=imp_ck,
                                    simple_checkpoint=sim_ck,
                                    infer_backend=opts["infer_backend"],
                                    onnx_path=opts["onnx_path"],
                                )
                                gt_kw = build_streamlit_denoise_kw(
                                    with_direct_uncertainty=False
                                )
                                if method == "Direct Denoising" and ckpt is not None:
                                    gt_kw["direct_checkpoint"] = ckpt

                                tile_fn = make_tile_denoise_fn(gt_svc, method, **gt_kw)

                                with tempfile.TemporaryDirectory() as gt_tmpdir:
                                    gt_in = Path(gt_tmpdir) / "input.tif"
                                    gt_out = Path(gt_tmpdir) / "denoised.tif"
                                    gt_in.write_bytes(gt_raw)
                                    with rasterio.open(gt_in) as _src:
                                        if _src.crs is None:
                                            raise ValueError(
                                                "Raster has no CRS — expected a georeferenced GeoTIFF."
                                            )
                                        if int(_src.count) != 1:
                                            raise ValueError(
                                                f"Expected a single-band raster; got count={_src.count}."
                                            )
                                    denoise_geotiff(
                                        gt_in,
                                        gt_out,
                                        tile_fn,
                                        tile_size=int(geotiff_tile),
                                        overlap=0,
                                    )
                                    gt_bytes = gt_out.read_bytes()

                                    with rasterio.open(gt_in) as s_in, rasterio.open(
                                        gt_out
                                    ) as s_out:
                                        arr_in = s_in.read(1).astype(np.float32)
                                        arr_out = s_out.read(1).astype(np.float32)
                                    if arr_in.shape != arr_out.shape:
                                        raise ValueError(
                                            f"Input/output shape mismatch: {arr_in.shape} vs {arr_out.shape}"
                                        )

                                    st.session_state["streamlit_geotiff_in_vis"] = (
                                        contrast_stretch_percentile_float01(arr_in)
                                    )
                                    st.session_state["streamlit_geotiff_out_vis"] = (
                                        contrast_stretch_percentile_float01(arr_out)
                                    )
                                    lo = float(min(arr_in.min(), arr_out.min()))
                                    hi = float(max(arr_in.max(), arr_out.max()))
                                    scale = hi - lo + 1e-8
                                    n_in = np.clip(
                                        (arr_in - lo) / scale, 0.0, 1.0
                                    ).astype(np.float32)
                                    n_out = np.clip(
                                        (arr_out - lo) / scale, 0.0, 1.0
                                    ).astype(np.float32)
                                    st.session_state["streamlit_geotiff_diff_vis"] = (
                                        comparison_abs_diff_map(n_in, n_out)
                                    )
                                    st.session_state["streamlit_geotiff_blind_qa"] = (
                                        compute_blind_qa(n_in, n_out)
                                    )
                                    st.session_state["streamlit_geotiff_enl_input"] = (
                                        calculate_enl(n_in)
                                    )
                                    st.session_state["streamlit_geotiff_enl_output"] = (
                                        calculate_enl(n_out)
                                    )
                                    st.session_state["streamlit_geotiff_norm_rmse"] = (
                                        float(
                                            np.sqrt(
                                                np.mean(
                                                    (n_in - n_out) ** 2,
                                                    dtype=np.float64,
                                                )
                                            )
                                        )
                                    )
                                    st.session_state["streamlit_geotiff_ssim_vs_input"] = (
                                        float(calculate_ssim(n_in, n_out))
                                    )
                                    st.session_state["streamlit_geotiff_psnr_vs_input"] = (
                                        float(calculate_psnr(n_in, n_out))
                                    )
                                    st.session_state["streamlit_geotiff_method_label"] = (
                                        method
                                    )

                                stem = Path(input_label).stem
                                safe_gt = re.sub(r"[^a-zA-Z0-9._-]+", "_", stem).strip(
                                    "._-"
                                )[:80] or "denoised"
                                st.session_state["streamlit_geotiff_out"] = gt_bytes
                                st.session_state["streamlit_geotiff_fn"] = (
                                    f"{safe_gt}_denoised.tif"
                                )
                                st.success(
                                    "GeoTIFF ready — preview & metrics below; download when needed."
                                )
                            except Exception as e:
                                _clear_geotiff_viz_state()
                                st.error(f"GeoTIFF processing failed: {e}")

        if st.session_state.get("streamlit_geotiff_out"):
            st.download_button(
                "Download denoised GeoTIFF",
                data=st.session_state["streamlit_geotiff_out"],
                file_name=st.session_state.get(
                    "streamlit_geotiff_fn", "denoised.tif"
                ),
                mime="image/tiff",
                key="streamlit_geotiff_download",
            )

        if st.session_state.get("streamlit_geotiff_in_vis") is not None:
            st.markdown("---")
            st.subheader("GeoTIFF — comparison & metrics")
            st.caption(
                "Display: **per-image 2–98% percentile** stretch (min–max would often look black "
                "on real SAR when a few pixels are much brighter). Metrics: **joint** min–max to "
                "[0, 1] for blind QA. **True reflectivity ground truth** is usually unknown for "
                "real SAR; PSNR/SSIM here are **vs. the observed input** (not vs. clean)."
            )
            _gml = st.session_state.get("streamlit_geotiff_method_label", method)
            gc1, gc2, gc3, gc4 = st.columns(4)
            with gc1:
                st.image(
                    st.session_state["streamlit_geotiff_in_vis"],
                    caption="Input (observed GeoTIFF)",
                    use_container_width=True,
                    clamp=True,
                )
            with gc2:
                st.image(
                    st.session_state["streamlit_geotiff_out_vis"],
                    caption=f"Denoised ({_gml})",
                    use_container_width=True,
                    clamp=True,
                )
            with gc3:
                st.image(
                    st.session_state["streamlit_geotiff_diff_vis"],
                    caption="|input − denoised| (joint norm., 99th pct. stretch)",
                    use_container_width=True,
                    clamp=True,
                )
            with gc4:
                st.markdown("**Reflectivity ground truth**")
                st.info(
                    "Not bundled for real Sentinel-1 / uploaded GeoTIFFs. "
                    "For **PSNR / SSIM vs clean**, use **SAMPLE** PNG patches (noisy + clean)."
                )

            bq = st.session_state.get("streamlit_geotiff_blind_qa") or {}
            ein = st.session_state.get("streamlit_geotiff_enl_input")
            eout = st.session_state.get("streamlit_geotiff_enl_output")

            def _fmt_enl(v: object) -> str:
                if v is None:
                    return "—"
                fv = float(v)
                if math.isinf(fv):
                    return "∞"
                if math.isnan(fv):
                    return "—"
                return f"{fv:.3f}"

            st.markdown("##### Metric summary")
            _markdown_metric_table(
                [
                    ("ENL (input, global)", _fmt_enl(ein)),
                    ("ENL (denoised, global)", _fmt_enl(eout)),
                    (
                        "ENL homog. (denoised)",
                        _format_metric_float(
                            float(bq.get("enl_homogeneous_median", 0.0))
                        ),
                    ),
                    (
                        "Edge pres. vs input",
                        _format_metric_float(
                            float(bq.get("edge_preservation_vs_input", 0.0))
                        ),
                    ),
                    (
                        "Norm. RMSE (in vs out)",
                        _format_metric_float(
                            float(
                                st.session_state.get(
                                    "streamlit_geotiff_norm_rmse", 0.0
                                )
                            )
                        ),
                    ),
                    (
                        "σ denoised (norm.)",
                        _format_metric_float(float(bq.get("std", 0.0))),
                    ),
                    (
                        "PSNR vs input (dB)",
                        _format_metric_float(
                            float(
                                st.session_state.get(
                                    "streamlit_geotiff_psnr_vs_input", 0.0
                                )
                            ),
                            nd=2,
                        ),
                    ),
                    (
                        "SSIM vs input",
                        _format_metric_float(
                            float(
                                st.session_state.get(
                                    "streamlit_geotiff_ssim_vs_input", 0.0
                                )
                            ),
                            nd=4,
                        ),
                    ),
                    (
                        "Var (denoised, norm.)",
                        _format_metric_float(float(bq.get("variance", 0.0))),
                    ),
                ]
            )
            st.caption(
                "**PSNR / SSIM** — vs joint-normalized **observed input** (higher PSNR → less change). "
                "Not vs. clean reflectivity."
            )

    # Load sample from SAMPLE dataset (user-selected patch, not random)
    sample_dir = "data/sample_sar/processed/test_patches"
    clean_dir = os.path.join(sample_dir, "clean")
    noisy_dir = os.path.join(sample_dir, "noisy")

    if os.path.exists(clean_dir) and os.path.exists(noisy_dir):
        sample_png_files = sorted(
            f for f in os.listdir(clean_dir) if f.lower().endswith(".png")
        )
        if not sample_png_files:
            st.warning("No PNG files found in SAMPLE dataset clean folder.")
        else:
            selected_sample = st.selectbox(
                "Select Sample Image",
                sample_png_files,
                key="sample_dataset_png",
            )
            if st.button("Load selected sample from SAMPLE dataset"):
                with st.spinner("Loading sample from SAMPLE dataset..."):
                    clean_path = os.path.join(clean_dir, selected_sample)
                    noisy_path = os.path.join(noisy_dir, selected_sample)
                    clean_image = None
                    noisy_image = None

                    if os.path.exists(clean_path) and os.path.exists(noisy_path):
                        clean_image = cv2.imread(clean_path, cv2.IMREAD_GRAYSCALE)
                        noisy_image = cv2.imread(noisy_path, cv2.IMREAD_GRAYSCALE)

                        if clean_image is not None and noisy_image is not None:
                            clean_image = clean_image.astype(np.float32) / 255.0
                            noisy_image = noisy_image.astype(np.float32) / 255.0
                        else:
                            st.error("Failed to load sample images")
                    else:
                        st.error("Sample image files not found for selected patch")

                    if clean_image is not None and noisy_image is not None:
                        st.session_state["clean_image"] = clean_image
                        st.session_state["noisy_image"] = noisy_image
                        st.session_state["image_source"] = "sample"
                        st.success("✅ Loaded selected sample from SAMPLE dataset!")
                    else:
                        st.session_state["clean_image"] = None
                        st.session_state["noisy_image"] = None

            with st.expander("📊 SAMPLE dataset grid viewer", expanded=False):
                st.caption(
                    "Preview multiple noisy patches at once. Does not replace the "
                    "single-image workspace above (selectbox + Load selected sample)."
                )
                max_grid = min(12, len(sample_png_files))
                default_grid_n = min(4, max_grid)
                n_grid = st.slider(
                    "Number of images",
                    min_value=1,
                    max_value=max_grid,
                    value=default_grid_n,
                    key="sample_dataset_grid_n",
                )
                prev_n = st.session_state.get("_sample_dataset_grid_n_prev")
                if prev_n is not None and prev_n != n_grid:
                    st.session_state.pop("sample_grid_random_filenames", None)
                st.session_state["_sample_dataset_grid_n_prev"] = n_grid

                btn_rand_col, btn_reset_col = st.columns(2)
                with btn_rand_col:
                    if st.button("Load N random samples", key="sample_grid_random_btn"):
                        k = min(n_grid, len(sample_png_files))
                        st.session_state["sample_grid_random_filenames"] = random.sample(
                            sample_png_files, k
                        )
                with btn_reset_col:
                    if st.button("Show first N (sorted)", key="sample_grid_first_n_btn"):
                        st.session_state.pop("sample_grid_random_filenames", None)

                picked = st.session_state.get("sample_grid_random_filenames")
                if picked is not None and len(picked) != n_grid:
                    st.session_state.pop("sample_grid_random_filenames", None)
                    picked = None
                if picked is None:
                    picked = sample_png_files[:n_grid]

                grid_images = []
                for fn in picked:
                    patch = load_sample_noisy_patch_float01(noisy_dir, fn)
                    if patch is not None:
                        grid_images.append((patch, fn))

                if not grid_images:
                    st.info("Could not load images for the current selection.")
                else:
                    grid_names = [fn for _, fn in grid_images]
                    stretch_thumbs = st.checkbox(
                        "Contrast-stretch thumbnails (per-image min–max, display only)",
                        value=False,
                        key="sample_grid_stretch_thumbs",
                    )

                    def _thumb(arr):
                        return (
                            contrast_stretch_display_float01(arr)
                            if stretch_thumbs
                            else arr
                        )

                    # One st.image() per row with a *list* of arrays — avoids Streamlit
                    # mis-binding repeated st.image calls inside nested st.columns() loops
                    # (symptom: same thumbnail shown for every cell).
                    ncols = 3
                    for row_start in range(0, len(grid_images), ncols):
                        chunk = grid_images[row_start : row_start + ncols]
                        st.image(
                            [_thumb(p[0]) for p in chunk],
                            caption=[p[1] for p in chunk],
                            use_container_width=True,
                            clamp=True,
                        )

                    # Streamlit forbids nested expanders — use subheaders inside the parent grid expander.
                    st.subheader("📈 Per-patch statistics")
                    st.caption(
                        "Raw **[0, 1]** float stats (same arrays used for inference / batch). "
                        "**Hash** = SHA256(float32 bytes), first 16 hex chars — unique per distinct array."
                    )
                    for arr, name in grid_images:
                        st.write(f"**{name}**")
                        st.write(sample_patch_stats(arr))
                        st.write("Hash:", sample_array_hash_prefix(arr))

                    st.divider()
                    st.subheader("🔬 Image Similarity Analysis")
                    st.caption(
                        "Quantitative **A vs B** comparison on raw **[0, 1]** patches (display-only). "
                        "**SSIM** treats A as reference and B as comparison (symmetric for equal shape)."
                    )
                    cmp_stretch = st.checkbox(
                        "Contrast-stretch A & B for display",
                        value=True,
                        key="sample_cmp_stretch_ab",
                    )
                    c1, c2 = st.columns(2)
                    with c1:
                        cmp_a = st.selectbox(
                            "Image A",
                            grid_names,
                            key="sample_cmp_img_a",
                        )
                    with c2:
                        cmp_b = st.selectbox(
                            "Image B",
                            grid_names,
                            index=min(1, len(grid_names) - 1),
                            key="sample_cmp_img_b",
                        )
                    arr_a = next(a for a, n in grid_images if n == cmp_a)
                    arr_b = next(a for a, n in grid_images if n == cmp_b)
                    disp_a = (
                        contrast_stretch_display_float01(arr_a)
                        if cmp_stretch
                        else arr_a
                    )
                    disp_b = (
                        contrast_stretch_display_float01(arr_b)
                        if cmp_stretch
                        else arr_b
                    )
                    diff_abs = np.abs(arr_a.astype(np.float64) - arr_b.astype(np.float64))
                    mad = float(np.mean(diff_abs))
                    max_diff = float(np.max(diff_abs))
                    diff_norm_max = sample_pair_diff_normalized_by_max(arr_a, arr_b)
                    try:
                        ssim_ab = float(calculate_ssim(arr_a, arr_b))
                    except Exception:
                        ssim_ab = float("nan")

                    st.image(
                        [disp_a, disp_b, diff_norm_max],
                        caption=[
                            f"A: {cmp_a}",
                            f"B: {cmp_b}",
                            "Normalized Difference Map",
                        ],
                        use_container_width=True,
                        clamp=True,
                    )

                    ha = sample_array_hash_prefix(arr_a)
                    hb = sample_array_hash_prefix(arr_b)
                    st.write(f"**Identity (SHA256 prefix)** — A: `{ha}` · B: `{hb}`")
                    if ha == hb:
                        st.warning(
                            "Hashes match — same float32 payload (identical arrays in memory)."
                        )

                    m1, m2, m3 = st.columns(3)
                    with m1:
                        st.metric("MAD", f"{mad:.6f}")
                    with m2:
                        st.metric("Max Diff", f"{max_diff:.6f}")
                    with m3:
                        st.metric(
                            "SSIM (A, B)",
                            "—" if np.isnan(ssim_ab) else f"{ssim_ab:.4f}",
                        )

                    verdict, level = similarity_verdict_mad(mad)
                    if level == "warning":
                        st.warning(f"**Verdict:** {verdict} (mean |A−B| < 0.002).")
                    elif level == "info":
                        st.info(f"**Verdict:** {verdict} (0.002 ≤ MAD < 0.02).")
                    else:
                        st.success(f"**Verdict:** {verdict} (MAD ≥ 0.02).")

                    show_ab_hist = st.checkbox(
                        "Histogram comparison (A vs B, matplotlib)",
                        value=False,
                        key="sample_sim_ab_hist",
                    )
                    if show_ab_hist:
                        fig, ax = plt.subplots(figsize=(6, 2.8))
                        ax.hist(
                            arr_a.ravel(),
                            bins=32,
                            range=(0.0, 1.0),
                            alpha=0.55,
                            color="tab:blue",
                            label="A",
                            edgecolor="white",
                            linewidth=0.4,
                        )
                        ax.hist(
                            arr_b.ravel(),
                            bins=32,
                            range=(0.0, 1.0),
                            alpha=0.55,
                            color="tab:orange",
                            label="B",
                            edgecolor="white",
                            linewidth=0.4,
                        )
                        ax.set_xlim(0.0, 1.0)
                        ax.set_xlabel("Intensity")
                        ax.set_ylabel("Count")
                        ax.set_title("Pixel distribution — A vs B")
                        ax.legend(loc="upper right")
                        ax.grid(True, alpha=0.3)
                        st.pyplot(fig)
                        plt.close(fig)

                    st.divider()
                    st.subheader("🔍 Zoom crop & histogram")
                    st.caption(
                        "Inspect a **ROI** with contrast-stretched view; histogram uses raw **[0, 1]** bins."
                    )
                    zoom_name = st.selectbox(
                        "Patch for zoom",
                        grid_names,
                        key="sample_zoom_pick",
                    )
                    zarr = next(a for a, n in grid_images if n == zoom_name)
                    zh, zw = int(zarr.shape[0]), int(zarr.shape[1])
                    zc1, zc2 = st.columns(2)
                    with zc1:
                        y1 = st.slider(
                            "y1 (row start)",
                            0,
                            max(0, zh - 2),
                            0,
                            key="sample_zoom_y1",
                        )
                        y2 = st.slider(
                            "y2 (row end, exclusive)",
                            min(y1 + 1, zh - 1),
                            zh,
                            min(y1 + min(32, zh), zh),
                            key="sample_zoom_y2",
                        )
                    with zc2:
                        x1 = st.slider(
                            "x1 (col start)",
                            0,
                            max(0, zw - 2),
                            0,
                            key="sample_zoom_x1",
                        )
                        x2 = st.slider(
                            "x2 (col end, exclusive)",
                            min(x1 + 1, zw - 1),
                            zw,
                            min(x1 + min(32, zw), zw),
                            key="sample_zoom_x2",
                        )
                    if y2 <= y1 or x2 <= x1:
                        st.warning("Invalid ROI — adjust sliders.")
                    else:
                        crop = zarr[y1:y2, x1:x2]
                        st.write(
                            {
                                "crop_shape": list(crop.shape),
                                "crop_mean": float(np.mean(crop)),
                                "crop_std": float(np.std(crop)),
                                "crop_min": float(np.min(crop)),
                                "crop_max": float(np.max(crop)),
                            }
                        )
                        st.image(
                            [
                                zarr,
                                contrast_stretch_display_float01(crop),
                            ],
                            caption=[
                                f"Full patch ({zoom_name})",
                                f"Zoom (contrast-stretched) [{y1}:{y2}, {x1}:{x2}]",
                            ],
                            use_container_width=True,
                            clamp=True,
                        )
                    show_hist = st.checkbox(
                        "Show pixel histogram (full patch, matplotlib)",
                        value=False,
                        key="sample_zoom_hist",
                    )
                    if show_hist:
                        fig, ax = plt.subplots(figsize=(6, 2.2))
                        ax.hist(
                            zarr.ravel(),
                            bins=32,
                            range=(0.0, 1.0),
                            color="steelblue",
                            edgecolor="white",
                            linewidth=0.5,
                        )
                        ax.set_xlim(0.0, 1.0)
                        ax.set_xlabel("Intensity")
                        ax.set_ylabel("Count")
                        ax.set_title(f"Histogram — {zoom_name}")
                        ax.grid(True, alpha=0.3)
                        st.pyplot(fig)
                        plt.close(fig)

                    st.divider()
                    st.markdown(
                        "**Batch denoising (SAMPLE)** — uses the sidebar **Denoising Method**; "
                        "at most **8** patches per run to limit compute."
                    )
                    _SAMPLE_BATCH_CAP = 8
                    batch_cap = min(_SAMPLE_BATCH_CAP, len(grid_images))
                    n_batch = st.slider(
                        "Batch denoise count",
                        min_value=1,
                        max_value=batch_cap,
                        value=min(4, batch_cap),
                        key="sample_dataset_batch_n",
                        help="Number of patches from the current grid (first N of the selection) to denoise together.",
                    )
                    batch_entries = grid_images[:n_batch]
                    new_batch_sig = tuple(fn for _, fn in batch_entries)
                    if st.session_state.get("_sample_batch_denoise_sig") != new_batch_sig:
                        st.session_state.pop("batch_denoise_results", None)
                    st.session_state["_sample_batch_denoise_sig"] = new_batch_sig
                    st.session_state["batch_images"] = [
                        np.ascontiguousarray(arr) for arr, _ in batch_entries
                    ]
                    st.session_state["batch_image_names"] = [
                        fn for _, fn in batch_entries
                    ]

                    if st.button("Run Batch Denoising", key="sample_dataset_batch_run_btn"):
                        imgs = st.session_state.get("batch_images") or []
                        names = st.session_state.get("batch_image_names") or []
                        if not imgs:
                            st.error("No batch images in session.")
                        else:
                            with st.spinner(
                                f"Batch denoising {len(imgs)} patch(es) with **{method}**…"
                            ):
                                device_str = (
                                    "cuda" if _cuda_available() else "cpu"
                                )
                                opts = service_options_from_merged()
                                batch_svc = SARDenoiseService(
                                    device=device_str,
                                    infer_backend=opts["infer_backend"],
                                    onnx_path=opts["onnx_path"],
                                )
                                batch_kw = build_streamlit_denoise_kw(
                                    with_direct_uncertainty=False
                                )
                                batch_results = []
                                for img_arr, fname in zip(imgs, names):
                                    try:
                                        if np.iscomplexobj(img_arr):
                                            raise ValueError(
                                                "Complex-valued batch item — use magnitude pipeline elsewhere."
                                            )
                                        bout = batch_svc.denoise_numpy(
                                            img_arr,
                                            method,
                                            **batch_kw,
                                            include_blind_qa=False,
                                        )
                                        batch_results.append(
                                            {
                                                "name": fname,
                                                "denoised": np.asarray(
                                                    bout["denoised"],
                                                    dtype=np.float32,
                                                ),
                                                "error": None,
                                            }
                                        )
                                    except Exception as e:
                                        batch_results.append(
                                            {
                                                "name": fname,
                                                "denoised": None,
                                                "error": str(e),
                                            }
                                        )
                                st.session_state["batch_denoise_results"] = batch_results
                            st.success(
                                f"✅ Batch denoising finished ({len(imgs)} patch(es))."
                            )

                    sample_batch_results = st.session_state.get("batch_denoise_results")
                    if sample_batch_results:
                        st.caption(
                            "Last batch run — denoised outputs (same order as batch)."
                        )
                        out_ncols = 3
                        for row0 in range(0, len(sample_batch_results), out_ncols):
                            out_row = st.columns(out_ncols)
                            for j, out_col in enumerate(out_row):
                                k = row0 + j
                                if k < len(sample_batch_results):
                                    item = sample_batch_results[k]
                                    with out_col:
                                        with st.container(
                                            key=f"sample_batch_out_{row0}_{j}"
                                        ):
                                            if item.get("error"):
                                                st.error(
                                                    f"{item['name']}: {item['error']}"
                                                )
                                            elif item.get("denoised") is not None:
                                                st.image(
                                                    item["denoised"],
                                                    caption=item["name"],
                                                    use_container_width=True,
                                                    clamp=True,
                                                )
    else:
        st.error("SAMPLE dataset not found. Please run: python download_sample_dataset.py")
    
    # Display uploaded/generated image
    if uploaded_file is not None:
        # Load uploaded image
        image = Image.open(uploaded_file).convert('L')
        image_array = np.array(image, dtype=np.float32) / 255.0
        
        st.session_state['noisy_image'] = image_array
        st.session_state['image_source'] = 'uploaded'
        
        st.image(image, caption="Uploaded Image", use_column_width=True)
    
    elif 'noisy_image' in st.session_state:
        # Display generated image
        noisy_img = st.session_state['noisy_image']
        st.image(noisy_img, caption="Input Image", use_column_width=True, clamp=True)

with col2:
    st.subheader("🔧 Denoising Results")
    run_col1, run_col2 = st.columns(2)
    with run_col1:
        run_single = st.button("🚀 Run Denoising", type="primary")
    with run_col2:
        run_all_methods = st.button("Run All Methods")

    if run_single:
        if 'noisy_image' not in st.session_state:
            st.error("Please upload an image or generate a synthetic one first!")
        else:
            with st.spinner("Running denoising algorithm..."):
                start_time = time.time()
                
                noisy_image = st.session_state['noisy_image']

                if np.iscomplexobj(noisy_image):
                    st.info("📡 Detected complex SAR data - extracting magnitude")

                device_str = "cuda" if _cuda_available() else "cpu"
                opts = service_options_from_merged()
                svc = SARDenoiseService(
                    device=device_str,
                    infer_backend=opts["infer_backend"],
                    onnx_path=opts["onnx_path"],
                )
                denoise_kw = build_streamlit_denoise_kw(with_direct_uncertainty=True)

                out = svc.denoise_numpy(
                    noisy_image,
                    method,
                    **denoise_kw,
                    include_blind_qa=include_blind_qa_sidebar,
                )
                replay_messages_streamlit(out["messages"], st)

                denoised_image = out["denoised"]
                energies = out["energies"]
                residuals = out["residuals"]
                _meta_out = out.get("meta") or {}
                st.session_state["last_blind_qa"] = _meta_out.get("blind_qa")
                st.session_state["last_blind_qa_error"] = _meta_out.get("blind_qa_error")

                end_time = time.time()
                processing_time = end_time - start_time
                
                # Store results
                st.session_state['denoised_image'] = denoised_image
                st.session_state['processing_time'] = processing_time
                st.session_state['energies'] = energies
                st.session_state['residuals'] = residuals
                u = out.get("uncertainty")
                if u is not None:
                    st.session_state['uncertainty_u8'] = uncertainty_to_vis_u8(u)
                    st.session_state['uncertainty_meta'] = {
                        k: out.get("meta", {}).get(k)
                        for k in ("uncertainty_mean", "uncertainty_max", "uncertainty_tta_passes")
                        if out.get("meta", {}).get(k) is not None
                    }
                else:
                    st.session_state.pop('uncertainty_u8', None)
                    st.session_state.pop('uncertainty_meta', None)

    if run_all_methods:
        if "noisy_image" not in st.session_state:
            st.error("Please upload an image or generate a synthetic one first!")
        else:
            with st.spinner("Running TV, Direct, and ADMM-PnP-DL…"):
                noisy_image = st.session_state["noisy_image"]
                if np.iscomplexobj(noisy_image):
                    st.info("📡 Detected complex SAR data - extracting magnitude")

                device_str = "cuda" if _cuda_available() else "cpu"
                opts = service_options_from_merged()
                svc = SARDenoiseService(
                    device=device_str,
                    infer_backend=opts["infer_backend"],
                    onnx_path=opts["onnx_path"],
                )
                kw = build_streamlit_denoise_kw(with_direct_uncertainty=False)
                raw_outputs = []
                seconds_list = []
                for mname in MULTI_METHOD_ORDER:
                    t0 = time.perf_counter()
                    out_m = svc.denoise_numpy(noisy_image, mname, **kw)
                    seconds_list.append(time.perf_counter() - t0)
                    raw_outputs.append(np.asarray(out_m["denoised"], dtype=np.float32))
                display_arrays = normalize_multi_method_display(raw_outputs)
                st.session_state["multi_method_comparison"] = [
                    {
                        "name": n,
                        "display": d,
                        "seconds": s,
                        "raw": np.ascontiguousarray(r, dtype=np.float32),
                    }
                    for n, d, s, r in zip(
                        MULTI_METHOD_ORDER,
                        display_arrays,
                        seconds_list,
                        raw_outputs,
                    )
                ]
            st.success(
                "✅ Multi-method run complete — scroll down in **Denoising Results** for comparison & charts."
            )

    # Display results
    if 'denoised_image' in st.session_state:
        denoised_img = st.session_state['denoised_image']
        if 'uncertainty_u8' in st.session_state:
            uc1, uc2 = st.columns(2)
            with uc1:
                st.image(denoised_img, caption="Denoised (TTA mean)", use_container_width=True, clamp=True)
            with uc2:
                st.image(
                    st.session_state['uncertainty_u8'],
                    caption="Uncertainty (TTA std, brighter = more spread)",
                    use_container_width=True,
                )
            um = st.session_state.get('uncertainty_meta') or {}
            if um:
                st.caption(
                    " · ".join(f"{k}: {float(v):.6g}" if isinstance(v, (int, float)) else f"{k}: {v}" for k, v in um.items())
                )
        else:
            st.image(denoised_img, caption="Denoised Image", use_container_width=True, clamp=True)
        
        # Calculate metrics when a clean reference exists (synthetic or SAMPLE dataset)
        if (
            "clean_image" in st.session_state
            and st.session_state["clean_image"] is not None
        ):
            clean_img = st.session_state["clean_image"]
            metrics = calculate_metrics(clean_img, denoised_img)
            
            st.subheader("📊 Performance Metrics")
            col_metric1, col_metric2, col_metric3 = st.columns(3)
            
            with col_metric1:
                st.metric("PSNR", f"{metrics['psnr']:.2f} dB")
            with col_metric2:
                st.metric("SSIM", f"{metrics['ssim']:.4f}")
            with col_metric3:
                st.metric("ENL", f"{metrics['enl']:.2f}")
        
        # Processing time
        if 'processing_time' in st.session_state:
            st.metric("Processing Time", f"{st.session_state['processing_time']:.2f} seconds")

        if st.session_state.get("last_blind_qa_error"):
            st.warning(
                f"Blind QA failed: {st.session_state['last_blind_qa_error']}"
            )
        _bq = st.session_state.get("last_blind_qa")
        if _bq:
            st.subheader("Blind QA (no reference)")
            st.caption(
                "Indicative only — scene-dependent; not a substitute for paired metrics."
            )
            _m1, _m2, _m3, _m4, _m5 = st.columns(5)
            with _m1:
                st.metric("ENL homo. median", f"{float(_bq['enl_homogeneous_median']):.2f}")
            with _m2:
                st.metric("Edge pres. vs input", f"{float(_bq['edge_preservation_vs_input']):.3f}")
            with _m3:
                st.metric("Variance", f"{float(_bq['variance']):.5f}")
            with _m4:
                st.metric("Std dev", f"{float(_bq['std']):.4f}")
            with _m5:
                st.metric("Var(log I)", f"{float(_bq['variance_log']):.5f}")
            with st.expander("Raw blind_qa JSON"):
                st.json(_bq)

    # Multi-method comparison (TV / Direct / ADMM in one run, shared display scaling)
    if st.session_state.get("multi_method_comparison"):
        st.markdown("---")
        st.subheader("Multi-method comparison")
        st.caption(
            "One run with **TV**, **Direct**, and **ADMM-PnP-DL** using the same sidebar settings. "
            "Each panel uses a **2–98% percentile stretch** so different output scales stay visible "
            "(a **shared** min–max would often wash out **Direct** when its range is smaller than TV/ADMM, "
            "e.g. no `checkpoints_simple` on Cloud). **Direct** still needs a real checkpoint for meaningful denoising."
        )
        mc = st.session_state["multi_method_comparison"]
        if any("raw" not in e for e in mc):
            st.warning(
                "Re-run **Run All Methods** once to enable comparison metrics and charts "
                "(this session was stored before raw outputs were saved)."
            )
        mcol1, mcol2, mcol3 = st.columns(3)
        for col, entry in zip((mcol1, mcol2, mcol3), mc):
            with col:
                st.image(
                    entry["display"],
                    caption=entry["name"],
                    use_container_width=True,
                    clamp=True,
                )
                st.metric("Inference time", f"{entry['seconds']:.3f} s")

        st.markdown("##### Comparison charts")
        names_short = [e["name"].replace(" Denoising", "").replace("-PnP-DL", "") for e in mc]
        secs = [float(e["seconds"]) for e in mc]
        fig_t, ax_t = plt.subplots(figsize=(8, 3.2))
        ax_t.bar(names_short, secs, color=["#4e79a7", "#f28e2b", "#59a14f"])
        ax_t.set_ylabel("Seconds")
        ax_t.set_title("Inference time (multi-method run)")
        ax_t.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig_t)
        plt.close(fig_t)

        noisy_m = st.session_state.get("noisy_image")
        clean_m = st.session_state.get("clean_image")
        ref_ok = (
            clean_m is not None
            and noisy_m is not None
            and all(
                e.get("raw") is not None
                and np.asarray(e["raw"]).shape == np.asarray(clean_m).shape
                for e in mc
            )
        )

        if ref_ok:
            st.markdown("##### Metrics vs ground truth (clean reference)")
            st.caption(
                "PSNR / SSIM / ENL use **`clean_image`** from SAMPLE or synthetic paired data "
                "(same shape as the noisy input)."
            )
            psnr_v, ssim_v, enl_v = [], [], []
            for e in mc:
                raw = np.asarray(e["raw"], dtype=np.float32)
                m = calculate_metrics(np.asarray(clean_m, dtype=np.float32), raw)
                psnr_v.append(float(m["psnr"]))
                ssim_v.append(float(m["ssim"]))
                enl_v.append(float(m["enl"]) if np.isfinite(m["enl"]) else float("nan"))

            mt1, mt2, mt3 = st.columns(3)
            for i, e in enumerate(mc):
                with [mt1, mt2, mt3][i]:
                    st.markdown(f"**{e['name']}**")
                    st.metric("PSNR", f"{psnr_v[i]:.2f} dB")
                    st.metric("SSIM", f"{ssim_v[i]:.4f}")
                    st.metric("ENL", f"{enl_v[i]:.2f}" if np.isfinite(enl_v[i]) else "—")

            fig_q, (ax_p, ax_s) = plt.subplots(1, 2, figsize=(9, 3.2))
            ax_p.bar(names_short, psnr_v, color=["#4e79a7", "#f28e2b", "#59a14f"])
            ax_p.set_ylabel("dB")
            ax_p.set_title("PSNR vs clean")
            ax_p.grid(True, axis="y", alpha=0.3)
            ax_s.bar(names_short, ssim_v, color=["#4e79a7", "#f28e2b", "#59a14f"])
            ax_s.set_ylabel("SSIM")
            ax_s.set_ylim(0.0, 1.05)
            ax_s.set_title("SSIM vs clean")
            ax_s.grid(True, axis="y", alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig_q)
            plt.close(fig_q)

            fig_e, ax_e = plt.subplots(figsize=(8, 3.0))
            enl_plot = [v if np.isfinite(v) else 0.0 for v in enl_v]
            ax_e.bar(names_short, enl_plot, color=["#4e79a7", "#f28e2b", "#59a14f"])
            ax_e.set_ylabel("ENL")
            ax_e.set_title("ENL of denoised vs clean (global)")
            ax_e.grid(True, axis="y", alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig_e)
            plt.close(fig_e)
        else:
            st.markdown("##### No-reference metrics (vs noisy input)")
            if clean_m is not None and noisy_m is not None:
                st.caption(
                    "**Clean reference** is present but **shape** does not match this multi-method "
                    "output — load a SAMPLE pair or matching synthetic image. Showing blind QA only."
                )
            else:
                st.caption(
                    "No **clean** image in session — load from **SAMPLE** (noisy + clean) or "
                    "**synthetic** for **PSNR/SSIM vs ground truth**. Below: blind proxies per method."
                )
            if noisy_m is None:
                st.info("No noisy image in session.")
            else:
                noisy_a = np.asarray(noisy_m, dtype=np.float32)
                bq_rows = []
                for e in mc:
                    raw = e.get("raw")
                    if raw is None:
                        continue
                    raw = np.asarray(raw, dtype=np.float32)
                    if raw.shape != noisy_a.shape:
                        bq_rows.append(
                            {
                                "Method": e["name"],
                                "Note": "shape mismatch vs noisy",
                            }
                        )
                        continue
                    lo = float(min(noisy_a.min(), raw.min()))
                    hi = float(max(noisy_a.max(), raw.max()))
                    sc = hi - lo + 1e-8
                    n0 = np.clip((noisy_a - lo) / sc, 0.0, 1.0).astype(np.float32)
                    n1 = np.clip((raw - lo) / sc, 0.0, 1.0).astype(np.float32)
                    try:
                        bq = compute_blind_qa(n0, n1)
                        bq_rows.append(
                            {
                                "Method": e["name"],
                                "ENL homog.": f"{float(bq['enl_homogeneous_median']):.3f}",
                                "Edge pres.": f"{float(bq['edge_preservation_vs_input']):.4f}",
                                "σ (norm.)": f"{float(bq['std']):.5f}",
                            }
                        )
                    except Exception as ex:
                        bq_rows.append({"Method": e["name"], "Note": str(ex)[:80]})

                if bq_rows:
                    st.dataframe(bq_rows, use_container_width=True, hide_index=True)

                homog = []
                edgep = []
                for e in mc:
                    raw = e.get("raw")
                    if raw is None or noisy_a.shape != np.asarray(raw).shape:
                        homog.append(0.0)
                        edgep.append(0.0)
                        continue
                    raw = np.asarray(raw, dtype=np.float32)
                    lo = float(min(noisy_a.min(), raw.min()))
                    hi = float(max(noisy_a.max(), raw.max()))
                    sc = hi - lo + 1e-8
                    n0 = np.clip((noisy_a - lo) / sc, 0.0, 1.0).astype(np.float32)
                    n1 = np.clip((raw - lo) / sc, 0.0, 1.0).astype(np.float32)
                    try:
                        bq = compute_blind_qa(n0, n1)
                        homog.append(float(bq["enl_homogeneous_median"]))
                        edgep.append(float(bq["edge_preservation_vs_input"]))
                    except Exception:
                        homog.append(0.0)
                        edgep.append(0.0)

                if any(h > 0 for h in homog):
                    fig_b, (ax_h, ax_b) = plt.subplots(1, 2, figsize=(9, 3.2))
                    ax_h.bar(names_short, homog, color=["#4e79a7", "#f28e2b", "#59a14f"])
                    ax_h.set_title("ENL homogeneous (median, denoised)")
                    ax_h.grid(True, axis="y", alpha=0.3)
                    ax_b.bar(names_short, edgep, color=["#4e79a7", "#f28e2b", "#59a14f"])
                    ax_b.set_title("Edge preservation vs input")
                    ax_b.set_ylim(0.0, max(1.05, max(edgep) * 1.15 if edgep else 1.0))
                    ax_b.grid(True, axis="y", alpha=0.3)
                    plt.tight_layout()
                    st.pyplot(fig_b)
                    plt.close(fig_b)

    # Comparison Dashboard — interactive blend, absolute difference, discrete view toggles
    if "denoised_image" in st.session_state and "noisy_image" in st.session_state:
        st.markdown("---")
        st.subheader("Comparison Dashboard")
        st.caption(
            "Inspect noisy vs denoised without re-running inference. "
            "Blend uses **α×denoised + (1−α)×noisy**; difference map uses |noisy−denoised| (display-normalized)."
        )
        noisy_cd = np.asarray(st.session_state["noisy_image"], dtype=np.float32)
        den_cd = np.asarray(st.session_state["denoised_image"], dtype=np.float32)
        if noisy_cd.shape != den_cd.shape:
            st.warning("Input and denoised shapes differ; comparison dashboard skipped.")
        else:
            view_mode = st.radio(
                "View",
                (
                    "Interactive blend (slider)",
                    "Original (noisy)",
                    "Denoised",
                    "Absolute difference",
                ),
                horizontal=True,
                key="comparison_dashboard_view",
            )
            if view_mode == "Interactive blend (slider)":
                alpha_blend = st.slider(
                    "Opacity on denoised (α): 0 = full noisy, 1 = full denoised",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.5,
                    step=0.01,
                    key="comparison_dashboard_alpha",
                )
                blended = comparison_blend_noisy_denoised(noisy_cd, den_cd, alpha_blend)
                st.image(
                    blended,
                    caption=(
                        f"Blend: {(1.0 - alpha_blend) * 100:.0f}% noisy + {alpha_blend * 100:.0f}% denoised "
                        f"(α={alpha_blend:.2f})"
                    ),
                    use_container_width=True,
                    clamp=True,
                )
            elif view_mode == "Original (noisy)":
                st.image(
                    noisy_cd,
                    caption="Original (noisy) input",
                    use_container_width=True,
                    clamp=True,
                )
            elif view_mode == "Denoised":
                st.image(
                    den_cd,
                    caption="Denoised output",
                    use_container_width=True,
                    clamp=True,
                )
            else:
                diff_vis = comparison_abs_diff_map(noisy_cd, den_cd)
                st.image(
                    diff_vis,
                    caption="Absolute difference |noisy − denoised| (99th pct. stretch)",
                    use_container_width=True,
                    clamp=True,
                )

    # Bottom section for detailed results
    if 'denoised_image' in st.session_state and 'energies' in st.session_state:
        st.markdown("---")
        st.subheader("📈 Algorithm Monitoring")
    
        col1, col2 = st.columns(2)
    
        with col1:
            # Energy plot
            if len(st.session_state['energies']) > 1:
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(st.session_state['energies'])
                ax.set_xlabel('Iteration')
                ax.set_ylabel('Energy')
                ax.set_title('ADMM Energy Convergence')
                ax.grid(True)
                st.pyplot(fig)
    
        with col2:
            # Residual plot
            if len(st.session_state['residuals']) > 1:
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(st.session_state['residuals'])
                ax.set_xlabel('Iteration')
                ax.set_ylabel('Residual')
                ax.set_title('ADMM Residual Convergence')
                ax.grid(True)
                st.pyplot(fig)

    # Comparison section
    if 'denoised_image' in st.session_state and 'noisy_image' in st.session_state:
        st.markdown("---")
        st.subheader("🔄 Image Comparison")
    
        n_compare_cols = 4 if 'uncertainty_u8' in st.session_state else 3
        cols = st.columns(n_compare_cols)
    
        with cols[0]:
            st.image(st.session_state['noisy_image'], caption="Noisy SAR Input", use_container_width=True, clamp=True)
    
        with cols[1]:
            cap = f"Denoised ({method})"
            st.image(st.session_state['denoised_image'], caption=cap, use_container_width=True, clamp=True)

        if 'uncertainty_u8' in st.session_state:
            with cols[2]:
                st.image(
                    st.session_state['uncertainty_u8'],
                    caption="Uncertainty (TTA)",
                    use_container_width=True,
                )
            gt_col = cols[3]
        else:
            gt_col = cols[2]
    
        with gt_col:
            if (
                "clean_image" in st.session_state
                and st.session_state["clean_image"] is not None
            ):
                st.image(
                    st.session_state["clean_image"],
                    caption="Ground Truth",
                    use_container_width=True,
                    clamp=True,
                )
            else:
                st.info("No ground truth available for uploaded images")
    
        # Quality Enhancement Mode comparison
        if quality_enhancement and 'denoised_image' in st.session_state:
            st.markdown("---")
            st.subheader("✨ Quality Enhancement Results")
            st.info("🔍 Enhanced processing applied: Gaussian blur + Non-local means denoising for superior clarity")
        
            # Show processing statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Input Range", f"{st.session_state['noisy_image'].min():.3f} - {st.session_state['noisy_image'].max():.3f}")
            with col2:
                st.metric("Output Range", f"{st.session_state['denoised_image'].min():.3f} - {st.session_state['denoised_image'].max():.3f}")
            with col3:
                improvement = st.session_state['denoised_image'].std() / (st.session_state['noisy_image'].std() + 1e-8)
                st.metric("Noise Reduction", f"{improvement:.2f}x")

# Async job history — read-only view of data/jobs (FastAPI + Redis/RQ)
st.markdown("---")
st.subheader("Async job history")
st.caption(
    f"Read-only listing of **`{job_storage.jobs_root()}`** "
    "(from **`POST /v1/jobs`**). Set **`SAR_JOBS_DIR`** to use a different root."
)
_root, _job_entries = load_job_history_entries(100)
if not _root.is_dir():
    st.info(
        "No jobs directory on disk yet. Enable the queue (**`SAR_USE_QUEUE=1`**, **`REDIS_URL`**) "
        "and submit jobs via the HTTP API to populate this folder."
    )
elif not _job_entries:
    st.info("No job folders containing **meta.json** were found.")
else:
    st.metric("Jobs shown", len(_job_entries))
    for _ent in _job_entries:
        _jid = _ent["job_id"]
        _method = _ent["meta"].get("method", "—")
        _exp_label = f"`{_jid[:8]}…` · **{_ent['status']}** · {_method}"
        with st.expander(_exp_label, expanded=False):
            st.caption(f"Folder mtime: {datetime.fromtimestamp(_ent['mtime'])}")
            if _ent.get("error"):
                st.error(str(_ent["error"])[:4000])
            st.markdown("**meta.json**")
            st.json(_ent["meta"])
            _jp = _ent["path"]
            _dc1, _dc2, _dc3, _dc4 = st.columns(4)
            with _dc1:
                _op = _jp / "output.png"
                if _op.is_file():
                    st.download_button(
                        "Download output.png",
                        data=_op.read_bytes(),
                        file_name=f"{_jid}_output.png",
                        mime="image/png",
                        key=f"job_hist_out_{_jid}",
                    )
                else:
                    st.caption("—")
            with _dc2:
                _mf = _jp / "meta.json"
                if _mf.is_file():
                    st.download_button(
                        "Download meta.json",
                        data=_mf.read_bytes(),
                        file_name=f"{_jid}_meta.json",
                        mime="application/json",
                        key=f"job_hist_meta_{_jid}",
                    )
            with _dc3:
                _sf = _jp / "status.json"
                if _sf.is_file():
                    st.download_button(
                        "Download status.json",
                        data=_sf.read_bytes(),
                        file_name=f"{_jid}_status.json",
                        mime="application/json",
                        key=f"job_hist_stat_{_jid}",
                    )
            with _dc4:
                _up = _jp / "uncertainty.png"
                if _up.is_file():
                    st.download_button(
                        "Download uncertainty.png",
                        data=_up.read_bytes(),
                        file_name=f"{_jid}_uncertainty.png",
                        mime="image/png",
                        key=f"job_hist_unc_{_jid}",
                    )

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>ADMM-PnP-DL SAR Image Denoising Demo</p>
    <p>Built with PyTorch, Streamlit, and advanced optimization techniques</p>
</div>
""", unsafe_allow_html=True)


def generate_synthetic_sar_image(size):
    """Generate synthetic SAR image for testing"""
    # Create base image with different regions
    image = np.zeros((size, size))
    center = size // 2
    
    # Add geometric shapes
    y, x = np.ogrid[:size, :size]
    
    # Circle
    mask = (x - center)**2 + (y - center)**2 < (size//4)**2
    image[mask] = 0.8
    
    # Rectangle
    image[center-size//8:center+size//8, center-size//4:center+size//4] = 0.6
    
    # Add some texture
    texture = np.random.normal(0, 0.1, (size, size))
    image = image + texture
    
    # Add some lines
    for i in range(0, size, size//8):
        image[i, :] = 0.4
        image[:, i] = 0.4
    
    # Smooth the image
    from scipy.ndimage import gaussian_filter
    image = gaussian_filter(image, sigma=1.0)
    
    # Normalize to [0, 1]
    image = (image - image.min()) / (image.max() - image.min())
    
    return image


if __name__ == "__main__":
    # This will be run when the script is executed directly
    # For Streamlit, use: streamlit run demo/streamlit_app.py
    pass
