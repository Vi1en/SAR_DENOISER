# Deploy on Streamlit Community Cloud (free)

**Repository:** https://github.com/Vi1en/SAR_DENOISER  

## Prerequisites

- Code pushed to GitHub (`main`).
- Main file: **`demo/streamlit_app.py`**.
- **Python 3.11:** set it in **Advanced settings** when you deploy. [Streamlit’s docs](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app/upgrade-python) say the runtime cannot be changed after deploy without **delete + redeploy**; **`runtime.txt` in the repo is not guaranteed to be read by Cloud** (it is still useful for other hosts).
- Dependencies: **`requirements.txt`** is slim (no **rasterio**, no FastAPI/Redis/RQ, **no ONNX** — those are in **`requirements-full.txt`**). **`PyYAML`** is listed explicitly. YAML is **lazy-imported** in code. **Do not** put `--extra-index-url` in `requirements.txt`: Streamlit’s **uv** step can fail resolving **`onnx` / `ml-dtypes` / `setuptools`** when the PyTorch index is listed first.
- **No `packages.txt`:** we removed it because mixed Debian sources on Cloud broke `apt` (e.g. `libglib2.0-0`). **`opencv-python-headless`** wheels usually do not need extra system packages on Cloud.

## Steps

1. Open [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
2. **New app** → pick **`Vi1en/SAR_DENOISER`** → branch **`main`**.
3. **Main file path:** `demo/streamlit_app.py`.
4. **Advanced settings → Python:** choose **3.11** (matches CI and `requirements.txt` pins). Do not rely on Cloud defaulting to 3.11.
5. Deploy.

After deploy, Streamlit shows your public URL (e.g. `https://<app>.streamlit.app`). Add it to the main README under **Live demo**.

## Behavior without checkpoints

`.gitignore` excludes **`*.pth` / `*.pt`**. On Cloud, deep-learning methods may fall back until you:

- Attach weights via **Streamlit Secrets** + a small download script, or  
- Use **Git LFS** / a **Release** asset and download at startup, or  
- Rely on **TV / classical** paths that need no learned checkpoint.

The app is written to degrade gracefully when improved checkpoints are missing.

## GeoTIFF / rasterio

GeoTIFF uses **`rasterio`** (may need GDAL on some hosts). If install fails on Cloud, use **upload + PNG workflow** only, or simplify GeoTIFF imports later. If you must add system libraries later, use a **`packages.txt`** with **one package name per line and no comments** (comment lines are passed to `apt-get` as bogus package names).

## Memory

- Prefer **CPU** PyTorch on free tier; avoid huge batch sizes in the sidebar.
- Close unused expanders to limit rerenders on weak instances.

## Troubleshooting

**`installer returned a non-zero exit code`:** That line is only a summary. The real reason is always **a few lines below** the latest `📦 Processing dependencies…` block (look for `uv`, `pip`, or `apt-get` stderr, e.g. `×`, `ERROR`, `E:`). Copy **that** chunk into an issue — not megabytes of older log from a previous day.

**Forks:** If your Cloud app uses **your fork** of this repo, open the fork on GitHub → **Sync fork** so `main` matches **`Vi1en/SAR_DENOISER`**. An unsynced fork still runs old `infer_config` / `packages.txt` / old `requirements.txt`.

**Logs look “stuck” on old errors:** Scroll to the **latest** timestamps after **Reboot**. If the traceback still shows `infer_config.py` **line 17** as `import yaml`, Cloud is not on current `main` (today that import is **lazy**, inside `_load_yaml_file`, not at module level). Confirm the app uses repo **`Vi1en/SAR_DENOISER`**, branch **`main`**, and that **`packages.txt`** is gone (it was removed on purpose). Set **Python 3.11** under **Advanced settings** so `numpy<2` installs from wheels instead of building on Python 3.13.

**`ModuleNotFoundError: yaml`:** `demo/streamlit_app.py` calls `st.set_page_config` first, then bootstraps PyYAML (up to two `pip install` attempts) **before** importing `api.infer_config`. On current `main`, `infer_config` only imports `yaml` inside `_load_yaml_file`. If your traceback still shows **`streamlit_app.py` line ~32** at `set_page_config` or **`infer_config.py` line 17** as `import yaml`, the app is **not** running this revision — confirm GitHub **`Vi1en/SAR_DENOISER` `main`** and reboot. **Only root `requirements.txt`** is used for deps (no `demo/requirements.txt`).

| Issue | Action |
|--------|--------|
| **Error installing requirements** | Almost always a heavy/compiled package. This repo uses a **slim** `requirements.txt` (no `rasterio`, no API stack). Pull latest `main` and **redeploy**. If it still fails, open **Manage app → Logs** and search for the first `ERROR` line from `pip`. |
| `ModuleNotFoundError` | Redeploy after sync; for local full stack use `pip install -r requirements-full.txt`. |
| OpenCV errors | Rare on Cloud with `opencv-python-headless`. If `cv2` still fails, add a minimal **`packages.txt`** (one package per line, no `#` lines) per Streamlit docs and redeploy. |
| Model not found | Expected without weights; use TV or host checkpoints. |
| Upload too large | Raise `maxUploadSize` in `.streamlit/config.toml` (MB). |
| Pip timeout (torch) | In Streamlit Cloud logs, if download times out, redeploy during off-peak or ask for a rebuild. |
