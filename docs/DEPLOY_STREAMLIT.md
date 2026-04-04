# Deploy on Streamlit Community Cloud (free)

**Repository:** https://github.com/Vi1en/SAR_DENOISER  

## Prerequisites

- Code pushed to GitHub (`main`).
- Main file: **`demo/streamlit_app.py`**.
- **`runtime.txt`**: pins **Python 3.11** so dependency pins (NumPy &lt; 2, etc.) resolve like local CI — Cloud otherwise defaults to **3.13** and may pull incompatible wheels.
- Dependencies: **`requirements.txt`** is slim (no **rasterio**, no FastAPI/Redis/RQ). **`PyYAML`** is listed explicitly (provides `import yaml`).
- **`packages.txt`**: **one apt package name per line only** — **no comments** (Streamlit passes each line to `apt-get install`).

## Steps

1. Open [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
2. **New app** → pick **`Vi1en/SAR_DENOISER`** → branch **`main`**.
3. **Main file path:** `demo/streamlit_app.py`.
4. **Python version:** 3.11 (matches CI) if the UI offers a selector.
5. Deploy.

After deploy, Streamlit shows your public URL (e.g. `https://<app>.streamlit.app`). Add it to the main README under **Live demo**.

## Behavior without checkpoints

`.gitignore` excludes **`*.pth` / `*.pt`**. On Cloud, deep-learning methods may fall back until you:

- Attach weights via **Streamlit Secrets** + a small download script, or  
- Use **Git LFS** / a **Release** asset and download at startup, or  
- Rely on **TV / classical** paths that need no learned checkpoint.

The app is written to degrade gracefully when improved checkpoints are missing.

## GeoTIFF / rasterio

GeoTIFF uses **`rasterio`** (may need GDAL on some hosts). If install fails on Cloud, use **upload + PNG workflow** only, or simplify GeoTIFF imports later. **`packages.txt`** includes common system libs for OpenCV; add GDAL packages only if Streamlit build logs require it.

## Memory

- Prefer **CPU** PyTorch on free tier; avoid huge batch sizes in the sidebar.
- Close unused expanders to limit rerenders on weak instances.

## Troubleshooting

| Issue | Action |
|--------|--------|
| **Error installing requirements** | Almost always a heavy/compiled package. This repo uses a **slim** `requirements.txt` (no `rasterio`, no API stack). Pull latest `main` and **redeploy**. If it still fails, open **Manage app → Logs** and search for the first `ERROR` line from `pip`. |
| `ModuleNotFoundError` | Redeploy after sync; for local full stack use `pip install -r requirements-full.txt`. |
| OpenCV errors | Ensure `packages.txt` is present; redeploy. |
| Model not found | Expected without weights; use TV or host checkpoints. |
| Upload too large | Raise `maxUploadSize` in `.streamlit/config.toml` (MB). |
| Pip timeout (torch) | In Streamlit Cloud logs, if download times out, redeploy during off-peak or ask for a rebuild. |
