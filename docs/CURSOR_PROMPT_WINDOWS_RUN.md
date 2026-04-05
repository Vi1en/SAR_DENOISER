# Cursor agent prompt: run SAR denoiser on Windows

Copy everything inside the block below into a **new Cursor chat** (or Composer) on Windows. The agent should run commands in **PowerShell** or **cmd** from a real terminal.

---

```
You are assisting on Windows (PowerShell or Command Prompt). Goal: run the SAR_DENOISER Streamlit app locally.

Rules:
- Run commands yourself in the integrated terminal; do not only suggest them.
- Use the repository root as cwd (folder containing `requirements.txt` and `demo/streamlit_app.py`).
- If Python is missing or wrong version, say so clearly after checking `python --version` or `py -3.11 --version`.
- Prefer a fresh virtual environment `.venv` in the repo root.
- After installs, start Streamlit and report the local URL.

Steps (execute in order):

1) Locate or obtain the repo
   - If the workspace is already this repo, `cd` to its root (where `requirements.txt` exists).
   - If not cloned: `git clone https://github.com/Vi1en/SAR_DENOISER.git` then `cd SAR_DENOISER` (or the user’s fork path).

2) Verify Python
   - Run: `python --version` OR `py -3.11 --version`
   - Need Python 3.10–3.12 (3.11 recommended). If `python` fails, try `py -3.11`.

3) Create and activate venv (repo root)
   - `python -m venv .venv`
   - PowerShell: `.\.venv\Scripts\Activate.ps1`
   - cmd: `.venv\Scripts\activate.bat`
   - If PowerShell blocks scripts: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` then activate again.

4) Upgrade pip
   - `python -m pip install --upgrade pip`

5) Install dependencies
   - `pip install -r requirements.txt`
   - If torch fails, retry after: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu` then `pip install -r requirements.txt` again.

6) Optional: SAMPLE dataset
   - `python scripts\download_sample_dataset.py`

7) Run the app
   - `streamlit run demo\streamlit_app.py`
   - Tell the user to open the printed URL (usually http://localhost:8501).

8) If `rasterio` or GDAL errors appear on Windows
   - Retry `pip install rasterio`; if still failing, note that conda-forge (`conda install -c conda-forge rasterio`) may be needed.

Stop when Streamlit is listening or after a clear, actionable error with the exact command output.
```

---

## One-line usage

Paste the quoted block (from `You are assisting...` through `...command output.`) into Cursor as your message.
