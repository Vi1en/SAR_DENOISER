# Step 01: Fix Documentation and Demo Entrypoint Mismatch

## 1. Objective

- Align **all** documentation, verification scripts, and printed instructions with the **actual** Streamlit application path (`demo/streamlit_app.py`).
- Eliminate references to the non-existent `demo/app.py` so new contributors, CI, and `verify_system.py` succeed on a clean clone.
- **Why it matters:** Broken entrypoints are the fastest way to lose trust; this step is zero-risk and unblocks every subsequent improvement.

## 2. Current Problem

- The repository ships **`demo/streamlit_app.py`** as the main UI, but **`README.md`**, **`PROJECT_REPORT.md`**, **`verify_system.py`**, **`setup.py`**, **`run_complete_workflow.py`**, **`run_demo.py`**, **`download_sample_dataset.py`**, **`train_sample.py`**, **`train.py`**, **`test_setup.py`**, and several `*_SUMMARY.md` files instruct `streamlit run demo/app.py` or check for `demo/app.py`.
- `verify_system.py` fails or reports failure when `demo/app.py` is missing, even though the project is otherwise valid.

## 3. Scope of Changes

### Files to modify (search and replace / edit)

| File | Change |
|------|--------|
| `README.md` | Replace `demo/app.py` → `demo/streamlit_app.py`; update project structure tree |
| `PROJECT_REPORT.md` | Same |
| `PROJECT_SUMMARY.md` | Same (if present) |
| `verify_system.py` | Check `demo/streamlit_app.py` |
| `setup.py` | Print correct launch command |
| `run_complete_workflow.py` | Print correct command |
| `run_demo.py` | Print correct command |
| `download_sample_dataset.py` | Print correct command |
| `train_sample.py` | Print correct command |
| `train.py` | Print correct command |
| `test_setup.py` | Print correct command |
| `FINAL_SUMMARY.md` | Same |
| `IMPROVEMENTS_SUMMARY.md` | Same |
| `BULLETPROOF_DISPLAY_SUMMARY.md` | Same |
| `SMART_DISPLAY_SUMMARY.md` | Same |
| `demo/streamlit_app.py` | Fix trailing comment if it says `streamlit run app.py` |

### New files

- Optional: `fixes/README.md` (index of fix steps) — **not required** for this step.

### Do not rename `streamlit_app.py` in this step (optional alternative)

- **Preferred (this step):** Keep filename `streamlit_app.py` and fix docs only — **lowest risk**.
- **Alternative later:** Add `demo/app.py` as a one-line shim (see Section 6) if you want the old command to work.

## 4. Detailed Implementation Steps

1. **Inventory references** (from repo root):

   ```bash
   rg "demo/app\.py" --glob '!fixes/**'
   ```

2. **Replace in each file**  
   - Find: `demo/app.py`  
   - Replace: `demo/streamlit_app.py`  
   - Also fix tree diagrams that show `app.py` under `demo/` → `streamlit_app.py`.

3. **Update `verify_system.py`**  
   - Locate the variable or string that references `demo/app.py` (approximately line 72).  
   - Change to:

   ```python
   demo_file = "demo/streamlit_app.py"
   ```

   - Ensure the existence check uses this path.

4. **`demo/streamlit_app.py` footer**  
   - Search for `app.py` in comments; update to `streamlit_app.py`.

5. **Consistency check**

   ```bash
   rg "demo/app\.py" --glob '!fixes/**'
   ```

   - Expected: **no matches** (or only this fix doc if you document the old name historically).

6. **Manual smoke test**

   ```bash
   streamlit run demo/streamlit_app.py
   ```

   - Confirm the app loads (even without GPU weights).

## 5. Code-Level Guidance

### BEFORE (`verify_system.py` pattern)

```python
demo_file = 'demo/app.py'
```

### AFTER

```python
demo_file = 'demo/streamlit_app.py'
```

### OPTIONAL shim (additive backward compatibility)

**New file:** `demo/app.py`

```python
"""Backward-compatible entrypoint: delegates to streamlit_app."""
# Re-export or instruct users — Streamlit requires a file to run.
# Option A: duplicate is bad. Option B: single line runner is not valid for streamlit.
# RECOMMENDED: Do NOT add empty shim; fix docs only.
```

**Note:** Streamlit must run a real script. The clean approach is **docs-only fix**. If you must support `streamlit run demo/app.py`, create **`demo/app.py`** with content:

```python
# demo/app.py — thin entrypoint (optional, Step 01b)
import runpy
from pathlib import Path
runpy.run_path(str(Path(__file__).parent / "streamlit_app.py"), run_name="__main__")
```

Actually `runpy.run_path` may not set up Streamlit’s `__main__` correctly. **Safer shim:**

```python
# demo/app.py
import importlib.util
from pathlib import Path

def main():
    path = Path(__file__).parent / "streamlit_app.py"
    spec = importlib.util.spec_from_file_location("streamlit_app", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

if __name__ == "__main__":
    main()
```

Streamlit expects to execute the file directly; **importing another module’s side effects is fragile**. **Recommendation for Step 01:** **Documentation + verify_system only.** Add `demo/app.py` as a copy symlink on Unix or duplicate `streamlit run` target only if product needs both — duplicate file is maintenance burden.

**Practical optional shim:** Copy `streamlit_app.py` → `app.py` once and maintain both (bad). **Best:** docs point to `streamlit_app.py` only.

## 6. Safety Constraints (VERY IMPORTANT)

- **MUST NOT** change algorithm code, training logic, or Streamlit behavior in `streamlit_app.py` except comment fixes.
- **MUST NOT** delete `streamlit_app.py` or rename it without updating Streamlit Cloud / deployment config (if deployed) — if GitHub/Streamlit uses `demo/app.py`, update the **Streamlit Cloud “main file path”** to `demo/streamlit_app.py` in the hosting dashboard.
- **Backward compatibility:** Old bookmarks saying `app.py` will break until users use the new path — communicate in README **“Entry point: `demo/streamlit_app.py`”** one release note line.

## 7. Testing & Verification

```bash
# 1. Verification script
python verify_system.py

# 2. No stale references
rg "demo/app\.py" --glob '!fixes/**'

# 3. Streamlit starts (timeout after manual check)
streamlit run demo/streamlit_app.py --server.headless true
```

**Expected:** `verify_system.py` reports demo file exists; Streamlit serves without import errors (may warn if checkpoints missing).

## 8. Rollback Plan

- Revert the commit: `git revert <commit>` or restore files from `main`.
- No database or schema changes; rollback is trivial.

## 9. Result After This Step

- Clone-and-run path matches documentation.
- `verify_system.py` reflects reality.
- Foundation for CI (later steps) that runs `verify_system.py` or path checks.
