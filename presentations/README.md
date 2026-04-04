# Presentations (PowerPoint)

**All `.pptx` decks for this project live in this folder** so they are easy to find.

| File | Description |
|------|-------------|
| **`project_improvements_presentation.pptx`** | **Regenerate:** `python scripts/build_improvements_presentation.py` — **17 slides**: title; problem (**I=R×n**); goals; evolution **chevrons**; **Res-UNet**; denoising **triptych** figure; **metrics** (MAD/SSIM/PSNR formulas); **method table** (TV from JSON); visualization; similarity; **TTA uncertainty**; batch; **FastAPI/Redis**; GeoTIFF/ONNX; before/after; impact; conclusion. Each slide: bullets + 2–3 line explanation + takeaway where specified. |
| **`ADMM_PnP_SAR_Denoising_Presentation.pptx`** | Earlier technical deck (see `PROJECT_PRESENTATION_SUMMARY.md`). |
| **`SAR_Denoising_Project_Evolution.pptx`** | Project evolution variant (manual / export). |
| **`Untitled presentation.pptx`** | Scratch / draft — rename or replace when final. |

## Regenerate the improvements deck

From the **repository root**:

```bash
pip install python-pptx   # once
python scripts/build_improvements_presentation.py
```

Output path: **`presentations/project_improvements_presentation.pptx`**.

## Adding new decks

Save new `.pptx` files **here** and add a row to the table above.
