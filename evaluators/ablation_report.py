"""Build Markdown tables from ablation aggregate JSON (Upgrade 6)."""
from __future__ import annotations

import json
from pathlib import Path


def row_cells(ablation_id: str, method: str, payload: dict) -> list[str]:
    def fmt(key: str, nd: int = 4) -> str:
        v = payload.get(key)
        if v is None:
            return ""
        if isinstance(v, (int, float)):
            return f"{float(v):.{nd}f}"
        return str(v)

    return [
        ablation_id,
        method,
        fmt("psnr_mean", 2),
        fmt("ssim_mean", 4),
        fmt("enl_mean", 2),
        fmt("gsm_corr_mean", 4),
        fmt("epi_mean", 4),
        fmt("grad_ssim_mean", 4),
    ]


def render_markdown(aggregate_path: Path) -> str:
    doc = json.loads(aggregate_path.read_text(encoding="utf-8"))
    runs = doc.get("runs") or doc
    if not isinstance(runs, dict):
        raise ValueError("Expected dict with 'runs' key or flat run dict")

    if "runs" in doc and isinstance(doc["runs"], dict) and all(
        isinstance(v, dict) for v in doc["runs"].values()
    ):
        runs = doc["runs"]

    lines = [
        "# Ablation summary",
        "",
        f"_Source: `{aggregate_path}`_",
        "",
        "| ablation_id | method | psnr_mean | ssim_mean | enl_mean | gsm_corr_mean | epi_mean | grad_ssim_mean |",
        "|---|---|--:|--:|--:|--:|--:|--:|",
    ]

    for ablation_id in sorted(runs.keys()):
        methods_payload = runs[ablation_id]
        if not isinstance(methods_payload, dict):
            continue
        for method in sorted(methods_payload.keys()):
            payload = methods_payload[method]
            if not isinstance(payload, dict):
                continue
            cells = row_cells(ablation_id, method, payload)
            lines.append("| " + " | ".join(cells) + " |")

    lines.append("")
    return "\n".join(lines)
