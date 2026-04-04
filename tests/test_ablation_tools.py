"""Tests for ablation aggregate → markdown (no dataset required)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_ablation_to_markdown_table(tmp_path: Path):
    from evaluators.ablation_report import render_markdown

    agg = {
        "manifest": "/tmp/x",
        "schema_version": 1,
        "runs": {
            "run_a": {
                "U-Net Direct": {
                    "psnr_mean": 30.5,
                    "ssim_mean": 0.91,
                    "enl_mean": 12.0,
                    "gsm_corr_mean": 0.88,
                    "epi_mean": 0.95,
                    "grad_ssim_mean": 0.9,
                }
            }
        },
    }
    p = tmp_path / "aggregate_ablation.json"
    p.write_text(json.dumps(agg), encoding="utf-8")
    md = render_markdown(p)
    assert "run_a" in md
    assert "U-Net Direct" in md
    assert "30.50" in md
