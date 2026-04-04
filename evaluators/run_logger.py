"""
Structured evaluation run artifacts under ``results/runs/<run_id>/``.
"""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


def get_git_sha_short(fallback: str = "nogit") -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True,
            cwd=Path(__file__).resolve().parent.parent,
        ).strip()
    except Exception:
        return fallback


class EvaluationRunContext:
    """Creates ``results/runs/<UTC_ts>_<git_sha>/`` with manifest + metrics JSON."""

    def __init__(self, base_dir: Path | None = None):
        base_dir = base_dir or Path("results/runs")
        sha = get_git_sha_short()
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.run_id = f"{ts}_{sha}"
        self.run_dir = Path(base_dir) / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "plots").mkdir(exist_ok=True)

    def write_manifest(self, extra: Dict[str, Any]) -> None:
        manifest = {
            "run_id": self.run_id,
            "git_sha": get_git_sha_short(),
            **extra,
        }
        (self.run_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, default=str) + "\n",
            encoding="utf-8",
        )

    def write_metrics(self, metrics: Dict[str, Any]) -> None:
        (self.run_dir / "metrics.json").write_text(
            json.dumps(metrics, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )

    def plots_dir(self) -> Path:
        return self.run_dir / "plots"
