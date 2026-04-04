#!/usr/bin/env python3
"""
Build ``results/baseline/metrics.json`` from a finished ``evaluate_sample.py`` output.

Typical flow (see ``results/baseline/README.md``):

1. Run evaluation with a fixed ``--save_dir`` (e.g. ``results/baseline/staging``).
2. Run: ``python scripts/capture_baseline.py --baseline-id sample_tv_v1``
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algos.evaluation import results_to_jsonable  # noqa: E402


def _relpath_or_abs(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def _git_sha_short(cwd: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True,
            cwd=cwd,
        ).strip()
    except Exception:
        return "unknown"


def main() -> int:
    p = argparse.ArgumentParser(description="Write results/baseline/metrics.json from evaluation JSON.")
    p.add_argument(
        "--eval-json",
        type=Path,
        default=ROOT / "results/baseline/staging/evaluation_results.json",
        help="Path to evaluation_results.json from evaluate_sample.py",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results/baseline/metrics.json",
        help="Output metrics.json",
    )
    p.add_argument("--baseline-id", type=str, default="sample_tv_v1", help="Stable label for this baseline")
    p.add_argument(
        "--command",
        type=str,
        default="",
        help="Exact shell command used for evaluate_sample.py (stored in provenance)",
    )
    p.add_argument("--data-dir", type=str, default="data/sample_sar/processed")
    args = p.parse_args()

    if not args.eval_json.is_file():
        print(f"Missing {args.eval_json}; run evaluate_sample.py first.", file=sys.stderr)
        return 1

    raw = json.loads(args.eval_json.read_text(encoding="utf-8"))
    trimmed = results_to_jsonable(raw, include_per_patch_lists=False)

    doc = {
        "baseline_id": args.baseline_id,
        "schema_version": 1,
        "provenance": {
            "captured_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "git_sha": _git_sha_short(ROOT),
            "source_file": _relpath_or_abs(args.eval_json, ROOT),
            "data_dir": args.data_dir,
            "evaluate_command": args.command or None,
        },
        "metrics": trimmed,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
