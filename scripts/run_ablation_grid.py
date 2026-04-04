#!/usr/bin/env python3
"""
Run multiple evaluate_sample.py configurations from a YAML manifest (Upgrade 6).

Writes ``results/ablation/aggregate_ablation.json`` mapping run id -> evaluation_results.json payload.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


def build_command(run: dict, device: str | None) -> list[str]:
    cmd: list[str] = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "evaluate_sample.py"),
        "--data_dir",
        str(run.get("data_dir", "data/sample_sar/processed")),
        "--save_dir",
        str(run["save_dir"]),
        "--model_type",
        str(run.get("model_type", "unet")),
        "--batch_size",
        str(int(run.get("batch_size", 8))),
        "--patch_size",
        str(int(run.get("patch_size", 128))),
        "--methods",
        *[str(m) for m in run["methods"]],
    ]
    if device:
        cmd.extend(["--device", device])
    if run.get("no_task_metrics"):
        cmd.append("--no-task-metrics")
    if run.get("no_run_log"):
        cmd.append("--no-run-log")
    layout = run.get("data_layout", "sample")
    if layout and layout != "sample":
        cmd.extend(["--data_layout", str(layout)])
    tag = run.get("dataset_tag")
    if tag:
        cmd.extend(["--dataset_tag", str(tag)])
    for a in run.get("extra_args", []):
        cmd.append(str(a))
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch evaluation from ablation manifest YAML")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPO_ROOT / "configs/ablation/manifest.yaml",
        help="Path to manifest (see configs/ablation/example_manifest.yaml)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional --device for evaluate_sample (e.g. cpu)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands only; do not execute",
    )
    parser.add_argument(
        "--aggregate-out",
        type=Path,
        default=REPO_ROOT / "results" / "ablation" / "aggregate_ablation.json",
        help="Output path for merged JSON",
    )
    args = parser.parse_args()

    if not args.manifest.is_file():
        print(f"Manifest not found: {args.manifest}", file=sys.stderr)
        print("Copy configs/ablation/example_manifest.yaml to configs/ablation/manifest.yaml", file=sys.stderr)
        sys.exit(1)

    doc = yaml.safe_load(args.manifest.read_text(encoding="utf-8"))
    runs = doc.get("runs") or []
    if not runs:
        print("No runs in manifest", file=sys.stderr)
        sys.exit(1)

    aggregate: dict[str, dict] = {}

    for run in runs:
        rid = run.get("id")
        if not rid:
            print("Each run must have 'id'", file=sys.stderr)
            sys.exit(1)
        cmd = build_command(run, args.device)
        print("---", rid, "---")
        print(" ", " ".join(cmd))
        if args.dry_run:
            continue
        subprocess.check_call(cmd, cwd=REPO_ROOT)
        save_dir = REPO_ROOT / run["save_dir"]
        metrics_path = save_dir / "evaluation_results.json"
        if metrics_path.is_file():
            aggregate[rid] = json.loads(metrics_path.read_text(encoding="utf-8"))
        else:
            print(f"Warning: missing {metrics_path}", file=sys.stderr)

    if args.dry_run:
        return

    args.aggregate_out.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "manifest": str(args.manifest.resolve()),
        "schema_version": doc.get("schema_version", 1),
        "runs": aggregate,
    }
    args.aggregate_out.write_text(json.dumps(meta, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Wrote aggregate: {args.aggregate_out}")


if __name__ == "__main__":
    main()
