#!/usr/bin/env python3
"""
Render ``aggregate_ablation.json`` (from run_ablation_grid.py) as a Markdown table.

Usage:
  python scripts/ablation_to_markdown.py
  python scripts/ablation_to_markdown.py --aggregate results/ablation/aggregate_ablation.json --out results/ablation/summary.md
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Ablation aggregate → Markdown table")
    parser.add_argument(
        "--aggregate",
        type=Path,
        default=REPO_ROOT / "results" / "ablation" / "aggregate_ablation.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write markdown to file (default: stdout only)",
    )
    args = parser.parse_args()

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from evaluators.ablation_report import render_markdown

    if not args.aggregate.is_file():
        print(f"Aggregate not found: {args.aggregate}", file=sys.stderr)
        print("Run: python scripts/run_ablation_grid.py --manifest configs/ablation/manifest.yaml", file=sys.stderr)
        sys.exit(1)

    md = render_markdown(args.aggregate)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(md, encoding="utf-8")
        print(f"Wrote {args.out}", file=sys.stderr)
    else:
        sys.stdout.write(md)


if __name__ == "__main__":
    main()
