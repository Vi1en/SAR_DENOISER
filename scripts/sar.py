#!/usr/bin/env python3
"""
Unified CLI for common repo tasks (roadmap: single entrypoint).

Examples::

    python sar.py train --config configs/train/default.yaml
    python sar.py eval --no-run-log --methods tv
    python sar.py verify
    python sar.py api -- --host 0.0.0.0 --port 8000
    python sar.py worker -- --url redis://localhost:6379/0
    python sar.py streamlit
    python sar.py streamlit -- --server.port 8502

See also ``docs/DEVELOPMENT.md``.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS_DIR = ROOT / "scripts"

_SCRIPTS: dict[str, Path] = {
    "train": _SCRIPTS_DIR / "train_improved.py",
    "train-sample": _SCRIPTS_DIR / "train_sample.py",
    "eval": _SCRIPTS_DIR / "evaluate_sample.py",
    "verify": _SCRIPTS_DIR / "verify_system.py",
    "capture-baseline": _SCRIPTS_DIR / "capture_baseline.py",
    "export-onnx": _SCRIPTS_DIR / "export_onnx.py",
    "compare-onnx": _SCRIPTS_DIR / "compare_pytorch_onnx.py",
    "denoise-geotiff": _SCRIPTS_DIR / "denoise_geotiff.py",
    "ablation-grid": _SCRIPTS_DIR / "run_ablation_grid.py",
    "ablation-md": _SCRIPTS_DIR / "ablation_to_markdown.py",
}


def _run_script(target: Path, forwarded: list[str]) -> int:
    cmd = [sys.executable, str(target), *forwarded]
    return subprocess.call(cmd, cwd=str(ROOT))


def _run_uvicorn(forwarded: list[str]) -> int:
    return subprocess.call(
        [sys.executable, "-m", "uvicorn", "api.main:app", *forwarded],
        cwd=str(ROOT),
    )


def _run_rq_worker(forwarded: list[str]) -> int:
    return subprocess.call(
        [sys.executable, "-m", "rq.cli", "worker", "sar_denoise", *forwarded],
        cwd=str(ROOT),
    )


def _run_streamlit(forwarded: list[str]) -> int:
    """``streamlit version`` / ``help`` passthrough; else ``streamlit run demo/streamlit_app.py``."""
    if forwarded and forwarded[0] in ("version", "help", "--version", "-h"):
        return subprocess.call([sys.executable, "-m", "streamlit", *forwarded], cwd=str(ROOT))
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "demo/streamlit_app.py",
        *forwarded,
    ]
    return subprocess.call(cmd, cwd=str(ROOT))


_SPECIAL: dict[str, Callable[[list[str]], int]] = {
    "api": _run_uvicorn,
    "worker": _run_rq_worker,
    "streamlit": _run_streamlit,
}


def _all_commands() -> list[str]:
    return sorted([*_SCRIPTS.keys(), *_SPECIAL.keys()])


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        prog="sar.py",
        description="SAR denoising project CLI — forwards to existing scripts or modules.",
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=_all_commands(),
        help="Subcommand to run",
    )
    parser.add_argument(
        "forwarded",
        nargs=argparse.REMAINDER,
        help="Arguments passed through (use -- before flags if needed)",
    )
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        print("\nCommands map to:", file=sys.stderr)
        for k, v in sorted(_SCRIPTS.items()):
            print(f"  {k:18} -> {v.relative_to(ROOT)}", file=sys.stderr)
        print("  api                -> python -m uvicorn api.main:app", file=sys.stderr)
        print("  worker             -> python -m rq.cli worker sar_denoise", file=sys.stderr)
        print("  streamlit          -> python -m streamlit run demo/streamlit_app.py", file=sys.stderr)
        return 1

    fwd = args.forwarded
    if fwd[:1] == ["--"]:
        fwd = fwd[1:]

    if args.command in _SPECIAL:
        return _SPECIAL[args.command](fwd)

    target = _SCRIPTS[args.command]
    if not target.is_file():
        print(f"Missing script: {target}", file=sys.stderr)
        return 1

    return _run_script(target, fwd)


if __name__ == "__main__":
    raise SystemExit(main())
