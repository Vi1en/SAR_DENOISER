"""Unified sar.py CLI forwards to existing scripts."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SAR = ROOT / "sar.py"


def test_sar_no_command_exits_nonzero():
    r = subprocess.run([sys.executable, str(SAR)], cwd=str(ROOT), capture_output=True, text=True)
    assert r.returncode == 1
    assert "train" in r.stderr or "train" in r.stdout


def test_sar_unknown_command():
    r = subprocess.run(
        [sys.executable, str(SAR), "not-a-real-command"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    assert r.returncode != 0


def test_sar_eval_forwards_help():
    r = subprocess.run(
        [sys.executable, str(SAR), "eval", "--help"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0
    assert "evaluate_sample" in r.stdout or "data_dir" in r.stdout or "Evaluation" in r.stdout


def test_sar_streamlit_version():
    pytest.importorskip("streamlit")
    r = subprocess.run(
        [sys.executable, str(SAR), "streamlit", "version"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0
    out = r.stdout + r.stderr
    assert "streamlit" in out.lower()


@pytest.mark.parametrize(
    "cmd,needle",
    [
        ("train", "config"),
        ("train-sample", "sample"),
        ("capture-baseline", "baseline"),
        ("ablation-grid", "manifest"),
        ("ablation-md", "aggregate"),
        ("compare-onnx", "onnx"),
        ("api", "uvicorn"),
        ("worker", "rq"),
    ],
)
def test_sar_forward_help(cmd: str, needle: str):
    r = subprocess.run(
        [sys.executable, str(SAR), cmd, "--help"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0
    out = r.stdout + r.stderr
    assert needle.lower() in out.lower()
