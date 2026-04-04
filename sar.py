#!/usr/bin/env python3
"""Forward to ``scripts/sar.py`` — run ``python sar.py`` from the repository root."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
raise SystemExit(
    subprocess.call(
        [sys.executable, str(ROOT / "scripts" / "sar.py"), *sys.argv[1:]],
        cwd=str(ROOT),
    )
)
