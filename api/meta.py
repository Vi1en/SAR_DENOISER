"""Build / deploy metadata for the HTTP API (health checks, ops)."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent


def git_sha_short() -> str | None:
    """Return short git SHA, or ``SAR_GIT_SHA`` when set (e.g. Docker build arg)."""
    env = os.environ.get("SAR_GIT_SHA", "").strip()
    if env:
        return env
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True,
            cwd=_REPO_ROOT,
        ).strip()
    except Exception:
        return None
