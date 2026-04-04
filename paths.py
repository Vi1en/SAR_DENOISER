"""Repository root paths — import after repo root is on sys.path (run from project root)."""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
ASSETS_IMAGES = REPO_ROOT / "assets" / "images"


def ensure_assets_images() -> Path:
    """Create ``assets/images`` if missing; return that path."""
    ASSETS_IMAGES.mkdir(parents=True, exist_ok=True)
    return ASSETS_IMAGES
