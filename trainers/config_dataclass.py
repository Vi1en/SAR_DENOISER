"""Training configuration for unified pipeline (YAML-ready in Step 03)."""
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class TrainingConfig:
    """Defaults match historical ``train_improved.py`` behavior."""

    data_dir: Path = Path("data/sample_sar/processed")
    patch_size: int = 128
    batch_size: int = 8
    num_workers: int = 2
    epochs: int = 20
    lr: float = 2e-4
    device: str = "auto"  # resolved to cuda/cpu in run_training / config_loader
    checkpoint_dir: Path = Path("checkpoints_improved")
    model_type: str = "unet"
    noise_conditioning: bool = False
    seed: int = 42
    results_png: Path = _REPO_ROOT / "assets" / "images" / "improved_training_results.png"
