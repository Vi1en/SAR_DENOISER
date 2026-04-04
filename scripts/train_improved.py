#!/usr/bin/env python3
"""
Improved training script for better SAR denoising performance.

Delegates to ``trainers.pipeline.run_training`` with defaults matching the
historical inline implementation. Optional ``--config`` loads YAML (see ``configs/train/``).
"""
import argparse
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from trainers.config_dataclass import TrainingConfig
from trainers.improved_trainer import ImprovedTrainer  # backward-compatible re-export
from trainers.pipeline import run_training

__all__ = ["ImprovedTrainer", "main", "run_training", "TrainingConfig"]


def main():
    parser = argparse.ArgumentParser(
        description="Improved SAR denoiser training (SAMPLE patches + ImprovedTrainer)"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="YAML training config (e.g. configs/train/default.yaml). Paths relative to repo root.",
    )
    args = parser.parse_args()

    if args.config:
        from trainers.config_loader import training_config_from_yaml_path

        cfg = training_config_from_yaml_path(Path(args.config))
    else:
        cfg = TrainingConfig()

    run_training(cfg)


if __name__ == "__main__":
    main()
