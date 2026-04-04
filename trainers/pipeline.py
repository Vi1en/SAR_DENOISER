"""
Unified improved training pipeline (SAMPLE patches + ImprovedTrainer).
Entry point: run_training(TrainingConfig).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from trainers.config_dataclass import TrainingConfig
from trainers.config_loader import resolve_training_device


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _ensure_repo_on_path() -> None:
    root = str(_repo_root())
    if root not in sys.path:
        sys.path.insert(0, root)


def run_training(config: TrainingConfig) -> None:
    """
    Run the same flow as legacy ``train_improved.py`` main(): dataloaders, train, quick test + PNG.
    """
    _ensure_repo_on_path()

    import torch

    from algos.evaluation import calculate_metrics
    from data.sample_dataset_loader import create_sample_dataloaders
    from models.unet import create_model
    from trainers.improved_trainer import ImprovedTrainer

    config = resolve_training_device(config)

    data_dir = Path(config.data_dir)
    if not data_dir.exists():
        print(f"❌ SAMPLE dataset not found at {data_dir}")
        print("Please run: python download_sample_dataset.py")
        return

    device = torch.device(config.device)

    print("🚀 IMPROVED ADMM-PnP-DL Training")
    print("=" * 50)
    print(f"Using device: {device}")

    print("📊 Creating improved data loaders...")
    train_loader, val_loader, test_loader = create_sample_dataloaders(
        str(data_dir),
        batch_size=config.batch_size,
        patch_size=config.patch_size,
        num_workers=config.num_workers,
    )

    print(f"✅ Data loaders created:")
    print(f"   Train: {len(train_loader.dataset)} samples, {len(train_loader)} batches")
    print(f"   Val: {len(val_loader.dataset)} samples, {len(val_loader)} batches")
    print(f"   Test: {len(test_loader.dataset)} samples, {len(test_loader)} batches")

    print("🧠 Creating improved model...")
    model = create_model(
        config.model_type,
        n_channels=1,
        noise_conditioning=config.noise_conditioning,
    )

    trainer = ImprovedTrainer(model, device=device, lr=config.lr)

    ckpt_dir = Path(config.checkpoint_dir)
    print("\n🔄 Starting improved training...")
    trainer.train(
        train_loader,
        val_loader,
        epochs=config.epochs,
        save_dir=str(ckpt_dir),
    )

    print("\n🧪 Testing improved model...")
    model.eval()
    with torch.no_grad():
        batch = next(iter(test_loader))
        clean = batch["clean"].to(device)
        noisy = batch["noisy"].to(device)

        pred = model(noisy)

        clean_np = clean[0, 0].cpu().numpy()
        noisy_np = noisy[0, 0].cpu().numpy()
        pred_np = pred[0, 0].cpu().numpy()

        metrics_noisy = calculate_metrics(clean_np, noisy_np)
        metrics_pred = calculate_metrics(clean_np, pred_np)

        print(f"📈 IMPROVED Results:")
        print(f"   Noisy PSNR: {metrics_noisy['psnr']:.2f} dB")
        print(f"   Denoised PSNR: {metrics_pred['psnr']:.2f} dB")
        print(f"   Improvement: {metrics_pred['psnr'] - metrics_noisy['psnr']:.2f} dB")

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(clean_np, cmap="gray")
        axes[0].set_title("Clean")
        axes[0].axis("off")

        axes[1].imshow(noisy_np, cmap="gray")
        axes[1].set_title(f'Noisy (PSNR: {metrics_noisy["psnr"]:.2f} dB)')
        axes[1].axis("off")

        axes[2].imshow(pred_np, cmap="gray")
        axes[2].set_title(f'Improved Denoised (PSNR: {metrics_pred["psnr"]:.2f} dB)')
        axes[2].axis("off")

        plt.tight_layout()
        out_png = Path(config.results_png)
        out_png.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(os.fspath(out_png), dpi=150, bbox_inches="tight")
        plt.close()

        print(f"✅ Improved results saved to: {out_png}")

    print("\n🎉 IMPROVED training completed successfully!")
    print(f"📁 Checkpoints saved in: {ckpt_dir}/")
