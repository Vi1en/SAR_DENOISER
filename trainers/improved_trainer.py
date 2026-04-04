"""
ImprovedTrainer — perceptual (L1 + gradient) loss, AdamW, cosine warm restarts.
Used by trainers.pipeline.run_training and re-exported from train_improved for compatibility.
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


class ImprovedTrainer:
    """Improved trainer with advanced techniques for better denoising"""

    def __init__(self, model, device="cpu", lr=1e-4):
        self.model = model.to(device)
        self.device = device
        self.lr = lr

        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()

        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-5)

        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )

        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float("inf")

    def perceptual_loss(self, pred, target):
        l1 = self.l1_loss(pred, target)

        pred_grad_x = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
        target_grad_x = torch.abs(target[:, :, :, 1:] - target[:, :, :, :-1])
        pred_grad_y = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])
        target_grad_y = torch.abs(target[:, :, 1:, :] - target[:, :, :-1, :])

        grad_loss = self.l1_loss(pred_grad_x, target_grad_x) + self.l1_loss(
            pred_grad_y, target_grad_y
        )

        return l1 + 0.1 * grad_loss

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0

        pbar = tqdm(train_loader, desc="Training")
        for batch in pbar:
            clean = batch["clean"].to(self.device)
            noisy = batch["noisy"].to(self.device)

            self.optimizer.zero_grad()
            pred = self.model(noisy)

            loss = self.perceptual_loss(pred, clean)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"Loss": f"{loss.item():.6f}"})

        return total_loss / len(train_loader)

    def validate_epoch(self, val_loader):
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                clean = batch["clean"].to(self.device)
                noisy = batch["noisy"].to(self.device)

                pred = self.model(noisy)
                loss = self.perceptual_loss(pred, clean)

                total_loss += loss.item()

        return total_loss / len(val_loader)

    def train(self, train_loader, val_loader, epochs=20, save_dir="checkpoints_improved"):
        save_dir = os.fspath(save_dir)
        os.makedirs(save_dir, exist_ok=True)

        print(f"🚀 Starting IMPROVED training for {epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Optimizer: AdamW with weight decay")
        print(f"Scheduler: CosineAnnealingWarmRestarts")
        print(f"Loss: Perceptual (L1 + Gradient)")

        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")

            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)

            val_loss = self.validate_epoch(val_loader)
            self.val_losses.append(val_loss)

            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]["lr"]

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "scheduler_state_dict": self.scheduler.state_dict(),
                        "val_loss": val_loss,
                        "train_loss": train_loss,
                        "lr": current_lr,
                    },
                    os.path.join(save_dir, "best_model.pth"),
                )

                print(f"🎯 New best model saved! Val loss: {val_loss:.6f}")

            print(f"Train Loss: {train_loss:.6f}")
            print(f"Val Loss: {val_loss:.6f}")
            print(f"LR: {current_lr:.2e}")

        torch.save(
            {
                "epoch": epochs,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "val_loss": val_loss,
                "train_loss": train_loss,
                "lr": current_lr,
                "train_losses": self.train_losses,
                "val_losses": self.val_losses,
            },
            os.path.join(save_dir, "final_model.pth"),
        )

        print(f"\n🎉 IMPROVED training completed!")
        print(f"Best validation loss: {self.best_val_loss:.6f}")

        self.plot_training_curves(save_dir)

    def plot_training_curves(self, save_dir):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        save_dir = os.fspath(save_dir)
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label="Train Loss")
        plt.plot(self.val_losses, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Curves")
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(self.train_losses, label="Train Loss")
        plt.plot(self.val_losses, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss (Log Scale)")
        plt.title("Training Curves (Log Scale)")
        plt.yscale("log")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "training_curves.png"), dpi=150, bbox_inches="tight")
        plt.close()

        print(f"📊 Training curves saved to: {save_dir}/training_curves.png")
