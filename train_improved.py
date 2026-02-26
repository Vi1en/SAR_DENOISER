#!/usr/bin/env python3
"""
Improved training script for better SAR denoising performance
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.sample_dataset_loader import create_sample_dataloaders
from models.unet import create_model
from algos.evaluation import calculate_metrics


class ImprovedTrainer:
    """Improved trainer with advanced techniques for better denoising"""
    
    def __init__(self, model, device='cpu', lr=1e-4):
        self.model = model.to(device)
        self.device = device
        self.lr = lr
        
        # Advanced loss function with multiple components
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        
        # Optimizer with weight decay
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-5)
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
    
    def perceptual_loss(self, pred, target):
        """Perceptual loss using L1 and gradient differences"""
        # L1 loss
        l1 = self.l1_loss(pred, target)
        
        # Gradient loss for better edge preservation
        pred_grad_x = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
        target_grad_x = torch.abs(target[:, :, :, 1:] - target[:, :, :, :-1])
        pred_grad_y = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])
        target_grad_y = torch.abs(target[:, :, 1:, :] - target[:, :, :-1, :])
        
        grad_loss = self.l1_loss(pred_grad_x, target_grad_x) + self.l1_loss(pred_grad_y, target_grad_y)
        
        return l1 + 0.1 * grad_loss
    
    def train_epoch(self, train_loader):
        """Train for one epoch with improved techniques"""
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(train_loader, desc="Training")
        for batch in pbar:
            clean = batch['clean'].to(self.device)
            noisy = batch['noisy'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            pred = self.model(noisy)
            
            # Combined loss
            loss = self.perceptual_loss(pred, clean)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            pbar.set_postfix({'Loss': f'{loss.item():.6f}'})
        
        return total_loss / len(train_loader)
    
    def validate_epoch(self, val_loader):
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                clean = batch['clean'].to(self.device)
                noisy = batch['noisy'].to(self.device)
                
                # Forward pass
                pred = self.model(noisy)
                loss = self.perceptual_loss(pred, clean)
                
                # Update metrics
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def train(self, train_loader, val_loader, epochs=20, save_dir='checkpoints_improved'):
        """Main training loop with improved techniques"""
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"🚀 Starting IMPROVED training for {epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Optimizer: AdamW with weight decay")
        print(f"Scheduler: CosineAnnealingWarmRestarts")
        print(f"Loss: Perceptual (L1 + Gradient)")
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            
            # Training
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # Validation
            val_loss = self.validate_epoch(val_loader)
            self.val_losses.append(val_loss)
            
            # Learning rate scheduling
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'val_loss': val_loss,
                    'train_loss': train_loss,
                    'lr': current_lr
                }, os.path.join(save_dir, 'best_model.pth'))
                
                print(f"🎯 New best model saved! Val loss: {val_loss:.6f}")
            
            # Print epoch results
            print(f"Train Loss: {train_loss:.6f}")
            print(f"Val Loss: {val_loss:.6f}")
            print(f"LR: {current_lr:.2e}")
        
        # Save final model
        torch.save({
            'epoch': epochs,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'train_loss': train_loss,
            'lr': current_lr,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }, os.path.join(save_dir, 'final_model.pth'))
        
        print(f"\n🎉 IMPROVED training completed!")
        print(f"Best validation loss: {self.best_val_loss:.6f}")
        
        # Plot training curves
        self.plot_training_curves(save_dir)
    
    def plot_training_curves(self, save_dir):
        """Plot training and validation curves"""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Curves')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (Log Scale)')
        plt.title('Training Curves (Log Scale)')
        plt.yscale('log')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Training curves saved to: {save_dir}/training_curves.png")


def main():
    """Main function for improved training"""
    print("🚀 IMPROVED ADMM-PnP-DL Training")
    print("=" * 50)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Check if SAMPLE dataset exists
    data_dir = 'data/sample_sar/processed'
    if not os.path.exists(data_dir):
        print(f"❌ SAMPLE dataset not found at {data_dir}")
        print("Please run: python download_sample_dataset.py")
        return
    
    # Create data loaders with larger batch size for better training
    print("📊 Creating improved data loaders...")
    train_loader, val_loader, test_loader = create_sample_dataloaders(
        data_dir, batch_size=8, patch_size=128, num_workers=2
    )
    
    print(f"✅ Data loaders created:")
    print(f"   Train: {len(train_loader.dataset)} samples, {len(train_loader)} batches")
    print(f"   Val: {len(val_loader.dataset)} samples, {len(val_loader)} batches")
    print(f"   Test: {len(test_loader.dataset)} samples, {len(test_loader)} batches")
    
    # Create improved model
    print("🧠 Creating improved model...")
    model = create_model('unet', n_channels=1, noise_conditioning=False)
    
    # Create improved trainer
    trainer = ImprovedTrainer(model, device=device, lr=2e-4)
    
    # Train model
    print("\n🔄 Starting improved training...")
    trainer.train(train_loader, val_loader, epochs=20, save_dir='checkpoints_improved')
    
    # Test the improved model
    print("\n🧪 Testing improved model...")
    model.eval()
    with torch.no_grad():
        batch = next(iter(test_loader))
        clean = batch['clean'].to(device)
        noisy = batch['noisy'].to(device)
        
        # Denoise
        pred = model(noisy)
        
        # Calculate metrics
        clean_np = clean[0, 0].cpu().numpy()
        noisy_np = noisy[0, 0].cpu().numpy()
        pred_np = pred[0, 0].cpu().numpy()
        
        metrics_noisy = calculate_metrics(clean_np, noisy_np)
        metrics_pred = calculate_metrics(clean_np, pred_np)
        
        print(f"📈 IMPROVED Results:")
        print(f"   Noisy PSNR: {metrics_noisy['psnr']:.2f} dB")
        print(f"   Denoised PSNR: {metrics_pred['psnr']:.2f} dB")
        print(f"   Improvement: {metrics_pred['psnr'] - metrics_noisy['psnr']:.2f} dB")
        
        # Save visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(clean_np, cmap='gray')
        axes[0].set_title('Clean')
        axes[0].axis('off')
        
        axes[1].imshow(noisy_np, cmap='gray')
        axes[1].set_title(f'Noisy (PSNR: {metrics_noisy["psnr"]:.2f} dB)')
        axes[1].axis('off')
        
        axes[2].imshow(pred_np, cmap='gray')
        axes[2].set_title(f'Improved Denoised (PSNR: {metrics_pred["psnr"]:.2f} dB)')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig('improved_training_results.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print("✅ Improved results saved to: improved_training_results.png")
    
    print("\n🎉 IMPROVED training completed successfully!")
    print("📁 Checkpoints saved in: checkpoints_improved/")


if __name__ == "__main__":
    main()


