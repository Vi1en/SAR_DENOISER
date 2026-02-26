#!/usr/bin/env python3
"""
Simple training script for ADMM-PnP-DL SAR image denoising
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


class SimpleTrainer:
    """Simple trainer without noise conditioning"""
    
    def __init__(self, model, device='cpu', lr=1e-4):
        self.model = model.to(device)
        self.device = device
        self.lr = lr
        
        # Loss function
        self.loss_fn = nn.L1Loss()
        
        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(train_loader, desc="Training")
        for batch in pbar:
            clean = batch['clean'].to(self.device)
            noisy = batch['noisy'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            pred = self.model(noisy)  # No noise conditioning
            loss = self.loss_fn(pred, clean)
            
            # Backward pass
            loss.backward()
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
                pred = self.model(noisy)  # No noise conditioning
                loss = self.loss_fn(pred, clean)
                
                # Update metrics
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def train(self, train_loader, val_loader, epochs=10, save_dir='checkpoints_simple'):
        """Main training loop"""
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"Starting simple training for {epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            
            # Training
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # Validation
            val_loss = self.validate_epoch(val_loader)
            self.val_losses.append(val_loss)
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'train_loss': train_loss
                }, os.path.join(save_dir, 'best_model.pth'))
                
                print(f"New best model saved! Val loss: {val_loss:.6f}")
            
            # Print epoch results
            print(f"Train Loss: {train_loss:.6f}")
            print(f"Val Loss: {val_loss:.6f}")
        
        # Save final model
        torch.save({
            'epoch': epochs,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'train_loss': train_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }, os.path.join(save_dir, 'final_model.pth'))
        
        print(f"\nSimple training completed! Best validation loss: {self.best_val_loss:.6f}")


def main():
    """Main function"""
    print("🚀 Simple ADMM-PnP-DL Training")
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
    
    # Create data loaders
    print("📊 Creating data loaders...")
    train_loader, val_loader, test_loader = create_sample_dataloaders(
        data_dir, batch_size=4, patch_size=128, num_workers=0
    )
    
    print(f"✅ Data loaders created:")
    print(f"   Train: {len(train_loader.dataset)} samples, {len(train_loader)} batches")
    print(f"   Val: {len(val_loader.dataset)} samples, {len(val_loader)} batches")
    print(f"   Test: {len(test_loader.dataset)} samples, {len(test_loader)} batches")
    
    # Create model (without noise conditioning)
    print("🧠 Creating model...")
    model = create_model('unet', n_channels=1, noise_conditioning=False)
    
    # Create trainer
    trainer = SimpleTrainer(model, device=device, lr=1e-4)
    
    # Train model
    print("\n🔄 Starting training...")
    trainer.train(train_loader, val_loader, epochs=5, save_dir='checkpoints_simple')
    
    # Test the trained model
    print("\n🧪 Testing trained model...")
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
        
        print(f"📈 Results:")
        print(f"   Noisy PSNR: {metrics_noisy['psnr']:.2f} dB")
        print(f"   Denoised PSNR: {metrics_pred['psnr']:.2f} dB")
        print(f"   Improvement: {metrics_pred['psnr'] - metrics_noisy['psnr']:.2f} dB")
        
        # Save visualization
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        
        axes[0].imshow(clean_np, cmap='gray')
        axes[0].set_title('Clean')
        axes[0].axis('off')
        
        axes[1].imshow(noisy_np, cmap='gray')
        axes[1].set_title(f'Noisy (PSNR: {metrics_noisy["psnr"]:.2f} dB)')
        axes[1].axis('off')
        
        axes[2].imshow(pred_np, cmap='gray')
        axes[2].set_title(f'Denoised (PSNR: {metrics_pred["psnr"]:.2f} dB)')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig('simple_training_results.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print("✅ Results saved to: simple_training_results.png")
    
    print("\n🎉 Simple training completed successfully!")
    print("📁 Checkpoints saved in: checkpoints_simple/")


if __name__ == "__main__":
    main()


