"""
Training script for SAR image denoiser
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import json
from datetime import datetime

from models.unet import create_model
from data.sar_simulation import create_dataloaders, create_synthetic_dataset, prepare_dataset
from algos.admm_pnp import ADMMPnP


class SSIMLoss(nn.Module):
    """SSIM loss implementation"""
    
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self.create_window(window_size, self.channel)

    def gaussian(self, window_size, sigma=1.5):
        gauss = torch.Tensor([np.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()

    def create_window(self, window_size, channel):
        _1D_window = self.gaussian(window_size).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = self.create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel

        mu1 = F.conv2d(img1, window, padding=self.window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=self.window_size//2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1*mu2

        sigma1_sq = F.conv2d(img1*img1, window, padding=self.window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, window, padding=self.window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1*img2, window, padding=self.window_size//2, groups=channel) - mu1_mu2

        C1 = 0.01**2
        C2 = 0.03**2

        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

        if self.size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)


class DenoiserTrainer:
    """Trainer for SAR image denoiser"""
    
    def __init__(self, model, device='cuda', lr=1e-4, l1_weight=1.0, ssim_weight=0.1):
        self.model = model.to(device)
        self.device = device
        self.lr = lr
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        
        # Loss functions
        self.l1_loss = nn.L1Loss()
        self.ssim_loss = SSIMLoss()
        
        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10
        )
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        
    def compute_loss(self, pred, target, noise_level=None):
        """Compute combined loss"""
        # L1 loss
        l1 = self.l1_loss(pred, target)
        
        # SSIM loss
        ssim = 1 - self.ssim_loss(pred, target)  # Convert to loss (1 - SSIM)
        
        # Total loss
        total_loss = self.l1_weight * l1 + self.ssim_weight * ssim
        
        return total_loss, l1, ssim
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        total_l1 = 0
        total_ssim = 0
        
        pbar = tqdm(train_loader, desc="Training")
        for batch in pbar:
            clean = batch['clean'].to(self.device)
            noisy = batch['noisy'].to(self.device)
            noise_level = batch['noise_level'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            if hasattr(self.model, 'noise_conditioning') and self.model.noise_conditioning:
                pred = self.model(noisy, noise_level)
            else:
                pred = self.model(noisy)
            
            # Compute loss
            loss, l1, ssim = self.compute_loss(pred, clean, noise_level)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            total_l1 += l1.item()
            total_ssim += ssim.item()
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.6f}',
                'L1': f'{l1.item():.6f}',
                'SSIM': f'{ssim.item():.6f}'
            })
        
        return total_loss / len(train_loader), total_l1 / len(train_loader), total_ssim / len(train_loader)
    
    def validate_epoch(self, val_loader):
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0
        total_l1 = 0
        total_ssim = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                clean = batch['clean'].to(self.device)
                noisy = batch['noisy'].to(self.device)
                noise_level = batch['noise_level'].to(self.device)
                
                # Forward pass
                if hasattr(self.model, 'noise_conditioning') and self.model.noise_conditioning:
                    pred = self.model(noisy, noise_level)
                else:
                    pred = self.model(noisy)
                
                # Compute loss
                loss, l1, ssim = self.compute_loss(pred, clean, noise_level)
                
                # Update metrics
                total_loss += loss.item()
                total_l1 += l1.item()
                total_ssim += ssim.item()
        
        return total_loss / len(val_loader), total_l1 / len(val_loader), total_ssim / len(val_loader)
    
    def train(self, train_loader, val_loader, epochs=100, save_dir='checkpoints'):
        """Main training loop"""
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"Starting training for {epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            
            # Training
            train_loss, train_l1, train_ssim = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # Validation
            val_loss, val_l1, val_ssim = self.validate_epoch(val_loader)
            self.val_losses.append(val_loss)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
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
            print(f"Train Loss: {train_loss:.6f} (L1: {train_l1:.6f}, SSIM: {train_ssim:.6f})")
            print(f"Val Loss: {val_loss:.6f} (L1: {val_l1:.6f}, SSIM: {val_ssim:.6f})")
            print(f"LR: {self.optimizer.param_groups[0]['lr']:.2e}")
            
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'train_loss': train_loss,
                    'train_losses': self.train_losses,
                    'val_losses': self.val_losses
                }, os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth'))
        
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
        
        # Plot training curves
        self.plot_training_curves(save_dir)
        
        print(f"\nTraining completed! Best validation loss: {self.best_val_loss:.6f}")
    
    def plot_training_curves(self, save_dir):
        """Plot and save training curves"""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss (Log Scale)')
        plt.yscale('log')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """Main training function"""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create synthetic dataset if it doesn't exist
    data_dir = 'data/synthetic_sar'
    if not os.path.exists(data_dir):
        print("Creating synthetic SAR dataset...")
        create_synthetic_dataset(data_dir, num_images=1000, image_size=256)
    
    # Prepare dataset
    train_paths, val_paths, test_paths = prepare_dataset(data_dir)
    train_loader, val_loader, test_loader = create_dataloaders(
        train_paths, val_paths, test_paths, 
        batch_size=16, patch_size=128, num_workers=4
    )
    
    print(f"Dataset sizes - Train: {len(train_paths)}, Val: {len(val_paths)}, Test: {len(test_paths)}")
    
    # Create model
    model = create_model('unet', n_channels=1, noise_conditioning=True)
    
    # Create trainer
    trainer = DenoiserTrainer(model, device=device, lr=1e-4, l1_weight=1.0, ssim_weight=0.1)
    
    # Train model
    trainer.train(train_loader, val_loader, epochs=100, save_dir='checkpoints')
    
    print("Training completed successfully!")


if __name__ == "__main__":
    main()
