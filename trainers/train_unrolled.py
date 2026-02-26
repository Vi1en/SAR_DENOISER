"""
Training script for unrolled ADMM-PnP
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from models.unet import create_model
from data.sar_simulation import create_dataloaders, prepare_dataset, create_synthetic_dataset
from algos.admm_pnp import ADMMPnP


class UnrolledADMM(nn.Module):
    """Unrolled ADMM-PnP with learnable parameters"""
    
    def __init__(self, denoiser, num_iterations=5, device='cuda'):
        super(UnrolledADMM, self).__init__()
        self.denoiser = denoiser
        self.num_iterations = num_iterations
        self.device = device
        
        # Learnable parameters for each iteration
        self.rho_params = nn.Parameter(torch.ones(num_iterations) * 1.0)
        self.alpha_params = nn.Parameter(torch.ones(num_iterations) * 0.5)
        self.theta_params = nn.Parameter(torch.ones(num_iterations) * 0.5)
        
        # Learnable PSF parameters
        self.psf_sigma = nn.Parameter(torch.tensor(1.0))
        
    def create_psf(self, image_shape):
        """Create learnable PSF"""
        h, w = image_shape
        psf = torch.zeros((h, w), device=self.device)
        center_h, center_w = h // 2, w // 2
        
        # Create Gaussian PSF with learnable sigma
        sigma = torch.clamp(self.psf_sigma, 0.1, 5.0)
        for i in range(h):
            for j in range(w):
                psf[i, j] = torch.exp(-((i - center_h)**2 + (j - center_w)**2) / (2 * sigma**2))
        
        psf = psf / psf.sum()
        return psf
    
    def fft_conv(self, image, psf):
        """Convolution using FFT"""
        # Add batch and channel dimensions if needed
        if image.dim() == 2:
            image = image.unsqueeze(0).unsqueeze(0)
        if psf.dim() == 2:
            psf = psf.unsqueeze(0).unsqueeze(0)
        
        # FFT convolution
        image_fft = torch.fft.fft2(image)
        psf_fft = torch.fft.fft2(psf)
        result_fft = image_fft * psf_fft
        result = torch.fft.ifft2(result_fft).real
        
        return result.squeeze()
    
    def x_update_fft(self, z, u, H, H_conj, rho):
        """x-update step using FFT"""
        # Compute H^T * y + rho * (z - u)
        rhs = H_conj + rho * (z - u)
        
        # Solve (H^T * H + rho * I) * x = rhs using FFT
        H_conj_H = torch.conj(H) * H
        denominator = H_conj_H + rho
        
        # Avoid division by zero
        denominator = torch.where(denominator.abs() < 1e-10, 
                                 torch.ones_like(denominator) * 1e-10, 
                                 denominator)
        
        x_fft = rhs / denominator
        x = torch.fft.ifft2(x_fft).real
        
        return x
    
    def forward(self, noisy_image, clean_image=None):
        """Forward pass through unrolled ADMM"""
        # Initialize variables
        x = noisy_image.clone()
        z = noisy_image.clone()
        u = torch.zeros_like(noisy_image)
        
        # Create PSF
        psf = self.create_psf(noisy_image.shape)
        
        # Create observation y
        y = self.fft_conv(noisy_image, psf)
        
        # FFT of PSF
        H = torch.fft.fft2(psf)
        H_conj = torch.conj(H) * torch.fft.fft2(y)
        
        # Storage for intermediate results
        intermediate_results = []
        
        # Unrolled ADMM iterations
        for iteration in range(self.num_iterations):
            # Get parameters for this iteration
            rho = torch.clamp(self.rho_params[iteration], 0.1, 10.0)
            alpha = torch.clamp(self.alpha_params[iteration], 0.0, 1.0)
            theta = torch.clamp(self.theta_params[iteration], 0.0, 1.0)
            
            # x-update
            x = self.x_update_fft(z, u, H, H_conj, rho)
            
            # x-bar update
            x_bar = alpha * x + (1 - alpha) * z
            
            # Denoising step
            den_input = torch.clamp(x_bar + u, 0, 1)
            
            # Add batch dimension for denoiser
            if den_input.dim() == 2:
                den_input_batch = den_input.unsqueeze(0).unsqueeze(0)
            else:
                den_input_batch = den_input.unsqueeze(0)
            
            # Denoise using the neural network
            if hasattr(self.denoiser, 'noise_conditioning') and self.denoiser.noise_conditioning:
                # Use a default noise level for unrolled training
                noise_level = torch.tensor(0.3, device=self.device)
                z_denoised = self.denoiser(den_input_batch, noise_level)
            else:
                z_denoised = self.denoiser(den_input_batch)
            
            z_denoised = z_denoised.squeeze()
            
            # z-update
            z = theta * z_denoised + (1 - theta) * den_input
            
            # Dual update
            u = u + x - z
            
            # Store intermediate result
            intermediate_results.append(z.clone())
        
        return z, intermediate_results


class UnrolledTrainer:
    """Trainer for unrolled ADMM-PnP"""
    
    def __init__(self, model, device='cuda', lr=1e-4, l1_weight=1.0, ssim_weight=0.1):
        self.model = model.to(device)
        self.device = device
        self.lr = lr
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        
        # Loss functions
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        
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
    
    def compute_loss(self, pred, target, intermediate_results=None):
        """Compute combined loss"""
        # Final output loss
        l1_final = self.l1_loss(pred, target)
        mse_final = self.mse_loss(pred, target)
        
        # Intermediate supervision loss (optional)
        intermediate_loss = 0
        if intermediate_results is not None:
            for intermediate in intermediate_results:
                intermediate_loss += self.l1_loss(intermediate, target)
            intermediate_loss = intermediate_loss / len(intermediate_results)
        
        # Total loss
        total_loss = self.l1_weight * l1_final + 0.1 * intermediate_loss
        
        return total_loss, l1_final, mse_final, intermediate_loss
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        total_l1 = 0
        total_mse = 0
        total_intermediate = 0
        
        pbar = tqdm(train_loader, desc="Training")
        for batch in pbar:
            clean = batch['clean'].to(self.device)
            noisy = batch['noisy'].to(self.device)
            
            # Remove batch dimension for unrolled ADMM
            clean = clean.squeeze(0)
            noisy = noisy.squeeze(0)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            pred, intermediate_results = self.model(noisy, clean)
            
            # Compute loss
            loss, l1, mse, intermediate = self.compute_loss(pred, clean, intermediate_results)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            total_l1 += l1.item()
            total_mse += mse.item()
            total_intermediate += intermediate.item()
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.6f}',
                'L1': f'{l1.item():.6f}',
                'MSE': f'{mse.item():.6f}',
                'Intermediate': f'{intermediate.item():.6f}'
            })
        
        return (total_loss / len(train_loader), total_l1 / len(train_loader), 
                total_mse / len(train_loader), total_intermediate / len(train_loader))
    
    def validate_epoch(self, val_loader):
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0
        total_l1 = 0
        total_mse = 0
        total_intermediate = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                clean = batch['clean'].to(self.device)
                noisy = batch['noisy'].to(self.device)
                
                # Remove batch dimension for unrolled ADMM
                clean = clean.squeeze(0)
                noisy = noisy.squeeze(0)
                
                # Forward pass
                pred, intermediate_results = self.model(noisy, clean)
                
                # Compute loss
                loss, l1, mse, intermediate = self.compute_loss(pred, clean, intermediate_results)
                
                # Update metrics
                total_loss += loss.item()
                total_l1 += l1.item()
                total_mse += mse.item()
                total_intermediate += intermediate.item()
        
        return (total_loss / len(val_loader), total_l1 / len(val_loader), 
                total_mse / len(val_loader), total_intermediate / len(val_loader))
    
    def train(self, train_loader, val_loader, epochs=50, save_dir='checkpoints_unrolled'):
        """Main training loop"""
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"Starting unrolled ADMM training for {epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            
            # Training
            train_loss, train_l1, train_mse, train_intermediate = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # Validation
            val_loss, val_l1, val_mse, val_intermediate = self.validate_epoch(val_loader)
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
                }, os.path.join(save_dir, 'best_unrolled_model.pth'))
                
                print(f"New best model saved! Val loss: {val_loss:.6f}")
            
            # Print epoch results
            print(f"Train Loss: {train_loss:.6f} (L1: {train_l1:.6f}, MSE: {train_mse:.6f}, Intermediate: {train_intermediate:.6f})")
            print(f"Val Loss: {val_loss:.6f} (L1: {val_l1:.6f}, MSE: {val_mse:.6f}, Intermediate: {val_intermediate:.6f})")
            print(f"LR: {self.optimizer.param_groups[0]['lr']:.2e}")
            
            # Print learned parameters
            print(f"Rho params: {self.model.rho_params.data.cpu().numpy()}")
            print(f"Alpha params: {self.model.alpha_params.data.cpu().numpy()}")
            print(f"Theta params: {self.model.theta_params.data.cpu().numpy()}")
            print(f"PSF sigma: {self.model.psf_sigma.data.item():.4f}")
            
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
                }, os.path.join(save_dir, f'unrolled_checkpoint_epoch_{epoch+1}.pth'))
        
        # Save final model
        torch.save({
            'epoch': epochs,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'train_loss': train_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }, os.path.join(save_dir, 'final_unrolled_model.pth'))
        
        # Plot training curves
        self.plot_training_curves(save_dir)
        
        print(f"\nUnrolled training completed! Best validation loss: {self.best_val_loss:.6f}")
    
    def plot_training_curves(self, save_dir):
        """Plot and save training curves"""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Unrolled ADMM Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Unrolled ADMM Training and Validation Loss (Log Scale)')
        plt.yscale('log')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'unrolled_training_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """Main training function for unrolled ADMM"""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create synthetic dataset if it doesn't exist
    data_dir = 'data/synthetic_sar'
    if not os.path.exists(data_dir):
        print("Creating synthetic SAR dataset...")
        from data.sar_simulation import create_synthetic_dataset
        create_synthetic_dataset(data_dir, num_images=1000, image_size=256)
    
    # Prepare dataset
    train_paths, val_paths, test_paths = prepare_dataset(data_dir)
    train_loader, val_loader, test_loader = create_dataloaders(
        train_paths, val_paths, test_paths, 
        batch_size=1, patch_size=128, num_workers=4  # Batch size 1 for unrolled ADMM
    )
    
    print(f"Dataset sizes - Train: {len(train_paths)}, Val: {len(val_paths)}, Test: {len(test_paths)}")
    
    # Create denoiser
    denoiser = create_model('unet', n_channels=1, noise_conditioning=True)
    
    # Create unrolled ADMM model
    model = UnrolledADMM(denoiser, num_iterations=5, device=device)
    
    # Create trainer
    trainer = UnrolledTrainer(model, device=device, lr=1e-4, l1_weight=1.0, ssim_weight=0.1)
    
    # Train model
    trainer.train(train_loader, val_loader, epochs=50, save_dir='checkpoints_unrolled')
    
    print("Unrolled ADMM training completed successfully!")


if __name__ == "__main__":
    main()
