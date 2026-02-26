#!/usr/bin/env python3
"""
Test script to verify the improved denoising parameters
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from models.unet import create_model
from algos.admm_pnp import ADMMPnP
from data.sample_dataset_loader import create_sample_dataloaders

def test_parameter_configurations():
    """Test different parameter configurations"""
    print("🧪 Testing Improved Denoising Parameters")
    print("=" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load a test image from the dataset
    try:
        _, _, test_loader = create_sample_dataloaders(
            'data/sample_sar/processed', batch_size=1, patch_size=128
        )
        batch = next(iter(test_loader))
        clean_image = batch['clean'][0, 0].numpy()
        noisy_image = batch['noisy'][0, 0].numpy()
        print("✅ Loaded test image from SAMPLE dataset")
    except Exception as e:
        print(f"⚠️ Could not load dataset: {e}")
        # Create synthetic test image
        clean_image = np.ones((128, 128)) * 0.5
        clean_image[32:96, 32:96] = 0.8  # Square in center
        # Add synthetic speckle noise
        speckle = np.random.gamma(1, 0.3, clean_image.shape)
        noisy_image = clean_image * speckle
        print("✅ Created synthetic test image")
    
    # Load improved model
    try:
        model = create_model('unet', n_channels=1, noise_conditioning=False)
        checkpoint = torch.load('checkpoints_improved/best_model.pth', map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print("✅ Loaded improved U-Net model")
    except Exception as e:
        print(f"❌ Could not load improved model: {e}")
        return
    
    # Test configurations
    configurations = [
        {
            'name': 'Original (Over-smooth)',
            'params': {'rho_init': 1.0, 'alpha': 0.5, 'theta': 0.5, 'max_iter': 20, 'use_log_transform': False}
        },
        {
            'name': 'Balanced (Recommended)',
            'params': {'rho_init': 3.0, 'alpha': 0.3, 'theta': 0.5, 'max_iter': 30, 'use_log_transform': True}
        },
        {
            'name': 'Sharp Edges',
            'params': {'rho_init': 5.0, 'alpha': 0.2, 'theta': 0.5, 'max_iter': 40, 'use_log_transform': True}
        },
        {
            'name': 'Conservative',
            'params': {'rho_init': 4.0, 'alpha': 0.25, 'theta': 0.45, 'max_iter': 35, 'use_log_transform': True}
        }
    ]
    
    results = {}
    
    for config in configurations:
        print(f"\n🔧 Testing: {config['name']}")
        print(f"   Parameters: {config['params']}")
        
        try:
            # Create ADMM instance
            admm = ADMMPnP(model, device=device, **config['params'])
            
            # Run denoising
            result = admm.denoise(noisy_image)
            denoised = result['denoised']
            energies = result['energies']
            residuals = result['residuals']
            
            results[config['name']] = {
                'denoised': denoised,
                'energies': energies,
                'residuals': residuals,
                'params': config['params']
            }
            
            print(f"   ✅ Completed in {len(energies)} iterations")
            print(f"   📊 Final energy: {energies[-1]:.2f}")
            print(f"   📊 Final residual: {residuals[-1]:.4f}")
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
    
    # Create comparison visualization
    if results:
        create_comparison_plot(clean_image, noisy_image, results)
    
    print("\n🎉 Parameter testing completed!")
    print("📊 Check 'denoising_comparison.png' for visual results")

def create_comparison_plot(clean, noisy, results):
    """Create comparison visualization"""
    n_configs = len(results) + 2  # +2 for clean and noisy
    fig, axes = plt.subplots(2, n_configs, figsize=(4*n_configs, 8))
    
    # Clean image
    axes[0, 0].imshow(clean, cmap='gray')
    axes[0, 0].set_title('Clean (Ground Truth)', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Noisy image
    axes[0, 1].imshow(noisy, cmap='gray')
    axes[0, 1].set_title('Noisy Input', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    
    # Denoised results
    for i, (name, result) in enumerate(results.items()):
        col = i + 2
        denoised = result['denoised']
        
        axes[0, col].imshow(denoised, cmap='gray')
        axes[0, col].set_title(name, fontsize=12, fontweight='bold')
        axes[0, col].axis('off')
        
        # Energy convergence
        axes[1, col].plot(result['energies'], 'b-', linewidth=2)
        axes[1, col].set_title(f'Energy Convergence', fontsize=10)
        axes[1, col].set_xlabel('Iteration')
        axes[1, col].set_ylabel('Energy')
        axes[1, col].grid(True, alpha=0.3)
    
    # Energy comparison
    axes[1, 0].text(0.5, 0.5, 'Energy\nComparison', ha='center', va='center', 
                    fontsize=12, transform=axes[1, 0].transAxes)
    axes[1, 0].axis('off')
    
    axes[1, 1].text(0.5, 0.5, 'Residual\nComparison', ha='center', va='center', 
                    fontsize=12, transform=axes[1, 1].transAxes)
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('denoising_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("📊 Comparison plot saved: denoising_comparison.png")

if __name__ == "__main__":
    test_parameter_configurations()


