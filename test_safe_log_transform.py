#!/usr/bin/env python3
"""
Test script for the improved safe log-transform implementation
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from models.unet import create_model
from algos.admm_pnp import ADMMPnP
from data.sample_dataset_loader import create_sample_dataloaders

def test_safe_log_transform():
    """Test the improved safe log-transform implementation"""
    print("🧪 Testing Safe Log-Transform Implementation")
    print("=" * 60)
    
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
        # Create synthetic test image with proper SAR characteristics
        clean_image = np.ones((128, 128)) * 0.5
        clean_image[32:96, 32:96] = 0.8  # Square in center
        # Add synthetic speckle noise (multiplicative)
        speckle = np.random.gamma(1, 0.3, clean_image.shape)
        noisy_image = clean_image * speckle
        print("✅ Created synthetic SAR test image")
    
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
    
    # Test configurations with your recommended safer parameters
    configurations = [
        {
            'name': 'No Log Transform (Baseline)',
            'params': {'rho_init': 3.0, 'alpha': 0.3, 'theta': 0.5, 'max_iter': 30, 'use_log_transform': False}
        },
        {
            'name': 'Safe Log Transform (Recommended)',
            'params': {'rho_init': 1.0, 'alpha': 0.3, 'theta': 0.5, 'max_iter': 30, 'use_log_transform': True}
        },
        {
            'name': 'Sharp Edges (Lower Rho)',
            'params': {'rho_init': 0.8, 'alpha': 0.2, 'theta': 0.5, 'max_iter': 40, 'use_log_transform': True}
        },
        {
            'name': 'Conservative (Very Low Rho)',
            'params': {'rho_init': 0.7, 'alpha': 0.25, 'theta': 0.45, 'max_iter': 35, 'use_log_transform': True}
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
            
            # Check for numerical stability
            if np.any(np.isnan(denoised)) or np.any(np.isinf(denoised)):
                print(f"   ⚠️ Warning: Numerical instability detected")
            else:
                print(f"   ✅ Numerically stable")
                
        except Exception as e:
            print(f"   ❌ Failed: {e}")
    
    # Create comparison visualization
    if results:
        create_safe_comparison_plot(clean_image, noisy_image, results)
    
    print("\n🎉 Safe log-transform testing completed!")
    print("📊 Check 'safe_log_transform_comparison.png' for visual results")

def create_safe_comparison_plot(clean, noisy, results):
    """Create comparison visualization for safe log-transform"""
    n_configs = len(results) + 2  # +2 for clean and noisy
    fig, axes = plt.subplots(3, n_configs, figsize=(4*n_configs, 12))
    
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
        
        # Residual convergence
        axes[2, col].plot(result['residuals'], 'r-', linewidth=2)
        axes[2, col].set_title(f'Residual Convergence', fontsize=10)
        axes[2, col].set_xlabel('Iteration')
        axes[2, col].set_ylabel('Residual')
        axes[2, col].grid(True, alpha=0.3)
    
    # Labels for empty spaces
    axes[1, 0].text(0.5, 0.5, 'Energy\nComparison', ha='center', va='center', 
                    fontsize=12, transform=axes[1, 0].transAxes)
    axes[1, 0].axis('off')
    
    axes[1, 1].text(0.5, 0.5, 'Residual\nComparison', ha='center', va='center', 
                    fontsize=12, transform=axes[1, 1].transAxes)
    axes[1, 1].axis('off')
    
    axes[2, 0].text(0.5, 0.5, 'Convergence\nAnalysis', ha='center', va='center', 
                    fontsize=12, transform=axes[2, 0].transAxes)
    axes[2, 0].axis('off')
    
    axes[2, 1].text(0.5, 0.5, 'Stability\nCheck', ha='center', va='center', 
                    fontsize=12, transform=axes[2, 1].transAxes)
    axes[2, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('safe_log_transform_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("📊 Safe comparison plot saved: safe_log_transform_comparison.png")

def test_log_transform_safety():
    """Test the safety of the log-transform implementation"""
    print("\n🔬 Testing Log-Transform Safety")
    print("-" * 40)
    
    # Test edge cases
    test_cases = [
        ("Normal range [0,1]", np.random.rand(64, 64)),
        ("High values [0,100]", np.random.rand(64, 64) * 100),
        ("Near zero", np.full((64, 64), 1e-8)),
        ("Mixed values", np.array([[1e-10, 0.5, 100], [1e-6, 1.0, 50]])),
    ]
    
    # Create a dummy model for testing transforms
    dummy_model = create_model('unet', n_channels=1, noise_conditioning=False)
    admm = ADMMPnP(dummy_model, device='cpu')  # Just for testing transforms
    
    for name, test_image in test_cases:
        print(f"Testing: {name}")
        
        try:
            # Forward transform
            log_image = admm.safe_log_transform(test_image)
            
            # Check for NaN or inf
            has_nan = np.any(np.isnan(log_image))
            has_inf = np.any(np.isinf(log_image))
            
            # Inverse transform
            exp_image = admm.safe_exp_transform(log_image, test_image.max())
            
            # Check for NaN or inf
            has_nan_inv = np.any(np.isnan(exp_image))
            has_inf_inv = np.any(np.isinf(exp_image))
            
            print(f"  ✅ Forward: NaN={has_nan}, Inf={has_inf}")
            print(f"  ✅ Inverse: NaN={has_nan_inv}, Inf={has_inf_inv}")
            print(f"  📊 Range: [{exp_image.min():.2e}, {exp_image.max():.2e}]")
            
        except Exception as e:
            print(f"  ❌ Failed: {e}")

if __name__ == "__main__":
    test_log_transform_safety()
    test_safe_log_transform()
