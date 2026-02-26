#!/usr/bin/env python3
"""
Debug script to identify the scaling/display issue
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from models.unet import create_model
from algos.admm_pnp import ADMMPnP
from data.sample_dataset_loader import create_sample_dataloaders

def debug_scaling_issue():
    """Debug the scaling issue step by step"""
    print("🔍 DEBUGGING SCALING ISSUE")
    print("=" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load a test image
    try:
        _, _, test_loader = create_sample_dataloaders(
            'data/sample_sar/processed', batch_size=1, patch_size=128
        )
        batch = next(iter(test_loader))
        clean_image = batch['clean'][0, 0].numpy()
        noisy_image = batch['noisy'][0, 0].numpy()
        print("✅ Loaded test image from SAMPLE dataset")
        print(f"📊 Original noisy image range: [{noisy_image.min():.6f}, {noisy_image.max():.6f}]")
    except Exception as e:
        print(f"⚠️ Could not load dataset: {e}")
        # Create synthetic test image
        clean_image = np.ones((128, 128)) * 0.5
        clean_image[32:96, 32:96] = 0.8  # Square in center
        # Add synthetic speckle noise
        speckle = np.random.gamma(1, 0.3, clean_image.shape)
        noisy_image = clean_image * speckle
        print("✅ Created synthetic test image")
        print(f"📊 Original noisy image range: [{noisy_image.min():.6f}, {noisy_image.max():.6f}]")
    
    # Load model
    try:
        model = create_model('unet', n_channels=1, noise_conditioning=False)
        checkpoint = torch.load('checkpoints_improved/best_model.pth', map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print("✅ Loaded improved U-Net model")
    except Exception as e:
        print(f"❌ Could not load improved model: {e}")
        return
    
    # Test with safe log-transform
    print("\n🧪 Testing with Safe Log-Transform")
    print("-" * 40)
    
    admm = ADMMPnP(
        model, 
        device=device,
        rho_init=1.0,
        alpha=0.3,
        theta=0.5,
        max_iter=5,  # Short run for debugging
        use_log_transform=True
    )
    
    # Run denoising
    result = admm.denoise(noisy_image)
    denoised = result['denoised']
    
    print(f"\n📊 FINAL RESULTS:")
    print(f"   x_hat.min(): {denoised.min():.6f}")
    print(f"   x_hat.max(): {denoised.max():.6f}")
    print(f"   x_hat.mean(): {denoised.mean():.6f}")
    
    # Test different display methods
    print(f"\n🎨 Testing Display Methods:")
    
    # Method 1: Direct display
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(denoised, cmap='gray')
    plt.title(f"Direct Display\nRange: [{denoised.min():.3f}, {denoised.max():.3f}]")
    plt.axis('off')
    
    # Method 2: Dynamic range compression (your suggestion)
    x_hat_disp = denoised - denoised.min()
    if x_hat_disp.max() > 0:
        x_hat_disp = x_hat_disp / x_hat_disp.max()
    x_hat_disp = np.power(x_hat_disp, 0.5)  # Gamma compression
    
    plt.subplot(1, 3, 2)
    plt.imshow(x_hat_disp, cmap='gray')
    plt.title(f"Dynamic Range Compression\nRange: [{x_hat_disp.min():.3f}, {x_hat_disp.max():.3f}]")
    plt.axis('off')
    
    # Method 3: Log visualization for ultra-low contrast
    plt.subplot(1, 3, 3)
    plt.imshow(np.log1p(denoised), cmap='gray')
    plt.title(f"Log Visualization\nRange: [{np.log1p(denoised).min():.3f}, {np.log1p(denoised).max():.3f}]")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('debug_scaling_issue.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("📊 Debug visualization saved: debug_scaling_issue.png")
    
    # Test without log-transform for comparison
    print(f"\n🧪 Testing WITHOUT Log-Transform (for comparison)")
    print("-" * 50)
    
    admm_no_log = ADMMPnP(
        model, 
        device=device,
        rho_init=3.0,
        alpha=0.3,
        theta=0.5,
        max_iter=5,
        use_log_transform=False
    )
    
    result_no_log = admm_no_log.denoise(noisy_image)
    denoised_no_log = result_no_log['denoised']
    
    print(f"📊 WITHOUT log-transform:")
    print(f"   x_hat.min(): {denoised_no_log.min():.6f}")
    print(f"   x_hat.max(): {denoised_no_log.max():.6f}")
    print(f"   x_hat.mean(): {denoised_no_log.mean():.6f}")
    
    # Create comparison plot
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 3, 1)
    plt.imshow(noisy_image, cmap='gray')
    plt.title('Input (Noisy)')
    plt.axis('off')
    
    plt.subplot(2, 3, 2)
    plt.imshow(denoised, cmap='gray')
    plt.title(f'With Log-Transform\nRange: [{denoised.min():.3f}, {denoised.max():.3f}]')
    plt.axis('off')
    
    plt.subplot(2, 3, 3)
    plt.imshow(denoised_no_log, cmap='gray')
    plt.title(f'Without Log-Transform\nRange: [{denoised_no_log.min():.3f}, {denoised_no_log.max():.3f}]')
    plt.axis('off')
    
    plt.subplot(2, 3, 4)
    plt.imshow(x_hat_disp, cmap='gray')
    plt.title('Dynamic Range Compression')
    plt.axis('off')
    
    plt.subplot(2, 3, 5)
    plt.imshow(np.log1p(denoised), cmap='gray')
    plt.title('Log Visualization')
    plt.axis('off')
    
    plt.subplot(2, 3, 6)
    plt.imshow(clean_image, cmap='gray')
    plt.title('Ground Truth')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('scaling_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("📊 Comparison plot saved: scaling_comparison.png")
    
    print(f"\n🎯 SUMMARY:")
    print(f"   Log-transform output range: [{denoised.min():.6f}, {denoised.max():.6f}]")
    print(f"   No log-transform range: [{denoised_no_log.min():.6f}, {denoised_no_log.max():.6f}]")
    
    if denoised.max() - denoised.min() < 0.01:
        print("   ⚠️ WARNING: Very narrow dynamic range detected!")
        print("   💡 Solution: Use dynamic range compression or log visualization")
    else:
        print("   ✅ Dynamic range looks reasonable")

if __name__ == "__main__":
    debug_scaling_issue()


