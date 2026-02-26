#!/usr/bin/env python3
"""
Test ADMM-PnP integration with trained denoiser
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.sample_dataset_loader import create_sample_dataloaders
from models.unet import create_model
from algos.admm_pnp import ADMMPnP
from algos.evaluation import calculate_metrics


def test_admm_integration():
    """Test ADMM-PnP with trained denoiser"""
    print("🧪 Testing ADMM-PnP Integration with Trained Denoiser")
    print("=" * 60)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load trained model
    checkpoint_path = 'checkpoints_simple/best_model.pth'
    if not os.path.exists(checkpoint_path):
        print(f"❌ Trained model not found at {checkpoint_path}")
        print("Please run: python train_simple.py")
        return
    
    # Create model
    model = create_model('unet', n_channels=1, noise_conditioning=False)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device)['model_state_dict'])
    model.eval()
    model.to(device)
    print("✅ Loaded trained denoiser model")
    
    # Create data loader
    data_dir = 'data/sample_sar/processed'
    if not os.path.exists(data_dir):
        print(f"❌ SAMPLE dataset not found at {data_dir}")
        return
    
    _, _, test_loader = create_sample_dataloaders(
        data_dir, batch_size=1, patch_size=128, num_workers=0
    )
    
    # Get a test sample
    batch = next(iter(test_loader))
    clean = batch['clean'].to(device)
    noisy = batch['noisy'].to(device)
    
    print(f"📊 Test sample shapes: Clean={clean.shape}, Noisy={noisy.shape}")
    
    # Test 1: Direct denoiser
    print("\n🔍 Test 1: Direct Denoiser")
    with torch.no_grad():
        direct_pred = model(noisy)
        
        # Calculate metrics
        clean_np = clean[0, 0].cpu().numpy()
        noisy_np = noisy[0, 0].cpu().numpy()
        direct_pred_np = direct_pred[0, 0].cpu().numpy()
        
        metrics_noisy = calculate_metrics(clean_np, noisy_np)
        metrics_direct = calculate_metrics(clean_np, direct_pred_np)
        
        print(f"   Noisy PSNR: {metrics_noisy['psnr']:.2f} dB")
        print(f"   Direct Denoised PSNR: {metrics_direct['psnr']:.2f} dB")
        print(f"   Direct Improvement: {metrics_direct['psnr'] - metrics_noisy['psnr']:.2f} dB")
    
    # Test 2: ADMM-PnP with trained denoiser
    print("\n🔍 Test 2: ADMM-PnP with Trained Denoiser")
    
    # Create ADMM-PnP instance
    admm = ADMMPnP(
        denoiser=model,
        device=device,
        rho_init=1.0,
        alpha=0.1,
        theta=0.1,
        max_iter=10,
        tol=1e-4
    )
    
    # Run ADMM-PnP
    with torch.no_grad():
        admm_result = admm.denoise(noisy)
        # The result is in 'denoised' key
        admm_pred = admm_result['denoised']
        if isinstance(admm_pred, torch.Tensor):
            admm_pred_np = admm_pred[0, 0].cpu().numpy()
        else:
            admm_pred_np = admm_pred[0, 0]
        
        metrics_admm = calculate_metrics(clean_np, admm_pred_np)
        
        print(f"   ADMM-PnP PSNR: {metrics_admm['psnr']:.2f} dB")
        print(f"   ADMM-PnP Improvement: {metrics_admm['psnr'] - metrics_noisy['psnr']:.2f} dB")
        print(f"   ADMM vs Direct: {metrics_admm['psnr'] - metrics_direct['psnr']:.2f} dB")
    
    # Test 3: Compare different ADMM parameters
    print("\n🔍 Test 3: ADMM Parameter Comparison")
    
    param_configs = [
        {'rho_init': 0.5, 'alpha': 0.05, 'theta': 0.05, 'name': 'Conservative'},
        {'rho_init': 1.0, 'alpha': 0.1, 'theta': 0.1, 'name': 'Standard'},
        {'rho_init': 2.0, 'alpha': 0.2, 'theta': 0.2, 'name': 'Aggressive'}
    ]
    
    results = []
    for config in param_configs:
        admm_test = ADMMPnP(
            denoiser=model,
            device=device,
            rho_init=config['rho_init'],
            alpha=config['alpha'],
            theta=config['theta'],
            max_iter=10,
            tol=1e-4
        )
        
        with torch.no_grad():
            result = admm_test.denoise(noisy)
            result_tensor = result['denoised']
            if isinstance(result_tensor, torch.Tensor):
                result_np = result_tensor[0, 0].cpu().numpy()
            else:
                result_np = result_tensor[0, 0]
            metrics = calculate_metrics(clean_np, result_np)
            
            results.append({
                'config': config['name'],
                'psnr': metrics['psnr'],
                'ssim': metrics['ssim']
            })
            
            print(f"   {config['name']}: PSNR={metrics['psnr']:.2f} dB, SSIM={metrics['ssim']:.4f}")
    
    # Create visualization
    print("\n📊 Creating visualization...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Row 1: Images
    axes[0, 0].imshow(clean_np, cmap='gray')
    axes[0, 0].set_title('Clean')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(noisy_np, cmap='gray')
    axes[0, 1].set_title(f'Noisy (PSNR: {metrics_noisy["psnr"]:.2f} dB)')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(direct_pred_np, cmap='gray')
    axes[0, 2].set_title(f'Direct Denoiser (PSNR: {metrics_direct["psnr"]:.2f} dB)')
    axes[0, 2].axis('off')
    
    # Row 2: ADMM results
    axes[1, 0].imshow(admm_pred_np, cmap='gray')
    axes[1, 0].set_title(f'ADMM-PnP (PSNR: {metrics_admm["psnr"]:.2f} dB)')
    axes[1, 0].axis('off')
    
    # Plot metrics comparison
    configs = [r['config'] for r in results]
    psnrs = [r['psnr'] for r in results]
    
    axes[1, 1].bar(configs, psnrs, color=['skyblue', 'lightgreen', 'lightcoral'])
    axes[1, 1].set_title('ADMM Parameter Comparison')
    axes[1, 1].set_ylabel('PSNR (dB)')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    # Plot improvement comparison
    improvements = [r['psnr'] - metrics_noisy['psnr'] for r in results]
    axes[1, 2].bar(configs, improvements, color=['skyblue', 'lightgreen', 'lightcoral'])
    axes[1, 2].set_title('PSNR Improvement')
    axes[1, 2].set_ylabel('Improvement (dB)')
    axes[1, 2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('admm_integration_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✅ ADMM integration test completed!")
    print("📁 Results saved to: admm_integration_results.png")
    
    # Summary
    print("\n📈 Summary:")
    print(f"   Direct Denoiser: {metrics_direct['psnr']:.2f} dB")
    print(f"   ADMM-PnP (Standard): {metrics_admm['psnr']:.2f} dB")
    print(f"   Best ADMM Config: {max(results, key=lambda x: x['psnr'])['config']} ({max(results, key=lambda x: x['psnr'])['psnr']:.2f} dB)")
    
    return True


if __name__ == "__main__":
    test_admm_integration()
