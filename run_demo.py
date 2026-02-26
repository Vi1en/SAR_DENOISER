#!/usr/bin/env python3
"""
Quick demo runner for ADMM-PnP-DL SAR image denoising
"""
import os
import sys
import subprocess
import torch
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def run_quick_demo():
    """Run a quick demo without training"""
    print("🚀 ADMM-PnP-DL Quick Demo")
    print("=" * 40)
    
    try:
        # Import modules
        from data.sar_simulation import SARSimulator, generate_synthetic_clean_image
        from models.unet import create_model
        from algos.admm_pnp import ADMMPnP
        from algos.evaluation import calculate_metrics
        
        print("✅ All modules imported successfully")
        
        # Create test data
        print("\n📊 Creating test data...")
        clean_image = generate_synthetic_clean_image(128)
        simulator = SARSimulator(psf_sigma=1.0, speckle_factor=0.3, noise_sigma=0.05)
        noisy_image = simulator.simulate_sar(clean_image)
        
        print(f"✅ Generated test images: {clean_image.shape}")
        
        # Create model
        print("\n🧠 Creating model...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        denoiser = create_model('unet', n_channels=1, noise_conditioning=True)
        print(f"✅ Model created on {device}")
        
        # Create ADMM-PnP
        print("\n🔄 Running ADMM-PnP...")
        admm = ADMMPnP(denoiser, device=device, max_iter=10)  # Short demo
        result = admm.denoise(noisy_image)
        denoised_image = result['denoised']
        
        print(f"✅ Denoising completed in {result['iterations']} iterations")
        
        # Calculate metrics
        metrics = calculate_metrics(clean_image, denoised_image)
        print(f"\n📈 Results:")
        print(f"  PSNR: {metrics['psnr']:.2f} dB")
        print(f"  SSIM: {metrics['ssim']:.4f}")
        print(f"  ENL: {metrics['enl']:.2f}")
        
        # Visualize results
        print("\n🖼️ Saving visualization...")
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        
        axes[0].imshow(clean_image, cmap='gray')
        axes[0].set_title('Clean Image')
        axes[0].axis('off')
        
        axes[1].imshow(noisy_image, cmap='gray')
        axes[1].set_title('Noisy Image')
        axes[1].axis('off')
        
        axes[2].imshow(denoised_image, cmap='gray')
        axes[2].set_title('Denoised Image')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig('demo_results.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print("✅ Demo completed successfully!")
        print("📁 Results saved to: demo_results.png")
        
        return True
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        return False

def main():
    """Main function"""
    print("This is a quick demo of ADMM-PnP-DL SAR image denoising.")
    print("It will create synthetic data and run denoising without training.")
    print()
    
    response = input("Continue with demo? (y/n): ").lower().strip()
    if response != 'y':
        print("Demo cancelled.")
        return
    
    success = run_quick_demo()
    
    if success:
        print("\n🎉 Demo completed successfully!")
        print("\nNext steps:")
        print("1. Train models: python train.py")
        print("2. Run full evaluation: python evaluate.py")
        print("3. Launch interactive demo: streamlit run demo/app.py")
    else:
        print("\n❌ Demo failed. Please check the error messages above.")
        print("Make sure all dependencies are installed: pip install -r requirements.txt")

if __name__ == "__main__":
    main()


