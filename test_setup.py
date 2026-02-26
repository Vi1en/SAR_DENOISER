#!/usr/bin/env python3
"""
Test script to verify the ADMM-PnP-DL setup
"""
import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test if all modules can be imported"""
    print("Testing imports...")
    
    try:
        from data.sar_simulation import SARSimulator, create_synthetic_dataset
        print("✅ Data simulation module imported successfully")
    except ImportError as e:
        print(f"❌ Data simulation import failed: {e}")
        return False
    
    try:
        from models.unet import create_model
        print("✅ Models module imported successfully")
    except ImportError as e:
        print(f"❌ Models import failed: {e}")
        return False
    
    try:
        from algos.admm_pnp import ADMMPnP, TVDenoiser
        print("✅ ADMM-PnP module imported successfully")
    except ImportError as e:
        print(f"❌ ADMM-PnP import failed: {e}")
        return False
    
    try:
        from algos.evaluation import calculate_metrics
        print("✅ Evaluation module imported successfully")
    except ImportError as e:
        print(f"❌ Evaluation import failed: {e}")
        return False
    
    return True

def test_models():
    """Test model creation and forward pass"""
    print("\nTesting models...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    try:
        from models.unet import create_model
        
        # Test U-Net
        unet = create_model('unet', n_channels=1, noise_conditioning=True)
        x = torch.randn(1, 1, 128, 128)
        noise_level = torch.randn(1)
        output = unet(x, noise_level)
        print(f"✅ U-Net: Input {x.shape} -> Output {output.shape}")
        
        # Test DnCNN
        dncnn = create_model('dncnn', channels=1, noise_conditioning=True)
        output = dncnn(x, noise_level)
        print(f"✅ DnCNN: Input {x.shape} -> Output {output.shape}")
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return False
    
    return True

def test_sar_simulation():
    """Test SAR image simulation"""
    print("\nTesting SAR simulation...")
    
    try:
        from data.sar_simulation import SARSimulator, generate_synthetic_clean_image
        
        # Create simulator
        simulator = SARSimulator(psf_sigma=1.0, speckle_factor=0.3, noise_sigma=0.05)
        
        # Generate clean image
        clean_image = generate_synthetic_clean_image(128)
        print(f"✅ Generated clean image: {clean_image.shape}, range [{clean_image.min():.3f}, {clean_image.max():.3f}]")
        
        # Simulate SAR degradation
        noisy_image = simulator.simulate_sar(clean_image)
        print(f"✅ Generated noisy image: {noisy_image.shape}, range [{noisy_image.min():.3f}, {noisy_image.max():.3f}]")
        
        # Test metrics
        from algos.evaluation import calculate_metrics
        metrics = calculate_metrics(clean_image, noisy_image)
        print(f"✅ Metrics calculated: PSNR={metrics['psnr']:.2f}, SSIM={metrics['ssim']:.4f}")
        
    except Exception as e:
        print(f"❌ SAR simulation test failed: {e}")
        return False
    
    return True

def test_admm_pnp():
    """Test ADMM-PnP algorithm"""
    print("\nTesting ADMM-PnP...")
    
    try:
        from models.unet import create_model
        from algos.admm_pnp import ADMMPnP
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create denoiser
        denoiser = create_model('unet', n_channels=1, noise_conditioning=True)
        
        # Create ADMM-PnP
        admm = ADMMPnP(denoiser, device=device, max_iter=5)  # Short test
        
        # Test image
        test_image = np.random.rand(64, 64)
        
        # Run denoising
        result = admm.denoise(test_image)
        print(f"✅ ADMM-PnP completed: {result['iterations']} iterations")
        print(f"   Final energy: {result['energies'][-1]:.6f}")
        print(f"   Final residual: {result['residuals'][-1]:.6f}")
        
    except Exception as e:
        print(f"❌ ADMM-PnP test failed: {e}")
        return False
    
    return True

def test_dataset_creation():
    """Test dataset creation"""
    print("\nTesting dataset creation...")
    
    try:
        from data.sar_simulation import create_synthetic_dataset, prepare_dataset, create_dataloaders
        
        # Create small test dataset
        test_dir = 'test_data'
        create_synthetic_dataset(test_dir, num_images=10, image_size=128)
        print(f"✅ Created test dataset in {test_dir}")
        
        # Prepare dataset
        train_paths, val_paths, test_paths = prepare_dataset(test_dir)
        print(f"✅ Dataset split: Train={len(train_paths)}, Val={len(val_paths)}, Test={len(test_paths)}")
        
        # Create data loaders
        train_loader, val_loader, test_loader = create_dataloaders(
            train_paths, val_paths, test_paths, batch_size=2, patch_size=64, num_workers=0
        )
        print(f"✅ Data loaders created: Train={len(train_loader)}, Val={len(val_loader)}, Test={len(test_loader)}")
        
        # Test batch
        batch = next(iter(train_loader))
        # Ensure tensors are contiguous
        clean_tensor = batch['clean'].contiguous()
        noisy_tensor = batch['noisy'].contiguous()
        print(f"✅ Batch test: Clean={clean_tensor.shape}, Noisy={noisy_tensor.shape}")
        
        # Clean up
        import shutil
        shutil.rmtree(test_dir)
        print("✅ Test data cleaned up")
        
    except Exception as e:
        print(f"❌ Dataset creation test failed: {e}")
        return False
    
    return True

def main():
    """Run all tests"""
    print("🧪 ADMM-PnP-DL Setup Test")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("Model Test", test_models),
        ("SAR Simulation Test", test_sar_simulation),
        ("ADMM-PnP Test", test_admm_pnp),
        ("Dataset Creation Test", test_dataset_creation)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 {test_name}")
        print("-" * 30)
        
        try:
            if test_func():
                print(f"✅ {test_name} PASSED")
                passed += 1
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} FAILED with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The setup is working correctly.")
        print("\nNext steps:")
        print("1. Run training: python train.py")
        print("2. Run evaluation: python evaluate.py")
        print("3. Launch demo: streamlit run demo/app.py")
    else:
        print("⚠️ Some tests failed. Please check the error messages above.")
        print("Make sure all dependencies are installed: pip install -r requirements.txt")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
