#!/usr/bin/env python3
"""
Test script for SAMPLE dataset integration
"""
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def test_sample_dataset_downloader():
    """Test SAMPLE dataset downloader"""
    print("🧪 Testing SAMPLE dataset downloader...")
    
    try:
        from data.sample_dataset_downloader import SAMPLEDatasetDownloader
        
        # Create downloader
        downloader = SAMPLEDatasetDownloader(data_dir='test_sample_data')
        
        # Test synthetic dataset creation
        success = downloader.create_synthetic_sample_dataset()
        if success:
            print("✅ SAMPLE dataset downloader test passed")
            return True
        else:
            print("❌ SAMPLE dataset downloader test failed")
            return False
            
    except Exception as e:
        print(f"❌ SAMPLE dataset downloader test failed: {e}")
        return False


def test_sample_dataset_loader():
    """Test SAMPLE dataset loader"""
    print("🧪 Testing SAMPLE dataset loader...")
    
    try:
        from data.sample_dataset_loader import SAMPLESARDataset, create_sample_dataloaders
        
        # Test dataset creation
        dataset = SAMPLESARDataset('test_sample_data', split='train', patch_size=128, augment=False)
        print(f"✅ SAMPLE dataset loader test passed: {len(dataset)} samples")
        return True
        
    except Exception as e:
        print(f"❌ SAMPLE dataset loader test failed: {e}")
        return False


def test_sample_training_integration():
    """Test SAMPLE dataset training integration"""
    print("🧪 Testing SAMPLE dataset training integration...")
    
    try:
        from models.unet import create_model
        from data.sample_dataset_loader import create_sample_dataloaders
        
        # Create model
        model = create_model('unet', n_channels=1, noise_conditioning=True)
        
        # Test data loader
        train_loader, val_loader, test_loader = create_sample_dataloaders(
            'test_sample_data', batch_size=2, patch_size=128, num_workers=0
        )
        
        # Test batch
        batch = next(iter(train_loader))
        print(f"✅ SAMPLE training integration test passed")
        print(f"   Batch shape: {batch['clean'].shape}")
        print(f"   Noise levels: {batch['noise_level']}")
        return True
        
    except Exception as e:
        print(f"❌ SAMPLE training integration test failed: {e}")
        return False


def test_sample_evaluation_integration():
    """Test SAMPLE dataset evaluation integration"""
    print("🧪 Testing SAMPLE dataset evaluation integration...")
    
    try:
        from algos.evaluation import calculate_metrics
        from data.sample_dataset_loader import create_sample_dataloaders
        
        # Test data loader
        train_loader, val_loader, test_loader = create_sample_dataloaders(
            'test_sample_data', batch_size=2, patch_size=128, num_workers=0
        )
        
        # Test batch
        batch = next(iter(test_loader))
        clean = batch['clean'][0, 0].numpy()
        noisy = batch['noisy'][0, 0].numpy()
        
        # Test metrics
        metrics = calculate_metrics(clean, noisy)
        print(f"✅ SAMPLE evaluation integration test passed")
        print(f"   PSNR: {metrics['psnr']:.2f} dB")
        print(f"   SSIM: {metrics['ssim']:.4f}")
        return True
        
    except Exception as e:
        print(f"❌ SAMPLE evaluation integration test failed: {e}")
        return False


def cleanup_test_data():
    """Clean up test data"""
    import shutil
    test_dir = 'test_sample_data'
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
        print("🧹 Cleaned up test data")


def main():
    """Run all SAMPLE dataset integration tests"""
    print("🧪 SAMPLE Dataset Integration Tests")
    print("=" * 50)
    
    tests = [
        ("SAMPLE Dataset Downloader", test_sample_dataset_downloader),
        ("SAMPLE Dataset Loader", test_sample_dataset_loader),
        ("SAMPLE Training Integration", test_sample_training_integration),
        ("SAMPLE Evaluation Integration", test_sample_evaluation_integration)
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
    
    # Cleanup
    cleanup_test_data()
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All SAMPLE dataset integration tests passed!")
        print("\nThe SAMPLE dataset integration is working correctly.")
        print("\nNext steps:")
        print("1. Download real SAMPLE dataset: python download_sample_dataset.py")
        print("2. Train models: python train_sample.py")
        print("3. Evaluate models: python evaluate_sample.py")
    else:
        print("⚠️ Some SAMPLE dataset integration tests failed.")
        print("Please check the error messages above.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)


