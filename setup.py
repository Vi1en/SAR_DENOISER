#!/usr/bin/env python3
"""
Setup script for ADMM-PnP-DL SAR image denoising project
"""
import os
import sys
import subprocess
import torch

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 7):
        print("❌ Python 3.7 or higher is required")
        return False
    print(f"✅ Python {sys.version.split()[0]} detected")
    return True

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'torch', 'torchvision', 'numpy', 'scipy', 'scikit-image',
        'opencv-python', 'matplotlib', 'streamlit', 'tqdm', 'Pillow'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'opencv-python':
                import cv2
            elif package == 'scikit-image':
                import skimage
            elif package == 'Pillow':
                import PIL
            else:
                __import__(package.replace('-', '_'))
            print(f"✅ {package} is installed")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} is missing")
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Install them with: pip install -r requirements.txt")
        return False
    
    return True

def check_cuda():
    """Check CUDA availability"""
    if torch.cuda.is_available():
        print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA version: {torch.version.cuda}")
        return True
    else:
        print("⚠️ CUDA not available, will use CPU")
        return False

def create_directories():
    """Create necessary directories"""
    directories = [
        'data', 'models', 'trainers', 'algos', 'demo', 'notebooks',
        'checkpoints', 'checkpoints_unrolled', 'results'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created directory: {directory}")

def test_imports():
    """Test if all modules can be imported"""
    try:
        # Test data module
        from data.sar_simulation import SARSimulator
        print("✅ Data simulation module imported")
        
        # Test models module
        from models.unet import create_model
        print("✅ Models module imported")
        
        # Test algorithms module
        from algos.admm_pnp import ADMMPnP
        print("✅ ADMM-PnP module imported")
        
        # Test evaluation module
        from algos.evaluation import calculate_metrics
        print("✅ Evaluation module imported")
        
        return True
    except ImportError as e:
        print(f"❌ Import test failed: {e}")
        return False

def main():
    """Main setup function"""
    print("🚀 ADMM-PnP-DL SAR Image Denoising Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Check dependencies
    if not check_dependencies():
        print("\nPlease install missing dependencies first:")
        print("pip install -r requirements.txt")
        return False
    
    # Check CUDA
    check_cuda()
    
    # Create directories
    print("\n📁 Creating directories...")
    create_directories()
    
    # Test imports
    print("\n🧪 Testing imports...")
    if not test_imports():
        print("❌ Import test failed. Please check the error messages above.")
        return False
    
    print("\n" + "=" * 50)
    print("🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Test the setup: python test_setup.py")
    print("2. Train models: python train.py")
    print("3. Evaluate models: python evaluate.py")
    print("4. Launch demo: streamlit run demo/app.py")
    print("\nFor detailed usage, see README.md")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
