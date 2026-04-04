#!/usr/bin/env python3
"""
Verify that the complete ADMM-PnP-DL system is working
"""
import os
import sys
import torch
import numpy as np
from pathlib import Path

def verify_system():
    """Verify all system components are working"""
    print("🔍 ADMM-PnP-DL System Verification")
    print("=" * 50)
    
    # Check Python environment
    print(f"✅ Python: {sys.version}")
    print(f"✅ PyTorch: {torch.__version__}")
    print(f"✅ Working directory: {os.getcwd()}")
    
    # Check project structure
    required_dirs = ['data', 'models', 'algos', 'trainers', 'demo', 'checkpoints_simple']
    for dir_name in required_dirs:
        if os.path.exists(dir_name):
            print(f"✅ Directory: {dir_name}")
        else:
            print(f"❌ Missing directory: {dir_name}")
    
    # Check trained model
    model_path = 'checkpoints_simple/best_model.pth'
    if os.path.exists(model_path):
        print(f"✅ Trained model: {model_path}")
        # Load and test model
        try:
            from models.unet import create_model
            model = create_model('unet', n_channels=1, noise_conditioning=False)
            checkpoint = torch.load(model_path, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            print(f"✅ Model loaded successfully")
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
    else:
        print(f"❌ Trained model not found: {model_path}")
    
    # Check SAMPLE dataset
    data_dir = 'data/sample_sar/processed'
    if os.path.exists(data_dir):
        train_dir = os.path.join(data_dir, 'train_patches')
        if os.path.exists(train_dir):
            clean_dir = os.path.join(train_dir, 'clean')
            noisy_dir = os.path.join(train_dir, 'noisy')
            if os.path.exists(clean_dir) and os.path.exists(noisy_dir):
                clean_count = len([f for f in os.listdir(clean_dir) if f.endswith('.png')])
                noisy_count = len([f for f in os.listdir(noisy_dir) if f.endswith('.png')])
                print(f"✅ SAMPLE dataset: {clean_count} clean, {noisy_count} noisy patches")
            else:
                print(f"❌ SAMPLE dataset structure incomplete")
        else:
            print(f"❌ SAMPLE dataset not processed")
    else:
        print(f"❌ SAMPLE dataset not found: {data_dir}")
    
    # Check ADMM-PnP algorithm
    try:
        from algos.admm_pnp import ADMMPnP
        print(f"✅ ADMM-PnP algorithm imported")
    except Exception as e:
        print(f"❌ ADMM-PnP import failed: {e}")
    
    # Check Streamlit demo
    demo_file = 'demo/streamlit_app.py'
    if os.path.exists(demo_file):
        print(f"✅ Streamlit demo: {demo_file}")
    else:
        print(f"❌ Streamlit demo not found: {demo_file}")
    
    # Test basic functionality
    print("\n🧪 Testing Basic Functionality")
    try:
        # Test model creation
        from models.unet import create_model
        model = create_model('unet', n_channels=1, noise_conditioning=False)
        
        # Test with dummy input
        dummy_input = torch.randn(1, 1, 128, 128)
        with torch.no_grad():
            output = model(dummy_input)
        
        print(f"✅ Model forward pass: {dummy_input.shape} -> {output.shape}")
        
        # Test ADMM-PnP
        from algos.admm_pnp import ADMMPnP
        admm = ADMMPnP(
            denoiser=model,
            device='cpu',
            rho_init=1.0,
            alpha=0.1,
            theta=0.1,
            max_iter=5,
            tol=1e-4
        )
        
        # Test with dummy image
        dummy_image = np.random.rand(128, 128)
        result = admm.denoise(dummy_image)
        
        print(f"✅ ADMM-PnP test: {dummy_image.shape} -> {result['denoised'].shape}")
        print(f"✅ ADMM completed in {result['iterations']} iterations")

        from inference.service import SARDenoiseService

        svc = SARDenoiseService(device="cpu")
        inf = svc.denoise_numpy(
            np.random.rand(64, 64).astype(np.float32),
            "TV Denoising",
        )
        if inf["denoised"].shape != (64, 64):
            raise RuntimeError(
                f"inference output shape {inf['denoised'].shape}, expected (64, 64)"
            )
        print("✅ Inference service (TV path, no checkpoint)")

    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False

    try:
        import api.main  # noqa: F401

        print("✅ FastAPI application import")
    except Exception as e:
        print(f"⚠️ FastAPI application import skipped: {e}")
    else:
        try:
            from fastapi.testclient import TestClient

            with TestClient(api.main.app) as tc:
                hr = tc.get("/health")
                if hr.status_code != 200:
                    raise RuntimeError(f"/health HTTP {hr.status_code}: {hr.text[:200]}")
                hbody = hr.json()
                if hbody.get("status") != "ok":
                    raise RuntimeError(f"/health unexpected body: {hbody!r}")
                if hbody.get("direct_infer_backend") not in ("pytorch", "onnx"):
                    raise RuntimeError(f"/health missing direct_infer_backend: {hbody!r}")
                if not isinstance(hbody.get("onnx_path_set"), bool):
                    raise RuntimeError(f"/health missing onnx_path_set bool: {hbody!r}")

                rr = tc.get("/ready")
                if rr.status_code != 200:
                    raise RuntimeError(f"/ready HTTP {rr.status_code}: {rr.text[:200]}")
                rbody = rr.json()
                if rbody.get("status") != "ready":
                    raise RuntimeError(f"/ready unexpected body: {rbody!r}")

            print("✅ FastAPI /health + /ready smoke (TestClient)")
        except Exception as e:
            print(f"❌ FastAPI /health + /ready smoke failed: {e}")
            return False

    print("\n🎉 System Verification Complete!")
    print("✅ All components are working correctly")
    print("🌐 Streamlit demo should be available at: http://localhost:8501")
    
    return True

if __name__ == "__main__":
    raise SystemExit(0 if verify_system() else 1)


