#!/usr/bin/env python3
"""
Test script to verify model loading works correctly
"""
import torch
import os
from models.unet import create_model

def test_model_loading():
    """Test loading both improved and simple models"""
    print("🧪 Testing Model Loading...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Test improved model
    improved_path = "checkpoints_improved/best_model.pth"
    if os.path.exists(improved_path):
        print(f"\n✅ Testing improved model: {improved_path}")
        try:
            checkpoint = torch.load(improved_path, map_location=device)
            state_dict_keys = list(checkpoint['model_state_dict'].keys())
            
            # Detect model type
            if any('inc.double_conv' in key for key in state_dict_keys):
                model_type = 'unet'
                print("🔍 Detected: U-Net model")
            elif any('dncnn' in key for key in state_dict_keys):
                model_type = 'dncnn'
                print("🔍 Detected: DnCNN model")
            else:
                model_type = 'unet'  # Default
                print("🔍 Default: U-Net model")
            
            # Create and load model
            model = create_model(model_type, n_channels=1, noise_conditioning=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            print(f"✅ Successfully loaded {model_type.upper()} model")
            print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
            print(f"   Training loss: {checkpoint.get('train_loss', 'N/A')}")
            print(f"   Validation loss: {checkpoint.get('val_loss', 'N/A')}")
            
        except Exception as e:
            print(f"❌ Failed to load improved model: {e}")
    else:
        print(f"⚠️ Improved model not found: {improved_path}")
    
    # Test simple model
    simple_path = "checkpoints_simple/best_model.pth"
    if os.path.exists(simple_path):
        print(f"\n✅ Testing simple model: {simple_path}")
        try:
            checkpoint = torch.load(simple_path, map_location=device)
            state_dict_keys = list(checkpoint['model_state_dict'].keys())
            
            # Detect model type
            if any('inc.double_conv' in key for key in state_dict_keys):
                model_type = 'unet'
                print("🔍 Detected: U-Net model")
            elif any('dncnn' in key for key in state_dict_keys):
                model_type = 'dncnn'
                print("🔍 Detected: DnCNN model")
            else:
                model_type = 'unet'  # Default
                print("🔍 Default: U-Net model")
            
            # Create and load model
            model = create_model(model_type, n_channels=1, noise_conditioning=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            print(f"✅ Successfully loaded {model_type.upper()} model")
            print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
            
        except Exception as e:
            print(f"❌ Failed to load simple model: {e}")
    else:
        print(f"⚠️ Simple model not found: {simple_path}")
    
    print("\n🎉 Model loading test completed!")

if __name__ == "__main__":
    test_model_loading()


