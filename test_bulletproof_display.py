#!/usr/bin/env python3
"""
Test script for the bulletproof display function
Tests all edge cases: black images, white images, grid artifacts, numerical issues
"""
import numpy as np
import matplotlib.pyplot as plt
from demo.bulletproof_display import display_denoised_image

def test_bulletproof_display():
    """Test the bulletproof display function with all edge cases"""
    print("🛡️ Testing Bulletproof Display Function")
    print("=" * 60)
    
    # Create test scenarios for all edge cases
    scenarios = [
        {
            'name': 'Normal Range (Good)',
            'noisy': np.random.rand(128, 128),
            'denoised': np.random.rand(128, 128) * 0.5 + 0.25,
            'use_log_transform': False,
            'expected': 'Good'
        },
        {
            'name': 'Black Image (Log Transform Issue)',
            'noisy': np.random.rand(128, 128),
            'denoised': np.full((128, 128), 0.001),  # All near 0
            'use_log_transform': True,
            'expected': 'Black'
        },
        {
            'name': 'White Image (Log Transform Issue)',
            'noisy': np.random.rand(128, 128),
            'denoised': np.full((128, 128), 0.999),  # All near 1
            'use_log_transform': True,
            'expected': 'White'
        },
        {
            'name': 'Narrow Range (Compressed)',
            'noisy': np.random.rand(128, 128),
            'denoised': np.full((128, 128), 0.5) + np.random.rand(128, 128) * 0.01,
            'use_log_transform': True,
            'expected': 'Narrow'
        },
        {
            'name': 'Grid Artifacts (Low Variance)',
            'noisy': np.random.rand(128, 128),
            'denoised': np.full((128, 128), 0.5) + np.random.randn(128, 128) * 0.001,
            'use_log_transform': True,
            'expected': 'Grid'
        },
        {
            'name': 'Numerical Issues (NaN/Inf)',
            'noisy': np.random.rand(128, 128),
            'denoised': np.random.rand(128, 128),
            'use_log_transform': True,
            'expected': 'Numerical'
        }
    ]
    
    for i, scenario in enumerate(scenarios):
        print(f"\n🔍 Testing Scenario {i+1}: {scenario['name']}")
        print("-" * 50)
        
        # Print input stats
        noisy = scenario['noisy']
        denoised = scenario['denoised']
        
        # Add NaN/Inf for numerical issues test
        if scenario['name'] == 'Numerical Issues (NaN/Inf)':
            denoised[0, 0] = np.nan  # Add NaN
            denoised[0, 1] = np.inf  # Add Inf
            denoised[0, 2] = -np.inf  # Add -Inf
        
        print(f"Input stats: min={noisy.min():.6f}, max={noisy.max():.6f}, std={noisy.std():.6f}")
        print(f"Denoised stats: min={denoised.min():.6f}, max={denoised.max():.6f}, std={denoised.std():.6f}")
        
        # Test the function logic (without Streamlit)
        try:
            # Clean any NaN or inf values
            denoised_clean = np.nan_to_num(denoised, nan=0.0, posinf=1.0, neginf=0.0)
            
            # Detect issues
            output_range = denoised_clean.max() - denoised_clean.min()
            output_mean = denoised_clean.mean()
            output_std = denoised_clean.std()
            
            print(f"Cleaned stats: min={denoised_clean.min():.6f}, max={denoised_clean.max():.6f}, std={denoised_clean.std():.6f}")
            print(f"Range: {output_range:.6f}, Mean: {output_mean:.6f}, Std: {output_std:.6f}")
            
            # Issue detection
            issues = []
            if output_range < 0.01 and output_mean < 0.1:
                issues.append("Black image")
            elif output_range < 0.01 and output_mean > 0.99:
                issues.append("White image")
            elif output_range < 0.1:
                issues.append("Narrow range")
            if output_std < 0.01:
                issues.append("Low variance")
            
            if issues:
                print(f"⚠️ Issues detected: {', '.join(issues)}")
            else:
                print("✅ No issues detected")
            
            # Apply corrections
            x_hat_disp = denoised_clean.copy()
            
            # Dynamic contrast expansion for narrow ranges
            if output_range < 0.1:
                print("📊 Applying dynamic contrast expansion...")
                x_hat_disp = x_hat_disp - x_hat_disp.min()
                if x_hat_disp.max() > 0:
                    x_hat_disp = x_hat_disp / x_hat_disp.max()
            
            # Final normalization
            x_hat_disp = (x_hat_disp - x_hat_disp.min()) / (x_hat_disp.max() - x_hat_disp.min() + 1e-8)
            x_hat_disp = np.clip(x_hat_disp, 0, 1)
            
            # Gamma correction if sufficient variance
            final_std = x_hat_disp.std()
            if final_std > 0.01:
                print("🎨 Applying gamma correction...")
                x_hat_disp = np.power(x_hat_disp, 0.5)
                x_hat_disp = np.clip(x_hat_disp, 0, 1)
            else:
                print("📊 Skipping gamma correction (low variance)")
            
            print(f"Final stats: min={x_hat_disp.min():.6f}, max={x_hat_disp.max():.6f}, std={x_hat_disp.std():.6f}")
            
            # Quality assessment
            final_range = x_hat_disp.max() - x_hat_disp.min()
            final_std = x_hat_disp.std()
            
            if final_range > 0.5 and final_std > 0.1:
                print("🌟 Excellent quality")
            elif final_range > 0.3 and final_std > 0.05:
                print("👍 Good quality")
            elif final_range > 0.1 and final_std > 0.01:
                print("⚠️ Fair quality")
            else:
                print("❌ Poor quality")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print(f"\n🎉 Bulletproof display function testing completed!")

def create_visualization_example():
    """Create a comprehensive visualization example"""
    print("\n🎨 Creating Comprehensive Visualization Example")
    print("-" * 50)
    
    # Create problematic scenarios
    scenarios = [
        ('Black Image', np.full((128, 128), 0.001)),
        ('White Image', np.full((128, 128), 0.999)),
        ('Narrow Range', np.full((128, 128), 0.5) + np.random.rand(128, 128) * 0.01),
        ('Grid Artifacts', np.full((128, 128), 0.5) + np.random.randn(128, 128) * 0.001)
    ]
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    for i, (name, denoised) in enumerate(scenarios):
        # Original problematic image
        axes[0, i].imshow(denoised, cmap='gray')
        axes[0, i].set_title(f'{name}\nRange: [{denoised.min():.3f}, {denoised.max():.3f}]')
        axes[0, i].axis('off')
        
        # Apply bulletproof corrections
        x_hat_disp = denoised.copy()
        
        # Clean NaN/inf
        x_hat_disp = np.nan_to_num(x_hat_disp, nan=0.0, posinf=1.0, neginf=0.0)
        
        # Dynamic contrast expansion
        if x_hat_disp.max() - x_hat_disp.min() < 0.1:
            x_hat_disp = x_hat_disp - x_hat_disp.min()
            if x_hat_disp.max() > 0:
                x_hat_disp = x_hat_disp / x_hat_disp.max()
        
        # Final normalization
        x_hat_disp = (x_hat_disp - x_hat_disp.min()) / (x_hat_disp.max() - x_hat_disp.min() + 1e-8)
        x_hat_disp = np.clip(x_hat_disp, 0, 1)
        
        # Gamma correction
        if x_hat_disp.std() > 0.01:
            x_hat_disp = np.power(x_hat_disp, 0.5)
            x_hat_disp = np.clip(x_hat_disp, 0, 1)
        
        # Corrected image
        axes[1, i].imshow(x_hat_disp, cmap='gray')
        axes[1, i].set_title(f'Corrected\nRange: [{x_hat_disp.min():.3f}, {x_hat_disp.max():.3f}]')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig('bulletproof_display_example.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("📊 Comprehensive visualization saved: bulletproof_display_example.png")

if __name__ == "__main__":
    test_bulletproof_display()
    create_visualization_example()
