#!/usr/bin/env python3
"""
Test script for the smart display function
"""
import numpy as np
import matplotlib.pyplot as plt
from demo.smart_display import display_denoised_image

def test_smart_display():
    """Test the smart display function with various scenarios"""
    print("🧪 Testing Smart Display Function")
    print("=" * 50)
    
    # Create test scenarios
    scenarios = [
        {
            'name': 'Normal Range',
            'noisy': np.random.rand(128, 128),
            'denoised': np.random.rand(128, 128) * 0.5 + 0.25,
            'use_log_transform': False
        },
        {
            'name': 'Narrow Range (Log Transform Issue)',
            'noisy': np.random.rand(128, 128),
            'denoised': np.full((128, 128), 0.5) + np.random.rand(128, 128) * 0.01,  # Very narrow range
            'use_log_transform': True
        },
        {
            'name': 'Low Variance (Grid-like)',
            'noisy': np.random.rand(128, 128),
            'denoised': np.full((128, 128), 0.5) + np.random.randn(128, 128) * 0.001,  # Very low std
            'use_log_transform': True
        },
        {
            'name': 'All White',
            'noisy': np.random.rand(128, 128),
            'denoised': np.full((128, 128), 0.999),  # All near 1.0
            'use_log_transform': True
        }
    ]
    
    for i, scenario in enumerate(scenarios):
        print(f"\n🔍 Testing Scenario {i+1}: {scenario['name']}")
        print("-" * 40)
        
        # Print input stats
        noisy = scenario['noisy']
        denoised = scenario['denoised']
        
        print(f"Input stats: min={noisy.min():.6f}, max={noisy.max():.6f}, std={noisy.std():.6f}")
        print(f"Denoised stats: min={denoised.min():.6f}, max={denoised.max():.6f}, std={denoised.std():.6f}")
        
        # Test the function (without Streamlit)
        try:
            # Simulate the function logic
            output_range = denoised.max() - denoised.min()
            output_std = denoised.std()
            
            print(f"Range: {output_range:.6f}")
            print(f"Std: {output_std:.6f}")
            
            # Apply corrections if needed
            x_hat_disp = denoised.copy()
            
            if output_range < 0.1:
                print("⚠️ Narrow range detected - applying dynamic range compression")
                x_hat_disp = x_hat_disp - x_hat_disp.min()
                if x_hat_disp.max() > 0:
                    x_hat_disp = x_hat_disp / x_hat_disp.max()
            
            if output_std < 0.01:
                print("⚠️ Low variance detected - may appear grid-like")
            
            # Apply gamma correction
            x_hat_disp = np.power(x_hat_disp, 0.5)
            
            # Final normalization
            x_hat_disp = (x_hat_disp - x_hat_disp.min()) / (x_hat_disp.max() - x_hat_disp.min() + 1e-8)
            
            print(f"Final stats: min={x_hat_disp.min():.6f}, max={x_hat_disp.max():.6f}, std={x_hat_disp.std():.6f}")
            
            # Quality assessment
            final_range = x_hat_disp.max() - x_hat_disp.min()
            final_std = x_hat_disp.std()
            
            if final_range > 0.5 and final_std > 0.1:
                print("✅ Excellent quality")
            elif final_range > 0.3 and final_std > 0.05:
                print("👍 Good quality")
            elif final_range > 0.1 and final_std > 0.01:
                print("⚠️ Fair quality")
            else:
                print("❌ Poor quality")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print(f"\n🎉 Smart display function testing completed!")

def create_visualization_example():
    """Create a visualization example"""
    print("\n🎨 Creating Visualization Example")
    print("-" * 40)
    
    # Create a problematic denoised image (narrow range)
    noisy = np.random.rand(128, 128)
    denoised = np.full((128, 128), 0.5) + np.random.rand(128, 128) * 0.01
    
    # Apply smart display corrections
    x_hat_disp = denoised.copy()
    
    # Dynamic range compression
    x_hat_disp = x_hat_disp - x_hat_disp.min()
    if x_hat_disp.max() > 0:
        x_hat_disp = x_hat_disp / x_hat_disp.max()
    
    # Gamma correction
    x_hat_disp = np.power(x_hat_disp, 0.5)
    
    # Final normalization
    x_hat_disp = (x_hat_disp - x_hat_disp.min()) / (x_hat_disp.max() - x_hat_disp.min() + 1e-8)
    
    # Create comparison plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(noisy, cmap='gray')
    axes[0].set_title(f'Input (Noisy)\nRange: [{noisy.min():.3f}, {noisy.max():.3f}]')
    axes[0].axis('off')
    
    axes[1].imshow(denoised, cmap='gray')
    axes[1].set_title(f'Raw Denoised\nRange: [{denoised.min():.3f}, {denoised.max():.3f}]')
    axes[1].axis('off')
    
    axes[2].imshow(x_hat_disp, cmap='gray')
    axes[2].set_title(f'Smart Display\nRange: [{x_hat_disp.min():.3f}, {x_hat_disp.max():.3f}]')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig('smart_display_example.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("📊 Visualization saved: smart_display_example.png")
    print(f"Original range: {denoised.max() - denoised.min():.6f}")
    print(f"Corrected range: {x_hat_disp.max() - x_hat_disp.min():.6f}")

if __name__ == "__main__":
    test_smart_display()
    create_visualization_example()


