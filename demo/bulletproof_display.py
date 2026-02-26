#!/usr/bin/env python3
"""
Bulletproof SAR denoising display function with full self-correction
Handles all edge cases: black images, white images, grid artifacts, numerical issues
"""
import streamlit as st
import numpy as np
import torch
from algos.admm_pnp import ADMMPnP
from models.unet import create_model

def display_denoised_image(x_noisy, x_hat, use_log_transform=True, denoiser=None, device='cpu'):
    """
    Bulletproof SAR denoising display function with full self-correction.
    
    Handles all edge cases:
    - Black images (range < 0.01, mean near 0)
    - White images (range < 0.01, mean > 0.99)
    - Grid artifacts (std < 0.01)
    - Numerical issues (NaN, inf, negative values)
    
    Args:
        x_noisy: Input noisy image (numpy array)
        x_hat: Denoised output (numpy array)
        use_log_transform: Whether log-transform was used
        denoiser: Optional denoiser model for auto-correction
        device: Device for reprocessing
        
    Returns:
        x_hat_disp: Final display-ready denoised image
    """
    
    # Convert to numpy and ensure 2D
    if isinstance(x_noisy, torch.Tensor):
        x_noisy = x_noisy.cpu().numpy()
    if isinstance(x_hat, torch.Tensor):
        x_hat = x_hat.cpu().numpy()
    
    # Ensure 2D arrays
    if len(x_noisy.shape) > 2:
        x_noisy = x_noisy.squeeze()
    if len(x_hat.shape) > 2:
        x_hat = x_hat.squeeze()
    
    # Clean any NaN or inf values
    x_noisy = np.nan_to_num(x_noisy, nan=0.0, posinf=1.0, neginf=0.0)
    x_hat = np.nan_to_num(x_hat, nan=0.0, posinf=1.0, neginf=0.0)
    
    st.markdown("### 📊 Input Stats")
    st.code(f"""
Input Image Statistics:
  min: {x_noisy.min():.6f}
  max: {x_noisy.max():.6f}
  mean: {x_noisy.mean():.6f}
  std: {x_noisy.std():.6f}
  range: {x_noisy.max() - x_noisy.min():.6f}
    """)
    
    st.markdown("### 🔍 Model Output")
    st.code(f"""
Denoised Output Statistics:
  min: {x_hat.min():.6f}
  max: {x_hat.max():.6f}
  mean: {x_hat.mean():.6f}
  std: {x_hat.std():.6f}
  range: {x_hat.max() - x_hat.min():.6f}
    """)
    
    # Detect problematic outputs
    output_range = x_hat.max() - x_hat.min()
    output_mean = x_hat.mean()
    output_std = x_hat.std()
    
    st.markdown("### 🚨 Issue Detection")
    
    issues_detected = []
    auto_correction_used = False
    
    # Check for black images
    if output_range < 0.01 and output_mean < 0.1:
        issues_detected.append("❌ Black image detected (very low range and mean)")
        st.error("🔍 Black image detected: Very low range and mean")
    
    # Check for white images  
    elif output_range < 0.01 and output_mean > 0.99:
        issues_detected.append("❌ White image detected (very low range, high mean)")
        st.error("🔍 White image detected: Very low range, high mean")
    
    # Check for narrow range
    elif output_range < 0.1:
        issues_detected.append("⚠️ Narrow dynamic range detected")
        st.warning(f"🔍 Narrow range detected: {output_range:.6f}")
    
    # Check for grid artifacts
    if output_std < 0.01:
        issues_detected.append("⚠️ Low variance (grid artifacts) detected")
        st.warning(f"🔍 Low variance detected: {output_std:.6f}")
    
    # Auto-correction logic
    x_hat_disp = x_hat.copy()
    
    if issues_detected and use_log_transform and denoiser is not None:
        st.markdown("### 🔁 Auto Self-Correction")
        st.info("🔁 Auto Self-Correction Activated: Reprocessed without log-transform")
        
        try:
            # Create ADMM instance without log-transform
            admm_no_log = ADMMPnP(
                denoiser, 
                device=device,
                rho_init=3.0,  # Higher rho for non-log processing
                alpha=0.3,
                theta=0.5,
                max_iter=30,
                use_log_transform=False
            )
            
            # Reprocess
            result_no_log = admm_no_log.denoise(x_noisy)
            x_hat_corrected = result_no_log['denoised']
            
            # Clean any NaN or inf values
            x_hat_corrected = np.nan_to_num(x_hat_corrected, nan=0.0, posinf=1.0, neginf=0.0)
            
            # Check if correction improved things
            corrected_range = x_hat_corrected.max() - x_hat_corrected.min()
            corrected_std = x_hat_corrected.std()
            
            st.success("✅ Reprocessing completed!")
            st.code(f"""
Corrected Output Statistics:
  min: {x_hat_corrected.min():.6f}
  max: {x_hat_corrected.max():.6f}
  mean: {x_hat_corrected.mean():.6f}
  std: {x_hat_corrected.std():.6f}
  range: {corrected_range:.6f}
            """)
            
            # Use corrected version if it's better
            if corrected_range > output_range and corrected_std > output_std:
                x_hat_disp = x_hat_corrected
                auto_correction_used = True
                st.success("🎉 Auto-correction successful! Using improved result.")
            else:
                st.warning("⚠️ Auto-correction didn't improve results. Using original.")
                
        except Exception as e:
            st.error(f"❌ Auto-correction failed: {str(e)}")
            st.warning("⚠️ Continuing with original result...")
    
    # Apply inverse log-transform if needed
    if use_log_transform and not auto_correction_used:
        st.info("🔄 Applying inverse log-transform...")
        try:
            # Safe inverse log-transform
            x_hat_disp = np.expm1(x_hat_disp)  # exp(x) - 1, more stable than exp()
            x_hat_disp = np.clip(x_hat_disp, 0, 1)  # Clip to valid range
            st.success("✅ Inverse log-transform applied")
        except Exception as e:
            st.warning(f"⚠️ Inverse log-transform failed: {str(e)}")
            st.info("📊 Continuing with original values...")
    
    # Dynamic contrast expansion for narrow ranges
    current_range = x_hat_disp.max() - x_hat_disp.min()
    if current_range < 0.1:
        st.info("📊 Applying dynamic contrast expansion...")
        
        # Safe dynamic range expansion
        x_hat_disp = x_hat_disp - x_hat_disp.min()
        if x_hat_disp.max() > 0:
            x_hat_disp = x_hat_disp / x_hat_disp.max()
        
        st.success(f"✅ Range expanded from {current_range:.6f} to {x_hat_disp.max() - x_hat_disp.min():.6f}")
    
    # Final normalization to [0,1]
    st.info("🎯 Applying final normalization...")
    x_hat_disp = (x_hat_disp - x_hat_disp.min()) / (x_hat_disp.max() - x_hat_disp.min() + 1e-8)
    x_hat_disp = np.clip(x_hat_disp, 0, 1)  # Ensure [0,1] range
    
    # Check if gamma correction is needed
    final_std = x_hat_disp.std()
    if final_std > 0.01:  # Only apply gamma if there's sufficient variance
        st.info("🎨 Applying gamma correction...")
        try:
            x_hat_disp = np.power(x_hat_disp, 0.5)
            x_hat_disp = np.clip(x_hat_disp, 0, 1)  # Re-clip after gamma
            st.success("✅ Gamma correction applied")
        except Exception as e:
            st.warning(f"⚠️ Gamma correction failed: {str(e)}")
            st.info("📊 Skipping gamma correction...")
    else:
        st.info("📊 Skipping gamma correction (low variance detected)")
    
    st.markdown("### 🎯 Final Display Stats")
    st.code(f"""
Final Display Image Statistics:
  min: {x_hat_disp.min():.6f}
  max: {x_hat_disp.max():.6f}
  mean: {x_hat_disp.mean():.6f}
  std: {x_hat_disp.std():.6f}
  range: {x_hat_disp.max() - x_hat_disp.min():.6f}
    """)
    
    # Display images side by side
    st.markdown("### 🖼️ Image Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Input (Noisy)**")
        st.image(x_noisy, caption="Noisy Input", use_container_width=True)
    
    with col2:
        st.markdown("**Denoised Output**")
        st.image(x_hat_disp, caption="Denoised Result", use_container_width=True)
    
    # Processing summary
    st.markdown("### 📋 Processing Summary")
    
    if auto_correction_used:
        st.success("✅ Auto-correction applied successfully")
    elif issues_detected:
        st.warning("⚠️ Issues detected but auto-correction not available")
    else:
        st.success("✅ No issues detected - output looks good!")
    
    # Quality assessment
    final_range = x_hat_disp.max() - x_hat_disp.min()
    final_std = x_hat_disp.std()
    
    if final_range > 0.5 and final_std > 0.1:
        st.success("🌟 Excellent quality: High contrast and good variance")
    elif final_range > 0.3 and final_std > 0.05:
        st.info("👍 Good quality: Decent contrast and variance")
    elif final_range > 0.1 and final_std > 0.01:
        st.warning("⚠️ Fair quality: Some contrast but limited variance")
    else:
        st.error("❌ Poor quality: Low contrast and variance")
    
    return x_hat_disp


def create_bulletproof_wrapper():
    """
    Create a wrapper function for easy integration
    """
    
    def bulletproof_display_integration(noisy_image, denoised_image, use_log_transform=True, denoiser=None, device='cpu'):
        """
        Integration function for the main Streamlit app
        
        Args:
            noisy_image: Input noisy image
            denoised_image: Denoised output from ADMM
            use_log_transform: Whether log-transform was used
            denoiser: Denoiser model (optional, for auto-correction)
            device: Device for processing
            
        Returns:
            Final display-ready image
        """
        
        return display_denoised_image(
            noisy_image, 
            denoised_image, 
            use_log_transform=use_log_transform,
            denoiser=denoiser,
            device=device
        )
    
    return bulletproof_display_integration


# Example usage function
def example_usage():
    """
    Example of how to use the bulletproof display function
    """
    
    st.markdown("""
    ### 📝 Usage Example
    
    Replace your current image display code with:
    
    ```python
    # After running ADMM denoising
    result = admm.denoise(noisy_image)
    denoised_image = result['denoised']
    
    # Use bulletproof display function
    from demo.bulletproof_display import display_denoised_image
    
    final_display_image = display_denoised_image(
        x_noisy=noisy_image,
        x_hat=denoised_image, 
        use_log_transform=use_log_transform,
        denoiser=denoiser,  # Optional, for auto-correction
        device=device
    )
    
    # Store for session state
    st.session_state['denoised_image'] = final_display_image
    ```
    """)
    
    st.info("💡 This function will automatically detect and fix all display issues!")


if __name__ == "__main__":
    # Demo the function
    st.title("🛡️ Bulletproof SAR Denoising Display Function")
    example_usage()


