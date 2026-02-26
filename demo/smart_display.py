#!/usr/bin/env python3
"""
Smart diagnostic and visualization function for SAR denoising
Automatically detects and fixes display issues with comprehensive debugging
"""
import streamlit as st
import numpy as np
import torch
from algos.admm_pnp import ADMMPnP
from models.unet import create_model

def display_denoised_image(x_noisy, x_hat, use_log_transform=True, denoiser=None, device='cpu'):
    """
    Smart diagnostic and visualization function for SAR denoising outputs.
    
    Args:
        x_noisy: Input noisy image (numpy array)
        x_hat: Denoised output (numpy array) 
        use_log_transform: Whether log-transform was used
        denoiser: Optional denoiser model for reprocessing
        device: Device for reprocessing
        
    Returns:
        x_hat_disp: Final display-ready denoised image
    """
    
    st.markdown("### 🔍 Input Stats")
    st.code(f"""
Input Image Statistics:
  min: {x_noisy.min():.6f}
  max: {x_noisy.max():.6f}
  mean: {x_noisy.mean():.6f}
  std: {x_noisy.std():.6f}
  range: {x_noisy.max() - x_noisy.min():.6f}
    """)
    
    st.markdown("### 📊 Denoised Stats")
    st.code(f"""
Denoised Image Statistics:
  min: {x_hat.min():.6f}
  max: {x_hat.max():.6f}
  mean: {x_hat.mean():.6f}
  std: {x_hat.std():.6f}
  range: {x_hat.max() - x_hat.min():.6f}
    """)
    
    # Detect problematic outputs
    output_range = x_hat.max() - x_hat.min()
    output_std = x_hat.std()
    
    st.markdown("### 🚨 Quality Diagnostics")
    
    issues_detected = []
    
    # Check for narrow dynamic range
    if output_range < 0.1:
        issues_detected.append(f"⚠️ Narrow dynamic range: {output_range:.6f} (< 0.1)")
        st.warning(f"🔍 Detected narrow output range: [{x_hat.min():.4f}, {x_hat.max():.4f}]")
    
    # Check for low variance (grid-like or washed out)
    if output_std < 0.01:
        issues_detected.append(f"⚠️ Low variance (grid-like): {output_std:.6f} (< 0.01)")
        st.warning(f"🔍 Detected low variance output: std = {output_std:.6f}")
    
    # Check for all-white or all-black
    if output_range < 0.001:
        issues_detected.append(f"❌ Extremely narrow range: {output_range:.6f}")
        st.error(f"🔍 Extremely narrow range detected: {output_range:.6f}")
    
    # Auto-correction logic
    x_hat_disp = x_hat.copy()
    auto_correction_used = False
    
    if issues_detected:
        st.markdown("### 🔧 Auto-Correction Analysis")
        
        # If log-transform was used and we have issues, try without it
        if use_log_transform and denoiser is not None:
            st.info("🔁 Auto Self-Correction Activated")
            st.info("🔄 Reprocessing without log-transform...")
            
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
                
                # Check if correction improved things
                corrected_range = x_hat_corrected.max() - x_hat_corrected.min()
                corrected_std = x_hat_corrected.std()
                
                st.success(f"✅ Reprocessing completed!")
                st.code(f"""
Corrected Image Statistics:
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
    
    # Apply dynamic range compression if needed
    if not auto_correction_used:
        current_range = x_hat_disp.max() - x_hat_disp.min()
        if current_range < 0.1:
            st.info("📊 Applying dynamic range compression...")
            
            # Dynamic range compression
            x_hat_disp = x_hat_disp - x_hat_disp.min()
            if x_hat_disp.max() > 0:
                x_hat_disp = x_hat_disp / x_hat_disp.max()
            
            st.success(f"✅ Range expanded from {current_range:.6f} to {x_hat_disp.max() - x_hat_disp.min():.6f}")
    
    # Apply gamma correction for better contrast
    st.info("🎨 Applying gamma correction (power 0.5)...")
    x_hat_disp = np.power(x_hat_disp, 0.5)
    
    # Final normalization to [0,1]
    x_hat_disp = (x_hat_disp - x_hat_disp.min()) / (x_hat_disp.max() - x_hat_disp.min() + 1e-8)
    
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
    
    # Summary
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


def create_smart_display_wrapper():
    """
    Create a wrapper function that can be easily integrated into the main Streamlit app
    """
    
    def smart_display_integration(noisy_image, denoised_image, use_log_transform, denoiser=None, device='cpu'):
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
        
        # Convert to numpy if needed
        if isinstance(noisy_image, torch.Tensor):
            noisy_image = noisy_image.cpu().numpy()
        if isinstance(denoised_image, torch.Tensor):
            denoised_image = denoised_image.cpu().numpy()
        
        # Ensure 2D arrays
        if len(noisy_image.shape) > 2:
            noisy_image = noisy_image.squeeze()
        if len(denoised_image.shape) > 2:
            denoised_image = denoised_image.squeeze()
        
        # Call the smart display function
        return display_denoised_image(
            noisy_image, 
            denoised_image, 
            use_log_transform=use_log_transform,
            denoiser=denoiser,
            device=device
        )
    
    return smart_display_integration


# Example usage function
def example_usage():
    """
    Example of how to use the smart display function in your Streamlit app
    """
    
    st.markdown("""
    ### 📝 Usage Example
    
    Replace your current image display code with:
    
    ```python
    # After running ADMM denoising
    result = admm.denoise(noisy_image)
    denoised_image = result['denoised']
    
    # Use smart display function
    from demo.smart_display import display_denoised_image
    
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
    
    st.info("💡 This function will automatically detect issues and apply corrections!")


if __name__ == "__main__":
    # Demo the function
    st.title("🧠 Smart SAR Denoising Display Function")
    example_usage()


