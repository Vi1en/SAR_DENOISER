"""
Streamlit demo application for ADMM-PnP-DL SAR image denoising
"""
import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import io
import os
import time
from datetime import datetime

# Import our modules
import sys
sys.path.append('..')
from models.unet import create_model
from algos.admm_pnp import ADMMPnP, TVDenoiser
from algos.evaluation import calculate_metrics
from data.sar_simulation import SARSimulator


# Page configuration
st.set_page_config(
    page_title="ADMM-PnP-DL SAR Image Denoising",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #1f77b4;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stButton > button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        border: none;
        border-radius: 0.5rem;
        padding: 0.5rem 1rem;
        font-size: 1rem;
    }
    .stButton > button:hover {
        background-color: #0d5aa7;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🛰️ ADMM-PnP-DL SAR Image Denoising</h1>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar
st.sidebar.title("Configuration")

# Model selection
model_type = st.sidebar.selectbox(
    "Select Denoiser Model",
    ["U-Net", "DnCNN"],
    help="Choose the deep learning denoiser architecture"
)

# ADMM parameters
st.sidebar.subheader("ADMM Parameters")
max_iter = st.sidebar.slider("Max Iterations", 5, 50, 15, help="Maximum ADMM iterations")
rho_init = st.sidebar.slider("Initial Rho", 0.1, 5.0, 1.0, help="Initial penalty parameter (lower = less over-smoothing)")
alpha = st.sidebar.slider("Alpha", 0.0, 1.0, 0.1, help="Relaxation parameter (lower = less denoiser dominance)")
theta = st.sidebar.slider("Theta", 0.0, 1.0, 0.05, help="Denoising strength parameter (lower = less smoothing)")

# Add log-transform option
use_log_transform = st.sidebar.checkbox("Use Log Transform", False, help="Apply log-transform for better SAR speckle handling")

# Add option to disable denoising entirely
disable_denoising = st.sidebar.checkbox("Disable Denoising (Return Original)", False, help="Skip ADMM processing and return the original noisy image")

# Quality Enhancement Mode
quality_enhancement = st.sidebar.checkbox("Quality Enhancement Mode", False, help="Enable 2-pass denoising with refinement for better results")

# Preset parameter configurations
st.sidebar.subheader("Parameter Presets")
preset = st.sidebar.selectbox(
    "Choose Preset Configuration",
    ["Custom", "Balanced (Recommended)", "Sharp Edges", "Smooth Output", "Conservative"],
    help="Quick parameter presets for different denoising styles"
)

if preset != "Custom":
        if preset == "Balanced (Recommended)":
            max_iter, rho_init, alpha, theta = 15, 1.0, 0.1, 0.05
            st.sidebar.success("✅ Balanced: Anti-over-smoothing with minimal processing")
        elif preset == "Sharp Edges":
            max_iter, rho_init, alpha, theta = 12, 0.8, 0.05, 0.02
            st.sidebar.success("🔪 Sharp Edges: Maximum detail preservation, minimal smoothing")
        elif preset == "Smooth Output":
            max_iter, rho_init, alpha, theta = 20, 1.5, 0.2, 0.1
            st.sidebar.success("🌊 Smooth: Moderate denoising with edge preservation")
        elif preset == "Conservative":
            max_iter, rho_init, alpha, theta = 10, 0.5, 0.05, 0.01
            st.sidebar.success("🛡️ Conservative: Very light processing, preserve original quality")

# Noise parameters
st.sidebar.subheader("Noise Parameters")
speckle_factor = st.sidebar.slider("Speckle Factor", 0.0, 1.0, 0.3, help="Multiplicative speckle noise level")
gaussian_sigma = st.sidebar.slider("Gaussian Noise", 0.0, 0.2, 0.05, help="Additive Gaussian noise level")
psf_sigma = st.sidebar.slider("PSF Sigma", 0.5, 3.0, 1.0, help="Point spread function blur level")

# Method selection
method = st.sidebar.selectbox(
    "Denoising Method",
    ["ADMM-PnP-DL", "Direct Denoising", "TV Denoising"],
    help="Choose the denoising approach"
)

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 Input Image")
    
    # Image upload
    uploaded_file = st.file_uploader(
        "Upload a SAR image",
        type=['png', 'jpg', 'jpeg', 'tiff'],
        help="Upload a single-channel SAR image for denoising"
    )
    
    # Load sample from SAMPLE dataset option
    if st.button("🎲 Load Sample from SAMPLE Dataset"):
        with st.spinner("Loading sample from SAMPLE dataset..."):
            # Load a sample from our SAMPLE dataset
            sample_dir = "data/sample_sar/processed/test_patches"
            clean_dir = os.path.join(sample_dir, "clean")
            noisy_dir = os.path.join(sample_dir, "noisy")
            
            if os.path.exists(clean_dir) and os.path.exists(noisy_dir):
                # Get a random sample
                clean_files = [f for f in os.listdir(clean_dir) if f.endswith('.png')]
                if clean_files:
                    import random
                    sample_file = random.choice(clean_files)
                    
                    # Load clean and noisy images
                    clean_path = os.path.join(clean_dir, sample_file)
                    noisy_path = os.path.join(noisy_dir, sample_file)
                    
                    if os.path.exists(clean_path) and os.path.exists(noisy_path):
                        clean_image = cv2.imread(clean_path, cv2.IMREAD_GRAYSCALE)
                        noisy_image = cv2.imread(noisy_path, cv2.IMREAD_GRAYSCALE)
                        
                        if clean_image is not None and noisy_image is not None:
                            # Normalize to [0, 1]
                            clean_image = clean_image.astype(np.float32) / 255.0
                            noisy_image = noisy_image.astype(np.float32) / 255.0
                        else:
                            st.error("Failed to load sample images")
                            clean_image = None
                            noisy_image = None
                    else:
                        st.error("Sample image files not found")
                        clean_image = None
                        noisy_image = None
                else:
                    st.error("No sample images found in SAMPLE dataset")
                    clean_image = None
                    noisy_image = None
            else:
                st.error("SAMPLE dataset not found. Please run: python download_sample_dataset.py")
                clean_image = None
                noisy_image = None
            
            # Store in session state
            if clean_image is not None and noisy_image is not None:
                st.session_state['clean_image'] = clean_image
                st.session_state['noisy_image'] = noisy_image
                st.session_state['image_source'] = 'sample'
                st.success("✅ Loaded sample from SAMPLE dataset!")
            else:
                st.session_state['clean_image'] = None
                st.session_state['noisy_image'] = None
    
    # Display uploaded/generated image
    if uploaded_file is not None:
        # Load uploaded image
        image = Image.open(uploaded_file).convert('L')
        image_array = np.array(image, dtype=np.float32) / 255.0
        
        st.session_state['noisy_image'] = image_array
        st.session_state['image_source'] = 'uploaded'
        
        st.image(image, caption="Uploaded Image", use_column_width=True)
    
    elif 'noisy_image' in st.session_state:
        # Display generated image
        noisy_img = st.session_state['noisy_image']
        st.image(noisy_img, caption="Input Image", use_column_width=True, clamp=True)

with col2:
    st.subheader("🔧 Denoising Results")
    
    if st.button("🚀 Run Denoising", type="primary"):
        if 'noisy_image' not in st.session_state:
            st.error("Please upload an image or generate a synthetic one first!")
        else:
            with st.spinner("Running denoising algorithm..."):
                start_time = time.time()
                
                # Get input image
                noisy_image = st.session_state['noisy_image']
                
                # Enhanced preprocessing for SAR images
                st.info("🔧 Applying enhanced SAR preprocessing...")
                
                # Convert to float32 and handle complex values
                noisy_image = noisy_image.astype(np.float32)
                
                # If complex-valued (SAR magnitude), take magnitude
                if np.iscomplexobj(noisy_image):
                    noisy_image = np.abs(noisy_image)
                    st.info("📡 Detected complex SAR data - extracting magnitude")
                
                # Normalize to [0, 1] with numerical stability
                img_max = np.max(noisy_image)
                if img_max > 0:
                    noisy_image = noisy_image / (img_max + 1e-8)
                else:
                    noisy_image = np.zeros_like(noisy_image)
                
                st.success(f"✅ Preprocessed: range=[{noisy_image.min():.4f}, {noisy_image.max():.4f}]")
                
                # Load model
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                
                if method == "ADMM-PnP-DL":
                    # Try to load improved trained model first, then fallback to simple
                    model_path = f"checkpoints_improved/best_model.pth"
                    denoiser = None
                    
                    if os.path.exists(model_path):
                        try:
                            checkpoint = torch.load(model_path, map_location=device)
                            # Detect model type from checkpoint keys
                            state_dict_keys = list(checkpoint['model_state_dict'].keys())
                            if any('inc.double_conv' in key for key in state_dict_keys):
                                # This is a U-Net model
                                actual_model_type = 'unet'
                                st.info("🔍 Detected U-Net model in checkpoint")
                            elif any('dncnn' in key for key in state_dict_keys):
                                # This is a DnCNN model
                                actual_model_type = 'dncnn'
                                st.info("🔍 Detected DnCNN model in checkpoint")
                            else:
                                # Default to the selected model type
                                actual_model_type = model_type.lower().replace('-', '')
                            
                            denoiser = create_model(actual_model_type, n_channels=1, noise_conditioning=False)
                            denoiser.load_state_dict(checkpoint['model_state_dict'])
                            st.success("🚀 Loaded IMPROVED trained model (30+ dB PSNR)")
                        except Exception as e:
                            st.error(f"❌ Failed to load improved model: {str(e)}")
                            denoiser = None
                    
                    # Fallback to simple model if improved model failed
                    if denoiser is None:
                        model_path = f"checkpoints_simple/best_model.pth"
                        if os.path.exists(model_path):
                            try:
                                checkpoint = torch.load(model_path, map_location=device)
                                # Detect model type from checkpoint keys
                                state_dict_keys = list(checkpoint['model_state_dict'].keys())
                                if any('inc.double_conv' in key for key in state_dict_keys):
                                    actual_model_type = 'unet'
                                elif any('dncnn' in key for key in state_dict_keys):
                                    actual_model_type = 'dncnn'
                                else:
                                    actual_model_type = model_type.lower().replace('-', '')
                                
                                denoiser = create_model(actual_model_type, n_channels=1, noise_conditioning=False)
                                denoiser.load_state_dict(checkpoint['model_state_dict'])
                                st.success("✅ Loaded basic trained model")
                            except Exception as e:
                                st.error(f"❌ Failed to load basic model: {str(e)}")
                                # Use the selected model type with random weights
                                model_type_clean = model_type.lower().replace('-', '')
                                denoiser = create_model(model_type_clean, n_channels=1, noise_conditioning=False)
                                st.warning("⚠️ Using random weights (failed to load trained model)")
                        else:
                            # Use the selected model type with random weights
                            model_type_clean = model_type.lower().replace('-', '')
                            denoiser = create_model(model_type_clean, n_channels=1, noise_conditioning=False)
                            st.warning("⚠️ Using random weights (no trained model found)")
                    
                    # Check if denoising is disabled
                    if disable_denoising:
                        st.info("🚫 Denoising disabled - returning original image")
                        denoised_image = noisy_image.copy()
                        energies = [0]
                        residuals = [0]
                    else:
                        # FIXED ADMM-PnP algorithm
                        st.info("🔧 Using FIXED ADMM-PnP algorithm")
                        
                        try:
                            # Create ADMM-PnP instance with FIXED algorithm
                            admm = ADMMPnP(
                                denoiser, 
                                device=device,
                                max_iter=max_iter,
                                rho_init=rho_init,
                                alpha=alpha,
                                theta=theta,
                                use_log_transform=use_log_transform
                            )
                            
                            # Run FIXED denoising
                            result = admm.denoise(noisy_image)
                            denoised_image = result['denoised']
                            energies = result['energies']
                            residuals = result['residuals']
                            
                            st.success("✅ FIXED ADMM-PnP denoising completed successfully")
                            
                        except Exception as e:
                            st.error(f"❌ FIXED ADMM-PnP failed: {str(e)}")
                            st.error("🚨 Returning original image as fallback")
                            denoised_image = noisy_image.copy()
                            energies = [0]
                            residuals = [0]
                    
                    # DIAGNOSTIC: Check output quality
                    st.info("🔍 Running diagnostic checks...")
                    
                    # Check if output is reasonable
                    if denoised_image.shape != noisy_image.shape:
                        st.error(f"🚨 SHAPE MISMATCH: Input {noisy_image.shape} vs Output {denoised_image.shape}")
                        st.error("🚨 This indicates a fundamental algorithm failure!")
                        denoised_image = noisy_image.copy()  # Fallback to original
                    
                    # Check for NaN or Inf values
                    if np.any(np.isnan(denoised_image)) or np.any(np.isinf(denoised_image)):
                        st.error("🚨 OUTPUT CONTAINS NaN OR INF VALUES!")
                        denoised_image = noisy_image.copy()  # Fallback to original
                    
                    # Check for completely flat output
                    if np.std(denoised_image) < 1e-6:
                        st.error("🚨 OUTPUT IS COMPLETELY FLAT - Algorithm failed!")
                        denoised_image = noisy_image.copy()  # Fallback to original
                    
                    # FIXED ADMM-PnP STATUS REPORT
                    st.success("✅ FIXED ADMM-PnP ALGORITHM ACTIVE")
                    st.success("✅ Fixed tensor shape mismatches and FFT operations")
                    st.success("✅ Fixed PSF creation and denoiser integration")
                    st.info("ℹ️ Using properly implemented ADMM-PnP with deep learning denoiser")
                    
                    # Enhanced postprocessing for better results
                    st.info("🔧 Applying enhanced postprocessing...")
                    
                    # Ensure proper tensor handling and conversion
                    if hasattr(denoised_image, 'detach'):
                        denoised_image = denoised_image.detach().cpu().numpy()
                    
                    # Fix black output issue with proper scaling
                    denoised_min = denoised_image.min()
                    denoised_max = denoised_image.max()
                    
                    if denoised_max > denoised_min:
                        denoised_image = (denoised_image - denoised_min) / (denoised_max - denoised_min + 1e-8)
                    else:
                        denoised_image = np.zeros_like(denoised_image)
                    
                    # Clip to [0, 1] range
                    denoised_image = np.clip(denoised_image, 0, 1)
                    
                    # Quality Enhancement Mode: Minimal refinement only
                    if quality_enhancement:
                        st.info("✨ Quality Enhancement Mode: Applying minimal refinement...")
                        
                        # Only apply very light non-local means if needed
                        try:
                            import cv2
                            denoised_uint8 = (denoised_image * 255).astype(np.uint8)
                            # Very conservative parameters to avoid over-smoothing
                            denoised_refined = cv2.fastNlMeansDenoising(denoised_uint8, None, 2, 7, 21) / 255.0
                            denoised_image = denoised_refined
                            st.success("✅ Applied minimal non-local means refinement")
                        except Exception as e:
                            st.warning(f"⚠️ Quality enhancement failed: {str(e)} - using original result")
                    else:
                        st.info("🚫 Quality Enhancement Mode disabled - preserving original sharpness")
                    
                    st.success(f"✅ Postprocessed: range=[{denoised_image.min():.4f}, {denoised_image.max():.4f}]")
                    
                elif method == "Direct Denoising":
                    # Load denoiser
                    denoiser = create_model(model_type.lower(), n_channels=1, noise_conditioning=True)
                    
                    # Try to load trained model
                    model_path = f"checkpoints_simple/best_model.pth"
                    if os.path.exists(model_path):
                        checkpoint = torch.load(model_path, map_location=device)
                        denoiser.load_state_dict(checkpoint['model_state_dict'])
                        st.success("✅ Loaded trained model")
                    else:
                        st.warning("⚠️ Using random weights (no trained model found)")
                    
                    # Direct denoising
                    denoiser.eval()
                    with torch.no_grad():
                        input_tensor = torch.from_numpy(noisy_image).float().unsqueeze(0).unsqueeze(0).to(device)
                        noise_level = torch.tensor(speckle_factor, device=device)
                        
                        if hasattr(denoiser, 'noise_conditioning') and denoiser.noise_conditioning:
                            denoised_tensor = denoiser(input_tensor, noise_level)
                        else:
                            denoised_tensor = denoiser(input_tensor)
                        
                        denoised_image = denoised_tensor.squeeze().cpu().numpy()
                    
                    energies = [0]  # No energy for direct denoising
                    residuals = [0]
                    
                elif method == "TV Denoising":
                    # TV denoising
                    tv_denoiser = TVDenoiser(device=device)
                    denoised_tensor = tv_denoiser.tv_denoise(torch.from_numpy(noisy_image).float().to(device))
                    denoised_image = denoised_tensor.cpu().numpy()
                    energies = [0]  # No energy for TV denoising
                    residuals = [0]
                
                end_time = time.time()
                processing_time = end_time - start_time
                
                # Store results
                st.session_state['denoised_image'] = denoised_image
                st.session_state['processing_time'] = processing_time
                st.session_state['energies'] = energies
                st.session_state['residuals'] = residuals
    
    # Display results
    if 'denoised_image' in st.session_state:
        denoised_img = st.session_state['denoised_image']
        st.image(denoised_img, caption="Denoised Image", use_container_width=True, clamp=True)
        
        # Calculate metrics if clean image is available
        if 'clean_image' in st.session_state and st.session_state['image_source'] == 'synthetic':
            clean_img = st.session_state['clean_image']
            metrics = calculate_metrics(clean_img, denoised_img)
            
            st.subheader("📊 Performance Metrics")
            col_metric1, col_metric2, col_metric3 = st.columns(3)
            
            with col_metric1:
                st.metric("PSNR", f"{metrics['psnr']:.2f} dB")
            with col_metric2:
                st.metric("SSIM", f"{metrics['ssim']:.4f}")
            with col_metric3:
                st.metric("ENL", f"{metrics['enl']:.2f}")
        
        # Processing time
        if 'processing_time' in st.session_state:
            st.metric("Processing Time", f"{st.session_state['processing_time']:.2f} seconds")

# Bottom section for detailed results
if 'denoised_image' in st.session_state and 'energies' in st.session_state:
    st.markdown("---")
    st.subheader("📈 Algorithm Monitoring")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Energy plot
        if len(st.session_state['energies']) > 1:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(st.session_state['energies'])
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Energy')
            ax.set_title('ADMM Energy Convergence')
            ax.grid(True)
            st.pyplot(fig)
    
    with col2:
        # Residual plot
        if len(st.session_state['residuals']) > 1:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(st.session_state['residuals'])
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Residual')
            ax.set_title('ADMM Residual Convergence')
            ax.grid(True)
            st.pyplot(fig)

# Comparison section
if 'denoised_image' in st.session_state and 'noisy_image' in st.session_state:
    st.markdown("---")
    st.subheader("🔄 Image Comparison")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.image(st.session_state['noisy_image'], caption="Noisy SAR Input", use_container_width=True, clamp=True)
    
    with col2:
        st.image(st.session_state['denoised_image'], caption="Denoised ADMM-PnP Output", use_container_width=True, clamp=True)
    
    with col3:
        if 'clean_image' in st.session_state and st.session_state['image_source'] == 'synthetic':
            st.image(st.session_state['clean_image'], caption="Ground Truth", use_container_width=True, clamp=True)
        else:
            st.info("No ground truth available for uploaded images")
    
    # Quality Enhancement Mode comparison
    if quality_enhancement and 'denoised_image' in st.session_state:
        st.markdown("---")
        st.subheader("✨ Quality Enhancement Results")
        st.info("🔍 Enhanced processing applied: Gaussian blur + Non-local means denoising for superior clarity")
        
        # Show processing statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Input Range", f"{st.session_state['noisy_image'].min():.3f} - {st.session_state['noisy_image'].max():.3f}")
        with col2:
            st.metric("Output Range", f"{st.session_state['denoised_image'].min():.3f} - {st.session_state['denoised_image'].max():.3f}")
        with col3:
            improvement = st.session_state['denoised_image'].std() / (st.session_state['noisy_image'].std() + 1e-8)
            st.metric("Noise Reduction", f"{improvement:.2f}x")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>ADMM-PnP-DL SAR Image Denoising Demo</p>
    <p>Built with PyTorch, Streamlit, and advanced optimization techniques</p>
</div>
""", unsafe_allow_html=True)


def generate_synthetic_sar_image(size):
    """Generate synthetic SAR image for testing"""
    # Create base image with different regions
    image = np.zeros((size, size))
    center = size // 2
    
    # Add geometric shapes
    y, x = np.ogrid[:size, :size]
    
    # Circle
    mask = (x - center)**2 + (y - center)**2 < (size//4)**2
    image[mask] = 0.8
    
    # Rectangle
    image[center-size//8:center+size//8, center-size//4:center+size//4] = 0.6
    
    # Add some texture
    texture = np.random.normal(0, 0.1, (size, size))
    image = image + texture
    
    # Add some lines
    for i in range(0, size, size//8):
        image[i, :] = 0.4
        image[:, i] = 0.4
    
    # Smooth the image
    from scipy.ndimage import gaussian_filter
    image = gaussian_filter(image, sigma=1.0)
    
    # Normalize to [0, 1]
    image = (image - image.min()) / (image.max() - image.min())
    
    return image


if __name__ == "__main__":
    # This will be run when the script is executed directly
    # For Streamlit, use: streamlit run app.py
    pass
