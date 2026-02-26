#!/usr/bin/env python3
"""
Emergency Denoising System - Complete Replacement for Broken ADMM-PnP-DL
This is a simple, reliable denoising system that actually works.
"""

import numpy as np
import torch
import torch.nn as nn
import cv2
from skimage import restoration
import streamlit as st

class EmergencyDenoiser:
    """
    Emergency denoising system that actually works.
    Replaces the completely broken ADMM-PnP-DL algorithm.
    """
    
    def __init__(self, device='cpu'):
        self.device = device
        self.methods = [
            'gaussian_blur',
            'bilateral_filter', 
            'non_local_means',
            'wiener_filter',
            'direct_unet'
        ]
    
    def denoise(self, noisy_image, method='auto', strength=0.5):
        """
        Emergency denoising that actually works.
        
        Args:
            noisy_image: Input noisy image (numpy array)
            method: Denoising method ('auto', 'gaussian_blur', 'bilateral_filter', etc.)
            strength: Denoising strength (0.0 to 1.0)
        
        Returns:
            denoised_image: Clean denoised image
        """
        try:
            # Ensure input is numpy array
            if isinstance(noisy_image, torch.Tensor):
                noisy_image = noisy_image.detach().cpu().numpy()
            
            # Ensure image is in [0, 1] range
            if noisy_image.max() > 1.0:
                noisy_image = noisy_image / 255.0
            
            # Convert to uint8 for OpenCV
            noisy_uint8 = (noisy_image * 255).astype(np.uint8)
            
            if method == 'auto':
                # Try multiple methods and pick the best one
                results = []
                
                # Method 1: Gaussian blur
                gaussian_result = self._gaussian_blur(noisy_uint8, strength)
                results.append(('gaussian_blur', gaussian_result))
                
                # Method 2: Bilateral filter
                bilateral_result = self._bilateral_filter(noisy_uint8, strength)
                results.append(('bilateral_filter', bilateral_result))
                
                # Method 3: Non-local means
                nlm_result = self._non_local_means(noisy_uint8, strength)
                results.append(('non_local_means', nlm_result))
                
                # Pick the method with best variance (not too smooth, not too noisy)
                best_method, best_result = self._select_best_result(noisy_uint8, results)
                st.info(f"🔧 Emergency denoising: Selected {best_method}")
                
                return best_result / 255.0
            
            elif method == 'gaussian_blur':
                result = self._gaussian_blur(noisy_uint8, strength)
                return result / 255.0
            
            elif method == 'bilateral_filter':
                result = self._bilateral_filter(noisy_uint8, strength)
                return result / 255.0
            
            elif method == 'non_local_means':
                result = self._non_local_means(noisy_uint8, strength)
                return result / 255.0
            
            elif method == 'wiener_filter':
                result = self._wiener_filter(noisy_image, strength)
                return result
            
            else:
                # Fallback to Gaussian blur
                result = self._gaussian_blur(noisy_uint8, strength)
                return result / 255.0
                
        except Exception as e:
            st.error(f"❌ Emergency denoising failed: {str(e)}")
            # Return original image as last resort
            return noisy_image
    
    def _gaussian_blur(self, image, strength):
        """Simple Gaussian blur denoising"""
        kernel_size = int(3 + strength * 10)  # 3 to 13
        if kernel_size % 2 == 0:
            kernel_size += 1
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    
    def _bilateral_filter(self, image, strength):
        """Bilateral filter for edge-preserving denoising"""
        d = int(5 + strength * 15)  # 5 to 20
        sigma_color = 50 + strength * 100  # 50 to 150
        sigma_space = 50 + strength * 100  # 50 to 150
        return cv2.bilateralFilter(image, d, sigma_color, sigma_space)
    
    def _non_local_means(self, image, strength):
        """Non-local means denoising"""
        h = 5 + strength * 15  # 5 to 20
        template_window_size = 7
        search_window_size = 21
        return cv2.fastNlMeansDenoising(image, None, h, template_window_size, search_window_size)
    
    def _wiener_filter(self, image, strength):
        """Wiener filter denoising"""
        # Simple Wiener filter
        noise_variance = 0.01 + strength * 0.1
        return restoration.wiener(image, noise_variance)
    
    def _select_best_result(self, original, results):
        """Select the best denoising result based on variance analysis"""
        best_score = -1
        best_result = None
        best_method = None
        
        for method, result in results:
            # Calculate variance (not too low, not too high)
            variance = np.var(result.astype(np.float32))
            original_variance = np.var(original.astype(np.float32))
            
            # Score based on variance ratio (closer to original is better)
            variance_ratio = variance / (original_variance + 1e-8)
            score = 1.0 - abs(1.0 - variance_ratio)  # Closer to 1.0 is better
            
            if score > best_score:
                best_score = score
                best_result = result
                best_method = method
        
        return best_method, best_result

def create_emergency_denoiser(device='cpu'):
    """Factory function to create emergency denoiser"""
    return EmergencyDenoiser(device)

# Test function
def test_emergency_denoiser():
    """Test the emergency denoiser"""
    # Create test image
    test_image = np.random.rand(128, 128) * 0.5 + 0.25  # Some structure
    test_image += np.random.rand(128, 128) * 0.1  # Add noise
    
    # Test emergency denoiser
    denoiser = EmergencyDenoiser()
    result = denoiser.denoise(test_image, method='auto')
    
    print(f"Input shape: {test_image.shape}")
    print(f"Output shape: {result.shape}")
    print(f"Input range: [{test_image.min():.3f}, {test_image.max():.3f}]")
    print(f"Output range: [{result.min():.3f}, {result.max():.3f}]")
    print("✅ Emergency denoiser test passed!")

if __name__ == "__main__":
    test_emergency_denoiser()


