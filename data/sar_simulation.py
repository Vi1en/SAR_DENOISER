"""
SAR image simulation and data preparation utilities
"""
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy import ndimage
from skimage import filters, transform
import cv2
import os
from PIL import Image
import matplotlib.pyplot as plt


class SARSimulator:
    """Simulate SAR-like images with blur, speckle, and noise"""
    
    def __init__(self, psf_size=5, psf_sigma=1.0, speckle_factor=0.3, noise_sigma=0.05):
        self.psf_size = psf_size
        self.psf_sigma = psf_sigma
        self.speckle_factor = speckle_factor
        self.noise_sigma = noise_sigma
        
    def create_psf(self, size=None, sigma=None):
        """Create point spread function (Gaussian blur kernel)"""
        if size is None:
            size = self.psf_size
        if sigma is None:
            sigma = self.psf_sigma
            
        # Create Gaussian PSF
        psf = np.zeros((size, size))
        center = size // 2
        for i in range(size):
            for j in range(size):
                psf[i, j] = np.exp(-((i - center)**2 + (j - center)**2) / (2 * sigma**2))
        psf = psf / np.sum(psf)
        return psf
    
    def add_speckle(self, image, factor=None):
        """Add multiplicative speckle noise (Rayleigh distribution)"""
        if factor is None:
            factor = self.speckle_factor
            
        # Generate Rayleigh distributed speckle
        speckle = np.random.rayleigh(1.0, image.shape)
        # Scale speckle to control noise level
        speckle = 1 + factor * (speckle - 1)
        
        # Apply multiplicative speckle
        noisy_image = image * speckle
        return noisy_image, speckle
    
    def add_gaussian_noise(self, image, sigma=None):
        """Add additive Gaussian noise"""
        if sigma is None:
            sigma = self.noise_sigma
        noise = np.random.normal(0, sigma, image.shape)
        return image + noise
    
    def simulate_sar(self, clean_image, add_blur=True, add_speckle=True, add_noise=True):
        """Simulate SAR degradation process"""
        image = clean_image.copy()
        
        # Step 1: Apply PSF blur
        if add_blur:
            psf = self.create_psf()
            image = ndimage.convolve(image, psf, mode='reflect')
        
        # Step 2: Add multiplicative speckle
        if add_speckle:
            image, speckle = self.add_speckle(image)
        
        # Step 3: Add additive Gaussian noise
        if add_noise:
            image = self.add_gaussian_noise(image)
        
        # Ensure non-negative values
        image = np.maximum(image, 0)
        
        return image


class SARDataset(Dataset):
    """Dataset for SAR image denoising"""
    
    def __init__(self, image_paths, patch_size=128, augment=True, simulator=None):
        self.image_paths = image_paths
        self.patch_size = patch_size
        self.augment = augment
        self.simulator = simulator or SARSimulator()
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        image_path = self.image_paths[idx]
        image = self.load_image(image_path)
        
        # Extract patches
        clean_patch, noisy_patch = self.extract_patch_pair(image)
        
        # Data augmentation
        if self.augment:
            clean_patch, noisy_patch = self.augment_pair(clean_patch, noisy_patch)
        
        # Convert to tensors (ensure contiguous arrays)
        clean_tensor = torch.from_numpy(clean_patch.copy()).float().unsqueeze(0)
        noisy_tensor = torch.from_numpy(noisy_patch.copy()).float().unsqueeze(0)
        
        return {
            'clean': clean_tensor,
            'noisy': noisy_tensor,
            'noise_level': torch.tensor(self.simulator.speckle_factor, dtype=torch.float32)
        }
    
    def load_image(self, path):
        """Load and preprocess image"""
        if isinstance(path, str):
            image = Image.open(path).convert('L')
        else:
            image = path
            
        # Convert to numpy array and normalize
        image = np.array(image, dtype=np.float32)
        image = image / 255.0  # Normalize to [0, 1]
        
        return image
    
    def extract_patch_pair(self, image):
        """Extract clean and noisy patch pair"""
        h, w = image.shape
        
        # Ensure image is large enough
        if h < self.patch_size or w < self.patch_size:
            image = self.pad_image(image)
            h, w = image.shape
        
        # Random crop
        top = np.random.randint(0, h - self.patch_size + 1)
        left = np.random.randint(0, w - self.patch_size + 1)
        
        clean_patch = image[top:top+self.patch_size, left:left+self.patch_size]
        
        # Generate noisy version
        noisy_patch = self.simulator.simulate_sar(clean_patch)
        
        return clean_patch, noisy_patch
    
    def pad_image(self, image):
        """Pad image to minimum size"""
        h, w = image.shape
        pad_h = max(0, self.patch_size - h)
        pad_w = max(0, self.patch_size - w)
        
        padded = np.pad(image, ((0, pad_h), (0, pad_w)), mode='reflect')
        return padded
    
    def augment_pair(self, clean, noisy):
        """Apply data augmentation to both clean and noisy patches"""
        # Make copies to avoid negative strides
        clean = clean.copy()
        noisy = noisy.copy()
        
        # Random horizontal flip
        if np.random.random() > 0.5:
            clean = np.fliplr(clean)
            noisy = np.fliplr(noisy)
        
        # Random vertical flip
        if np.random.random() > 0.5:
            clean = np.flipud(clean)
            noisy = np.flipud(noisy)
        
        # Random rotation (90, 180, 270 degrees)
        if np.random.random() > 0.5:
            k = np.random.randint(1, 4)
            clean = np.rot90(clean, k)
            noisy = np.rot90(noisy, k)
        
        return clean, noisy


def create_synthetic_dataset(output_dir, num_images=1000, image_size=256):
    """Create synthetic SAR dataset"""
    os.makedirs(output_dir, exist_ok=True)
    
    simulator = SARSimulator()
    
    for i in range(num_images):
        # Generate synthetic clean image
        clean_image = generate_synthetic_clean_image(image_size)
        
        # Save clean image
        clean_path = os.path.join(output_dir, f'clean_{i:04d}.png')
        Image.fromarray((clean_image * 255).astype(np.uint8)).save(clean_path)
        
        # Generate and save noisy image
        noisy_image = simulator.simulate_sar(clean_image)
        noisy_path = os.path.join(output_dir, f'noisy_{i:04d}.png')
        Image.fromarray((noisy_image * 255).astype(np.uint8)).save(noisy_path)
    
    print(f"Created {num_images} synthetic SAR image pairs in {output_dir}")


def generate_synthetic_clean_image(size):
    """Generate synthetic clean image with various structures"""
    # Create base image with different regions
    image = np.zeros((size, size))
    
    # Add geometric shapes
    center = size // 2
    
    # Circle
    y, x = np.ogrid[:size, :size]
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
    image = filters.gaussian(image, sigma=1.0)
    
    # Normalize to [0, 1]
    image = (image - image.min()) / (image.max() - image.min())
    
    return image


def prepare_dataset(data_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """Prepare dataset splits"""
    # Get all image paths
    image_paths = []
    for filename in os.listdir(data_dir):
        if filename.startswith('clean_') and filename.endswith('.png'):
            image_paths.append(os.path.join(data_dir, filename))
    
    # Shuffle and split
    np.random.shuffle(image_paths)
    
    n_total = len(image_paths)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    train_paths = image_paths[:n_train]
    val_paths = image_paths[n_train:n_train+n_val]
    test_paths = image_paths[n_train+n_val:]
    
    return train_paths, val_paths, test_paths


def create_dataloaders(train_paths, val_paths, test_paths, batch_size=16, patch_size=128, num_workers=4):
    """Create data loaders for training"""
    simulator = SARSimulator()
    
    train_dataset = SARDataset(train_paths, patch_size=patch_size, augment=True, simulator=simulator)
    val_dataset = SARDataset(val_paths, patch_size=patch_size, augment=False, simulator=simulator)
    test_dataset = SARDataset(test_paths, patch_size=patch_size, augment=False, simulator=simulator)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test SAR simulation
    simulator = SARSimulator()
    
    # Create synthetic clean image
    clean_image = generate_synthetic_clean_image(256)
    
    # Simulate SAR degradation
    noisy_image = simulator.simulate_sar(clean_image)
    
    # Visualize results
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(clean_image, cmap='gray')
    axes[0].set_title('Clean Image')
    axes[0].axis('off')
    
    axes[1].imshow(noisy_image, cmap='gray')
    axes[1].set_title('SAR-like Noisy Image')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig('sar_simulation_test.png')
    plt.show()
    
    print("SAR simulation test completed!")
