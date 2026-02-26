"""
SAMPLE SAR Dataset Loader
Loads and processes the organized SAMPLE SAR dataset for training
"""
import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import random
from pathlib import Path
import matplotlib.pyplot as plt


class SAMPLESARDataset(Dataset):
    """Dataset class for SAMPLE SAR images"""
    
    def __init__(self, data_dir, split='train', patch_size=128, augment=True, noise_conditioning=True):
        self.data_dir = Path(data_dir)
        self.split = split
        self.patch_size = patch_size
        self.augment = augment
        self.noise_conditioning = noise_conditioning
        
        # Set up paths
        self.clean_dir = self.data_dir / f'{split}_patches' / 'clean'
        self.noisy_dir = self.data_dir / f'{split}_patches' / 'noisy'
        
        # Get all clean image files
        self.clean_files = list(self.clean_dir.glob('*.png'))
        self.clean_files.sort()  # Ensure consistent ordering
        
        # Verify dataset
        if not self.clean_dir.exists():
            raise FileNotFoundError(f"Clean images directory not found: {self.clean_dir}")
        if not self.noisy_dir.exists():
            raise FileNotFoundError(f"Noisy images directory not found: {self.noisy_dir}")
        
        print(f"📁 Loaded {len(self.clean_files)} {split} samples from {self.data_dir}")
    
    def __len__(self):
        return len(self.clean_files)
    
    def __getitem__(self, idx):
        # Get file paths
        clean_file = self.clean_files[idx]
        noisy_file = self.noisy_dir / clean_file.name
        
        # Load images
        clean_img = self.load_image(clean_file)
        noisy_img = self.load_image(noisy_file)
        
        # Ensure images are the same size
        if clean_img.shape != noisy_img.shape:
            # Resize to match
            min_h = min(clean_img.shape[0], noisy_img.shape[0])
            min_w = min(clean_img.shape[1], noisy_img.shape[1])
            clean_img = clean_img[:min_h, :min_w]
            noisy_img = noisy_img[:min_h, :min_w]
        
        # Extract patch if needed
        if clean_img.shape[0] > self.patch_size or clean_img.shape[1] > self.patch_size:
            clean_patch, noisy_patch = self.extract_patch(clean_img, noisy_img)
        else:
            clean_patch, noisy_patch = clean_img, noisy_img
        
        # Data augmentation
        if self.augment and self.split == 'train':
            clean_patch, noisy_patch = self.augment_pair(clean_patch, noisy_patch)
        
        # Convert to tensors (ensure contiguous arrays)
        clean_tensor = torch.from_numpy(clean_patch.copy()).float().unsqueeze(0)
        noisy_tensor = torch.from_numpy(noisy_patch.copy()).float().unsqueeze(0)
        
        # Calculate noise level (standard deviation of noise)
        noise_level = torch.tensor(np.std(noisy_patch - clean_patch), dtype=torch.float32)
        
        return {
            'clean': clean_tensor,
            'noisy': noisy_tensor,
            'noise_level': noise_level,
            'clean_path': str(clean_file),
            'noisy_path': str(noisy_file)
        }
    
    def load_image(self, path):
        """Load and preprocess image"""
        try:
            image = Image.open(path).convert('L')
            image_array = np.array(image, dtype=np.float32) / 255.0
            return image_array
        except Exception as e:
            print(f"Error loading image {path}: {e}")
            # Return a random image as fallback
            return np.random.rand(self.patch_size, self.patch_size)
    
    def extract_patch(self, clean_img, noisy_img):
        """Extract random patch from images"""
        h, w = clean_img.shape
        
        # Random crop
        if h > self.patch_size:
            top = random.randint(0, h - self.patch_size)
        else:
            top = 0
        
        if w > self.patch_size:
            left = random.randint(0, w - self.patch_size)
        else:
            left = 0
        
        # Extract patches
        clean_patch = clean_img[top:top+self.patch_size, left:left+self.patch_size]
        noisy_patch = noisy_img[top:top+self.patch_size, left:left+self.patch_size]
        
        return clean_patch, noisy_patch
    
    def augment_pair(self, clean, noisy):
        """Apply data augmentation to both clean and noisy patches"""
        # Make copies to avoid negative strides
        clean = clean.copy()
        noisy = noisy.copy()
        
        # Random horizontal flip
        if random.random() > 0.5:
            clean = np.fliplr(clean)
            noisy = np.fliplr(noisy)
        
        # Random vertical flip
        if random.random() > 0.5:
            clean = np.flipud(clean)
            noisy = np.flipud(noisy)
        
        # Random rotation (90, 180, 270 degrees)
        if random.random() > 0.5:
            k = random.randint(1, 3)
            clean = np.rot90(clean, k)
            noisy = np.rot90(noisy, k)
        
        return clean, noisy


def create_sample_dataloaders(data_dir, batch_size=16, patch_size=128, num_workers=4, augment=True):
    """Create data loaders for SAMPLE dataset"""
    print(f"📊 Creating SAMPLE SAR data loaders...")
    print(f"   Data directory: {data_dir}")
    print(f"   Batch size: {batch_size}")
    print(f"   Patch size: {patch_size}")
    
    # Create datasets
    train_dataset = SAMPLESARDataset(
        data_dir, split='train', patch_size=patch_size, augment=augment
    )
    val_dataset = SAMPLESARDataset(
        data_dir, split='val', patch_size=patch_size, augment=False
    )
    test_dataset = SAMPLESARDataset(
        data_dir, split='test', patch_size=patch_size, augment=False
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=True
    )
    
    print(f"✅ Data loaders created:")
    print(f"   Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"   Val: {len(val_dataset)} samples, {len(val_loader)} batches")
    print(f"   Test: {len(test_dataset)} samples, {len(test_loader)} batches")
    
    return train_loader, val_loader, test_loader


def visualize_sample_dataset(data_dir, num_samples=4):
    """Visualize SAMPLE dataset samples"""
    print("📊 Visualizing SAMPLE dataset...")
    
    # Create dataset
    dataset = SAMPLESARDataset(data_dir, split='train', patch_size=128, augment=False)
    
    # Get samples
    samples = []
    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]
        samples.append(sample)
    
    # Create visualization
    fig, axes = plt.subplots(2, num_samples, figsize=(4*num_samples, 8))
    
    for i, sample in enumerate(samples):
        clean_img = sample['clean'].squeeze().numpy()
        noisy_img = sample['noisy'].squeeze().numpy()
        noise_level = sample['noise_level'].item()
        
        # Plot clean image
        axes[0, i].imshow(clean_img, cmap='gray')
        axes[0, i].set_title(f'Clean {i+1}')
        axes[0, i].axis('off')
        
        # Plot noisy image
        axes[1, i].imshow(noisy_img, cmap='gray')
        axes[1, i].set_title(f'Noisy {i+1}\n(Noise: {noise_level:.3f})')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig('sample_dataset_visualization.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✅ SAMPLE dataset visualization saved!")


def get_sample_dataset_stats(data_dir):
    """Get SAMPLE dataset statistics"""
    print("📊 Computing SAMPLE dataset statistics...")
    
    stats = {}
    
    for split in ['train', 'val', 'test']:
        try:
            dataset = SAMPLESARDataset(data_dir, split=split, patch_size=128, augment=False)
            
            # Collect statistics
            clean_values = []
            noisy_values = []
            noise_levels = []
            
            for i in range(min(100, len(dataset))):  # Sample first 100 for efficiency
                sample = dataset[i]
                clean_values.append(sample['clean'].numpy().flatten())
                noisy_values.append(sample['noisy'].numpy().flatten())
                noise_levels.append(sample['noise_level'].item())
            
            # Concatenate all values
            clean_values = np.concatenate(clean_values)
            noisy_values = np.concatenate(noisy_values)
            
            stats[split] = {
                'num_samples': len(dataset),
                'clean_mean': np.mean(clean_values),
                'clean_std': np.std(clean_values),
                'noisy_mean': np.mean(noisy_values),
                'noisy_std': np.std(noisy_values),
                'noise_level_mean': np.mean(noise_levels),
                'noise_level_std': np.std(noise_levels)
            }
            
        except FileNotFoundError:
            stats[split] = {'num_samples': 0}
    
    # Print statistics
    print("\n📈 SAMPLE Dataset Statistics:")
    print("-" * 50)
    for split, stat in stats.items():
        if stat['num_samples'] > 0:
            print(f"{split.upper()}:")
            print(f"  Samples: {stat['num_samples']}")
            print(f"  Clean mean: {stat['clean_mean']:.4f} ± {stat['clean_std']:.4f}")
            print(f"  Noisy mean: {stat['noisy_mean']:.4f} ± {stat['noisy_std']:.4f}")
            print(f"  Noise level: {stat['noise_level_mean']:.4f} ± {stat['noise_level_std']:.4f}")
            print()
        else:
            print(f"{split.upper()}: No data found")
    
    return stats


def main():
    """Test SAMPLE dataset loader"""
    data_dir = 'data/sample_sar/processed'
    
    if not os.path.exists(data_dir):
        print(f"❌ SAMPLE dataset not found at {data_dir}")
        print("Please run the dataset downloader first:")
        print("python data/sample_dataset_downloader.py")
        return
    
    # Test dataset loading
    print("🧪 Testing SAMPLE dataset loader...")
    
    try:
        # Create data loaders
        train_loader, val_loader, test_loader = create_sample_dataloaders(
            data_dir, batch_size=4, patch_size=128, num_workers=0
        )
        
        # Test batch
        batch = next(iter(train_loader))
        print(f"✅ Batch test successful:")
        print(f"   Clean shape: {batch['clean'].shape}")
        print(f"   Noisy shape: {batch['noisy'].shape}")
        print(f"   Noise levels: {batch['noise_level']}")
        
        # Visualize dataset
        visualize_sample_dataset(data_dir)
        
        # Get statistics
        stats = get_sample_dataset_stats(data_dir)
        
        print("✅ SAMPLE dataset loader test completed!")
        
    except Exception as e:
        print(f"❌ SAMPLE dataset loader test failed: {e}")


if __name__ == "__main__":
    main()
