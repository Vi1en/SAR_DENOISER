#!/usr/bin/env python3
"""
Main script to download and organize SAMPLE SAR dataset
"""
import os
import sys
import argparse
import subprocess
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.sample_dataset_downloader import SAMPLEDatasetDownloader
from data.sample_dataset_loader import create_sample_dataloaders, get_sample_dataset_stats, visualize_sample_dataset


def main():
    parser = argparse.ArgumentParser(description='Download and organize SAMPLE SAR dataset')
    parser.add_argument('--data_dir', type=str, default='data/sample_sar',
                       help='Directory to store SAMPLE dataset')
    parser.add_argument('--force_download', action='store_true',
                       help='Force re-download of dataset')
    parser.add_argument('--patch_size', type=int, default=128,
                       help='Patch size for training')
    parser.add_argument('--overlap', type=float, default=0.5,
                       help='Overlap ratio for patch extraction')
    parser.add_argument('--visualize', action='store_true',
                       help='Create dataset visualization')
    parser.add_argument('--test_loader', action='store_true',
                       help='Test data loader after organization')
    
    args = parser.parse_args()
    
    print("🚀 SAMPLE SAR Dataset Downloader and Organizer")
    print("=" * 60)
    print(f"Data directory: {args.data_dir}")
    print(f"Patch size: {args.patch_size}")
    print(f"Overlap: {args.overlap}")
    
    # Step 1: Download and organize dataset
    print("\n📥 Step 1: Downloading and organizing SAMPLE dataset...")
    downloader = SAMPLEDatasetDownloader(data_dir=args.data_dir)
    
    if not downloader.download_and_organize():
        print("❌ Dataset download and organization failed!")
        return False
    
    # Step 2: Test data loader
    if args.test_loader:
        print("\n🧪 Step 2: Testing data loader...")
        processed_dir = Path(args.data_dir) / 'processed'
        
        if not processed_dir.exists():
            print(f"❌ Processed dataset not found at {processed_dir}")
            return False
        
        try:
            # Test data loader
            train_loader, val_loader, test_loader = create_sample_dataloaders(
                str(processed_dir), 
                batch_size=4, 
                patch_size=args.patch_size, 
                num_workers=0
            )
            
            # Test batch
            batch = next(iter(train_loader))
            print(f"✅ Data loader test successful:")
            print(f"   Clean shape: {batch['clean'].shape}")
            print(f"   Noisy shape: {batch['noisy'].shape}")
            print(f"   Noise levels: {batch['noise_level']}")
            
            # Get statistics
            stats = get_sample_dataset_stats(str(processed_dir))
            
        except Exception as e:
            print(f"❌ Data loader test failed: {e}")
            return False
    
    # Step 3: Create visualization
    if args.visualize:
        print("\n📊 Step 3: Creating dataset visualization...")
        processed_dir = Path(args.data_dir) / 'processed'
        
        if processed_dir.exists():
            try:
                visualize_sample_dataset(str(processed_dir))
                print("✅ Dataset visualization created!")
            except Exception as e:
                print(f"⚠️ Visualization failed: {e}")
        else:
            print("⚠️ Processed dataset not found, skipping visualization")
    
    print("\n" + "=" * 60)
    print("🎉 SAMPLE SAR Dataset preparation completed!")
    print("=" * 60)
    print(f"📁 Dataset location: {args.data_dir}")
    print(f"📊 Processed data: {args.data_dir}/processed")
    print(f"🖼️ Raw data: {args.data_dir}/raw")
    
    print("\nNext steps:")
    print("1. Train models: python train_sample.py")
    print("2. Evaluate models: python evaluate_sample.py")
    print("3. Run demo: streamlit run demo/app.py")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)


