#!/usr/bin/env python3
"""
Training script for ADMM-PnP-DL SAR image denoising using SAMPLE dataset
"""
import argparse
import torch
import os
import sys
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.sample_dataset_downloader import SAMPLEDatasetDownloader
from data.sample_dataset_loader import create_sample_dataloaders, get_sample_dataset_stats
from trainers.train_denoiser import DenoiserTrainer
from trainers.train_unrolled import UnrolledTrainer, UnrolledADMM
from models.unet import create_model


def download_sample_dataset(data_dir, force_download=False):
    """Download and organize SAMPLE dataset"""
    if os.path.exists(data_dir) and not force_download:
        print(f"✅ SAMPLE dataset already exists at {data_dir}")
        return True
    
    print("📥 Downloading and organizing SAMPLE SAR dataset...")
    downloader = SAMPLEDatasetDownloader(data_dir=data_dir)
    success = downloader.download_and_organize()
    
    if success:
        print("✅ SAMPLE dataset preparation completed!")
        return True
    else:
        print("❌ SAMPLE dataset preparation failed!")
        return False


def main():
    parser = argparse.ArgumentParser(description='Train ADMM-PnP-DL SAR denoising models on SAMPLE dataset')
    parser.add_argument('--mode', type=str, choices=['denoiser', 'unrolled', 'both'], 
                       default='both', help='Training mode')
    parser.add_argument('--data_dir', type=str, default='data/sample_sar/processed',
                       help='Directory for SAMPLE SAR data')
    parser.add_argument('--download_dir', type=str, default='data/sample_sar',
                       help='Directory to download SAMPLE dataset')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for training')
    parser.add_argument('--patch_size', type=int, default=128,
                       help='Patch size for training')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cuda/cpu/auto)')
    parser.add_argument('--model_type', type=str, choices=['unet', 'dncnn'], 
                       default='unet', help='Denoiser model type')
    parser.add_argument('--force_download', action='store_true',
                       help='Force re-download of dataset')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loader workers')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    print(f"Training mode: {args.mode}")
    print(f"Model type: {args.model_type}")
    
    # Download and organize SAMPLE dataset
    if not download_sample_dataset(args.download_dir, args.force_download):
        print("❌ Failed to prepare SAMPLE dataset!")
        return
    
    # Check if processed data exists
    if not os.path.exists(args.data_dir):
        print(f"❌ Processed SAMPLE dataset not found at {args.data_dir}")
        print("Please run the dataset downloader first:")
        print("python data/sample_dataset_downloader.py")
        return
    
    # Get dataset statistics
    print("\n📊 SAMPLE Dataset Statistics:")
    stats = get_sample_dataset_stats(args.data_dir)
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_sample_dataloaders(
        args.data_dir, 
        batch_size=args.batch_size, 
        patch_size=args.patch_size, 
        num_workers=args.num_workers
    )
    
    print(f"\n📈 Dataset Summary:")
    print(f"   Train: {len(train_loader.dataset)} samples, {len(train_loader)} batches")
    print(f"   Val: {len(val_loader.dataset)} samples, {len(val_loader)} batches")
    print(f"   Test: {len(test_loader.dataset)} samples, {len(test_loader)} batches")
    
    # Train denoiser
    if args.mode in ['denoiser', 'both']:
        print("\n" + "="*60)
        print("TRAINING DENOISER ON SAMPLE DATASET")
        print("="*60)
        
        # Create model
        model = create_model(args.model_type, n_channels=1, noise_conditioning=True)
        
        # Create trainer
        trainer = DenoiserTrainer(
            model, 
            device=device, 
            lr=args.lr, 
            l1_weight=1.0, 
            ssim_weight=0.1
        )
        
        # Train model
        save_dir = f'checkpoints_sample_{args.model_type}'
        trainer.train(train_loader, val_loader, epochs=args.epochs, save_dir=save_dir)
        
        print("✅ Denoiser training completed!")
    
    # Train unrolled ADMM
    if args.mode in ['unrolled', 'both']:
        print("\n" + "="*60)
        print("TRAINING UNROLLED ADMM ON SAMPLE DATASET")
        print("="*60)
        
        # Create denoiser
        denoiser = create_model(args.model_type, n_channels=1, noise_conditioning=True)
        
        # Try to load trained denoiser
        denoiser_checkpoint = f'checkpoints_sample_{args.model_type}/best_model.pth'
        if os.path.exists(denoiser_checkpoint):
            checkpoint = torch.load(denoiser_checkpoint, map_location=device)
            denoiser.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ Loaded trained denoiser from {denoiser_checkpoint}")
        else:
            print("⚠️ No trained denoiser found, using random weights")
        
        # Create unrolled ADMM model
        unrolled_model = UnrolledADMM(denoiser, num_iterations=5, device=device)
        
        # Create trainer
        unrolled_trainer = UnrolledTrainer(
            unrolled_model, 
            device=device, 
            lr=args.lr, 
            l1_weight=1.0, 
            ssim_weight=0.1
        )
        
        # Create data loaders for unrolled ADMM (batch size 1)
        unrolled_train_loader, _, _ = create_sample_dataloaders(
            args.data_dir, 
            batch_size=1, 
            patch_size=args.patch_size, 
            num_workers=args.num_workers
        )
        
        # Train model
        save_dir = f'checkpoints_unrolled_sample_{args.model_type}'
        unrolled_trainer.train(
            unrolled_train_loader, val_loader, 
            epochs=min(args.epochs, 50), save_dir=save_dir
        )
        
        print("✅ Unrolled ADMM training completed!")
    
    print("\n" + "="*60)
    print("SAMPLE DATASET TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"Checkpoints saved in:")
    if args.mode in ['denoiser', 'both']:
        print(f"  - checkpoints_sample_{args.model_type}/ (denoiser)")
    if args.mode in ['unrolled', 'both']:
        print(f"  - checkpoints_unrolled_sample_{args.model_type}/ (unrolled ADMM)")
    print(f"\nTo evaluate: python evaluate_sample.py")
    print(f"To run demo: streamlit run demo/app.py")


if __name__ == "__main__":
    main()


