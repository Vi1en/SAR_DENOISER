#!/usr/bin/env python3
"""
Main training script for ADMM-PnP-DL SAR image denoising
"""
import argparse
import torch
import os
import sys
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.sar_simulation import create_synthetic_dataset, prepare_dataset, create_dataloaders
from trainers.train_denoiser import DenoiserTrainer
from trainers.train_unrolled import UnrolledTrainer, UnrolledADMM
from models.unet import create_model


def main():
    parser = argparse.ArgumentParser(description='Train ADMM-PnP-DL SAR denoising models')
    parser.add_argument('--mode', type=str, choices=['denoiser', 'unrolled', 'both'], 
                       default='both', help='Training mode')
    parser.add_argument('--data_dir', type=str, default='data/synthetic_sar',
                       help='Directory for synthetic SAR data')
    parser.add_argument('--num_images', type=int, default=1000,
                       help='Number of synthetic images to generate')
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
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    print(f"Training mode: {args.mode}")
    print(f"Model type: {args.model_type}")
    
    # Create synthetic dataset if it doesn't exist
    if not os.path.exists(args.data_dir):
        print(f"Creating synthetic SAR dataset with {args.num_images} images...")
        create_synthetic_dataset(args.data_dir, num_images=args.num_images, image_size=256)
        print("Dataset creation completed!")
    else:
        print(f"Using existing dataset at {args.data_dir}")
    
    # Prepare dataset
    train_paths, val_paths, test_paths = prepare_dataset(args.data_dir)
    train_loader, val_loader, test_loader = create_dataloaders(
        train_paths, val_paths, test_paths,
        batch_size=args.batch_size, patch_size=args.patch_size, num_workers=4
    )
    
    print(f"Dataset sizes - Train: {len(train_paths)}, Val: {len(val_paths)}, Test: {len(test_paths)}")
    
    # Train denoiser
    if args.mode in ['denoiser', 'both']:
        print("\n" + "="*60)
        print("TRAINING DENOISER")
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
        trainer.train(train_loader, val_loader, epochs=args.epochs, save_dir='checkpoints')
        
        print("Denoiser training completed!")
    
    # Train unrolled ADMM
    if args.mode in ['unrolled', 'both']:
        print("\n" + "="*60)
        print("TRAINING UNROLLED ADMM")
        print("="*60)
        
        # Create denoiser
        denoiser = create_model(args.model_type, n_channels=1, noise_conditioning=True)
        
        # Try to load trained denoiser
        if os.path.exists('checkpoints/best_model.pth'):
            checkpoint = torch.load('checkpoints/best_model.pth', map_location=device)
            denoiser.load_state_dict(checkpoint['model_state_dict'])
            print("Loaded trained denoiser for unrolled ADMM")
        else:
            print("Warning: No trained denoiser found, using random weights")
        
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
        
        # Train model (use smaller batch size for unrolled ADMM)
        unrolled_train_loader, _, _ = create_dataloaders(
            train_paths, val_paths, test_paths,
            batch_size=1, patch_size=args.patch_size, num_workers=4
        )
        
        unrolled_trainer.train(
            unrolled_train_loader, val_loader, 
            epochs=min(args.epochs, 50), save_dir='checkpoints_unrolled'
        )
        
        print("Unrolled ADMM training completed!")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"Checkpoints saved in:")
    if args.mode in ['denoiser', 'both']:
        print(f"  - checkpoints/ (denoiser)")
    if args.mode in ['unrolled', 'both']:
        print(f"  - checkpoints_unrolled/ (unrolled ADMM)")
    print(f"\nTo run the demo: streamlit run demo/app.py")


if __name__ == "__main__":
    main()


