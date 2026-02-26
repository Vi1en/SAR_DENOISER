#!/usr/bin/env python3
"""
Evaluation script for ADMM-PnP-DL SAR image denoising
"""
import argparse
import torch
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.sar_simulation import prepare_dataset, create_dataloaders
from algos.evaluation import SARDenoisingEvaluator, run_comprehensive_evaluation
from models.unet import create_model
from algos.admm_pnp import ADMMPnP, TVDenoiser


def main():
    parser = argparse.ArgumentParser(description='Evaluate ADMM-PnP-DL SAR denoising models')
    parser.add_argument('--data_dir', type=str, default='data/synthetic_sar',
                       help='Directory for test data')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cuda/cpu/auto)')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for evaluation')
    parser.add_argument('--patch_size', type=int, default=128,
                       help='Patch size for evaluation')
    parser.add_argument('--save_dir', type=str, default='results',
                       help='Directory to save results')
    parser.add_argument('--methods', nargs='+', 
                       choices=['unet', 'dncnn', 'admm-pnp', 'unrolled', 'tv', 'all'],
                       default=['all'], help='Methods to evaluate')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    print(f"Evaluation methods: {args.methods}")
    
    # Check if data exists
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory {args.data_dir} not found!")
        print("Please run training first to create synthetic data.")
        return
    
    # Prepare test data
    train_paths, val_paths, test_paths = prepare_dataset(args.data_dir)
    _, _, test_loader = create_dataloaders(
        train_paths, val_paths, test_paths,
        batch_size=args.batch_size, patch_size=args.patch_size, num_workers=4
    )
    
    print(f"Test dataset size: {len(test_paths)} images")
    
    # Create evaluator
    evaluator = SARDenoisingEvaluator(device=device)
    
    # Load models
    models = {}
    
    if 'unet' in args.methods or 'all' in args.methods:
        print("Loading U-Net model...")
        unet_model = create_model('unet', n_channels=1, noise_conditioning=True)
        if os.path.exists('checkpoints/best_model.pth'):
            checkpoint = torch.load('checkpoints/best_model.pth', map_location=device)
            unet_model.load_state_dict(checkpoint['model_state_dict'])
            print("✅ Loaded trained U-Net model")
        else:
            print("⚠️ No trained U-Net model found, using random weights")
        models['U-Net Direct'] = unet_model
    
    if 'dncnn' in args.methods or 'all' in args.methods:
        print("Loading DnCNN model...")
        dncnn_model = create_model('dncnn', channels=1, noise_conditioning=True)
        if os.path.exists('checkpoints/best_model.pth'):
            checkpoint = torch.load('checkpoints/best_model.pth', map_location=device)
            dncnn_model.load_state_dict(checkpoint['model_state_dict'])
            print("✅ Loaded trained DnCNN model")
        else:
            print("⚠️ No trained DnCNN model found, using random weights")
        models['DnCNN Direct'] = dncnn_model
    
    if 'admm-pnp' in args.methods or 'all' in args.methods:
        print("Loading ADMM-PnP model...")
        if 'unet' in models:
            admm_model = models['U-Net Direct']
        else:
            admm_model = create_model('unet', n_channels=1, noise_conditioning=True)
            if os.path.exists('checkpoints/best_model.pth'):
                checkpoint = torch.load('checkpoints/best_model.pth', map_location=device)
                admm_model.load_state_dict(checkpoint['model_state_dict'])
        models['ADMM-PnP-DL'] = admm_model
    
    if 'unrolled' in args.methods or 'all' in args.methods:
        print("Loading Unrolled ADMM model...")
        from trainers.train_unrolled import UnrolledADMM
        denoiser = create_model('unet', n_channels=1, noise_conditioning=True)
        if os.path.exists('checkpoints/best_model.pth'):
            checkpoint = torch.load('checkpoints/best_model.pth', map_location=device)
            denoiser.load_state_dict(checkpoint['model_state_dict'])
        
        unrolled_model = UnrolledADMM(denoiser, num_iterations=5, device=device)
        if os.path.exists('checkpoints_unrolled/best_unrolled_model.pth'):
            checkpoint = torch.load('checkpoints_unrolled/best_unrolled_model.pth', map_location=device)
            unrolled_model.load_state_dict(checkpoint['model_state_dict'])
            print("✅ Loaded trained Unrolled ADMM model")
        else:
            print("⚠️ No trained Unrolled ADMM model found, using random weights")
        models['Unrolled ADMM'] = unrolled_model
    
    if 'tv' in args.methods or 'all' in args.methods:
        print("Loading TV denoising model...")
        tv_model = TVDenoiser(device=device)
        models['TV Denoising'] = tv_model
    
    # Run evaluation
    print("\n" + "="*60)
    print("RUNNING EVALUATION")
    print("="*60)
    
    for method_name, model in models.items():
        print(f"\nEvaluating {method_name}...")
        
        if method_name == 'ADMM-PnP-DL':
            # ADMM-PnP evaluation
            admm_params = {'max_iter': 20, 'rho_init': 1.0, 'alpha': 0.5, 'theta': 0.5}
            evaluator.evaluate_method(method_name, model, test_loader, admm_params)
        else:
            # Direct evaluation
            evaluator.evaluate_method(method_name, model, test_loader)
    
    # Compare results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    evaluator.compare_methods(evaluator.results)
    
    # Save results
    os.makedirs(args.save_dir, exist_ok=True)
    evaluator.plot_comparison(args.save_dir)
    evaluator.save_results(args.save_dir)
    
    print(f"\nResults saved to {args.save_dir}/")
    print("Evaluation completed!")


if __name__ == "__main__":
    main()


