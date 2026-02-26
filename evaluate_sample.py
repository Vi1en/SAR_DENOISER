#!/usr/bin/env python3
"""
Evaluation script for ADMM-PnP-DL SAR image denoising on SAMPLE dataset
"""
import argparse
import torch
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.sample_dataset_loader import create_sample_dataloaders, get_sample_dataset_stats
from algos.evaluation import SARDenoisingEvaluator
from models.unet import create_model
from algos.admm_pnp import ADMMPnP, TVDenoiser


def main():
    parser = argparse.ArgumentParser(description='Evaluate ADMM-PnP-DL SAR denoising models on SAMPLE dataset')
    parser.add_argument('--data_dir', type=str, default='data/sample_sar/processed',
                       help='Directory for SAMPLE SAR data')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cuda/cpu/auto)')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for evaluation')
    parser.add_argument('--patch_size', type=int, default=128,
                       help='Patch size for evaluation')
    parser.add_argument('--save_dir', type=str, default='results_sample',
                       help='Directory to save results')
    parser.add_argument('--methods', nargs='+', 
                       choices=['unet', 'dncnn', 'admm-pnp', 'unrolled', 'tv', 'all'],
                       default=['all'], help='Methods to evaluate')
    parser.add_argument('--model_type', type=str, choices=['unet', 'dncnn'], 
                       default='unet', help='Model type to evaluate')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    print(f"Evaluation methods: {args.methods}")
    print(f"Model type: {args.model_type}")
    
    # Check if data exists
    if not os.path.exists(args.data_dir):
        print(f"❌ SAMPLE dataset not found at {args.data_dir}")
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
        num_workers=4
    )
    
    print(f"\n📈 Test Dataset Summary:")
    print(f"   Test samples: {len(test_loader.dataset)}")
    print(f"   Test batches: {len(test_loader)}")
    
    # Create evaluator
    evaluator = SARDenoisingEvaluator(device=device)
    
    # Load models
    models = {}
    
    if 'unet' in args.methods or 'all' in args.methods:
        print("Loading U-Net model...")
        unet_model = create_model('unet', n_channels=1, noise_conditioning=True)
        checkpoint_path = f'checkpoints_sample_{args.model_type}/best_model.pth'
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            unet_model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ Loaded trained U-Net model from {checkpoint_path}")
        else:
            print("⚠️ No trained U-Net model found, using random weights")
        models['U-Net Direct'] = unet_model
    
    if 'dncnn' in args.methods or 'all' in args.methods:
        print("Loading DnCNN model...")
        dncnn_model = create_model('dncnn', channels=1, noise_conditioning=True)
        checkpoint_path = f'checkpoints_sample_{args.model_type}/best_model.pth'
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            dncnn_model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ Loaded trained DnCNN model from {checkpoint_path}")
        else:
            print("⚠️ No trained DnCNN model found, using random weights")
        models['DnCNN Direct'] = dncnn_model
    
    if 'admm-pnp' in args.methods or 'all' in args.methods:
        print("Loading ADMM-PnP model...")
        if 'U-Net Direct' in models:
            admm_model = models['U-Net Direct']
        else:
            admm_model = create_model('unet', n_channels=1, noise_conditioning=True)
            checkpoint_path = f'checkpoints_sample_{args.model_type}/best_model.pth'
            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path, map_location=device)
                admm_model.load_state_dict(checkpoint['model_state_dict'])
        models['ADMM-PnP-DL'] = admm_model
    
    if 'unrolled' in args.methods or 'all' in args.methods:
        print("Loading Unrolled ADMM model...")
        from trainers.train_unrolled import UnrolledADMM
        denoiser = create_model('unet', n_channels=1, noise_conditioning=True)
        checkpoint_path = f'checkpoints_sample_{args.model_type}/best_model.pth'
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            denoiser.load_state_dict(checkpoint['model_state_dict'])
        
        unrolled_model = UnrolledADMM(denoiser, num_iterations=5, device=device)
        unrolled_checkpoint_path = f'checkpoints_unrolled_sample_{args.model_type}/best_unrolled_model.pth'
        if os.path.exists(unrolled_checkpoint_path):
            checkpoint = torch.load(unrolled_checkpoint_path, map_location=device)
            unrolled_model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ Loaded trained Unrolled ADMM model from {unrolled_checkpoint_path}")
        else:
            print("⚠️ No trained Unrolled ADMM model found, using random weights")
        models['Unrolled ADMM'] = unrolled_model
    
    if 'tv' in args.methods or 'all' in args.methods:
        print("Loading TV denoising model...")
        tv_model = TVDenoiser(device=device)
        models['TV Denoising'] = tv_model
    
    # Run evaluation
    print("\n" + "="*60)
    print("RUNNING EVALUATION ON SAMPLE DATASET")
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
    print("SAMPLE DATASET EVALUATION RESULTS")
    print("="*60)
    
    evaluator.compare_methods(evaluator.results)
    
    # Save results
    os.makedirs(args.save_dir, exist_ok=True)
    evaluator.plot_comparison(args.save_dir)
    evaluator.save_results(args.save_dir)
    
    # Create SAMPLE-specific visualization
    create_sample_visualization(evaluator.results, args.save_dir)
    
    print(f"\nResults saved to {args.save_dir}/")
    print("SAMPLE dataset evaluation completed!")


def create_sample_visualization(results, save_dir):
    """Create SAMPLE-specific visualization"""
    print("📊 Creating SAMPLE dataset visualization...")
    
    # Create comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # PSNR comparison
    methods = list(results.keys())
    psnr_means = [results[method]['psnr_mean'] for method in methods]
    psnr_stds = [results[method]['psnr_std'] for method in methods]
    
    axes[0, 0].bar(methods, psnr_means, yerr=psnr_stds, capsize=5, alpha=0.7)
    axes[0, 0].set_title('PSNR Comparison on SAMPLE Dataset')
    axes[0, 0].set_ylabel('PSNR (dB)')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # SSIM comparison
    ssim_means = [results[method]['ssim_mean'] for method in methods]
    ssim_stds = [results[method]['ssim_std'] for method in methods]
    
    axes[0, 1].bar(methods, ssim_means, yerr=ssim_stds, capsize=5, alpha=0.7)
    axes[0, 1].set_title('SSIM Comparison on SAMPLE Dataset')
    axes[0, 1].set_ylabel('SSIM')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # ENL comparison
    enl_means = [results[method]['enl_mean'] for method in methods]
    enl_stds = [results[method]['enl_std'] for method in methods]
    
    axes[1, 0].bar(methods, enl_means, yerr=enl_stds, capsize=5, alpha=0.7)
    axes[1, 0].set_title('ENL Comparison on SAMPLE Dataset')
    axes[1, 0].set_ylabel('ENL')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Performance summary
    axes[1, 1].text(0.1, 0.8, 'SAMPLE Dataset Results', fontsize=14, fontweight='bold')
    axes[1, 1].text(0.1, 0.7, f'Best PSNR: {max(psnr_means):.2f} dB', fontsize=12)
    axes[1, 1].text(0.1, 0.6, f'Best SSIM: {max(ssim_means):.4f}', fontsize=12)
    axes[1, 1].text(0.1, 0.5, f'Best ENL: {max(enl_means):.2f}', fontsize=12)
    axes[1, 1].text(0.1, 0.4, f'Methods: {len(methods)}', fontsize=12)
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'sample_dataset_results.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ SAMPLE dataset visualization saved!")


if __name__ == "__main__":
    main()


