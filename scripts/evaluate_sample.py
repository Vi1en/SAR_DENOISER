#!/usr/bin/env python3
"""
Evaluation script for ADMM-PnP-DL SAR denoising on SAMPLE layout or paired-folder OOD data.
"""
import argparse
import shutil
import torch
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from data.paired_folder_loader import create_paired_eval_dataloader
from data.sample_dataset_loader import create_sample_dataloaders, get_sample_dataset_stats
from algos.evaluation import SARDenoisingEvaluator, results_to_jsonable
from evaluators.run_logger import EvaluationRunContext
from models.unet import create_model
from algos.admm_pnp import TVDenoiser


def main():
    parser = argparse.ArgumentParser(description='Evaluate ADMM-PnP-DL SAR denoising models on SAMPLE dataset')
    parser.add_argument('--data_dir', type=str, default='data/sample_sar/processed',
                       help='SAMPLE processed root, or paired-folder root (see --data_layout)')
    parser.add_argument(
        '--data_layout',
        type=str,
        choices=['sample', 'paired'],
        default='sample',
        help='sample: train/val/test_patches; paired: data_dir/clean + data_dir/noisy (OOD)',
    )
    parser.add_argument(
        '--dataset_tag',
        type=str,
        default='',
        help='Optional label for run logs (e.g. ood_sen1)',
    )
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
    parser.add_argument(
        '--model_type',
        type=str,
        choices=['unet', 'dncnn', 'res_unet'],
        default='unet',
        help='Backbone for checkpoints_sample_<type> / sample unrolled paths',
    )
    parser.add_argument(
        '--no-run-log',
        action='store_true',
        help='Do not write results/runs/<run_id>/ manifest and metrics.json',
    )
    parser.add_argument(
        '--no-task-metrics',
        action='store_true',
        help='Skip structure/edge task metrics (faster; PSNR/SSIM/ENL only)',
    )

    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    print(f"Evaluation methods: {args.methods}")
    print(f"Model type: {args.model_type}")
    
    if not os.path.exists(args.data_dir):
        print(f"❌ Data directory not found: {args.data_dir}")
        if args.data_layout == "sample":
            print("For SAMPLE layout, run: python data/sample_dataset_downloader.py")
        else:
            print("For paired layout, create data_dir/clean and data_dir/noisy with matching filenames.")
        return

    if args.data_layout == "sample":
        print("\n📊 SAMPLE Dataset Statistics:")
        get_sample_dataset_stats(args.data_dir)
        train_loader, val_loader, test_loader = create_sample_dataloaders(
            args.data_dir,
            batch_size=args.batch_size,
            patch_size=args.patch_size,
            num_workers=4,
        )
    else:
        print("\n📊 Paired-folder (OOD) layout: clean/ + noisy/")
        test_loader = create_paired_eval_dataloader(
            args.data_dir,
            batch_size=args.batch_size,
            patch_size=args.patch_size,
            num_workers=4,
        )
    
    print("\n📈 Test Dataset Summary:")
    print(f"   Test samples: {len(test_loader.dataset)}")
    print(f"   Test batches: {len(test_loader)}")
    
    # Create evaluator
    evaluator = SARDenoisingEvaluator(device=device)
    
    # Load models (backbone matches --model_type for unet vs res_unet; dncnn uses its own branch)
    models = {}
    denoiser_arch = args.model_type if args.model_type in ("unet", "res_unet") else "unet"

    if 'unet' in args.methods or 'all' in args.methods:
        print(f"Loading direct denoiser ({denoiser_arch})...")
        unet_model = create_model(denoiser_arch, n_channels=1, noise_conditioning=True)
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
            admm_model = create_model(denoiser_arch, n_channels=1, noise_conditioning=True)
            checkpoint_path = f'checkpoints_sample_{args.model_type}/best_model.pth'
            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path, map_location=device)
                admm_model.load_state_dict(checkpoint['model_state_dict'])
        models['ADMM-PnP-DL'] = admm_model
    
    if 'unrolled' in args.methods or 'all' in args.methods:
        print("Loading Unrolled ADMM model...")
        from trainers.train_unrolled import UnrolledADMM
        denoiser = create_model(denoiser_arch, n_channels=1, noise_conditioning=True)
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
            evaluator.evaluate_method(
                method_name,
                model,
                test_loader,
                admm_params,
                include_task_metrics=not args.no_task_metrics,
            )
        else:
            # Direct evaluation
            evaluator.evaluate_method(
                method_name,
                model,
                test_loader,
                include_task_metrics=not args.no_task_metrics,
            )
    
    # Compare results
    tag = (args.dataset_tag or args.data_layout).upper()
    print("\n" + "="*60)
    print(f"EVALUATION RESULTS ({tag})")
    print("="*60)
    
    evaluator.compare_methods(evaluator.results)
    
    # Save results
    os.makedirs(args.save_dir, exist_ok=True)
    evaluator.plot_comparison(args.save_dir)
    evaluator.save_results(args.save_dir)
    
    # Create SAMPLE-specific visualization
    create_sample_visualization(
        evaluator.results, args.save_dir, dataset_label=args.dataset_tag or args.data_layout
    )

    if not args.no_run_log:
        primary_ckpt = f'checkpoints_sample_{args.model_type}/best_model.pth'
        unrolled_ckpt = f'checkpoints_unrolled_sample_{args.model_type}/best_unrolled_model.pth'
        ctx = EvaluationRunContext()
        ctx.write_manifest({
            "script": "scripts/evaluate_sample.py",
            "argv": sys.argv,
            "data_dir": args.data_dir,
            "data_layout": args.data_layout,
            "dataset_tag": args.dataset_tag or None,
            "model_type": args.model_type,
            "methods": list(args.methods),
            "device": str(device),
            "save_dir": args.save_dir,
            "batch_size": args.batch_size,
            "patch_size": args.patch_size,
            "task_metrics_enabled": not args.no_task_metrics,
            "checkpoint_sample": primary_ckpt if os.path.exists(primary_ckpt) else None,
            "checkpoint_unrolled": unrolled_ckpt if os.path.exists(unrolled_ckpt) else None,
        })
        # Summary metrics only (per-patch lists remain in save_dir/evaluation_results.json)
        ctx.write_metrics(
            results_to_jsonable(evaluator.results, include_per_patch_lists=False)
        )
        for name in (
            "method_comparison.png",
            "metric_distributions.png",
            "task_metric_comparison.png",
            "sample_dataset_results.png",
        ):
            src = os.path.join(args.save_dir, name)
            if os.path.isfile(src):
                shutil.copy2(src, ctx.plots_dir() / name)
        print(f"Run log saved to {ctx.run_dir}/")

    print(f"\nResults saved to {args.save_dir}/")
    print("Evaluation completed!")


def create_sample_visualization(results, save_dir, dataset_label: str = "sample"):
    """Bar charts for method comparison (filename unchanged for downstream tools)."""
    print(f"📊 Creating evaluation visualization ({dataset_label})...")
    
    # Create comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # PSNR comparison
    methods = list(results.keys())
    psnr_means = [results[method]['psnr_mean'] for method in methods]
    psnr_stds = [results[method]['psnr_std'] for method in methods]
    
    axes[0, 0].bar(methods, psnr_means, yerr=psnr_stds, capsize=5, alpha=0.7)
    axes[0, 0].set_title(f'PSNR ({dataset_label})')
    axes[0, 0].set_ylabel('PSNR (dB)')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # SSIM comparison
    ssim_means = [results[method]['ssim_mean'] for method in methods]
    ssim_stds = [results[method]['ssim_std'] for method in methods]
    
    axes[0, 1].bar(methods, ssim_means, yerr=ssim_stds, capsize=5, alpha=0.7)
    axes[0, 1].set_title(f'SSIM ({dataset_label})')
    axes[0, 1].set_ylabel('SSIM')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # ENL comparison
    enl_means = [results[method]['enl_mean'] for method in methods]
    enl_stds = [results[method]['enl_std'] for method in methods]
    
    axes[1, 0].bar(methods, enl_means, yerr=enl_stds, capsize=5, alpha=0.7)
    axes[1, 0].set_title(f'ENL ({dataset_label})')
    axes[1, 0].set_ylabel('ENL')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Performance summary
    axes[1, 1].text(0.1, 0.8, f'Results ({dataset_label})', fontsize=14, fontweight='bold')
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


