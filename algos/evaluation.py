"""
Evaluation utilities for SAR image denoising
"""
import torch
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import matplotlib.pyplot as plt
from tqdm import tqdm
import os


def calculate_psnr(clean, denoised):
    """Calculate Peak Signal-to-Noise Ratio"""
    if isinstance(clean, torch.Tensor):
        clean = clean.cpu().numpy()
    if isinstance(denoised, torch.Tensor):
        denoised = denoised.cpu().numpy()
    
    # Ensure images are in [0, 1] range
    clean = np.clip(clean, 0, 1)
    denoised = np.clip(denoised, 0, 1)
    
    return peak_signal_noise_ratio(clean, denoised, data_range=1.0)


def calculate_ssim(clean, denoised):
    """Calculate Structural Similarity Index"""
    if isinstance(clean, torch.Tensor):
        clean = clean.cpu().numpy()
    if isinstance(denoised, torch.Tensor):
        denoised = denoised.cpu().numpy()
    
    # Ensure images are in [0, 1] range
    clean = np.clip(clean, 0, 1)
    denoised = np.clip(denoised, 0, 1)
    
    return structural_similarity(clean, denoised, data_range=1.0)


def calculate_enl(denoised):
    """Calculate Equivalent Number of Looks (ENL) for SAR images"""
    if isinstance(denoised, torch.Tensor):
        denoised = denoised.cpu().numpy()
    
    # ENL = (mean^2) / variance
    mean_val = np.mean(denoised)
    var_val = np.var(denoised)
    
    if var_val == 0:
        return float('inf')
    
    enl = (mean_val ** 2) / var_val
    return enl


def calculate_metrics(clean, denoised):
    """Calculate all evaluation metrics"""
    psnr = calculate_psnr(clean, denoised)
    ssim = calculate_ssim(clean, denoised)
    enl = calculate_enl(denoised)
    
    return {
        'psnr': psnr,
        'ssim': ssim,
        'enl': enl
    }


def results_to_jsonable(results: dict, include_per_patch_lists: bool = True) -> dict:
    """
    Convert evaluator ``results`` (numpy scalars, lists, inf) to JSON-safe Python types.
    If ``include_per_patch_lists`` is False, drops ``*_values`` keys to keep files small.
    """
    import math

    def convert(x):
        if isinstance(x, (np.floating, float)):
            v = float(x)
            if math.isnan(v) or math.isinf(v):
                return None
            return v
        if isinstance(x, (np.integer, int)):
            return int(x)
        if isinstance(x, np.ndarray):
            return [convert(i) for i in x.tolist()]
        if isinstance(x, dict):
            return {k: convert(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [convert(i) for i in x]
        return x

    out = convert(results)
    if not include_per_patch_lists and isinstance(out, dict):
        trimmed = {}
        for method, payload in out.items():
            if not isinstance(payload, dict):
                trimmed[method] = payload
                continue
            trimmed[method] = {
                k: v
                for k, v in payload.items()
                if not k.endswith("_values")
            }
        return trimmed
    return out


class SARDenoisingEvaluator:
    """Comprehensive evaluator for SAR denoising methods"""
    
    def __init__(self, device='cuda'):
        self.device = device
        self.results = {}
    
    def evaluate_method(
        self,
        method_name,
        denoiser,
        test_loader,
        admm_params=None,
        *,
        include_task_metrics: bool = True,
    ):
        """Evaluate a denoising method on test dataset.

        When ``include_task_metrics`` is True, also computes structure/edge proxies
        from ``evaluators.task_metrics`` (gradient correlation, EPI, Laplacian MSE, grad SSIM).
        """
        print(f"Evaluating {method_name}...")
        
        psnr_values = []
        ssim_values = []
        enl_values = []
        gsm_corr_values = []
        epi_values = []
        laplacian_mse_values = []
        grad_ssim_values = []

        if hasattr(denoiser, 'eval'):
            denoiser.eval()

        if include_task_metrics:
            from evaluators.task_metrics import compute_task_metrics
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Evaluating {method_name}"):
                clean = batch['clean'].to(self.device)
                noisy = batch['noisy'].to(self.device)
                noise_level = batch['noise_level'].to(self.device)
                
                # Denoise
                if method_name == 'ADMM-PnP-DL':
                    # Use ADMM-PnP with deep learning denoiser
                    from algos.admm_pnp import ADMMPnP
                    admm = ADMMPnP(denoiser, device=self.device, **(admm_params or {}))
                    result = admm.denoise(noisy.squeeze().cpu().numpy(), noise_level=noise_level.item())
                    denoised = torch.from_numpy(result['denoised']).float()
                elif method_name == 'TV Denoising':
                    with torch.enable_grad():
                        denoised = denoiser.tv_denoise(noisy)
                else:
                    # Direct denoising
                    if hasattr(denoiser, 'noise_conditioning') and denoiser.noise_conditioning:
                        denoised = denoiser(noisy, noise_level)
                    else:
                        denoised = denoiser(noisy)
                
                # Calculate metrics
                clean_np = clean.squeeze().cpu().numpy()
                denoised_np = denoised.squeeze().cpu().numpy()
                
                metrics = calculate_metrics(clean_np, denoised_np)
                psnr_values.append(metrics['psnr'])
                ssim_values.append(metrics['ssim'])
                enl_values.append(metrics['enl'])

                if include_task_metrics:
                    tm = compute_task_metrics(clean_np, denoised_np, noisy=noisy.squeeze().cpu().numpy())
                    gsm_corr_values.append(tm["gsm_corr"])
                    epi_values.append(tm["epi"])
                    laplacian_mse_values.append(tm["laplacian_mse"])
                    grad_ssim_values.append(tm["grad_ssim"])
        
        # Calculate statistics
        row = {
            'psnr_mean': np.mean(psnr_values),
            'psnr_std': np.std(psnr_values),
            'ssim_mean': np.mean(ssim_values),
            'ssim_std': np.std(ssim_values),
            'enl_mean': np.mean(enl_values),
            'enl_std': np.std(enl_values),
            'psnr_values': psnr_values,
            'ssim_values': ssim_values,
            'enl_values': enl_values
        }
        if include_task_metrics:
            row.update({
                'gsm_corr_mean': np.mean(gsm_corr_values),
                'gsm_corr_std': np.std(gsm_corr_values),
                'epi_mean': np.mean(epi_values),
                'epi_std': np.std(epi_values),
                'laplacian_mse_mean': np.mean(laplacian_mse_values),
                'laplacian_mse_std': np.std(laplacian_mse_values),
                'grad_ssim_mean': np.mean(grad_ssim_values),
                'grad_ssim_std': np.std(grad_ssim_values),
                'gsm_corr_values': gsm_corr_values,
                'epi_values': epi_values,
                'laplacian_mse_values': laplacian_mse_values,
                'grad_ssim_values': grad_ssim_values,
            })
        self.results[method_name] = row
        
        print(f"{method_name} Results:")
        print(f"  PSNR: {np.mean(psnr_values):.2f} ± {np.std(psnr_values):.2f}")
        print(f"  SSIM: {np.mean(ssim_values):.4f} ± {np.std(ssim_values):.4f}")
        print(f"  ENL: {np.mean(enl_values):.2f} ± {np.std(enl_values):.2f}")
        if include_task_metrics:
            print(f"  GSM corr: {row['gsm_corr_mean']:.4f} ± {row['gsm_corr_std']:.4f}")
            print(f"  EPI: {row['epi_mean']:.4f} ± {row['epi_std']:.4f}")
            print(f"  Laplacian MSE: {row['laplacian_mse_mean']:.6f} ± {row['laplacian_mse_std']:.6f}")
            print(f"  Grad SSIM: {row['grad_ssim_mean']:.4f} ± {row['grad_ssim_std']:.4f}")
    
    def compare_methods(self, methods_results):
        """Compare multiple denoising methods"""
        print("\n" + "="*60)
        print("METHOD COMPARISON RESULTS")
        print("="*60)
        
        # Create comparison table
        methods = list(methods_results.keys())

        print(f"{'Method':<20} {'PSNR':<10} {'SSIM':<10} {'ENL':<10}")
        print("-" * 60)
        
        for method in methods:
            result = methods_results[method]
            psnr = f"{result['psnr_mean']:.2f}±{result['psnr_std']:.2f}"
            ssim = f"{result['ssim_mean']:.4f}±{result['ssim_std']:.4f}"
            enl = f"{result['enl_mean']:.2f}±{result['enl_std']:.2f}"
            print(f"{method:<20} {psnr:<10} {ssim:<10} {enl:<10}")
        if methods and 'gsm_corr_mean' in methods_results[methods[0]]:
            print("-" * 60)
            print(f"{'Method':<20} {'GSM-corr':<12} {'EPI':<12} {'Grad-SSIM':<12}")
            print("-" * 60)
            for method in methods:
                r = methods_results[method]
                g = f"{r['gsm_corr_mean']:.4f}±{r['gsm_corr_std']:.4f}"
                e = f"{r['epi_mean']:.4f}±{r['epi_std']:.4f}"
                gs = f"{r['grad_ssim_mean']:.4f}±{r['grad_ssim_std']:.4f}"
                print(f"{method:<20} {g:<12} {e:<12} {gs:<12}")
    
    def plot_comparison(self, save_dir='results'):
        """Plot comparison results"""
        os.makedirs(save_dir, exist_ok=True)
        
        methods = list(self.results.keys())
        metrics = ['psnr', 'ssim', 'enl']
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            
            # Extract data
            means = [self.results[method][f'{metric}_mean'] for method in methods]
            stds = [self.results[method][f'{metric}_std'] for method in methods]
            
            # Create bar plot
            x_pos = np.arange(len(methods))
            bars = ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7)
            
            # Customize plot
            ax.set_xlabel('Method')
            ax.set_ylabel(metric.upper())
            ax.set_title(f'{metric.upper()} Comparison')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(methods, rotation=45, ha='right')
            ax.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, mean, std in zip(bars, means, stds):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
                       f'{mean:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'method_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot distribution of metrics
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            
            for method in methods:
                values = self.results[method][f'{metric}_values']
                ax.hist(values, alpha=0.7, label=method, bins=20)
            
            ax.set_xlabel(metric.upper())
            ax.set_ylabel('Frequency')
            ax.set_title(f'{metric.upper()} Distribution')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'metric_distributions.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # Task / structure metrics (optional second figure)
        if methods and 'gsm_corr_mean' in self.results[methods[0]]:
            task_keys = [
                ('gsm_corr', 'GSM correlation'),
                ('epi', 'Edge preservation index'),
                ('grad_ssim', 'Grad SSIM'),
            ]
            fig, axes = plt.subplots(1, len(task_keys), figsize=(5 * len(task_keys), 5))
            if len(task_keys) == 1:
                axes = [axes]
            for ax, (key, label) in zip(axes, task_keys):
                means = [self.results[m][f'{key}_mean'] for m in methods]
                stds = [self.results[m][f'{key}_std'] for m in methods]
                x_pos = np.arange(len(methods))
                ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7)
                ax.set_xlabel('Method')
                ax.set_ylabel(label)
                ax.set_title(label)
                ax.set_xticks(x_pos)
                ax.set_xticklabels(methods, rotation=45, ha='right')
                ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'task_metric_comparison.png'), dpi=300, bbox_inches='tight')
            plt.close()
    
    def save_results(self, save_dir='results'):
        """Save evaluation results to file"""
        os.makedirs(save_dir, exist_ok=True)
        
        # Save detailed results
        import json
        with open(os.path.join(save_dir, 'evaluation_results.json'), 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            results_serializable = {}
            for method, result in self.results.items():
                results_serializable[method] = {}
                for key, value in result.items():
                    if isinstance(value, np.ndarray):
                        results_serializable[method][key] = value.tolist()
                    else:
                        results_serializable[method][key] = value
            json.dump(results_serializable, f, indent=2)
        
        print(f"Results saved to {save_dir}/evaluation_results.json")


def run_comprehensive_evaluation(test_loader, device='cuda'):
    """Run comprehensive evaluation of all methods"""
    from models.unet import create_model
    from algos.admm_pnp import TVDenoiser
    
    evaluator = SARDenoisingEvaluator(device=device)
    
    # Load trained models
    print("Loading trained models...")
    
    # Load U-Net denoiser
    unet_model = create_model('unet', n_channels=1, noise_conditioning=True)
    if os.path.exists('checkpoints/best_model.pth'):
        checkpoint = torch.load('checkpoints/best_model.pth', map_location=device)
        unet_model.load_state_dict(checkpoint['model_state_dict'])
        print("Loaded trained U-Net model")
    else:
        print("Warning: No trained U-Net model found, using random weights")
    
    # Load DnCNN denoiser
    dncnn_model = create_model('dncnn', channels=1, noise_conditioning=True)
    if os.path.exists('checkpoints/best_model.pth'):
        checkpoint = torch.load('checkpoints/best_model.pth', map_location=device)
        dncnn_model.load_state_dict(checkpoint['model_state_dict'])
        print("Loaded trained DnCNN model")
    else:
        print("Warning: No trained DnCNN model found, using random weights")
    
    # Load unrolled ADMM model
    from trainers.train_unrolled import UnrolledADMM
    unrolled_model = UnrolledADMM(unet_model, num_iterations=5, device=device)
    if os.path.exists('checkpoints_unrolled/best_unrolled_model.pth'):
        checkpoint = torch.load('checkpoints_unrolled/best_unrolled_model.pth', map_location=device)
        unrolled_model.load_state_dict(checkpoint['model_state_dict'])
        print("Loaded trained unrolled ADMM model")
    else:
        print("Warning: No trained unrolled ADMM model found, using random weights")
    
    # Evaluate methods
    print("\nEvaluating methods...")
    
    # 1. Direct U-Net denoising
    evaluator.evaluate_method('U-Net Direct', unet_model, test_loader)
    
    # 2. Direct DnCNN denoising
    evaluator.evaluate_method('DnCNN Direct', dncnn_model, test_loader)
    
    # 3. ADMM-PnP with U-Net
    admm_params = {'max_iter': 20, 'rho_init': 1.0, 'alpha': 0.5, 'theta': 0.5}
    evaluator.evaluate_method('ADMM-PnP-DL', unet_model, test_loader, admm_params)
    
    # 4. Unrolled ADMM
    evaluator.evaluate_method('Unrolled ADMM', unrolled_model, test_loader)
    
    # 5. TV denoising baseline
    tv_denoiser = TVDenoiser(device=device)
    evaluator.evaluate_method('TV Denoising', tv_denoiser, test_loader)
    
    # Compare and save results
    evaluator.compare_methods(evaluator.results)
    evaluator.plot_comparison()
    evaluator.save_results()
    
    return evaluator.results


if __name__ == "__main__":
    # Test evaluation functions
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create test data
    clean = np.random.rand(128, 128)
    noisy = clean + 0.1 * np.random.randn(128, 128)
    
    # Test metrics
    metrics = calculate_metrics(clean, noisy)
    print("Test metrics:", metrics)
    
    print("Evaluation utilities test completed!")


