"""
ADMM-PnP algorithm for SAR image denoising with deep learning denoiser
"""
import torch
import torch.nn.functional as F
import numpy as np
from scipy.fft import fft2, ifft2, fftshift, ifftshift
import matplotlib.pyplot as plt
from tqdm import tqdm


class ADMMPnP:
    """ADMM-PnP algorithm with deep learning denoiser"""
    
    def __init__(self, denoiser, device='cuda', rho_init=1.0, alpha=0.3, theta=0.5, 
                 adaptive_rho=True, max_iter=30, tol=1e-4, use_log_transform=True):
        self.denoiser = denoiser
        self.device = device
        self.rho_init = rho_init
        self.alpha = alpha
        self.theta = theta
        self.adaptive_rho = adaptive_rho
        self.max_iter = max_iter
        self.tol = tol
        self.use_log_transform = use_log_transform  # For SAR speckle handling
        
        # Move denoiser to device
        self.denoiser.to(device)
        self.denoiser.eval()
    
    def safe_log_transform(self, image, eps=1e-6):
        """Safe log-transform with proper normalization for SAR images"""
        if isinstance(image, torch.Tensor):
            img = image.float()
            # Normalize to [0, 1] to prevent overflow
            img_max = img.max()
            if img_max > 0:
                img = img / (img_max + eps)
            # Clip to avoid log(0)
            img = torch.clip(img, eps, 1.0)
            return torch.log(img)
        else:
            img = image.astype(np.float32)
            # Normalize to [0, 1] to prevent overflow
            img_max = img.max()
            if img_max > 0:
                img = img / (img_max + eps)
            # Clip to avoid log(0)
            img = np.clip(img, eps, 1.0)
            return np.log(img)
    
    def safe_exp_transform(self, image_log, original_max=None):
        """Safe inverse log-transform with proper scaling"""
        if isinstance(image_log, torch.Tensor):
            # Apply exp
            img_exp = torch.exp(image_log)
            # If we have original max, scale back
            if original_max is not None:
                img_exp = img_exp * original_max
            # Clip to valid range
            return torch.clip(img_exp, 0, 1)
        else:
            # Apply exp
            img_exp = np.exp(image_log)
            # If we have original max, scale back
            if original_max is not None:
                img_exp = img_exp * original_max
            # Clip to valid range
            return np.clip(img_exp, 0, 1)
        
    def create_psf(self, image_shape, psf_size=5, psf_sigma=1.0):
        """Create point spread function for blur - FIXED VERSION"""
        h, w = image_shape
        center_h, center_w = h // 2, w // 2
        
        # Create coordinate grids efficiently
        y, x = np.ogrid[:h, :w]
        
        # Gaussian PSF
        psf = np.exp(-((x - center_w)**2 + (y - center_h)**2) / (2 * psf_sigma**2))
        psf = psf / (np.sum(psf) + 1e-8)  # Normalize with numerical stability
        
        return psf
    
    def fft_conv(self, image, psf):
        """Convolution using FFT"""
        # Convert to torch tensors
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).float()
        if isinstance(psf, np.ndarray):
            psf = torch.from_numpy(psf).float()
        
        # Move to device
        image = image.to(self.device)
        psf = psf.to(self.device)
        
        # Add batch and channel dimensions if needed
        if image.dim() == 2:
            image = image.unsqueeze(0).unsqueeze(0)
        if psf.dim() == 2:
            psf = psf.unsqueeze(0).unsqueeze(0)
        
        # FFT convolution
        image_fft = torch.fft.fft2(image)
        psf_fft = torch.fft.fft2(psf)
        result_fft = image_fft * psf_fft
        result = torch.fft.ifft2(result_fft).real
        
        return result.squeeze()
    
    def x_update_fft(self, z, u, H, H_conj, rho):
        """x-update step using FFT - FIXED VERSION"""
        # Ensure z and u are the same shape as H
        if z.dim() == 2:
            z = z.unsqueeze(0).unsqueeze(0)
        if u.dim() == 2:
            u = u.unsqueeze(0).unsqueeze(0)
        
        # Compute H^T * y + rho * (z - u)
        z_u_diff = z - u
        if z_u_diff.dim() == 4:
            z_u_diff = z_u_diff.squeeze(0).squeeze(0)  # Remove batch and channel dims
        
        # Convert to frequency domain
        z_u_fft = torch.fft.fft2(z_u_diff)
        
        # Compute rhs in frequency domain
        rhs_fft = H_conj + rho * z_u_fft
        
        # Solve (H^T * H + rho * I) * x = rhs using FFT
        H_conj_H = torch.conj(H) * H
        denominator = H_conj_H + rho
        
        # Avoid division by zero
        denominator = torch.where(denominator.abs() < 1e-10, 
                                 torch.ones_like(denominator) * 1e-10, 
                                 denominator)
        
        x_fft = rhs_fft / denominator
        x = torch.fft.ifft2(x_fft).real
        
        return x
    
    def denoise_step(self, x_bar, noise_level=None):
        """Denoising step using deep learning denoiser - FIXED VERSION"""
        # Clip to valid range
        x_bar = torch.clamp(x_bar, 0, 1)
        
        # Ensure proper tensor shape for denoiser
        original_shape = x_bar.shape
        if x_bar.dim() == 2:
            x_bar = x_bar.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        elif x_bar.dim() == 3:
            x_bar = x_bar.unsqueeze(0)  # [1, C, H, W]
        elif x_bar.dim() == 4:
            pass  # Already correct shape [B, C, H, W]
        else:
            raise ValueError(f"Unexpected tensor dimension: {x_bar.dim()}")
        
        # Denoise using the neural network
        with torch.no_grad():
            try:
                if hasattr(self.denoiser, 'noise_conditioning') and self.denoiser.noise_conditioning:
                    # Create noise level tensor if not provided
                    if noise_level is None:
                        noise_level = torch.tensor(0.3, device=x_bar.device)
                    elif isinstance(noise_level, (int, float)):
                        noise_level = torch.tensor(noise_level, device=x_bar.device)
                    denoised = self.denoiser(x_bar, noise_level)
                else:
                    denoised = self.denoiser(x_bar)
                
                # Ensure output has same spatial dimensions as input
                if denoised.dim() == 4:
                    denoised = denoised.squeeze(0).squeeze(0)  # Remove batch and channel dims
                elif denoised.dim() == 3:
                    denoised = denoised.squeeze(0)  # Remove batch dim
                
                # Ensure output matches original shape
                if denoised.shape != original_shape:
                    denoised = denoised.view(original_shape)
                
            except Exception as e:
                print(f"❌ Denoiser failed: {str(e)}")
                # Return input as fallback
                denoised = x_bar.squeeze() if x_bar.dim() > 2 else x_bar
        
        return denoised
    
    def compute_energy(self, x, z, u, H, y, rho):
        """Compute ADMM energy function"""
        # Data fidelity term: ||Hx - y||^2
        Hx = self.fft_conv(x, H)
        data_fidelity = torch.norm(Hx - y) ** 2
        
        # Augmented Lagrangian term: rho/2 * ||x - z + u||^2
        aug_lag = (rho / 2) * torch.norm(x - z + u) ** 2
        
        energy = data_fidelity + aug_lag
        return energy.item()
    
    def compute_residual(self, x, z, u, rho):
        """Compute primal and dual residuals"""
        # Primal residual: r = x - z
        r = torch.norm(x - z)
        
        # Dual residual: s = rho * (z - z_prev)
        # For simplicity, we'll use the change in z
        s = rho * torch.norm(z)
        
        return r.item(), s.item()
    
    def adaptive_rho_update(self, r, s, rho, tau=2.0, mu=10.0):
        """Adaptive rho update based on residuals"""
        if r > mu * s:
            rho = tau * rho
        elif s > mu * r:
            rho = rho / tau
        
        return rho
    
    def denoise(self, noisy_image, psf=None, y=None, noise_level=None):
        """Main ADMM-PnP denoising algorithm"""
        # Convert to torch tensor
        if isinstance(noisy_image, np.ndarray):
            noisy_image = torch.from_numpy(noisy_image).float()
        
        noisy_image = noisy_image.to(self.device)
        
        # Store original max for proper inverse scaling
        original_max = None
        if self.use_log_transform:
            if isinstance(noisy_image, torch.Tensor):
                original_max = noisy_image.max().item()
            else:
                original_max = float(noisy_image.max())
            
            # Apply safe log-transform for SAR speckle handling
            noisy_image_log = self.safe_log_transform(noisy_image)
            # print(f"🔍 Input log range: {noisy_image_log.min():.6f}, {noisy_image_log.max():.6f}")
            noisy_image = noisy_image_log
            # Also transform y if provided
            if y is not None:
                if isinstance(y, np.ndarray):
                    y = torch.from_numpy(y).float()
                y = y.to(self.device)
                y = self.safe_log_transform(y)
        
        # Initialize variables
        x = noisy_image.clone()
        z = noisy_image.clone()
        u = torch.zeros_like(noisy_image)
        
        # Create PSF if not provided - FIXED VERSION
        if psf is None:
            # Extract spatial dimensions from tensor shape
            if len(noisy_image.shape) == 4:  # [B, C, H, W]
                h, w = noisy_image.shape[2], noisy_image.shape[3]
            elif len(noisy_image.shape) == 3:  # [C, H, W]
                h, w = noisy_image.shape[1], noisy_image.shape[2]
            else:  # [H, W]
                h, w = noisy_image.shape
            psf = self.create_psf((h, w))
        
        if isinstance(psf, np.ndarray):
            psf = torch.from_numpy(psf).float().to(self.device)
        
        # Create observation y if not provided
        if y is None:
            y = self.fft_conv(noisy_image, psf)
        
        # FFT of PSF - FIXED VERSION
        H = torch.fft.fft2(psf)
        
        # Ensure y is the right shape for FFT
        if y.dim() == 2:
            y_fft = torch.fft.fft2(y)
        else:
            # Handle multi-dimensional case
            y_2d = y.squeeze() if y.dim() > 2 else y
            y_fft = torch.fft.fft2(y_2d)
        
        H_conj = torch.conj(H) * y_fft
        
        # Initialize rho
        rho = self.rho_init
        
        # Storage for monitoring
        energies = []
        residuals = []
        
        # Main ADMM loop
        for iteration in tqdm(range(self.max_iter), desc="ADMM-PnP"):
            # x-update: solve (H^T H + rho I) x = H^T y + rho(z - u)
            x = self.x_update_fft(z, u, H, H_conj, rho)
            
            # x-bar update: x_bar = alpha * x + (1 - alpha) * z
            x_bar = self.alpha * x + (1 - self.alpha) * z
            
            # Denoising step: z = D_phi(x_bar + u)
            den_input = torch.clamp(x_bar + u, 0, 1)
            z_denoised = self.denoise_step(den_input, noise_level)
            
            # z-update: z = theta * z_denoised + (1 - theta) * den_input
            z = self.theta * z_denoised + (1 - self.theta) * den_input
            
            # Dual update: u = u + x - z
            u = u + x - z
            
            # Compute energy and residuals
            energy = self.compute_energy(x, z, u, H, y, rho)
            r_primal, r_dual = self.compute_residual(x, z, u, rho)
            
            energies.append(energy)
            residuals.append(r_primal)
            
            # Adaptive rho update
            if self.adaptive_rho and iteration > 0:
                rho = self.adaptive_rho_update(r_primal, r_dual, rho)
            
            # Check convergence
            if r_primal < self.tol:
                print(f"Converged at iteration {iteration}")
                break
        
        # Apply safe inverse log-transform if used
        denoised_result = z
        if self.use_log_transform:
            denoised_result = self.safe_exp_transform(z, original_max)
            # Debug: Print scaling information (only in debug mode)
            # if isinstance(denoised_result, torch.Tensor):
            #     denoised_np = denoised_result.cpu().numpy()
            # else:
            #     denoised_np = denoised_result
            # print(f"🔍 Final output stats → min: {denoised_np.min():.6f}, max: {denoised_np.max():.6f}, mean: {denoised_np.mean():.6f}")
        
        # Convert to numpy for return
        if isinstance(denoised_result, torch.Tensor):
            denoised_result = denoised_result.cpu().numpy()
        
        return {
            'denoised': denoised_result,
            'energies': energies,
            'residuals': residuals,
            'iterations': iteration + 1
        }
    
    def denoise_batch(self, noisy_batch, psf=None, noise_levels=None):
        """Denoise a batch of images"""
        results = []
        
        for i in range(noisy_batch.shape[0]):
            noisy_img = noisy_batch[i, 0].cpu().numpy()  # Remove channel dimension
            noise_level = noise_levels[i] if noise_levels is not None else None
            
            result = self.denoise(noisy_img, psf=psf, noise_level=noise_level)
            results.append(result)
        
        return results


class TVDenoiser:
    """Total Variation denoiser for baseline comparison"""
    
    def __init__(self, device='cuda', lambda_tv=0.1, max_iter=100):
        self.device = device
        self.lambda_tv = lambda_tv
        self.max_iter = max_iter
    
    def tv_denoise(self, noisy_image):
        """TV denoising using gradient descent"""
        x = noisy_image.clone().requires_grad_(True)
        optimizer = torch.optim.Adam([x], lr=0.01)
        
        for _ in range(self.max_iter):
            optimizer.zero_grad()
            
            # TV regularization
            tv_loss = self.compute_tv_loss(x)
            
            # Data fidelity
            data_loss = torch.norm(x - noisy_image) ** 2
            
            # Total loss
            loss = data_loss + self.lambda_tv * tv_loss
            loss.backward()
            optimizer.step()
            
            # Clamp to valid range
            with torch.no_grad():
                x = torch.clamp(x, 0, 1)
        
        return x.detach()
    
    def compute_tv_loss(self, x):
        """Compute total variation loss"""
        # Compute gradients
        grad_x = torch.diff(x, dim=1, prepend=x[:, :, :1])
        grad_y = torch.diff(x, dim=0, prepend=x[:1, :, :])
        
        # TV norm
        tv = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8).sum()
        return tv


def compare_denoisers(noisy_image, denoiser_dl, denoiser_tv, psf=None):
    """Compare deep learning and TV denoisers"""
    # Deep learning denoiser
    result_dl = denoiser_dl.denoise(noisy_image, psf=psf)
    
    # TV denoiser
    tv_denoiser = TVDenoiser()
    result_tv = tv_denoiser.tv_denoise(torch.from_numpy(noisy_image).float())
    
    return {
        'dl_result': result_dl,
        'tv_result': result_tv.cpu().numpy()
    }


if __name__ == "__main__":
    # Test ADMM-PnP algorithm
    from models.unet import create_model
    
    # Create a simple denoiser for testing
    denoiser = create_model('unet', n_channels=1, noise_conditioning=True)
    
    # Create ADMM-PnP instance
    admm = ADMMPnP(denoiser, device='cpu')
    
    # Create test image
    test_image = np.random.rand(128, 128)
    
    # Test denoising
    result = admm.denoise(test_image)
    
    print(f"Denoising completed in {result['iterations']} iterations")
    print(f"Final energy: {result['energies'][-1]:.6f}")
    print(f"Final residual: {result['residuals'][-1]:.6f}")
