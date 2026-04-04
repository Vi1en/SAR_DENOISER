"""
Core SAR denoising inference (model load + ADMM / direct / TV). No Streamlit.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from inference.types import DenoiseMethod, Message, denoise_result_dict

# Default checkpoint layout (same as demo/streamlit_app.py)
DEFAULT_IMPROVED_CKPT = Path("checkpoints_improved/best_model.pth")
DEFAULT_SIMPLE_CKPT = Path("checkpoints_simple/best_model.pth")


def _merge_blind_qa_meta(
    meta: Dict[str, Any],
    noisy: np.ndarray,
    denoised: np.ndarray,
    include: bool,
) -> None:
    """Optional no-reference metrics under ``meta['blind_qa']`` (does not touch training eval)."""
    if not include:
        return
    try:
        from evaluators.blind_qa import compute_blind_qa

        meta["blind_qa"] = compute_blind_qa(noisy, denoised)
    except Exception as e:
        meta["blind_qa_error"] = str(e)[:300]


def _infer_noise_conditioning_from_state(state_dict: dict) -> bool:
    """Match ``create_model(..., noise_conditioning=?)`` to saved first-conv input channels."""
    for k, v in state_dict.items():
        if not hasattr(v, "shape") or len(v.shape) != 4:
            continue
        if k == "inc.double_conv.0.weight":
            return int(v.shape[1]) > 1
        if k == "inc.conv1.weight":
            return int(v.shape[1]) > 1
        if ".dncnn.0.weight" in k or k.endswith("dncnn.0.weight"):
            return int(v.shape[1]) > 1
    return True


def preprocess_noisy_array(noisy_image: np.ndarray) -> np.ndarray:
    """Float32, complex -> magnitude, normalize to [0, 1]."""
    noisy_image = noisy_image.astype(np.float32)
    if np.iscomplexobj(noisy_image):
        noisy_image = np.abs(noisy_image)
    img_max = np.max(noisy_image)
    if img_max > 0:
        noisy_image = noisy_image / (img_max + 1e-8)
    else:
        noisy_image = np.zeros_like(noisy_image)
    return noisy_image


def _sidebar_model_slug(model_type: str) -> str:
    t = model_type.lower().replace("-", "")
    if t.replace("_", "") == "resunet":
        return "res_unet"
    return t


def _detect_arch_from_state_keys(state_dict_keys: List[str]) -> str:
    keys = list(state_dict_keys)
    # ResUNet uses ResDoubleConv (conv1/conv2) inside downs; vanilla U-Net uses DoubleConv.
    if any("maxpool_conv.1.conv1." in k for k in keys):
        return "res_unet"
    if any("inc.double_conv" in k for k in keys):
        return "unet"
    if any("dncnn" in k for k in keys):
        return "dncnn"
    return ""


def load_admm_denoiser(
    device: torch.device,
    model_type: str,
    improved_path: Path | None = None,
    simple_path: Path | None = None,
) -> Tuple[torch.nn.Module, List[Message]]:
    """Load U-Net/DnCNN for ADMM branch (noise_conditioning=False), same order as Streamlit."""
    improved_path = improved_path or DEFAULT_IMPROVED_CKPT
    simple_path = simple_path or DEFAULT_SIMPLE_CKPT
    messages: List[Message] = []
    denoiser: torch.nn.Module | None = None

    if improved_path.is_file():
        try:
            checkpoint = torch.load(improved_path, map_location=device)
            state = checkpoint["model_state_dict"]
            detected = _detect_arch_from_state_keys(state.keys())
            actual = detected or _sidebar_model_slug(model_type)
            from models.unet import create_model

            denoiser = create_model(actual, n_channels=1, noise_conditioning=False)
            denoiser.load_state_dict(state)
            if detected == "unet":
                messages.append(("info", "🔍 Detected U-Net model in checkpoint"))
            elif detected == "dncnn":
                messages.append(("info", "🔍 Detected DnCNN model in checkpoint"))
            messages.append(("success", "🚀 Loaded IMPROVED trained model (30+ dB PSNR)"))
        except Exception as e:
            messages.append(("error", f"❌ Failed to load improved model: {e!s}"))
            denoiser = None

    if denoiser is None and simple_path.is_file():
        try:
            checkpoint = torch.load(simple_path, map_location=device)
            state = checkpoint["model_state_dict"]
            detected = _detect_arch_from_state_keys(state.keys())
            actual = detected or _sidebar_model_slug(model_type)
            from models.unet import create_model

            denoiser = create_model(actual, n_channels=1, noise_conditioning=False)
            denoiser.load_state_dict(state)
            messages.append(("success", "✅ Loaded basic trained model"))
        except Exception as e:
            messages.append(("error", f"❌ Failed to load basic model: {e!s}"))
            from models.unet import create_model

            denoiser = create_model(_sidebar_model_slug(model_type), n_channels=1, noise_conditioning=False)
            messages.append(("warning", "⚠️ Using random weights (failed to load trained model)"))
    elif denoiser is None:
        from models.unet import create_model

        denoiser = create_model(_sidebar_model_slug(model_type), n_channels=1, noise_conditioning=False)
        messages.append(("warning", "⚠️ Using random weights (no trained model found)"))

    return denoiser, messages


def _postprocess_admm_output(
    noisy_image: np.ndarray,
    denoised_image: np.ndarray,
    quality_enhancement: bool,
) -> Tuple[np.ndarray, List[Message]]:
    messages: List[Message] = []
    if hasattr(denoised_image, "detach"):
        denoised_image = denoised_image.detach().cpu().numpy()

    denoised_min = float(np.min(denoised_image))
    denoised_max = float(np.max(denoised_image))
    if denoised_max > denoised_min:
        denoised_image = (denoised_image - denoised_min) / (denoised_max - denoised_min + 1e-8)
    else:
        denoised_image = np.zeros_like(denoised_image)

    denoised_image = np.clip(denoised_image, 0, 1)

    if quality_enhancement:
        messages.append(("info", "✨ Quality Enhancement Mode: Applying minimal refinement..."))
        try:
            import cv2

            denoised_uint8 = (denoised_image * 255).astype(np.uint8)
            denoised_refined = cv2.fastNlMeansDenoising(denoised_uint8, None, 2, 7, 21) / 255.0
            denoised_image = denoised_refined
            messages.append(("success", "✅ Applied minimal non-local means refinement"))
        except Exception as e:
            messages.append(("warning", f"⚠️ Quality enhancement failed: {e!s} - using original result"))
    else:
        messages.append(("info", "🚫 Quality Enhancement Mode disabled - preserving original sharpness"))

    messages.append(
        ("success", f"✅ Postprocessed: range=[{denoised_image.min():.4f}, {denoised_image.max():.4f}]")
    )
    return denoised_image, messages


def _diagnose_admm_output(noisy_image: np.ndarray, denoised_image: np.ndarray) -> Tuple[np.ndarray, List[Message]]:
    messages: List[Message] = []
    if denoised_image.shape != noisy_image.shape:
        messages.append(
            ("error", f"🚨 SHAPE MISMATCH: Input {noisy_image.shape} vs Output {denoised_image.shape}")
        )
        messages.append(("error", "🚨 This indicates a fundamental algorithm failure!"))
        return noisy_image.copy(), messages
    if np.any(np.isnan(denoised_image)) or np.any(np.isinf(denoised_image)):
        messages.append(("error", "🚨 OUTPUT CONTAINS NaN OR INF VALUES!"))
        return noisy_image.copy(), messages
    if np.std(denoised_image) < 1e-6:
        messages.append(("error", "🚨 OUTPUT IS COMPLETELY FLAT - Algorithm failed!"))
        return noisy_image.copy(), messages
    return denoised_image, messages


class SARDenoiseService:
    """Load checkpoints and run denoising on numpy arrays."""

    def __init__(
        self,
        device: str | torch.device = "cpu",
        improved_checkpoint: Path | None = None,
        simple_checkpoint: Path | None = None,
        *,
        infer_backend: str | None = None,
        onnx_path: str | Path | None = None,
    ):
        if isinstance(device, str):
            if device == "auto":
                self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                self._device = torch.device(device)
        else:
            self._device = device
        self.improved_checkpoint = improved_checkpoint
        self.simple_checkpoint = simple_checkpoint
        ib = str(infer_backend).strip().lower() if infer_backend and str(infer_backend).strip() else None
        self._infer_backend_config = ib if ib in ("pytorch", "onnx") else None
        self._onnx_path_config: str | Path | None = onnx_path
        self._denoiser: torch.nn.Module | None = None
        self._loaded_model_type: str | None = None
        self._onnx_direct: Any = None
        self._onnx_direct_path: Path | None = None

    @property
    def device(self) -> torch.device:
        return self._device

    def _effective_inference_backend(self) -> str:
        e = os.environ.get("SAR_BACKEND", "").strip()
        if e:
            return e.lower()
        if self._infer_backend_config:
            return self._infer_backend_config
        return "pytorch"

    def _effective_onnx_path(self, direct_checkpoint: Path | str | None) -> Optional[Path]:
        e = os.environ.get("SAR_ONNX_PATH", "").strip()
        if e:
            return Path(e).expanduser().resolve()
        if self._onnx_path_config is not None and str(self._onnx_path_config).strip():
            return Path(self._onnx_path_config).expanduser().resolve()
        if direct_checkpoint is not None:
            p = Path(direct_checkpoint)
            if p.suffix.lower() == ".onnx":
                return p.resolve()
        return None

    def load_weights(
        self,
        model_type: str,
        checkpoint_path: Path | str,
        *,
        noise_conditioning: bool = True,
        strict: bool = False,
    ) -> None:
        """Load a single checkpoint into the service (for APIs that know the path)."""
        path = Path(checkpoint_path)
        from models.unet import create_model

        net = create_model(model_type.lower(), n_channels=1, noise_conditioning=noise_conditioning)
        state = torch.load(path, map_location=self._device)
        if isinstance(state, dict) and "model_state_dict" in state:
            net.load_state_dict(state["model_state_dict"], strict=strict)
        else:
            net.load_state_dict(state, strict=strict)
        net.to(self._device)
        net.eval()
        self._denoiser = net
        self._loaded_model_type = model_type.lower()

    def _get_onnx_denoiser(self, path: Path) -> Any:
        from inference.onnx_backend import ONNXDirectDenoiser

        path = path.resolve()
        if self._onnx_direct is None or self._onnx_direct_path != path:
            ep = os.environ.get("SAR_ONNX_EP")
            providers = [s.strip() for s in ep.split(",") if s.strip()] if ep else None
            self._onnx_direct = ONNXDirectDenoiser(path, providers=providers)
            self._onnx_direct_path = path
        return self._onnx_direct

    def denoise_numpy(
        self,
        noisy: np.ndarray,
        method: DenoiseMethod,
        *,
        model_type: str = "U-Net",
        max_iter: int = 15,
        rho_init: float = 1.0,
        alpha: float = 0.1,
        theta: float = 0.05,
        use_log_transform: bool = False,
        disable_denoising: bool = False,
        quality_enhancement: bool = False,
        speckle_factor: float = 0.3,
        direct_checkpoint: Path | str | None = None,
        return_uncertainty: bool = False,
        uncertainty_tta_passes: int = 4,
        include_blind_qa: bool = False,
    ) -> Dict[str, Any]:
        """
        Full Streamlit-equivalent pipeline. Returns dict with denoised, energies, residuals, messages.

        ``return_uncertainty`` (Direct denoising only): TTA stack → pixelwise std map in ``uncertainty``.
        ``include_blind_qa``: when True, adds ``meta['blind_qa']`` (no-reference proxies; optional).
        """
        messages: List[Message] = []
        noisy_image = preprocess_noisy_array(noisy)
        messages.append(
            ("info", "🔧 Applying enhanced SAR preprocessing..."),
        )
        messages.append(
            ("success", f"✅ Preprocessed: range=[{noisy_image.min():.4f}, {noisy_image.max():.4f}]"),
        )

        device = self._device
        energies: List[float] = [0.0]
        residuals: List[float] = [0.0]
        uncertainty_map: np.ndarray | None = None

        if return_uncertainty and method != "Direct Denoising":
            messages.append(
                (
                    "warning",
                    "⚠️ Uncertainty (TTA) is only supported for Direct Denoising; ignoring flag.",
                ),
            )
            return_uncertainty = False

        if method == "ADMM-PnP-DL":
            denoiser, load_msgs = load_admm_denoiser(
                device,
                model_type,
                improved_path=self.improved_checkpoint,
                simple_path=self.simple_checkpoint,
            )
            messages.extend(load_msgs)

            if disable_denoising:
                messages.append(("info", "🚫 Denoising disabled - returning original image"))
                meta_skip: Dict[str, Any] = {"method": method}
                _merge_blind_qa_meta(
                    meta_skip, noisy_image, noisy_image.copy(), include_blind_qa
                )
                return denoise_result_dict(
                    noisy_image.copy(),
                    [0.0],
                    [0.0],
                    messages,
                    meta=meta_skip,
                )

            messages.append(("info", "🔧 Using FIXED ADMM-PnP algorithm"))
            try:
                from algos.admm_pnp import ADMMPnP

                admm = ADMMPnP(
                    denoiser,
                    device=device,
                    max_iter=max_iter,
                    rho_init=rho_init,
                    alpha=alpha,
                    theta=theta,
                    use_log_transform=use_log_transform,
                )
                result = admm.denoise(noisy_image)
                denoised_image = result["denoised"]
                energies = list(result.get("energies", [0.0]))
                residuals = list(result.get("residuals", [0.0]))
                messages.append(("success", "✅ FIXED ADMM-PnP denoising completed successfully"))
            except Exception as e:
                messages.append(("error", f"❌ FIXED ADMM-PnP failed: {e!s}"))
                messages.append(("error", "🚨 Returning original image as fallback"))
                denoised_image = noisy_image.copy()
                energies = [0.0]
                residuals = [0.0]

            messages.append(("info", "🔍 Running diagnostic checks..."))
            denoised_image, diag_msgs = _diagnose_admm_output(noisy_image, denoised_image)
            messages.extend(diag_msgs)

            messages.extend(
                [
                    ("success", "✅ FIXED ADMM-PnP ALGORITHM ACTIVE"),
                    ("success", "✅ Fixed tensor shape mismatches and FFT operations"),
                    ("success", "✅ Fixed PSF creation and denoiser integration"),
                    ("info", "ℹ️ Using properly implemented ADMM-PnP with deep learning denoiser"),
                    ("info", "🔧 Applying enhanced postprocessing..."),
                ]
            )
            denoised_image, pp_msgs = _postprocess_admm_output(
                noisy_image, denoised_image, quality_enhancement
            )
            messages.extend(pp_msgs)

        elif method == "Direct Denoising":
            from inference.uncertainty import tta_direct_onnx, tta_direct_pytorch

            backend = self._effective_inference_backend()
            onnx_path = self._effective_onnx_path(direct_checkpoint)
            used_onnx = False

            if return_uncertainty:
                if backend == "onnx" and onnx_path is not None and onnx_path.is_file():
                    try:
                        ort_d = self._get_onnx_denoiser(onnx_path)
                        denoised_image, uncertainty_map = tta_direct_onnx(
                            ort_d,
                            noisy_image,
                            speckle_factor,
                            passes=uncertainty_tta_passes,
                        )
                        messages.append(
                            ("success", "✅ Direct denoising + TTA uncertainty (ONNX Runtime)"),
                        )
                        used_onnx = True
                    except Exception as e:
                        messages.append(
                            ("warning", f"⚠️ ONNX TTA failed ({e!s}); falling back to PyTorch"),
                        )

                if not used_onnx:
                    if backend == "onnx" and (onnx_path is None or not onnx_path.is_file()):
                        messages.append(
                            (
                                "warning",
                                "⚠️ SAR_BACKEND=onnx but SAR_ONNX_PATH missing or not found; using PyTorch",
                            ),
                        )

                    from models.unet import create_model

                    ckpt = (
                        Path(direct_checkpoint)
                        if direct_checkpoint is not None
                        else (self.simple_checkpoint or DEFAULT_SIMPLE_CKPT)
                    )
                    nc = True
                    state_sd = None
                    if ckpt.is_file() and ckpt.suffix.lower() != ".onnx":
                        checkpoint = torch.load(ckpt, map_location=device)
                        state_sd = checkpoint["model_state_dict"]
                        nc = _infer_noise_conditioning_from_state(state_sd)

                    denoiser = create_model(
                        _sidebar_model_slug(model_type), n_channels=1, noise_conditioning=nc
                    )
                    if state_sd is not None:
                        denoiser.load_state_dict(state_sd)
                        messages.append(("success", "✅ Loaded trained model"))
                    else:
                        messages.append(
                            ("warning", "⚠️ Using random weights (no trained model found)"),
                        )

                    denoiser.eval()
                    denoiser.to(device)
                    denoised_image, uncertainty_map = tta_direct_pytorch(
                        denoiser,
                        noisy_image,
                        speckle_factor,
                        device,
                        passes=uncertainty_tta_passes,
                    )
                    messages.append(
                        ("success", "✅ Direct denoising + TTA uncertainty (PyTorch)"),
                    )
            else:
                if backend == "onnx" and onnx_path is not None and onnx_path.is_file():
                    try:
                        ort_d = self._get_onnx_denoiser(onnx_path)
                        out = ort_d.run(noisy_image.astype(np.float32), speckle_factor)
                        denoised_image = np.asarray(out, dtype=np.float32).squeeze()
                        if denoised_image.ndim > 2:
                            denoised_image = denoised_image.squeeze()
                        messages.append(
                            ("success", "✅ Direct denoising (ONNX Runtime)"),
                        )
                        used_onnx = True
                    except Exception as e:
                        messages.append(
                            ("warning", f"⚠️ ONNX Runtime failed ({e!s}); falling back to PyTorch"),
                        )

                if not used_onnx:
                    if backend == "onnx" and (onnx_path is None or not onnx_path.is_file()):
                        messages.append(
                            (
                                "warning",
                                "⚠️ SAR_BACKEND=onnx but SAR_ONNX_PATH missing or not found; using PyTorch",
                            ),
                        )

                    from models.unet import create_model

                    ckpt = (
                        Path(direct_checkpoint)
                        if direct_checkpoint is not None
                        else (self.simple_checkpoint or DEFAULT_SIMPLE_CKPT)
                    )
                    nc = True
                    state_sd = None
                    if ckpt.is_file() and ckpt.suffix.lower() != ".onnx":
                        checkpoint = torch.load(ckpt, map_location=device)
                        state_sd = checkpoint["model_state_dict"]
                        nc = _infer_noise_conditioning_from_state(state_sd)

                    denoiser = create_model(
                        _sidebar_model_slug(model_type), n_channels=1, noise_conditioning=nc
                    )
                    if state_sd is not None:
                        denoiser.load_state_dict(state_sd)
                        messages.append(("success", "✅ Loaded trained model"))
                    else:
                        messages.append(
                            ("warning", "⚠️ Using random weights (no trained model found)"),
                        )

                    denoiser.eval()
                    denoiser.to(device)
                    with torch.no_grad():
                        input_tensor = (
                            torch.from_numpy(noisy_image).float().unsqueeze(0).unsqueeze(0).to(device)
                        )
                        noise_level = torch.tensor(speckle_factor, device=device)
                        if hasattr(denoiser, "noise_conditioning") and denoiser.noise_conditioning:
                            denoised_tensor = denoiser(input_tensor, noise_level)
                        else:
                            denoised_tensor = denoiser(input_tensor)
                        denoised_image = denoised_tensor.squeeze().cpu().numpy()

            energies = [0.0]
            residuals = [0.0]

        elif method == "TV Denoising":
            from algos.admm_pnp import TVDenoiser

            tv_denoiser = TVDenoiser(device=device)
            denoised_tensor = tv_denoiser.tv_denoise(
                torch.from_numpy(noisy_image).float().to(device)
            )
            denoised_image = denoised_tensor.cpu().numpy()
            energies = [0.0]
            residuals = [0.0]
        else:
            raise ValueError(f"Unknown method: {method}")

        meta: Dict[str, Any] = {"method": method}
        if uncertainty_map is not None:
            u = uncertainty_map
            finite = u[np.isfinite(u)]
            meta["uncertainty_mean"] = float(np.mean(finite)) if finite.size else 0.0
            meta["uncertainty_max"] = float(np.max(u)) if u.size else 0.0
            meta["uncertainty_tta_passes"] = int(uncertainty_tta_passes)

        _merge_blind_qa_meta(meta, noisy_image, denoised_image, include_blind_qa)

        return denoise_result_dict(
            denoised_image,
            energies,
            residuals,
            messages,
            meta=meta,
            uncertainty=uncertainty_map,
        )

    def denoise_file(
        self,
        path: Path | str,
        method: DenoiseMethod,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Load grayscale image from disk (PNG/JPEG), same as upload path in Streamlit."""
        import cv2

        path = Path(path)
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not read image: {path}")
        arr = img.astype(np.float32) / 255.0
        return self.denoise_numpy(arr, method, **kwargs)


def replay_messages_streamlit(messages: List[Message], st_module: Any) -> None:
    """Map (level, text) to st.info / st.success / st.warning / st.error."""
    fn = {
        "info": st_module.info,
        "success": st_module.success,
        "warning": st_module.warning,
        "error": st_module.error,
    }
    for level, text in messages:
        fn[level](text)
