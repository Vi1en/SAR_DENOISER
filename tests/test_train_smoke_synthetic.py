"""Single-batch forward smoke (no dataset, no training loop)."""

import torch


def test_one_batch_forward_unconditioned():
    from models.unet import create_model

    m = create_model("unet", n_channels=1, noise_conditioning=False)
    m.eval()
    x = torch.randn(1, 1, 64, 64)
    y = m(x)
    assert y.shape == x.shape


def test_one_batch_forward_noise_conditioned():
    from models.unet import create_model

    m = create_model("unet", n_channels=1, noise_conditioning=True)
    m.eval()
    x = torch.randn(2, 1, 64, 64)
    noise_level = torch.tensor([0.5, 1.0])
    y = m(x, noise_level=noise_level)
    assert y.shape == x.shape


def test_res_unet_forward_smoke():
    from models.unet import create_model

    m = create_model("res_unet", n_channels=1, noise_conditioning=False)
    m.eval()
    x = torch.randn(1, 1, 64, 64)
    y = m(x)
    assert y.shape == x.shape
