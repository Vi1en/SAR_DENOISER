"""
Residual U-Net denoiser (Upgrade 4): same topology as ``UNet`` with residual blocks
for stronger gradient flow. Distinct state-dict keys (``res_double_conv``) so inference
can auto-detect vs vanilla U-Net.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResDoubleConv(nn.Module):
    """Two 3x3 convs + BN + ReLU with a 1x1 shortcut when channels change."""

    def __init__(self, in_channels: int, out_channels: int, mid_channels: int | None = None):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, 1, bias=False)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out + self.shortcut(x))


class ResDown(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2), ResDoubleConv(in_channels, out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class ResUp(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = ResDoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = ResDoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)
        x1 = F.pad(
            x1,
            [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2],
        )
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class ResOutConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ResUNet(nn.Module):
    """Residual U-Net; ``forward(x, noise_level=None)`` matches ``UNet``."""

    def __init__(
        self,
        n_channels: int = 1,
        n_classes: int = 1,
        bilinear: bool = False,
        noise_conditioning: bool = False,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.noise_conditioning = noise_conditioning
        input_channels = n_channels + 1 if noise_conditioning else n_channels

        self.inc = ResDoubleConv(input_channels, 64)
        self.down1 = ResDown(64, 128)
        self.down2 = ResDown(128, 256)
        self.down3 = ResDown(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = ResDown(512, 1024 // factor)
        self.up1 = ResUp(1024, 512 // factor, bilinear)
        self.up2 = ResUp(512, 256 // factor, bilinear)
        self.up3 = ResUp(256, 128 // factor, bilinear)
        self.up4 = ResUp(128, 64, bilinear)
        self.outc = ResOutConv(64, n_classes)

    def forward(self, x: torch.Tensor, noise_level: torch.Tensor | None = None) -> torch.Tensor:
        if self.noise_conditioning and noise_level is not None:
            batch_size, _, height, width = x.shape
            nl = noise_level
            if nl.dim() == 0:
                nl = nl.reshape(1).expand(batch_size)
            elif nl.dim() == 1:
                if nl.shape[0] == 1:
                    nl = nl.expand(batch_size)
                elif nl.shape[0] != batch_size:
                    nl = nl[0].reshape(1).expand(batch_size)
            elif nl.dim() == 2 and nl.shape[0] == batch_size and nl.shape[1] == 1:
                nl = nl.squeeze(1)
            elif nl.dim() == 2 and nl.shape == (1, batch_size):
                nl = nl.squeeze(0)
            noise_channel = nl.reshape(batch_size, 1, 1, 1).expand(batch_size, 1, height, width)
            x = torch.cat([x, noise_channel], dim=1)

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)
