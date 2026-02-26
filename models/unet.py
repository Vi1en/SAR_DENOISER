"""
U-Net architecture for SAR image denoising
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """U-Net for SAR image denoising"""
    
    def __init__(self, n_channels=1, n_classes=1, bilinear=False, noise_conditioning=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.noise_conditioning = noise_conditioning
        
        # Add noise channel if conditioning is enabled
        input_channels = n_channels + 1 if noise_conditioning else n_channels
        
        self.inc = (DoubleConv(input_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x, noise_level=None):
        if self.noise_conditioning and noise_level is not None:
            # Add noise level as additional channel
            batch_size, _, height, width = x.shape
            # Ensure noise_level has the right shape
            if noise_level.dim() == 0:  # scalar
                noise_level = noise_level.unsqueeze(0).expand(batch_size)
            elif noise_level.dim() == 1 and noise_level.shape[0] == 1:  # single value
                noise_level = noise_level.expand(batch_size)
            elif noise_level.dim() == 1 and noise_level.shape[0] != batch_size:  # wrong batch size
                noise_level = noise_level[0].unsqueeze(0).expand(batch_size)
            noise_channel = noise_level.unsqueeze(-1).unsqueeze(-1).expand(batch_size, 1, height, width)
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
        logits = self.outc(x)
        return logits


class ResidualBlock(nn.Module):
    """Residual block for DnCNN"""
    
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return out


class DnCNN(nn.Module):
    """DnCNN for SAR image denoising"""
    
    def __init__(self, channels=1, num_of_layers=17, num_of_filters=64, noise_conditioning=False):
        super(DnCNN, self).__init__()
        self.noise_conditioning = noise_conditioning
        input_channels = channels + 1 if noise_conditioning else channels
        
        kernel_size = 3
        padding = 1
        features = num_of_filters
        
        layers = []
        layers.append(nn.Conv2d(in_channels=input_channels, out_channels=features, 
                               kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(nn.ReLU(inplace=True))
        
        for _ in range(num_of_layers-2):
            layers.append(nn.Conv2d(in_channels=features, out_channels=features, 
                                   kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        
        layers.append(nn.Conv2d(in_channels=features, out_channels=channels, 
                               kernel_size=kernel_size, padding=padding, bias=False))
        
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x, noise_level=None):
        if self.noise_conditioning and noise_level is not None:
            # Add noise level as additional channel
            batch_size, _, height, width = x.shape
            # Ensure noise_level has the right shape
            if noise_level.dim() == 0:  # scalar
                noise_level = noise_level.unsqueeze(0).expand(batch_size)
            elif noise_level.dim() == 1 and noise_level.shape[0] == 1:  # single value
                noise_level = noise_level.expand(batch_size)
            elif noise_level.dim() == 1 and noise_level.shape[0] != batch_size:  # wrong batch size
                noise_level = noise_level[0].unsqueeze(0).expand(batch_size)
            noise_channel = noise_level.unsqueeze(-1).unsqueeze(-1).expand(batch_size, 1, height, width)
            x = torch.cat([x, noise_channel], dim=1)
        
        out = self.dncnn(x)
        return out


def create_model(model_type='unet', **kwargs):
    """Factory function to create models"""
    if model_type.lower() == 'unet':
        return UNet(**kwargs)
    elif model_type.lower() == 'dncnn':
        # Map n_channels to channels for DnCNN
        if 'n_channels' in kwargs:
            kwargs['channels'] = kwargs.pop('n_channels')
        return DnCNN(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Test the models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test U-Net
    unet = UNet(n_channels=1, noise_conditioning=True).to(device)
    x = torch.randn(2, 1, 128, 128).to(device)
    noise_level = torch.randn(2).to(device)
    output = unet(x, noise_level)
    print(f"U-Net output shape: {output.shape}")
    
    # Test DnCNN
    dncnn = DnCNN(channels=1, noise_conditioning=True).to(device)
    output = dncnn(x, noise_level)
    print(f"DnCNN output shape: {output.shape}")
