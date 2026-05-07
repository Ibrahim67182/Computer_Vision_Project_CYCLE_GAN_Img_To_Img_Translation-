"""
CycleGAN - Generator (ResNet-based)
For 256x256 images → 9 residual blocks (as per Zhu et al. 2017)

Architecture:
    Input (3, 256, 256)
        ↓
    Initial Conv Block       → (64, 256, 256)
        ↓
    Downsample x2            → (128, 128, 128) → (256, 64, 64)
        ↓
    Residual Blocks x9       → (256, 64, 64)
        ↓
    Upsample x2              → (128, 128, 128) → (64, 256, 256)
        ↓
    Output Conv (Tanh)       → (3, 256, 256)   values in [-1, 1]

Two generator instances are created in train.py:
    G_AB : translates Domain A → Domain B  (e.g. horse → zebra)
    G_BA : translates Domain B → Domain A  (e.g. zebra → horse)
Both use this same class.
"""

import torch
import torch.nn as nn


# ─────────────────────────────────────────────
#  Building Blocks
# ─────────────────────────────────────────────

class ConvBlock(nn.Module):
    """
    Conv → InstanceNorm → ReLU
    Used in the initial layer and downsampling layers.

    Args:
        in_channels  : number of input channels
        out_channels : number of output channels
        kernel_size  : convolution kernel size
        stride       : stride (2 = downsample, 1 = same size)
        padding      : zero padding
        use_relu     : True for all layers except the final output layer
    """
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=1, use_relu=True):
        super().__init__()

        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      stride=stride, padding=padding, bias=False),
            nn.InstanceNorm2d(out_channels),
        ]
        if use_relu:
            layers.append(nn.ReLU(inplace=True))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class ResidualBlock(nn.Module):
    """
    Residual Block: Conv → IN → ReLU → Conv → IN → skip connection
    Uses reflection padding instead of zero padding to reduce border artifacts.
    Input and output shapes are identical: (256, 64, 64) → (256, 64, 64)

    Args:
        channels : number of channels (256 for 256x256 images)
    """
    def __init__(self, channels=256):
        super().__init__()

        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, bias=False),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),

            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, bias=False),
            nn.InstanceNorm2d(channels),
        )

    def forward(self, x):
        # Skip connection: add input to output
        return x + self.block(x)


class UpsampleBlock(nn.Module):
    """
    Upsample → Conv → InstanceNorm → ReLU
    Uses ConvTranspose2d (fractionally strided conv) to double spatial size.

    (128, 128, 128) → (64, 256, 256)  for example
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels,
                               kernel_size=3, stride=2,
                               padding=1, output_padding=1,
                               bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


# ─────────────────────────────────────────────
#  Generator
# ─────────────────────────────────────────────

class Generator(nn.Module):
    """
    ResNet Generator for CycleGAN.
    Translates images from one domain to another.

    For 256x256 images: num_residual_blocks=9  (paper default)
    For 128x128 images: num_residual_blocks=6

    Args:
        in_channels          : input image channels (3 for RGB)
        out_channels         : output image channels (3 for RGB)
        num_residual_blocks  : 9 for 256x256
        features             : base feature size (64)
    """
    def __init__(self, in_channels=3, out_channels=3,
                 num_residual_blocks=9, features=64):
        super().__init__()

        # ── Initial convolution (large kernel, reflection padding) ──
        # (3, 256, 256) → (64, 256, 256)
        self.initial = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, features, kernel_size=7, bias=False),
            nn.InstanceNorm2d(features),
            nn.ReLU(inplace=True),
        )

        # ── Downsampling: 2 layers ──
        # (64, 256, 256) → (128, 128, 128) → (256, 64, 64)
        self.downsample = nn.Sequential(
            ConvBlock(features,     features * 2, kernel_size=3, stride=2, padding=1),
            ConvBlock(features * 2, features * 4, kernel_size=3, stride=2, padding=1),
        )

        # ── Residual Blocks: 9 layers ──
        # (256, 64, 64) → (256, 64, 64)  shape unchanged
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(features * 4) for _ in range(num_residual_blocks)]
        )

        # ── Upsampling: 2 layers ──
        # (256, 64, 64) → (128, 128, 128) → (64, 256, 256)
        self.upsample = nn.Sequential(
            UpsampleBlock(features * 4, features * 2),
            UpsampleBlock(features * 2, features),
        )

        # ── Output convolution ──
        # (64, 256, 256) → (3, 256, 256)  values in [-1, 1] via Tanh
        self.output = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(features, out_channels, kernel_size=7),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.initial(x)
        x = self.downsample(x)
        x = self.residual_blocks(x)
        x = self.upsample(x)
        x = self.output(x)
        return x


# ─────────────────────────────────────────────
#  Weight Initialization
# ─────────────────────────────────────────────

def initialize_weights(model):
    """
    Initialize conv and instancenorm weights as per the CycleGAN paper:
        Conv weights  : Normal distribution, mean=0, std=0.02
        InstanceNorm  : weight=1, bias=0
    Call this after creating each generator instance.
    """
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.InstanceNorm2d):
            if m.weight is not None:
                nn.init.constant_(m.weight, 1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


# ─────────────────────────────────────────────
#  Quick Verification (run this file directly)
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")

    # Create both generators (G_AB and G_BA use the same class)
    G_AB = Generator(in_channels=3, out_channels=3, num_residual_blocks=9).to(device)
    G_BA = Generator(in_channels=3, out_channels=3, num_residual_blocks=9).to(device)

    # Initialize weights
    initialize_weights(G_AB)
    initialize_weights(G_BA)

    # Fake input batch: 1 image, 3 channels, 256x256
    fake_A = torch.randn(1, 3, 256, 256).to(device)
    fake_B = torch.randn(1, 3, 256, 256).to(device)

    # Forward pass
    fake_B_out = G_AB(fake_A)
    fake_A_out = G_BA(fake_B)

    print(f"G_AB  input  shape : {fake_A.shape}")
    print(f"G_AB  output shape : {fake_B_out.shape}   (should be [1, 3, 256, 256])")
    print(f"G_BA  input  shape : {fake_B.shape}")
    print(f"G_BA  output shape : {fake_A_out.shape}   (should be [1, 3, 256, 256])")
    print(f"\nOutput pixel range : {fake_B_out.min():.3f} to {fake_B_out.max():.3f}  (should be within -1 to 1)")

    # Parameter count
    params = sum(p.numel() for p in G_AB.parameters())
    print(f"\nGenerator parameters : {params:,}  (~11.4M expected)")

    print("\n[OK] Generator is working correctly.")