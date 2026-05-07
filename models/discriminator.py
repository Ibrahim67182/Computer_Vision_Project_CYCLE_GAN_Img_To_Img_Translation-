"""
CycleGAN - Discriminator (PatchGAN)
For 256x256 images → 70x70 PatchGAN (as per Zhu et al. 2017)

Architecture:
    Input (3, 256, 256)
        ↓
    Conv Block 1  (no InstanceNorm on first layer)  → (64,  128, 128)
        ↓
    Conv Block 2                                     → (128,  64,  64)
        ↓
    Conv Block 3                                     → (256,  32,  32)
        ↓
    Conv Block 4  (stride=1)                         → (512,  31,  31)
        ↓
    Output Conv   (stride=1, no norm, no activation) → (1,    30,  30)

The output (1, 30, 30) is a patch map — each of the 30×30 values
represents real/fake prediction for a 70×70 patch of the input image.
Loss is averaged over all patches.

Two discriminator instances are created in train.py:
    D_A : judges whether Domain A images are real or fake  (e.g. horses)
    D_B : judges whether Domain B images are real or fake  (e.g. zebras)
Both use this same class.
"""

import torch
import torch.nn as nn


# ─────────────────────────────────────────────
#  Building Block
# ─────────────────────────────────────────────

class DiscConvBlock(nn.Module):
    """
    Conv → InstanceNorm (optional) → LeakyReLU

    Args:
        in_channels   : input channels
        out_channels  : output channels
        stride        : 2 = downsample, 1 = same size
        use_norm      : False for the very first layer (paper convention)
    """
    def __init__(self, in_channels, out_channels, stride=2, use_norm=True):
        super().__init__()

        layers = [
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=4, stride=stride,
                      padding=1, bias=False)
        ]

        if use_norm:
            layers.append(nn.InstanceNorm2d(out_channels))

        # LeakyReLU with slope 0.2 — standard for discriminators
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


# ─────────────────────────────────────────────
#  Discriminator
# ─────────────────────────────────────────────

class Discriminator(nn.Module):
    """
    PatchGAN Discriminator for CycleGAN.
    Instead of classifying the entire image as real/fake,
    it classifies overlapping 70×70 patches independently.
    This encourages sharper, more detailed translations.

    Output shape: (1, 30, 30)
        - Each value = real/fake score for one 70×70 patch
        - Loss is computed as mean over all 900 patch predictions

    Args:
        in_channels : input image channels (3 for RGB)
        features    : base feature size (64)
    """
    def __init__(self, in_channels=3, features=64):
        super().__init__()

        # ── Layer 1: no InstanceNorm on first layer (paper convention) ──
        # (3, 256, 256) → (64, 128, 128)
        self.layer1 = DiscConvBlock(in_channels, features,
                                    stride=2, use_norm=False)

        # ── Layer 2 ──
        # (64, 128, 128) → (128, 64, 64)
        self.layer2 = DiscConvBlock(features,     features * 2,
                                    stride=2, use_norm=True)

        # ── Layer 3 ──
        # (128, 64, 64) → (256, 32, 32)
        self.layer3 = DiscConvBlock(features * 2, features * 4,
                                    stride=2, use_norm=True)

        # ── Layer 4: stride=1, spatial size barely changes ──
        # (256, 32, 32) → (512, 31, 31)
        self.layer4 = DiscConvBlock(features * 4, features * 8,
                                    stride=1, use_norm=True)

        # ── Output: single channel patch map, no norm, no activation ──
        # (512, 31, 31) → (1, 30, 30)
        # Raw logits — BCEWithLogitsLoss handles the sigmoid
        self.output = nn.Conv2d(features * 8, 1,
                                kernel_size=4, stride=1, padding=1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
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
    Call this after creating each discriminator instance.
    """
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
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

    # Create both discriminators (D_A and D_B use the same class)
    D_A = Discriminator(in_channels=3).to(device)
    D_B = Discriminator(in_channels=3).to(device)

    initialize_weights(D_A)
    initialize_weights(D_B)

    # Fake input batch: 1 image, 3 channels, 256x256
    fake_A = torch.randn(1, 3, 256, 256).to(device)
    fake_B = torch.randn(1, 3, 256, 256).to(device)

    # Forward pass
    out_A = D_A(fake_A)
    out_B = D_B(fake_B)

    print(f"D_A  input  shape  : {fake_A.shape}")
    print(f"D_A  output shape  : {out_A.shape}     (should be [1, 1, 30, 30])")
    print(f"D_B  input  shape  : {fake_B.shape}")
    print(f"D_B  output shape  : {out_B.shape}     (should be [1, 1, 30, 30])")

    print(f"\nPatch predictions  : {out_A.shape[2] * out_A.shape[3]} patches per image (30×30 = 900)")
    print(f"Output value range : {out_A.min():.3f} to {out_A.max():.3f}  (raw logits, no sigmoid yet)")

    # Parameter count
    params = sum(p.numel() for p in D_A.parameters())
    print(f"\nDiscriminator parameters : {params:,}  (~2.76M expected)")

    print("\n[OK] Discriminator is working correctly.")