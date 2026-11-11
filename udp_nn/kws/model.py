"""
kws/model.py — DS-CNN for MLPerf Tiny Keyword Spotting

Architecture: depthwise-separable CNN (DS-CNN small variant)
Reference   : mlcommons/tiny benchmark/training/keyword_spotting/keras_model.py

Input  : (batch, 1, 49, 10)  — 49 MFCC frames × 10 coefficients
Output : (batch, 12)          — 10 keywords + silence + unknown

Topology:
  Stem : Conv2d(1→64, 10×4, stride 2×2) → BN → ReLU
  ×4   : DepthwiseConv2d(64, 3×3) → BN → ReLU
           PointwiseConv2d(64→64, 1×1) → BN → ReLU
  Head : AdaptiveAvgPool → Flatten → Linear(64→12)

Parameters: ~38 K
"""

import torch
import torch.nn as nn


class DSConvBlock(nn.Module):
    """Depthwise + Pointwise separable conv block."""

    def __init__(self, channels: int):
        super().__init__()
        self.dw = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1,
                      groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.pw = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pw(self.dw(x))


class DSCNN(nn.Module):
    """
    MLPerf Tiny DS-CNN (small) for keyword spotting.

    Args:
        num_classes : number of output classes (default 12)
        num_filters : base filter count (default 64)
    """

    def __init__(self, num_classes: int = 12, num_filters: int = 64):
        super().__init__()

        # Stem — initial large-kernel conv
        # Input : (B, 1, 49, 10)
        # Output: (B, 64, 25, 5)  [stride 2×2, padding 5×1]
        self.stem = nn.Sequential(
            nn.Conv2d(1, num_filters, kernel_size=(10, 4),
                      stride=(2, 2), padding=(5, 1), bias=False),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(inplace=True),
        )

        # Four DS blocks (no spatial downsampling)
        self.blocks = nn.Sequential(*[DSConvBlock(num_filters) for _ in range(4)])

        # Head
        self.pool    = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc      = nn.Linear(num_filters, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        x = self.pool(x)
        x = self.flatten(x)
        return self.fc(x)


def build_dscnn(num_classes: int = 12, num_filters: int = 64) -> DSCNN:
    return DSCNN(num_classes=num_classes, num_filters=num_filters)


if __name__ == "__main__":
    m = build_dscnn()
    x = torch.randn(1, 1, 49, 10)
    out = m(x)
    total = sum(p.numel() for p in m.parameters())
    print(f"Output shape : {out.shape}")   # expect (1, 12)
    print(f"Total params : {total:,}")     # expect ~38 K
