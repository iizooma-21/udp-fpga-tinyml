"""
vww/model.py — MobileNetV1 (width × 0.25) for MLPerf Tiny Visual Wake Words

Reference : mlcommons/tiny benchmark/training/visual_wake_words/vww_model.py
            MobileNetV1 paper (Howard et al. 2017)

Input  : (batch, 3, 96, 96)  — RGB image
Output : (batch, 2)           — person (1) / no-person (0)

Architecture (α = 0.25):
  Conv    : 3→8,  3×3, stride 2  → 48×48
  DS ×13  : depthwise (3×3) + pointwise (1×1) with channel doubling
  AvgPool : 3×3 → 1×1
  Linear  : 256→2

Parameters: ~217 K
"""

import torch
import torch.nn as nn


def _round_filters(c: int, alpha: float = 0.25) -> int:
    return max(1, int(c * alpha))


class DWBlock(nn.Module):
    """Depthwise-Separable block: DW conv → BN → ReLU → PW conv → BN → ReLU."""

    def __init__(self, in_c: int, out_c: int, stride: int = 1):
        super().__init__()
        self.dw = nn.Sequential(
            nn.Conv2d(in_c, in_c, kernel_size=3, stride=stride,
                      padding=1, groups=in_c, bias=False),
            nn.BatchNorm2d(in_c),
            nn.ReLU(inplace=True),
        )
        self.pw = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pw(self.dw(x))


class MobileNetV1(nn.Module):
    """
    MobileNetV1 with width multiplier α.

    Args:
        num_classes : 2 (person / no-person) for VWW
        alpha       : width multiplier (0.25 for MLPerf Tiny)
    """

    # (in_channels, out_channels, stride) — before applying alpha
    _CONFIG = [
        # First regular conv is handled separately
        (32,   64,  1),
        (64,  128,  2),
        (128, 128,  1),
        (128, 256,  2),
        (256, 256,  1),
        (256, 512,  2),
        (512, 512,  1),   # × 5 in original; here index 6-10
        (512, 512,  1),
        (512, 512,  1),
        (512, 512,  1),
        (512, 512,  1),
        (512, 1024, 2),
        (1024, 1024, 1),
    ]

    def __init__(self, num_classes: int = 2, alpha: float = 0.25):
        super().__init__()
        self.alpha = alpha

        first_c = _round_filters(32, alpha)  # 8

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, first_c, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(first_c),
            nn.ReLU(inplace=True),
        )

        # Build DS blocks
        layers = []
        in_c = first_c
        for (_, out_raw, stride) in self._CONFIG:
            out_c = _round_filters(out_raw, alpha)
            layers.append(DWBlock(in_c, out_c, stride))
            in_c = out_c

        self.features = nn.Sequential(*layers)
        self.pool     = nn.AdaptiveAvgPool2d(1)
        self.flatten  = nn.Flatten()
        self.fc       = nn.Linear(in_c, num_classes)

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
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.features(x)
        x = self.pool(x)
        x = self.flatten(x)
        return self.fc(x)

    # ── convenience: per-block forward for UDP sim ──────────────────────────
    def forward_blocks(self, x: torch.Tensor):
        """Yield intermediate tensors for calibration."""
        x = self.stem(x)
        yield "stem", x
        for i, blk in enumerate(self.features):
            x = blk(x)
            yield f"block{i}", x


def build_mobilenet_vww(num_classes: int = 2, alpha: float = 0.25) -> MobileNetV1:
    return MobileNetV1(num_classes=num_classes, alpha=alpha)


if __name__ == "__main__":
    m = build_mobilenet_vww()
    x = torch.randn(1, 3, 96, 96)
    out = m(x)
    total = sum(p.numel() for p in m.parameters())
    print(f"Output shape : {out.shape}")   # expect (1, 2)
    print(f"Total params : {total:,}")     # expect ~217 K
