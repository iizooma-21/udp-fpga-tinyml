"""
resnet8.py — PyTorch port of the official MLPerf Tiny ResNet-8 (resnet_v1_eembc)

Architecture reference:
  https://github.com/mlcommons/tiny/blob/master/benchmark/training/
  image_classification/keras_model.py

Topology (conv_filters=26):
  Stem      : Conv(26, 3×3, s=1) → BN → ReLU
  Stack 1   : [Conv(26,3×3,s=1)→BN→ReLU→Conv(26,3×3,s=1)→BN] + identity → ReLU
  Stack 2   : [Conv(52,3×3,s=2)→BN→ReLU→Conv(52,3×3,s=1)→BN]
              + Conv(52,1×1,s=2) shortcut → ReLU
  Stack 3   : [Conv(104,3×3,s=2)→BN→ReLU→Conv(104,3×3,s=1)→BN]
              + Conv(104,1×1,s=2) shortcut → ReLU
  Head      : AvgPool(8×8) → Flatten → Linear(104→10)

Input : 3×32×32 (CIFAR-10)
Output: 10-class logits (softmax applied externally during training)
"""

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """One residual stack: two 3×3 convs with optional projection shortcut."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)

        # Projection shortcut when spatial size or channel count changes
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(out + self.shortcut(x))
        return out


class ResNet8(nn.Module):
    """
    MLPerf Tiny ResNet-8 (resnet_v1_eembc) in PyTorch.

    Args:
        conv_filters: base filter count (default 26, matching MLPerf Tiny spec)
        num_classes:  number of output classes (default 10 for CIFAR-10)
    """

    def __init__(self, conv_filters: int = 26, num_classes: int = 10):
        super().__init__()
        f = conv_filters

        # ── Stem ──────────────────────────────────────────────────────────────
        self.stem = nn.Sequential(
            nn.Conv2d(3, f, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(f),
            nn.ReLU(inplace=True),
        )

        # ── Stacks ────────────────────────────────────────────────────────────
        self.stack1 = ResidualBlock(f,     f,     stride=1)   # 32×32 → 32×32
        self.stack2 = ResidualBlock(f,     f * 2, stride=2)   # 32×32 → 16×16
        self.stack3 = ResidualBlock(f * 2, f * 4, stride=2)   # 16×16 →  8×8

        # ── Head ──────────────────────────────────────────────────────────────
        # After stack3 the feature map is 8×8 for CIFAR-10 input (32/2/2=8)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc      = nn.Linear(f * 4, num_classes)

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
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stack1(x)
        x = self.stack2(x)
        x = self.stack3(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


def build_resnet8(conv_filters: int = 26, num_classes: int = 10) -> ResNet8:
    """Convenience factory function."""
    return ResNet8(conv_filters=conv_filters, num_classes=num_classes)


# ── Quick sanity check ────────────────────────────────────────────────────────
if __name__ == "__main__":
    model = build_resnet8()
    dummy = torch.randn(1, 3, 32, 32)
    out   = model(dummy)
    total = sum(p.numel() for p in model.parameters())
    print(f"Output shape : {out.shape}")        # expect (1, 10)
    print(f"Total params : {total:,}")          # expect ~200k
