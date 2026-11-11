"""
ad/model.py — Deep AutoEncoder for MLPerf Tiny Anomaly Detection

Reference : mlcommons/tiny benchmark/training/anomaly_detection/keras_model.py
Dataset   : DCASE2020 Task 2 / ToyADMOS (ToyCar machine)

Architecture:
  Dense (640→128→128→128→128→8→128→128→128→128→640)
  ReLU activations between every layer.
  Input = 640 = 5 consecutive 128-dim log-mel frames (flattened)
  Output = reconstructed input (reconstruction loss = anomaly score)

Parameters: ~860 K
"""

import torch
import torch.nn as nn


class DenseAutoEncoder(nn.Module):
    """
    MLPerf Tiny Anomaly Detection autoencoder.

    Args:
        input_dim  : feature dimension (default 640 = 5 × 128 mel bins)
        bottleneck : bottleneck dimension (default 8)
    """

    HIDDEN = [128, 128, 128, 128]  # encoder + decoder hidden layers

    def __init__(self, input_dim: int = 640, bottleneck: int = 8):
        super().__init__()

        dims   = [input_dim] + self.HIDDEN + [bottleneck]
        dims  += list(reversed(self.HIDDEN)) + [input_dim]

        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:          # no ReLU after final output
                layers.append(nn.ReLU(inplace=True))

        self.net       = nn.Sequential(*layers)
        self.input_dim = input_dim

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return reconstruction; shape identical to input."""
        return self.net(x)

    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sample mean-squared reconstruction error (anomaly score)."""
        recon = self.forward(x)
        return ((x - recon) ** 2).mean(dim=-1)


def build_autoencoder(input_dim: int = 640, bottleneck: int = 8) -> DenseAutoEncoder:
    return DenseAutoEncoder(input_dim=input_dim, bottleneck=bottleneck)


if __name__ == "__main__":
    m = build_autoencoder()
    x = torch.randn(4, 640)
    out   = m(x)
    score = m.anomaly_score(x)
    total = sum(p.numel() for p in m.parameters())
    print(f"Input  : {x.shape}")
    print(f"Output : {out.shape}")      # expect (4, 640)
    print(f"Score  : {score.shape}")    # expect (4,)
    print(f"Params : {total:,}")        # expect ~860 K
