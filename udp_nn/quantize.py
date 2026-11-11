"""
quantize.py — INT4 Post-Training Quantization (PTQ) for ResNet-8

Maps the trained FP32 model to the UDP hardware's number format:
  Weights     : signed INT4,   range [-8, 7],   symmetric,  per-tensor scale
  Activations : unsigned INT4, range [0, 15],   asymmetric, per-tensor scale
                (after ReLU, values are always >= 0)

Method: fake quantization
  All arithmetic stays in float32, but every weight and activation is passed
  through the quantization grid (round → clamp) so the model experiences the
  same precision loss as the real INT4 hardware.  The resulting accuracy is
  therefore a faithful estimate of what the UDP array will produce.

Calibration:
  A small subset of the training set (default 1024 images) is used to measure
  the min/max of each layer's activation.  Scale factors are computed once and
  then frozen for evaluation.

Usage:
  python quantize.py                             # quantize + evaluate
  python quantize.py --weights checkpoints/best.pth
  python quantize.py --calib-batches 8          # more calibration data

Outputs:
  checkpoints/quantized_int4.pth  — fake-quantised state dict + scale factors
"""

import argparse
import os
import time
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from resnet8 import build_resnet8

# ── CIFAR-10 normalisation ────────────────────────────────────────────────────
CIFAR10_MEAN    = (0.4914, 0.4822, 0.4465)
CIFAR10_STD     = (0.2023, 0.1994, 0.2010)
CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
                   "dog",      "frog",       "horse", "ship", "truck"]

# INT4 ranges
W_BITS  = 4
A_BITS  = 4
W_QMIN  = -(2 ** (W_BITS - 1))      # -8
W_QMAX  =  (2 ** (W_BITS - 1)) - 1  #  7
A_QMIN  = 0                          #  0   (unsigned, post-ReLU)
A_QMAX  = (2 ** A_BITS) - 1          # 15


# =============================================================================
# Quantisation helpers
# =============================================================================

def symmetric_scale(tensor: torch.Tensor) -> torch.Tensor:
    """Per-tensor symmetric scale for weight quantisation."""
    return tensor.abs().max() / W_QMAX


def asymmetric_scale_zp(x_min: float, x_max: float):
    """Per-tensor asymmetric scale + zero-point for activation quantisation."""
    x_min = min(x_min, 0.0)   # ensure 0 is representable
    x_max = max(x_max, 0.0)
    scale = (x_max - x_min) / (A_QMAX - A_QMIN)
    if scale == 0:
        scale = 1e-8
    zero_point = A_QMIN - round(x_min / scale)
    zero_point = int(max(A_QMIN, min(A_QMAX, zero_point)))
    return scale, zero_point


def fake_quantize_weight(w: torch.Tensor) -> torch.Tensor:
    """Round weights through the INT4 symmetric grid (fake quant)."""
    scale = symmetric_scale(w)
    w_q   = (w / scale).round().clamp(W_QMIN, W_QMAX)
    return w_q * scale


def fake_quantize_activation(x: torch.Tensor,
                              scale: float,
                              zero_point: int) -> torch.Tensor:
    """Round activations through the INT4 unsigned grid (fake quant)."""
    x_q = (x / scale + zero_point).round().clamp(A_QMIN, A_QMAX)
    return (x_q - zero_point) * scale


# =============================================================================
# Quantised layer wrappers
# =============================================================================

class FakeQuantConv2d(nn.Module):
    """
    Conv2d with fake-quantised weights (INT4, symmetric).
    Activation quantisation is applied by a separate FakeQuantAct module
    inserted after each BN+ReLU block.
    """

    def __init__(self, conv: nn.Conv2d):
        super().__init__()
        self.conv = conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_fq = fake_quantize_weight(self.conv.weight)
        return nn.functional.conv2d(
            x, w_fq, self.conv.bias,
            self.conv.stride, self.conv.padding,
            self.conv.dilation, self.conv.groups,
        )


class FakeQuantAct(nn.Module):
    """
    Fake-quantises activations to INT4 (unsigned, asymmetric).
    Scale and zero_point are set during calibration and frozen afterwards.
    """

    def __init__(self):
        super().__init__()
        self.register_buffer("scale",      torch.tensor(1.0))
        self.register_buffer("zero_point", torch.tensor(0))
        self.calibrating = True
        self._obs_min =  float("inf")
        self._obs_max = -float("inf")

    def observe(self, x: torch.Tensor):
        self._obs_min = min(self._obs_min, x.min().item())
        self._obs_max = max(self._obs_max, x.max().item())

    def freeze(self):
        s, zp = asymmetric_scale_zp(self._obs_min, self._obs_max)
        self.scale.fill_(s)
        self.zero_point.fill_(zp)
        self.calibrating = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.calibrating:
            self.observe(x)
            return x
        return fake_quantize_activation(x, self.scale.item(),
                                        self.zero_point.item())


# =============================================================================
# Model-level quantisation
# =============================================================================

def insert_fake_quant(model: nn.Module) -> nn.Module:
    """
    Walk the model and:
      1. Replace every Conv2d with FakeQuantConv2d (weight quant).
      2. Insert a FakeQuantAct after every ReLU (activation quant).

    The walk targets the named submodules of ResNet8 directly so we
    can insert modules without breaking the forward graph.
    """
    # Replace Conv2d layers
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            parent, attr = _get_parent(model, name)
            setattr(parent, attr, FakeQuantConv2d(module))

    # Wrap each block's ReLU with a Sequential([ReLU, FakeQuantAct])
    # ResNet8 has ReLUs in: stem, stack1.relu, stack2.relu, stack3.relu
    _wrap_relus(model)

    return model


def _get_parent(model, name):
    parts  = name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def _wrap_relus(model):
    """Replace each nn.ReLU with Sequential(ReLU, FakeQuantAct)."""
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.ReLU):
            parent, attr = _get_parent(model, name)
            setattr(parent, attr,
                    nn.Sequential(OrderedDict([
                        ("relu", nn.ReLU(inplace=False)),
                        ("fqa",  FakeQuantAct()),
                    ])))


def collect_fqa_modules(model):
    """Return all FakeQuantAct modules in the model."""
    return [m for m in model.modules() if isinstance(m, FakeQuantAct)]


def calibrate(model, loader, device, num_batches: int):
    """Run calibration: feed data through the model, collect activation stats."""
    model.eval()
    fqa_mods = collect_fqa_modules(model)
    for m in fqa_mods:
        m.calibrating = True

    with torch.no_grad():
        for i, (imgs, _) in enumerate(loader):
            if i >= num_batches:
                break
            model(imgs.to(device))

    for m in fqa_mods:
        m.freeze()

    print(f"  Calibrated {len(fqa_mods)} activation quantisers "
          f"on {min(num_batches, len(loader))} batches.")


# =============================================================================
# Evaluation
# =============================================================================

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    t0 = time.time()
    for imgs, labels in loader:
        imgs   = imgs.to(device)
        preds  = model(imgs).argmax(1).cpu()
        correct += preds.eq(labels).sum().item()
        total   += labels.size(0)
    elapsed = time.time() - t0
    return 100.0 * correct / total, elapsed


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights",       default="./checkpoints/best.pth")
    parser.add_argument("--data-dir",      default="./data")
    parser.add_argument("--ckpt-dir",      default="./checkpoints")
    parser.add_argument("--calib-batches", type=int, default=8,
                        help="Number of training mini-batches for calibration")
    parser.add_argument("--batch-size",    type=int, default=128)
    parser.add_argument("--num-workers",   type=int, default=2)
    parser.add_argument("--conv-filters",  type=int, default=26)
    args = parser.parse_args()

    os.makedirs(args.ckpt_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device  : {device}")
    print(f"Weights : {args.weights}")

    # ── Load FP32 model ───────────────────────────────────────────────────────
    model  = build_resnet8(conv_filters=args.conv_filters)
    state  = torch.load(args.weights, map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state)

    # ── Insert fake-quant wrappers ────────────────────────────────────────────
    model = insert_fake_quant(model).to(device)

    # ── Data loaders ─────────────────────────────────────────────────────────
    base_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    train_ds = datasets.CIFAR10(args.data_dir, train=True,
                                download=True, transform=base_tf)
    test_ds  = datasets.CIFAR10(args.data_dir, train=False,
                                download=True, transform=base_tf)

    calib_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers)
    test_loader  = DataLoader(test_ds,  batch_size=256,
                              shuffle=False, num_workers=args.num_workers)

    # ── Calibration ───────────────────────────────────────────────────────────
    print("\nCalibrating activation scales...")
    calibrate(model, calib_loader, device, args.calib_batches)

    # ── Evaluation ────────────────────────────────────────────────────────────
    print("\nEvaluating INT4 quantised model on CIFAR-10 test set...")
    acc, elapsed = evaluate(model, test_loader, device)

    print(f"\n{'='*50}")
    print(f"  INT4 Top-1 Accuracy  : {acc:.2f}%")
    print(f"  Inference time       : {elapsed:.2f}s  (10,000 images)")
    mlperf_pass = "PASS" if acc >= 85.0 else "FAIL"
    print(f"  MLPerf Tiny target   : >=85%  →  {mlperf_pass}")
    print(f"{'='*50}")

    # ── Save quantised checkpoint ─────────────────────────────────────────────
    save_path = os.path.join(args.ckpt_dir, "quantized_int4.pth")

    # Collect scale/zero-point metadata for each FakeQuantAct
    fqa_scales = {}
    for name, m in model.named_modules():
        if isinstance(m, FakeQuantAct):
            fqa_scales[name] = {
                "scale":      m.scale.item(),
                "zero_point": m.zero_point.item(),
            }

    torch.save({
        "model_state":  model.state_dict(),
        "fqa_scales":   fqa_scales,
        "int4_acc":     acc,
        "conv_filters": args.conv_filters,
    }, save_path)
    print(f"\nQuantised checkpoint saved to: {save_path}")
    print("  (contains state dict + per-layer INT4 scale factors)")


if __name__ == "__main__":
    main()
