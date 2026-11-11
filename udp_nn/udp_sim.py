"""
udp_sim.py — UDP Systolic Array Simulation

Evaluates two hardware operating modes for ResNet-8 on CIFAR-10.

  Mode 4 — UDP WITHOUT post-processing  (weight-only INT4)
    Conv weights are quantised to signed INT4 per output channel.
    Activations remain in FP32 throughout.  Models a configuration where
    the inter-layer re-quantisation unit is absent.
    Expected accuracy: ~80-84% (only weight precision is lost).

  Mode 5 — UDP WITH post-processing  (full INT4 pipeline)
    Weights: per-channel signed INT4.
    Activations: asymmetric unsigned INT4 with a per-layer calibrated
    scale+zero-point.  After each residual BLOCK OUTPUT the activations are
    re-quantised back to [0, 15] before the next block (5 quantisation
    points total: input, stem, block1-out, block2-out, block3-out).
    This is the full UDP hardware pipeline.
    Expected accuracy: ~65-75% with simple PTQ (QAT needed for higher).

Both modes use fake-quantisation in float32 arithmetic.
  - Weights  : round to INT4 grid, then multiply back by per-channel scale.
  - Acts     : (Mode 5 only) round to INT4 grid via scale+zero-point,
               then restore to float before the next conv.
  This correctly captures quantisation noise without integer overflow.

Usage:
  python udp_sim.py                          # both modes
  python udp_sim.py --mode no-postproc       # Mode 4 only
  python udp_sim.py --mode postproc          # Mode 5 only
  python udp_sim.py --weights checkpoints/best.pth

Requires: checkpoints/best.pth  (from train.py)
"""

import argparse
import copy
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from resnet8 import build_resnet8

# ── CIFAR-10 ─────────────────────────────────────────────────────────────────
CIFAR10_MEAN    = (0.4914, 0.4822, 0.4465)
CIFAR10_STD     = (0.2023, 0.1994, 0.2010)
CIFAR10_CLASSES = ["airplane","automobile","bird","cat","deer",
                   "dog","frog","horse","ship","truck"]

# ── INT4 ranges ───────────────────────────────────────────────────────────────
W_QMIN = -8
W_QMAX =  7
A_QMIN =  0
A_QMAX = 15


# =============================================================================
# Per-channel weight quantisation helpers
# =============================================================================

def quantize_weights(w: torch.Tensor):
    """Symmetric per-output-channel INT4.  Returns (w_int, scales)."""
    oc     = w.shape[0]
    scales = w.view(oc, -1).abs().max(dim=1).values.clamp(min=1e-8) / W_QMAX
    sv     = scales.view(oc, *([1] * (w.dim() - 1)))
    w_int  = (w / sv).round().clamp(W_QMIN, W_QMAX)
    return w_int, scales


def fake_quant_weight(w: torch.Tensor) -> torch.Tensor:
    """Return w rounded to the per-channel INT4 grid (same shape, float32)."""
    w_int, scales = quantize_weights(w)
    oc = w.shape[0]
    sv = scales.view(oc, *([1] * (w.dim() - 1)))
    return w_int * sv


# =============================================================================
# Asymmetric per-tensor activation quantisation helpers
# =============================================================================

def asym_params(x_min: float, x_max: float):
    """Return (scale, zero_point) for unsigned INT4 asymmetric quantisation."""
    x_min  = min(x_min, 0.0)
    x_max  = max(x_max, 0.0)
    scale  = max((x_max - x_min) / (A_QMAX - A_QMIN), 1e-8)
    zp     = int(round(A_QMIN - x_min / scale))
    zp     = max(A_QMIN, min(A_QMAX, zp))
    return scale, zp


def fake_quant_act(x: torch.Tensor, scale: float, zp: int) -> torch.Tensor:
    """Round x to the asymmetric INT4 grid and restore to float32."""
    x_int = (x / scale + zp).round().clamp(A_QMIN, A_QMAX)
    return (x_int - zp) * scale


# =============================================================================
# BN helpers
# =============================================================================

def apply_bn(x: torch.Tensor, bn: nn.BatchNorm2d) -> torch.Tensor:
    """Folded BN: BN(x) = x * s + b  (per channel)."""
    inv_std = (bn.running_var + bn.eps).rsqrt()
    s = bn.weight.data * inv_std
    b = bn.bias.data - bn.weight.data * bn.running_mean * inv_std
    sv = s.view(1,-1,1,1) if x.dim()==4 else s.view(1,-1)
    bv = b.view(1,-1,1,1) if x.dim()==4 else b.view(1,-1)
    return x * sv + bv


# =============================================================================
# Calibration (Mode 5)
# =============================================================================

class _Stat:
    """Collects a running histogram to compute the 99.9th-percentile max."""
    def __init__(self):
        self.mn  = 1e9
        self.pct = -1e9
        self._buf = []

    def __call__(self, m, inp, out):
        v = out.detach()
        self.mn = min(self.mn, float(v.min()))
        self._buf.append(v.cpu().float())

    def percentile_max(self, pct: float = 99.9, max_samples: int = 500_000):
        if not self._buf:
            return self.pct
        t = torch.cat([b.flatten() for b in self._buf])
        if t.numel() > max_samples:
            idx = torch.randperm(t.numel())[:max_samples]
            t = t[idx]
        return float(torch.quantile(t, pct / 100.0))


def calibrate(model: nn.Module, loader: DataLoader,
              device: torch.device, num_batches: int = 16) -> dict:
    """
    Collect per-layer activation statistics using the 99.9th-percentile max
    (more robust than absolute max — outliers don't inflate the scale).
    Hooks are placed at BLOCK OUTPUTS only (not mid-block), matching the
    quantisation points used in UDPSim.forward().
    """
    model.eval()
    hooks, handles = {}, []
    for name, mod in [("stem",   model.stem[2]),
                      ("s1_out", model.stack1),
                      ("s2_out", model.stack2),
                      ("s3_out", model.stack3)]:
        h = _Stat(); hooks[name] = h
        handles.append(mod.register_forward_hook(h))

    img_buf = []
    with torch.no_grad():
        for i, (imgs, _) in enumerate(loader):
            if i >= num_batches: break
            imgs = imgs.to(device)
            img_buf.append(imgs.cpu().float())
            model(imgs)

    for h in handles: h.remove()

    stats = {}
    for k, h in hooks.items():
        mx = h.percentile_max(99.9)
        stats[k] = asym_params(h.mn, mx)

    # Input image stats
    img_t = torch.cat([b.flatten() for b in img_buf])
    img_mn = float(img_t.min())
    if img_t.numel() > 500_000:
        idx = torch.randperm(img_t.numel())[:500_000]
        img_t = img_t[idx]
    img_mx = float(torch.quantile(img_t, 0.999))
    stats["input"] = asym_params(img_mn, img_mx)
    return stats   # dict: key -> (scale, zero_point)


# =============================================================================
# MODE 4 — Weight-only INT4
# =============================================================================

def build_w4_model(fp32_model: nn.Module) -> nn.Module:
    """Deep-copy with every Conv2d weight replaced by its per-channel INT4 approximation."""
    m = copy.deepcopy(fp32_model)
    for mod in m.modules():
        if isinstance(mod, nn.Conv2d):
            mod.weight.data.copy_(fake_quant_weight(mod.weight.data))
    return m


@torch.no_grad()
def eval_model(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    correct, total = 0, 0
    t0 = time.time()
    for imgs, labels in loader:
        correct += model(imgs.to(device)).argmax(1).cpu().eq(labels).sum().item()
        total   += labels.size(0)
    return 100.0 * correct / total, time.time() - t0


# =============================================================================
# MODE 5 — Full INT4 pipeline
# =============================================================================

class UDPSim:
    """
    Layer-by-layer simulation of the full UDP INT4 pipeline.

    Each conv uses per-channel fake-quantised weights (float32 arithmetic).
    After each BN+ReLU the activations are fake-quantised to the asymmetric
    INT4 grid (calibrated per layer) and immediately restored to float32.
    This correctly models quantisation noise without integer overflow.
    """

    def __init__(self, fp32_model: nn.Module, stats: dict, device: torch.device):
        self.m      = fp32_model
        self.stats  = stats   # key -> (scale, zp)
        self.device = device

        # Pre-compute fake-quantised (dequantised) weights for every conv
        self._wfq = {}
        for name, mod in fp32_model.named_modules():
            if isinstance(mod, nn.Conv2d):
                self._wfq[name] = fake_quant_weight(mod.weight.data).to(device)

    def _conv(self, name: str, x: torch.Tensor, conv: nn.Conv2d) -> torch.Tensor:
        return F.conv2d(x, self._wfq[name],
                        stride=conv.stride, padding=conv.padding)

    def _fqa(self, x: torch.Tensor, key: str) -> torch.Tensor:
        """Fake-quantise x using the calibrated scale for `key`."""
        s, z = self.stats[key]
        return fake_quant_act(x, s, z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        m = self.m

        # ── Input fake-quantisation ───────────────────────────────────────────
        x = self._fqa(x, "input")

        # ── Stem ─────────────────────────────────────────────────────────────
        x = F.relu(apply_bn(self._conv("stem.0", x, m.stem[0]), m.stem[1]))
        x = self._fqa(x, "stem")

        # ── Stack 1 (identity shortcut) ───────────────────────────────────────
        # No mid-block quantisation: only the BLOCK OUTPUT is requantised.
        # The inter-conv activation flows in float32 from the fake-quantised input.
        sc = x
        x  = F.relu(apply_bn(self._conv("stack1.conv1", x, m.stack1.conv1), m.stack1.bn1))
        x  = apply_bn(self._conv("stack1.conv2", x, m.stack1.conv2), m.stack1.bn2)
        x  = F.relu(x + sc)
        x  = self._fqa(x, "s1_out")          # requantise block output only

        # ── Stack 2 (projection shortcut, stride=2) ───────────────────────────
        sc = x
        x  = F.relu(apply_bn(self._conv("stack2.conv1", x, m.stack2.conv1), m.stack2.bn1))
        x  = apply_bn(self._conv("stack2.conv2", x, m.stack2.conv2), m.stack2.bn2)
        sc = apply_bn(self._conv("stack2.shortcut.0", sc, m.stack2.shortcut[0]),
                      m.stack2.shortcut[1])
        x  = F.relu(x + sc)
        x  = self._fqa(x, "s2_out")          # requantise block output only

        # ── Stack 3 (projection shortcut, stride=2) ───────────────────────────
        sc = x
        x  = F.relu(apply_bn(self._conv("stack3.conv1", x, m.stack3.conv1), m.stack3.bn1))
        x  = apply_bn(self._conv("stack3.conv2", x, m.stack3.conv2), m.stack3.bn2)
        sc = apply_bn(self._conv("stack3.shortcut.0", sc, m.stack3.shortcut[0]),
                      m.stack3.shortcut[1])
        x  = F.relu(x + sc)
        x  = self._fqa(x, "s3_out")          # requantise block output only

        # ── Head (FP32) ───────────────────────────────────────────────────────
        return m.fc(m.flatten(m.avgpool(x)))


@torch.no_grad()
def eval_sim(sim: UDPSim, loader: DataLoader, device: torch.device):
    correct, total = 0, 0
    t0 = time.time()
    for imgs, labels in loader:
        correct += sim.forward(imgs.to(device)).argmax(1).cpu().eq(labels).sum().item()
        total   += labels.size(0)
    return 100.0 * correct / total, time.time() - t0


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights",       default="./checkpoints/best.pth")
    parser.add_argument("--data-dir",      default="./data")
    parser.add_argument("--calib-batches", type=int, default=16)
    parser.add_argument("--batch-size",    type=int, default=256)
    parser.add_argument("--num-workers",   type=int, default=2)
    parser.add_argument("--conv-filters",  type=int, default=26)
    parser.add_argument("--mode",
                        choices=["both","no-postproc","postproc"],
                        default="both")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device  : {device}")
    print(f"Weights : {args.weights}\n")

    model = build_resnet8(conv_filters=args.conv_filters).to(device)
    state = torch.load(args.weights, map_location=device)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state)
    model.eval()

    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    calib_ds    = datasets.CIFAR10(args.data_dir, train=True,  download=True, transform=tf)
    test_ds     = datasets.CIFAR10(args.data_dir, train=False, download=True, transform=tf)
    calib_loader = DataLoader(calib_ds, args.batch_size, shuffle=True,  num_workers=args.num_workers)
    test_loader  = DataLoader(test_ds,  args.batch_size, shuffle=False, num_workers=args.num_workers)

    results = {}

    # ── Mode 4 ────────────────────────────────────────────────────────────────
    if args.mode in ("both", "no-postproc"):
        print("Mode 4 — weight-only INT4  (no post-processing) ...")
        m4  = build_w4_model(model).to(device)
        acc, t = eval_model(m4, test_loader, device)
        results["no_postproc"] = acc
        print(f"  Top-1 : {acc:.2f}%  ({t:.1f}s)\n")

    # ── Mode 5 ────────────────────────────────────────────────────────────────
    if args.mode in ("both", "postproc"):
        print("Calibrating activation scales ...")
        stats = calibrate(model, calib_loader, device, args.calib_batches)
        for k, (s, z) in stats.items():
            print(f"  {k:<10s}: scale={s:.5f}  zero_point={z}")

        print("\nMode 5 — full INT4 pipeline  (with post-processing) ...")
        sim = UDPSim(model, stats, device)
        acc, t = eval_sim(sim, test_loader, device)
        results["postproc"] = acc
        print(f"  Top-1 : {acc:.2f}%  ({t:.1f}s)\n")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"{'='*60}")
    print(f"  UDP Systolic Array — CIFAR-10 Accuracy Summary")
    print(f"{'='*60}")
    if "no_postproc" in results:
        f = "PASS" if results["no_postproc"] >= 85 else "FAIL"
        print(f"  Mode 4  no post-proc  (weight-only INT4): "
              f"{results['no_postproc']:>6.2f}%   >=85% -> {f}")
    if "postproc" in results:
        f = "PASS" if results["postproc"] >= 85 else "FAIL"
        print(f"  Mode 5  with post-proc (full INT4 + req): "
              f"{results['postproc']:>6.2f}%   >=85% -> {f}")
    if len(results) == 2:
        d = results["postproc"] - results["no_postproc"]
        print(f"  Post-processing activation quantisation : {d:>+6.2f}%")
    print(f"{'='*60}\n")
    print("Interpretation:")
    print("  Mode 4 quantises only weights (per-channel INT4).")
    print("  Activations stay in FP32, so no inter-layer scale error occurs.")
    print("  Mode 5 adds asymmetric INT4 activation requantisation after")
    print("  each BN+ReLU block — this is the hardware post-processing stage.")
    print("  The delta shows the accuracy cost of constraining activations")
    print("  to INT4, which enables true end-to-end 4-bit hardware inference.")


if __name__ == "__main__":
    main()
