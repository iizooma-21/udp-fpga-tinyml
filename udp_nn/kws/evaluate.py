"""
kws/evaluate.py — Full evaluation pipeline for MLPerf Tiny KWS (DS-CNN)

Evaluates:
  1. FP32 accuracy                  (direct checkpoint)
  2. Naive INT4 PTQ accuracy         (per-tensor weight + act fake-quant)
  3. UDP Mode 4 — weight-only INT4   (INT4 weights, FP32 activations)
  4. UDP Mode 5 — full INT4 pipeline (INT4 weights + activations, fake-quant)

Usage:
  python kws/evaluate.py --ckpt checkpoints/kws_best.pth

Outputs a summary table matching the IC benchmark format.
"""

import argparse
import os
import sys
from copy import deepcopy
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from kws.model import build_dscnn, DSCNN
from kws.dataset import SpeechCommandsDataset, CLASSES

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── INT4 helpers (identical to IC pipeline) ─────────────────────────────────

def fake_quant(x: torch.Tensor, bits: int = 4,
               per_channel: bool = False, ch_dim: int = 0) -> torch.Tensor:
    n = 2 ** (bits - 1)          # 8 for int4
    if per_channel:
        shape = [-1 if i == ch_dim else 1 for i in range(x.dim())]
        scale = x.abs().amax(dim=[i for i in range(x.dim()) if i != ch_dim],
                              keepdim=True)
        scale = scale.view(shape)
    else:
        scale = x.abs().max()
    scale = scale.clamp(min=1e-8) / n
    return (x / scale).round().clamp(-n, n - 1) * scale


def fake_quant_sym(x: torch.Tensor, bits: int = 4,
                   zero: float = 0.0) -> torch.Tensor:
    """Symmetric per-tensor fake-quant (for activations)."""
    n = 2 ** (bits - 1)
    scale = x.abs().max().clamp(min=1e-8) / n
    return (x / scale).round().clamp(-n, n - 1) * scale


# ─── FP32 evaluation ─────────────────────────────────────────────────────────

@torch.no_grad()
def eval_accuracy(model: nn.Module, loader: DataLoader) -> float:
    model.eval()
    correct, total = 0, 0
    for feats, labels in loader:
        correct += model(feats.to(DEVICE)).argmax(1).cpu().eq(labels).sum().item()
        total   += labels.size(0)
    return 100.0 * correct / total


# ─── Naive INT4 PTQ ──────────────────────────────────────────────────────────

class NaiveINT4DSCNN(nn.Module):
    """Wraps DS-CNN with per-tensor fake-quant on all conv + linear weights."""

    def __init__(self, base: DSCNN, calib_loader: DataLoader,
                 n_calib: int = 8, bits: int = 4):
        super().__init__()
        self.m    = deepcopy(base)
        self.bits = bits

        # Calibrate activation ranges on first n_calib batches
        self._act_stats: Dict[str, torch.Tensor] = {}
        self._calibrate(calib_loader, n_calib)

    def _calibrate(self, loader: DataLoader, n: int):
        self.m.eval()
        with torch.no_grad():
            for i, (feats, _) in enumerate(loader):
                if i >= n:
                    break
                feats = feats.to(DEVICE)
                # Collect stats at input, after stem, after each block
                act = feats
                self._update("input", act)
                act = self.m.stem(act)
                self._update("stem",  act)
                for j, blk in enumerate(self.m.blocks):
                    act = blk(act)
                    self._update(f"block{j}", act)

    def _update(self, key: str, t: torch.Tensor):
        v = t.abs().max().item()
        if key not in self._act_stats:
            self._act_stats[key] = v
        else:
            self._act_stats[key] = max(self._act_stats[key], v)

    def _fq_w(self, w):
        return fake_quant(w, self.bits, per_channel=True, ch_dim=0)

    def _fq_a(self, x, key):
        bits = self.bits
        n    = 2 ** (bits - 1)
        v    = self._act_stats.get(key, x.abs().max().item())
        scale = max(v, 1e-8) / n
        return (x / scale).round().clamp(-n, n - 1) * scale

    def forward(self, x):
        m = self.m
        x = self._fq_a(x, "input")

        # Stem
        x = F.conv2d(x, self._fq_w(m.stem[0].weight),
                     stride=m.stem[0].stride, padding=m.stem[0].padding)
        x = m.stem[1](x)           # BN
        x = m.stem[2](x)           # ReLU
        x = self._fq_a(x, "stem")

        for j, blk in enumerate(m.blocks):
            dw, pw = blk.dw, blk.pw
            # Depthwise
            x = F.conv2d(x, self._fq_w(dw[0].weight),
                         padding=dw[0].padding, groups=dw[0].groups)
            x = dw[1](x); x = dw[2](x)
            # Pointwise
            x = F.conv2d(x, self._fq_w(pw[0].weight))
            x = pw[1](x); x = pw[2](x)
            x = self._fq_a(x, f"block{j}")

        x = m.pool(x)
        x = m.flatten(x)
        x = F.linear(x, self._fq_w(m.fc.weight), m.fc.bias)
        return x


# ─── UDP Mode 4 (weight-only INT4) ───────────────────────────────────────────

class UDPMode4DSCNN(nn.Module):
    """Weight-only INT4 fake-quant; activations remain FP32."""

    def __init__(self, base: DSCNN, bits: int = 4):
        super().__init__()
        self.m    = deepcopy(base)
        self.bits = bits

    def _fq_w(self, w):
        return fake_quant(w, self.bits, per_channel=True, ch_dim=0)

    def forward(self, x):
        m = self.m
        x = F.conv2d(x, self._fq_w(m.stem[0].weight),
                     stride=m.stem[0].stride, padding=m.stem[0].padding)
        x = m.stem[1](x); x = m.stem[2](x)

        for blk in m.blocks:
            dw, pw = blk.dw, blk.pw
            x = F.conv2d(x, self._fq_w(dw[0].weight),
                         padding=dw[0].padding, groups=dw[0].groups)
            x = dw[1](x); x = dw[2](x)
            x = F.conv2d(x, self._fq_w(pw[0].weight))
            x = pw[1](x); x = pw[2](x)

        x = m.pool(x); x = m.flatten(x)
        x = F.linear(x, self._fq_w(m.fc.weight), m.fc.bias)
        return x


# ─── UDP Mode 5 (full INT4 pipeline) ─────────────────────────────────────────

class UDPMode5DSCNN(nn.Module):
    """
    Full INT4 fake-quant simulation.
    Block-output activation scale calibrated from calibration data (99.9th pct).
    """

    def __init__(self, base: DSCNN, calib_loader: DataLoader,
                 n_calib: int = 8, bits: int = 4):
        super().__init__()
        self.m    = deepcopy(base)
        self.bits = bits
        self._scales: Dict[str, float] = {}
        self._calibrate(calib_loader, n_calib)

    def _calibrate(self, loader: DataLoader, n: int):
        self.m.eval()
        buf: Dict[str, list] = {}
        with torch.no_grad():
            for i, (feats, _) in enumerate(loader):
                if i >= n:
                    break
                feats = feats.to(DEVICE)
                act   = feats
                buf.setdefault("input", []).append(act.abs().flatten())
                act = self.m.stem(act)
                buf.setdefault("stem",  []).append(act.abs().flatten())
                for j, blk in enumerate(self.m.blocks):
                    act = blk(act)
                    buf.setdefault(f"block{j}", []).append(act.abs().flatten())

        n_int = 2 ** (self.bits - 1)
        for key, tensors in buf.items():
            all_vals = torch.cat(tensors)
            pct      = min(all_vals.numel(), 500_000)
            if all_vals.numel() > pct:
                idx      = torch.randperm(all_vals.numel())[:pct]
                all_vals = all_vals[idx]
            self._scales[key] = torch.quantile(
                all_vals, 0.999).item() / n_int

    def _fq_w(self, w):
        return fake_quant(w, self.bits, per_channel=True, ch_dim=0)

    def _fq_a(self, x, key):
        n     = 2 ** (self.bits - 1)
        scale = self._scales.get(key, x.abs().max().item() / n)
        scale = max(scale, 1e-8)
        return (x / scale).round().clamp(-n, n - 1) * scale

    def forward(self, x):
        m = self.m
        x = self._fq_a(x, "input")
        x = F.conv2d(x, self._fq_w(m.stem[0].weight),
                     stride=m.stem[0].stride, padding=m.stem[0].padding)
        x = m.stem[1](x); x = m.stem[2](x)
        x = self._fq_a(x, "stem")

        for j, blk in enumerate(m.blocks):
            dw, pw = blk.dw, blk.pw
            x = F.conv2d(x, self._fq_w(dw[0].weight),
                         padding=dw[0].padding, groups=dw[0].groups)
            x = dw[1](x); x = dw[2](x)
            x = F.conv2d(x, self._fq_w(pw[0].weight))
            x = pw[1](x); x = pw[2](x)
            x = self._fq_a(x, f"block{j}")

        x = m.pool(x); x = m.flatten(x)
        x = F.linear(x, self._fq_w(m.fc.weight), m.fc.bias)
        return x


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",        required=True,
                        help="Path to kws_best.pth")
    parser.add_argument("--data-dir",    default="./data/speech_commands")
    parser.add_argument("--calib-batches", type=int, default=8)
    parser.add_argument("--batch-size",  type=int,   default=256)
    parser.add_argument("--num-workers", type=int,   default=2)
    args = parser.parse_args()

    print(f"Device  : {DEVICE}")

    # Load checkpoint
    ckpt = torch.load(args.ckpt, map_location=DEVICE)
    num_filters = ckpt.get("num_filters", 64)
    num_classes = ckpt.get("num_classes", len(CLASSES))

    fp32 = build_dscnn(num_classes=num_classes,
                       num_filters=num_filters).to(DEVICE)
    fp32.load_state_dict(ckpt["model"])
    fp32.eval()

    # Test & calibration loaders
    test_ds   = SpeechCommandsDataset(args.data_dir, split="testing",  download=True)
    calib_ds  = SpeechCommandsDataset(args.data_dir, split="training", download=True)
    test_loader  = DataLoader(test_ds,  args.batch_size, shuffle=False,
                              num_workers=args.num_workers)
    calib_loader = DataLoader(calib_ds, args.batch_size, shuffle=True,
                              num_workers=args.num_workers)

    results: Dict[str, float] = {}

    # 1 — FP32
    print("\n[1/4] FP32 accuracy ...")
    results["FP32"] = eval_accuracy(fp32, test_loader)

    # 2 — Naive INT4 PTQ
    print("[2/4] Naive INT4 PTQ ...")
    naive = NaiveINT4DSCNN(fp32, calib_loader,
                            n_calib=args.calib_batches).to(DEVICE)
    results["Naive INT4 PTQ"] = eval_accuracy(naive, test_loader)

    # 3 — UDP Mode 4
    print("[3/4] UDP Mode 4 (weight-only INT4) ...")
    mode4 = UDPMode4DSCNN(fp32).to(DEVICE)
    results["UDP Mode 4 (w-only)"] = eval_accuracy(mode4, test_loader)

    # 4 — UDP Mode 5
    print("[4/4] UDP Mode 5 (full INT4 pipeline) ...")
    mode5 = UDPMode5DSCNN(fp32, calib_loader,
                           n_calib=args.calib_batches).to(DEVICE)
    results["UDP Mode 5 (full INT4)"] = eval_accuracy(mode5, test_loader)

    # Summary
    print()
    print("=" * 52)
    print("KWS (DS-CNN) Accuracy Summary")
    print("=" * 52)
    print(f"{'Mode':<28}  {'Acc':>8}  {'>=90%':>6}")
    print("-" * 52)
    for name, acc in results.items():
        status = "PASS" if acc >= 90.0 else "FAIL"
        print(f"{name:<28}  {acc:>7.2f}%  {status:>6}")
    print("=" * 52)

    # Save JSON
    import json
    out_path = os.path.join(os.path.dirname(args.ckpt), "kws_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved -> {out_path}")


if __name__ == "__main__":
    main()
