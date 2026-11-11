"""
vww/evaluate.py — Full evaluation pipeline for MLPerf Tiny VWW (MobileNetV1)

Evaluates:
  1. FP32 accuracy
  2. Naive INT4 PTQ (per-tensor weight + activation fake-quant)
  3. UDP Mode 4 — weight-only INT4
  4. UDP Mode 5 — full INT4 pipeline

Usage:
  python vww/evaluate.py --ckpt checkpoints/vww_best.pth --data-dir ./data/vww_96
"""

import argparse
import json
import os
import sys
from copy import deepcopy
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from vww.model   import build_mobilenet_vww, MobileNetV1, DWBlock
from vww.dataset import VWWDataset, CLASSES

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── INT4 helpers ─────────────────────────────────────────────────────────────

def fake_quant(x: torch.Tensor, bits: int = 4,
               per_channel: bool = False, ch_dim: int = 0) -> torch.Tensor:
    n = 2 ** (bits - 1)
    if per_channel:
        shape = [-1 if i == ch_dim else 1 for i in range(x.dim())]
        scale = x.abs().amax(
            dim=[i for i in range(x.dim()) if i != ch_dim], keepdim=True
        ).view(shape)
    else:
        scale = x.abs().max()
    scale = scale.clamp(min=1e-8) / n
    return (x / scale).round().clamp(-n, n - 1) * scale


# ─── FP32 evaluation ─────────────────────────────────────────────────────────

@torch.no_grad()
def eval_accuracy(model: nn.Module, loader: DataLoader) -> float:
    model.eval()
    correct, total = 0, 0
    for imgs, labels in loader:
        correct += model(imgs.to(DEVICE)).argmax(1).cpu().eq(labels).sum().item()
        total   += labels.size(0)
    return 100.0 * correct / total


# ─── Helper: apply fake-quant to a DWBlock ───────────────────────────────────

def _dw_block_fq_w(blk: DWBlock, fq_fn, x: torch.Tensor) -> torch.Tensor:
    dw_conv = blk.dw[0]
    pw_conv = blk.pw[0]
    x = F.conv2d(x, fq_fn(dw_conv.weight),
                 stride=dw_conv.stride, padding=dw_conv.padding,
                 groups=dw_conv.groups)
    x = blk.dw[1](x); x = blk.dw[2](x)  # BN + ReLU
    x = F.conv2d(x, fq_fn(pw_conv.weight))
    x = blk.pw[1](x); x = blk.pw[2](x)  # BN + ReLU
    return x


# ─── Naive INT4 PTQ ──────────────────────────────────────────────────────────

class NaiveINT4MobileNet(nn.Module):
    def __init__(self, base: MobileNetV1, calib_loader, n_calib=8, bits=4):
        super().__init__()
        self.m    = deepcopy(base)
        self.bits = bits
        self._scales: Dict[str, float] = {}
        self._calibrate(calib_loader, n_calib)

    def _calibrate(self, loader, n):
        self.m.eval()
        with torch.no_grad():
            for i, (imgs, _) in enumerate(loader):
                if i >= n:
                    break
                imgs = imgs.to(DEVICE)
                self._up("input", imgs)
                x = self.m.stem(imgs)
                self._up("stem", x)
                for j, blk in enumerate(self.m.features):
                    x = blk(x)
                    self._up(f"feat{j}", x)

    def _up(self, key, t):
        v = t.abs().max().item()
        self._scales[key] = max(self._scales.get(key, 0.0), v)

    def _fq_w(self, w):
        return fake_quant(w, self.bits, per_channel=True, ch_dim=0)

    def _fq_a(self, x, key):
        n = 2 ** (self.bits - 1)
        s = max(self._scales.get(key, x.abs().max().item()), 1e-8) / n
        return (x / s).round().clamp(-n, n - 1) * s

    def forward(self, x):
        m = self.m
        x = self._fq_a(x, "input")
        stem_conv = m.stem[0]
        x = F.conv2d(x, self._fq_w(stem_conv.weight),
                     stride=stem_conv.stride, padding=stem_conv.padding)
        x = m.stem[1](x); x = m.stem[2](x)
        x = self._fq_a(x, "stem")
        for j, blk in enumerate(m.features):
            x = _dw_block_fq_w(blk, self._fq_w, x)
            x = self._fq_a(x, f"feat{j}")
        x = m.pool(x); x = m.flatten(x)
        x = F.linear(x, self._fq_w(m.fc.weight), m.fc.bias)
        return x


# ─── UDP Mode 4 (weight-only) ────────────────────────────────────────────────

class UDPMode4MobileNet(nn.Module):
    def __init__(self, base: MobileNetV1, bits=4):
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
        for blk in m.features:
            x = _dw_block_fq_w(blk, self._fq_w, x)
        x = m.pool(x); x = m.flatten(x)
        x = F.linear(x, self._fq_w(m.fc.weight), m.fc.bias)
        return x


# ─── UDP Mode 5 (full INT4 pipeline) ─────────────────────────────────────────

class UDPMode5MobileNet(nn.Module):
    def __init__(self, base: MobileNetV1, calib_loader, n_calib=8, bits=4):
        super().__init__()
        self.m    = deepcopy(base)
        self.bits = bits
        self._scales: Dict[str, float] = {}
        self._calibrate(calib_loader, n_calib)

    def _calibrate(self, loader, n):
        self.m.eval()
        buf: Dict[str, List[torch.Tensor]] = {}
        with torch.no_grad():
            for i, (imgs, _) in enumerate(loader):
                if i >= n:
                    break
                imgs = imgs.to(DEVICE)
                buf.setdefault("input", []).append(imgs.abs().flatten())
                x = self.m.stem(imgs)
                buf.setdefault("stem", []).append(x.abs().flatten())
                for j, blk in enumerate(self.m.features):
                    x = blk(x)
                    buf.setdefault(f"feat{j}", []).append(x.abs().flatten())

        n_int = 2 ** (self.bits - 1)
        for key, tensors in buf.items():
            all_vals = torch.cat(tensors)
            pct = min(all_vals.numel(), 500_000)
            if all_vals.numel() > pct:
                all_vals = all_vals[torch.randperm(all_vals.numel())[:pct]]
            self._scales[key] = torch.quantile(all_vals, 0.999).item() / n_int

    def _fq_w(self, w):
        return fake_quant(w, self.bits, per_channel=True, ch_dim=0)

    def _fq_a(self, x, key):
        n = 2 ** (self.bits - 1)
        s = max(self._scales.get(key, x.abs().max().item() / n), 1e-8)
        return (x / s).round().clamp(-n, n - 1) * s

    def forward(self, x):
        m = self.m
        x = self._fq_a(x, "input")
        x = F.conv2d(x, self._fq_w(m.stem[0].weight),
                     stride=m.stem[0].stride, padding=m.stem[0].padding)
        x = m.stem[1](x); x = m.stem[2](x)
        x = self._fq_a(x, "stem")
        for j, blk in enumerate(m.features):
            x = _dw_block_fq_w(blk, self._fq_w, x)
            x = self._fq_a(x, f"feat{j}")
        x = m.pool(x); x = m.flatten(x)
        x = F.linear(x, self._fq_w(m.fc.weight), m.fc.bias)
        return x


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",          required=True)
    parser.add_argument("--data-dir",      required=True)
    parser.add_argument("--calib-batches", type=int, default=8)
    parser.add_argument("--batch-size",    type=int, default=64)
    parser.add_argument("--num-workers",   type=int, default=2)
    args = parser.parse_args()

    print(f"Device : {DEVICE}")

    ckpt = torch.load(args.ckpt, map_location=DEVICE)
    alpha       = ckpt.get("alpha",       0.25)
    num_classes = ckpt.get("num_classes", len(CLASSES))

    fp32 = build_mobilenet_vww(num_classes=num_classes, alpha=alpha).to(DEVICE)
    fp32.load_state_dict(ckpt["model"])
    fp32.eval()

    val_ds       = VWWDataset(args.data_dir, split="val")
    train_ds     = VWWDataset(args.data_dir, split="train")
    val_loader   = DataLoader(val_ds,   args.batch_size, shuffle=False,
                              num_workers=args.num_workers)
    calib_loader = DataLoader(train_ds, args.batch_size, shuffle=True,
                              num_workers=args.num_workers)

    results: Dict[str, float] = {}

    print("\n[1/4] FP32 accuracy ...")
    results["FP32"] = eval_accuracy(fp32, val_loader)

    print("[2/4] Naive INT4 PTQ ...")
    naive = NaiveINT4MobileNet(fp32, calib_loader,
                               n_calib=args.calib_batches).to(DEVICE)
    results["Naive INT4 PTQ"] = eval_accuracy(naive, val_loader)

    print("[3/4] UDP Mode 4 (weight-only INT4) ...")
    mode4 = UDPMode4MobileNet(fp32).to(DEVICE)
    results["UDP Mode 4 (w-only)"] = eval_accuracy(mode4, val_loader)

    print("[4/4] UDP Mode 5 (full INT4 pipeline) ...")
    mode5 = UDPMode5MobileNet(fp32, calib_loader,
                              n_calib=args.calib_batches).to(DEVICE)
    results["UDP Mode 5 (full INT4)"] = eval_accuracy(mode5, val_loader)

    print()
    print("=" * 52)
    print("VWW (MobileNetV1 0.25x) Accuracy Summary")
    print("=" * 52)
    print(f"{'Mode':<28}  {'Acc':>8}  {'>=80%':>6}")
    print("-" * 52)
    for name, acc in results.items():
        status = "PASS" if acc >= 80.0 else "FAIL"
        print(f"{name:<28}  {acc:>7.2f}%  {status:>6}")
    print("=" * 52)

    out_path = os.path.join(os.path.dirname(args.ckpt), "vww_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved -> {out_path}")


if __name__ == "__main__":
    main()
