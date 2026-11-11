"""
kws/qat_train.py — Quantization-Aware Training (QAT) for DS-CNN (KWS)

Mode w4  : INT4 weight-only QAT  (activations stay FP32)
Mode full: INT4 weight + activation QAT (block-boundary activation quantization)

Loads FP32 checkpoint, fine-tunes with fake-quant + STE, saves QAT checkpoint.

Usage:
  python kws/qat_train.py --ckpt checkpoints/kws_best.pth --mode w4
  python kws/qat_train.py --ckpt checkpoints/kws_best.pth --mode full
"""

import argparse
import os
import sys
import time
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from kws.model   import build_dscnn, DSCNN
from kws.dataset import SpeechCommandsDataset, CLASSES

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─── STE fake-quant (common to all layers) ───────────────────────────────────

class _STERound(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, n):
        return x.round().clamp(-n, n - 1)

    @staticmethod
    def backward(ctx, g):
        return g, None   # straight-through


def ste_round(x: torch.Tensor, n: int) -> torch.Tensor:
    return _STERound.apply(x, n)


def fq_weight_ste(w: torch.Tensor, bits: int = 4) -> torch.Tensor:
    """Per-output-channel INT4 fake-quant with STE."""
    n     = 2 ** (bits - 1)
    scale = w.abs().amax(dim=list(range(1, w.dim())), keepdim=True).clamp(1e-8) / n
    return ste_round(w / scale, n) * scale


def fq_act_ste(x: torch.Tensor, scale: float, bits: int = 4) -> torch.Tensor:
    """Per-tensor symmetric INT4 fake-quant on activation with STE."""
    n     = 2 ** (bits - 1)
    scale = max(scale, 1e-8)
    return ste_round(x / scale, n) * scale


# ─── QAT Model: weight-only (mode=w4) ────────────────────────────────────────

class QATWeightDSCNN(nn.Module):
    """DS-CNN with per-channel INT4 weight fake-quant; activations in FP32."""

    def __init__(self, base: DSCNN, bits: int = 4):
        super().__init__()
        self.m    = deepcopy(base)
        self.bits = bits

    def _fq(self, w):
        return fq_weight_ste(w, self.bits)

    def forward(self, x):
        m = self.m
        x = F.conv2d(x, self._fq(m.stem[0].weight),
                     stride=m.stem[0].stride, padding=m.stem[0].padding)
        x = m.stem[1](x); x = m.stem[2](x)
        for blk in m.blocks:
            x = F.conv2d(x, self._fq(blk.dw[0].weight),
                         padding=blk.dw[0].padding, groups=blk.dw[0].groups)
            x = blk.dw[1](x); x = blk.dw[2](x)
            x = F.conv2d(x, self._fq(blk.pw[0].weight))
            x = blk.pw[1](x); x = blk.pw[2](x)
        x = m.pool(x); x = m.flatten(x)
        x = F.linear(x, self._fq(m.fc.weight), m.fc.bias)
        return x


# ─── QAT Model: full INT4 (mode=full) ────────────────────────────────────────

class QATFullDSCNN(nn.Module):
    """DS-CNN with INT4 weights AND INT4 activations (block-boundary quant)."""

    def __init__(self, base: DSCNN, calib_loader: DataLoader,
                 n_calib: int = 8, bits: int = 4):
        super().__init__()
        self.m     = deepcopy(base)
        self.bits  = bits
        self._scales: dict = {}
        self._calibrate(calib_loader, n_calib)

    def _calibrate(self, loader, n):
        self.m.eval()
        buf = {}
        with torch.no_grad():
            for i, (feats, _) in enumerate(loader):
                if i >= n:
                    break
                feats = feats.to(DEVICE)
                buf.setdefault("input", []).append(feats.abs().flatten())
                x = self.m.stem(feats)
                buf.setdefault("stem", []).append(x.abs().flatten())
                for j, blk in enumerate(self.m.blocks):
                    x = blk(x)
                    buf.setdefault(f"block{j}", []).append(x.abs().flatten())
        n_int = 2 ** (self.bits - 1)
        for key, tensors in buf.items():
            all_v = torch.cat(tensors)
            pct   = min(all_v.numel(), 500_000)
            if all_v.numel() > pct:
                all_v = all_v[torch.randperm(all_v.numel())[:pct]]
            self._scales[key] = torch.quantile(all_v, 0.999).item() / n_int

    def _fq_w(self, w):
        return fq_weight_ste(w, self.bits)

    def _fq_a(self, x, key):
        return fq_act_ste(x, self._scales.get(key, x.abs().max().item() / (2**(self.bits-1))), self.bits)

    def forward(self, x):
        m = self.m
        x = self._fq_a(x, "input")
        x = F.conv2d(x, self._fq_w(m.stem[0].weight),
                     stride=m.stem[0].stride, padding=m.stem[0].padding)
        x = m.stem[1](x); x = m.stem[2](x)
        x = self._fq_a(x, "stem")
        for j, blk in enumerate(m.blocks):
            x = F.conv2d(x, self._fq_w(blk.dw[0].weight),
                         padding=blk.dw[0].padding, groups=blk.dw[0].groups)
            x = blk.dw[1](x); x = blk.dw[2](x)
            x = F.conv2d(x, self._fq_w(blk.pw[0].weight))
            x = blk.pw[1](x); x = blk.pw[2](x)
            x = self._fq_a(x, f"block{j}")
        x = m.pool(x); x = m.flatten(x)
        x = F.linear(x, self._fq_w(m.fc.weight), m.fc.bias)
        return x


# ─── Training loop ────────────────────────────────────────────────────────────

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for feats, labels in loader:
        feats, labels = feats.to(device), labels.to(device)
        optimizer.zero_grad()
        out  = model(feats)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * feats.size(0)
        correct    += out.argmax(1).eq(labels).sum().item()
        total      += feats.size(0)
    return total_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for feats, labels in loader:
        correct += model(feats.to(device)).argmax(1).cpu().eq(labels).sum().item()
        total   += labels.size(0)
    return 100.0 * correct / total


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",           required=True,
                        help="Path to FP32 checkpoint (kws_best.pth)")
    parser.add_argument("--data-dir",       default="./data/speech_commands")
    parser.add_argument("--mode",           choices=["w4", "full"], default="w4")
    parser.add_argument("--epochs",         type=int,   default=20)
    parser.add_argument("--lr",             type=float, default=1e-4)
    parser.add_argument("--batch-size",     type=int,   default=100)
    parser.add_argument("--num-workers",    type=int,   default=2)
    parser.add_argument("--calib-batches",  type=int,   default=8)
    args = parser.parse_args()

    print(f"Device : {DEVICE}   Mode : {args.mode}")

    ckpt        = torch.load(args.ckpt, map_location=DEVICE)
    num_filters = ckpt.get("num_filters", 64)
    num_classes = ckpt.get("num_classes", len(CLASSES))

    fp32 = build_dscnn(num_classes=num_classes,
                       num_filters=num_filters).to(DEVICE)
    fp32.load_state_dict(ckpt["model"])

    train_ds = SpeechCommandsDataset(args.data_dir, split="training",   download=True)
    val_ds   = SpeechCommandsDataset(args.data_dir, split="validation", download=True)
    test_ds  = SpeechCommandsDataset(args.data_dir, split="testing",    download=True)

    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True,
                              num_workers=args.num_workers)
    val_loader   = DataLoader(val_ds,   256, shuffle=False, num_workers=args.num_workers)
    test_loader  = DataLoader(test_ds,  256, shuffle=False, num_workers=args.num_workers)
    calib_loader = DataLoader(train_ds, args.batch_size, shuffle=True,
                              num_workers=args.num_workers)

    if args.mode == "w4":
        model = QATWeightDSCNN(fp32).to(DEVICE)
    else:
        model = QATFullDSCNN(fp32, calib_loader, args.calib_batches).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val  = 0.0
    out_name  = f"kws_qat_{args.mode}.pth"
    best_path = os.path.join(os.path.dirname(args.ckpt), out_name)

    for epoch in range(args.epochs):
        t0 = time.time()
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        va_acc = evaluate(model, val_loader, DEVICE)
        scheduler.step()
        print(f"QAT Epoch {epoch+1:3d}/{args.epochs}  "
              f"loss={tr_loss:.4f}  train={tr_acc:.2f}%  "
              f"val={va_acc:.2f}%  {time.time()-t0:.1f}s")
        if va_acc > best_val:
            best_val = va_acc
            torch.save({"model":       model.state_dict(),
                        "base_model":  fp32.state_dict(),
                        "mode":        args.mode,
                        "val_acc":     best_val,
                        "num_filters": num_filters,
                        "num_classes": num_classes}, best_path)
            print(f"  -> saved best ({best_val:.2f}%)")

    ckpt2 = torch.load(best_path, map_location=DEVICE)
    if args.mode == "w4":
        model.load_state_dict(ckpt2["model"])
    else:
        model.load_state_dict(ckpt2["model"])
    test_acc = evaluate(model, test_loader, DEVICE)
    status   = "PASS" if test_acc >= 90.0 else "FAIL"
    print(f"\nQAT-{args.mode} Test acc : {test_acc:.2f}%  (MLPerf >=90% -> {status})")
    print(f"Checkpoint saved -> {best_path}")


if __name__ == "__main__":
    main()
