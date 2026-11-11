"""
vww/qat_train.py — QAT for MobileNetV1 0.25× (VWW)

Mode w4  : INT4 weight-only QAT
Mode full: INT4 weight + activation QAT

Usage:
  python vww/qat_train.py --ckpt checkpoints/vww_best.pth --data-dir ./data/vww_96 --mode w4
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
from vww.model   import build_mobilenet_vww, MobileNetV1
from vww.dataset import VWWDataset, CLASSES

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─── STE helpers ─────────────────────────────────────────────────────────────

class _STERound(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, n):
        return x.round().clamp(-n, n - 1)
    @staticmethod
    def backward(ctx, g):
        return g, None


def fq_weight_ste(w: torch.Tensor, bits: int = 4) -> torch.Tensor:
    n     = 2 ** (bits - 1)
    scale = w.abs().amax(dim=list(range(1, w.dim())), keepdim=True).clamp(1e-8) / n
    return _STERound.apply(w / scale, n) * scale


def fq_act_ste(x: torch.Tensor, scale: float, bits: int = 4) -> torch.Tensor:
    n = 2 ** (bits - 1)
    return _STERound.apply(x / max(scale, 1e-8), n) * max(scale, 1e-8)


# ─── QAT weight-only ─────────────────────────────────────────────────────────

class QATWeightMobileNet(nn.Module):
    def __init__(self, base: MobileNetV1, bits: int = 4):
        super().__init__()
        self.m    = deepcopy(base)
        self.bits = bits

    def _fq(self, w):
        return fq_weight_ste(w, self.bits)

    def _blk_fwd(self, blk, x):
        dw, pw = blk.dw[0], blk.pw[0]
        x = F.conv2d(x, self._fq(dw.weight),
                     stride=dw.stride, padding=dw.padding, groups=dw.groups)
        x = blk.dw[1](x); x = blk.dw[2](x)
        x = F.conv2d(x, self._fq(pw.weight))
        x = blk.pw[1](x); x = blk.pw[2](x)
        return x

    def forward(self, x):
        m = self.m
        sc = m.stem[0]
        x  = F.conv2d(x, self._fq(sc.weight), stride=sc.stride, padding=sc.padding)
        x  = m.stem[1](x); x = m.stem[2](x)
        for blk in m.features:
            x = self._blk_fwd(blk, x)
        x = m.pool(x); x = m.flatten(x)
        x = F.linear(x, self._fq(m.fc.weight), m.fc.bias)
        return x


# ─── QAT full INT4 ───────────────────────────────────────────────────────────

class QATFullMobileNet(nn.Module):
    def __init__(self, base: MobileNetV1, calib_loader, n_calib=8, bits=4):
        super().__init__()
        self.m     = deepcopy(base)
        self.bits  = bits
        self._scales: dict = {}
        self._calibrate(calib_loader, n_calib)

    def _calibrate(self, loader, n):
        self.m.eval()
        buf = {}
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
            all_v = torch.cat(tensors)
            pct   = min(all_v.numel(), 500_000)
            if all_v.numel() > pct:
                all_v = all_v[torch.randperm(all_v.numel())[:pct]]
            self._scales[key] = torch.quantile(all_v, 0.999).item() / n_int

    def _fq_w(self, w):
        return fq_weight_ste(w, self.bits)

    def _fq_a(self, x, key):
        return fq_act_ste(x, self._scales.get(key, x.abs().max().item() / (2**(self.bits-1))), self.bits)

    def _blk_fwd(self, blk, x):
        dw, pw = blk.dw[0], blk.pw[0]
        x = F.conv2d(x, self._fq_w(dw.weight),
                     stride=dw.stride, padding=dw.padding, groups=dw.groups)
        x = blk.dw[1](x); x = blk.dw[2](x)
        x = F.conv2d(x, self._fq_w(pw.weight))
        x = blk.pw[1](x); x = blk.pw[2](x)
        return x

    def forward(self, x):
        m = self.m
        x = self._fq_a(x, "input")
        sc = m.stem[0]
        x  = F.conv2d(x, self._fq_w(sc.weight), stride=sc.stride, padding=sc.padding)
        x  = m.stem[1](x); x = m.stem[2](x)
        x  = self._fq_a(x, "stem")
        for j, blk in enumerate(m.features):
            x = self._blk_fwd(blk, x)
            x = self._fq_a(x, f"feat{j}")
        x = m.pool(x); x = m.flatten(x)
        x = F.linear(x, self._fq_w(m.fc.weight), m.fc.bias)
        return x


# ─── Training helpers ─────────────────────────────────────────────────────────

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        out  = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        correct    += out.argmax(1).eq(labels).sum().item()
        total      += imgs.size(0)
    return total_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for imgs, labels in loader:
        correct += model(imgs.to(device)).argmax(1).cpu().eq(labels).sum().item()
        total   += labels.size(0)
    return 100.0 * correct / total


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",           required=True)
    parser.add_argument("--data-dir",       required=True)
    parser.add_argument("--mode",           choices=["w4", "full"], default="w4")
    parser.add_argument("--epochs",         type=int,   default=20)
    parser.add_argument("--lr",             type=float, default=1e-4)
    parser.add_argument("--batch-size",     type=int,   default=64)
    parser.add_argument("--num-workers",    type=int,   default=2)
    parser.add_argument("--calib-batches",  type=int,   default=8)
    args = parser.parse_args()

    print(f"Device : {DEVICE}   Mode : {args.mode}")

    ckpt        = torch.load(args.ckpt, map_location=DEVICE)
    alpha       = ckpt.get("alpha",       0.25)
    num_classes = ckpt.get("num_classes", len(CLASSES))

    fp32 = build_mobilenet_vww(num_classes=num_classes, alpha=alpha).to(DEVICE)
    fp32.load_state_dict(ckpt["model"])

    train_ds = VWWDataset(args.data_dir, split="train")
    val_ds   = VWWDataset(args.data_dir, split="val")

    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   64, shuffle=False, num_workers=args.num_workers)
    calib_loader = DataLoader(train_ds, args.batch_size, shuffle=True,
                              num_workers=args.num_workers)

    if args.mode == "w4":
        model = QATWeightMobileNet(fp32).to(DEVICE)
    else:
        model = QATFullMobileNet(fp32, calib_loader, args.calib_batches).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val  = 0.0
    best_path = os.path.join(os.path.dirname(args.ckpt), f"vww_qat_{args.mode}.pth")

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
                        "alpha":       alpha,
                        "num_classes": num_classes}, best_path)
            print(f"  -> saved best ({best_val:.2f}%)")

    val_acc = evaluate(model, val_loader, DEVICE)
    status  = "PASS" if val_acc >= 80.0 else "FAIL"
    print(f"\nQAT-{args.mode} Val acc : {val_acc:.2f}%  (MLPerf >=80% -> {status})")
    print(f"Checkpoint saved -> {best_path}")


if __name__ == "__main__":
    main()
