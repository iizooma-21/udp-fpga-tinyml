"""
vww/train.py — Train MobileNetV1 (0.25×) on Visual Wake Words

Usage:
  python vww/train.py --data-dir ./data/vww_96

Target : >= 80% top-1 (MLPerf Tiny closed division)
"""

import argparse
import os
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from vww.model   import build_mobilenet_vww
from vww.dataset import VWWDataset, CLASSES

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    loss_sum, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        out  = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * imgs.size(0)
        correct  += out.argmax(1).eq(labels).sum().item()
        total    += imgs.size(0)
    return loss_sum / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for imgs, labels in loader:
        correct += model(imgs.to(device)).argmax(1).cpu().eq(labels).sum().item()
        total   += labels.size(0)
    return 100.0 * correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir",    required=True,
                        help="Path to VWW dataset root (with train/ and val/)")
    parser.add_argument("--ckpt-dir",    default="./checkpoints")
    parser.add_argument("--epochs",      type=int,   default=50)
    parser.add_argument("--lr",          type=float, default=0.01)
    parser.add_argument("--batch-size",  type=int,   default=64)
    parser.add_argument("--num-workers", type=int,   default=2)
    parser.add_argument("--alpha",       type=float, default=0.25)
    args = parser.parse_args()

    os.makedirs(args.ckpt_dir, exist_ok=True)
    print(f"Device : {DEVICE}")

    train_ds = VWWDataset(args.data_dir, split="train")
    val_ds   = VWWDataset(args.data_dir, split="val")

    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   256, shuffle=False,
                              num_workers=args.num_workers)

    model     = build_mobilenet_vww(num_classes=len(CLASSES),
                                    alpha=args.alpha).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val  = 0.0
    best_path = os.path.join(args.ckpt_dir, "vww_best.pth")

    for epoch in range(args.epochs):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, DEVICE)
        va_acc = evaluate(model, val_loader, DEVICE)
        scheduler.step()

        print(f"Epoch {epoch+1:3d}/{args.epochs}  "
              f"loss={tr_loss:.4f}  train={tr_acc:.2f}%  "
              f"val={va_acc:.2f}%  {time.time()-t0:.1f}s")

        if va_acc > best_val:
            best_val = va_acc
            torch.save({"model": model.state_dict(),
                        "val_acc": best_val, "epoch": epoch,
                        "alpha": args.alpha,
                        "num_classes": len(CLASSES)}, best_path)
            print(f"  -> saved best ({best_val:.2f}%)")

    # Final eval
    ckpt = torch.load(best_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    val_acc = evaluate(model, val_loader, DEVICE)
    status  = "PASS" if val_acc >= 80.0 else "FAIL"
    print(f"\nFP32 Val acc : {val_acc:.2f}%  (MLPerf >= 80% -> {status})")


if __name__ == "__main__":
    main()
