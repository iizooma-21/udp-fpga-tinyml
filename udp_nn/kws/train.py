"""
kws/train.py — Train DS-CNN on Google Speech Commands (MLPerf Tiny KWS)

Dataset  : torchaudio.datasets.SPEECHCOMMANDS (auto-download)
Classes  : 10 keywords + silence + unknown = 12
Target   : >= 90% top-1 (MLPerf Tiny closed division)

Usage:
  python kws/train.py
  python kws/train.py --epochs 30 --lr 0.01
"""

import argparse
import os
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from kws.model import build_dscnn
from kws.dataset import SpeechCommandsDataset, CLASSES

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_loaders(data_dir, batch_size, num_workers, val_frac=0.1):
    train_ds = SpeechCommandsDataset(data_dir, split="training",  download=True)
    val_ds   = SpeechCommandsDataset(data_dir, split="validation", download=True)
    test_ds  = SpeechCommandsDataset(data_dir, split="testing",    download=True)

    train_loader = DataLoader(train_ds, batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   256, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_ds,  256, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader


def train_one_epoch(model, loader, criterion, optimizer, device):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir",     default="./data/speech_commands")
    parser.add_argument("--ckpt-dir",     default="./checkpoints")
    parser.add_argument("--epochs",       type=int,   default=30)
    parser.add_argument("--lr",           type=float, default=0.01)
    parser.add_argument("--batch-size",   type=int,   default=100)
    parser.add_argument("--num-workers",  type=int,   default=0)
    parser.add_argument("--num-filters",  type=int,   default=64)
    args = parser.parse_args()

    os.makedirs(args.ckpt_dir, exist_ok=True)
    print(f"Device  : {DEVICE}")
    print(f"Classes : {len(CLASSES)} — {CLASSES}")

    train_loader, val_loader, test_loader = get_loaders(
        args.data_dir, args.batch_size, args.num_workers)

    model     = build_dscnn(num_classes=len(CLASSES),
                            num_filters=args.num_filters).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val, best_path = 0.0, os.path.join(args.ckpt_dir, "kws_best.pth")

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
                        "num_filters": args.num_filters,
                        "num_classes": len(CLASSES)}, best_path)
            print(f"  -> saved best ({best_val:.2f}%)")

    # Final test
    ckpt = torch.load(best_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    test_acc = evaluate(model, test_loader, DEVICE)
    status = "PASS" if test_acc >= 90.0 else "FAIL"
    print(f"\nFP32 Test acc : {test_acc:.2f}%  (MLPerf >=90% -> {status})")


if __name__ == "__main__":
    main()
