"""
train.py — Train ResNet-8 on CIFAR-10 (MLPerf Tiny image classification task)

Hyperparameters match the MLPerf Tiny reference submission:
  Optimizer  : SGD, momentum=0.9, weight_decay=1e-4
  LR schedule: cosine annealing from 0.1 → 0
  Epochs     : 200
  Batch size : 128
  Augment    : random horizontal flip + random crop (pad=4)

Usage:
  python train.py                      # train with defaults
  python train.py --epochs 200 --lr 0.1
  python train.py --resume              # resume from checkpoint

Outputs:
  checkpoints/best.pth   — best validation accuracy weights
  checkpoints/last.pth   — latest epoch weights
"""

import argparse
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from resnet8 import build_resnet8

# ── CIFAR-10 normalisation constants ─────────────────────────────────────────
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)


def get_loaders(data_dir: str, batch_size: int, num_workers: int):
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    val_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    train_ds = datasets.CIFAR10(data_dir, train=True,  download=True, transform=train_tf)
    val_ds   = datasets.CIFAR10(data_dir, train=False, download=True, transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=256,        shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader


def train_one_epoch(model, loader, criterion, optimizer, device):
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
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        out  = model(imgs)
        loss = criterion(out, labels)
        total_loss += loss.item() * imgs.size(0)
        correct    += out.argmax(1).eq(labels).sum().item()
        total      += imgs.size(0)
    return total_loss / total, 100.0 * correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir",    default="./data")
    parser.add_argument("--ckpt-dir",    default="./checkpoints")
    parser.add_argument("--epochs",      type=int,   default=200)
    parser.add_argument("--batch-size",  type=int,   default=128)
    parser.add_argument("--lr",          type=float, default=0.1)
    parser.add_argument("--num-workers", type=int,   default=2)
    parser.add_argument("--conv-filters",type=int,   default=26)
    parser.add_argument("--resume",      action="store_true")
    args = parser.parse_args()

    os.makedirs(args.ckpt_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, val_loader = get_loaders(
        args.data_dir, args.batch_size, args.num_workers)

    model     = build_resnet8(conv_filters=args.conv_filters).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    start_epoch = 0
    best_acc    = 0.0

    if args.resume:
        ckpt_path = os.path.join(args.ckpt_dir, "last.pth")
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            start_epoch = ckpt["epoch"] + 1
            best_acc    = ckpt.get("best_acc", 0.0)
            print(f"Resumed from epoch {start_epoch}, best_acc={best_acc:.2f}%")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {total_params:,}")

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - t0

        print(f"Epoch {epoch+1:3d}/{args.epochs}  "
              f"train_loss={tr_loss:.4f}  train_acc={tr_acc:.2f}%  "
              f"val_loss={va_loss:.4f}  val_acc={va_acc:.2f}%  "
              f"lr={scheduler.get_last_lr()[0]:.5f}  {elapsed:.1f}s")

        # Save latest checkpoint
        torch.save({
            "epoch":     epoch,
            "model":     model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_acc":  best_acc,
        }, os.path.join(args.ckpt_dir, "last.pth"))

        # Save best checkpoint
        if va_acc > best_acc:
            best_acc = va_acc
            torch.save(model.state_dict(),
                       os.path.join(args.ckpt_dir, "best.pth"))
            print(f"  ★ New best: {best_acc:.2f}%")

    print(f"\nTraining complete. Best val accuracy: {best_acc:.2f}%")
    print(f"Best weights saved to: {args.ckpt_dir}/best.pth")


if __name__ == "__main__":
    main()
