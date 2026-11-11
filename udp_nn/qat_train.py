"""
qat_train.py — Quantization-Aware Training for ResNet-8

Fine-tunes a trained FP32 checkpoint with INT4 fake-quantisation active
so the model learns weights robust to 4-bit rounding.

Modes:
  w4   — per-channel INT4 weight QAT only (UDP Mode 4 target)
  full — weight + activation QAT at block boundaries (UDP Mode 5 target)

Usage:
  python qat_train.py --mode w4
  python qat_train.py --mode full
  python qat_train.py --mode both
  python qat_train.py --mode w4 --epochs 20 --lr 1e-4

Outputs:
  checkpoints/qat_w4_best.pth
  checkpoints/qat_full_best.pth
"""

import argparse
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from qat import (
    QATResNet8Full,
    build_qat_from_checkpoint,
    calibrate_full_acts,
    evaluate_qat,
    load_resnet8_state_dict,
)

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)


def get_loaders(data_dir: str, batch_size: int, num_workers: int,
                val_split: int = 5000):
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

    full_train = datasets.CIFAR10(data_dir, train=True, download=True,
                                  transform=train_tf)
    val_ds = datasets.CIFAR10(data_dir, train=True, download=True,
                              transform=val_tf)
    test_ds = datasets.CIFAR10(data_dir, train=False, download=True,
                               transform=val_tf)

    train_len = len(full_train) - val_split
    train_ds, _ = random_split(
        full_train, [train_len, val_split],
        generator=torch.Generator().manual_seed(42),
    )
    val_indices = list(range(train_len, train_len + val_split))
    val_subset = torch.utils.data.Subset(val_ds, val_indices)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=256, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    calib_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers)
    return train_loader, val_loader, test_loader, calib_loader


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        correct += out.argmax(1).eq(labels).sum().item()
        total += imgs.size(0)
    return total_loss / total, 100.0 * correct / total


def run_qat(mode: str, args, device, train_loader, val_loader,
            test_loader, calib_loader):
    print(f"\n{'='*60}")
    print(f"  QAT training — mode: {mode}")
    print(f"{'='*60}")

    model = build_qat_from_checkpoint(mode, args.weights, device,
                                      args.conv_filters)

    if mode == "full":
        print("Calibrating activation scales before QAT ...")
        stats = calibrate_full_acts(model, calib_loader, device,
                                    args.calib_batches)
        for k, (s, z) in stats.items():
            print(f"  {k:<10s}: scale={s:.5f}  zero_point={z}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    ckpt_name = f"qat_{mode}_best.pth"
    ckpt_path = os.path.join(args.ckpt_dir, ckpt_name)
    best_val = 0.0

    fp32_val = evaluate_qat(model, val_loader, device)
    print(f"  Pre-QAT val accuracy : {fp32_val:.2f}%")

    for epoch in range(args.epochs):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device)
        va_acc = evaluate_qat(model, val_loader, device)
        scheduler.step()
        elapsed = time.time() - t0

        print(f"  Epoch {epoch+1:2d}/{args.epochs}  "
              f"loss={tr_loss:.4f}  train={tr_acc:.2f}%  "
              f"val={va_acc:.2f}%  lr={scheduler.get_last_lr()[0]:.2e}  "
              f"{elapsed:.1f}s")

        if va_acc > best_val:
            best_val = va_acc
            payload = {
                "mode": mode,
                "model": model.base_state_dict(),
                "best_val_acc": best_val,
                "epoch": epoch,
                "conv_filters": args.conv_filters,
            }
            if mode == "full" and isinstance(model, QATResNet8Full):
                payload["act_stats"] = model.activation_stats()
            torch.save(payload, ckpt_path)
            print(f"    -> saved best ({best_val:.2f}%) to {ckpt_name}")

    # Final test eval with best checkpoint
    ckpt = torch.load(ckpt_path, map_location=device)
    load_resnet8_state_dict(model, ckpt["model"])
    if mode == "full" and isinstance(model, QATResNet8Full):
        for k, (s, z) in ckpt.get("act_stats", {}).items():
            q = model.act_quants[k]
            q.scale.fill_(s)
            q.zero_point.fill_(float(z))
            q.calibrating = False
    test_acc = evaluate_qat(model, test_loader, device)

    print(f"\n  Best val  : {best_val:.2f}%")
    print(f"  Test acc  : {test_acc:.2f}%  "
          f"({'PASS' if test_acc >= 85 else 'FAIL'} vs MLPerf >=85%)")
    return {"mode": mode, "val": best_val, "test": test_acc, "ckpt": ckpt_path}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["w4", "full", "both"], default="both")
    parser.add_argument("--weights", default="./checkpoints/best.pth")
    parser.add_argument("--ckpt-dir", default="./checkpoints")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--calib-batches", type=int, default=16)
    parser.add_argument("--conv-filters", type=int, default=26)
    args = parser.parse_args()

    os.makedirs(args.ckpt_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device  : {device}")
    print(f"Weights : {args.weights}")
    print(f"Epochs  : {args.epochs}  LR: {args.lr}")

    train_loader, val_loader, test_loader, calib_loader = get_loaders(
        args.data_dir, args.batch_size, args.num_workers)

    modes = ["w4", "full"] if args.mode == "both" else [args.mode]
    results = []
    for mode in modes:
        results.append(run_qat(
            mode, args, device, train_loader, val_loader,
            test_loader, calib_loader))

    print(f"\n{'='*60}")
    print("  QAT Summary")
    print(f"{'='*60}")
    for r in results:
        status = "PASS" if r["test"] >= 85 else "FAIL"
        print(f"  {r['mode']:<6s}  test={r['test']:>6.2f}%  val={r['val']:>6.2f}%  "
              f">=85% -> {status}")
        print(f"         checkpoint: {r['ckpt']}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
