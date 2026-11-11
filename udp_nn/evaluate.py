"""
evaluate.py — FP32 accuracy evaluation for ResNet-8 on CIFAR-10

Loads a trained checkpoint and reports:
  - Top-1 accuracy on the 10,000-image CIFAR-10 test set
  - Per-class accuracy breakdown
  - Total inference time

MLPerf Tiny closed-division accuracy target: >= 85% top-1

Usage:
  python evaluate.py                              # uses checkpoints/best.pth
  python evaluate.py --weights checkpoints/best.pth
  python evaluate.py --weights checkpoints/best.pth --batch-size 512
"""

import argparse
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from resnet8 import build_resnet8

CIFAR10_MEAN    = (0.4914, 0.4822, 0.4465)
CIFAR10_STD     = (0.2023, 0.1994, 0.2010)
CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
                   "dog",      "frog",       "horse", "ship", "truck"]


def get_test_loader(data_dir: str, batch_size: int, num_workers: int):
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    ds = datasets.CIFAR10(data_dir, train=False, download=True, transform=tf)
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True)


@torch.no_grad()
def run_evaluation(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    t0 = time.time()

    for imgs, labels in loader:
        imgs   = imgs.to(device)
        logits = model(imgs)
        preds  = logits.argmax(dim=1).cpu()
        all_preds.append(preds)
        all_labels.append(labels)

    elapsed = time.time() - t0
    preds  = torch.cat(all_preds)
    labels = torch.cat(all_labels)
    return preds, labels, elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights",      default="./checkpoints/best.pth")
    parser.add_argument("--data-dir",     default="./data")
    parser.add_argument("--batch-size",   type=int, default=256)
    parser.add_argument("--num-workers",  type=int, default=2)
    parser.add_argument("--conv-filters", type=int, default=26)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device  : {device}")
    print(f"Weights : {args.weights}")

    model = build_resnet8(conv_filters=args.conv_filters).to(device)
    state = torch.load(args.weights, map_location=device)
    # Support both raw state_dict and checkpoint dicts saved by train.py
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state)
    print(f"Params  : {sum(p.numel() for p in model.parameters()):,}")

    loader = get_test_loader(args.data_dir, args.batch_size, args.num_workers)
    preds, labels, elapsed = run_evaluation(model, loader, device)

    # ── Overall accuracy ──────────────────────────────────────────────────────
    top1 = preds.eq(labels).float().mean().item() * 100.0
    print(f"\n{'='*50}")
    print(f"  FP32 Top-1 Accuracy : {top1:.2f}%")
    print(f"  Inference time      : {elapsed:.2f}s  ({len(labels)} images)")
    mlperf_pass = "PASS" if top1 >= 85.0 else "FAIL"
    print(f"  MLPerf Tiny target  : >=85%  →  {mlperf_pass}")
    print(f"{'='*50}\n")

    # ── Per-class accuracy ────────────────────────────────────────────────────
    print("Per-class accuracy:")
    print(f"  {'Class':<12}  {'Correct':>7}  {'Total':>5}  {'Acc':>6}")
    print(f"  {'-'*38}")
    for c, name in enumerate(CIFAR10_CLASSES):
        mask    = labels.eq(c)
        correct = preds[mask].eq(labels[mask]).sum().item()
        total   = mask.sum().item()
        acc     = 100.0 * correct / total if total > 0 else 0.0
        print(f"  {name:<12}  {correct:>7}  {total:>5}  {acc:>5.1f}%")


if __name__ == "__main__":
    main()
