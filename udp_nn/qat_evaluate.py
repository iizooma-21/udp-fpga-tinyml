"""
qat_evaluate.py — Evaluate QAT checkpoints

Reports accuracy three ways for each QAT checkpoint:
  1. Direct QAT model eval  (quant active in training forward path)
  2. UDP Mode 4 simulation  (weight-only INT4, per-channel)
  3. UDP Mode 5 simulation  (full INT4 pipeline with post-processing)

Usage:
  python qat_evaluate.py --mode w4
  python qat_evaluate.py --mode full
  python qat_evaluate.py --mode both
  python qat_evaluate.py --qat-weights checkpoints/qat_w4_best.pth
"""

import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from qat import (
    QATResNet8Full,
    build_qat_model,
    evaluate_qat,
    load_resnet8_state_dict,
)
from resnet8 import build_resnet8
from udp_sim import (
    UDPSim,
    build_w4_model,
    calibrate,
    eval_model,
    eval_sim,
)

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)


def load_qat_checkpoint(path: str, device: torch.device):
    ckpt = torch.load(path, map_location=device)
    mode = ckpt.get("mode", "w4")
    conv_filters = ckpt.get("conv_filters", 26)

    qat_model = build_qat_model(mode, conv_filters).to(device)
    load_resnet8_state_dict(qat_model, ckpt["model"])

    if mode == "full" and isinstance(qat_model, QATResNet8Full):
        act_stats = ckpt.get("act_stats", {})
        for k, (s, z) in act_stats.items():
            q = qat_model.act_quants[k]
            q.scale.fill_(s)
            q.zero_point.fill_(float(z))
            q.calibrating = False

    return qat_model, mode, conv_filters


def evaluate_one(qat_path: str, args, device, test_loader, calib_loader):
    qat_model, mode, conv_filters = load_qat_checkpoint(qat_path, device)

    print(f"\n{'='*60}")
    print(f"  QAT checkpoint: {qat_path}  (mode={mode})")
    print(f"{'='*60}")

    # 1) Direct QAT forward (training graph)
    direct_acc = evaluate_qat(qat_model, test_loader, device)
    print(f"  Direct QAT eval          : {direct_acc:.2f}%")

    # Build plain ResNet8 with QAT-learned weights for UDP sim
    fp32 = build_resnet8(conv_filters=conv_filters).to(device)
    fp32.load_state_dict(qat_model.base_state_dict())
    fp32.eval()

    # 2) UDP Mode 4 — weight-only INT4
    m4 = build_w4_model(fp32).to(device)
    mode4_acc, _ = eval_model(m4, test_loader, device)
    print(f"  UDP Mode 4 (weight-only) : {mode4_acc:.2f}%")

    # 3) UDP Mode 5 — full INT4 pipeline
    if mode == "full" and isinstance(qat_model, QATResNet8Full):
        stats = qat_model.activation_stats()
    else:
        stats = calibrate(fp32, calib_loader, device, args.calib_batches)
    sim = UDPSim(fp32, stats, device)
    mode5_acc, _ = eval_sim(sim, test_loader, device)
    print(f"  UDP Mode 5 (full INT4)   : {mode5_acc:.2f}%")

    print(f"  MLPerf target >=85%      : "
          f"direct={'PASS' if direct_acc >= 85 else 'FAIL'}  "
          f"mode4={'PASS' if mode4_acc >= 85 else 'FAIL'}  "
          f"mode5={'PASS' if mode5_acc >= 85 else 'FAIL'}")

    return {
        "path": qat_path,
        "mode": mode,
        "direct": direct_acc,
        "mode4": mode4_acc,
        "mode5": mode5_acc,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["w4", "full", "both"], default="both")
    parser.add_argument("--qat-weights", default=None,
                        help="Override checkpoint path (single mode only)")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--calib-batches", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    test_ds = datasets.CIFAR10(args.data_dir, train=False, download=True, transform=tf)
    calib_ds = datasets.CIFAR10(args.data_dir, train=True, download=True, transform=tf)
    test_loader = DataLoader(test_ds, args.batch_size, shuffle=False,
                             num_workers=args.num_workers)
    calib_loader = DataLoader(calib_ds, args.batch_size, shuffle=True,
                              num_workers=args.num_workers)

    if args.qat_weights:
        paths = [args.qat_weights]
    else:
        paths = []
        if args.mode in ("w4", "both"):
            paths.append("./checkpoints/qat_w4_best.pth")
        if args.mode in ("full", "both"):
            paths.append("./checkpoints/qat_full_best.pth")

    results = []
    for p in paths:
        results.append(evaluate_one(p, args, device, test_loader, calib_loader))

    if len(results) > 1:
        print(f"\n{'='*60}")
        print("  Comparison")
        print(f"{'='*60}")
        print(f"  {'Checkpoint':<28} {'Direct':>8} {'Mode4':>8} {'Mode5':>8}")
        for r in results:
            name = r["mode"]
            print(f"  {name:<28} {r['direct']:>7.2f}% {r['mode4']:>7.2f}% "
                  f"{r['mode5']:>7.2f}%")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
