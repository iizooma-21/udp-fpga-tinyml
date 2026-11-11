"""
vww/qat_evaluate.py — Evaluate VWW QAT checkpoint (MobileNetV1)

Usage:
  python vww/qat_evaluate.py --ckpt checkpoints/vww_qat_w4.pth --data-dir ./data/vww_96
"""

import argparse
import json
import os
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from vww.model       import build_mobilenet_vww
from vww.dataset     import VWWDataset, CLASSES
from vww.qat_train   import QATWeightMobileNet
from vww.evaluate    import UDPMode4MobileNet, UDPMode5MobileNet, eval_accuracy

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",          required=True)
    parser.add_argument("--data-dir",      required=True)
    parser.add_argument("--calib-batches", type=int, default=8)
    parser.add_argument("--batch-size",    type=int, default=64)
    parser.add_argument("--num-workers",   type=int, default=2)
    args = parser.parse_args()

    print(f"Device : {DEVICE}")

    ckpt        = torch.load(args.ckpt, map_location=DEVICE)
    alpha       = ckpt.get("alpha",       0.25)
    num_classes = ckpt.get("num_classes", len(CLASSES))

    fp32 = build_mobilenet_vww(num_classes=num_classes, alpha=alpha).to(DEVICE)
    if "base_model" in ckpt:
        fp32.load_state_dict(ckpt["base_model"])

    qat_model = QATWeightMobileNet(fp32).to(DEVICE)
    qat_model.load_state_dict(ckpt["model"])
    qat_model.eval()

    fp32_from_qat = build_mobilenet_vww(num_classes=num_classes, alpha=alpha).to(DEVICE)
    fp32_from_qat.load_state_dict(qat_model.m.state_dict())

    val_ds   = VWWDataset(args.data_dir, split="val")
    train_ds = VWWDataset(args.data_dir, split="train")
    val_loader   = DataLoader(val_ds,   args.batch_size, shuffle=False,
                              num_workers=args.num_workers)
    calib_loader = DataLoader(train_ds, args.batch_size, shuffle=True,
                              num_workers=args.num_workers)

    results = {}

    print("\n[1/3] QAT-w4 direct accuracy ...")
    results["QAT-w4 direct"] = eval_accuracy(qat_model, val_loader)

    print("[2/3] UDP Mode 4 with QAT-trained weights ...")
    mode4 = UDPMode4MobileNet(fp32_from_qat).to(DEVICE)
    results["UDP Mode 4 (QAT weights)"] = eval_accuracy(mode4, val_loader)

    print("[3/3] UDP Mode 5 with QAT-trained weights ...")
    mode5 = UDPMode5MobileNet(fp32_from_qat, calib_loader,
                              n_calib=args.calib_batches).to(DEVICE)
    results["UDP Mode 5 (QAT weights)"] = eval_accuracy(mode5, val_loader)

    print()
    print("=" * 52)
    print("VWW QAT Evaluation")
    print("=" * 52)
    for name, acc in results.items():
        status = "PASS" if acc >= 80.0 else "FAIL"
        print(f"  {name:<32} {acc:>7.2f}%  {status}")
    print("=" * 52)

    out_path = os.path.join(os.path.dirname(args.ckpt), "vww_qat_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved -> {out_path}")


if __name__ == "__main__":
    main()
