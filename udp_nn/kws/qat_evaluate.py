"""
kws/qat_evaluate.py — Evaluate KWS QAT checkpoint

Loads a kws_qat_w4.pth checkpoint and reports:
  1. QAT-w4 direct accuracy  (fake-quant active)
  2. UDP Mode 4 with QAT weights  (same INT4 weights, standard forward)
  3. UDP Mode 5 with QAT weights  (INT4 weights + calibrated act quant)

Results written to checkpoints/kws_qat_results.json

Usage:
  python kws/qat_evaluate.py --ckpt checkpoints/kws_qat_w4.pth
"""

import argparse
import json
import os
import sys
from copy import deepcopy

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from kws.model       import build_dscnn
from kws.dataset     import SpeechCommandsDataset, CLASSES
from kws.qat_train   import QATWeightDSCNN, fq_weight_ste
from kws.evaluate    import UDPMode4DSCNN, UDPMode5DSCNN, eval_accuracy

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",          required=True)
    parser.add_argument("--data-dir",      default="./data/speech_commands")
    parser.add_argument("--calib-batches", type=int, default=8)
    parser.add_argument("--batch-size",    type=int, default=256)
    parser.add_argument("--num-workers",   type=int, default=2)
    args = parser.parse_args()

    print(f"Device : {DEVICE}")

    ckpt        = torch.load(args.ckpt, map_location=DEVICE)
    num_filters = ckpt.get("num_filters", 64)
    num_classes = ckpt.get("num_classes", len(CLASSES))

    # Rebuild QAT model and load QAT-trained weights
    fp32 = build_dscnn(num_classes=num_classes, num_filters=num_filters).to(DEVICE)
    if "base_model" in ckpt:
        fp32.load_state_dict(ckpt["base_model"])

    qat_model = QATWeightDSCNN(fp32).to(DEVICE)
    qat_model.load_state_dict(ckpt["model"])
    qat_model.eval()

    # Also build a plain FP32 model from QAT-trained weights
    # (weights are fine-tuned under fake-quant, but the underlying m is still FP32)
    fp32_from_qat = build_dscnn(num_classes=num_classes, num_filters=num_filters).to(DEVICE)
    fp32_from_qat.load_state_dict(qat_model.m.state_dict())

    test_ds  = SpeechCommandsDataset(args.data_dir, split="testing",  download=True)
    train_ds = SpeechCommandsDataset(args.data_dir, split="training", download=True)
    test_loader  = DataLoader(test_ds,  args.batch_size, shuffle=False,
                              num_workers=args.num_workers)
    calib_loader = DataLoader(train_ds, args.batch_size, shuffle=True,
                              num_workers=args.num_workers)

    results = {}

    print("\n[1/3] QAT-w4 direct accuracy (fake-quant active) ...")
    results["QAT-w4 direct"] = eval_accuracy(qat_model, test_loader)

    print("[2/3] UDP Mode 4 with QAT-trained weights ...")
    mode4 = UDPMode4DSCNN(fp32_from_qat).to(DEVICE)
    results["UDP Mode 4 (QAT weights)"] = eval_accuracy(mode4, test_loader)

    print("[3/3] UDP Mode 5 with QAT-trained weights ...")
    mode5 = UDPMode5DSCNN(fp32_from_qat, calib_loader,
                           n_calib=args.calib_batches).to(DEVICE)
    results["UDP Mode 5 (QAT weights)"] = eval_accuracy(mode5, test_loader)

    print()
    print("=" * 52)
    print("KWS QAT Evaluation")
    print("=" * 52)
    for name, acc in results.items():
        status = "PASS" if acc >= 90.0 else "FAIL"
        print(f"  {name:<32} {acc:>7.2f}%  {status}")
    print("=" * 52)

    out_path = os.path.join(os.path.dirname(args.ckpt), "kws_qat_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved -> {out_path}")


if __name__ == "__main__":
    main()
