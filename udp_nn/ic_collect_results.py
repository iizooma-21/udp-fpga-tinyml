"""
ic_collect_results.py — Run full IC evaluation and save to ic_results.json

Runs: evaluate.py (FP32), quantize.py (PTQ), udp_sim.py (Mode4/5),
      qat_evaluate.py (QAT-w4, QAT-full) and aggregates everything.

Usage:
  python ic_collect_results.py
  python ic_collect_results.py --weights checkpoints/best.pth
"""

import argparse
import json
import os
import re
import subprocess
import sys

CKPT_DIR = "./checkpoints"
DEVICE   = None  # auto

def _run_capture(cmd):
    """Run command and return stdout as string."""
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.stdout + r.stderr


def _extract_float(text, pattern):
    m = re.search(pattern, text)
    return float(m.group(1)) if m else float("nan")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights",     default="./checkpoints/best.pth")
    parser.add_argument("--qat-w4",     default="./checkpoints/qat_w4_best.pth")
    parser.add_argument("--qat-full",   default="./checkpoints/qat_full_best.pth")
    parser.add_argument("--data-dir",   default="./data")
    parser.add_argument("--calib-batches", type=int, default=8)
    args = parser.parse_args()

    results = {}

    # ── FP32 ─────────────────────────────────────────────────────────────────
    print("[1/5] FP32 ...")
    out = _run_capture([sys.executable, "evaluate.py",
                        "--weights", args.weights,
                        "--data-dir", args.data_dir])
    print(out[-500:])
    results["FP32"] = _extract_float(out, r"FP32 Top-1 Accuracy\s*:\s*([\d.]+)")

    # ── Naive INT4 PTQ ───────────────────────────────────────────────────────
    print("[2/5] Naive INT4 PTQ ...")
    out = _run_capture([sys.executable, "quantize.py",
                        "--weights", args.weights,
                        "--data-dir", args.data_dir])
    print(out[-500:])
    # quantize.py prints something like "INT4 Top-1: 52.10%"
    results["Naive INT4 PTQ"] = _extract_float(
        out, r"(?:INT4|Quantized).*?(\d+\.\d+)\s*%")

    # ── Mode 4 + Mode 5 ──────────────────────────────────────────────────────
    print("[3/5] UDP Mode 4 + Mode 5 ...")
    out = _run_capture([sys.executable, "udp_sim.py",
                        "--weights",       args.weights,
                        "--data-dir",      args.data_dir,
                        "--calib-batches", str(args.calib_batches)])
    print(out[-1000:])
    results["UDP Mode 4 (w-only)"]    = _extract_float(
        out, r"Mode 4.*?(\d+\.\d+)\s*%")
    results["UDP Mode 5 (full INT4)"] = _extract_float(
        out, r"Mode 5.*?(\d+\.\d+)\s*%")

    # ── QAT-w4 ───────────────────────────────────────────────────────────────
    if os.path.exists(args.qat_w4):
        print("[4/5] QAT-w4 ...")
        out = _run_capture([sys.executable, "qat_evaluate.py",
                            "--mode", "w4",
                            "--qat-weights", args.qat_w4,
                            "--data-dir",    args.data_dir])
        print(out[-500:])
        results["QAT-w4 direct"] = _extract_float(
            out, r"Direct.*?(\d+\.\d+)\s*%")
        results["UDP Mode 4 (QAT weights)"] = _extract_float(
            out, r"Mode\s*4.*?(\d+\.\d+)\s*%")
        results["UDP Mode 5 (QAT weights)"] = _extract_float(
            out, r"Mode\s*5.*?(\d+\.\d+)\s*%")
    else:
        print(f"[4/5] QAT-w4 checkpoint not found: {args.qat_w4}")

    # ── QAT-full ─────────────────────────────────────────────────────────────
    if os.path.exists(args.qat_full):
        print("[5/5] QAT-full ...")
        out = _run_capture([sys.executable, "qat_evaluate.py",
                            "--mode", "full",
                            "--qat-weights", args.qat_full,
                            "--data-dir",    args.data_dir])
        print(out[-500:])
        results["QAT-full direct"] = _extract_float(
            out, r"Direct.*?(\d+\.\d+)\s*%")
    else:
        print(f"[5/5] QAT-full checkpoint not found: {args.qat_full}")

    # ── Print + save ──────────────────────────────────────────────────────────
    print()
    print("=" * 55)
    print("IC (ResNet-8) — Full Results")
    print("=" * 55)
    for k, v in results.items():
        ok = v >= 85.0 if v == v else False
        print(f"  {k:<30} {v:>7.2f}%  {'PASS' if ok else 'FAIL'}")
    print("=" * 55)

    out_path = os.path.join(CKPT_DIR, "ic_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
