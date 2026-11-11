"""
run_all_benchmarks.py — Master runner for all MLPerf Tiny benchmark evaluations

Runs accuracy evaluations for every benchmark that has a trained checkpoint.
Skips benchmarks whose checkpoint or dataset is missing (with a clear warning).

Usage:
  python run_all_benchmarks.py

Edit the PATHS section below to point to your checkpoints and datasets.
─────────────────────────────────────────────────────────────────────────────
"""

import json
import os
import subprocess
import sys

# ─── PATHS — edit these to match your setup ──────────────────────────────────

CHECKPOINTS = {
    "ic":  "./checkpoints/resnet8_best.pth",
    "kws": "./checkpoints/kws_best.pth",
    "vww": "./checkpoints/vww_best.pth",
    "ad":  "./checkpoints/ad_best.pth",
}

DATASETS = {
    "ic":  None,                                   # CIFAR-10 auto-downloads
    "kws": "./data/speech_commands",               # auto-downloaded by torchaudio
    "vww": "./data/vww_96",                        # manual download required
    "ad":  "./data/DCASE2020/dev_data/ToyCar",    # manual download required
}

# ─── Evaluation scripts ───────────────────────────────────────────────────────

EVAL_CMDS = {
    "ic": [
        sys.executable, "evaluate.py",
    ],
    "kws": [
        sys.executable, "kws/evaluate.py",
        "--ckpt", CHECKPOINTS["kws"],
        "--data-dir", DATASETS["kws"],
    ],
    "vww": [
        sys.executable, "vww/evaluate.py",
        "--ckpt", CHECKPOINTS["vww"],
        "--data-dir", DATASETS["vww"],
    ],
    "ad": [
        sys.executable, "ad/evaluate.py",
        "--ckpt", CHECKPOINTS["ad"],
        "--data-dir", DATASETS["ad"],
    ],
}

MLPerfTargets = {
    "ic":  {"metric": "Top-1 Acc",  "target": 85.0},
    "kws": {"metric": "Top-1 Acc",  "target": 90.0},
    "vww": {"metric": "Top-1 Acc",  "target": 80.0},
    "ad":  {"metric": "ROC-AUC",    "target": 0.85},
}

RESULT_FILES = {
    "ic":  "./checkpoints/ic_results.json",    # produced by evaluate.py
    "kws": "./checkpoints/kws_results.json",
    "vww": "./checkpoints/vww_results.json",
    "ad":  "./checkpoints/ad_results.json",
}

# ─── Helpers ─────────────────────────────────────────────────────────────────

def _missing(benchmark: str) -> str | None:
    ckpt = CHECKPOINTS[benchmark]
    if not os.path.exists(ckpt):
        return f"checkpoint not found: {ckpt}"
    ds = DATASETS[benchmark]
    if ds and not os.path.exists(ds):
        return f"dataset not found: {ds}"
    return None


def _run(benchmark: str):
    reason = _missing(benchmark)
    if reason:
        print(f"\n[SKIP] {benchmark.upper()} — {reason}")
        return False

    cmd = EVAL_CMDS[benchmark]
    print(f"\n{'='*60}")
    print(f"  Running: {benchmark.upper()} evaluation")
    print(f"{'='*60}")
    result = subprocess.run(cmd, cwd=os.path.dirname(__file__) or ".")
    return result.returncode == 0


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    benchmarks = ["ic", "kws", "vww", "ad"]
    ran, skipped, failed = [], [], []

    for bm in benchmarks:
        reason = _missing(bm)
        if reason:
            print(f"[SKIP] {bm.upper()}: {reason}")
            skipped.append(bm)
            continue
        ok = _run(bm)
        if ok:
            ran.append(bm)
        else:
            failed.append(bm)

    # ── Aggregate summary ────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("  MLPerf Tiny — All Benchmarks Summary")
    print(f"{'='*65}")
    print(f"{'Benchmark':<12} {'Model':<18} {'FP32':>10} {'Mode5':>10} {'Target':>8} {'Status':>6}")
    print("-" * 65)

    model_names = {
        "ic":  "ResNet-8",
        "kws": "DS-CNN",
        "vww": "MobileNetV1-0.25",
        "ad":  "AutoEncoder",
    }

    for bm in benchmarks:
        rf = RESULT_FILES[bm]
        tgt = MLPerfTargets[bm]
        if bm in skipped:
            print(f"{bm.upper():<12} {model_names[bm]:<18} {'N/A':>10} {'N/A':>10} {str(tgt['target']):>8} {'SKIP':>6}")
            continue
        if not os.path.exists(rf):
            print(f"{bm.upper():<12} {model_names[bm]:<18} {'NO JSON':>10} {'N/A':>10} {str(tgt['target']):>8} {'ERR':>6}")
            continue
        with open(rf) as f:
            res = json.load(f)

        fp32_key  = "FP32"
        mode5_key = "UDP Mode 5 (full INT4)" if bm != "ad" else None
        auc_key   = "FP32 AUC" if bm == "ad" else None

        if bm == "ad":
            fp32_val  = res.get(auc_key, float("nan"))
            mode5_val = res.get("INT4 weight-only AUC", float("nan"))
            target    = tgt["target"]
            fp32_str  = f"{fp32_val:.4f}"
            m5_str    = f"{mode5_val:.4f}"
            ok        = fp32_val >= target
        else:
            fp32_val  = res.get(fp32_key, float("nan"))
            mode5_val = res.get(mode5_key, float("nan"))
            target    = tgt["target"]
            fp32_str  = f"{fp32_val:.2f}%"
            m5_str    = f"{mode5_val:.2f}%"
            ok        = fp32_val >= target

        status = "PASS" if ok else "FAIL"
        print(f"{bm.upper():<12} {model_names[bm]:<18} "
              f"{fp32_str:>10} {m5_str:>10} {str(target):>8} {status:>6}")

    print(f"{'='*65}")
    print(f"Ran: {len(ran)}   Skipped: {len(skipped)}   Failed: {len(failed)}")
    if skipped:
        print(f"Skipped ({', '.join(s.upper() for s in skipped)}): "
              "checkpoint/dataset not found — see PATHS at top of this file.")


if __name__ == "__main__":
    main()
