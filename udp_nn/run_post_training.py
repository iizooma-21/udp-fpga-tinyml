"""
run_post_training.py — Evaluate + QAT for all benchmarks after training.

Polls for checkpoint existence every 60s and runs the next step automatically.
Leave this running; it will pick up each benchmark as it finishes training.

Usage:
  python -u run_post_training.py
"""

import json
import os
import subprocess
import sys
import time

ROOT     = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(ROOT, "checkpoints")

CKPTS = {
    "kws":        os.path.join(CKPT_DIR, "kws_best.pth"),
    "kws_qat_w4": os.path.join(CKPT_DIR, "kws_qat_w4.pth"),
    "vww":        os.path.join(CKPT_DIR, "vww_best.pth"),
    "vww_qat_w4": os.path.join(CKPT_DIR, "vww_qat_w4.pth"),
    "ad":         os.path.join(CKPT_DIR, "ad_best.pth"),
}

DATA = {
    "sc":    os.path.join(ROOT, "data", "speech_commands"),
    "vww":   os.path.join(ROOT, "data", "vww_96"),
    "toyCar": os.path.join(ROOT, "data", "DCASE2020", "dev_data", "ToyCar"),
}

DONE = set()   # track which steps have been completed in this session


def _run(cmd, label=""):
    if label:
        print(f"\n[RUN] {label}")
    t0  = time.time()
    ret = subprocess.run(cmd, cwd=ROOT)
    ok  = ret.returncode == 0
    print(f"  {'OK' if ok else 'FAILED'} ({time.time()-t0:.0f}s)")
    return ok


def _exists(key):
    return os.path.exists(CKPTS.get(key, ""))


def _vww_ready():
    """Check VWW dataset has enough images."""
    p = os.path.join(DATA["vww"], "train", "person")
    if not os.path.exists(p):
        return False
    return len([f for f in os.listdir(p) if f.endswith(".jpg")]) > 500


def step_kws():
    if "kws_eval" in DONE:
        return
    if not _exists("kws"):
        return  # still training
    print("\n===== KWS: checkpoint found — running evaluation + QAT =====")

    _run([sys.executable, "-u", "kws/evaluate.py",
          "--ckpt", CKPTS["kws"],
          "--data-dir", DATA["sc"]], "KWS FP32+PTQ+Mode4+Mode5")

    if not _exists("kws_qat_w4"):
        _run([sys.executable, "-u", "kws/qat_train.py",
              "--ckpt", CKPTS["kws"],
              "--data-dir", DATA["sc"],
              "--mode", "w4", "--epochs", "20"], "KWS QAT-w4 train")

    if _exists("kws_qat_w4"):
        _run([sys.executable, "-u", "kws/qat_evaluate.py",
              "--ckpt", CKPTS["kws_qat_w4"],
              "--data-dir", DATA["sc"]], "KWS QAT-w4 evaluate")

    DONE.add("kws_eval")
    print("===== KWS done =====")


def step_vww():
    if "vww_eval" in DONE:
        return
    if not _exists("vww"):
        # Try to train if dataset is ready but checkpoint missing
        if _vww_ready():
            print("\n===== VWW: dataset ready — training =====")
            _run([sys.executable, "-u", "vww/train.py",
                  "--data-dir", DATA["vww"],
                  "--ckpt-dir", CKPT_DIR,
                  "--epochs", "50"], "VWW train")
        return

    print("\n===== VWW: checkpoint found — running evaluation + QAT =====")

    _run([sys.executable, "-u", "vww/evaluate.py",
          "--ckpt", CKPTS["vww"],
          "--data-dir", DATA["vww"]], "VWW FP32+PTQ+Mode4+Mode5")

    if not _exists("vww_qat_w4"):
        _run([sys.executable, "-u", "vww/qat_train.py",
              "--ckpt", CKPTS["vww"],
              "--data-dir", DATA["vww"],
              "--mode", "w4", "--epochs", "20"], "VWW QAT-w4 train")

    if _exists("vww_qat_w4"):
        _run([sys.executable, "-u", "vww/qat_evaluate.py",
              "--ckpt", CKPTS["vww_qat_w4"],
              "--data-dir", DATA["vww"]], "VWW QAT-w4 evaluate")

    DONE.add("vww_eval")
    print("===== VWW done =====")


def step_ad():
    if "ad_eval" in DONE:
        return
    if not _exists("ad"):
        return  # still training

    print("\n===== AD: checkpoint found — running evaluation =====")
    _run([sys.executable, "-u", "ad/evaluate.py",
          "--ckpt", CKPTS["ad"],
          "--data-dir", DATA["toyCar"]], "AD evaluate")

    DONE.add("ad_eval")
    print("===== AD done =====")


def print_combined_table():
    """Print a combined results table from all saved JSON files."""
    result_files = {
        "IC":      os.path.join(CKPT_DIR, "ic_results.json"),
        "KWS":     os.path.join(CKPT_DIR, "kws_results.json"),
        "KWS_QAT": os.path.join(CKPT_DIR, "kws_qat_results.json"),
        "VWW":     os.path.join(CKPT_DIR, "vww_results.json"),
        "VWW_QAT": os.path.join(CKPT_DIR, "vww_qat_results.json"),
        "AD":      os.path.join(CKPT_DIR, "ad_results.json"),
    }

    data = {}
    for bm, path in result_files.items():
        if os.path.exists(path):
            with open(path) as f:
                data[bm] = json.load(f)

    if not data:
        return

    nan = float("nan")
    def g(d, *keys):
        for k in keys:
            v = d.get(k, nan)
            if v == v:
                return v
        return nan

    print("\n" + "=" * 85)
    print(" MLPerf Tiny — Combined Results (as of now)")
    print("=" * 85)
    fmt = "{:<5} {:<16} {:>8} {:>8} {:>8} {:>8} {:>8}  {:>7} {:>6}"
    print(fmt.format("BM", "Model", "FP32", "PTQ", "Mode4", "Mode5", "QAT-w4", "Target", "Status"))
    print("-" * 85)

    rows = [
        ("IC",  "ResNet-8",          data.get("IC", {}),  data.get("IC", {}),   85.0, True),
        ("KWS", "DS-CNN",            data.get("KWS", {}), data.get("KWS_QAT",{}), 90.0, True),
        ("VWW", "MobileNet-0.25",    data.get("VWW", {}), data.get("VWW_QAT",{}), 80.0, True),
        ("AD",  "AutoEncoder",       data.get("AD", {}),  {},                   0.85, False),
    ]

    for bm, model, res, qat_res, target, is_acc in rows:
        if is_acc:
            fp32 = g(res, "FP32")
            ptq  = g(res, "Naive INT4 PTQ")
            m4   = g(res, "UDP Mode 4 (w-only)")
            m5   = g(res, "UDP Mode 5 (full INT4)")
            qat  = g(qat_res, "QAT-w4 direct")
            ok   = fp32 >= target if fp32 == fp32 else False
            pct  = lambda v: f"{v:.2f}%" if v == v else "---"
            print(fmt.format(bm, model,
                  pct(fp32), pct(ptq), pct(m4), pct(m5), pct(qat),
                  f"{target}%", "PASS" if ok else "FAIL"))
        else:
            fp32 = g(res, "FP32 AUC")
            ptq  = g(res, "INT4 weight-only AUC")
            ok   = fp32 >= target if fp32 == fp32 else False
            flt  = lambda v: f"{v:.4f}" if v == v else "---"
            print(fmt.format(bm, model,
                  flt(fp32), flt(ptq), "---", "---", "---",
                  str(target), "PASS" if ok else "FAIL"))

    print("=" * 85)


def main():
    print("run_post_training.py — polling every 60s for new checkpoints")
    print("Press Ctrl+C to stop.\n")

    all_done = {"kws_eval", "vww_eval", "ad_eval"}

    while True:
        step_ad()
        step_kws()
        step_vww()

        print_combined_table()

        if DONE >= all_done:
            print("\nAll benchmarks complete!")
            break

        # Check if we have nothing left to wait for
        kws_done = "kws_eval" in DONE
        vww_done = "vww_eval" in DONE
        ad_done  = "ad_eval" in DONE

        remaining = []
        if not kws_done: remaining.append("KWS")
        if not vww_done: remaining.append("VWW")
        if not ad_done:  remaining.append("AD")

        print(f"\nWaiting for: {', '.join(remaining)}. Next check in 60s ...")
        time.sleep(60)


if __name__ == "__main__":
    main()
