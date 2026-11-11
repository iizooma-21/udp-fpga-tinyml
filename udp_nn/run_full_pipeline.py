"""
run_full_pipeline.py — Complete MLPerf Tiny evaluation pipeline

Runs in order:
  1. Install required packages
  2. Download datasets (Speech Commands auto, VWW via TFDS, DCASE from Zenodo)
  3. IC  : evaluate existing checkpoint (FP32, PTQ, Mode4, Mode5, QAT-w4)
  4. KWS : train → evaluate (FP32, PTQ, Mode4, Mode5) → QAT-w4 → QAT-full evaluate
  5. VWW : download → train → evaluate → QAT-w4 → QAT-full evaluate
  6. AD  : download → train → evaluate (FP32-AUC, INT4-AUC)
  7. Print combined results table

All checkpoints saved to ./checkpoints/
All results saved to ./checkpoints/all_results.json

This script skips steps that are already done (checkpoint exists).
Run from inside the udp_nn/ directory:
  cd "d:/Uni work/Semester 8/BACHELOR/Codes/udp_nn"
  python run_full_pipeline.py
"""

import json
import os
import subprocess
import sys
import time

# ─── Config ──────────────────────────────────────────────────────────────────

ROOT      = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR  = os.path.join(ROOT, "checkpoints")
DATA_DIR  = os.path.join(ROOT, "data")

CKPTS = {
    # IC checkpoints kept under their original names from previous training
    "ic":          os.path.join(CKPT_DIR, "best.pth"),
    "ic_qat_w4":   os.path.join(CKPT_DIR, "qat_w4_best.pth"),
    "ic_qat_full": os.path.join(CKPT_DIR, "qat_full_best.pth"),
    # New benchmarks
    "kws":         os.path.join(CKPT_DIR, "kws_best.pth"),
    "kws_qat_w4":  os.path.join(CKPT_DIR, "kws_qat_w4.pth"),
    "vww":         os.path.join(CKPT_DIR, "vww_best.pth"),
    "vww_qat_w4":  os.path.join(CKPT_DIR, "vww_qat_w4.pth"),
    "ad":          os.path.join(CKPT_DIR, "ad_best.pth"),
}

DATA = {
    "sc":    os.path.join(DATA_DIR, "speech_commands"),  # auto-download
    "vww":   os.path.join(DATA_DIR, "vww_96"),
    "dcase": os.path.join(DATA_DIR, "DCASE2020"),
    "toyCar": os.path.join(DATA_DIR, "DCASE2020", "dev_data", "ToyCar"),
}

RESULTS_FILE = os.path.join(CKPT_DIR, "all_results.json")

# ─── Helpers ─────────────────────────────────────────────────────────────────

def _run(cmd, cwd=None, label=""):
    cwd = cwd or ROOT
    if label:
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"{'='*60}")
    t0  = time.time()
    ret = subprocess.run(cmd, cwd=cwd)
    elapsed = time.time() - t0
    ok  = ret.returncode == 0
    print(f"{'OK' if ok else 'FAILED'} in {elapsed:.0f}s")
    return ok


def _pip(*pkgs):
    _run([sys.executable, "-m", "pip", "install", "-q"] + list(pkgs),
         label=f"pip install {' '.join(pkgs)}")


def _exists_ckpt(key):
    return os.path.exists(CKPTS[key])


def _exists_data(key):
    p = DATA[key]
    if not os.path.exists(p):
        return False
    # For directories, check there are files inside
    if os.path.isdir(p):
        return any(True for _ in os.scandir(p))
    return True


def _section(title):
    print(f"\n{'#'*65}")
    print(f"  {title}")
    print(f"{'#'*65}")


# ─── Step 1: Install packages ─────────────────────────────────────────────────

def step_install():
    _section("STEP 1/7 — Install required packages")
    # Only install what's missing (skip torch/torchvision if already present)
    import importlib
    for pkg, import_name in [
        ("scikit-learn",    "sklearn"),
        ("torchaudio",      "torchaudio"),
        ("Pillow",          "PIL"),
    ]:
        try:
            importlib.import_module(import_name)
            print(f"  {pkg}: already installed")
        except ImportError:
            _pip(pkg)
    print("Packages ready.")


# ─── Step 2: Download datasets ───────────────────────────────────────────────

def step_download():
    _section("STEP 2/7 — Download datasets")

    # Speech Commands — auto-downloaded by torchaudio during KWS training
    print("Speech Commands: will auto-download during KWS training.")

    # VWW
    if not _exists_data("vww"):
        print("\nDownloading VWW dataset ...")
        _run([sys.executable, "vww/download_vww.py",
              "--output-dir", DATA["vww"]],
             label="VWW download")
    else:
        print(f"VWW dataset already at {DATA['vww']}")

    # DCASE (AD)
    if not _exists_data("toyCar"):
        print("\nDownloading DCASE2020 ToyCar dataset ...")
        _run([sys.executable, "ad/download_ad.py",
              "--output-dir", DATA["dcase"]],
             label="DCASE download")
    else:
        print(f"DCASE dataset already at {DATA['toyCar']}")


# ─── Step 3: IC evaluation ────────────────────────────────────────────────────

def step_ic():
    _section("STEP 3/7 — IC: ResNet-8 (CIFAR-10)")

    if not _exists_ckpt("ic"):
        print("No IC checkpoint found — training first ...")
        _run([sys.executable, "train.py"], label="IC train")

    # QAT (train if missing)
    if not _exists_ckpt("ic_qat_w4"):
        _run([sys.executable, "qat_train.py", "--mode", "w4", "--epochs", "20"],
             label="IC QAT-w4 train")

    # Collect all IC results into ic_results.json
    _run([sys.executable, "ic_collect_results.py",
          "--weights",   CKPTS["ic"],
          "--qat-w4",    CKPTS["ic_qat_w4"],
          "--qat-full",  CKPTS["ic_qat_full"]],
         label="IC full evaluation -> ic_results.json")


# ─── Step 4: KWS training + evaluation ───────────────────────────────────────

def step_kws():
    _section("STEP 4/7 — KWS: DS-CNN (Speech Commands)")

    if not _exists_ckpt("kws"):
        _run([sys.executable, "kws/train.py",
              "--data-dir", DATA["sc"],
              "--ckpt-dir", CKPT_DIR,
              "--epochs", "30"],
             label="KWS train (Speech Commands auto-download)")

    _run([sys.executable, "kws/evaluate.py",
          "--ckpt",     CKPTS["kws"],
          "--data-dir", DATA["sc"]],
         label="KWS evaluate (FP32 + PTQ + Mode4 + Mode5)")

    if not _exists_ckpt("kws_qat_w4"):
        _run([sys.executable, "kws/qat_train.py",
              "--ckpt",     CKPTS["kws"],
              "--data-dir", DATA["sc"],
              "--mode", "w4", "--epochs", "20"],
             label="KWS QAT-w4 train")

    _run([sys.executable, "kws/qat_evaluate.py",
          "--ckpt",     CKPTS["kws_qat_w4"],
          "--data-dir", DATA["sc"]],
         label="KWS QAT-w4 evaluate")


# ─── Step 5: VWW training + evaluation ───────────────────────────────────────

def step_vww():
    _section("STEP 5/7 — VWW: MobileNetV1 0.25× (Visual Wake Words)")

    if not _exists_data("vww"):
        print("VWW dataset not found — skipping VWW benchmark.")
        print("Re-run after dataset download completes.")
        return

    if not _exists_ckpt("vww"):
        _run([sys.executable, "vww/train.py",
              "--data-dir", DATA["vww"],
              "--ckpt-dir", CKPT_DIR,
              "--epochs", "50"],
             label="VWW train")

    _run([sys.executable, "vww/evaluate.py",
          "--ckpt",     CKPTS["vww"],
          "--data-dir", DATA["vww"]],
         label="VWW evaluate (FP32 + PTQ + Mode4 + Mode5)")

    if not _exists_ckpt("vww_qat_w4"):
        _run([sys.executable, "vww/qat_train.py",
              "--ckpt",     CKPTS["vww"],
              "--data-dir", DATA["vww"],
              "--mode", "w4", "--epochs", "20"],
             label="VWW QAT-w4 train")

    _run([sys.executable, "vww/qat_evaluate.py",
          "--ckpt",     CKPTS["vww_qat_w4"],
          "--data-dir", DATA["vww"]],
         label="VWW QAT-w4 evaluate")


# ─── Step 6: AD training + evaluation ────────────────────────────────────────

def step_ad():
    _section("STEP 6/7 — AD: AutoEncoder (DCASE2020 ToyCar)")

    if not _exists_data("toyCar"):
        print("DCASE dataset not found — skipping AD benchmark.")
        return

    if not _exists_ckpt("ad"):
        _run([sys.executable, "ad/train.py",
              "--data-dir", DATA["toyCar"],
              "--ckpt-dir", CKPT_DIR,
              "--epochs", "100"],
             label="AD train")

    _run([sys.executable, "ad/evaluate.py",
          "--ckpt",     CKPTS["ad"],
          "--data-dir", DATA["toyCar"]],
         label="AD evaluate (FP32 AUC + INT4 AUC)")


# ─── Step 7: Print combined results table ────────────────────────────────────

RESULT_FILES = {
    "IC":      os.path.join(CKPT_DIR, "ic_results.json"),
    "IC_QAT":  os.path.join(CKPT_DIR, "qat_results.json"),     # from qat_evaluate.py
    "KWS":     os.path.join(CKPT_DIR, "kws_results.json"),
    "KWS_QAT": os.path.join(CKPT_DIR, "kws_qat_results.json"),
    "VWW":     os.path.join(CKPT_DIR, "vww_results.json"),
    "VWW_QAT": os.path.join(CKPT_DIR, "vww_qat_results.json"),
    "AD":      os.path.join(CKPT_DIR, "ad_results.json"),
}
TARGETS = {
    "IC":  85.0,
    "KWS": 90.0,
    "VWW": 80.0,
    "AD":  0.85,
}
MODELS = {
    "IC":  "ResNet-8",
    "KWS": "DS-CNN",
    "VWW": "MobileNetV1-0.25",
    "AD":  "AutoEncoder",
}


def _load_json(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def step_results():
    _section("STEP 7/7 — Combined Results Table")

    # Load all result JSONs
    ic_res      = _load_json(RESULT_FILES["IC"])
    ic_qat_res  = _load_json(RESULT_FILES["IC_QAT"])
    kws_res     = _load_json(RESULT_FILES["KWS"])
    kws_qat_res = _load_json(RESULT_FILES["KWS_QAT"])
    vww_res     = _load_json(RESULT_FILES["VWW"])
    vww_qat_res = _load_json(RESULT_FILES["VWW_QAT"])
    ad_res      = _load_json(RESULT_FILES["AD"])

    combined = {
        "IC": ic_res, "IC_QAT": ic_qat_res,
        "KWS": kws_res, "KWS_QAT": kws_qat_res,
        "VWW": vww_res, "VWW_QAT": vww_qat_res,
        "AD": ad_res,
    }

    nan = float("nan")
    def _p(v, pct=True):
        if v != v:
            return "N/A"
        return f"{v:.2f}%" if pct else f"{v:.4f}"

    print()
    print("=" * 85)
    print(" MLPerf Tiny — Full Evaluation Summary")
    print("=" * 85)
    hdr = f"{'Benchmark':<10} {'Model':<18} {'FP32':>8} {'PTQ':>8} {'Mode4':>8} {'Mode5':>8} {'QAT-w4':>8}  {'Target':>7} {'Status':>6}"
    print(hdr)
    print("-" * 85)

    rows = [
        ("IC",  "ResNet-8",          ic_res,  ic_qat_res,  85.0,  True),
        ("KWS", "DS-CNN",            kws_res, kws_qat_res, 90.0,  True),
        ("VWW", "MobileNetV1-0.25",  vww_res, vww_qat_res, 80.0,  True),
        ("AD",  "AutoEncoder",       ad_res,  {},          0.85,  False),
    ]

    for bm, model, res, qat_res, target, is_acc in rows:
        if is_acc:
            fp32 = res.get("FP32",                   nan)
            ptq  = res.get("Naive INT4 PTQ",         nan)
            m4   = res.get("UDP Mode 4 (w-only)",    nan)
            m5   = res.get("UDP Mode 5 (full INT4)", nan)
            # QAT-w4: direct fake-quant accuracy from qat_evaluate
            qat_direct = qat_res.get("QAT-w4 direct",           nan)
            ok   = fp32 >= target if fp32 == fp32 else False
            print(f"{bm:<10} {model:<18} {_p(fp32):>8} {_p(ptq):>8} "
                  f"{_p(m4):>8} {_p(m5):>8} {_p(qat_direct):>8}  "
                  f"{str(target):>7} {'PASS' if ok else 'FAIL':>6}")
        else:
            # AD uses AUC
            fp32 = res.get("FP32 AUC",              nan)
            ptq  = res.get("INT4 weight-only AUC",  nan)
            ok   = fp32 >= target if fp32 == fp32 else False
            print(f"{bm:<10} {model:<18} {_p(fp32,False):>8} {_p(ptq,False):>8} "
                  f"{'N/A':>8} {'N/A':>8} {'N/A':>8}  "
                  f"{str(target):>7} {'PASS' if ok else 'FAIL':>6}")

    print("=" * 80)
    print()
    print("NOTE: For Timeloop hardware results (cycles, utilisation),")
    print("      run in WSL from the timeloop/ directory:")
    print("  ./run_mapper.sh all_benchmarks")
    print("  ./run_mapper.sh all_kws_opt_a  all_kws_opt_b")
    print("  ./run_mapper.sh all_vww_opt_a  all_vww_opt_b")
    print()

    # Save combined
    with open(RESULTS_FILE, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"Combined results saved -> {RESULTS_FILE}")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(CKPT_DIR, exist_ok=True)

    t_start = time.time()
    print("MLPerf Tiny — Full Evaluation Pipeline")
    print(f"Working directory : {ROOT}")
    print(f"Checkpoints       : {CKPT_DIR}")
    print(f"Data              : {DATA_DIR}")
    print()

    step_install()
    step_download()
    step_ic()
    step_kws()
    step_vww()
    step_ad()
    step_results()

    elapsed = time.time() - t_start
    h, m    = divmod(int(elapsed), 3600)
    m, s    = divmod(m, 60)
    print(f"\nTotal wall time: {h}h {m}m {s}s")


if __name__ == "__main__":
    main()
