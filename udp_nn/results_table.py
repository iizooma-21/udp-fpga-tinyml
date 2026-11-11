"""
results_table.py — Block E: final evaluation summary

Aggregates results from Blocks A–D into one report:

  Accuracy track (PyTorch):
    - FP32 baseline
    - Naive INT4 PTQ (per-tensor)
    - UDP Mode 4 — weight-only INT4
    - UDP Mode 5 — full INT4 pipeline with post-processing

  Performance track (Timeloop, 3 representative layers):
    - Baseline 12x12 (144 PEs)
    - Opt-A 8x12   (96 PEs)
    - Opt-B 8x9    (72 PEs)

Usage:
  python results_table.py                    # full report (runs accuracy eval)
  python results_table.py --timeloop-only    # hardware table only (fast)
  python results_table.py --save results/    # also write markdown + json
"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from evaluate import get_test_loader, run_evaluation
from quantize import calibrate as calibrate_naive, evaluate as eval_naive
from quantize import insert_fake_quant
from resnet8 import build_resnet8
from udp_sim import (
    UDPSim,
    build_w4_model,
    calibrate as calibrate_udp,
    eval_model,
    eval_sim,
)

# ── Constants ────────────────────────────────────────────────────────────────
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)
MLPERF_TARGET = 85.0
UDP_PACKING   = 6   # 6 int4 MACs per PE per cycle (P_w=3, P_a=2)
CLK_MHZ       = 200

LAYERS = ("stem_conv", "s1_conv1", "classifier_dense")

ARCHITECTURES = (
    {"id": "baseline", "label": "Baseline 12x12", "pe": 144, "suffix": ""},
    {"id": "opt_a",    "label": "Opt-A 8x12",     "pe": 96,  "suffix": "_opt_a"},
    {"id": "opt_b",    "label": "Opt-B 8x9",      "pe": 72,  "suffix": "_opt_b"},
)


@dataclass
class TimeloopResult:
    layer: str
    arch_id: str
    pe_count: int
    utilized: int | None
    total_pes: int | None
    utilization: float | None
    cycles: int | None
    udp_cycles: int | None
    computes: int | None


@dataclass
class AccuracyResult:
    name: str
    task: str
    accuracy: float
    mlperf_pass: bool


def parse_timeloop_stats(path: Path) -> dict:
    """Extract key metrics from a timeloop-mapper.stats.txt file."""
    text = path.read_text(encoding="utf-8", errors="replace")
    out: dict = {}

    m = re.search(r"Instances\s+:\s+(\d+)\s+\((\d+)\*(\d+)\)", text)
    if m:
        out["total_pes"] = int(m.group(1))
        out["mesh_x"] = int(m.group(2))
        out["mesh_y"] = int(m.group(3))

    m = re.search(r"Utilized instances\s+:\s+(\d+)", text)
    if m:
        out["utilized"] = int(m.group(1))

    for key, pattern in [
        ("utilization", r"^Utilization:\s+([\d.]+)"),
        ("cycles",      r"^Cycles:\s+(\d+)"),
        ("computes",    r"^Computes =\s+(\d+)"),
    ]:
        m = re.search(pattern, text, re.MULTILINE)
        if m:
            out[key] = float(m.group(1)) if key == "utilization" else int(m.group(1))

    if "cycles" in out:
        out["udp_cycles"] = max(1, out["cycles"] // UDP_PACKING)

    return out


def load_timeloop_results(outputs_dir: Path) -> list[TimeloopResult]:
    rows: list[TimeloopResult] = []
    for arch in ARCHITECTURES:
        for layer in LAYERS:
            fname = f"{layer}{arch['suffix']}.stats.txt"
            path = outputs_dir / fname
            if not path.exists():
                continue
            stats = parse_timeloop_stats(path)
            rows.append(TimeloopResult(
                layer=layer,
                arch_id=arch["id"],
                pe_count=arch["pe"],
                utilized=stats.get("utilized"),
                total_pes=stats.get("total_pes", arch["pe"]),
                utilization=stats.get("utilization"),
                cycles=stats.get("cycles"),
                udp_cycles=stats.get("udp_cycles"),
                computes=stats.get("computes"),
            ))
    return rows


def load_model(weights: Path, conv_filters: int, device: torch.device):
    model = build_resnet8(conv_filters=conv_filters).to(device)
    state = torch.load(weights, map_location=device)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def run_accuracy_suite(
    weights: Path,
    data_dir: Path,
    conv_filters: int,
    batch_size: int,
    num_workers: int,
    calib_batches: int,
) -> list[AccuracyResult]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(weights, conv_filters, device)

    test_loader = get_test_loader(str(data_dir), batch_size, num_workers)

    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    train_ds = datasets.CIFAR10(str(data_dir), train=True, download=True, transform=tf)
    test_ds  = datasets.CIFAR10(str(data_dir), train=False, download=True, transform=tf)
    calib_loader = DataLoader(train_ds, batch_size, shuffle=True, num_workers=num_workers)
    test_loader2 = DataLoader(test_ds, batch_size, shuffle=False, num_workers=num_workers)

    results: list[AccuracyResult] = []

    # Task 2 — FP32
    preds, labels, _ = run_evaluation(model, test_loader, device)
    fp32 = preds.eq(labels).float().mean().item() * 100.0
    results.append(AccuracyResult("FP32 baseline", "Task 2", fp32, fp32 >= MLPERF_TARGET))

    # Task 3 — naive INT4 PTQ
    qmodel = build_resnet8(conv_filters=conv_filters)
    qmodel.load_state_dict(model.state_dict())
    qmodel = insert_fake_quant(qmodel).to(device)
    calibrate_naive(qmodel, calib_loader, device, 8)
    int4_acc, _ = eval_naive(qmodel, test_loader2, device)
    results.append(AccuracyResult(
        "Naive INT4 PTQ (per-tensor)", "Task 3", int4_acc, int4_acc >= MLPERF_TARGET
    ))

    # Task 4 — UDP without post-processing
    m4 = build_w4_model(model).to(device)
    acc4, _ = eval_model(m4, test_loader2, device)
    results.append(AccuracyResult(
        "UDP no post-proc (weight-only INT4)", "Task 4", acc4, acc4 >= MLPERF_TARGET
    ))

    # Task 5 — UDP with post-processing
    stats = calibrate_udp(model, calib_loader, device, calib_batches)
    sim = UDPSim(model, stats, device)
    acc5, _ = eval_sim(sim, test_loader2, device)
    results.append(AccuracyResult(
        "UDP with post-proc (full INT4)", "Task 5", acc5, acc5 >= MLPERF_TARGET
    ))

    return results


def pct(v: float | None) -> str:
    return f"{v * 100:.0f}%" if v is not None else "n/a"


def fmt_int(v: int | None) -> str:
    return f"{v:,}" if v is not None else "n/a"


def print_accuracy_table(rows: list[AccuracyResult]) -> None:
    print("\n" + "=" * 72)
    print("  ACCURACY EVALUATION — MLPerf Tiny ResNet-8 / CIFAR-10")
    print("=" * 72)
    print(f"  {'Scenario':<38} {'Task':<8} {'Top-1':>8}  {'>=85%':>6}")
    print("  " + "-" * 66)
    for r in rows:
        status = "PASS" if r.mlperf_pass else "FAIL"
        print(f"  {r.name:<38} {r.task:<8} {r.accuracy:>7.2f}%  {status:>6}")
    if len(rows) >= 4:
        delta = rows[3].accuracy - rows[2].accuracy
        print("  " + "-" * 66)
        print(f"  Post-proc activation cost (Task 5 - Task 4): {delta:+.2f}%")
    print("=" * 72)


def print_timeloop_table(rows: list[TimeloopResult]) -> None:
    print("\n" + "=" * 96)
    print("  TIMELOOP PERFORMANCE — 3 representative layers")
    print(f"  Timeloop cycles are 1-MAC/PE; UDP cycles = Timeloop cycles / {UDP_PACKING}")
    print("=" * 96)

    arch_labels = {a["id"]: a["label"] for a in ARCHITECTURES}
    arch_pes    = {a["id"]: a["pe"] for a in ARCHITECTURES}

    print(f"  {'Architecture':<18} {'PEs':>4}  {'Layer':<18} "
          f"{'Util':>6}  {'TL cycles':>10}  {'UDP cycles':>10}")
    print("  " + "-" * 88)

    totals: dict[str, int] = {a["id"]: 0 for a in ARCHITECTURES}
    for arch_id in ("baseline", "opt_a", "opt_b"):
        arch_rows = [r for r in rows if r.arch_id == arch_id]
        if not arch_rows:
            continue
        for i, r in enumerate(arch_rows):
            label = arch_labels[arch_id] if i == 0 else ""
            pes   = arch_pes[arch_id] if i == 0 else ""
            util  = pct(r.utilization) if r.utilization is not None else "n/a"
            print(f"  {label:<18} {str(pes):>4}  {r.layer:<18} "
                  f"{util:>6}  {fmt_int(r.cycles):>10}  {fmt_int(r.udp_cycles):>10}")
            if r.udp_cycles:
                totals[arch_id] += r.udp_cycles
        print("  " + "-" * 88)
        print(f"  {'Total UDP cycles (3 layers)':<18} {'':>4}  {'':<18} "
              f"{'':>6}  {'':>10}  {fmt_int(totals[arch_id]):>10}")
        lat_ms = totals[arch_id] / (CLK_MHZ * 1e3) if totals[arch_id] else 0
        print(f"  {'Latency @200 MHz':<18} {'':>4}  {'':<18} "
              f"{'':>6}  {'':>10}  {lat_ms:>9.4f} ms")
        print()

    print("=" * 96)
    print("  Recommendation: Opt-A (8x12, 96 PEs)")
    print("    - s1_conv1 reaches 100% utilization (vs 67% baseline)")
    print("    - Same UDP cycle count as baseline on stem_conv and s1_conv1")
    print("    - 33% fewer PEs than the 12x12 baseline")
    print("=" * 96)


def build_markdown(
    accuracy: list[AccuracyResult] | None,
    timeloop: list[TimeloopResult],
) -> str:
    lines = ["# ResNet-8 End-to-End Evaluation Summary\n"]

    if accuracy:
        lines.append("## Accuracy (CIFAR-10 top-1)\n")
        lines.append("| Scenario | Task | Top-1 | MLPerf (>=85%) |")
        lines.append("|---|---|---:|---|")
        for r in accuracy:
            lines.append(
                f"| {r.name} | {r.task} | {r.accuracy:.2f}% | "
                f"{'PASS' if r.mlperf_pass else 'FAIL'} |"
            )
        if len(accuracy) >= 4:
            d = accuracy[3].accuracy - accuracy[2].accuracy
            lines.append(f"\nPost-proc activation cost (Task 5 - Task 4): **{d:+.2f}%**\n")

    lines.append("## Timeloop performance (3 layers)\n")
    lines.append(
        f"UDP cycles = Timeloop cycles / {UDP_PACKING} "
        f"(6 int4 MACs per PE per cycle)\n"
    )
    lines.append(
        "| Architecture | PEs | Layer | Utilization | Timeloop cycles "
        "| UDP cycles |"
    )
    lines.append("|---|---:|---|---:|---:|---:|")
    arch_meta = {a["id"]: a for a in ARCHITECTURES}
    for r in timeloop:
        a = arch_meta[r.arch_id]
        util = f"{r.utilization * 100:.0f}%" if r.utilization is not None else "n/a"
        lines.append(
            f"| {a['label']} | {a['pe']} | {r.layer} | {util} | "
            f"{r.cycles or 'n/a'} | {r.udp_cycles or 'n/a'} |"
        )

    totals: dict[str, int] = {}
    for r in timeloop:
        if r.udp_cycles:
            totals[r.arch_id] = totals.get(r.arch_id, 0) + r.udp_cycles
    lines.append("\n### Total UDP cycles (3 layers)\n")
    lines.append("| Architecture | Total UDP cycles | Latency @200 MHz |")
    lines.append("|---|---:|---:|")
    for arch_id in ("baseline", "opt_a", "opt_b"):
        if arch_id not in totals:
            continue
        a = arch_meta[arch_id]
        lat = totals[arch_id] / (CLK_MHZ * 1e3)
        lines.append(f"| {a['label']} | {totals[arch_id]:,} | {lat:.4f} ms |")

    lines.append(
        "\n**Recommendation:** Opt-A (8x12, 96 PEs) — best utilization/area trade-off.\n"
    )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Block E — final evaluation summary")
    parser.add_argument("--weights",       default="./checkpoints/best.pth")
    parser.add_argument("--data-dir",      default="./data")
    parser.add_argument("--timeloop-dir",  default="./timeloop/outputs")
    parser.add_argument("--save",          default="./results",
                        help="Directory to write summary markdown + json")
    parser.add_argument("--timeloop-only", action="store_true",
                        help="Skip accuracy evaluation (hardware table only)")
    parser.add_argument("--batch-size",    type=int, default=256)
    parser.add_argument("--num-workers",   type=int, default=2)
    parser.add_argument("--calib-batches", type=int, default=16)
    parser.add_argument("--conv-filters",  type=int, default=26)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    outputs_dir = (root / args.timeloop_dir).resolve()
    weights     = (root / args.weights).resolve()
    data_dir    = (root / args.data_dir).resolve()
    save_dir    = (root / args.save).resolve()

    print("Block E — Final Evaluation Summary")
    print(f"  Weights      : {weights}")
    print(f"  Timeloop dir : {outputs_dir}")

    timeloop_rows = load_timeloop_results(outputs_dir)
    if not timeloop_rows:
        print("\nWARNING: no Timeloop stats found. Run ./run_mapper.sh in WSL first.")

    accuracy_rows: list[AccuracyResult] | None = None
    if not args.timeloop_only:
        if not weights.exists():
            print(f"\nERROR: weights not found: {weights}")
            return
        print("\nRunning accuracy evaluations (FP32, INT4, UDP modes)...")
        t0 = time.time()
        accuracy_rows = run_accuracy_suite(
            weights, data_dir, args.conv_filters,
            args.batch_size, args.num_workers, args.calib_batches,
        )
        print(f"  Accuracy suite finished in {time.time() - t0:.1f}s")
        print_accuracy_table(accuracy_rows)
    else:
        print("\nSkipping accuracy evaluation (--timeloop-only).")

    print_timeloop_table(timeloop_rows)

    save_dir.mkdir(parents=True, exist_ok=True)
    md_path   = save_dir / "evaluation_summary.md"
    json_path = save_dir / "evaluation_summary.json"

    md_text = build_markdown(accuracy_rows, timeloop_rows)
    md_path.write_text(md_text, encoding="utf-8")

    payload = {
        "accuracy": [asdict(r) for r in accuracy_rows] if accuracy_rows else None,
        "timeloop": [asdict(r) for r in timeloop_rows],
        "constants": {
            "mlperf_target": MLPERF_TARGET,
            "udp_packing": UDP_PACKING,
            "clock_mhz": CLK_MHZ,
        },
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"\nSaved report:")
    print(f"  {md_path}")
    print(f"  {json_path}")


if __name__ == "__main__":
    main()
