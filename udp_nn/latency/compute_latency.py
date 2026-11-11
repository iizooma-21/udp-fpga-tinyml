#!/usr/bin/env python3
"""
compute_latency.py
==================
Computes end-to-end network latency for all MLPerf Tiny benchmarks.

Strategy
--------
For layers that were already run through Timeloop (stem_conv, s1_conv1,
classifier_dense, kws_*, vww_*, ad_dense1), use the actual Timeloop cycles.

For layers that Timeloop could not map (due to constraint mismatch), use
analytical MAC-proportional estimation:

    est_cycles(L) = ref_cycles * MACs(L) / MACs(ref)

where ref is the closest same-type layer with a measured result.
This is valid because, for the same layer type on a fixed array, the
cycle count scales linearly with the number of MACs (constant efficiency).
Estimated values are marked with "(est)" in the report.

Latency model
-------------
    real_cycles = timeloop_cycles / 6     (UDP: 6 MACs/PE/cycle)
    latency_us  = real_cycles / 200e6 * 1e6

Usage
-----
  python compute_latency.py
  (or with optional flags, see --help)
"""

import argparse
import os
import re
import sys

# ---------------------------------------------------------------------------
# Hard-wired Timeloop results from all existing outputs
# Format: {(stem, arch_tag): cycles}
# ---------------------------------------------------------------------------
# These are read from the actual stats files at runtime; the dict below is
# populated by load_existing_results().
KNOWN_CYCLES: dict[tuple[str, str], int] = {}

# Arch tags
TAGS = ["base", "opt_a", "opt_b"]

UDP_MAC_FACTOR = 6
DEFAULT_CLOCK_MHZ = 200.0

# ---------------------------------------------------------------------------
# Network layer definitions
# Each entry: (name, C, K, R, S, P, Q, stride, count, ref_stem)
#
#   ref_stem: the TL output stem to use for cycle estimation when the
#             layer's own result is missing.  None = no fallback needed
#             (layer has measured results for all three archs).
# ---------------------------------------------------------------------------

IC_LAYERS = [
    # name         C    K   R  S   P   Q  s  cnt  ref_stem
    ("ic_stem",    3,  26,  3, 3, 32, 32, 1,  1,  None),          # measured
    ("ic_s1c",    26,  26,  3, 3, 32, 32, 1,  2,  None),          # measured (x2)
    ("ic_s2c1",   26,  52,  3, 3, 16, 16, 2,  1,  "s1_conv1"),    # estimate from s1
    ("ic_s2c2",   52,  52,  3, 3, 16, 16, 1,  1,  "s1_conv1"),
    ("ic_s2sc",   26,  52,  1, 1, 16, 16, 2,  1,  "classifier_dense"),  # 1x1 conv
    ("ic_s3c1",   52, 104,  3, 3,  8,  8, 2,  1,  "s1_conv1"),
    ("ic_s3c2",  104, 104,  3, 3,  8,  8, 1,  1,  "s1_conv1"),
    ("ic_s3sc",   52, 104,  1, 1,  8,  8, 2,  1,  "classifier_dense"),
    ("ic_cls",   104,  10,  1, 1,  1,  1, 1,  1,  None),          # measured
]

IC_STEMS = {      # layer name -> existing TL stem (for measured layers)
    "ic_stem": "stem_conv",
    "ic_s1c":  "s1_conv1",
    "ic_cls":  "classifier_dense",
}

# ---------------------------------------------------------------------------
KWS_LAYERS = [
    # name          C   K   R   S   P  Q  s  cnt  ref_stem
    ("kws_conv1",   1, 64, 10,  4, 25, 5, 2,  1,  None),   # measured
    ("kws_dw",      1, 64,  3,  3, 25, 5, 1,  4,  None),   # measured (x4)
    ("kws_pw",     64, 64,  1,  1, 25, 5, 1,  4,  None),   # measured (x4)
    ("kws_fc",     64, 12,  1,  1,  1, 1, 1,  1,  "kws_pw1"),  # 1x1 FC
]

KWS_STEMS = {
    "kws_conv1": "kws_conv1",
    "kws_dw":    "kws_dw1",
    "kws_pw":    "kws_pw1",
}

# KWS conv1 is incompatible with opt_a/opt_b; fall back to base cycles
KWS_FALLBACK_BASE = {"kws_conv1"}

# ---------------------------------------------------------------------------
VWW_LAYERS = [
    # name          C    K   R  S   P   Q  s  cnt  ref_stem
    ("vww_stem",    3,   8,  3, 3, 48, 48, 2,  1,  None),     # measured
    ("vww_dw0",     1,   8,  3, 3, 48, 48, 1,  1,  "vww_dw5"),  # DW type
    ("vww_pw0",     8,  16,  1, 1, 48, 48, 1,  1,  "vww_pw5"),
    ("vww_dw1",     1,  16,  3, 3, 24, 24, 2,  1,  "vww_dw5"),
    ("vww_pw1",    16,  32,  1, 1, 24, 24, 1,  1,  "vww_pw5"),
    ("vww_dw2",     1,  32,  3, 3, 24, 24, 1,  1,  "vww_dw5"),
    ("vww_pw2",    32,  32,  1, 1, 24, 24, 1,  1,  "vww_pw5"),
    ("vww_dw3",     1,  32,  3, 3, 12, 12, 2,  1,  None),     # measured
    ("vww_pw3",    32,  64,  1, 1, 12, 12, 1,  1,  None),     # measured
    ("vww_dw4",     1,  64,  3, 3, 12, 12, 1,  1,  "vww_dw5"),
    ("vww_pw4",    64,  64,  1, 1, 12, 12, 1,  1,  "vww_pw5"),
    ("vww_dw5b",    1,  64,  3, 3,  6,  6, 2,  1,  "vww_dw5"),
    ("vww_pw5b",   64, 128,  1, 1,  6,  6, 1,  1,  "vww_pw5"),
    ("vww_dw6",     1, 128,  3, 3,  6,  6, 1,  5,  "vww_dw5"),
    ("vww_pw6",   128, 128,  1, 1,  6,  6, 1,  5,  "vww_pw5"),
    ("vww_dw11",    1, 128,  3, 3,  3,  3, 2,  1,  "vww_dw5"),
    ("vww_pw11",  128, 256,  1, 1,  3,  3, 1,  1,  "vww_pw5"),
    ("vww_dw12",    1, 256,  3, 3,  3,  3, 1,  1,  "vww_dw5"),
    ("vww_pw12",  256, 256,  1, 1,  3,  3, 1,  1,  "vww_pw5"),
    ("vww_fc",    256,   2,  1, 1,  1,  1, 1,  1,  "vww_pw5"),
]

VWW_STEMS = {
    "vww_stem": "vww_conv1",
    "vww_dw3":  "vww_dw5",
    "vww_pw3":  "vww_pw5",
}

# ---------------------------------------------------------------------------
AD_LAYERS = [
    # name           C    K   R  S  P  Q  s  cnt  ref_stem
    ("ad_enc_in",  640, 128,  1, 1, 1, 1, 1,  1,  None),        # measured
    ("ad_hidden",  128, 128,  1, 1, 1, 1, 1,  6,  "ad_dense1"),  # same type
    ("ad_bn_enc",  128,   8,  1, 1, 1, 1, 1,  1,  "ad_dense1"),
    ("ad_bn_dec",    8, 128,  1, 1, 1, 1, 1,  1,  "ad_dense1"),
    ("ad_out",     128, 640,  1, 1, 1, 1, 1,  1,  "ad_dense1"),
]

AD_STEMS = {
    "ad_enc_in": "ad_dense1",
}

# ---------------------------------------------------------------------------
ALL_NETWORKS = {
    "ic":  (IC_LAYERS,  IC_STEMS,  KWS_FALLBACK_BASE & set()),
    "kws": (KWS_LAYERS, KWS_STEMS, KWS_FALLBACK_BASE),
    "vww": (VWW_LAYERS, VWW_STEMS, set()),
    "ad":  (AD_LAYERS,  AD_STEMS,  set()),
}

NET_LABELS = {
    "ic":  "ResNet-8 (IC)",
    "kws": "DS-CNN (KWS)",
    "vww": "MobileNetV1 (VWW)",
    "ad":  "AutoEncoder (AD)",
}


# ---------------------------------------------------------------------------
# Timeloop output reading
# ---------------------------------------------------------------------------

def macs(entry):
    _, C, K, R, S, P, Q, stride, count, _ = entry
    return C * K * R * S * P * Q


def extract_cycles(path: str):
    """Extract cycle count from a Timeloop stats or mapper output file."""
    if not os.path.isfile(path):
        return None
    with open(path, errors="replace") as f:
        text = f.read()
    if text.strip() == "FAILED":
        return None
    m = re.search(r"Cycles\s*:\s*(\d+)", text)
    return int(m.group(1)) if m else None


def load_existing_results(tl_dir: str):
    """
    Scan tl_dir for all stats/mapper files and build KNOWN_CYCLES.
    Key: (stem, tag)  e.g. ("s1_conv1", "opt_a")
    """
    if not os.path.isdir(tl_dir):
        return
    for fname in os.listdir(tl_dir):
        if not (fname.endswith(".stats.txt") or fname.endswith("_mapper.txt")):
            continue
        fpath = os.path.join(tl_dir, fname)
        c = extract_cycles(fpath)
        if c is None:
            continue
        # Parse stem and tag from filename
        # Patterns: <stem>.stats.txt  |  <stem>_<tag>.stats.txt  |  <stem>_mapper.txt
        base = fname.replace("_mapper.txt", "").replace(".stats.txt", "")
        # Check if it ends with a known tag
        tag = "base"
        for t in ("opt_a", "opt_b"):
            if base.endswith("_" + t):
                stem = base[: -(len(t) + 1)]
                tag = t
                break
        else:
            stem = base
        key = (stem, tag)
        # Keep the highest cycle count per key (avoid partial reads)
        if key not in KNOWN_CYCLES or c > KNOWN_CYCLES[key]:
            KNOWN_CYCLES[key] = c


def get_measured_cycles(stem: str, tag: str):
    """Return measured cycles for (stem, tag), or None."""
    return KNOWN_CYCLES.get((stem, tag))


def get_estimated_cycles(ref_stem: str, ref_tag: str, new_macs: int, ref_macs: int):
    """
    Estimate cycles for a new layer from a reference measurement.
    est = ref_cycles * new_macs / ref_macs
    """
    ref_c = get_measured_cycles(ref_stem, ref_tag)
    if ref_c is None or ref_macs == 0:
        return None
    return max(1, round(ref_c * new_macs / ref_macs))


# ---------------------------------------------------------------------------
# Per-layer cycle resolution
# ---------------------------------------------------------------------------

def resolve_cycles(entry, tag: str, stems: dict, fallback_base: set):
    """
    Return (cycles, source_label) for a layer × arch combination.
    source_label: 'measured', 'estimated', or 'fallback-base'
    """
    name, C, K, R, S, P, Q, stride, count, ref_stem = entry
    layer_macs = macs(entry)

    # 1. Try direct measured result
    if name in stems:
        stem = stems[name]
        # For KWS conv1 with opt arrays: fall back to base
        if name in fallback_base and tag != "base":
            c = get_measured_cycles(stem, "base")
            if c is not None:
                return c, "fallback-base"
        c = get_measured_cycles(stem, tag)
        if c is not None:
            return c, "measured"

    # 2. Estimate from reference layer
    if ref_stem is not None:
        # Use the same arch tag; if that's also missing, fall back to base
        ref_c = None
        used_tag = tag
        for try_tag in [tag, "base"]:
            ref_c = get_measured_cycles(ref_stem, try_tag)
            if ref_c is not None:
                used_tag = try_tag
                break
        if ref_c is not None:
            # Compute reference MACs
            ref_macs_val = KNOWN_CYCLES.get(("__macs__", ref_stem), 0)
            # Find ref_macs from the layers list — we need to compute it
            # Store it in a side channel below
            ref_m = _REF_MACS.get(ref_stem, 0)
            if ref_m > 0:
                est = max(1, round(ref_c * layer_macs / ref_m))
                label = f"est from {ref_stem}/{used_tag}"
                return est, label

    return None, "MISSING"


# Reference MACs table — populated once from layer definitions
_REF_MACS: dict[str, int] = {}


def build_ref_macs():
    """Compute MACs for each reference stem from the layer definitions."""
    # Known reference stems and their (C, K, R, S, P, Q):
    refs = {
        "stem_conv":         (3,  26,  3, 3, 32, 32),
        "s1_conv1":          (26, 26,  3, 3, 32, 32),
        "classifier_dense":  (104, 10, 1, 1,  1,  1),
        "kws_conv1":         (1,  64, 10, 4, 25,  5),
        "kws_dw1":           (1,  64,  3, 3, 25,  5),
        "kws_pw1":           (64, 64,  1, 1, 25,  5),
        "vww_conv1":         (3,   8,  3, 3, 48, 48),
        "vww_dw5":           (1,  32,  3, 3, 12, 12),
        "vww_pw5":           (32, 64,  1, 1, 12, 12),
        "ad_dense1":         (640, 128, 1, 1, 1,  1),
    }
    for stem, (C, K, R, S, P, Q) in refs.items():
        _REF_MACS[stem] = C * K * R * S * P * Q


# ---------------------------------------------------------------------------
# Network latency computation
# ---------------------------------------------------------------------------

def compute_network(net: str, layers, stems: dict, fallback_base: set,
                    clock_hz: float):
    results = {}
    for tag in TAGS:
        total_tl = 0
        layer_rows = []
        all_ok = True

        for entry in layers:
            name = entry[0]
            count = entry[8]
            layer_macs = macs(entry)

            cyc_single, src = resolve_cycles(entry, tag, stems, fallback_base)

            if cyc_single is None:
                all_ok = False
                layer_rows.append({
                    "name": name, "count": count,
                    "macs": layer_macs,
                    "cyc_single": None, "cyc_total": None,
                    "src": "MISSING",
                })
            else:
                cyc_total = cyc_single * count
                total_tl += cyc_total
                layer_rows.append({
                    "name": name, "count": count,
                    "macs": layer_macs,
                    "cyc_single": cyc_single,
                    "cyc_total": cyc_total,
                    "src": src,
                })

        if all_ok and total_tl > 0:
            real_cyc = total_tl // UDP_MAC_FACTOR
            lat_us = real_cyc / clock_hz * 1e6
        else:
            real_cyc = None
            lat_us = None

        results[tag] = {
            "total_tl":  total_tl if all_ok else None,
            "real_cyc":  real_cyc,
            "lat_us":    lat_us,
            "rows":      layer_rows,
            "all_ok":    all_ok,
        }
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tl-outputs", default="", help="Path to timeloop/outputs dir")
    ap.add_argument("--lat-outputs", default="", help="Path to latency/outputs dir")
    ap.add_argument("--out", default="./latency_report.txt")
    ap.add_argument("--clock-mhz", type=float, default=DEFAULT_CLOCK_MHZ)
    ap.add_argument("--filter", default="all",
                    choices=["all", "ic", "kws", "vww", "ad"])
    return ap.parse_args()


def fmt(v, spec=".1f"):
    return format(v, spec) if v is not None else "N/A"


def main():
    args = parse_args()
    clock_hz = args.clock_mhz * 1e6

    # Locate tl-outputs automatically if not given
    script_dir = os.path.dirname(os.path.abspath(__file__))
    tl_dir = args.tl_outputs or os.path.join(script_dir, "..", "timeloop", "outputs")
    lat_dir = args.lat_outputs or os.path.join(script_dir, "outputs")

    tl_dir  = os.path.realpath(tl_dir)
    lat_dir = os.path.realpath(lat_dir)

    load_existing_results(tl_dir)
    load_existing_results(lat_dir)  # also check new latency outputs (if any ran OK)
    build_ref_macs()

    lines = []
    def out(*a):
        s = " ".join(str(x) for x in a)
        print(s)
        lines.append(s)

    out("=" * 72)
    out(" MLPerf Tiny - End-to-End Latency Estimation")
    out(f" Clock    : {args.clock_mhz:.0f} MHz")
    out(f" UDP      : div6  (P_w=3 P_a=2, 6 MACs/PE/cycle)")
    out(f" Arrays   : base 12x12 (144 PEs) | opt_a 12x8 (96 PEs) | opt_b 9x8 (72 PEs)")
    out(f" Method   : measured Timeloop cycles where available;")
    out(f"            '(est)' = MAC-proportional estimate from same-type reference layer")
    out("=" * 72)

    nets_to_run = list(ALL_NETWORKS.keys()) if args.filter == "all" else [args.filter]

    summary = []

    for net in nets_to_run:
        layers, stems, fallback = ALL_NETWORKS[net]
        net_results = compute_network(net, layers, stems, fallback, clock_hz)

        label = NET_LABELS[net]
        out("")
        out("-" * 72)
        out(f"  {label}")
        out("-" * 72)

        # Per-layer detail (show base for brevity)
        out(f"  {'Layer':<18} {'cnt':>3}  {'MACs':>10}  {'TL-cycles':>10}  Source")
        out(f"  {'-'*18}  {'-'*3}  {'-'*10}  {'-'*10}  {'-'*20}")
        for row in net_results["base"]["rows"]:
            c = f"{row['cyc_single']:,}" if row['cyc_single'] is not None else "MISSING"
            out(f"  {row['name']:<18} {row['count']:>3}  {row['macs']:>10,}  {c:>10}  {row['src']}")

        out("")
        out(f"  {'Metric':<36}  {'base':>12}  {'opt_a':>12}  {'opt_b':>12}")
        out(f"  {'-'*36}  {'-'*12}  {'-'*12}  {'-'*12}")

        row_tl  = f"  {'TL cycles total (x1/PE)':<36}"
        row_rc  = f"  {'UDP real cycles (div6)':<36}"
        row_lat = f"  {'Latency (us)':<36}"

        for tag in TAGS:
            r = net_results[tag]
            row_tl  += f"  {fmt(r['total_tl'], ',d') if r['total_tl'] else 'N/A':>12}"
            row_rc  += f"  {fmt(r['real_cyc'],  ',d') if r['real_cyc']  else 'N/A':>12}"
            row_lat += f"  {fmt(r['lat_us'], '.1f') if r['lat_us'] else 'N/A':>12}"

        out(row_tl)
        out(row_rc)
        out(row_lat)

        summary.append((
            label,
            net_results["base"]["lat_us"],
            net_results["opt_a"]["lat_us"],
            net_results["opt_b"]["lat_us"],
        ))

    # Summary
    out("")
    out("=" * 72)
    out("  SUMMARY - End-to-End Inference Latency (us) at 200 MHz")
    out("=" * 72)
    out(f"  {'Network':<26}  {'base 12x12':>12}  {'opt_a 12x8':>12}  {'opt_b 9x8':>12}")
    out(f"  {'-'*26}  {'-'*12}  {'-'*12}  {'-'*12}")
    for label, b, oa, ob in summary:
        out(f"  {label:<26}  {fmt(b):>12}  {fmt(oa):>12}  {fmt(ob):>12}")

    out("")
    out("  Methodology:")
    out("  - 'measured': actual Timeloop mapper result")
    out("  - '(est)': cycles = ref_cycles x new_MACs/ref_MACs (same layer type)")
    out("  - real_cycles = TL_cycles / 6  (UDP: P_w=3, P_a=2, 6 MACs/PE/cycle)")
    out("  - latency = real_cycles / 200e6")
    out("  - KWS conv1 uses base cycles for opt_a/opt_b (10x4 kernel, no valid map)")
    out("=" * 72)

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nReport -> {args.out}")


if __name__ == "__main__":
    main()
