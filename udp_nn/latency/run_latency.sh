#!/usr/bin/env bash
# =============================================================================
# run_latency.sh  —  End-to-end latency estimation for MLPerf Tiny models
# =============================================================================
# Usage:  ./run_latency.sh [network]
#   network : ic | kws | vww | ad | all   (default: all)
#
# This script:
#   1. Generates problem YAMLs for all layers not yet computed
#   2. Runs timeloop-mapper for each (layer × arch) pair
#   3. Calls compute_latency.py to build the final latency table
#
# Prereqs: timeloop-mapper must be on PATH (source the Timeloop env first)
# =============================================================================

set -euo pipefail

# ── Timeloop PATH setup (mirrors run_mapper.sh) ───────────────────────────────
TIMELOOP_HOME="${TIMELOOP_HOME:-$HOME/timeloop}"
export PATH="$TIMELOOP_HOME/bin:/usr/bin:/bin:$PATH"
export LD_LIBRARY_PATH="$TIMELOOP_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

if ! command -v timeloop-mapper >/dev/null 2>&1; then
    echo "ERROR: timeloop-mapper not found at $TIMELOOP_HOME/bin"
    echo "  Set TIMELOOP_HOME env var if your installation is elsewhere."
    exit 1
fi
echo "Using timeloop-mapper: $(which timeloop-mapper)"

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIMELOOP_DIR="$(realpath "$SCRIPT_DIR/../timeloop")"
PROBLEMS_DIR="$SCRIPT_DIR/problems"
OUTPUTS_DIR="$SCRIPT_DIR/outputs"
TL_OUTPUTS="$TIMELOOP_DIR/outputs"

mkdir -p "$PROBLEMS_DIR" "$OUTPUTS_DIR"

# ── Architecture YAML files ───────────────────────────────────────────────────
ARCH_BASE="$TIMELOOP_DIR/udp_arch.yaml"
ARCH_OPT_A="$TIMELOOP_DIR/udp_arch_opt_a.yaml"
ARCH_OPT_B="$TIMELOOP_DIR/udp_arch_opt_b.yaml"
ARCH_OPT_A_KWS="$TIMELOOP_DIR/udp_arch_opt_a_kws.yaml"
ARCH_OPT_B_KWS="$TIMELOOP_DIR/udp_arch_opt_b_kws.yaml"

# ── Filter ─────────────────────────────────────────────────────────────────────
FILTER="${1:-all}"

# ── Helper: run timeloop-mapper and save stats ─────────────────────────────────
run_one() {
    local label="$1"     # e.g. ic_s2c1_base
    local arch="$2"
    local problem="$3"
    local out_dir="$OUTPUTS_DIR"

    # Use dot convention to match timeloop/outputs naming:  <label>.stats.txt
    local stats_file="$out_dir/${label}.stats.txt"
    local mapper_file="$out_dir/${label}_mapper.txt"

    # Skip only if stats file exists AND is NOT a FAILED sentinel
    if [[ -f "$stats_file" ]]; then
        if grep -q "^FAILED$" "$stats_file" 2>/dev/null; then
            echo "  [retry] $label  (previous run failed, retrying)"
            rm -f "$stats_file" "$mapper_file"
        else
            echo "  [skip] $label  (already done)"
            return 0
        fi
    fi

    echo "  [run ] $label"
    echo "    arch   : $arch"
    echo "    problem: $problem"

    # Run mapper; capture output; do NOT exit on failure (some layers may fail
    # for opt_a/opt_b — that is expected and handled in compute_latency.py)
    if timeloop-mapper "$arch" "$problem" > "$mapper_file" 2>&1; then
        # Copy mapper output as the stats file (both named for easy lookup)
        cp "$mapper_file" "$stats_file"
        # Also grab the key lines for quick inspection (|| true avoids set -e on grep miss)
        cycles=$(grep -oP 'Cycles:\s+\K[0-9]+' "$mapper_file" 2>/dev/null | tail -1 || true)
        util=$(grep -oP 'Utilization:\s+\K[0-9.]+' "$mapper_file" 2>/dev/null | tail -1 || true)
        echo "    → cycles    : ${cycles:-N/A}"
        echo "    → util      : ${util:-N/A}"
    else
        echo "    ✗ FAILED (no valid mapping — will fall back to base in latency table)"
        # Write a sentinel so we don't retry endlessly
        echo "FAILED" > "$stats_file"
    fi
}

# ── Step 1: Generate problem YAMLs ────────────────────────────────────────────
echo "================================================================="
echo " Step 1: Generating problem YAML files"
echo "================================================================="
python3 "$SCRIPT_DIR/generate_problems.py"

# ── Helper: decide which arch tags to use for a given layer/network ────────────
# Returns a list of "tag:arch_yaml" pairs
archs_for() {
    local net="$1"
    local layer="$2"
    # KWS conv1 skips opt_a / opt_b (incompatible kernel)
    if [[ "$net" == "kws" && "$layer" == "kws_conv1" ]]; then
        echo "base:$ARCH_BASE"
    elif [[ "$net" == "kws" ]]; then
        echo "base:$ARCH_BASE"
        echo "opt_a:$ARCH_OPT_A_KWS"
        echo "opt_b:$ARCH_OPT_B_KWS"
    else
        echo "base:$ARCH_BASE"
        echo "opt_a:$ARCH_OPT_A"
        echo "opt_b:$ARCH_OPT_B"
    fi
}

# ── Step 2: Run Timeloop for new layers ───────────────────────────────────────
echo ""
echo "================================================================="
echo " Step 2: Running Timeloop mapper for new layers"
echo "================================================================="

run_network() {
    local net="$1"
    shift

    echo ""
    echo "--- Network: $net ---"

    for name in "$@"; do
        local problem="$PROBLEMS_DIR/${name}.yaml"

        if [[ ! -f "$problem" ]]; then
            # This layer uses an existing Timeloop result — skip
            echo "  [skip] $name  (uses existing timeloop result)"
            continue
        fi

        # Determine arch configs for this layer
        if [[ "$net" == "kws" && "$name" == "kws_conv1" ]]; then
            run_one "${name}_base" "$ARCH_BASE" "$problem"
        elif [[ "$net" == "kws" ]]; then
            run_one "${name}_base"  "$ARCH_BASE"       "$problem"
            run_one "${name}_opt_a" "$ARCH_OPT_A_KWS"  "$problem"
            run_one "${name}_opt_b" "$ARCH_OPT_B_KWS"  "$problem"
        else
            run_one "${name}_base"  "$ARCH_BASE"   "$problem"
            run_one "${name}_opt_a" "$ARCH_OPT_A"  "$problem"
            run_one "${name}_opt_b" "$ARCH_OPT_B"  "$problem"
        fi
    done
}

# ── IC layers ─────────────────────────────────────────────────────────────────
if [[ "$FILTER" == "all" || "$FILTER" == "ic" ]]; then
    run_network ic \
        "ic_stem" "ic_s1c" "ic_s2c1" "ic_s2c2" "ic_s2sc" \
        "ic_s3c1" "ic_s3c2" "ic_s3sc" "ic_cls"
fi

# ── KWS layers ────────────────────────────────────────────────────────────────
if [[ "$FILTER" == "all" || "$FILTER" == "kws" ]]; then
    run_network kws "kws_conv1" "kws_dw" "kws_pw" "kws_fc"
fi

# ── VWW layers ────────────────────────────────────────────────────────────────
if [[ "$FILTER" == "all" || "$FILTER" == "vww" ]]; then
    run_network vww \
        "vww_stem" "vww_dw0" "vww_pw0" \
        "vww_dw1"  "vww_pw1" \
        "vww_dw2"  "vww_pw2" \
        "vww_dw3"  "vww_pw3" \
        "vww_dw4"  "vww_pw4" \
        "vww_dw5b" "vww_pw5b" \
        "vww_dw6"  "vww_pw6" \
        "vww_dw11" "vww_pw11" \
        "vww_dw12" "vww_pw12" \
        "vww_fc"
fi

# ── AD layers ─────────────────────────────────────────────────────────────────
if [[ "$FILTER" == "all" || "$FILTER" == "ad" ]]; then
    run_network ad "ad_enc_in" "ad_hidden" "ad_bn_enc" "ad_bn_dec" "ad_out"
fi

# ── Step 3: Compute latency table ─────────────────────────────────────────────
echo ""
echo "================================================================="
echo " Step 3: Computing latency table"
echo "================================================================="
python3 "$SCRIPT_DIR/compute_latency.py" \
    --filter "$FILTER" \
    --tl-outputs "$TL_OUTPUTS" \
    --lat-outputs "$OUTPUTS_DIR" \
    --out "$SCRIPT_DIR/latency_report.txt"

echo ""
echo "Done. Results saved to $SCRIPT_DIR/latency_report.txt"
