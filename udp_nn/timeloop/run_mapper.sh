#!/usr/bin/env bash
# Run Timeloop mapper for UDP systolic array problem configs.
#
# Usage:
#   ./run_mapper.sh <target>
#
# Targets (original 1-MAC-per-PE model, requires /6 post-correction):
#   stem_conv            s1_conv1            classifier_dense
#   all                  (all three above)
#
# Targets (6-MAC-per-PE UDP model, cycles reported directly):
#   stem_conv_6mac       s1_conv1_6mac       classifier_dense_6mac
#   all_6mac             (all three above)
#
# Requires timeloop-mapper at ~/timeloop/bin.
# PATH is set automatically if ~/timeloop/bin exists.

set -eu

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

TIMELOOP_HOME="${TIMELOOP_HOME:-$HOME/timeloop}"
export PATH="$TIMELOOP_HOME/bin:/usr/bin:/bin:$PATH"
export LD_LIBRARY_PATH="$TIMELOOP_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

if ! command -v timeloop-mapper >/dev/null 2>&1; then
    echo "ERROR: timeloop-mapper not found."
    echo "  Expected: $TIMELOOP_HOME/bin/timeloop-mapper"
    echo "  Run from PowerShell: wsl -e bash /home/hazem/run_timeloop_fixed.sh"
    exit 1
fi

ARCH_1MAC="$SCRIPT_DIR/udp_arch.yaml"
ARCH_6MAC="$SCRIPT_DIR/udp_arch_6mac.yaml"
ARCH_6X="$SCRIPT_DIR/udp_arch_6x.yaml"
ARCH_OPT_A="$SCRIPT_DIR/udp_arch_opt_a.yaml"       # 8×12=96 PEs, meshX=8
ARCH_OPT_B="$SCRIPT_DIR/udp_arch_opt_b.yaml"       # 8×9=72 PEs, meshX=8
ARCH_OPT_A_KWS="$SCRIPT_DIR/udp_arch_opt_a_kws.yaml"  # opt_a relaxed for KWS
ARCH_OPT_B_KWS="$SCRIPT_DIR/udp_arch_opt_b_kws.yaml"  # opt_b relaxed for KWS
OUT_DIR="$SCRIPT_DIR/outputs"
mkdir -p "$OUT_DIR"

run_one() {
    local name="$1"
    local arch="$2"
    local problem="$3"
    local extra="${4:-}"          # optional extra constraints/override YAML
    local out="$OUT_DIR/${name}_mapper.txt"
    local stats="$OUT_DIR/${name}.stats.txt"

    echo "=== Timeloop mapper: $name ==="
    echo "  arch   : $arch"
    echo "  problem: $problem"
    echo "  log    : $out"

    rm -f "$SCRIPT_DIR/timeloop-mapper.stats.txt"
    if ! timeloop-mapper "$arch" "$problem" ${extra:+"$extra"} 2>&1 | tee "$out"; then
        echo "ERROR: timeloop-mapper failed for $name" >&2
        exit 1
    fi
    if [ ! -f "$SCRIPT_DIR/timeloop-mapper.stats.txt" ]; then
        echo "ERROR: timeloop-mapper did not write stats for $name" >&2
        exit 1
    fi
    cp "$SCRIPT_DIR/timeloop-mapper.stats.txt" "$stats"
    echo ""
    grep -E "Summary stats|Utilization" "$out" || true
    grep -E "^(Utilization:|Cycles:|Computes =)" "$stats" | tail -3 || true
    echo ""
}

case "${1:-all}" in

    # ── Original 1-MAC-per-PE model (divide cycles by 6 to get UDP cycles) ──
    stem_conv)
        run_one stem_conv "$ARCH_1MAC" "$SCRIPT_DIR/problem_stem_conv.yaml"
        ;;
    s1_conv1)
        run_one s1_conv1 "$ARCH_1MAC" "$SCRIPT_DIR/problem_s1_conv1.yaml"
        ;;
    classifier_dense)
        run_one classifier_dense "$ARCH_1MAC" "$SCRIPT_DIR/problem_classifier_dense.yaml"
        ;;
    all)
        run_one stem_conv      "$ARCH_1MAC" "$SCRIPT_DIR/problem_stem_conv.yaml"
        run_one s1_conv1       "$ARCH_1MAC" "$SCRIPT_DIR/problem_s1_conv1.yaml"
        run_one classifier_dense "$ARCH_1MAC" "$SCRIPT_DIR/problem_classifier_dense.yaml"
        ;;

    # ── 6-MAC-per-PE UDP model (cycles reported directly, no /6 needed) ──────
    stem_conv_6mac)
        run_one stem_conv_6mac "$ARCH_6MAC" "$SCRIPT_DIR/problem_stem_conv.yaml"
        ;;
    s1_conv1_6mac)
        run_one s1_conv1_6mac "$ARCH_6MAC" "$SCRIPT_DIR/problem_s1_conv1.yaml"
        ;;
    classifier_dense_6mac)
        run_one classifier_dense_6mac "$ARCH_6MAC" "$SCRIPT_DIR/problem_classifier_dense.yaml"
        ;;
    all_6mac)
        run_one stem_conv_6mac       "$ARCH_6MAC" "$SCRIPT_DIR/problem_stem_conv.yaml"
        run_one s1_conv1_6mac        "$ARCH_6MAC" "$SCRIPT_DIR/problem_s1_conv1.yaml"
        run_one classifier_dense_6mac "$ARCH_6MAC" "$SCRIPT_DIR/problem_classifier_dense.yaml"
        ;;

    # ── 6× scaled 1-MAC array (same dataflow, 6× more PEs for comparison) ───────
    stem_conv_6x)
        run_one stem_conv_6x "$ARCH_6X" "$SCRIPT_DIR/problem_stem_conv.yaml"
        ;;
    s1_conv1_6x)
        run_one s1_conv1_6x "$ARCH_6X" "$SCRIPT_DIR/problem_s1_conv1.yaml"
        ;;
    classifier_dense_6x)
        run_one classifier_dense_6x "$ARCH_6X" "$SCRIPT_DIR/problem_classifier_dense.yaml"
        ;;
    all_6x)
        run_one stem_conv_6x         "$ARCH_6X" "$SCRIPT_DIR/problem_stem_conv.yaml"
        run_one s1_conv1_6x          "$ARCH_6X" "$SCRIPT_DIR/problem_s1_conv1.yaml"
        run_one classifier_dense_6x  "$ARCH_6X" "$SCRIPT_DIR/problem_classifier_dense.yaml"
        ;;

    # ── Opt-A: 8×12=96 PEs — fixes column waste (meshX=8) ───────────────────
    stem_conv_opt_a)
        run_one stem_conv_opt_a "$ARCH_OPT_A" "$SCRIPT_DIR/problem_stem_conv.yaml"
        ;;
    s1_conv1_opt_a)
        run_one s1_conv1_opt_a "$ARCH_OPT_A" "$SCRIPT_DIR/problem_s1_conv1.yaml"
        ;;
    classifier_dense_opt_a)
        run_one classifier_dense_opt_a "$ARCH_OPT_A" "$SCRIPT_DIR/problem_classifier_dense.yaml"
        ;;
    all_opt_a)
        run_one stem_conv_opt_a         "$ARCH_OPT_A" "$SCRIPT_DIR/problem_stem_conv.yaml"
        run_one s1_conv1_opt_a          "$ARCH_OPT_A" "$SCRIPT_DIR/problem_s1_conv1.yaml"
        run_one classifier_dense_opt_a  "$ARCH_OPT_A" "$SCRIPT_DIR/problem_classifier_dense.yaml"
        ;;

    # ── Opt-B: 8×9=72 PEs — fixes both column and row waste ─────────────────
    stem_conv_opt_b)
        run_one stem_conv_opt_b "$ARCH_OPT_B" "$SCRIPT_DIR/problem_stem_conv.yaml"
        ;;
    s1_conv1_opt_b)
        run_one s1_conv1_opt_b "$ARCH_OPT_B" "$SCRIPT_DIR/problem_s1_conv1.yaml"
        ;;
    classifier_dense_opt_b)
        run_one classifier_dense_opt_b "$ARCH_OPT_B" "$SCRIPT_DIR/problem_classifier_dense.yaml"
        ;;
    all_opt_b)
        run_one stem_conv_opt_b         "$ARCH_OPT_B" "$SCRIPT_DIR/problem_stem_conv.yaml"
        run_one s1_conv1_opt_b          "$ARCH_OPT_B" "$SCRIPT_DIR/problem_s1_conv1.yaml"
        run_one classifier_dense_opt_b  "$ARCH_OPT_B" "$SCRIPT_DIR/problem_classifier_dense.yaml"
        ;;

    # ── KWS (DS-CNN) layers ───────────────────────────────────────────────────
    kws_conv1)
        run_one kws_conv1 "$ARCH_1MAC" "$SCRIPT_DIR/problem_kws_conv1.yaml"
        ;;
    kws_dw1)
        run_one kws_dw1   "$ARCH_1MAC" "$SCRIPT_DIR/problem_kws_dw1.yaml"
        ;;
    kws_pw1)
        run_one kws_pw1   "$ARCH_1MAC" "$SCRIPT_DIR/problem_kws_pw1.yaml"
        ;;
    all_kws)
        run_one kws_conv1 "$ARCH_1MAC" "$SCRIPT_DIR/problem_kws_conv1.yaml"
        run_one kws_dw1   "$ARCH_1MAC" "$SCRIPT_DIR/problem_kws_dw1.yaml"
        run_one kws_pw1   "$ARCH_1MAC" "$SCRIPT_DIR/problem_kws_pw1.yaml"
        ;;
    # KWS individual layers with opt arrays (for debugging)
    kws_conv1_opt_a)
        run_one kws_conv1_opt_a "$ARCH_OPT_A_KWS" "$SCRIPT_DIR/problem_kws_conv1.yaml"
        ;;
    kws_dw1_opt_a)
        run_one kws_dw1_opt_a "$ARCH_OPT_A" "$SCRIPT_DIR/problem_kws_dw1.yaml"
        ;;
    kws_pw1_opt_a)
        run_one kws_pw1_opt_a "$ARCH_OPT_A" "$SCRIPT_DIR/problem_kws_pw1.yaml"
        ;;
    kws_conv1_opt_b)
        run_one kws_conv1_opt_b "$ARCH_OPT_B_KWS" "$SCRIPT_DIR/problem_kws_conv1.yaml"
        ;;
    kws_dw1_opt_b)
        run_one kws_dw1_opt_b "$ARCH_OPT_B" "$SCRIPT_DIR/problem_kws_dw1.yaml"
        ;;
    kws_pw1_opt_b)
        run_one kws_pw1_opt_b "$ARCH_OPT_B" "$SCRIPT_DIR/problem_kws_pw1.yaml"
        ;;

    # KWS with optimised arrays (KWS-specific arch with relaxed spatial for K-dim)
    all_kws_opt_a)
        run_one kws_conv1_opt_a "$ARCH_OPT_A_KWS" "$SCRIPT_DIR/problem_kws_conv1.yaml"
        run_one kws_dw1_opt_a   "$ARCH_OPT_A_KWS" "$SCRIPT_DIR/problem_kws_dw1.yaml"
        run_one kws_pw1_opt_a   "$ARCH_OPT_A_KWS" "$SCRIPT_DIR/problem_kws_pw1.yaml"
        ;;
    all_kws_opt_b)
        run_one kws_conv1_opt_b "$ARCH_OPT_B_KWS" "$SCRIPT_DIR/problem_kws_conv1.yaml"
        run_one kws_dw1_opt_b   "$ARCH_OPT_B_KWS" "$SCRIPT_DIR/problem_kws_dw1.yaml"
        run_one kws_pw1_opt_b   "$ARCH_OPT_B_KWS" "$SCRIPT_DIR/problem_kws_pw1.yaml"
        ;;

    # ── VWW (MobileNetV1) layers ──────────────────────────────────────────────
    vww_conv1)
        run_one vww_conv1 "$ARCH_1MAC" "$SCRIPT_DIR/problem_vww_conv1.yaml"
        ;;
    vww_dw5)
        run_one vww_dw5   "$ARCH_1MAC" "$SCRIPT_DIR/problem_vww_dw5.yaml"
        ;;
    vww_pw5)
        run_one vww_pw5   "$ARCH_1MAC" "$SCRIPT_DIR/problem_vww_pw5.yaml"
        ;;
    all_vww)
        run_one vww_conv1 "$ARCH_1MAC" "$SCRIPT_DIR/problem_vww_conv1.yaml"
        run_one vww_dw5   "$ARCH_1MAC" "$SCRIPT_DIR/problem_vww_dw5.yaml"
        run_one vww_pw5   "$ARCH_1MAC" "$SCRIPT_DIR/problem_vww_pw5.yaml"
        ;;
    # VWW with optimised arrays
    all_vww_opt_a)
        run_one vww_conv1_opt_a "$ARCH_OPT_A" "$SCRIPT_DIR/problem_vww_conv1.yaml"
        run_one vww_dw5_opt_a   "$ARCH_OPT_A" "$SCRIPT_DIR/problem_vww_dw5.yaml"
        run_one vww_pw5_opt_a   "$ARCH_OPT_A" "$SCRIPT_DIR/problem_vww_pw5.yaml"
        ;;
    all_vww_opt_b)
        run_one vww_conv1_opt_b "$ARCH_OPT_B" "$SCRIPT_DIR/problem_vww_conv1.yaml"
        run_one vww_dw5_opt_b   "$ARCH_OPT_B" "$SCRIPT_DIR/problem_vww_dw5.yaml"
        run_one vww_pw5_opt_b   "$ARCH_OPT_B" "$SCRIPT_DIR/problem_vww_pw5.yaml"
        ;;

    # ── AD (AutoEncoder dense layer) ──────────────────────────────────────────
    ad_dense1)
        run_one ad_dense1 "$ARCH_1MAC" "$SCRIPT_DIR/problem_ad_dense1.yaml"
        ;;
    ad_dense1_opt_a)
        run_one ad_dense1_opt_a "$ARCH_OPT_A" "$SCRIPT_DIR/problem_ad_dense1.yaml"
        ;;
    ad_dense1_opt_b)
        run_one ad_dense1_opt_b "$ARCH_OPT_B" "$SCRIPT_DIR/problem_ad_dense1.yaml"
        ;;
    all_ad)
        run_one ad_dense1       "$ARCH_1MAC"  "$SCRIPT_DIR/problem_ad_dense1.yaml"
        run_one ad_dense1_opt_a "$ARCH_OPT_A" "$SCRIPT_DIR/problem_ad_dense1.yaml"
        run_one ad_dense1_opt_b "$ARCH_OPT_B" "$SCRIPT_DIR/problem_ad_dense1.yaml"
        ;;

    # ── Run EVERYTHING ────────────────────────────────────────────────────────
    all_benchmarks)
        # IC (ResNet-8)
        run_one stem_conv         "$ARCH_1MAC" "$SCRIPT_DIR/problem_stem_conv.yaml"
        run_one s1_conv1          "$ARCH_1MAC" "$SCRIPT_DIR/problem_s1_conv1.yaml"
        run_one classifier_dense  "$ARCH_1MAC" "$SCRIPT_DIR/problem_classifier_dense.yaml"
        # KWS (DS-CNN)
        run_one kws_conv1 "$ARCH_1MAC" "$SCRIPT_DIR/problem_kws_conv1.yaml"
        run_one kws_dw1   "$ARCH_1MAC" "$SCRIPT_DIR/problem_kws_dw1.yaml"
        run_one kws_pw1   "$ARCH_1MAC" "$SCRIPT_DIR/problem_kws_pw1.yaml"
        # VWW (MobileNetV1)
        run_one vww_conv1 "$ARCH_1MAC" "$SCRIPT_DIR/problem_vww_conv1.yaml"
        run_one vww_dw5   "$ARCH_1MAC" "$SCRIPT_DIR/problem_vww_dw5.yaml"
        run_one vww_pw5   "$ARCH_1MAC" "$SCRIPT_DIR/problem_vww_pw5.yaml"
        # AD (dense, for completeness)
        run_one ad_dense1 "$ARCH_1MAC" "$SCRIPT_DIR/problem_ad_dense1.yaml"
        ;;

    *)
        echo "Unknown target: $1"
        echo ""
        echo "IC (ResNet-8):"
        echo "  stem_conv | s1_conv1 | classifier_dense | all"
        echo ""
        echo "KWS (DS-CNN):"
        echo "  kws_conv1 | kws_dw1 | kws_pw1 | all_kws"
        echo "  all_kws_opt_a | all_kws_opt_b"
        echo ""
        echo "VWW (MobileNetV1):"
        echo "  vww_conv1 | vww_dw5 | vww_pw5 | all_vww"
        echo "  all_vww_opt_a | all_vww_opt_b"
        echo ""
        echo "AD (AutoEncoder):"
        echo "  ad_dense1"
        echo ""
        echo "All benchmarks: all_benchmarks"
        exit 1
        ;;
esac
