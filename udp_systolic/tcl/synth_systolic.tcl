# =============================================================================
# synth_systolic.tcl
# Out-of-context synthesis for the 12×12 int4 UDP systolic array.
#
# Targets: Zynq UltraScale+ ZCU106 (xczu7ev-ffvc1156-2-e)
#
# USAGE (on the server):
#   vivado -mode batch -source tcl/synth_systolic.tcl
#
# WHAT THIS SCRIPT DOES
# =====================
#  1. Creates a Vivado in-memory project (no .xpr file written)
#  2. Adds all RTL source files
#  3. Runs out-of-context (OOC) synthesis
#     • OOC means the top-level ports are treated as black-box I/Os — the
#       synthesiser cannot assume constants on any input and cannot prune
#       "unused" outputs.  This gives a fair resource count.
#  4. Runs opt_design to apply physical-level optimisations
#     (important: LUT count DROPS after opt_design as Vivado merges
#      redundant logic across the 144 identical PE instances)
#  5. Reports:
#       • DSP utilisation  → should be 144
#       • LUT utilisation  → target ≈ 1466 (UDP logic) + accumulator overhead
#       • FF utilisation   → activation pipeline + accumulator registers
# =============================================================================

# ── Paths ─────────────────────────────────────────────────────────────────────
set SCRIPT_DIR [file dirname [file normalize [info script]]]
set PROJ_DIR   [file dirname $SCRIPT_DIR]

set PE_RTL_DIR  "$PROJ_DIR/../udp_rtl/rtl"
set SYS_RTL_DIR "$PROJ_DIR/rtl"
set RESULTS_DIR "$PROJ_DIR/synth_results"

file mkdir $RESULTS_DIR

# ── Target device ─────────────────────────────────────────────────────────────
set PART "xczu7ev-ffvc1156-2-e"   ;# Zynq UltraScale+ ZCU106

# ── All RTL source files ───────────────────────────────────────────────────────
set rtl_files [list \
    "$PE_RTL_DIR/dsp48e2_model.v"   \
    "$PE_RTL_DIR/udp_pack.v"        \
    "$PE_RTL_DIR/udp_preprocess.v"  \
    "$PE_RTL_DIR/udp_postprocess.v" \
    "$PE_RTL_DIR/udp_mac_top.v"     \
    "$SYS_RTL_DIR/udp_pe.v"         \
    "$SYS_RTL_DIR/systolic_array.v" \
]

# ── Create in-memory project ───────────────────────────────────────────────────
puts "=== Creating project for $PART ==="
create_project -in_memory -part $PART

# ── Add source files ───────────────────────────────────────────────────────────
puts "=== Adding RTL files ==="
foreach f $rtl_files {
    puts "  Adding: $f"
    add_files -norecurse $f
}

# ── Set top module ─────────────────────────────────────────────────────────────
set_property top systolic_array [current_fileset]

# ── Synthesis pass ─────────────────────────────────────────────────────────────
puts "\n=== Running synthesis (out-of-context) ==="
# -mode out_of_context: treats top-level ports as unknown inputs/outputs.
# This gives a worst-case (no constant propagation across top-level ports)
# and matches how the paper measures a standalone accelerator block.
synth_design \
    -top      systolic_array \
    -part     $PART          \
    -mode     out_of_context \
    -flatten_hierarchy rebuilt

# ── Immediate post-synthesis report ───────────────────────────────────────────
puts "\n=== POST-SYNTHESIS UTILISATION ==="
report_utilization -file "$RESULTS_DIR/util_post_synth.rpt"

# Extract key numbers and print to console
set post_synth_util [report_utilization -return_string]
puts $post_synth_util

# ── opt_design — critical for accurate LUT count ──────────────────────────────
# Vivado merges identical sub-circuits across the 144 PE instances here.
# The LUT count typically drops significantly compared to raw synthesis.
puts "\n=== Running opt_design (this reduces LUT count by sharing logic) ==="
opt_design

# ── Post-opt report ────────────────────────────────────────────────────────────
puts "\n=== POST-OPT UTILISATION (this is the comparable number to the paper) ==="
report_utilization -file "$RESULTS_DIR/util_post_opt.rpt"
set post_opt_util [report_utilization -return_string]
puts $post_opt_util

# ── Timing report ──────────────────────────────────────────────────────────────
puts "\n=== TIMING REPORT ==="
report_timing_summary -file "$RESULTS_DIR/timing_post_opt.rpt" -max_paths 5
report_timing_summary -max_paths 5

# ── Parse and pretty-print key utilisation numbers ────────────────────────────
puts "\n=========================================================="
puts " SUMMARY — 12x12 UDP int4 Systolic Array"
puts "=========================================================="

set rpt_text [read [open "$RESULTS_DIR/util_post_opt.rpt" r]]

proc extract_util { rpt_text keyword } {
    # Search for a line like "| LUT as Logic       |  1466 |" and return the number
    foreach line [split $rpt_text "\n"] {
        if {[string match "*${keyword}*" $line]} {
            set fields [split $line "|"]
            if {[llength $fields] >= 3} {
                set val [string trim [lindex $fields 2]]
                if {[string is integer $val] && $val > 0} {
                    return $val
                }
            }
        }
    }
    return "N/A"
}

set lut_logic [extract_util $rpt_text "LUT as Logic"]
set lut_total [extract_util $rpt_text "CLB LUTs"]
set dsp_count [extract_util $rpt_text "DSPs"]
set ff_count  [extract_util $rpt_text "CLB Registers"]

puts " DSPs            : $dsp_count  (target: 144)"
puts " LUT as Logic    : $lut_logic"
puts " CLB LUTs total  : $lut_total  (paper target: ~1466 for mult. logic only)"
puts " CLB FFs         : $ff_count   (activation pipeline + accumulators)"
puts ""
puts " NOTE: ACC_WIDTH=12 supports K_MAX=31 inner-dimension accumulations."
puts "       Increasing ACC_WIDTH raises LUTs by ~864 per extra bit (6 acc × 144 PEs)."
puts "       ACC_WIDTH=16 gave 17,864 LUTs; ACC_WIDTH=12 targets ~14,408 LUTs."
puts "=========================================================="

puts "\n=== Synthesis complete. Reports saved to $RESULTS_DIR ==="
