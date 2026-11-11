# =============================================================================
# synth_check.tcl
# Out-of-context synthesis script for udp_mac_top targeting ZCU106.
#
# Runs synthesis in batch mode and reports LUT and DSP utilization.
# Expected result after fix:
#   DSPs   : 1   (was 2 before the port-width fix in dsp48e2_model.v)
#   LUTs   : ~20 (reduced by postprocessor optimisation from ~50)
#
# Usage (from udp_rtl/ directory):
#   vivado -mode batch -source tcl/synth_check.tcl
# =============================================================================

set PART "xczu7ev-ffvc1156-2-e"   ;# Zynq UltraScale+ ZCU106
set TOP  "udp_mac_top"
set RTL_DIR [file join [pwd] rtl]
set OUT_DIR [file join [pwd] synth_out]

file mkdir $OUT_DIR

puts ""
puts "============================================================"
puts "  UDP Synthesis Check — ZCU106 (xczu7ev-ffvc1156-2-e)"
puts "  Target: 1 DSP, minimal LUTs for 6x MAC reduction"
puts "============================================================"

# -----------------------------------------------------------------------
# Create an in-memory project (no .xpr file needed for batch synthesis)
# -----------------------------------------------------------------------
create_project -in_memory -part $PART

# Add all RTL sources
read_verilog [glob ${RTL_DIR}/*.v]

set_property top $TOP [current_fileset]

# -----------------------------------------------------------------------
# Out-of-context synthesis
# Flags:
#   -mode out_of_context  : no I/O buffers inserted (pure logic)
#   -flatten_hierarchy none: keep hierarchy for readability in reports
# -----------------------------------------------------------------------
puts "\n--- Running synthesis ---"
synth_design \
    -top  $TOP \
    -part $PART \
    -mode out_of_context

# -----------------------------------------------------------------------
# Report utilization — look for DSP and LUT counts
# -----------------------------------------------------------------------
puts "\n--- Utilization Report ---"
report_utilization -file [file join $OUT_DIR utilization.rpt]
report_utilization

# -----------------------------------------------------------------------
# Parse key metrics for quick summary
# -----------------------------------------------------------------------
set dsp_count [llength [get_cells -hierarchical -filter {PRIMITIVE_TYPE =~ DSP*}]]
set lut_count [llength [get_cells -hierarchical -filter {PRIMITIVE_TYPE =~ LUT*}]]

puts ""
puts "============================================================"
puts "  QUICK SUMMARY"
puts "  DSP blocks  : $dsp_count  (target: 1)"
puts "  LUT cells   : $lut_count  (target: ~10-25)"
puts "============================================================"

if {$dsp_count == 1} {
    puts "  [OK]  1 DSP — synthesis fix confirmed."
} elseif {$dsp_count == 0} {
    puts "  [WARN] 0 DSPs — multiplier may have been mapped to LUTs."
    puts "         Try adding (* use_dsp = yes *) in the RTL."
} else {
    puts "  [FAIL] $dsp_count DSPs — expected 1. Check dsp48e2_model.v port widths."
}

# Save the elaborated netlist for inspection
write_verilog -force [file join $OUT_DIR netlist.v]

puts "\nReports written to: $OUT_DIR"
puts "  utilization.rpt — full resource breakdown"
puts "  netlist.v       — elaborated gate-level netlist"
