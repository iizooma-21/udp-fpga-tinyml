# =============================================================================
# run_all.tcl
# Master TCL script: runs both simulation testbenches and summarises results.
#
# Usage (from the udp_rtl/ project root):
#   vivado -mode batch -source tcl/run_all.tcl  |& tee run_all.log
#
# What it does:
#   1. Sources tcl/sim_dsp.tcl    → DSP48E2 model tests
#   2. Sources tcl/sim_udp_mac.tcl → Full UDP MAC tests (6x reduction demo)
#   3. Prints a summary of both runs
# =============================================================================

set script_dir [file dirname [info script]]

puts ""
puts "############################################################"
puts "  UDP RTL Project — Run All Simulations"
puts "  Vivado xsim batch mode"
puts "############################################################"
puts ""

# Save original directory
set orig_dir [pwd]

# =============================================================================
# RUN 1: DSP48E2 model testbench
# =============================================================================
puts "############################################################"
puts "  RUN 1: DSP48E2 Behavioral Model"
puts "############################################################"

cd $orig_dir
if {[catch {source "$script_dir/sim_dsp.tcl"} err]} {
    puts "ERROR in sim_dsp.tcl: $err"
}

# =============================================================================
# RUN 2: Full UDP MAC testbench
# =============================================================================
cd $orig_dir
puts ""
puts "############################################################"
puts "  RUN 2: Full UDP MAC (int4 x int4, P_w=3, P_a=2)"
puts "############################################################"

if {[catch {source "$script_dir/sim_udp_mac.tcl"} err]} {
    puts "ERROR in sim_udp_mac.tcl: $err"
}

# =============================================================================
# Final summary
# =============================================================================
cd $orig_dir
puts ""
puts "############################################################"
puts "  ALL SIMULATIONS COMPLETE"
puts ""
puts "  Key result to look for:"
puts "    'Test A (scalar MACs): 84 operations'"
puts "    'Test B (UDP DSP calls): 14 operations'"
puts "    'Reduction factor: 6x'"
puts ""
puts "  Simulation artifacts:"
puts "    ./sim_dsp_work/   — DSP testbench files"
puts "    ./sim_udp_work/   — UDP MAC testbench files"
puts "############################################################"
