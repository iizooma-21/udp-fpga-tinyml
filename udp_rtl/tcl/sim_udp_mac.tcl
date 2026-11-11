# =============================================================================
# sim_udp_mac.tcl
# Compiles and simulates the full UDP MAC testbench using Vivado xsim.
#
# Usage (from the udp_rtl/ project root):
#   vivado -mode batch -source tcl/sim_udp_mac.tcl
#
# Or run interactively in Vivado TCL console:
#   source tcl/sim_udp_mac.tcl
# =============================================================================

# --- Configuration ---
set rtl_dir  "../rtl"
set tb_dir   "../tb"
set sim_dir  "./sim_udp_work"
set snapshot "tb_udp_mac_snap"

puts "============================================================"
puts "  UDP Project — Full UDP MAC Simulation"
puts "  Config: int4 x int4, P_w=3, P_a=2 (6 MACs per DSP call)"
puts "============================================================"

# Create and enter simulation working directory
file mkdir $sim_dir
cd $sim_dir
puts "\nWorking directory: [pwd]"

# -----------------------------------------------------------------------------
# Step 1: Compile all RTL sources and testbench
# Note: order matters — submodules before top-level, top before testbench
# -----------------------------------------------------------------------------
puts "\n--- Step 1: Compiling sources ---"

set all_files [list \
    "$rtl_dir/dsp48e2_model.v"    \
    "$rtl_dir/udp_pack.v"         \
    "$rtl_dir/udp_preprocess.v"   \
    "$rtl_dir/udp_postprocess.v"  \
    "$rtl_dir/udp_mac_top.v"      \
    "$tb_dir/tb_udp_mac.v"        \
]

foreach f $all_files {
    set basename [file tail $f]
    puts "  Compiling: $basename"
    if {[catch {exec xvlog $f} result]} {
        puts "ERROR compiling $basename:\n$result"
        exit 1
    }
}
puts "  All sources compiled OK"

# -----------------------------------------------------------------------------
# Step 2: Elaborate
# -----------------------------------------------------------------------------
puts "\n--- Step 2: Elaborating (snapshot: $snapshot) ---"
if {[catch {exec xelab -debug typical tb_udp_mac -s $snapshot} result]} {
    puts "ERROR during elaboration:\n$result"
    exit 1
}
puts "  Elaboration OK"

# -----------------------------------------------------------------------------
# Step 3: Simulate and capture output
# -----------------------------------------------------------------------------
puts "\n--- Step 3: Running simulation ---"
set sim_output [exec xsim $snapshot -runall]
puts $sim_output

# Parse pass/fail counts from simulation output
set pass_line [regexp -line -inline {PASSED: (\d+)} $sim_output]
set fail_line [regexp -line -inline {FAILED: (\d+)} $sim_output]

puts "\n============================================================"
puts "  sim_udp_mac.tcl complete."
puts "  Check 'LOOP COUNT COMPARISON' in output for 6x reduction."
puts "============================================================"
