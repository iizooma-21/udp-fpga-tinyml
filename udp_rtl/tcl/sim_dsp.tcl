# =============================================================================
# sim_dsp.tcl
# Compiles and simulates the DSP48E2 testbench using Vivado xsim.
#
# Usage (from the udp_rtl/ project root):
#   vivado -mode batch -source tcl/sim_dsp.tcl
#
# Or run interactively in Vivado TCL console:
#   source tcl/sim_dsp.tcl
# =============================================================================

# --- Configuration ---
set rtl_dir  "../rtl"
set tb_dir   "../tb"
set sim_dir  "./sim_dsp_work"
set snapshot "tb_dsp48e2_snap"

puts "============================================================"
puts "  UDP Project — DSP48E2 Simulation"
puts "============================================================"

# Create and enter the simulation working directory
file mkdir $sim_dir
cd $sim_dir
puts "\nWorking directory: [pwd]"

# -----------------------------------------------------------------------------
# Step 1: Compile RTL and testbench sources with xvlog
# -----------------------------------------------------------------------------
puts "\n--- Step 1: Compiling sources ---"

set rtl_files [list \
    "$rtl_dir/dsp48e2_model.v" \
]

set tb_files [list \
    "$tb_dir/tb_dsp48e2.v" \
]

foreach f $rtl_files {
    puts "  Compiling RTL: $f"
    if {[catch {exec xvlog $f} result]} {
        puts "ERROR compiling $f:\n$result"
        exit 1
    }
}

foreach f $tb_files {
    puts "  Compiling TB:  $f"
    if {[catch {exec xvlog $f} result]} {
        puts "ERROR compiling $f:\n$result"
        exit 1
    }
}

puts "  Compilation OK"

# -----------------------------------------------------------------------------
# Step 2: Elaborate — link design, create simulation snapshot
# -----------------------------------------------------------------------------
puts "\n--- Step 2: Elaborating (snapshot: $snapshot) ---"
if {[catch {exec xelab -debug typical tb_dsp48e2 -s $snapshot} result]} {
    puts "ERROR during elaboration:\n$result"
    exit 1
}
puts "  Elaboration OK"

# -----------------------------------------------------------------------------
# Step 3: Simulate
# -----------------------------------------------------------------------------
puts "\n--- Step 3: Running simulation ---"
set xsim_result [exec xsim $snapshot -runall]
puts $xsim_result

puts "\n============================================================"
puts "  sim_dsp.tcl complete.  See output above for PASS/FAIL."
puts "============================================================"
