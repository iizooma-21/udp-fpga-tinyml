# =============================================================================
# sim_systolic.tcl
# Run xvlog/xelab/xsim to simulate the 12×12 int4 systolic array.
#
# USAGE (on the server, from the udp_systolic directory):
#   vivado -mode batch -source tcl/sim_systolic.tcl
# OR (directly using xvlog/xelab/xsim if Vivado is sourced):
#   xvlog -f tcl/filelist.f && xelab tb_systolic -s tb_systolic_sim && ...
#
# This script uses Vivado's built-in simulation flow for simplicity.
# =============================================================================

# ── Paths ─────────────────────────────────────────────────────────────────────
set SCRIPT_DIR [file dirname [file normalize [info script]]]
set PROJ_DIR   [file dirname $SCRIPT_DIR]

# Single-PE RTL dependency directory (existing project — do NOT overwrite)
set PE_RTL_DIR "$PROJ_DIR/../udp_rtl/rtl"

# New systolic RTL and testbench
set SYS_RTL_DIR "$PROJ_DIR/rtl"
set TB_DIR      "$PROJ_DIR/tb"
set WORK_DIR    "$PROJ_DIR/sim_work"

# ── Create working directory ──────────────────────────────────────────────────
file mkdir $WORK_DIR

# ── Compile all RTL source files (order matters for dependencies) ─────────────
puts "=== Compiling RTL sources ==="

# Shared single-PE files (from udp_rtl project)
set shared_rtl [list \
    "$PE_RTL_DIR/dsp48e2_model.v"   \
    "$PE_RTL_DIR/udp_pack.v"        \
    "$PE_RTL_DIR/udp_preprocess.v"  \
    "$PE_RTL_DIR/udp_postprocess.v" \
    "$PE_RTL_DIR/udp_mac_top.v"     \
]

# New systolic files
set sys_rtl [list \
    "$SYS_RTL_DIR/udp_pe.v"          \
    "$SYS_RTL_DIR/systolic_array.v"  \
]

# Testbench
set tb_files [list \
    "$TB_DIR/tb_systolic.v" \
]

# Compile all files
foreach f [concat $shared_rtl $sys_rtl $tb_files] {
    puts "  Compiling: $f"
    exec xvlog -d SIMULATION "$f" -work work 2>@stdout
}

# ── Elaborate ─────────────────────────────────────────────────────────────────
puts "\n=== Elaborating tb_systolic ==="
exec xelab -debug typical tb_systolic -s tb_systolic_sim -timescale 1ns/1ps 2>@stdout

# ── Simulate ──────────────────────────────────────────────────────────────────
puts "\n=== Running Simulation ==="
exec xsim tb_systolic_sim --runall 2>@stdout

puts "\n=== Simulation complete ==="
