# =============================================================================
# create_project.tcl
# Creates a persistent Vivado project (.xpr) for the 12×12 UDP systolic array.
# Once created, open the .xpr file in the Vivado GUI to run synthesis,
# implementation, and simulation interactively.
#
# USAGE (on the server):
#   vivado -mode batch -source tcl/create_project.tcl
#
# After running this script:
#   1. Open Vivado GUI (via x2Go)
#   2. File → Open Project → select ~/udp_project/udp_systolic/vivado_proj/udp_systolic.xpr
#   3. In the GUI you can: Run Synthesis, Run Simulation, view schematic, etc.
# =============================================================================

set SCRIPT_DIR [file dirname [file normalize [info script]]]
set PROJ_DIR   [file dirname $SCRIPT_DIR]

set PE_RTL_DIR  "$PROJ_DIR/../udp_rtl/rtl"
set SYS_RTL_DIR "$PROJ_DIR/rtl"
set TB_DIR      "$PROJ_DIR/tb"
set VIVADO_DIR  "$PROJ_DIR/vivado_proj"

set PART "xczu7ev-ffvc1156-2-e"

# ── Create the project ────────────────────────────────────────────────────────
puts "=== Creating Vivado project at $VIVADO_DIR ==="
create_project udp_systolic $VIVADO_DIR -part $PART -force

# ── Add RTL source files ──────────────────────────────────────────────────────
puts "=== Adding RTL source files ==="
add_files -norecurse [list \
    "$PE_RTL_DIR/dsp48e2_model.v"   \
    "$PE_RTL_DIR/udp_pack.v"        \
    "$PE_RTL_DIR/udp_preprocess.v"  \
    "$PE_RTL_DIR/udp_postprocess.v" \
    "$PE_RTL_DIR/udp_mac_top.v"     \
    "$SYS_RTL_DIR/udp_pe.v"         \
    "$SYS_RTL_DIR/systolic_array.v" \
]

# ── Add simulation (testbench) files ─────────────────────────────────────────
puts "=== Adding testbench ==="
add_files -fileset sim_1 -norecurse "$TB_DIR/tb_systolic.v"

# ── Set top modules ───────────────────────────────────────────────────────────
set_property top systolic_array   [get_filesets sources_1]
set_property top tb_systolic      [get_filesets sim_1]

# ── Set target language ───────────────────────────────────────────────────────
set_property target_language Verilog [current_project]

# ── Save project ──────────────────────────────────────────────────────────────
puts "=== Saving project ==="
save_project_as udp_systolic $VIVADO_DIR -force

puts ""
puts "=========================================================="
puts " PROJECT CREATED SUCCESSFULLY"
puts "=========================================================="
puts " Location : $VIVADO_DIR/udp_systolic.xpr"
puts " Top RTL  : systolic_array"
puts " Top SIM  : tb_systolic"
puts " Part     : $PART"
puts ""
puts " To open in Vivado GUI:"
puts "   1. Connect via x2Go"
puts "   2. Open Vivado"
puts "   3. File → Open Project → $VIVADO_DIR/udp_systolic.xpr"
puts "   OR in the Vivado TCL console:"
puts "      open_project $VIVADO_DIR/udp_systolic.xpr"
puts "=========================================================="
