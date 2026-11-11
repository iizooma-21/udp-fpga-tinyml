# =============================================================================
# create_project.tcl
# Creates a Vivado project with all RTL and testbench sources properly added.
#
# Usage (from the udp_rtl/ project root via SSH or x2GO TCL console):
#   vivado -mode batch -source tcl/create_project.tcl
#
# Or in Vivado TCL Console:
#   source tcl/create_project.tcl
#
# After running: open Vivado GUI and open the project at:
#   ~/udp_project/vivado_proj/udp_rtl.xpr
# =============================================================================

set proj_name "udp_rtl"
set proj_dir  "$::env(HOME)/udp_project/vivado_proj"
set rtl_dir   "$::env(HOME)/udp_project/udp_rtl/rtl"
set tb_dir    "$::env(HOME)/udp_project/udp_rtl/tb"

# Target part — generic Artix-7 (change if your supervisor specifies a board)
set part "xc7a35tcpg236-1"

puts "============================================================"
puts "  Creating Vivado Project: $proj_name"
puts "  Location: $proj_dir"
puts "============================================================"

# Remove old project if it exists, so this script is re-runnable
if {[file exists "$proj_dir/$proj_name.xpr"]} {
    puts "  Removing existing project..."
    file delete -force $proj_dir
}

# Create project
create_project $proj_name $proj_dir -part $part
puts "  Project created."

# -----------------------------------------------------------------------------
# Add RTL source files (design sources)
# -----------------------------------------------------------------------------
puts "\n  Adding RTL sources..."
set rtl_files [list \
    "$rtl_dir/dsp48e2_model.v"   \
    "$rtl_dir/udp_pack.v"        \
    "$rtl_dir/udp_preprocess.v"  \
    "$rtl_dir/udp_postprocess.v" \
    "$rtl_dir/udp_mac_top.v"     \
]

foreach f $rtl_files {
    add_files -norecurse $f
    puts "    + [file tail $f]"
}

# Set the top-level design module
set_property top udp_mac_top [current_fileset]
puts "  Top module set to: udp_mac_top"

# -----------------------------------------------------------------------------
# Add testbench files (simulation sources)
# -----------------------------------------------------------------------------
puts "\n  Adding simulation sources..."
set tb_files [list \
    "$tb_dir/tb_dsp48e2.v" \
    "$tb_dir/tb_udp_mac.v" \
]

foreach f $tb_files {
    add_files -fileset sim_1 -norecurse $f
    puts "    + [file tail $f]"
}

# Set the simulation top to the full UDP MAC testbench
set_property top tb_udp_mac [get_filesets sim_1]
puts "  Simulation top set to: tb_udp_mac"

# -----------------------------------------------------------------------------
# Save and report
# -----------------------------------------------------------------------------
save_project_as $proj_name $proj_dir -force

puts "\n============================================================"
puts "  Project ready."
puts "  To open in Vivado GUI, run:"
puts "    vivado $proj_dir/$proj_name.xpr"
puts ""
puts "  All source files are now visible in the Sources panel."
puts "  Double-click any .v file in Sources to open it."
puts "============================================================"
