# Copyright 2020 ETH Zurich and University of Bologna.
# Solderpad Hardware License, Version 0.51, see LICENSE for details.
# SPDX-License-Identifier: SHL-0.51
#
# Nils Wistoff <nwistoff@iis.ee.ethz.ch>

# Parse arguments (Vivado's boolean properties are not compatible with all tcl boolean variables)
set DEBUG false
set EXT_JTAG false

if {$argc > 0 && [lindex $argv 0]} { set DEBUG true }
if {$argc > 1 && [lindex $argv 1]} { set EXT_JTAG true }

# Create project
set project hemaia_chip

create_project $project ./hemaia_chip -force -part xcvp1802-lsvc4072-2MP-e-S
set_property XPM_LIBRARIES XPM_MEMORY [current_project]

# Define sources
source define-sources.tcl

# Add constraints
set constraint_file synth_constraints.xdc
if {[file exists ${constraint_file}]} {
    add_files -fileset constrs_1 -norecurse ${constraint_file}
    set_property USED_IN {synthesis out_of_context} [get_files ${constraint_file}]
}

# Buggy Vivado doesn't like these files. That's ok, we don't need them anyways.
set_property IS_ENABLED 0 [get_files -regex .*/axi_intf.sv]
set_property IS_ENABLED 0 [get_files -regex .*/reg_intf.sv]

# Package IP
set_property top occamy_chip [current_fileset]

update_compile_order -fileset sources_1
# This is just a quick synthesize to ensure there is no errors in source code
synth_design -rtl -name rtl_1

ipx::package_project -root_dir . -vendor MICAS_KUL -library user -taxonomy /UserIP -set_current true

# Clock interface
ipx::infer_bus_interface clk_i xilinx.com:signal:clock_rtl:1.0 [ipx::current_core]

# Reset interface
ipx::infer_bus_interface rst_ni xilinx.com:signal:reset_rtl:1.0 [ipx::current_core]

# Export
set_property core_revision 1 [ipx::current_core]
ipx::create_xgui_files [ipx::current_core]
ipx::update_checksums [ipx::current_core]
ipx::save_core [ipx::current_core]
ipx::check_integrity [ipx::current_core]
ipx::save_core [ipx::current_core]
