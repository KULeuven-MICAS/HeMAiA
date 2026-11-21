# Solderpad Hardware License, Version 0.51, see LICENSE for details.
# SPDX-License-Identifier: SHL-0.51

# Package separately from the SoC so that its original hierarchy and D2D constraints remain valid.
if {[llength [ipx::get_ports -quiet external_memory_req_o -of_objects [ipx::current_core]]]} {
    set bridge_dir [file dirname [file normalize [info script]]]
    set bridge_root [file normalize [file join $bridge_dir ../../..]]
    set bridge_soc_pkg [get_property NAME [get_files */occamy_pkg.sv]]
    set bridge_part [get_property PART [current_project]]
    set bridge_axi [exec bender -d $bridge_root path axi]
    set bridge_dbg [exec bender -d $bridge_root path riscv-dbg]
    set bridge_reg [exec bender -d $bridge_root path register_interface]
    set bridge_apb [exec bender -d $bridge_root path apb]

    ipx::unload_core [ipx::current_core]
    close_project
    create_project hemaia_vpk180_dram_bridge dram_bridge/build -force -part $bridge_part
    add_files -norecurse [list \
        $bridge_axi/src/axi_pkg.sv \
        $bridge_axi/src/axi_modify_address.sv \
        $bridge_dbg/src/dm_pkg.sv \
        $bridge_apb/src/apb_pkg.sv \
        $bridge_soc_pkg \
        $bridge_dir/hemaia_dram_address_mapper.sv \
        $bridge_dir/hemaia_vpk180_dram_bridge.sv]
    set_property include_dirs [list $bridge_axi/include \
        $bridge_reg/include $bridge_apb/include] [current_fileset]
    set_property top hemaia_vpk180_dram_bridge [current_fileset]
    update_compile_order -fileset sources_1
    synth_design -rtl -name bridge_rtl
    source $bridge_dir/hemaia_fpga_ip_ports.tcl
    set bridge_widths [hemaia_elaborated_port_widths {host_req_i host_resp_o}]
    if {[dict size $bridge_widths] != 2} {
        error "Could not obtain both elaborated AXI port widths for the VPK180 DRAM bridge"
    }

    ipx::package_project -root_dir dram_bridge -vendor MICAS_KUL -library user -taxonomy /UserIP -set_current true
    set bridge_core [ipx::current_core]
    hemaia_apply_packaged_port_widths $bridge_core $bridge_widths
    if {![llength [ipx::get_bus_interfaces -quiet m_axi -of_objects $bridge_core]]} {
        ipx::infer_bus_interface m_axi xilinx.com:interface:aximm_rtl:1.0 $bridge_core
    }
    foreach clock_name {host_clk_i m_axi_aclk_o} {
        ipx::infer_bus_interface $clock_name xilinx.com:signal:clock_rtl:1.0 $bridge_core
        set bridge_clock [ipx::get_bus_interfaces $clock_name -of_objects $bridge_core]
        set bridge_frequency [ipx::get_bus_parameters -quiet FREQ_HZ -of_objects $bridge_clock]
        if {![llength $bridge_frequency]} {
            set bridge_frequency [ipx::add_bus_parameter FREQ_HZ $bridge_clock]
        }
        # The FPGA constraints time the host at the 240 MHz reference divided by six.
        set_property value 40000000 $bridge_frequency
    }
    ipx::associate_bus_interfaces -busif m_axi -clock m_axi_aclk_o $bridge_core
    ipx::create_xgui_files $bridge_core
    ipx::update_checksums $bridge_core
    ipx::check_integrity $bridge_core
    ipx::save_core $bridge_core
} elseif {[file exists dram_bridge/component.xml]} {
    file delete dram_bridge/component.xml
}
