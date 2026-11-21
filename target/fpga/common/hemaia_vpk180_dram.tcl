# Solderpad Hardware License, Version 0.51, see LICENSE for details.
# SPDX-License-Identifier: SHL-0.51

source [file join [file dirname [info script]] hemaia_vpk180_dram_bd.tcl]

proc hemaia_connect_dram {chip cips chip_resetn} {
    if {![llength [get_bd_pins -quiet $chip/external_memory_req_o]]} {
        return
    }
    set bridge [create_bd_cell -type ip -vlnv MICAS_KUL:user:hemaia_vpk180_dram_bridge:1.0 dram_bridge]
    foreach {chip_port bridge_port} {
        external_memory_req_o host_req_i
        external_memory_rsp_i host_resp_o
    } {
        set chip_pin [get_bd_pins $chip/$chip_port]
        set bridge_pin [get_bd_pins $bridge/$bridge_port]
        set left [get_property LEFT $chip_pin]
        if {![string is integer -strict $left] || $left < 511 ||
            $left != [get_property LEFT $bridge_pin] ||
            [get_property RIGHT $chip_pin] ne [get_property RIGHT $bridge_pin]} {
            error "DRAM AXI port width mismatch between $chip_pin and $bridge_pin"
        }
        connect_bd_net $chip_pin $bridge_pin
    }
    connect_bd_net [get_bd_pins $chip/external_memory_clk_o] [get_bd_pins $bridge/host_clk_i]
    connect_bd_net -net [get_bd_nets -of_objects [get_bd_pins $chip/chip_id_i]] \
        [get_bd_pins $bridge/chip_id_i]
    connect_bd_net -net [get_bd_nets -of_objects $chip_resetn] [get_bd_pins $bridge/chip_rst_ni]
    hemaia_configure_vpk180_dram_pmc $cips
    set clock [get_bd_pins $bridge/m_axi_aclk_o]
    hemaia_connect_vpk180_dram [get_bd_intf_pins $bridge/m_axi] $clock $cips
}
