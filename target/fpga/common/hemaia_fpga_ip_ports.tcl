# Solderpad Hardware License, Version 0.51, see LICENSE for details.
# SPDX-License-Identifier: SHL-0.51

# IP Packager can miscount SystemVerilog packed structs. The elaborated design
# exposes their actual flattened bits even when a typedef hides the dimensions.
proc hemaia_elaborated_port_widths {port_names} {
    set widths {}
    foreach name $port_names {
        set bits [get_ports -quiet -regexp [format {^%s\[.*\]$} $name]]
        if {[llength $bits]} {
            dict set widths $name [llength $bits]
        } elseif {[llength [get_ports -quiet $name]]} {
            error "Expected packed AXI port $name to elaborate as a vector"
        }
    }
    return $widths
}

proc hemaia_apply_packaged_port_widths {core widths} {
    dict for {name width} $widths {
        set port [ipx::get_ports $name -of_objects $core]
        set_property -dict [list SIZE_LEFT [expr {$width - 1}] SIZE_RIGHT 0] $port
    }
}
