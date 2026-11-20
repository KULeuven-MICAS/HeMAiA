# Copyright 2026 KU Leuven.
# Solderpad Hardware License, Version 0.51, see LICENSE for details.
# SPDX-License-Identifier: SHL-0.51

# Implementation-only constraints shared by the VPK180 east and west designs.
# The board XDC has already defined the main, peripheral and PHY receive clocks.
set chip hemaia_system_i/occamy_chip/inst
set cluster_clocks [get_pins -quiet -hierarchical -filter "NAME =~ $chip/i_occamy/clk_acc_i*"]
set num_clusters [llength $cluster_clocks]
set main_clock [get_clocks clk_main]
set peripheral_clock [get_clocks clk_peri]
set main_period [get_property PERIOD $main_clock]
set peripheral_period [get_property PERIOD $peripheral_clock]
set host_period [expr {$main_period * 6.0}]
set accelerator_period [expr {$main_period * 8.0}]
set source_pin [get_pins -hierarchical -filter "NAME == $chip/clk_i"]

# Divider order matches occamy_chip. A design without clusters places east
# directly after the host channel.
set clock_domains [list [list clk_host_gen $host_period]]
for {set index 0} {$index < $num_clusters} {incr index} {
    set clock_name [expr {$index == 0 ? "clk_acc_gen" : "clk_acc_gen_$index"}]
    lappend clock_domains [list $clock_name $accelerator_period]
}
foreach direction {east west north south} {
    lappend clock_domains [list clk_d2d_phy_${direction}_gen $main_period]
}

set chip_clock_names {clk_pl_0 clk_main clk_pl_1 clk_pl_2 clk_peri}
set channel -1
foreach domain $clock_domains {
    incr channel
    lassign $domain clock_name period
    set divider_path [format {%s/i_hemaia_clk_rst_controller/gen_clock_divider[%d].i_clk_divider} $chip $channel]
    set output_pin [get_pins -quiet -hierarchical -filter [format {NAME == "%s/clk_o"} $divider_path]]
    if {![llength $output_pin]} {
        # An unused PHY direction can be absent after synthesis.
        continue
    }

    # Divider candidates stay related to clk_main but are mutually exclusive.
    set candidate_clocks {}
    foreach {candidate input divide_by} {even clk0_i 2 odd clk1_i 3} {
        set mux_pin [get_pins -quiet -hierarchical -filter [format {NAME == "%s/i_clk_divided_mux/%s"} $divider_path $input]]
        if {[llength $mux_pin]} {
            set candidate_clock clk_divider_${channel}_$candidate
            create_generated_clock -name $candidate_clock -source $source_pin \
                -divide_by $divide_by -add -master_clock clk_main $mux_pin
            lappend candidate_clocks [get_clocks $candidate_clock]
        }
    }
    if {[llength $candidate_clocks] == 2} {
        set_clock_groups -logically_exclusive \
            -group [lindex $candidate_clocks 0] -group [lindex $candidate_clocks 1]
    }

    create_clock -name $clock_name -period $period $output_pin
    lappend chip_clock_names $clock_name
}

# Vivado uses explicit datapath limits and hold-only false paths for these CDCs.
proc hemaia_set_cdc_constraints {clock_a clock_b max_delay} {
    set_max_delay -datapath_only $max_delay -from $clock_a -to $clock_b
    set_max_delay -datapath_only $max_delay -from $clock_b -to $clock_a
    set_false_path -hold -from $clock_a -to $clock_b
    set_false_path -hold -from $clock_b -to $clock_a
}

set host_clock [get_clocks clk_host_gen]
set_clock_groups -asynchronous -group [get_clocks clk_pl_1]
hemaia_set_cdc_constraints $host_clock $peripheral_clock $host_period
hemaia_set_cdc_constraints $main_clock $peripheral_clock $main_period
foreach accelerator_clock [get_clocks -quiet clk_acc_gen*] {
    hemaia_set_cdc_constraints $host_clock $accelerator_clock $host_period
    hemaia_set_cdc_constraints $peripheral_clock $accelerator_clock \
        [expr {min($peripheral_period, $accelerator_period)}]
}
set host_and_peripheral_clocks [get_clocks {clk_host_gen clk_peri}]
foreach phy_clock [get_clocks -quiet clk_d2d_phy_*_gen] {
    hemaia_set_cdc_constraints $host_and_peripheral_clocks $phy_clock $main_period
}

set_clock_groups -asynchronous \
    -group [get_clocks {rx_clk_1 rx_clk_s1_a_gen_1 rx_clk_s1_b_gen_1 rx_clk_s2_a_gen_1 rx_clk_s2_b_gen_1}] \
    -group [get_clocks {rx_clk_2 rx_clk_s1_a_gen_2 rx_clk_s1_b_gen_2 rx_clk_s2_a_gen_2 rx_clk_s2_b_gen_2}] \
    -group [get_clocks {rx_clk_3 rx_clk_s1_a_gen_3 rx_clk_s1_b_gen_3 rx_clk_s2_a_gen_3 rx_clk_s2_b_gen_3}] \
    -group [get_clocks $chip_clock_names]
