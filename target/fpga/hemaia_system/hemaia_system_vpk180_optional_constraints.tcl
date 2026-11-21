# Solderpad Hardware License, Version 0.51, see LICENSE for details.
# SPDX-License-Identifier: SHL-0.51

# These interfaces and clock domains depend on the generated SoC.
# Six-wires SPIx4 (Slave)
if {[llength [get_ports -quiet spis_sck_i]]} {
    # J1.39 - LA09_N - CD46
    set_property PACKAGE_PIN CD46 [get_ports {spis_sd_io[0]}]
    set_property IOSTANDARD LVCMOS15 [get_ports {spis_sd_io[0]}]
    # J1.37 - LA09_P - CC45
    set_property PACKAGE_PIN CC45 [get_ports {spis_sd_io[1]}]
    set_property IOSTANDARD LVCMOS15 [get_ports {spis_sd_io[1]}]
    # J1.35 - LA08_N - BY50
    set_property PACKAGE_PIN BY50 [get_ports {spis_sd_io[2]}]
    set_property IOSTANDARD LVCMOS15 [get_ports {spis_sd_io[2]}]
    # J1.33 - LA08_P - BY49
    set_property PACKAGE_PIN BY49 [get_ports {spis_sd_io[3]}]
    set_property IOSTANDARD LVCMOS15 [get_ports {spis_sd_io[3]}]
    # J1.31 - LA07_N - CC50
    set_property PACKAGE_PIN CC50 [get_ports spis_csb_i]
    set_property IOSTANDARD LVCMOS15 [get_ports spis_csb_i]
    # J1.29 - LA07_P - CB49
    set_property PACKAGE_PIN CB49 [get_ports spis_sck_i]
    set_property IOSTANDARD LVCMOS15 [get_ports spis_sck_i]

    create_clock -period 50.0 -name spi_s_sck [get_ports spis_sck_i]
    set_property CLOCK_DEDICATED_ROUTE FALSE [get_nets spis_sck_i_IBUF]
    set_property CLOCK_DEDICATED_ROUTE ANY_CMT_REGION [get_nets spis_sck_i_IBUF_BUFGCE]
}

# Set RTC as false path when it reaches the CLINT.
set rtc_sync [get_pins -quiet {hemaia_system_i/occamy_chip/inst/i_occamy/i_clint/i_sync_edge/i_sync/reg_q_reg[0]/D}]
if {[llength $rtc_sync]} {
    set_false_path -to $rtc_sync
}

if {[llength [get_pins -quiet hemaia_system_i/occamy_chip/inst/i_occamy/clk_acc_i*]]} {
    create_generated_clock \
        -name clk_acc \
        -source [get_pins hemaia_system_i/occamy_chip/inst/clk_i] \
        -divide_by 6 \
        [get_pins {hemaia_system_i/occamy_chip/inst/i_hemaia_clk_rst_controller/gen_clock_divider[1].i_clk_divider/clk_o}]
}

set clock_group_args {}
foreach domain {{clk_pl_0 clk_main} clk_core clk_acc clk_pl_1 {clk_pl_2 clk_peri} spi_s_sck} {
    set domain_clocks [get_clocks -quiet $domain]
    if {[llength $domain_clocks]} {
        lappend clock_group_args -group $domain_clocks
    }
}
set_clock_groups -asynchronous {*}$clock_group_args
