# Copyright 2024 KU Leuven and ETH Zurich.
# Solderpad Hardware License, Version 0.51, see LICENSE for details.
# SPDX-License-Identifier: SHL-0.51
#
# Nils Wistoff <nwistoff@iis.ee.ethz.ch>
# Yunhao Deng <yunhao.deng@kuleuven.be>

set iostd LVCMOS12
set d2d_io_port west_d2d_io_0
set d2d_fc_cts_i_port flow_control_west_cts_i_0
set d2d_fc_cts_o_port flow_control_west_cts_o_0
set d2d_fc_rts_i_port flow_control_west_rts_i_0
set d2d_fc_rts_o_port flow_control_west_rts_o_0
set d2d_test_req_port west_test_request_o_0
set d2d_test_brq_port west_test_being_requested_i_0

## Channel per cable mode pinout
## This pinout does not match FPGA banks
#  Channel 1 -> cable 1,2
#  Channel 2 -> cable 2
#  Channel 3 -> cable 1
set_property -dict "PACKAGE_PIN CD51 IOSTANDARD $iostd SLEW FAST" [get_ports -filter {NAME == "west_d2d_io_0[0]"}]
set_property -dict "PACKAGE_PIN CD52 IOSTANDARD $iostd SLEW FAST" [get_ports -filter {NAME == "west_d2d_io_0[1]"}]
set_property -dict "PACKAGE_PIN BU41 IOSTANDARD $iostd SLEW FAST" [get_ports -filter {NAME == "west_d2d_io_0[2]"}]
set_property -dict "PACKAGE_PIN BR42 IOSTANDARD $iostd SLEW FAST" [get_ports -filter {NAME == "west_d2d_io_0[3]"}]
set_property -dict "PACKAGE_PIN BU42 IOSTANDARD $iostd SLEW FAST" [get_ports -filter {NAME == "west_d2d_io_0[4]"}]
set_property -dict "PACKAGE_PIN BT41 IOSTANDARD $iostd SLEW FAST" [get_ports -filter {NAME == "west_d2d_io_0[5]"}]
set_property -dict "PACKAGE_PIN BN40 IOSTANDARD $iostd SLEW FAST" [get_ports -filter {NAME == "west_d2d_io_0[6]"}]
set_property -dict "PACKAGE_PIN BW39 IOSTANDARD $iostd SLEW FAST" [get_ports -filter {NAME == "west_d2d_io_0[7]"}]
set_property -dict "PACKAGE_PIN BP40 IOSTANDARD $iostd SLEW FAST" [get_ports -filter {NAME == "west_d2d_io_0[8]"}]
set_property -dict "PACKAGE_PIN BY39 IOSTANDARD $iostd SLEW FAST" [get_ports -filter {NAME == "west_d2d_io_0[9]"}]
set_property -dict "PACKAGE_PIN CB40 IOSTANDARD $iostd SLEW FAST" [get_ports -filter {NAME == "west_d2d_io_0[10]"}]
set_property -dict "PACKAGE_PIN CD42 IOSTANDARD $iostd SLEW FAST" [get_ports -filter {NAME == "west_d2d_io_0[11]"}]
set_property -dict "PACKAGE_PIN CC40 IOSTANDARD $iostd SLEW FAST" [get_ports -filter {NAME == "west_d2d_io_0[12]"}]
set_property -dict "PACKAGE_PIN CD43 IOSTANDARD $iostd SLEW FAST" [get_ports -filter {NAME == "west_d2d_io_0[13]"}]
set_property -dict "PACKAGE_PIN CD40 IOSTANDARD $iostd SLEW FAST" [get_ports -filter {NAME == "west_d2d_io_0[14]"}]
set_property -dict "PACKAGE_PIN CD41 IOSTANDARD $iostd SLEW FAST" [get_ports -filter {NAME == "west_d2d_io_0[15]"}]
set_property -dict "PACKAGE_PIN CB41 IOSTANDARD $iostd SLEW FAST" [get_ports -filter {NAME == "west_d2d_io_0[16]"}]
set_property -dict "PACKAGE_PIN CA38 IOSTANDARD $iostd SLEW FAST" [get_ports -filter {NAME == "west_d2d_io_0[17]"}]
set_property -dict "PACKAGE_PIN CC42 IOSTANDARD $iostd SLEW FAST" [get_ports -filter {NAME == "west_d2d_io_0[18]"}]
set_property -dict "PACKAGE_PIN CC38 IOSTANDARD $iostd SLEW FAST" [get_ports -filter {NAME == "west_d2d_io_0[19]"}]
set_property -dict "PACKAGE_PIN CB39 IOSTANDARD $iostd SLEW FAST" [get_ports -filter {NAME == "west_d2d_io_0[20]"}]
set_property -dict "PACKAGE_PIN BT37 IOSTANDARD $iostd SLEW FAST" [get_ports -filter {NAME == "west_d2d_io_0[21]"}]
set_property -dict "PACKAGE_PIN BY40 IOSTANDARD $iostd SLEW FAST" [get_ports -filter {NAME == "west_d2d_io_0[22]"}]
set_property -dict "PACKAGE_PIN CC39 IOSTANDARD $iostd SLEW FAST" [get_ports -filter {NAME == "west_d2d_io_0[23]"}]
set_property -dict "PACKAGE_PIN BU37 IOSTANDARD $iostd SLEW FAST" [get_ports -filter {NAME == "west_d2d_io_0[24]"}]
set_property -dict "PACKAGE_PIN CA39 IOSTANDARD $iostd SLEW FAST" [get_ports -filter {NAME == "west_d2d_io_0[25]"}]
set_property -dict "PACKAGE_PIN BY38 IOSTANDARD $iostd SLEW FAST" [get_ports -filter {NAME == "west_d2d_io_0[26]"}]
set_property -dict "PACKAGE_PIN BR37 IOSTANDARD $iostd SLEW FAST" [get_ports -filter {NAME == "west_d2d_io_0[27]"}]
set_property -dict "PACKAGE_PIN CA37 IOSTANDARD $iostd SLEW FAST" [get_ports -filter {NAME == "west_d2d_io_0[28]"}]
set_property -dict "PACKAGE_PIN BR38 IOSTANDARD $iostd SLEW FAST" [get_ports -filter {NAME == "west_d2d_io_0[29]"}]
set_property -dict "PACKAGE_PIN BR41 IOSTANDARD $iostd SLEW FAST" [get_ports -filter {NAME == "west_d2d_io_0[30]"}]
set_property -dict "PACKAGE_PIN BR39 IOSTANDARD $iostd SLEW FAST" [get_ports -filter {NAME == "west_d2d_io_0[31]"}]
set_property -dict "PACKAGE_PIN BT40 IOSTANDARD $iostd SLEW FAST" [get_ports -filter {NAME == "west_d2d_io_0[32]"}]
set_property -dict "PACKAGE_PIN BT39 IOSTANDARD $iostd SLEW FAST" [get_ports -filter {NAME == "west_d2d_io_0[33]"}]
set_property -dict "PACKAGE_PIN BV41 IOSTANDARD $iostd SLEW FAST" [get_ports -filter {NAME == "west_d2d_io_0[34]"}]
set_property -dict "PACKAGE_PIN BY41 IOSTANDARD $iostd SLEW FAST" [get_ports -filter {NAME == "west_d2d_io_0[35]"}]
set_property -dict "PACKAGE_PIN BW41 IOSTANDARD $iostd SLEW FAST" [get_ports -filter {NAME == "west_d2d_io_0[36]"}]
set_property -dict "PACKAGE_PIN CA41 IOSTANDARD $iostd SLEW FAST" [get_ports -filter {NAME == "west_d2d_io_0[37]"}]
set_property -dict "PACKAGE_PIN BY43 IOSTANDARD $iostd SLEW FAST" [get_ports -filter {NAME == "west_d2d_io_0[38]"}]
set_property -dict "PACKAGE_PIN BY44 IOSTANDARD $iostd SLEW FAST" [get_ports -filter {NAME == "west_d2d_io_0[39]"}]
set_property -dict "PACKAGE_PIN BV49 IOSTANDARD $iostd SLEW FAST" [get_ports -filter {NAME == "west_d2d_io_0[40]"}]
set_property -dict "PACKAGE_PIN CB46 IOSTANDARD $iostd SLEW FAST" [get_ports -filter {NAME == "west_d2d_io_0[41]"}]
set_property -dict "PACKAGE_PIN BV50 IOSTANDARD $iostd SLEW FAST" [get_ports -filter {NAME == "west_d2d_io_0[42]"}]
set_property -dict "PACKAGE_PIN CC47 IOSTANDARD $iostd SLEW FAST" [get_ports -filter {NAME == "west_d2d_io_0[43]"}]
set_property -dict "PACKAGE_PIN BW51 IOSTANDARD $iostd SLEW FAST" [get_ports -filter {NAME == "west_d2d_io_0[44]"}]
set_property -dict "PACKAGE_PIN BW52 IOSTANDARD $iostd SLEW FAST" [get_ports -filter {NAME == "west_d2d_io_0[45]"}]
set_property -dict "PACKAGE_PIN CD47 IOSTANDARD $iostd SLEW FAST" [get_ports -filter {NAME == "west_d2d_io_0[46]"}]
set_property -dict "PACKAGE_PIN CA44 IOSTANDARD $iostd SLEW FAST" [get_ports -filter {NAME == "west_d2d_io_0[47]"}]
set_property -dict "PACKAGE_PIN CA49 IOSTANDARD $iostd SLEW FAST" [get_ports -filter {NAME == "west_d2d_io_0[48]"}]
set_property -dict "PACKAGE_PIN CD48 IOSTANDARD $iostd SLEW FAST" [get_ports -filter {NAME == "west_d2d_io_0[49]"}]
set_property -dict "PACKAGE_PIN CC43 IOSTANDARD $iostd SLEW FAST" [get_ports -filter {NAME == "west_d2d_io_0[50]"}]
set_property -dict "PACKAGE_PIN CB45 IOSTANDARD $iostd SLEW FAST" [get_ports -filter {NAME == "west_d2d_io_0[51]"}]
set_property -dict "PACKAGE_PIN CB50 IOSTANDARD $iostd SLEW FAST" [get_ports -filter {NAME == "west_d2d_io_0[52]"}]
set_property -dict "PACKAGE_PIN CB44 IOSTANDARD $iostd SLEW FAST" [get_ports -filter {NAME == "west_d2d_io_0[53]"}]
set_property -dict "PACKAGE_PIN BY49 IOSTANDARD $iostd SLEW FAST" [get_ports -filter {NAME == "west_d2d_io_0[54]"}]
set_property -dict "PACKAGE_PIN BY50 IOSTANDARD $iostd SLEW FAST" [get_ports -filter {NAME == "west_d2d_io_0[55]"}]
set_property -dict "PACKAGE_PIN CB49 IOSTANDARD $iostd SLEW FAST" [get_ports -filter {NAME == "west_d2d_io_0[56]"}]
set_property -dict "PACKAGE_PIN CC45 IOSTANDARD $iostd SLEW FAST" [get_ports -filter {NAME == "west_d2d_io_0[57]"}]
set_property -dict "PACKAGE_PIN CC44 IOSTANDARD $iostd SLEW FAST" [get_ports -filter {NAME == "west_d2d_io_0[58]"}]
set_property -dict "PACKAGE_PIN CC50 IOSTANDARD $iostd SLEW FAST" [get_ports -filter {NAME == "west_d2d_io_0[59]"}]

# Test Pins
set_property -dict "PACKAGE_PIN BW50 IOSTANDARD $iostd SLEW FAST" [get_ports "$d2d_test_brq_port"] 
set_property -dict "PACKAGE_PIN CB51 IOSTANDARD $iostd SLEW FAST" [get_ports "$d2d_test_req_port"] 

# Flow Control
set_property -dict "PACKAGE_PIN CD45 IOSTANDARD $iostd SLEW FAST" [get_ports "$d2d_fc_cts_i_port"] 
set_property -dict "PACKAGE_PIN CD46 IOSTANDARD $iostd SLEW FAST" [get_ports "$d2d_fc_cts_o_port"] 
set_property -dict "PACKAGE_PIN BW42 IOSTANDARD $iostd SLEW FAST" [get_ports "$d2d_fc_rts_i_port"] 
set_property -dict "PACKAGE_PIN BW49 IOSTANDARD $iostd SLEW FAST" [get_ports "$d2d_fc_rts_o_port"] 

# JTAG
set_property -dict "PACKAGE_PIN BY51 IOSTANDARD $iostd" [get_ports {jtag_tck_i}] 
set_property -dict "PACKAGE_PIN CD50 IOSTANDARD $iostd" [get_ports {jtag_tdi_i}] 
set_property -dict "PACKAGE_PIN CB52 IOSTANDARD $iostd" [get_ports {jtag_tdo_o}] 
set_property -dict "PACKAGE_PIN CA52 IOSTANDARD $iostd" [get_ports {jtag_tms_i}] 


# UART FC
set_property -dict "PACKAGE_PIN BV43 IOSTANDARD $iostd PULLTYPE PULLUP" [get_ports {uart_cts_ni}]
set_property -dict "PACKAGE_PIN CC52 IOSTANDARD $iostd" [get_ports {uart_rts_no}] 
set_property -dict "PACKAGE_PIN CC49 IOSTANDARD $iostd" [get_ports {uart_rx_i}] 
set_property -dict "PACKAGE_PIN CA51 IOSTANDARD $iostd" [get_ports {uart_tx_o}] 

#VREF
set_property -dict "PACKAGE_PIN BN39 IOSTANDARD $iostd DRIVE 8" [get_ports {vref_vdd_o[0]}] 
set_property -dict "PACKAGE_PIN BP38 IOSTANDARD $iostd DRIVE 8" [get_ports {vref_gnd_o[0]}] 

# Four-wires GPIO_O connected to LEDs
set_property PACKAGE_PIN BA49 [get_ports {gpio_d_o[0]}]
set_property IOSTANDARD LVCMOS15 [get_ports {gpio_d_o[0]}]
set_property PACKAGE_PIN AY50 [get_ports {gpio_d_o[1]}]
set_property IOSTANDARD LVCMOS15 [get_ports {gpio_d_o[1]}]
set_property PACKAGE_PIN BA48 [get_ports {gpio_d_o[2]}]
set_property IOSTANDARD LVCMOS15 [get_ports {gpio_d_o[2]}]
set_property PACKAGE_PIN AY49 [get_ports {gpio_d_o[3]}]
set_property IOSTANDARD LVCMOS15 [get_ports {gpio_d_o[3]}]


# CPU_RESET pushbutton switch
# SW4 is the reset button
set_false_path -from [get_ports reset]
set_false_path -through [get_pins {hemaia_system_i/axis_vio_0/probe_out0[0]}]
set_false_path -through [get_pins {hemaia_system_i/axis_vio_0/probe_out4[0]}]
set_false_path -through [get_pins hemaia_system_i/axis_vio_0/probe_out3*]
set_false_path -through [get_pins hemaia_system_i/axis_vio_0/probe_out2*]
set_false_path -through [get_pins hemaia_system_i/axis_vio_0/probe_out1*]
set_false_path -through [get_pins hemaia_system_i/occamy_chip/*test_being_requested_i]
set_false_path -from [get_ports $d2d_test_brq_port]
set_false_path -to [get_ports $d2d_test_req_port]
set_false_path -to [get_ports {vref_gnd_o[0]}]
set_false_path -to [get_ports {vref_vdd_o[0]}]
set_false_path -through [get_pins hemaia_system_i/occamy_chip/rst_ni]
set_false_path -through [get_pins hemaia_system_i/occamy_chip/rst_periph_ni]
set_property PACKAGE_PIN BT48 [get_ports reset]
set_property IOSTANDARD LVCMOS15 [get_ports reset]

# Set RTC as false path
set_false_path -to [get_pins {hemaia_system_i/occamy_chip/inst/i_occamy/i_clint/i_sync_edge/i_sync/reg_q_reg[0]/D}]

################################################################################
# Clock Domains
################################################################################

create_generated_clock \
    -name clk_main \
    -source [get_pins hemaia_system_i/versal_cips_0/pl0_ref_clk] \
    -divide_by 1 \
    [get_pins hemaia_system_i/occamy_chip/inst/clk_i]

create_generated_clock \
    -name clk_peri \
    -source [get_pins hemaia_system_i/versal_cips_0/pl2_ref_clk] \
    -divide_by 1 \
    [get_pins hemaia_system_i/occamy_chip/inst/clk_periph_i]

set clk_main_period [get_property PERIOD [get_clocks clk_main]]
set clk_peri_period [get_property PERIOD [get_clocks clk_peri]]
set clock_host_period [expr {$clk_main_period * 6.0}]
set clock_acc_period [expr {$clk_main_period * 8.0}]
set clock_d2d_phy_period [expr {$clk_main_period * 1.0}]

# Model the programmable divider internals like the chip SDC. The even/odd
# divider candidates stay related to clk_main; each final channel clock is an
# independent output-domain clock at the divider output.
create_generated_clock -name clk_divider_0_even \
  -source [get_pins -hierarchical -filter {NAME == "hemaia_system_i/occamy_chip/inst/clk_i"}] \
  -divide_by 2 -master_clock clk_main \
  [get_pins -hierarchical -filter {NAME == "hemaia_system_i/occamy_chip/inst/i_hemaia_clk_rst_controller/gen_clock_divider[0].i_clk_divider/i_clk_divided_mux/clk0_i"}]
create_generated_clock -name clk_divider_0_odd \
  -source [get_pins -hierarchical -filter {NAME == "hemaia_system_i/occamy_chip/inst/clk_i"}] \
  -divide_by 3 -master_clock clk_main \
  [get_pins -hierarchical -filter {NAME == "hemaia_system_i/occamy_chip/inst/i_hemaia_clk_rst_controller/gen_clock_divider[0].i_clk_divider/i_clk_divided_mux/clk1_i"}]
set_clock_groups -logically_exclusive -group [get_clocks clk_divider_0_even] -group [get_clocks clk_divider_0_odd]

create_generated_clock -name clk_divider_1_even \
  -source [get_pins -hierarchical -filter {NAME == "hemaia_system_i/occamy_chip/inst/clk_i"}] \
  -divide_by 2 -master_clock clk_main \
  [get_pins -hierarchical -filter {NAME == "hemaia_system_i/occamy_chip/inst/i_hemaia_clk_rst_controller/gen_clock_divider[1].i_clk_divider/i_clk_divided_mux/clk0_i"}]
create_generated_clock -name clk_divider_1_odd \
  -source [get_pins -hierarchical -filter {NAME == "hemaia_system_i/occamy_chip/inst/clk_i"}] \
  -divide_by 3 -master_clock clk_main \
  [get_pins -hierarchical -filter {NAME == "hemaia_system_i/occamy_chip/inst/i_hemaia_clk_rst_controller/gen_clock_divider[1].i_clk_divider/i_clk_divided_mux/clk1_i"}]
set_clock_groups -logically_exclusive -group [get_clocks clk_divider_1_even] -group [get_clocks clk_divider_1_odd]

create_generated_clock -name clk_divider_2_even \
  -source [get_pins -hierarchical -filter {NAME == "hemaia_system_i/occamy_chip/inst/clk_i"}] \
  -divide_by 2 -master_clock clk_main \
  [get_pins -hierarchical -filter {NAME == "hemaia_system_i/occamy_chip/inst/i_hemaia_clk_rst_controller/gen_clock_divider[2].i_clk_divider/i_clk_divided_mux/clk0_i"}]
create_generated_clock -name clk_divider_2_odd \
  -source [get_pins -hierarchical -filter {NAME == "hemaia_system_i/occamy_chip/inst/clk_i"}] \
  -divide_by 3 -master_clock clk_main \
  [get_pins -hierarchical -filter {NAME == "hemaia_system_i/occamy_chip/inst/i_hemaia_clk_rst_controller/gen_clock_divider[2].i_clk_divider/i_clk_divided_mux/clk1_i"}]
set_clock_groups -logically_exclusive -group [get_clocks clk_divider_2_even] -group [get_clocks clk_divider_2_odd]

create_generated_clock -name clk_divider_3_even \
  -source [get_pins -hierarchical -filter {NAME == "hemaia_system_i/occamy_chip/inst/clk_i"}] \
  -divide_by 2 -master_clock clk_main \
  [get_pins -hierarchical -filter {NAME == "hemaia_system_i/occamy_chip/inst/i_hemaia_clk_rst_controller/gen_clock_divider[3].i_clk_divider/i_clk_divided_mux/clk0_i"}]
create_generated_clock -name clk_divider_3_odd \
  -source [get_pins -hierarchical -filter {NAME == "hemaia_system_i/occamy_chip/inst/clk_i"}] \
  -divide_by 3 -master_clock clk_main \
  [get_pins -hierarchical -filter {NAME == "hemaia_system_i/occamy_chip/inst/i_hemaia_clk_rst_controller/gen_clock_divider[3].i_clk_divider/i_clk_divided_mux/clk1_i"}]
set_clock_groups -logically_exclusive -group [get_clocks clk_divider_3_even] -group [get_clocks clk_divider_3_odd]

create_generated_clock -name clk_divider_4_even \
  -source [get_pins -hierarchical -filter {NAME == "hemaia_system_i/occamy_chip/inst/clk_i"}] \
  -divide_by 2 -master_clock clk_main \
  [get_pins -hierarchical -filter {NAME == "hemaia_system_i/occamy_chip/inst/i_hemaia_clk_rst_controller/gen_clock_divider[4].i_clk_divider/i_clk_divided_mux/clk0_i"}]
create_generated_clock -name clk_divider_4_odd \
  -source [get_pins -hierarchical -filter {NAME == "hemaia_system_i/occamy_chip/inst/clk_i"}] \
  -divide_by 3 -master_clock clk_main \
  [get_pins -hierarchical -filter {NAME == "hemaia_system_i/occamy_chip/inst/i_hemaia_clk_rst_controller/gen_clock_divider[4].i_clk_divider/i_clk_divided_mux/clk1_i"}]
set_clock_groups -logically_exclusive -group [get_clocks clk_divider_4_even] -group [get_clocks clk_divider_4_odd]

create_generated_clock -name clk_divider_5_even \
  -source [get_pins -hierarchical -filter {NAME == "hemaia_system_i/occamy_chip/inst/clk_i"}] \
  -divide_by 2 -master_clock clk_main \
  [get_pins -hierarchical -filter {NAME == "hemaia_system_i/occamy_chip/inst/i_hemaia_clk_rst_controller/gen_clock_divider[5].i_clk_divider/i_clk_divided_mux/clk0_i"}]
create_generated_clock -name clk_divider_5_odd \
  -source [get_pins -hierarchical -filter {NAME == "hemaia_system_i/occamy_chip/inst/clk_i"}] \
  -divide_by 3 -master_clock clk_main \
  [get_pins -hierarchical -filter {NAME == "hemaia_system_i/occamy_chip/inst/i_hemaia_clk_rst_controller/gen_clock_divider[5].i_clk_divider/i_clk_divided_mux/clk1_i"}]
set_clock_groups -logically_exclusive -group [get_clocks clk_divider_5_even] -group [get_clocks clk_divider_5_odd]

create_clock -name clk_host_gen -period $clock_host_period \
  [get_pins -hierarchical -filter {NAME == "hemaia_system_i/occamy_chip/inst/i_hemaia_clk_rst_controller/gen_clock_divider[0].i_clk_divider/clk_o"}]

create_clock -name clk_acc_gen -period $clock_acc_period \
  [get_pins -hierarchical -filter {NAME == "hemaia_system_i/occamy_chip/inst/i_hemaia_clk_rst_controller/gen_clock_divider[1].i_clk_divider/clk_o"}]

create_clock -name clk_d2d_phy_east_gen -period $clock_d2d_phy_period \
  [get_pins -hierarchical -filter {NAME == "hemaia_system_i/occamy_chip/inst/i_hemaia_clk_rst_controller/gen_clock_divider[2].i_clk_divider/clk_o"}]

create_clock -name clk_d2d_phy_west_gen -period $clock_d2d_phy_period \
  [get_pins -hierarchical -filter {NAME == "hemaia_system_i/occamy_chip/inst/i_hemaia_clk_rst_controller/gen_clock_divider[3].i_clk_divider/clk_o"}]

create_clock -name clk_d2d_phy_north_gen -period $clock_d2d_phy_period \
  [get_pins -hierarchical -filter {NAME == "hemaia_system_i/occamy_chip/inst/i_hemaia_clk_rst_controller/gen_clock_divider[4].i_clk_divider/clk_o"}]

create_clock -name clk_d2d_phy_south_gen -period $clock_d2d_phy_period \
  [get_pins -hierarchical -filter {NAME == "hemaia_system_i/occamy_chip/inst/i_hemaia_clk_rst_controller/gen_clock_divider[5].i_clk_divider/clk_o"}]

# Vivado does not support the Synopsys -allow_paths clock-group style used in
# the chip SDC. Use explicit CDC max-delay plus hold-only false paths instead.
set_clock_groups -asynchronous -group [get_clocks clk_pl_1]

set_max_delay -datapath_only $clock_host_period -from [get_clocks clk_host_gen] -to [get_clocks clk_peri]
set_max_delay -datapath_only $clock_host_period -from [get_clocks clk_peri] -to [get_clocks clk_host_gen]
set_false_path -hold -from [get_clocks clk_host_gen] -to [get_clocks clk_peri]
set_false_path -hold -from [get_clocks clk_peri] -to [get_clocks clk_host_gen]

set_max_delay -datapath_only $clk_main_period -from [get_clocks clk_main] -to [get_clocks clk_peri]
set_max_delay -datapath_only $clk_main_period -from [get_clocks clk_peri] -to [get_clocks clk_main]
set_false_path -hold -from [get_clocks clk_main] -to [get_clocks clk_peri]
set_false_path -hold -from [get_clocks clk_peri] -to [get_clocks clk_main]

set_max_delay -datapath_only $clock_host_period -from [get_clocks clk_host_gen] -to [get_clocks clk_acc_gen]
set_max_delay -datapath_only $clock_host_period -from [get_clocks clk_acc_gen] -to [get_clocks clk_host_gen]
set_false_path -hold -from [get_clocks clk_host_gen] -to [get_clocks clk_acc_gen]
set_false_path -hold -from [get_clocks clk_acc_gen] -to [get_clocks clk_host_gen]

set_max_delay -datapath_only $clock_d2d_phy_period -from [get_clocks {clk_host_gen clk_peri}] -to [get_clocks clk_d2d_phy_east_gen]
set_max_delay -datapath_only $clock_d2d_phy_period -from [get_clocks clk_d2d_phy_east_gen] -to [get_clocks {clk_host_gen clk_peri}]
set_false_path -hold -from [get_clocks {clk_host_gen clk_peri}] -to [get_clocks clk_d2d_phy_east_gen]
set_false_path -hold -from [get_clocks clk_d2d_phy_east_gen] -to [get_clocks {clk_host_gen clk_peri}]

set_max_delay -datapath_only $clock_d2d_phy_period -from [get_clocks {clk_host_gen clk_peri}] -to [get_clocks clk_d2d_phy_west_gen]
set_max_delay -datapath_only $clock_d2d_phy_period -from [get_clocks clk_d2d_phy_west_gen] -to [get_clocks {clk_host_gen clk_peri}]
set_false_path -hold -from [get_clocks {clk_host_gen clk_peri}] -to [get_clocks clk_d2d_phy_west_gen]
set_false_path -hold -from [get_clocks clk_d2d_phy_west_gen] -to [get_clocks {clk_host_gen clk_peri}]

set_max_delay -datapath_only $clock_d2d_phy_period -from [get_clocks {clk_host_gen clk_peri}] -to [get_clocks clk_d2d_phy_north_gen]
set_max_delay -datapath_only $clock_d2d_phy_period -from [get_clocks clk_d2d_phy_north_gen] -to [get_clocks {clk_host_gen clk_peri}]
set_false_path -hold -from [get_clocks {clk_host_gen clk_peri}] -to [get_clocks clk_d2d_phy_north_gen]
set_false_path -hold -from [get_clocks clk_d2d_phy_north_gen] -to [get_clocks {clk_host_gen clk_peri}]

set_max_delay -datapath_only $clock_d2d_phy_period -from [get_clocks {clk_host_gen clk_peri}] -to [get_clocks clk_d2d_phy_south_gen]
set_max_delay -datapath_only $clock_d2d_phy_period -from [get_clocks clk_d2d_phy_south_gen] -to [get_clocks {clk_host_gen clk_peri}]
set_false_path -hold -from [get_clocks {clk_host_gen clk_peri}] -to [get_clocks clk_d2d_phy_south_gen]
set_false_path -hold -from [get_clocks clk_d2d_phy_south_gen] -to [get_clocks {clk_host_gen clk_peri}]

# CDC 2phase clearable of DM: i_cdc_resp/i_cdc_req
# CONSTRAINT: Requires max_delay of min_period(src_clk_i, dst_clk_i) through the paths async_req, async_ack, async_data.
set_max_delay 10.000 -through [get_nets -quiet -hierarchical -filter {NAME =~ *i_cdc_resp/async_req*}]
set_false_path -hold -through [get_nets -quiet -hierarchical -filter {NAME =~ *i_cdc_resp/async_req*}]
set_max_delay 10.000 -through [get_nets -quiet -hierarchical -filter {NAME =~ *i_cdc_resp/async_ack*}]
set_false_path -hold -through [get_nets -quiet -hierarchical -filter {NAME =~ *i_cdc_resp/async_ack*}]
set_max_delay 10.000 -through [get_nets -quiet -hierarchical -filter {NAME =~ *i_cdc_resp/async_data*}]
set_false_path -hold -through [get_nets -quiet -hierarchical -filter {NAME =~ *i_cdc_resp/async_data*}]
set_max_delay 10.000 -through [get_nets -quiet -hierarchical -filter {NAME =~ *i_cdc_req/async_req*}]
set_false_path -hold -through [get_nets -quiet -hierarchical -filter {NAME =~ *i_cdc_req/async_req*}]
set_max_delay 10.000 -through [get_nets -quiet -hierarchical -filter {NAME =~ *i_cdc_req/async_ack*}]
set_false_path -hold -through [get_nets -quiet -hierarchical -filter {NAME =~ *i_cdc_req/async_ack*}]
set_max_delay 10.000 -through [get_nets -quiet -hierarchical -filter {NAME =~ *i_cdc_req/async_data*}]
set_false_path -hold -through [get_nets -quiet -hierarchical -filter {NAME =~ *i_cdc_req/async_data*}]

################################################################################
# False Paths at IO
################################################################################
set_false_path -through [get_pins hemaia_system_i/occamy_chip/inst/chip_id_i]
set_false_path -through [get_pins hemaia_system_i/occamy_chip/inst/boot_mode_i]
set_false_path -through [get_pins hemaia_system_i/occamy_chip/inst/test_mode_i]
set_false_path -through [get_pins hemaia_system_i/occamy_chip/inst/gpio_d_i]
set_false_path -through [get_pins hemaia_system_i/occamy_chip/inst/gpio_d_o]
set_false_path -through [get_pins hemaia_system_i/occamy_chip/inst/gpio_oe_o]


########## D2D #############
set d2d_clk_period [get_property PERIOD [get_clocks clk_d2d_phy_west_gen]]
set d2d_rx_mux_cdc_period [expr {0.5 * $d2d_clk_period}]
set d2d_phy_async_period [expr {0.5 * $d2d_clk_period}]

create_generated_clock -name tx_clk_1 \
  -source [get_pins -hierarchical -filter {NAME == "hemaia_system_i/occamy_chip/inst/i_hemaia_clk_rst_controller/gen_clock_divider[3].i_clk_divider/clk_o"}] \
  -divide_by 1 -master_clock clk_d2d_phy_west_gen \
  [get_pins -hierarchical -filter {NAME == "hemaia_system_i/occamy_chip/inst/i_d2d_link/gen_west_phy_enabled.gen_phy_channels[0].i_phy_interface_west/i_phy_digital_interface/clk_phy_tx_i"}]
create_generated_clock -name tx_clk_2 \
  -source [get_pins -hierarchical -filter {NAME == "hemaia_system_i/occamy_chip/inst/i_hemaia_clk_rst_controller/gen_clock_divider[3].i_clk_divider/clk_o"}] \
  -divide_by 1 -master_clock clk_d2d_phy_west_gen \
  [get_pins -hierarchical -filter {NAME == "hemaia_system_i/occamy_chip/inst/i_d2d_link/gen_west_phy_enabled.gen_phy_channels[1].i_phy_interface_west/i_phy_digital_interface/clk_phy_tx_i"}]
create_generated_clock -name tx_clk_3 \
  -source [get_pins -hierarchical -filter {NAME == "hemaia_system_i/occamy_chip/inst/i_hemaia_clk_rst_controller/gen_clock_divider[3].i_clk_divider/clk_o"}] \
  -divide_by 1 -master_clock clk_d2d_phy_west_gen \
  [get_pins -hierarchical -filter {NAME == "hemaia_system_i/occamy_chip/inst/i_d2d_link/gen_west_phy_enabled.gen_phy_channels[2].i_phy_interface_west/i_phy_digital_interface/clk_phy_tx_i"}]

create_clock -period $d2d_clk_period -name rx_clk_1 [get_ports -filter {NAME == "west_d2d_io_0[0]"}]
create_generated_clock -name rx_clk_1_sel0 \
  -source [get_ports -filter {NAME == "west_d2d_io_0[0]"}] \
  -combinational -master_clock rx_clk_1 \
  [get_pins -hierarchical -filter {NAME == "hemaia_system_i/occamy_chip/inst/i_d2d_link/gen_west_phy_enabled.gen_phy_channels[0].i_phy_interface_west/i_phy_digital_interface/i_rx_clk_delay_controller/i_final_clk_mux/clk0"}]
create_generated_clock -name rx_clk_1_sel1 \
  -source [get_ports -filter {NAME == "west_d2d_io_0[0]"}] \
  -combinational -master_clock rx_clk_1 \
  [get_pins -hierarchical -filter {NAME == "hemaia_system_i/occamy_chip/inst/i_d2d_link/gen_west_phy_enabled.gen_phy_channels[0].i_phy_interface_west/i_phy_digital_interface/i_rx_clk_delay_controller/i_final_clk_mux/clk1"}]
create_clock -period $d2d_clk_period -name rx_clk_1_gen \
  [get_pins -hierarchical -filter {NAME == "hemaia_system_i/occamy_chip/inst/i_d2d_link/gen_west_phy_enabled.gen_phy_channels[0].i_phy_interface_west/i_phy_digital_interface/i_rx_clk_delay_controller/data_clk_o"}]

create_clock -period $d2d_clk_period -name rx_clk_2 [get_ports -filter {NAME == "west_d2d_io_0[20]"}]
create_generated_clock -name rx_clk_2_sel0 \
  -source [get_ports -filter {NAME == "west_d2d_io_0[20]"}] \
  -combinational -master_clock rx_clk_2 \
  [get_pins -hierarchical -filter {NAME == "hemaia_system_i/occamy_chip/inst/i_d2d_link/gen_west_phy_enabled.gen_phy_channels[1].i_phy_interface_west/i_phy_digital_interface/i_rx_clk_delay_controller/i_final_clk_mux/clk0"}]
create_generated_clock -name rx_clk_2_sel1 \
  -source [get_ports -filter {NAME == "west_d2d_io_0[20]"}] \
  -combinational -master_clock rx_clk_2 \
  [get_pins -hierarchical -filter {NAME == "hemaia_system_i/occamy_chip/inst/i_d2d_link/gen_west_phy_enabled.gen_phy_channels[1].i_phy_interface_west/i_phy_digital_interface/i_rx_clk_delay_controller/i_final_clk_mux/clk1"}]
create_clock -period $d2d_clk_period -name rx_clk_2_gen \
  [get_pins -hierarchical -filter {NAME == "hemaia_system_i/occamy_chip/inst/i_d2d_link/gen_west_phy_enabled.gen_phy_channels[1].i_phy_interface_west/i_phy_digital_interface/i_rx_clk_delay_controller/data_clk_o"}]

create_clock -period $d2d_clk_period -name rx_clk_3 [get_ports -filter {NAME == "west_d2d_io_0[40]"}]
create_generated_clock -name rx_clk_3_sel0 \
  -source [get_ports -filter {NAME == "west_d2d_io_0[40]"}] \
  -combinational -master_clock rx_clk_3 \
  [get_pins -hierarchical -filter {NAME == "hemaia_system_i/occamy_chip/inst/i_d2d_link/gen_west_phy_enabled.gen_phy_channels[2].i_phy_interface_west/i_phy_digital_interface/i_rx_clk_delay_controller/i_final_clk_mux/clk0"}]
create_generated_clock -name rx_clk_3_sel1 \
  -source [get_ports -filter {NAME == "west_d2d_io_0[40]"}] \
  -combinational -master_clock rx_clk_3 \
  [get_pins -hierarchical -filter {NAME == "hemaia_system_i/occamy_chip/inst/i_d2d_link/gen_west_phy_enabled.gen_phy_channels[2].i_phy_interface_west/i_phy_digital_interface/i_rx_clk_delay_controller/i_final_clk_mux/clk1"}]
create_clock -period $d2d_clk_period -name rx_clk_3_gen \
  [get_pins -hierarchical -filter {NAME == "hemaia_system_i/occamy_chip/inst/i_d2d_link/gen_west_phy_enabled.gen_phy_channels[2].i_phy_interface_west/i_phy_digital_interface/i_rx_clk_delay_controller/data_clk_o"}]

set_max_delay -datapath_only $d2d_clk_period -from [get_clocks {clk_host_gen clk_peri}] -to [get_clocks tx_clk_1]
set_max_delay -datapath_only $d2d_clk_period -from [get_clocks tx_clk_1] -to [get_clocks {clk_host_gen clk_peri}]
set_false_path -hold -from [get_clocks {clk_host_gen clk_peri}] -to [get_clocks tx_clk_1]
set_false_path -hold -from [get_clocks tx_clk_1] -to [get_clocks {clk_host_gen clk_peri}]
set_max_delay -datapath_only $d2d_clk_period -from [get_clocks {clk_host_gen clk_peri}] -to [get_clocks rx_clk_1_gen]
set_max_delay -datapath_only $d2d_clk_period -from [get_clocks rx_clk_1_gen] -to [get_clocks {clk_host_gen clk_peri}]
set_false_path -hold -from [get_clocks {clk_host_gen clk_peri}] -to [get_clocks rx_clk_1_gen]
set_false_path -hold -from [get_clocks rx_clk_1_gen] -to [get_clocks {clk_host_gen clk_peri}]
set_max_delay -datapath_only $d2d_rx_mux_cdc_period -from [get_clocks rx_clk_1_sel0] -to [get_clocks rx_clk_1_sel1]
set_max_delay -datapath_only $d2d_rx_mux_cdc_period -from [get_clocks rx_clk_1_sel1] -to [get_clocks rx_clk_1_sel0]
set_false_path -hold -from [get_clocks rx_clk_1_sel0] -to [get_clocks rx_clk_1_sel1]
set_false_path -hold -from [get_clocks rx_clk_1_sel1] -to [get_clocks rx_clk_1_sel0]

set_max_delay -datapath_only $d2d_clk_period -from [get_clocks {clk_host_gen clk_peri}] -to [get_clocks tx_clk_2]
set_max_delay -datapath_only $d2d_clk_period -from [get_clocks tx_clk_2] -to [get_clocks {clk_host_gen clk_peri}]
set_false_path -hold -from [get_clocks {clk_host_gen clk_peri}] -to [get_clocks tx_clk_2]
set_false_path -hold -from [get_clocks tx_clk_2] -to [get_clocks {clk_host_gen clk_peri}]
set_max_delay -datapath_only $d2d_clk_period -from [get_clocks {clk_host_gen clk_peri}] -to [get_clocks rx_clk_2_gen]
set_max_delay -datapath_only $d2d_clk_period -from [get_clocks rx_clk_2_gen] -to [get_clocks {clk_host_gen clk_peri}]
set_false_path -hold -from [get_clocks {clk_host_gen clk_peri}] -to [get_clocks rx_clk_2_gen]
set_false_path -hold -from [get_clocks rx_clk_2_gen] -to [get_clocks {clk_host_gen clk_peri}]
set_max_delay -datapath_only $d2d_rx_mux_cdc_period -from [get_clocks rx_clk_2_sel0] -to [get_clocks rx_clk_2_sel1]
set_max_delay -datapath_only $d2d_rx_mux_cdc_period -from [get_clocks rx_clk_2_sel1] -to [get_clocks rx_clk_2_sel0]
set_false_path -hold -from [get_clocks rx_clk_2_sel0] -to [get_clocks rx_clk_2_sel1]
set_false_path -hold -from [get_clocks rx_clk_2_sel1] -to [get_clocks rx_clk_2_sel0]

set_max_delay -datapath_only $d2d_clk_period -from [get_clocks {clk_host_gen clk_peri}] -to [get_clocks tx_clk_3]
set_max_delay -datapath_only $d2d_clk_period -from [get_clocks tx_clk_3] -to [get_clocks {clk_host_gen clk_peri}]
set_false_path -hold -from [get_clocks {clk_host_gen clk_peri}] -to [get_clocks tx_clk_3]
set_false_path -hold -from [get_clocks tx_clk_3] -to [get_clocks {clk_host_gen clk_peri}]
set_max_delay -datapath_only $d2d_clk_period -from [get_clocks {clk_host_gen clk_peri}] -to [get_clocks rx_clk_3_gen]
set_max_delay -datapath_only $d2d_clk_period -from [get_clocks rx_clk_3_gen] -to [get_clocks {clk_host_gen clk_peri}]
set_false_path -hold -from [get_clocks {clk_host_gen clk_peri}] -to [get_clocks rx_clk_3_gen]
set_false_path -hold -from [get_clocks rx_clk_3_gen] -to [get_clocks {clk_host_gen clk_peri}]
set_max_delay -datapath_only $d2d_rx_mux_cdc_period -from [get_clocks rx_clk_3_sel0] -to [get_clocks rx_clk_3_sel1]
set_max_delay -datapath_only $d2d_rx_mux_cdc_period -from [get_clocks rx_clk_3_sel1] -to [get_clocks rx_clk_3_sel0]
set_false_path -hold -from [get_clocks rx_clk_3_sel0] -to [get_clocks rx_clk_3_sel1]
set_false_path -hold -from [get_clocks rx_clk_3_sel1] -to [get_clocks rx_clk_3_sel0]

set_max_delay $d2d_phy_async_period -to [get_pins -quiet -hierarchical -filter {NAME =~ *i_d2d_link*async_tx_mode_i}]
set_false_path -hold -to [get_pins -quiet -hierarchical -filter {NAME =~ *i_d2d_link*async_tx_mode_i}]
set_max_delay $d2d_phy_async_period -to [get_pins -quiet -hierarchical -filter {NAME =~ *i_d2d_link*async_phy_ready_i}]
set_false_path -hold -to [get_pins -quiet -hierarchical -filter {NAME =~ *i_d2d_link*async_phy_ready_i}]
set_max_delay $d2d_phy_async_period -from [get_pins -quiet -hierarchical -filter {NAME =~ *i_d2d_link*async_buffer_empty_o}]
set_false_path -hold -from [get_pins -quiet -hierarchical -filter {NAME =~ *i_d2d_link*async_buffer_empty_o}]

#### The t-pin oddrs are active one cycle before the data
set_multicycle_path -to [get_pins -hierarchical *oddre1_tpin/D*] 2


### D2D false paths
set_false_path -through [get_nets -quiet -hierarchical -filter {NAME =~ *i_d2d_link*false_path*}]
set_false_path -through [get_pins -quiet -hierarchical -filter {NAME =~ *i_d2d_link*false_path*}]
set_false_path -through [get_pins -quiet {hemaia_system_i/occamy_chip/inst/i_d2d_link/*.i_phy_interface*/*tx_strength_thermometer_i*}]
set_false_path -through [get_pins -quiet -hierarchical -filter {NAME =~ *i_d2d_link*i_rx_clk_delay_controller/i_final_clk_mux/sel}]
set_false_path -through [get_pins -quiet -hierarchical -filter {NAME =~ *i_d2d_link*link_available*}]
set_false_path -through [get_pins -quiet -hierarchical -filter {NAME =~ *i_d2d_link*multicast_available*}]
set_false_path -through [get_pins -quiet {hemaia_system_i/occamy_chip/inst/i_d2d_link/i_controller_reg_to_hw/reg2hw[availability_register][*_link_available][q]}]
set_false_path -through [get_pins -quiet {hemaia_system_i/occamy_chip/inst/i_d2d_link/i_controller_reg_to_hw/reg2hw[multicast_fence_register][*][q]}]
set_false_path -through [get_nets -quiet {hemaia_system_i/occamy_chip/inst/i_d2d_link/hw2reg[fec_unrecoverable_error_register][*][d][*]}]
set_false_path -through [get_nets -quiet {hemaia_system_i/occamy_chip/inst/i_d2d_link/i_controller_reg_to_hw/reg2hw[rx_buffer_threshold_register][*_rx_buffer_threshold][q][*]}]
set_false_path -through [get_nets -quiet {hemaia_system_i/occamy_chip/inst/i_d2d_link/i_controller_reg_to_hw/u_tx_backoff_period_register_*_tx_backoff_period/reg2hw[tx_backoff_period_register][*_tx_backoff_period][q][*]}]
set_false_path -through [get_nets -quiet {hemaia_system_i/occamy_chip/inst/i_d2d_link/reg2hw[tx_hold_period_register][*_tx_hold_period][q][*]}]
set_false_path -through [get_nets -quiet {hemaia_system_i/occamy_chip/inst/i_d2d_link/reg2hw[tx_yield_period_register][*_tx_yield_period][q][*]}]
set_false_path -through [get_nets -quiet {hemaia_system_i/occamy_chip/inst/i_d2d_link/reg2hw[tx_turnaround_silence_period_register][*_tx_turnaround_silence_period][q][*]}]
set_false_path -through [get_nets -quiet {hemaia_system_i/occamy_chip/inst/i_d2d_link/reg2hw[*_broken_wire_register][*][q][*]}]
set_false_path -through [get_nets -quiet {hemaia_system_i/occamy_chip/inst/i_d2d_link/reg2hw[programmable_drive_strength_register][*_tx_strength][q][*]}]
set_false_path -through [get_nets -quiet {hemaia_system_i/occamy_chip/inst/i_d2d_link/reg2hw[phy_clock_gating_delay_register][*][q][*]}]
set_false_path -through [get_nets -quiet {hemaia_system_i/occamy_chip/inst/i_d2d_link/reg2hw[phy_data_transmission_delay_register][*][q][*]}]
set_false_path -through [get_nets -quiet {hemaia_system_i/occamy_chip/inst/i_d2d_link/reg2hw[*_programmable_clock_delay_register][clock_delay_c*][q][*]}]
set_false_path -through [get_nets -quiet {hemaia_system_i/occamy_chip/inst/i_d2d_link/reg2hw[phy_ddr_mode_register][*][q]}]
set_false_path -through [get_nets -quiet {hemaia_system_i/occamy_chip/inst/i_d2d_link/reg2hw[test_mode_register][*_test_request][q]}]


################################################################################
# TIMING GROUPS
################################################################################

# Create timing groups through the FPU to help meet timing

# ingress and egress same for all pipe configs
group_path -default -through [get_pins -of [get_cells -hierarchical -filter {ORIG_REF_NAME == fpnew_sdotp_multi || REF_NAME == fpnew_sdotp_multi}] -filter { DIRECTION == "IN" && NAME !~ *out_ready_i && NAME !~ *rst_ni && NAME !~ *clk_i}]
group_path -name {sdotp_ingress} -through [get_pins -of [get_cells -hierarchical -filter {ORIG_REF_NAME == fpnew_sdotp_multi || REF_NAME == fpnew_sdotp_multi}] -filter { DIRECTION == "IN" && NAME !~ *out_ready_i && NAME !~ *rst_ni && NAME !~ *clk_i}]
group_path -default -through [get_pins -of [get_cells -hierarchical -filter {ORIG_REF_NAME == fpnew_fma_multi || REF_NAME == fpnew_fma_multi}] -filter { DIRECTION == "IN" && NAME !~ *out_ready_i && NAME !~ *rst_ni && NAME !~ *clk_i}]
group_path -name {fma_ingress} -through [get_pins -of [get_cells -hierarchical -filter {ORIG_REF_NAME == fpnew_fma_multi || REF_NAME == fpnew_fma_multi}] -filter { DIRECTION == "IN" && NAME !~ *out_ready_i && NAME !~ *rst_ni && NAME !~ *clk_i}]
group_path -default -through [get_pins -of [get_cells -hierarchical -filter {ORIG_REF_NAME == fpnew_sdotp_multi || REF_NAME == fpnew_sdotp_multi}] -filter { DIRECTION == "OUT" && NAME !~ *in_ready_o}]
group_path -name {sdotp_egress} -through [get_pins -of [get_cells -hierarchical -filter {ORIG_REF_NAME == fpnew_sdotp_multi || REF_NAME == fpnew_sdotp_multi}] -filter { DIRECTION == "OUT" && NAME !~ *in_ready_o}]
group_path -default -through [get_pins -of [get_cells -hierarchical -filter {ORIG_REF_NAME == fpnew_fma_multi || REF_NAME == fpnew_fma_multi}] -filter { DIRECTION == "OUT" && NAME !~ *in_ready_o}]
group_path -name {fma_egress} -through [get_pins -of [get_cells -hierarchical -filter {ORIG_REF_NAME == fpnew_fma_multi || REF_NAME == fpnew_fma_multi}] -filter { DIRECTION == "OUT" && NAME !~ *in_ready_o}]

# For 2 DISTRIBUTED pipe registers, registers are placed on input and mid
# The inside path therefore goes through the registers created in `gen_inside_pipeline[0]`

# The inside path groups
group_path -default -through [get_pins -filter {NAME =~ "*/D"} -of [get_cells -hier -filter { NAME =~  "*gen_inside_pipeline[0]*" && PARENT =~  "*fpnew_sdotp_multi*" }]]
group_path -name {sdotp_fu0} -through [get_pins -filter {NAME =~ "*/D"} -of [get_cells -hier -filter { NAME =~  "*gen_inside_pipeline[0]*" && PARENT =~  "*fpnew_sdotp_multi*" }]]
group_path -default -through [get_pins -filter {NAME =~ "*/D"} -of [get_cells -hier -filter { NAME =~  "*gen_inside_pipeline[0]*" && PARENT =~  "*fpnew_fma_multi*" }]]
group_path -name {fma_fu0} -through [get_pins -filter {NAME =~ "*/D"} -of [get_cells -hier -filter { NAME =~  "*gen_inside_pipeline[0]*" && PARENT =~  "*fpnew_fma_multi*" }]]

# Force bufg on out_clk of phy_digital mux
set_property CLOCK_BUFFER_TYPE BUFG [get_nets -hierarchical -filter {NAME =~ *i_final_clk_mux/out_clk*}]

################################################################################
# BIT_STREAM: Versal uses PS to configure PL. Still investigating how to configure... 
################################################################################
# set_property BITSTREAM.CONFIG.SPI_32BIT_ADDR Yes [current_design]
# set_property BITSTREAM.CONFIG.SPI_BUSWIDTH 4 [current_design]
# set_property BITSTREAM.CONFIG.SPI_FALL_EDGE Yes [current_design]
# set_property BITSTREAM.CONFIG.CONFIGRATE 127.5 [current_design]
# set_property CONFIG_VOLTAGE 1.8 [current_design]
# set_property CFGBVS GND [current_design]
