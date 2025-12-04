# Copyright 2024 KU Leuven and ETH Zurich.
# Solderpad Hardware License, Version 0.51, see LICENSE for details.
# SPDX-License-Identifier: SHL-0.51
#
# Nils Wistoff <nwistoff@iis.ee.ethz.ch>
# Yunhao Deng <yunhao.deng@kuleuven.be>

set iostd LVCMOS12
set d2d_io_port east_d2d_io_0
set d2d_fc_cts_i_port flow_control_east_cts_i_0
set d2d_fc_cts_o_port flow_control_east_cts_o_0
set d2d_fc_rts_i_port flow_control_east_rts_i_0
set d2d_fc_rts_o_port flow_control_east_rts_o_0
set d2d_test_req_port east_test_request_o_0
set d2d_test_brq_port east_test_being_requested_i_0

set_property -dict "PACKAGE_PIN BT37 IOSTANDARD $iostd SLEW FAST" [get_ports "$d2d_io_port[0]"]
set_property -dict "PACKAGE_PIN BU37 IOSTANDARD $iostd SLEW FAST" [get_ports "$d2d_io_port[1]"] 
set_property -dict "PACKAGE_PIN BT39 IOSTANDARD $iostd SLEW FAST" [get_ports "$d2d_io_port[2]"]
set_property -dict "PACKAGE_PIN BR39 IOSTANDARD $iostd SLEW FAST" [get_ports "$d2d_io_port[3]"]
set_property -dict "PACKAGE_PIN BR38 IOSTANDARD $iostd SLEW FAST" [get_ports "$d2d_io_port[4]"]
set_property -dict "PACKAGE_PIN BR37 IOSTANDARD $iostd SLEW FAST" [get_ports "$d2d_io_port[5]"]
set_property -dict "PACKAGE_PIN BT40 IOSTANDARD $iostd SLEW FAST" [get_ports "$d2d_io_port[6]"] 
set_property -dict "PACKAGE_PIN BR41 IOSTANDARD $iostd SLEW FAST" [get_ports "$d2d_io_port[7]"]
set_property -dict "PACKAGE_PIN BT41 IOSTANDARD $iostd SLEW FAST" [get_ports "$d2d_io_port[8]"]
set_property -dict "PACKAGE_PIN BR42 IOSTANDARD $iostd SLEW FAST" [get_ports "$d2d_io_port[9]"]
set_property -dict "PACKAGE_PIN BP40 IOSTANDARD $iostd SLEW FAST" [get_ports "$d2d_io_port[10]"]
set_property -dict "PACKAGE_PIN BN40 IOSTANDARD $iostd SLEW FAST" [get_ports "$d2d_io_port[11]"]
set_property -dict "PACKAGE_PIN BP38 IOSTANDARD $iostd SLEW FAST" [get_ports "$d2d_io_port[12]"]
set_property -dict "PACKAGE_PIN BN39 IOSTANDARD $iostd SLEW FAST" [get_ports "$d2d_io_port[13]"]
set_property -dict "PACKAGE_PIN CA39 IOSTANDARD $iostd SLEW FAST" [get_ports "$d2d_io_port[14]"]
set_property -dict "PACKAGE_PIN BY40 IOSTANDARD $iostd SLEW FAST" [get_ports "$d2d_io_port[15]"]
set_property -dict "PACKAGE_PIN CA41 IOSTANDARD $iostd SLEW FAST" [get_ports "$d2d_io_port[16]"]
set_property -dict "PACKAGE_PIN BY41 IOSTANDARD $iostd SLEW FAST" [get_ports "$d2d_io_port[17]"]
set_property -dict "PACKAGE_PIN BY39 IOSTANDARD $iostd SLEW FAST" [get_ports "$d2d_io_port[18]"]
set_property -dict "PACKAGE_PIN BW39 IOSTANDARD $iostd SLEW FAST" [get_ports "$d2d_io_port[19]"]
set_property -dict "PACKAGE_PIN CA37 IOSTANDARD $iostd SLEW FAST" [get_ports "$d2d_io_port[38]"]
set_property -dict "PACKAGE_PIN BY38 IOSTANDARD $iostd SLEW FAST" [get_ports "$d2d_io_port[21]"]
set_property -dict "PACKAGE_PIN BW41 IOSTANDARD $iostd SLEW FAST" [get_ports "$d2d_io_port[39]"]
set_property -dict "PACKAGE_PIN BV41 IOSTANDARD $iostd SLEW FAST" [get_ports "$d2d_io_port[23]"]
set_property -dict "PACKAGE_PIN BU42 IOSTANDARD $iostd SLEW FAST" [get_ports "$d2d_io_port[24]"]
set_property -dict "PACKAGE_PIN BU41 IOSTANDARD $iostd SLEW FAST" [get_ports "$d2d_io_port[25]"]
set_property -dict "PACKAGE_PIN CD43 IOSTANDARD $iostd SLEW FAST" [get_ports "$d2d_io_port[26]"]
set_property -dict "PACKAGE_PIN CD42 IOSTANDARD $iostd SLEW FAST" [get_ports "$d2d_io_port[27]"]
set_property -dict "PACKAGE_PIN CD41 IOSTANDARD $iostd SLEW FAST" [get_ports "$d2d_io_port[28]"]
set_property -dict "PACKAGE_PIN CD40 IOSTANDARD $iostd SLEW FAST" [get_ports "$d2d_io_port[29]"]
set_property -dict "PACKAGE_PIN CC42 IOSTANDARD $iostd SLEW FAST" [get_ports "$d2d_io_port[30]"]
set_property -dict "PACKAGE_PIN CB41 IOSTANDARD $iostd SLEW FAST" [get_ports "$d2d_io_port[31]"]
set_property -dict "PACKAGE_PIN CC40 IOSTANDARD $iostd SLEW FAST" [get_ports "$d2d_io_port[32]"]
set_property -dict "PACKAGE_PIN CB40 IOSTANDARD $iostd SLEW FAST" [get_ports "$d2d_io_port[33]"]
set_property -dict "PACKAGE_PIN CC39 IOSTANDARD $iostd SLEW FAST" [get_ports "$d2d_io_port[34]"]
set_property -dict "PACKAGE_PIN CC38 IOSTANDARD $iostd SLEW FAST" [get_ports "$d2d_io_port[35]"]
set_property -dict "PACKAGE_PIN CB39 IOSTANDARD $iostd SLEW FAST" [get_ports "$d2d_io_port[36]"]
set_property -dict "PACKAGE_PIN CA38 IOSTANDARD $iostd SLEW FAST" [get_ports "$d2d_io_port[37]"]
set_property -dict "PACKAGE_PIN BV43 IOSTANDARD $iostd SLEW FAST" [get_ports "$d2d_io_port[20]"]
set_property -dict "PACKAGE_PIN BW42 IOSTANDARD $iostd SLEW FAST" [get_ports "$d2d_io_port[22]"]
set_property -dict "PACKAGE_PIN CC47 IOSTANDARD $iostd SLEW FAST" [get_ports "$d2d_io_port[40]"]
set_property -dict "PACKAGE_PIN CB46 IOSTANDARD $iostd SLEW FAST" [get_ports "$d2d_io_port[41]"]
set_property -dict "PACKAGE_PIN CD46 IOSTANDARD $iostd SLEW FAST" [get_ports "$d2d_io_port[42]"]
set_property -dict "PACKAGE_PIN CC45 IOSTANDARD $iostd SLEW FAST" [get_ports "$d2d_io_port[43]"]
set_property -dict "PACKAGE_PIN CD45 IOSTANDARD $iostd SLEW FAST" [get_ports "$d2d_io_port[44]"]
set_property -dict "PACKAGE_PIN CC44 IOSTANDARD $iostd SLEW FAST" [get_ports "$d2d_io_port[45]"]
set_property -dict "PACKAGE_PIN CB44 IOSTANDARD $iostd SLEW FAST" [get_ports "$d2d_io_port[46]"]
set_property -dict "PACKAGE_PIN CC43 IOSTANDARD $iostd SLEW FAST" [get_ports "$d2d_io_port[47]"]
set_property -dict "PACKAGE_PIN CB45 IOSTANDARD $iostd SLEW FAST" [get_ports "$d2d_io_port[48]"]
set_property -dict "PACKAGE_PIN CA44 IOSTANDARD $iostd SLEW FAST" [get_ports "$d2d_io_port[49]"]
set_property -dict "PACKAGE_PIN BY44 IOSTANDARD $iostd SLEW FAST" [get_ports "$d2d_io_port[50]"]
set_property -dict "PACKAGE_PIN BY43 IOSTANDARD $iostd SLEW FAST" [get_ports "$d2d_io_port[51]"]
set_property -dict "PACKAGE_PIN CB52 IOSTANDARD $iostd SLEW FAST" [get_ports "$d2d_io_port[52]"]
set_property -dict "PACKAGE_PIN CA51 IOSTANDARD $iostd SLEW FAST" [get_ports "$d2d_io_port[53]"] 
set_property -dict "PACKAGE_PIN CA52 IOSTANDARD $iostd SLEW FAST" [get_ports "$d2d_io_port[54]"]
set_property -dict "PACKAGE_PIN BY51 IOSTANDARD $iostd SLEW FAST" [get_ports "$d2d_io_port[55]"] 
set_property -dict "PACKAGE_PIN BW52 IOSTANDARD $iostd SLEW FAST" [get_ports "$d2d_io_port[56]"]
set_property -dict "PACKAGE_PIN BW51 IOSTANDARD $iostd SLEW FAST" [get_ports "$d2d_io_port[57]"] 
set_property -dict "PACKAGE_PIN BY50 IOSTANDARD $iostd SLEW FAST" [get_ports "$d2d_io_port[58]"]
set_property -dict "PACKAGE_PIN BY49 IOSTANDARD $iostd SLEW FAST" [get_ports "$d2d_io_port[59]"] 

# Test Pins
set_property -dict "PACKAGE_PIN CD52 IOSTANDARD $iostd SLEW FAST" [get_ports "$d2d_test_brq_port"] 
set_property -dict "PACKAGE_PIN CD51 IOSTANDARD $iostd SLEW FAST" [get_ports "$d2d_test_req_port"] 

# Flow Control
set_property -dict "PACKAGE_PIN BW50 IOSTANDARD $iostd SLEW FAST" [get_ports "$d2d_fc_cts_i_port"] 
set_property -dict "PACKAGE_PIN BW49 IOSTANDARD $iostd SLEW FAST" [get_ports "$d2d_fc_cts_o_port"] 
set_property -dict "PACKAGE_PIN BV50 IOSTANDARD $iostd SLEW FAST" [get_ports "$d2d_fc_rts_i_port"] 
set_property -dict "PACKAGE_PIN BV49 IOSTANDARD $iostd SLEW FAST" [get_ports "$d2d_fc_rts_o_port"] 

# JTAG
set_property -dict "PACKAGE_PIN CC50 IOSTANDARD $iostd" [get_ports {jtag_tck_i}] 
set_property -dict "PACKAGE_PIN CB49 IOSTANDARD $iostd" [get_ports {jtag_tdi_i}] 
set_property -dict "PACKAGE_PIN CB50 IOSTANDARD $iostd" [get_ports {jtag_tdo_o}] 
set_property -dict "PACKAGE_PIN CA49 IOSTANDARD $iostd" [get_ports {jtag_tms_i}] 


# UART FC
set_property -dict "PACKAGE_PIN CC52 IOSTANDARD $iostd PULLTYPE PULLUP" [get_ports {uart_cts_ni}]
set_property -dict "PACKAGE_PIN CB51 IOSTANDARD $iostd" [get_ports {uart_rts_no}] 
set_property -dict "PACKAGE_PIN CD50 IOSTANDARD $iostd" [get_ports {uart_rx_i}] 
set_property -dict "PACKAGE_PIN CC49 IOSTANDARD $iostd" [get_ports {uart_tx_o}] 

#VREF
set_property -dict "PACKAGE_PIN CD48 IOSTANDARD $iostd DRIVE 8" [get_ports {vref_vdd_o[0]}] 
set_property -dict "PACKAGE_PIN CD47 IOSTANDARD $iostd DRIVE 8" [get_ports {vref_gnd_o[0]}] 

# Four-wires GPIO_O connected to LEDs
set_property PACKAGE_PIN BA49 [get_ports gpio_d_o[0]]
set_property IOSTANDARD LVCMOS15 [get_ports gpio_d_o[0]]
set_property PACKAGE_PIN AY50 [get_ports gpio_d_o[1]]
set_property IOSTANDARD LVCMOS15 [get_ports gpio_d_o[1]]
set_property PACKAGE_PIN BA48 [get_ports gpio_d_o[2]]
set_property IOSTANDARD LVCMOS15 [get_ports gpio_d_o[2]]
set_property PACKAGE_PIN AY49 [get_ports gpio_d_o[3]]
set_property IOSTANDARD LVCMOS15 [get_ports gpio_d_o[3]]


# CPU_RESET pushbutton switch
# SW4 is the reset button
set_false_path -from [get_ports reset] -to [all_registers]
set_property PACKAGE_PIN BT48 [get_ports reset]
set_property IOSTANDARD LVCMOS15 [get_ports reset]

# Set RTC as false path
set_false_path -to [get_pins hemaia_system_i/occamy_chip/inst/i_occamy/i_clint/i_sync_edge/i_sync/reg_q_reg[0]/D]

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

create_generated_clock \
    -name clk_core \
    -source [get_pins hemaia_system_i/occamy_chip/inst/clk_i] \
    -divide_by 6 \
    [get_pins hemaia_system_i/occamy_chip/inst/i_hemaia_clk_rst_controller/gen_clock_divider[0].i_clk_divider/clk_o]

create_generated_clock \
    -name clk_acc \
    -source [get_pins hemaia_system_i/occamy_chip/inst/clk_i] \
    -divide_by 6 \
    [get_pins hemaia_system_i/occamy_chip/inst/i_hemaia_clk_rst_controller/gen_clock_divider[1].i_clk_divider/clk_o]

set_clock_groups -asynchronous \
    -group {clk_pl_0 clk_main} \
    -group {clk_core} \
    -group {clk_acc} \
    -group {clk_pl_1} \
    -group {clk_pl_2 clk_peri} 

# CDC 2phase clearable of DM: i_cdc_resp/i_cdc_req
# CONSTRAINT: Requires max_delay of min_period(src_clk_i, dst_clk_i) through the paths async_req, async_ack, async_data.
set_max_delay -through [get_nets -hier -filter {NAME =~ "*i_cdc_resp/async_req*"}] 10.000
set_max_delay -through [get_nets -hier -filter {NAME =~ "*i_cdc_resp/async_ack*"}] 10.000
set_max_delay -through [get_nets -hier -filter {NAME =~ "*i_cdc_resp/async_data*"}] 10.000
set_max_delay -through [get_nets -hier -filter {NAME =~ "*i_cdc_req/async_req*"}] 10.000
set_max_delay -through [get_nets -hier -filter {NAME =~ "*i_cdc_req/async_ack*"}] 10.000
set_max_delay -through [get_nets -hier -filter {NAME =~ "*i_cdc_req/async_data*"}] 10.000

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
set rx_clk_1_io_port "$d2d_io_port[42]"
set rx_clk_2_io_port "$d2d_io_port[22]"
set rx_clk_3_io_port "$d2d_io_port[2]"
set d2d_phy_clk clk_main 
set d2d_clk_period [get_property PERIOD [get_clocks $d2d_phy_clk]]
set d2d_path hemaia_system_i/occamy_chip/inst/i_d2d_link
set phy_digital_channel1 "$d2d_path/gen_east_phy_enabled.gen_phy_channels[0].i_phy_interface_east/i_phy_digital_interface"
set phy_digital_channel2 "$d2d_path/gen_east_phy_enabled.gen_phy_channels[1].i_phy_interface_east/i_phy_digital_interface"
set phy_digital_channel3 "$d2d_path/gen_east_phy_enabled.gen_phy_channels[2].i_phy_interface_east/i_phy_digital_interface"

# create rx clocks
create_clock -period $d2d_clk_period -name rx_clk_1 [get_ports $rx_clk_1_io_port]
create_clock -period $d2d_clk_period -name rx_clk_2 [get_ports $rx_clk_2_io_port]
create_clock -period $d2d_clk_period -name rx_clk_3 [get_ports $rx_clk_3_io_port]


create_generated_clock -name rx_clk_s1_a_gen_1 \
    -source [get_ports $rx_clk_1_io_port] \
    -multiply_by 1 \
    -combinational \
    -add \
    -master_clock rx_clk_1 \
    [get_pins "$phy_digital_channel1/i_rx_clk_delay_controller/i_clk_mux_a/clk_o"]

create_generated_clock -name rx_clk_s1_a_gen_2 \
    -source [get_ports $rx_clk_2_io_port] \
    -multiply_by 1 \
    -combinational \
    -add \
    -master_clock rx_clk_2 \
    [get_pins "$phy_digital_channel2/i_rx_clk_delay_controller/i_clk_mux_a/clk_o"]

create_generated_clock -name rx_clk_s1_a_gen_3 \
    -source [get_ports $rx_clk_3_io_port] \
    -multiply_by 1 \
    -combinational \
    -add \
    -master_clock rx_clk_3 \
    [get_pins "$phy_digital_channel3/i_rx_clk_delay_controller/i_clk_mux_a/clk_o"]


create_generated_clock -name rx_clk_s1_b_gen_1 \
    -source [get_ports $rx_clk_1_io_port] \
    -multiply_by 1 \
    -combinational \
    -add \
    -master_clock rx_clk_1 \
    [get_pins "$phy_digital_channel1/i_rx_clk_delay_controller/i_clk_mux_b/clk_o"]

create_generated_clock -name rx_clk_s1_b_gen_2 \
    -source [get_ports $rx_clk_2_io_port] \
    -multiply_by 1 \
    -combinational \
    -add \
    -master_clock rx_clk_2 \
    [get_pins "$phy_digital_channel2/i_rx_clk_delay_controller/i_clk_mux_b/clk_o"]

create_generated_clock -name rx_clk_s1_b_gen_3 \
    -source [get_ports $rx_clk_3_io_port] \
    -multiply_by 1 \
    -combinational \
    -add \
    -master_clock rx_clk_3 \
    [get_pins "$phy_digital_channel3/i_rx_clk_delay_controller/i_clk_mux_b/clk_o"]

set_clock_groups -asynchronous \
    -group rx_clk_s1_a_gen_1 \
    -group rx_clk_s1_a_gen_2 \
    -group rx_clk_s1_a_gen_3 \
    -group rx_clk_s1_b_gen_1 \
    -group rx_clk_s1_b_gen_2 \
    -group rx_clk_s1_b_gen_3

################## Stage 2 Mux constraints
create_generated_clock -name rx_clk_s2_a_gen_1 \
    -source [get_pins "$phy_digital_channel1/i_rx_clk_delay_controller/i_final_clk_mux/clk0"] \
    -multiply_by 1 \
    -add \
    -combinational \
    -master_clock rx_clk_s1_a_gen_1 \
    [get_pins "$phy_digital_channel1/i_rx_clk_delay_controller/i_final_clk_mux/out_clk"]

create_generated_clock -name rx_clk_s2_a_gen_2 \
    -source [get_pins "$phy_digital_channel2/i_rx_clk_delay_controller/i_final_clk_mux/clk0"] \
    -multiply_by 1 \
    -add \
    -combinational \
    -master_clock rx_clk_s1_a_gen_2 \
    [get_pins "$phy_digital_channel2/i_rx_clk_delay_controller/i_final_clk_mux/out_clk"]

create_generated_clock -name rx_clk_s2_a_gen_3 \
    -source [get_pins "$phy_digital_channel3/i_rx_clk_delay_controller/i_final_clk_mux/clk0"] \
    -multiply_by 1 \
    -add \
    -combinational \
    -master_clock rx_clk_s1_a_gen_3 \
    [get_pins "$phy_digital_channel3/i_rx_clk_delay_controller/i_final_clk_mux/out_clk"]



create_generated_clock -name rx_clk_s2_b_gen_1 \
    -source [get_pins "$phy_digital_channel1/i_rx_clk_delay_controller/i_final_clk_mux/clk1"] \
    -multiply_by 1 \
    -add \
    -combinational \
    -master_clock rx_clk_s1_b_gen_1 \
    [get_pins "$phy_digital_channel1/i_rx_clk_delay_controller/i_final_clk_mux/out_clk"]

create_generated_clock -name rx_clk_s2_b_gen_2 \
    -source [get_pins "$phy_digital_channel2/i_rx_clk_delay_controller/i_final_clk_mux/clk1"] \
    -multiply_by 1 \
    -add \
    -combinational \
    -master_clock rx_clk_s1_b_gen_2 \
    [get_pins "$phy_digital_channel2/i_rx_clk_delay_controller/i_final_clk_mux/out_clk"]

create_generated_clock -name rx_clk_s2_b_gen_3 \
    -source [get_pins "$phy_digital_channel3/i_rx_clk_delay_controller/i_final_clk_mux/clk1"] \
    -multiply_by 1 \
    -add \
    -combinational \
    -master_clock rx_clk_s1_b_gen_3 \
    [get_pins "$phy_digital_channel3/i_rx_clk_delay_controller/i_final_clk_mux/out_clk"]


################### Groups 
set_clock_groups -logically_exclusive \
    -group rx_clk_s2_a_gen_1 \
    -group rx_clk_s2_b_gen_1 \
    -group rx_clk_s2_a_gen_2 \
    -group rx_clk_s2_b_gen_2 \
    -group rx_clk_s2_a_gen_3 \
    -group rx_clk_s2_b_gen_3


set_clock_groups -asynchronous \
    -group {rx_clk_1 rx_clk_s1_a_gen_1 rx_clk_s1_b_gen_1 rx_clk_s2_a_gen_1 rx_clk_s2_b_gen_1} \
    -group {rx_clk_2 rx_clk_s1_a_gen_2 rx_clk_s1_b_gen_2 rx_clk_s2_a_gen_2 rx_clk_s2_b_gen_2} \
    -group {rx_clk_3 rx_clk_s1_a_gen_3 rx_clk_s1_b_gen_3 rx_clk_s2_a_gen_3 rx_clk_s2_b_gen_3} \
    -group {clk_pl_0 clk_main} \
    -group {clk_core} \
    -group {clk_acc} \
    -group {clk_pl_1} \
    -group {clk_pl_2 clk_peri} 

#### The t-pin oddrs are active one cycle before the data
set_multicycle_path -to [get_pins -hierarchical *oddre1_tpin/D*] 2

set_false_path -through [get_pins hemaia_system_i/occamy_chip/inst/i_d2d_link/*.i_phy_interface*/*tx_strength_thermometer_i*]
set_false_path -through [get_pins {hemaia_system_i/occamy_chip/inst/i_d2d_link/i_controller_reg_to_hw/reg2hw[availability_register][*_link_available][q]}]
set_false_path -through [get_nets hemaia_system_i/occamy_chip/inst/i_d2d_link/**/*false_path*]
set_false_path -through [get_nets {hemaia_system_i/occamy_chip/inst/i_d2d_link/i_controller_reg_to_hw/reg2hw[rx_buffer_threshold_register][east_rx_buffer_threshold][q][*]}]
set_false_path -through [get_nets {hemaia_system_i/occamy_chip/inst/i_d2d_link/i_controller_reg_to_hw/u_tx_backoff_period_register_east_tx_backoff_period/reg2hw[tx_backoff_period_register][east_tx_backoff_period][q][*]}]
set_false_path -through [get_nets {hemaia_system_i/occamy_chip/inst/i_d2d_link/reg2hw[tx_hold_period_register][east_tx_hold_period][q][*]}]
set_false_path -through [get_nets {hemaia_system_i/occamy_chip/inst/i_d2d_link/hw2reg[fec_unrecoverable_error_register][*][d][*]}]
set_false_path -through [get_nets {hemaia_system_i/occamy_chip/inst/i_d2d_link/i_controller_reg_to_hw/reg2hw[rx_buffer_threshold_register][*_rx_buffer_threshold][q][*]}]
set_false_path -through [get_nets {hemaia_system_i/occamy_chip/inst/i_d2d_link/i_controller_reg_to_hw/u_tx_backoff_period_register_*_tx_backoff_period/reg2hw[tx_backoff_period_register][*_tx_backoff_period][q][*]}]
set_false_path -through [get_nets {hemaia_system_i/occamy_chip/inst/i_d2d_link/reg2hw[tx_hold_period_register][*_tx_hold_period][q][*]}]
set _xlnx_shared_i0 [get_nets {hemaia_system_i/occamy_chip/inst/i_d2d_link/gen_*_phy_enabled.gen_phy_channels[*].i_phy_interface_*/i_phy_analog_interface/*/tx_strength_thermometer_i[*]}]
set_false_path -through $_xlnx_shared_i0
set_false_path -through [get_nets {hemaia_system_i/occamy_chip/inst/i_d2d_link/reg2hw[phy_clock_gating_delay_register][*][q][*]}]
set_false_path -through [get_nets {hemaia_system_i/occamy_chip/inst/i_d2d_link/reg2hw[phy_data_transmission_delay_register][*][q][*]}]
set_false_path -through [get_nets {hemaia_system_i/occamy_chip/inst/i_d2d_link/reg2hw[*_programmable_clock_delay_register][clock_delay_c*][q][*]}]
set_false_path -through [get_nets {hemaia_system_i/occamy_chip/inst/i_d2d_link/reg2hw[phy_ddr_mode_register][*][q]}]


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

################################################################################
# BIT_STREAM: Versal uses PS to configure PL. Still investigating how to configure... 
################################################################################
# set_property BITSTREAM.CONFIG.SPI_32BIT_ADDR Yes [current_design]
# set_property BITSTREAM.CONFIG.SPI_BUSWIDTH 4 [current_design]
# set_property BITSTREAM.CONFIG.SPI_FALL_EDGE Yes [current_design]
# set_property BITSTREAM.CONFIG.CONFIGRATE 127.5 [current_design]
# set_property CONFIG_VOLTAGE 1.8 [current_design]
# set_property CFGBVS GND [current_design]
