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
foreach {d2d_io_idx d2d_io_pin} {
  0 CD51 1 CD52 2 BU41 3 BR42 4 BU42 5 BT41
  6 BN40 7 BW39 8 BP40 9 BY39 10 CB40 11 CD42
  12 CC40 13 CD43 14 CD40 15 CD41 16 CB41 17 CA38
  18 CC42 19 CC38 20 CB39 21 BT37 22 BY40 23 CC39
  24 BU37 25 CA39 26 BY38 27 BR37 28 CA37 29 BR38
  30 BR41 31 BR39 32 BT40 33 BT39 34 BV41 35 BY41
  36 BW41 37 CA41 38 BY43 39 BY44 40 BV49 41 CB46
  42 BV50 43 CC47 44 BW51 45 BW52 46 CD47 47 CA44
  48 CA49 49 CD48 50 CC43 51 CB45 52 CB50 53 CB44
  54 BY49 55 BY50 56 CB49 57 CC45 58 CC44 59 CC50
} {
  set_property -dict "PACKAGE_PIN $d2d_io_pin IOSTANDARD $iostd SLEW FAST" \
    [get_ports [format {%s[%d]} $d2d_io_port $d2d_io_idx]]
}

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

set hemaia_clk_rst_path {hemaia_system_i/occamy_chip/inst/i_hemaia_clk_rst_controller}
set clk_main_period [get_property PERIOD [get_clocks clk_main]]
set clk_peri_period [get_property PERIOD [get_clocks clk_peri]]
set clock_host_period [expr {$clk_main_period * 6.0}]
set clock_acc_period [expr {$clk_main_period * 8.0}]
set clock_d2d_phy_period [expr {$clk_main_period * 1.0}]

proc hemaia_set_cdc_max_delay {delay from_clocks to_clocks} {
  set from_objs [get_clocks -quiet $from_clocks]
  set to_objs [get_clocks -quiet $to_clocks]
  if {[llength $from_objs] != 0 && [llength $to_objs] != 0} {
    set_max_delay -datapath_only $delay -from $from_objs -to $to_objs
    set_false_path -hold -from $from_objs -to $to_objs
  }
}

proc hemaia_set_cdc_max_delay_pair {delay clocks_a clocks_b} {
  hemaia_set_cdc_max_delay $delay $clocks_a $clocks_b
  hemaia_set_cdc_max_delay $delay $clocks_b $clocks_a
}

proc hemaia_set_max_delay_through {delay objects} {
  if {[llength $objects] != 0} {
    set_max_delay $delay -through $objects
    set_false_path -hold -through $objects
  }
}

proc hemaia_set_max_delay_to {delay objects} {
  if {[llength $objects] != 0} {
    set_max_delay $delay -to $objects
    set_false_path -hold -to $objects
  }
}

proc hemaia_set_max_delay_from {delay objects} {
  if {[llength $objects] != 0} {
    set_max_delay $delay -from $objects
    set_false_path -hold -from $objects
  }
}

proc hemaia_set_false_path_through {objects} {
  if {[llength $objects] != 0} {
    set_false_path -through $objects
  }
}

# Model the programmable divider internals like the chip SDC. The even/odd
# divider candidates stay related to clk_main; each final channel clock is an
# independent output-domain clock at the divider output.
for {set i 0} {$i < 6} {incr i} {
  set div_path [format {%s/gen_clock_divider[%d].i_clk_divider} $hemaia_clk_rst_path $i]
  set even_pin [format {%s/i_clk_divided_mux/clk0_i} $div_path]
  set odd_pin [format {%s/i_clk_divided_mux/clk1_i} $div_path]

  create_generated_clock -name [format {clk_divider_%d_even} $i] \
    -source [get_pins hemaia_system_i/occamy_chip/inst/clk_i] \
    -divide_by 2 -master_clock clk_main \
    [get_pins $even_pin]

  create_generated_clock -name [format {clk_divider_%d_odd} $i] \
    -source [get_pins hemaia_system_i/occamy_chip/inst/clk_i] \
    -divide_by 3 -master_clock clk_main \
    [get_pins $odd_pin]

  set_clock_groups -logically_exclusive \
    -group [get_clocks [format {clk_divider_%d_even} $i]] \
    -group [get_clocks [format {clk_divider_%d_odd} $i]]
}

create_clock -name clk_host_gen -period $clock_host_period \
  [get_pins [format {%s/gen_clock_divider[0].i_clk_divider/clk_o} $hemaia_clk_rst_path]]

create_clock -name clk_acc_gen -period $clock_acc_period \
  [get_pins [format {%s/gen_clock_divider[1].i_clk_divider/clk_o} $hemaia_clk_rst_path]]

create_clock -name clk_d2d_phy_east_gen -period $clock_d2d_phy_period \
  [get_pins [format {%s/gen_clock_divider[2].i_clk_divider/clk_o} $hemaia_clk_rst_path]]

create_clock -name clk_d2d_phy_west_gen -period $clock_d2d_phy_period \
  [get_pins [format {%s/gen_clock_divider[3].i_clk_divider/clk_o} $hemaia_clk_rst_path]]

create_clock -name clk_d2d_phy_north_gen -period $clock_d2d_phy_period \
  [get_pins [format {%s/gen_clock_divider[4].i_clk_divider/clk_o} $hemaia_clk_rst_path]]

create_clock -name clk_d2d_phy_south_gen -period $clock_d2d_phy_period \
  [get_pins [format {%s/gen_clock_divider[5].i_clk_divider/clk_o} $hemaia_clk_rst_path]]

# Vivado does not support the Synopsys -allow_paths clock-group style used in
# the chip SDC. Use explicit CDC max-delay plus hold-only false paths instead.
if {[llength [get_clocks -quiet clk_pl_1]] != 0} {
  set_clock_groups -asynchronous -group [get_clocks clk_pl_1]
}

hemaia_set_cdc_max_delay_pair $clock_host_period clk_host_gen clk_peri
hemaia_set_cdc_max_delay_pair $clk_main_period clk_main clk_peri
hemaia_set_cdc_max_delay_pair $clock_host_period clk_host_gen clk_acc_gen

foreach d2d_clk {
  clk_d2d_phy_east_gen
  clk_d2d_phy_west_gen
  clk_d2d_phy_north_gen
  clk_d2d_phy_south_gen
} {
  hemaia_set_cdc_max_delay $clock_d2d_phy_period {clk_host_gen clk_peri} $d2d_clk
  hemaia_set_cdc_max_delay $clock_d2d_phy_period $d2d_clk {clk_host_gen clk_peri}
}

# CDC 2phase clearable of DM: i_cdc_resp/i_cdc_req
# CONSTRAINT: Requires max_delay of min_period(src_clk_i, dst_clk_i) through the paths async_req, async_ack, async_data.
foreach cdc_net_pattern {
  *i_cdc_resp/async_req*
  *i_cdc_resp/async_ack*
  *i_cdc_resp/async_data*
  *i_cdc_req/async_req*
  *i_cdc_req/async_ack*
  *i_cdc_req/async_data*
} {
  hemaia_set_max_delay_through 10.000 \
    [get_nets -quiet -hier -filter "NAME =~ \"$cdc_net_pattern\""]
}

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
set d2d_path hemaia_system_i/occamy_chip/inst/i_d2d_link
set rx_clk_1_io_port [format {%s[%d]} $d2d_io_port 0]
set rx_clk_2_io_port [format {%s[%d]} $d2d_io_port 20]
set rx_clk_3_io_port [format {%s[%d]} $d2d_io_port 40]
set d2d_phy_clk clk_d2d_phy_west_gen
set d2d_clk_period [get_property PERIOD [get_clocks $d2d_phy_clk]]
set d2d_rx_mux_cdc_period [expr {0.5 * $d2d_clk_period}]
set d2d_phy_async_period [expr {0.5 * $d2d_clk_period}]
set d2d_tx_clk_pin [format {%s/gen_clock_divider[3].i_clk_divider/clk_o} $hemaia_clk_rst_path]
set phy_digital_channel1 [format {%s/gen_west_phy_enabled.gen_phy_channels[0].i_phy_interface_west/i_phy_digital_interface} $d2d_path]
set phy_digital_channel2 [format {%s/gen_west_phy_enabled.gen_phy_channels[1].i_phy_interface_west/i_phy_digital_interface} $d2d_path]
set phy_digital_channel3 [format {%s/gen_west_phy_enabled.gen_phy_channels[2].i_phy_interface_west/i_phy_digital_interface} $d2d_path]

proc hemaia_create_d2d_tx_clock {idx phy_path src_pin master_clock} {
  set tx_clock [format {tx_clk_%d} $idx]
  create_generated_clock -name $tx_clock \
    -source [get_pins $src_pin] \
    -divide_by 1 -master_clock $master_clock \
    [get_pins [format {%s/clk_phy_tx_i} $phy_path]]
}

proc hemaia_create_d2d_rx_clocks {idx rx_io_port phy_path period} {
  set rx_port_clock [format {rx_clk_%d} $idx]
  set rx_sel0_clock [format {rx_clk_%d_sel0} $idx]
  set rx_sel1_clock [format {rx_clk_%d_sel1} $idx]
  set rx_out_clock [format {rx_clk_%d_gen} $idx]

  create_clock -period $period -name $rx_port_clock [get_ports $rx_io_port]

  create_generated_clock -name $rx_sel0_clock \
    -source [get_ports $rx_io_port] \
    -combinational \
    -master_clock $rx_port_clock \
    [get_pins [format {%s/i_rx_clk_delay_controller/i_final_clk_mux/clk0} $phy_path]]

  create_generated_clock -name $rx_sel1_clock \
    -source [get_ports $rx_io_port] \
    -combinational \
    -master_clock $rx_port_clock \
    [get_pins [format {%s/i_rx_clk_delay_controller/i_final_clk_mux/clk1} $phy_path]]

  create_clock -period $period -name $rx_out_clock \
    [get_pins [format {%s/i_rx_clk_delay_controller/data_clk_o} $phy_path]]
}

hemaia_create_d2d_tx_clock 1 $phy_digital_channel1 $d2d_tx_clk_pin $d2d_phy_clk
hemaia_create_d2d_tx_clock 2 $phy_digital_channel2 $d2d_tx_clk_pin $d2d_phy_clk
hemaia_create_d2d_tx_clock 3 $phy_digital_channel3 $d2d_tx_clk_pin $d2d_phy_clk

hemaia_create_d2d_rx_clocks 1 $rx_clk_1_io_port $phy_digital_channel1 $d2d_clk_period
hemaia_create_d2d_rx_clocks 2 $rx_clk_2_io_port $phy_digital_channel2 $d2d_clk_period
hemaia_create_d2d_rx_clocks 3 $rx_clk_3_io_port $phy_digital_channel3 $d2d_clk_period

foreach idx {1 2 3} {
  hemaia_set_cdc_max_delay_pair $d2d_clk_period {clk_host_gen clk_peri} [format {tx_clk_%d} $idx]
  hemaia_set_cdc_max_delay_pair $d2d_clk_period {clk_host_gen clk_peri} [format {rx_clk_%d_gen} $idx]
  hemaia_set_cdc_max_delay_pair $d2d_rx_mux_cdc_period \
    [format {rx_clk_%d_sel0} $idx] [format {rx_clk_%d_sel1} $idx]
}

foreach async_in_filter {
  {NAME =~ *i_d2d_link*async_tx_mode_i}
  {NAME =~ *i_d2d_link*async_phy_ready_i}
} {
  hemaia_set_max_delay_to $d2d_phy_async_period \
    [get_pins -quiet -hierarchical -filter $async_in_filter]
}

hemaia_set_max_delay_from $d2d_phy_async_period \
  [get_pins -quiet -hierarchical -filter {NAME =~ *i_d2d_link*async_buffer_empty_o}]

#### The t-pin oddrs are active one cycle before the data
set_multicycle_path -to [get_pins -hierarchical *oddre1_tpin/D*] 2


### D2D false paths
hemaia_set_false_path_through [get_nets -quiet -hierarchical -filter {NAME =~ *i_d2d_link*false_path*}]
hemaia_set_false_path_through [get_pins -quiet -hierarchical -filter {NAME =~ *i_d2d_link*false_path*}]
hemaia_set_false_path_through [get_pins -quiet {hemaia_system_i/occamy_chip/inst/i_d2d_link/*.i_phy_interface*/*tx_strength_thermometer_i*}]
hemaia_set_false_path_through [get_pins -quiet -hierarchical -filter {NAME =~ *i_d2d_link*i_rx_clk_delay_controller/i_final_clk_mux/sel}]
hemaia_set_false_path_through [get_pins -quiet -hierarchical -filter {NAME =~ *i_d2d_link*link_available*}]
hemaia_set_false_path_through [get_pins -quiet -hierarchical -filter {NAME =~ *i_d2d_link*multicast_available*}]

foreach fp_pin_pattern {
  {hemaia_system_i/occamy_chip/inst/i_d2d_link/i_controller_reg_to_hw/reg2hw[availability_register][*_link_available][q]}
  {hemaia_system_i/occamy_chip/inst/i_d2d_link/i_controller_reg_to_hw/reg2hw[multicast_fence_register][*][q]}
} {
  hemaia_set_false_path_through [get_pins -quiet $fp_pin_pattern]
}

foreach fp_net_pattern {
  {hemaia_system_i/occamy_chip/inst/i_d2d_link/hw2reg[fec_unrecoverable_error_register][*][d][*]}
  {hemaia_system_i/occamy_chip/inst/i_d2d_link/i_controller_reg_to_hw/reg2hw[rx_buffer_threshold_register][*_rx_buffer_threshold][q][*]}
  {hemaia_system_i/occamy_chip/inst/i_d2d_link/i_controller_reg_to_hw/u_tx_backoff_period_register_*_tx_backoff_period/reg2hw[tx_backoff_period_register][*_tx_backoff_period][q][*]}
  {hemaia_system_i/occamy_chip/inst/i_d2d_link/reg2hw[tx_hold_period_register][*_tx_hold_period][q][*]}
  {hemaia_system_i/occamy_chip/inst/i_d2d_link/reg2hw[tx_yield_period_register][*_tx_yield_period][q][*]}
  {hemaia_system_i/occamy_chip/inst/i_d2d_link/reg2hw[tx_turnaround_silence_period_register][*_tx_turnaround_silence_period][q][*]}
  {hemaia_system_i/occamy_chip/inst/i_d2d_link/reg2hw[*_broken_wire_register][*][q][*]}
  {hemaia_system_i/occamy_chip/inst/i_d2d_link/reg2hw[programmable_drive_strength_register][*_tx_strength][q][*]}
  {hemaia_system_i/occamy_chip/inst/i_d2d_link/reg2hw[phy_clock_gating_delay_register][*][q][*]}
  {hemaia_system_i/occamy_chip/inst/i_d2d_link/reg2hw[phy_data_transmission_delay_register][*][q][*]}
  {hemaia_system_i/occamy_chip/inst/i_d2d_link/reg2hw[*_programmable_clock_delay_register][clock_delay_c*][q][*]}
  {hemaia_system_i/occamy_chip/inst/i_d2d_link/reg2hw[phy_ddr_mode_register][*][q]}
  {hemaia_system_i/occamy_chip/inst/i_d2d_link/reg2hw[test_mode_register][*_test_request][q]}
} {
  hemaia_set_false_path_through [get_nets -quiet $fp_net_pattern]
}


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
