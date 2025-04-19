# Copyright 2024 KU Leuven and ETH Zurich.
# Solderpad Hardware License, Version 0.51, see LICENSE for details.
# SPDX-License-Identifier: SHL-0.51
#
# Nils Wistoff <nwistoff@iis.ee.ethz.ch>
# Yunhao Deng <yunhao.deng@kuleuven.be>

# 1 and 0 voltage reference
# 1: J20.3 - LA20_N - BT21
set_property PACKAGE_PIN BT41 [get_ports vref_vdd_o]
set_property IOSTANDARD LVCMOS15 [get_ports vref_vdd_o]
set_property DRIVE 12 [get_ports vref_vdd_o]

# 0: J20.1 - LA20_P - BR42
set_property PACKAGE_PIN BR42 [get_ports vref_gnd_o]
set_property IOSTANDARD LVCMOS15 [get_ports vref_gnd_o]
set_property DRIVE 12 [get_ports vref_gnd_o]


# Four-wires UART with flow control
# 2CD0 - J1.10 - LA12_P - BW49
set_property PACKAGE_PIN BW49 [get_ports uart_rx_i]
set_property IOSTANDARD LVCMOS15 [get_ports uart_rx_i]
# 2CD1 - J1.12 - LA12_N - BW50
set_property PACKAGE_PIN BW50 [get_ports uart_tx_o]
set_property IOSTANDARD LVCMOS15 [get_ports uart_tx_o]
# Flow Control
# 2CD2 - J1.14 - LA13_P - CC49
set_property PACKAGE_PIN CC49 [get_ports uart_cts_ni]
set_property IOSTANDARD LVCMOS15 [get_ports uart_cts_ni]
set_property PULLUP TRUE [get_ports uart_cts_ni]
# 2CD3 - J1.16 - LA13_N - CD50
set_property PACKAGE_PIN CD50 [get_ports uart_rts_no]
set_property IOSTANDARD LVCMOS15 [get_ports uart_rts_no]

# Six-wires SPIx4 (Slave)
# J1.39 - LA09_N - CD46
set_property PACKAGE_PIN CD46 [get_ports spis_sd_io[0]]
set_property IOSTANDARD LVCMOS15 [get_ports spis_sd_io[0]]
# J1.37 - LA09_P - CC45
set_property PACKAGE_PIN CC45 [get_ports spis_sd_io[1]]
set_property IOSTANDARD LVCMOS15 [get_ports spis_sd_io[1]]
# J1.35 - LA08_N - BY50
set_property PACKAGE_PIN BY50 [get_ports spis_sd_io[2]]
set_property IOSTANDARD LVCMOS15 [get_ports spis_sd_io[2]]
# J1.33 - LA08_P - BY49
set_property PACKAGE_PIN BY49 [get_ports spis_sd_io[3]]
set_property IOSTANDARD LVCMOS15 [get_ports spis_sd_io[3]]
# J1.31 - LA07_N - CC50
set_property PACKAGE_PIN CC50 [get_ports spis_csb_i]
set_property IOSTANDARD LVCMOS15 [get_ports spis_csb_i]
# J1.29 - LA07_P - CB49
set_property PACKAGE_PIN CB49 [get_ports spis_sck_i]
set_property IOSTANDARD LVCMOS15 [get_ports spis_sck_i]

create_clock -period 50.0 -name spi_s_sck [get_ports spis_sck_i]
set_property CLOCK_DEDICATED_ROUTE FALSE [get_nets spis_sck_i_IBUF]
set_property CLOCK_DEDICATED_ROUTE ANY_CMT_REGION [get_nets spis_sck_i_IBUF_BUFGCE]

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
# Crossing Clock Domains
################################################################################

create_clock -period 20.0 -name clk_core [get_pins hemaia_system_i/versal_cips_0/pl0_ref_clk]
create_clock -period 62.5 -name clk_peri [get_pins hemaia_system_i/versal_cips_0/pl2_ref_clk]

set_clock_groups -asynchronous -group [get_clocks -of [get_pins hemaia_system_i/occamy_chip/clk_i]] -group [get_clocks -of [get_pins hemaia_system_i/occamy_chip/clk_periph_i]]

# CDC 2phase clearable of DM: i_cdc_resp/i_cdc_req
# CONSTRAINT: Requires max_delay of min_period(src_clk_i, dst_clk_i) through the paths async_req, async_ack, async_data.
set_max_delay -through [get_nets -hier -filter {NAME =~ "*i_cdc_resp/async_req*"}] 10.000
set_max_delay -through [get_nets -hier -filter {NAME =~ "*i_cdc_resp/async_ack*"}] 10.000
set_max_delay -through [get_nets -hier -filter {NAME =~ "*i_cdc_resp/async_data*"}] 10.000
set_max_delay -through [get_nets -hier -filter {NAME =~ "*i_cdc_req/async_req*"}] 10.000
set_max_delay -through [get_nets -hier -filter {NAME =~ "*i_cdc_req/async_ack*"}] 10.000
set_max_delay -through [get_nets -hier -filter {NAME =~ "*i_cdc_req/async_data*"}] 10.000

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
