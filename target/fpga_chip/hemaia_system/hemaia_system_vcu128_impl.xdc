# Copyright 2024 KU Leuven and ETH Zurich.
# Solderpad Hardware License, Version 0.51, see LICENSE for details.
# SPDX-License-Identifier: SHL-0.51
#
# Nils Wistoff <nwistoff@iis.ee.ethz.ch>
# Yunhao Deng <yunhao.deng@kuleuven.be>

# 1 and 0 voltage reference
# 1: LA20_P - A21
set_property PACKAGE_PIN A21 [get_ports vref_vdd_o]
set_property IOSTANDARD LVCMOS18 [get_ports vref_vdd_o]
set_property DRIVE 12 [get_ports vref_vdd_o]

# 0: LA20_N - A20
set_property PACKAGE_PIN A20 [get_ports vref_gnd_o]
set_property IOSTANDARD LVCMOS18 [get_ports vref_gnd_o]
set_property DRIVE 12 [get_ports vref_gnd_o]


# Four-wires UART with flow control
# LA26P - D17
set_property PACKAGE_PIN D17 [get_ports uart_rx_i]
set_property IOSTANDARD LVCMOS18 [get_ports uart_rx_i]
# LA26N - D16
set_property PACKAGE_PIN D16 [get_ports uart_tx_o]
set_property IOSTANDARD LVCMOS18 [get_ports uart_tx_o]
# Flow Control
# LA27P - E21
set_property PACKAGE_PIN E21 [get_ports uart_cts_ni]
set_property IOSTANDARD LVCMOS18 [get_ports uart_cts_ni]
set_property PULLUP TRUE [get_ports uart_cts_ni]
# LA27N - D21
set_property PACKAGE_PIN D21 [get_ports uart_rts_no]
set_property IOSTANDARD LVCMOS18 [get_ports uart_rts_no]

# Six-wires SPIx4 (Master)
# LA10_P - B23
set_property PACKAGE_PIN B23 [get_ports spim_sd_io[0]]
set_property IOSTANDARD LVCMOS18 [get_ports spim_sd_io[0]]
# LA10_N - A23
set_property PACKAGE_PIN A23 [get_ports spim_sd_io[1]]
set_property IOSTANDARD LVCMOS18 [get_ports spim_sd_io[1]]
# LA11_P - B26
set_property PACKAGE_PIN B26 [get_ports spim_sd_io[2]]
set_property IOSTANDARD LVCMOS18 [get_ports spim_sd_io[2]]
# LA11_N - B25
set_property PACKAGE_PIN B25 [get_ports spim_sd_io[3]]
set_property IOSTANDARD LVCMOS18 [get_ports spim_sd_io[3]]
# LA12_P - J22
set_property PACKAGE_PIN J22 [get_ports spim_csb_o[0]]
set_property IOSTANDARD LVCMOS18 [get_ports spim_csb_o[0]]
# LA12_N - H22
set_property PACKAGE_PIN H22 [get_ports spim_csb_o[1]]
set_property IOSTANDARD LVCMOS18 [get_ports spim_csb_o[1]]
# LA13_P - A25
set_property PACKAGE_PIN A25 [get_ports spim_sck_o]
set_property IOSTANDARD LVCMOS18 [get_ports spim_sck_o]

create_clock -period 10.000 -name spi_m_sck [get_ports spim_sck_o]

# Two-wires I2C
# LA14_P - C23
set_property PACKAGE_PIN C23 [get_ports i2c_sda_io]
set_property IOSTANDARD LVCMOS18 [get_ports i2c_sda_io]
# LA14_N - B22
set_property PACKAGE_PIN B22 [get_ports i2c_scl_io]
set_property IOSTANDARD LVCMOS18 [get_ports i2c_scl_io]

# Six-wires SPIx4 (Slave)
# LA15_N - H20
set_property PACKAGE_PIN H20 [get_ports spis_sd_io[0]]
set_property IOSTANDARD LVCMOS18 [get_ports spis_sd_io[0]]
# LA16_P - K24
set_property PACKAGE_PIN K24 [get_ports spis_sd_io[1]]
set_property IOSTANDARD LVCMOS18 [get_ports spis_sd_io[1]]
# LA16_N - K23
set_property PACKAGE_PIN K23 [get_ports spis_sd_io[2]]
set_property IOSTANDARD LVCMOS18 [get_ports spis_sd_io[2]]
# LA17_P - D20
set_property PACKAGE_PIN D20 [get_ports spis_sd_io[3]]
set_property IOSTANDARD LVCMOS18 [get_ports spis_sd_io[3]]
# LA17_N - D21
set_property PACKAGE_PIN D21 [get_ports spis_csb_i]
set_property IOSTANDARD LVCMOS18 [get_ports spim_csb_o]
# LA18_P - C22
set_property PACKAGE_PIN C22 [get_ports spis_sck_i]
set_property IOSTANDARD LVCMOS18 [get_ports spis_sck_i]

create_clock -period 10.000 -name spi_s_sck [get_ports spis_sck_i]

# Eight-wires GPIO_O connected to LEDs
set_property PACKAGE_PIN BH24 [get_ports gpio_d_o[0]]
set_property IOSTANDARD LVCMOS18 [get_ports gpio_d_o[0]]
set_property PACKAGE_PIN BG24 [get_ports gpio_d_o[1]]
set_property IOSTANDARD LVCMOS18 [get_ports gpio_d_o[1]]
set_property PACKAGE_PIN BG25 [get_ports gpio_d_o[2]]
set_property IOSTANDARD LVCMOS18 [get_ports gpio_d_o[2]]
set_property PACKAGE_PIN BF25 [get_ports gpio_d_o[3]]
set_property IOSTANDARD LVCMOS18 [get_ports gpio_d_o[3]]
set_property PACKAGE_PIN BF26 [get_ports gpio_d_o[4]]
set_property IOSTANDARD LVCMOS18 [get_ports gpio_d_o[4]]
set_property PACKAGE_PIN BF27 [get_ports gpio_d_o[5]]
set_property IOSTANDARD LVCMOS18 [get_ports gpio_d_o[5]]
set_property PACKAGE_PIN BG27 [get_ports gpio_d_o[6]]
set_property IOSTANDARD LVCMOS18 [get_ports gpio_d_o[6]]
set_property PACKAGE_PIN BG28 [get_ports gpio_d_o[7]]
set_property IOSTANDARD LVCMOS18 [get_ports gpio_d_o[7]]


# CPU_RESET pushbutton switch
set_false_path -from [get_ports reset] -to [all_registers]
set_property PACKAGE_PIN BM29 [get_ports reset]
set_property IOSTANDARD LVCMOS12 [get_ports reset]

# Set RTC as false path
set_false_path -to [get_pins hemaia_system_i/occamy_chip/inst/i_occamy/i_clint/i_sync_edge/i_sync/reg_q_reg[0]/D]

################################################################################
# Crossing Clock Domains
################################################################################

create_clock -period 20.0 -name clk_core [get_pins hemaia_system_i/clk_wiz/clk_core]
create_clock -period 62.5 -name clk_peri [get_pins hemaia_system_i/clk_wiz/clk_peri]

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
# BIT_STREAM
################################################################################
set_property BITSTREAM.CONFIG.SPI_32BIT_ADDR Yes [current_design]
set_property BITSTREAM.CONFIG.SPI_BUSWIDTH 4 [current_design]
set_property BITSTREAM.CONFIG.SPI_FALL_EDGE Yes [current_design]
set_property BITSTREAM.CONFIG.CONFIGRATE 127.5 [current_design]
set_property CONFIG_VOLTAGE 1.8 [current_design]
set_property CFGBVS GND [current_design]


