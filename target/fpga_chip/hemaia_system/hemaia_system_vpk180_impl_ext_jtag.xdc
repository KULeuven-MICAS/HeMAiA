# Copyright 2024 KU Leuven, ETH Zurich and University of Bologna.
# Solderpad Hardware License, Version 0.51, see LICENSE for details.
# SPDX-License-Identifier: SHL-0.51
#
# Cyril Koenig <cykoenig@iis.ee.ethz.ch>

# This constraint file is written for VCU128 + FMC XM105 Debug Card and is included only when EXT_JTAG = 1

# 5 MHz max JTAG
create_clock -period 200 -name jtag_tck_i [get_pins hemaia_system_i/jtag_tck_i]
set_property CLOCK_DEDICATED_ROUTE FALSE [get_nets -of [get_pins jtag_tck_i_IBUF_inst/O]]
set_property CLOCK_BUFFER_TYPE NONE [get_nets -of [get_pins jtag_tck_i_IBUF_inst/O]]
set_input_jitter jtag_tck_i 1.000

# JTAG clock is asynchronous with every other clocks.
set_clock_groups -asynchronous -group [get_clocks jtag_tck_i]

# Minimize routing delay
set_input_delay  -clock jtag_tck_i -clock_fall 5 [get_ports jtag_tdi_i]
set_input_delay  -clock jtag_tck_i -clock_fall 5 [get_ports jtag_tms_i]
set_output_delay -clock jtag_tck_i             5 [get_ports jtag_tdo_o]

set_max_delay -to   [get_ports { jtag_tdo_o }] 20
set_max_delay -from [get_ports { jtag_tms_i }] 20
set_max_delay -from [get_ports { jtag_tdi_i }] 20

# 2AD0 - J1.1 - LA00_CC_P - BV49
set_property PACKAGE_PIN BV49    [get_ports jtag_tck_i]
set_property IOSTANDARD LVCMOS15 [get_ports jtag_tck_i]
# 2AD1 - J1.3 - LA00_CC_N - BV50
set_property PACKAGE_PIN BV50    [get_ports jtag_tdi_i]
set_property IOSTANDARD LVCMOS15 [get_ports jtag_tdi_i]
# 2AD2 - J1.5 - LA01_CC_P - BW51
set_property PACKAGE_PIN BW51    [get_ports jtag_tdo_o]
set_property IOSTANDARD LVCMOS15 [get_ports jtag_tdo_o]
# 2AD3 - J1.7 - LA01_CC_N - BW52
set_property PACKAGE_PIN BW52    [get_ports jtag_tms_i]
set_property IOSTANDARD LVCMOS15 [get_ports jtag_tms_i]
