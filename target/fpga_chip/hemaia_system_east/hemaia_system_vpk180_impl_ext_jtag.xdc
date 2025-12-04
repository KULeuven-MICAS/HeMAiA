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


set_property LOC SLICE_X306Y0 [get_cells {hemaia_system_i/occamy_chip/inst/i_d2d_link/gen_east_phy_enabled.gen_phy_channels[1].i_phy_interface_east/i_phy_analog_interface/tx_en_delayed_reg_replica_42}]
create_pblock pblock_i_d2d_link
add_cells_to_pblock [get_pblocks pblock_i_d2d_link] [get_cells -quiet [list hemaia_system_i/occamy_chip/inst/i_d2d_link]]
resize_pblock [get_pblocks pblock_i_d2d_link] -add {SLICE_X164Y0:SLICE_X379Y43}
resize_pblock [get_pblocks pblock_i_d2d_link] -add {BLI_A_GRP0_X74Y0:BLI_A_GRP0_X211Y0}
resize_pblock [get_pblocks pblock_i_d2d_link] -add {BLI_A_GRP1_X74Y0:BLI_A_GRP1_X211Y0}
resize_pblock [get_pblocks pblock_i_d2d_link] -add {BLI_A_GRP2_X74Y0:BLI_A_GRP2_X211Y0}
resize_pblock [get_pblocks pblock_i_d2d_link] -add {BLI_B_GRP0_X74Y0:BLI_B_GRP0_X211Y0}
resize_pblock [get_pblocks pblock_i_d2d_link] -add {BLI_B_GRP1_X74Y0:BLI_B_GRP1_X211Y0}
resize_pblock [get_pblocks pblock_i_d2d_link] -add {BLI_B_GRP2_X74Y0:BLI_B_GRP2_X211Y0}
resize_pblock [get_pblocks pblock_i_d2d_link] -add {BLI_C_GRP0_X74Y0:BLI_C_GRP0_X211Y0}
resize_pblock [get_pblocks pblock_i_d2d_link] -add {BLI_C_GRP1_X74Y0:BLI_C_GRP1_X211Y0}
resize_pblock [get_pblocks pblock_i_d2d_link] -add {BLI_C_GRP2_X74Y0:BLI_C_GRP2_X211Y0}
resize_pblock [get_pblocks pblock_i_d2d_link] -add {BLI_D_GRP4_X74Y0:BLI_D_GRP4_X211Y0}
resize_pblock [get_pblocks pblock_i_d2d_link] -add {BLI_D_GRP5_X74Y0:BLI_D_GRP5_X211Y0}
resize_pblock [get_pblocks pblock_i_d2d_link] -add {BLI_D_GRP6_X74Y0:BLI_D_GRP6_X211Y0}
resize_pblock [get_pblocks pblock_i_d2d_link] -add {BLI_D_GRP7_X74Y0:BLI_D_GRP7_X211Y0}
resize_pblock [get_pblocks pblock_i_d2d_link] -add {BLI_TMR_X74Y0:BLI_TMR_X211Y0}
resize_pblock [get_pblocks pblock_i_d2d_link] -add {DSP58_CPLX_X4Y0:DSP58_CPLX_X11Y21}
resize_pblock [get_pblocks pblock_i_d2d_link] -add {DSP_X8Y0:DSP_X23Y21}
resize_pblock [get_pblocks pblock_i_d2d_link] -add {IRI_QUAD_X83Y0:IRI_QUAD_X230Y203}
resize_pblock [get_pblocks pblock_i_d2d_link] -add {NOC_NMU512_X2Y0:NOC_NMU512_X3Y0}
resize_pblock [get_pblocks pblock_i_d2d_link] -add {NOC_NPS_VNOC_X2Y0:NOC_NPS_VNOC_X3Y1}
resize_pblock [get_pblocks pblock_i_d2d_link] -add {NOC_NSU512_X2Y0:NOC_NSU512_X3Y0}
resize_pblock [get_pblocks pblock_i_d2d_link] -add {PCIE50_X1Y0:PCIE50_X1Y0}
resize_pblock [get_pblocks pblock_i_d2d_link] -add {RAMB18_X6Y23:RAMB18_X16Y0}
resize_pblock [get_pblocks pblock_i_d2d_link] -add {RAMB36_X6Y11:RAMB36_X16Y0}
resize_pblock [get_pblocks pblock_i_d2d_link] -add {URAM288_X4Y11:URAM288_X8Y0}
