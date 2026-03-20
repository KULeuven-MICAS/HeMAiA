// Copyright 2026 KU Leuven.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51
// Yunhao Deng <yunhao.deng@kuleuven.be>
// Fanchen Kong <fanchen.kong@kuleuven.be>
// This module is the wrapper for the hemaia_io.
// The inputs are the wires from the hemaia rtl
// and the outputs are the pad wires
module hemaia_io_pad (
    // Hemaia RTL signals
    // CLKS
    inout wire              io_clk_i,
    inout wire              io_rst_ni,
    inout wire              io_clk_obs_o,
    inout wire              io_clk_periph_i,
    inout wire              io_rst_periph_ni,
    // PLL signal
    inout wire              io_pll_bypass_i,
    inout wire              io_pll_en_i,
    inout wire [1:0]        io_pll_post_div_sel_i,
    inout wire              io_pll_lock_o,
    // Aux signals
    inout wire              io_test_mode_i,
    inout wire [ 7:0]       io_chip_id_i,
    inout wire              io_boot_mode_i,
    // East side (66)
    inout wire              io_east_test_being_requested_i,
    inout wire              io_east_test_request_o,
    inout wire              io_flow_control_east_rts_o,
    inout wire              io_flow_control_east_cts_i,
    inout wire              io_flow_control_east_rts_i,
    inout wire              io_flow_control_east_cts_o,
    inout wire [2:0][19:0]  io_east_d2d,
    // West side (66)
    inout wire              io_west_test_being_requested_i,
    inout wire              io_west_test_request_o,
    inout wire              io_flow_control_west_rts_o,
    inout wire              io_flow_control_west_cts_i,
    inout wire              io_flow_control_west_rts_i,
    inout wire              io_flow_control_west_cts_o,
    inout wire [2:0][19:0]  io_west_d2d,
    // North side (66)
    inout wire              io_north_test_being_requested_i,
    inout wire              io_north_test_request_o,
    inout wire              io_flow_control_north_rts_o,
    inout wire              io_flow_control_north_cts_i,
    inout wire              io_flow_control_north_rts_i,
    inout wire              io_flow_control_north_cts_o,
    inout wire [2:0][19:0]  io_north_d2d,
    // South side (66)
    inout wire              io_south_test_being_requested_i,
    inout wire              io_south_test_request_o,
    inout wire              io_flow_control_south_rts_o,
    inout wire              io_flow_control_south_cts_i,
    inout wire              io_flow_control_south_rts_i,
    inout wire              io_flow_control_south_cts_o,
    inout wire [2:0][19:0]  io_south_d2d,
    // `uart` Interface (4)
    inout wire              io_uart_tx_o,
    inout wire              io_uart_rx_i,
    inout wire              io_uart_rts_no,
    inout wire              io_uart_cts_ni,
    // `gpio` Interface (4)
    inout wire [ 3:0]       io_gpio,
    // SPI Master Interface (6)
    inout wire              io_spim_sck_o,
    inout wire              io_spim_csb_o,
    inout wire [ 3:0]       io_spim_sd,
    // I2C Interface (2)
    inout wire              io_i2c_sda,
    inout wire              io_i2c_scl,
    // `jtag` Interface (5)
    inout wire              io_jtag_trst_ni,
    inout wire              io_jtag_tck_i,
    inout wire              io_jtag_tms_i,
    inout wire              io_jtag_tdi_i,
    inout wire              io_jtag_tdo_o,

    // Actual pad wires
    inout wire              CRBL_X1_Y4,
    inout wire              CRBL_X1_Y3,
    inout wire              CRBL_X2_Y3,
    inout wire              CRBL_X2_Y2,
    inout wire              CRBL_X1_Y2,
    inout wire              CRBL_X3_Y3,
    inout wire              CRBL_X4_Y4,
    inout wire              CRBL_X1_Y1,
    inout wire              CRBL_X2_Y4,
    inout wire              CRBL_X4_Y2,
    inout wire              CRBL_X5_Y2,
    inout wire              CRBL_X2_Y1,
    inout wire              CRBL_X3_Y1,
    inout wire              CRBL_X3_Y2,
    inout wire              CRBL_X4_Y1,
    inout wire              CRBL_X5_Y1,
    inout wire              CRBL_X0_Y0,
    inout wire              CRBL_X0_Y4,
    inout wire              CRBL_X0_Y1,
    inout wire              CRBL_X0_Y3,
    inout wire              CRBL_X3_Y0,
    inout wire              CRBL_X2_Y0,
    inout wire              CRBL_X1_Y0,
    inout wire              CRBL_X5_Y0,
    inout wire              CRBL_X4_Y0,
    inout wire              CRBL_X0_Y2,
    inout wire              CRBL_X3_Y4,
    inout wire              CRBL_X4_Y3,
    inout wire              CRBL_X5_Y4,
    inout wire              CRBL_X5_Y3,
    inout wire              CRBR_X3_Y2,
    inout wire              CRBR_X3_Y3,
    inout wire              CRBR_X3_Y4,
    inout wire              CRBR_X3_Y5,
    inout wire              CRBR_X2_Y3,
    inout wire              CRBR_X1_Y4,
    inout wire              CRBR_X0_Y3,
    inout wire              CRBR_X2_Y5,
    inout wire              CRBR_X2_Y4,
    inout wire              CRBR_X0_Y2,
    inout wire              CRBR_X2_Y1,
    inout wire              CRBR_X1_Y2,
    inout wire              CRBR_X1_Y1,
    inout wire              CRBR_X2_Y2,
    inout wire              CRBR_X0_Y1,
    inout wire              CRBR_X3_Y1,
    inout wire              CRBR_X1_Y5,
    inout wire              CRBR_X4_Y0,
    inout wire              CRBR_X0_Y0,
    inout wire              CRBR_X3_Y0,
    inout wire              CRBR_X1_Y0,
    inout wire              CRBR_X4_Y3,
    inout wire              CRBR_X4_Y2,
    inout wire              CRBR_X4_Y1,
    inout wire              CRBR_X4_Y5,
    inout wire              CRBR_X4_Y4,
    inout wire              CRBR_X2_Y0,
    inout wire              CRBR_X0_Y5,
    inout wire              CRBR_X0_Y4,
    inout wire              CRBR_X1_Y3,
    inout wire              CRTL_X4_Y4,
    inout wire              CRTL_X3_Y4,
    inout wire              CRTL_X2_Y0,
    inout wire              CRTL_X2_Y1,
    inout wire              CRTL_X1_Y4,
    inout wire              CRTL_X1_Y3,
    inout wire              CRTL_X1_Y1,
    inout wire              CRTL_X1_Y2,
    inout wire              CRTL_X1_Y0,
    inout wire              CRTL_X2_Y4,
    inout wire              CRTL_X2_Y3,
    inout wire              CRTL_X2_Y2,
    inout wire              CRTL_X3_Y3,
    inout wire              CRTL_X4_Y3,
    inout wire              CRTL_X3_Y2,
    inout wire              CRTL_X4_Y2,
    inout wire              CRTL_X4_Y1,
    inout wire              CRTL_X3_Y1,
    inout wire              CRTL_X4_Y5,
    inout wire              CRTL_X3_Y5,
    inout wire              CRTL_X2_Y5,
    inout wire              CRTL_X1_Y5,
    inout wire              CRTL_X0_Y5,
    inout wire              CRTL_X0_Y4,
    inout wire              CRTL_X0_Y3,
    inout wire              CRTL_X0_Y2,
    inout wire              CRTL_X0_Y1,
    inout wire              CRTL_X0_Y0,
    inout wire              CRTL_X3_Y0,
    inout wire              CRTL_X4_Y0,
    inout wire              CRTR_X2_Y3,
    inout wire              CRTR_X1_Y3,
    inout wire              CRTR_X0_Y3,
    inout wire              CRTR_X1_Y2,
    inout wire              CRTR_X2_Y2,
    inout wire              CRTR_X3_Y2,
    inout wire              CRTR_X0_Y1,
    inout wire              CRTR_X1_Y1,
    inout wire              CRTR_X3_Y0,
    inout wire              CRTR_X3_Y1,
    inout wire              CRTR_X0_Y2,
    inout wire              CRTR_X4_Y0,
    inout wire              CRTR_X4_Y1,
    inout wire              CRTR_X4_Y2,
    inout wire              CRTR_X4_Y3,
    inout wire              CRTR_X3_Y3,
    inout wire              CRTR_X5_Y4,
    inout wire              CRTR_X4_Y4,
    inout wire              CRTR_X3_Y4,
    inout wire              CRTR_X2_Y4,
    inout wire              CRTR_X1_Y4,
    inout wire              CRTR_X0_Y4,
    inout wire              CRTR_X5_Y3,
    inout wire              CRTR_X5_Y2,
    inout wire              CRTR_X5_Y1,
    inout wire              CRTR_X5_Y0,
    inout wire              CRTR_X2_Y1,
    inout wire              CRTR_X0_Y0,
    inout wire              CRTR_X1_Y0,
    inout wire              CRTR_X2_Y0,
    inout wire              M_X0_Y9,
    inout wire              M_X8_Y9,
    inout wire              M_X9_Y9,
    inout wire              M_X1_Y9,
    inout wire              M_X3_Y9,
    inout wire              M_X2_Y9,
    inout wire              M_X4_Y9,
    inout wire              M_X6_Y9,
    inout wire              M_X7_Y9,
    inout wire              M_X5_Y9,
    inout wire              M_X9_Y8,
    inout wire              M_X9_Y7,
    inout wire              M_X9_Y6,
    inout wire              M_X9_Y5,
    inout wire              M_X9_Y4,
    inout wire              M_X9_Y3,
    inout wire              M_X9_Y2,
    inout wire              M_X9_Y1,
    inout wire              M_X9_Y0,
    inout wire              M_X0_Y8,
    inout wire              M_X0_Y7,
    inout wire              M_X0_Y6,
    inout wire              M_X0_Y5,
    inout wire              M_X0_Y4,
    inout wire              M_X0_Y3,
    inout wire              M_X0_Y2,
    inout wire              M_X0_Y1,
    inout wire              M_X0_Y0,
    inout wire              M_X8_Y0,
    inout wire              M_X1_Y0,
    inout wire              M_X3_Y0,
    inout wire              M_X2_Y0,
    inout wire              M_X4_Y0,
    inout wire              M_X6_Y0,
    inout wire              M_X7_Y0,
    inout wire              M_X5_Y0,
    inout wire              D2DE_X5_Y6,
    inout wire              D2DE_X5_Y7,
    inout wire              D2DE_X5_Y8,
    inout wire              D2DE_X5_Y9,
    inout wire              D2DE_X5_Y10,
    inout wire              D2DE_X5_Y1,
    inout wire              D2DE_X5_Y2,
    inout wire              D2DE_X5_Y3,
    inout wire              D2DE_X5_Y4,
    inout wire              D2DE_X5_Y5,
    inout wire              D2DE_X5_Y0,
    inout wire              D2DE_X4_Y8,
    inout wire              D2DE_X4_Y7,
    inout wire              D2DE_X4_Y6,
    inout wire              D2DE_X3_Y6,
    inout wire              D2DE_X2_Y6,
    inout wire              D2DE_X1_Y6,
    inout wire              D2DE_X0_Y6,
    inout wire              D2DE_X4_Y5,
    inout wire              D2DE_X3_Y5,
    inout wire              D2DE_X2_Y5,
    inout wire              D2DE_X1_Y5,
    inout wire              D2DE_X4_Y4,
    inout wire              D2DE_X3_Y4,
    inout wire              D2DE_X2_Y4,
    inout wire              D2DE_X1_Y4,
    inout wire              D2DE_X4_Y3,
    inout wire              D2DE_X4_Y2,
    inout wire              D2DE_X4_Y1,
    inout wire              D2DE_X0_Y4,
    inout wire              D2DE_X0_Y5,
    inout wire              D2DE_X4_Y0,
    inout wire              D2DE_X0_Y0,
    inout wire              D2DE_X0_Y1,
    inout wire              D2DE_X0_Y2,
    inout wire              D2DE_X0_Y3,
    inout wire              D2DE_X1_Y0,
    inout wire              D2DE_X1_Y1,
    inout wire              D2DE_X1_Y2,
    inout wire              D2DE_X1_Y3,
    inout wire              D2DE_X2_Y0,
    inout wire              D2DE_X2_Y1,
    inout wire              D2DE_X2_Y2,
    inout wire              D2DE_X2_Y3,
    inout wire              D2DE_X3_Y0,
    inout wire              D2DE_X3_Y1,
    inout wire              D2DE_X3_Y2,
    inout wire              D2DE_X3_Y3,
    inout wire              D2DE_X0_Y7,
    inout wire              D2DE_X0_Y8,
    inout wire              D2DE_X0_Y9,
    inout wire              D2DE_X0_Y10,
    inout wire              D2DE_X1_Y7,
    inout wire              D2DE_X1_Y8,
    inout wire              D2DE_X1_Y9,
    inout wire              D2DE_X1_Y10,
    inout wire              D2DE_X2_Y7,
    inout wire              D2DE_X2_Y8,
    inout wire              D2DE_X2_Y9,
    inout wire              D2DE_X2_Y10,
    inout wire              D2DE_X3_Y7,
    inout wire              D2DE_X3_Y8,
    inout wire              D2DE_X3_Y9,
    inout wire              D2DE_X3_Y10,
    inout wire              D2DE_X4_Y10,
    inout wire              D2DE_X4_Y9,
    inout wire              D2DN_X0_Y5,
    inout wire              D2DN_X1_Y5,
    inout wire              D2DN_X2_Y5,
    inout wire              D2DN_X0_Y0,
    inout wire              D2DN_X1_Y0,
    inout wire              D2DN_X7_Y1,
    inout wire              D2DN_X7_Y2,
    inout wire              D2DN_X7_Y3,
    inout wire              D2DN_X7_Y4,
    inout wire              D2DN_X8_Y1,
    inout wire              D2DN_X8_Y2,
    inout wire              D2DN_X8_Y3,
    inout wire              D2DN_X8_Y4,
    inout wire              D2DN_X9_Y1,
    inout wire              D2DN_X9_Y2,
    inout wire              D2DN_X9_Y3,
    inout wire              D2DN_X9_Y4,
    inout wire              D2DN_X10_Y1,
    inout wire              D2DN_X10_Y2,
    inout wire              D2DN_X10_Y3,
    inout wire              D2DN_X10_Y4,
    inout wire              D2DN_X0_Y1,
    inout wire              D2DN_X0_Y2,
    inout wire              D2DN_X0_Y3,
    inout wire              D2DN_X0_Y4,
    inout wire              D2DN_X1_Y1,
    inout wire              D2DN_X1_Y2,
    inout wire              D2DN_X1_Y3,
    inout wire              D2DN_X1_Y4,
    inout wire              D2DN_X2_Y1,
    inout wire              D2DN_X2_Y2,
    inout wire              D2DN_X2_Y3,
    inout wire              D2DN_X2_Y4,
    inout wire              D2DN_X3_Y1,
    inout wire              D2DN_X3_Y2,
    inout wire              D2DN_X3_Y3,
    inout wire              D2DN_X3_Y4,
    inout wire              D2DN_X9_Y5,
    inout wire              D2DN_X10_Y5,
    inout wire              D2DN_X9_Y0,
    inout wire              D2DN_X10_Y0,
    inout wire              D2DN_X2_Y0,
    inout wire              D2DN_X4_Y0,
    inout wire              D2DN_X4_Y1,
    inout wire              D2DN_X4_Y2,
    inout wire              D2DN_X3_Y0,
    inout wire              D2DN_X5_Y0,
    inout wire              D2DN_X5_Y3,
    inout wire              D2DN_X5_Y4,
    inout wire              D2DN_X7_Y0,
    inout wire              D2DN_X6_Y1,
    inout wire              D2DN_X5_Y1,
    inout wire              D2DN_X6_Y3,
    inout wire              D2DN_X8_Y0,
    inout wire              D2DN_X6_Y0,
    inout wire              D2DN_X6_Y2,
    inout wire              D2DN_X6_Y4,
    inout wire              D2DN_X5_Y5,
    inout wire              D2DN_X6_Y5,
    inout wire              D2DN_X4_Y3,
    inout wire              D2DN_X4_Y4,
    inout wire              D2DN_X8_Y5,
    inout wire              D2DN_X7_Y5,
    inout wire              D2DN_X4_Y5,
    inout wire              D2DN_X3_Y5,
    inout wire              D2DN_X5_Y2,
    inout wire              D2DS_X0_Y0,
    inout wire              D2DS_X1_Y0,
    inout wire              D2DS_X8_Y0,
    inout wire              D2DS_X9_Y0,
    inout wire              D2DS_X4_Y0,
    inout wire              D2DS_X5_Y0,
    inout wire              D2DS_X7_Y0,
    inout wire              D2DS_X6_Y0,
    inout wire              D2DS_X3_Y0,
    inout wire              D2DS_X2_Y0,
    inout wire              D2DS_X10_Y0,
    inout wire              D2DS_X0_Y1,
    inout wire              D2DS_X1_Y1,
    inout wire              D2DS_X7_Y2,
    inout wire              D2DS_X7_Y3,
    inout wire              D2DS_X7_Y4,
    inout wire              D2DS_X7_Y5,
    inout wire              D2DS_X8_Y2,
    inout wire              D2DS_X8_Y3,
    inout wire              D2DS_X8_Y4,
    inout wire              D2DS_X8_Y5,
    inout wire              D2DS_X9_Y2,
    inout wire              D2DS_X9_Y3,
    inout wire              D2DS_X9_Y4,
    inout wire              D2DS_X9_Y5,
    inout wire              D2DS_X10_Y2,
    inout wire              D2DS_X10_Y3,
    inout wire              D2DS_X10_Y4,
    inout wire              D2DS_X10_Y5,
    inout wire              D2DS_X0_Y2,
    inout wire              D2DS_X0_Y3,
    inout wire              D2DS_X0_Y4,
    inout wire              D2DS_X0_Y5,
    inout wire              D2DS_X1_Y2,
    inout wire              D2DS_X1_Y3,
    inout wire              D2DS_X1_Y4,
    inout wire              D2DS_X1_Y5,
    inout wire              D2DS_X2_Y2,
    inout wire              D2DS_X2_Y3,
    inout wire              D2DS_X2_Y4,
    inout wire              D2DS_X2_Y5,
    inout wire              D2DS_X3_Y2,
    inout wire              D2DS_X3_Y3,
    inout wire              D2DS_X3_Y4,
    inout wire              D2DS_X3_Y5,
    inout wire              D2DS_X9_Y1,
    inout wire              D2DS_X10_Y1,
    inout wire              D2DS_X2_Y1,
    inout wire              D2DS_X4_Y1,
    inout wire              D2DS_X4_Y2,
    inout wire              D2DS_X4_Y3,
    inout wire              D2DS_X3_Y1,
    inout wire              D2DS_X5_Y1,
    inout wire              D2DS_X5_Y4,
    inout wire              D2DS_X5_Y5,
    inout wire              D2DS_X7_Y1,
    inout wire              D2DS_X6_Y2,
    inout wire              D2DS_X5_Y2,
    inout wire              D2DS_X6_Y4,
    inout wire              D2DS_X8_Y1,
    inout wire              D2DS_X6_Y1,
    inout wire              D2DS_X6_Y3,
    inout wire              D2DS_X6_Y5,
    inout wire              D2DS_X4_Y4,
    inout wire              D2DS_X4_Y5,
    inout wire              D2DS_X5_Y3,
    inout wire              D2DW_X0_Y10,
    inout wire              D2DW_X0_Y9,
    inout wire              D2DW_X0_Y8,
    inout wire              D2DW_X0_Y7,
    inout wire              D2DW_X0_Y6,
    inout wire              D2DW_X0_Y5,
    inout wire              D2DW_X0_Y4,
    inout wire              D2DW_X0_Y3,
    inout wire              D2DW_X0_Y2,
    inout wire              D2DW_X0_Y1,
    inout wire              D2DW_X0_Y0,
    inout wire              D2DW_X5_Y8,
    inout wire              D2DW_X5_Y7,
    inout wire              D2DW_X5_Y6,
    inout wire              D2DW_X4_Y6,
    inout wire              D2DW_X3_Y6,
    inout wire              D2DW_X2_Y6,
    inout wire              D2DW_X1_Y6,
    inout wire              D2DW_X5_Y5,
    inout wire              D2DW_X4_Y5,
    inout wire              D2DW_X3_Y5,
    inout wire              D2DW_X2_Y5,
    inout wire              D2DW_X5_Y4,
    inout wire              D2DW_X4_Y4,
    inout wire              D2DW_X3_Y4,
    inout wire              D2DW_X2_Y4,
    inout wire              D2DW_X5_Y3,
    inout wire              D2DW_X5_Y2,
    inout wire              D2DW_X5_Y1,
    inout wire              D2DW_X1_Y4,
    inout wire              D2DW_X1_Y5,
    inout wire              D2DW_X5_Y0,
    inout wire              D2DW_X1_Y0,
    inout wire              D2DW_X1_Y1,
    inout wire              D2DW_X1_Y2,
    inout wire              D2DW_X1_Y3,
    inout wire              D2DW_X2_Y0,
    inout wire              D2DW_X2_Y1,
    inout wire              D2DW_X2_Y2,
    inout wire              D2DW_X2_Y3,
    inout wire              D2DW_X3_Y0,
    inout wire              D2DW_X3_Y1,
    inout wire              D2DW_X3_Y2,
    inout wire              D2DW_X3_Y3,
    inout wire              D2DW_X4_Y0,
    inout wire              D2DW_X4_Y1,
    inout wire              D2DW_X4_Y2,
    inout wire              D2DW_X4_Y3,
    inout wire              D2DW_X1_Y7,
    inout wire              D2DW_X1_Y8,
    inout wire              D2DW_X1_Y9,
    inout wire              D2DW_X1_Y10,
    inout wire              D2DW_X2_Y7,
    inout wire              D2DW_X2_Y8,
    inout wire              D2DW_X2_Y9,
    inout wire              D2DW_X2_Y10,
    inout wire              D2DW_X3_Y7,
    inout wire              D2DW_X3_Y8,
    inout wire              D2DW_X3_Y9,
    inout wire              D2DW_X3_Y10,
    inout wire              D2DW_X4_Y7,
    inout wire              D2DW_X4_Y8,
    inout wire              D2DW_X4_Y9,
    inout wire              D2DW_X4_Y10,
    inout wire              D2DW_X5_Y10,
    inout wire              D2DW_X5_Y9
);

    cds_thru i_cds_thru_0 (.src(io_clk_i), .dst(CRTL_X1_Y3));
    cds_thru i_cds_thru_1 (.src(io_rst_ni), .dst(CRTL_X1_Y1));
    cds_thru i_cds_thru_2 (.src(io_clk_obs_o), .dst(CRBL_X5_Y2));
    cds_thru i_cds_thru_3 (.src(io_clk_periph_i), .dst(CRTL_X2_Y4));
    cds_thru i_cds_thru_4 (.src(io_rst_periph_ni), .dst(CRTL_X1_Y0));
    cds_thru i_cds_thru_5 (.src(io_pll_bypass_i), .dst(CRTL_X1_Y4));
    cds_thru i_cds_thru_6 (.src(io_pll_en_i), .dst(CRTL_X3_Y1));
    cds_thru i_cds_thru_7 (.src(io_pll_post_div_sel_i[1]), .dst(CRTL_X4_Y1));
    cds_thru i_cds_thru_8 (.src(io_pll_post_div_sel_i[0]), .dst(CRTL_X4_Y2));
    cds_thru i_cds_thru_9 (.src(io_pll_lock_o), .dst(CRTL_X3_Y2));
    cds_thru i_cds_thru_10 (.src(io_test_mode_i), .dst(CRTL_X1_Y2));
    cds_thru i_cds_thru_11 (.src(io_chip_id_i[7]), .dst(CRBL_X2_Y1));
    cds_thru i_cds_thru_12 (.src(io_chip_id_i[6]), .dst(CRBL_X3_Y1));
    cds_thru i_cds_thru_13 (.src(io_chip_id_i[5]), .dst(CRBL_X4_Y1));
    cds_thru i_cds_thru_14 (.src(io_chip_id_i[4]), .dst(CRBL_X5_Y1));
    cds_thru i_cds_thru_15 (.src(io_chip_id_i[3]), .dst(CRBR_X0_Y1));
    cds_thru i_cds_thru_16 (.src(io_chip_id_i[2]), .dst(CRBR_X1_Y1));
    cds_thru i_cds_thru_17 (.src(io_chip_id_i[1]), .dst(CRBR_X2_Y1));
    cds_thru i_cds_thru_18 (.src(io_chip_id_i[0]), .dst(CRBR_X3_Y1));
    cds_thru i_cds_thru_19 (.src(io_boot_mode_i), .dst(CRBL_X3_Y4));
    cds_thru i_cds_thru_20 (.src(io_east_test_being_requested_i), .dst(CRTR_X3_Y0));
    cds_thru i_cds_thru_21 (.src(io_east_test_request_o), .dst(CRTR_X2_Y1));
    cds_thru i_cds_thru_22 (.src(io_flow_control_east_rts_o), .dst(CRBR_X2_Y4));
    cds_thru i_cds_thru_23 (.src(io_flow_control_east_cts_i), .dst(CRBR_X2_Y5));
    cds_thru i_cds_thru_24 (.src(io_flow_control_east_rts_i), .dst(CRBR_X2_Y3));
    cds_thru i_cds_thru_25 (.src(io_flow_control_east_cts_o), .dst(CRBR_X1_Y3));
    cds_thru i_cds_thru_26 (.src(io_east_d2d[2][19]), .dst(D2DE_X3_Y10));
    cds_thru i_cds_thru_27 (.src(io_east_d2d[2][18]), .dst(D2DE_X3_Y9));
    cds_thru i_cds_thru_28 (.src(io_east_d2d[2][17]), .dst(D2DE_X3_Y8));
    cds_thru i_cds_thru_29 (.src(io_east_d2d[2][16]), .dst(D2DE_X3_Y7));
    cds_thru i_cds_thru_30 (.src(io_east_d2d[2][15]), .dst(D2DE_X2_Y10));
    cds_thru i_cds_thru_31 (.src(io_east_d2d[2][14]), .dst(D2DE_X2_Y9));
    cds_thru i_cds_thru_32 (.src(io_east_d2d[2][13]), .dst(D2DE_X2_Y8));
    cds_thru i_cds_thru_33 (.src(io_east_d2d[2][12]), .dst(D2DE_X2_Y7));
    cds_thru i_cds_thru_34 (.src(io_east_d2d[2][11]), .dst(D2DE_X1_Y10));
    cds_thru i_cds_thru_35 (.src(io_east_d2d[2][10]), .dst(D2DE_X1_Y9));
    cds_thru i_cds_thru_36 (.src(io_east_d2d[2][9]), .dst(D2DE_X1_Y8));
    cds_thru i_cds_thru_37 (.src(io_east_d2d[2][8]), .dst(D2DE_X1_Y7));
    cds_thru i_cds_thru_38 (.src(io_east_d2d[2][7]), .dst(D2DE_X0_Y10));
    cds_thru i_cds_thru_39 (.src(io_east_d2d[2][6]), .dst(D2DE_X0_Y9));
    cds_thru i_cds_thru_40 (.src(io_east_d2d[2][5]), .dst(D2DE_X0_Y8));
    cds_thru i_cds_thru_41 (.src(io_east_d2d[2][4]), .dst(D2DE_X0_Y7));
    cds_thru i_cds_thru_42 (.src(io_east_d2d[2][3]), .dst(D2DE_X4_Y10));
    cds_thru i_cds_thru_43 (.src(io_east_d2d[2][2]), .dst(D2DE_X4_Y9));
    cds_thru i_cds_thru_44 (.src(io_east_d2d[2][1]), .dst(D2DE_X5_Y9));
    cds_thru i_cds_thru_45 (.src(io_east_d2d[2][0]), .dst(D2DE_X5_Y8));
    cds_thru i_cds_thru_46 (.src(io_east_d2d[1][19]), .dst(D2DE_X4_Y8));
    cds_thru i_cds_thru_47 (.src(io_east_d2d[1][18]), .dst(D2DE_X4_Y7));
    cds_thru i_cds_thru_48 (.src(io_east_d2d[1][17]), .dst(D2DE_X4_Y3));
    cds_thru i_cds_thru_49 (.src(io_east_d2d[1][16]), .dst(D2DE_X4_Y2));
    cds_thru i_cds_thru_50 (.src(io_east_d2d[1][15]), .dst(D2DE_X3_Y6));
    cds_thru i_cds_thru_51 (.src(io_east_d2d[1][14]), .dst(D2DE_X4_Y6));
    cds_thru i_cds_thru_52 (.src(io_east_d2d[1][13]), .dst(D2DE_X4_Y4));
    cds_thru i_cds_thru_53 (.src(io_east_d2d[1][12]), .dst(D2DE_X3_Y4));
    cds_thru i_cds_thru_54 (.src(io_east_d2d[1][11]), .dst(D2DE_X2_Y6));
    cds_thru i_cds_thru_55 (.src(io_east_d2d[1][10]), .dst(D2DE_X4_Y5));
    cds_thru i_cds_thru_56 (.src(io_east_d2d[1][9]), .dst(D2DE_X3_Y5));
    cds_thru i_cds_thru_57 (.src(io_east_d2d[1][8]), .dst(D2DE_X2_Y4));
    cds_thru i_cds_thru_58 (.src(io_east_d2d[1][7]), .dst(D2DE_X1_Y6));
    cds_thru i_cds_thru_59 (.src(io_east_d2d[1][6]), .dst(D2DE_X1_Y5));
    cds_thru i_cds_thru_60 (.src(io_east_d2d[1][5]), .dst(D2DE_X1_Y4));
    cds_thru i_cds_thru_61 (.src(io_east_d2d[1][4]), .dst(D2DE_X0_Y4));
    cds_thru i_cds_thru_62 (.src(io_east_d2d[1][3]), .dst(D2DE_X0_Y6));
    cds_thru i_cds_thru_63 (.src(io_east_d2d[1][2]), .dst(D2DE_X0_Y5));
    cds_thru i_cds_thru_64 (.src(io_east_d2d[1][1]), .dst(D2DE_X5_Y5));
    cds_thru i_cds_thru_65 (.src(io_east_d2d[1][0]), .dst(D2DE_X5_Y4));
    cds_thru i_cds_thru_66 (.src(io_east_d2d[0][19]), .dst(D2DE_X3_Y3));
    cds_thru i_cds_thru_67 (.src(io_east_d2d[0][18]), .dst(D2DE_X3_Y2));
    cds_thru i_cds_thru_68 (.src(io_east_d2d[0][17]), .dst(D2DE_X3_Y1));
    cds_thru i_cds_thru_69 (.src(io_east_d2d[0][16]), .dst(D2DE_X3_Y0));
    cds_thru i_cds_thru_70 (.src(io_east_d2d[0][15]), .dst(D2DE_X2_Y3));
    cds_thru i_cds_thru_71 (.src(io_east_d2d[0][14]), .dst(D2DE_X2_Y2));
    cds_thru i_cds_thru_72 (.src(io_east_d2d[0][13]), .dst(D2DE_X2_Y1));
    cds_thru i_cds_thru_73 (.src(io_east_d2d[0][12]), .dst(D2DE_X2_Y0));
    cds_thru i_cds_thru_74 (.src(io_east_d2d[0][11]), .dst(D2DE_X1_Y3));
    cds_thru i_cds_thru_75 (.src(io_east_d2d[0][10]), .dst(D2DE_X1_Y2));
    cds_thru i_cds_thru_76 (.src(io_east_d2d[0][9]), .dst(D2DE_X1_Y1));
    cds_thru i_cds_thru_77 (.src(io_east_d2d[0][8]), .dst(D2DE_X1_Y0));
    cds_thru i_cds_thru_78 (.src(io_east_d2d[0][7]), .dst(D2DE_X0_Y3));
    cds_thru i_cds_thru_79 (.src(io_east_d2d[0][6]), .dst(D2DE_X0_Y2));
    cds_thru i_cds_thru_80 (.src(io_east_d2d[0][5]), .dst(D2DE_X0_Y1));
    cds_thru i_cds_thru_81 (.src(io_east_d2d[0][4]), .dst(D2DE_X0_Y0));
    cds_thru i_cds_thru_82 (.src(io_east_d2d[0][3]), .dst(D2DE_X4_Y1));
    cds_thru i_cds_thru_83 (.src(io_east_d2d[0][2]), .dst(D2DE_X4_Y0));
    cds_thru i_cds_thru_84 (.src(io_east_d2d[0][1]), .dst(D2DE_X5_Y1));
    cds_thru i_cds_thru_85 (.src(io_east_d2d[0][0]), .dst(D2DE_X5_Y0));
    cds_thru i_cds_thru_86 (.src(io_west_test_being_requested_i), .dst(CRTL_X2_Y2));
    cds_thru i_cds_thru_87 (.src(io_west_test_request_o), .dst(CRTL_X2_Y3));
    cds_thru i_cds_thru_88 (.src(io_flow_control_west_rts_o), .dst(CRBL_X2_Y2));
    cds_thru i_cds_thru_89 (.src(io_flow_control_west_cts_i), .dst(CRBL_X2_Y4));
    cds_thru i_cds_thru_90 (.src(io_flow_control_west_rts_i), .dst(CRBL_X3_Y3));
    cds_thru i_cds_thru_91 (.src(io_flow_control_west_cts_o), .dst(CRBL_X2_Y3));
    cds_thru i_cds_thru_92 (.src(io_west_d2d[2][19]), .dst(D2DW_X4_Y10));
    cds_thru i_cds_thru_93 (.src(io_west_d2d[2][18]), .dst(D2DW_X4_Y9));
    cds_thru i_cds_thru_94 (.src(io_west_d2d[2][17]), .dst(D2DW_X4_Y8));
    cds_thru i_cds_thru_95 (.src(io_west_d2d[2][16]), .dst(D2DW_X4_Y7));
    cds_thru i_cds_thru_96 (.src(io_west_d2d[2][15]), .dst(D2DW_X3_Y10));
    cds_thru i_cds_thru_97 (.src(io_west_d2d[2][14]), .dst(D2DW_X3_Y9));
    cds_thru i_cds_thru_98 (.src(io_west_d2d[2][13]), .dst(D2DW_X3_Y8));
    cds_thru i_cds_thru_99 (.src(io_west_d2d[2][12]), .dst(D2DW_X3_Y7));
    cds_thru i_cds_thru_100 (.src(io_west_d2d[2][11]), .dst(D2DW_X2_Y10));
    cds_thru i_cds_thru_101 (.src(io_west_d2d[2][10]), .dst(D2DW_X2_Y9));
    cds_thru i_cds_thru_102 (.src(io_west_d2d[2][9]), .dst(D2DW_X2_Y8));
    cds_thru i_cds_thru_103 (.src(io_west_d2d[2][8]), .dst(D2DW_X2_Y7));
    cds_thru i_cds_thru_104 (.src(io_west_d2d[2][7]), .dst(D2DW_X1_Y10));
    cds_thru i_cds_thru_105 (.src(io_west_d2d[2][6]), .dst(D2DW_X1_Y9));
    cds_thru i_cds_thru_106 (.src(io_west_d2d[2][5]), .dst(D2DW_X1_Y8));
    cds_thru i_cds_thru_107 (.src(io_west_d2d[2][4]), .dst(D2DW_X1_Y7));
    cds_thru i_cds_thru_108 (.src(io_west_d2d[2][3]), .dst(D2DW_X5_Y10));
    cds_thru i_cds_thru_109 (.src(io_west_d2d[2][2]), .dst(D2DW_X5_Y9));
    cds_thru i_cds_thru_110 (.src(io_west_d2d[2][1]), .dst(D2DW_X0_Y10));
    cds_thru i_cds_thru_111 (.src(io_west_d2d[2][0]), .dst(D2DW_X0_Y9));
    cds_thru i_cds_thru_112 (.src(io_west_d2d[1][19]), .dst(D2DW_X5_Y8));
    cds_thru i_cds_thru_113 (.src(io_west_d2d[1][18]), .dst(D2DW_X5_Y7));
    cds_thru i_cds_thru_114 (.src(io_west_d2d[1][17]), .dst(D2DW_X5_Y3));
    cds_thru i_cds_thru_115 (.src(io_west_d2d[1][16]), .dst(D2DW_X5_Y2));
    cds_thru i_cds_thru_116 (.src(io_west_d2d[1][15]), .dst(D2DW_X4_Y6));
    cds_thru i_cds_thru_117 (.src(io_west_d2d[1][14]), .dst(D2DW_X5_Y6));
    cds_thru i_cds_thru_118 (.src(io_west_d2d[1][13]), .dst(D2DW_X5_Y4));
    cds_thru i_cds_thru_119 (.src(io_west_d2d[1][12]), .dst(D2DW_X4_Y4));
    cds_thru i_cds_thru_120 (.src(io_west_d2d[1][11]), .dst(D2DW_X3_Y6));
    cds_thru i_cds_thru_121 (.src(io_west_d2d[1][10]), .dst(D2DW_X5_Y5));
    cds_thru i_cds_thru_122 (.src(io_west_d2d[1][9]), .dst(D2DW_X4_Y5));
    cds_thru i_cds_thru_123 (.src(io_west_d2d[1][8]), .dst(D2DW_X3_Y4));
    cds_thru i_cds_thru_124 (.src(io_west_d2d[1][7]), .dst(D2DW_X2_Y6));
    cds_thru i_cds_thru_125 (.src(io_west_d2d[1][6]), .dst(D2DW_X2_Y5));
    cds_thru i_cds_thru_126 (.src(io_west_d2d[1][5]), .dst(D2DW_X2_Y4));
    cds_thru i_cds_thru_127 (.src(io_west_d2d[1][4]), .dst(D2DW_X1_Y4));
    cds_thru i_cds_thru_128 (.src(io_west_d2d[1][3]), .dst(D2DW_X1_Y6));
    cds_thru i_cds_thru_129 (.src(io_west_d2d[1][2]), .dst(D2DW_X1_Y5));
    cds_thru i_cds_thru_130 (.src(io_west_d2d[1][1]), .dst(D2DW_X0_Y6));
    cds_thru i_cds_thru_131 (.src(io_west_d2d[1][0]), .dst(D2DW_X0_Y5));
    cds_thru i_cds_thru_132 (.src(io_west_d2d[0][19]), .dst(D2DW_X4_Y3));
    cds_thru i_cds_thru_133 (.src(io_west_d2d[0][18]), .dst(D2DW_X4_Y2));
    cds_thru i_cds_thru_134 (.src(io_west_d2d[0][17]), .dst(D2DW_X4_Y1));
    cds_thru i_cds_thru_135 (.src(io_west_d2d[0][16]), .dst(D2DW_X4_Y0));
    cds_thru i_cds_thru_136 (.src(io_west_d2d[0][15]), .dst(D2DW_X3_Y3));
    cds_thru i_cds_thru_137 (.src(io_west_d2d[0][14]), .dst(D2DW_X3_Y2));
    cds_thru i_cds_thru_138 (.src(io_west_d2d[0][13]), .dst(D2DW_X3_Y1));
    cds_thru i_cds_thru_139 (.src(io_west_d2d[0][12]), .dst(D2DW_X3_Y0));
    cds_thru i_cds_thru_140 (.src(io_west_d2d[0][11]), .dst(D2DW_X2_Y3));
    cds_thru i_cds_thru_141 (.src(io_west_d2d[0][10]), .dst(D2DW_X2_Y2));
    cds_thru i_cds_thru_142 (.src(io_west_d2d[0][9]), .dst(D2DW_X2_Y1));
    cds_thru i_cds_thru_143 (.src(io_west_d2d[0][8]), .dst(D2DW_X2_Y0));
    cds_thru i_cds_thru_144 (.src(io_west_d2d[0][7]), .dst(D2DW_X1_Y3));
    cds_thru i_cds_thru_145 (.src(io_west_d2d[0][6]), .dst(D2DW_X1_Y2));
    cds_thru i_cds_thru_146 (.src(io_west_d2d[0][5]), .dst(D2DW_X1_Y1));
    cds_thru i_cds_thru_147 (.src(io_west_d2d[0][4]), .dst(D2DW_X1_Y0));
    cds_thru i_cds_thru_148 (.src(io_west_d2d[0][3]), .dst(D2DW_X5_Y1));
    cds_thru i_cds_thru_149 (.src(io_west_d2d[0][2]), .dst(D2DW_X5_Y0));
    cds_thru i_cds_thru_150 (.src(io_west_d2d[0][1]), .dst(D2DW_X0_Y2));
    cds_thru i_cds_thru_151 (.src(io_west_d2d[0][0]), .dst(D2DW_X0_Y1));
    cds_thru i_cds_thru_152 (.src(io_north_test_being_requested_i), .dst(CRTL_X3_Y3));
    cds_thru i_cds_thru_153 (.src(io_north_test_request_o), .dst(CRTL_X4_Y3));
    cds_thru i_cds_thru_154 (.src(io_flow_control_north_rts_o), .dst(CRTR_X3_Y2));
    cds_thru i_cds_thru_155 (.src(io_flow_control_north_cts_i), .dst(CRTR_X0_Y2));
    cds_thru i_cds_thru_156 (.src(io_flow_control_north_rts_i), .dst(CRTR_X2_Y2));
    cds_thru i_cds_thru_157 (.src(io_flow_control_north_cts_o), .dst(CRTR_X1_Y2));
    cds_thru i_cds_thru_158 (.src(io_north_d2d[2][19]), .dst(D2DN_X10_Y4));
    cds_thru i_cds_thru_159 (.src(io_north_d2d[2][18]), .dst(D2DN_X10_Y3));
    cds_thru i_cds_thru_160 (.src(io_north_d2d[2][17]), .dst(D2DN_X10_Y2));
    cds_thru i_cds_thru_161 (.src(io_north_d2d[2][16]), .dst(D2DN_X10_Y1));
    cds_thru i_cds_thru_162 (.src(io_north_d2d[2][15]), .dst(D2DN_X9_Y4));
    cds_thru i_cds_thru_163 (.src(io_north_d2d[2][14]), .dst(D2DN_X9_Y3));
    cds_thru i_cds_thru_164 (.src(io_north_d2d[2][13]), .dst(D2DN_X9_Y2));
    cds_thru i_cds_thru_165 (.src(io_north_d2d[2][12]), .dst(D2DN_X9_Y1));
    cds_thru i_cds_thru_166 (.src(io_north_d2d[2][11]), .dst(D2DN_X8_Y4));
    cds_thru i_cds_thru_167 (.src(io_north_d2d[2][10]), .dst(D2DN_X8_Y3));
    cds_thru i_cds_thru_168 (.src(io_north_d2d[2][9]), .dst(D2DN_X8_Y2));
    cds_thru i_cds_thru_169 (.src(io_north_d2d[2][8]), .dst(D2DN_X8_Y1));
    cds_thru i_cds_thru_170 (.src(io_north_d2d[2][7]), .dst(D2DN_X7_Y4));
    cds_thru i_cds_thru_171 (.src(io_north_d2d[2][6]), .dst(D2DN_X7_Y3));
    cds_thru i_cds_thru_172 (.src(io_north_d2d[2][5]), .dst(D2DN_X7_Y2));
    cds_thru i_cds_thru_173 (.src(io_north_d2d[2][4]), .dst(D2DN_X7_Y1));
    cds_thru i_cds_thru_174 (.src(io_north_d2d[2][3]), .dst(D2DN_X10_Y0));
    cds_thru i_cds_thru_175 (.src(io_north_d2d[2][2]), .dst(D2DN_X9_Y0));
    cds_thru i_cds_thru_176 (.src(io_north_d2d[2][1]), .dst(D2DN_X10_Y5));
    cds_thru i_cds_thru_177 (.src(io_north_d2d[2][0]), .dst(D2DN_X9_Y5));
    cds_thru i_cds_thru_178 (.src(io_north_d2d[1][19]), .dst(D2DN_X6_Y4));
    cds_thru i_cds_thru_179 (.src(io_north_d2d[1][18]), .dst(D2DN_X6_Y2));
    cds_thru i_cds_thru_180 (.src(io_north_d2d[1][17]), .dst(D2DN_X6_Y0));
    cds_thru i_cds_thru_181 (.src(io_north_d2d[1][16]), .dst(D2DN_X8_Y0));
    cds_thru i_cds_thru_182 (.src(io_north_d2d[1][15]), .dst(D2DN_X6_Y3));
    cds_thru i_cds_thru_183 (.src(io_north_d2d[1][14]), .dst(D2DN_X5_Y1));
    cds_thru i_cds_thru_184 (.src(io_north_d2d[1][13]), .dst(D2DN_X6_Y1));
    cds_thru i_cds_thru_185 (.src(io_north_d2d[1][12]), .dst(D2DN_X7_Y0));
    cds_thru i_cds_thru_186 (.src(io_north_d2d[1][11]), .dst(D2DN_X5_Y4));
    cds_thru i_cds_thru_187 (.src(io_north_d2d[1][10]), .dst(D2DN_X5_Y3));
    cds_thru i_cds_thru_188 (.src(io_north_d2d[1][9]), .dst(D2DN_X5_Y0));
    cds_thru i_cds_thru_189 (.src(io_north_d2d[1][8]), .dst(D2DN_X3_Y0));
    cds_thru i_cds_thru_190 (.src(io_north_d2d[1][7]), .dst(D2DN_X4_Y2));
    cds_thru i_cds_thru_191 (.src(io_north_d2d[1][6]), .dst(D2DN_X4_Y1));
    cds_thru i_cds_thru_192 (.src(io_north_d2d[1][5]), .dst(D2DN_X4_Y0));
    cds_thru i_cds_thru_193 (.src(io_north_d2d[1][4]), .dst(D2DN_X2_Y0));
    cds_thru i_cds_thru_194 (.src(io_north_d2d[1][3]), .dst(D2DN_X4_Y4));
    cds_thru i_cds_thru_195 (.src(io_north_d2d[1][2]), .dst(D2DN_X4_Y3));
    cds_thru i_cds_thru_196 (.src(io_north_d2d[1][1]), .dst(D2DN_X6_Y5));
    cds_thru i_cds_thru_197 (.src(io_north_d2d[1][0]), .dst(D2DN_X5_Y5));
    cds_thru i_cds_thru_198 (.src(io_north_d2d[0][19]), .dst(D2DN_X3_Y4));
    cds_thru i_cds_thru_199 (.src(io_north_d2d[0][18]), .dst(D2DN_X3_Y3));
    cds_thru i_cds_thru_200 (.src(io_north_d2d[0][17]), .dst(D2DN_X3_Y2));
    cds_thru i_cds_thru_201 (.src(io_north_d2d[0][16]), .dst(D2DN_X3_Y1));
    cds_thru i_cds_thru_202 (.src(io_north_d2d[0][15]), .dst(D2DN_X2_Y4));
    cds_thru i_cds_thru_203 (.src(io_north_d2d[0][14]), .dst(D2DN_X2_Y3));
    cds_thru i_cds_thru_204 (.src(io_north_d2d[0][13]), .dst(D2DN_X2_Y2));
    cds_thru i_cds_thru_205 (.src(io_north_d2d[0][12]), .dst(D2DN_X2_Y1));
    cds_thru i_cds_thru_206 (.src(io_north_d2d[0][11]), .dst(D2DN_X1_Y4));
    cds_thru i_cds_thru_207 (.src(io_north_d2d[0][10]), .dst(D2DN_X1_Y3));
    cds_thru i_cds_thru_208 (.src(io_north_d2d[0][9]), .dst(D2DN_X1_Y2));
    cds_thru i_cds_thru_209 (.src(io_north_d2d[0][8]), .dst(D2DN_X1_Y1));
    cds_thru i_cds_thru_210 (.src(io_north_d2d[0][7]), .dst(D2DN_X0_Y4));
    cds_thru i_cds_thru_211 (.src(io_north_d2d[0][6]), .dst(D2DN_X0_Y3));
    cds_thru i_cds_thru_212 (.src(io_north_d2d[0][5]), .dst(D2DN_X0_Y2));
    cds_thru i_cds_thru_213 (.src(io_north_d2d[0][4]), .dst(D2DN_X0_Y1));
    cds_thru i_cds_thru_214 (.src(io_north_d2d[0][3]), .dst(D2DN_X1_Y0));
    cds_thru i_cds_thru_215 (.src(io_north_d2d[0][2]), .dst(D2DN_X0_Y0));
    cds_thru i_cds_thru_216 (.src(io_north_d2d[0][1]), .dst(D2DN_X2_Y5));
    cds_thru i_cds_thru_217 (.src(io_north_d2d[0][0]), .dst(D2DN_X1_Y5));
    cds_thru i_cds_thru_218 (.src(io_south_test_being_requested_i), .dst(CRBL_X4_Y2));
    cds_thru i_cds_thru_219 (.src(io_south_test_request_o), .dst(CRBL_X3_Y2));
    cds_thru i_cds_thru_220 (.src(io_flow_control_south_rts_o), .dst(CRBR_X0_Y2));
    cds_thru i_cds_thru_221 (.src(io_flow_control_south_cts_i), .dst(CRBR_X2_Y2));
    cds_thru i_cds_thru_222 (.src(io_flow_control_south_rts_i), .dst(CRBR_X1_Y2));
    cds_thru i_cds_thru_223 (.src(io_flow_control_south_cts_o), .dst(CRBR_X0_Y3));
    cds_thru i_cds_thru_224 (.src(io_south_d2d[2][19]), .dst(D2DS_X10_Y5));
    cds_thru i_cds_thru_225 (.src(io_south_d2d[2][18]), .dst(D2DS_X10_Y4));
    cds_thru i_cds_thru_226 (.src(io_south_d2d[2][17]), .dst(D2DS_X10_Y3));
    cds_thru i_cds_thru_227 (.src(io_south_d2d[2][16]), .dst(D2DS_X10_Y2));
    cds_thru i_cds_thru_228 (.src(io_south_d2d[2][15]), .dst(D2DS_X9_Y5));
    cds_thru i_cds_thru_229 (.src(io_south_d2d[2][14]), .dst(D2DS_X9_Y4));
    cds_thru i_cds_thru_230 (.src(io_south_d2d[2][13]), .dst(D2DS_X9_Y3));
    cds_thru i_cds_thru_231 (.src(io_south_d2d[2][12]), .dst(D2DS_X9_Y2));
    cds_thru i_cds_thru_232 (.src(io_south_d2d[2][11]), .dst(D2DS_X8_Y5));
    cds_thru i_cds_thru_233 (.src(io_south_d2d[2][10]), .dst(D2DS_X8_Y4));
    cds_thru i_cds_thru_234 (.src(io_south_d2d[2][9]), .dst(D2DS_X8_Y3));
    cds_thru i_cds_thru_235 (.src(io_south_d2d[2][8]), .dst(D2DS_X8_Y2));
    cds_thru i_cds_thru_236 (.src(io_south_d2d[2][7]), .dst(D2DS_X7_Y5));
    cds_thru i_cds_thru_237 (.src(io_south_d2d[2][6]), .dst(D2DS_X7_Y4));
    cds_thru i_cds_thru_238 (.src(io_south_d2d[2][5]), .dst(D2DS_X7_Y3));
    cds_thru i_cds_thru_239 (.src(io_south_d2d[2][4]), .dst(D2DS_X7_Y2));
    cds_thru i_cds_thru_240 (.src(io_south_d2d[2][3]), .dst(D2DS_X10_Y1));
    cds_thru i_cds_thru_241 (.src(io_south_d2d[2][2]), .dst(D2DS_X9_Y1));
    cds_thru i_cds_thru_242 (.src(io_south_d2d[2][1]), .dst(D2DS_X9_Y0));
    cds_thru i_cds_thru_243 (.src(io_south_d2d[2][0]), .dst(D2DS_X8_Y0));
    cds_thru i_cds_thru_244 (.src(io_south_d2d[1][19]), .dst(D2DS_X6_Y5));
    cds_thru i_cds_thru_245 (.src(io_south_d2d[1][18]), .dst(D2DS_X6_Y3));
    cds_thru i_cds_thru_246 (.src(io_south_d2d[1][17]), .dst(D2DS_X6_Y1));
    cds_thru i_cds_thru_247 (.src(io_south_d2d[1][16]), .dst(D2DS_X8_Y1));
    cds_thru i_cds_thru_248 (.src(io_south_d2d[1][15]), .dst(D2DS_X6_Y4));
    cds_thru i_cds_thru_249 (.src(io_south_d2d[1][14]), .dst(D2DS_X5_Y2));
    cds_thru i_cds_thru_250 (.src(io_south_d2d[1][13]), .dst(D2DS_X6_Y2));
    cds_thru i_cds_thru_251 (.src(io_south_d2d[1][12]), .dst(D2DS_X7_Y1));
    cds_thru i_cds_thru_252 (.src(io_south_d2d[1][11]), .dst(D2DS_X5_Y5));
    cds_thru i_cds_thru_253 (.src(io_south_d2d[1][10]), .dst(D2DS_X5_Y4));
    cds_thru i_cds_thru_254 (.src(io_south_d2d[1][9]), .dst(D2DS_X5_Y1));
    cds_thru i_cds_thru_255 (.src(io_south_d2d[1][8]), .dst(D2DS_X3_Y1));
    cds_thru i_cds_thru_256 (.src(io_south_d2d[1][7]), .dst(D2DS_X4_Y3));
    cds_thru i_cds_thru_257 (.src(io_south_d2d[1][6]), .dst(D2DS_X4_Y2));
    cds_thru i_cds_thru_258 (.src(io_south_d2d[1][5]), .dst(D2DS_X4_Y1));
    cds_thru i_cds_thru_259 (.src(io_south_d2d[1][4]), .dst(D2DS_X2_Y1));
    cds_thru i_cds_thru_260 (.src(io_south_d2d[1][3]), .dst(D2DS_X4_Y5));
    cds_thru i_cds_thru_261 (.src(io_south_d2d[1][2]), .dst(D2DS_X4_Y4));
    cds_thru i_cds_thru_262 (.src(io_south_d2d[1][1]), .dst(D2DS_X5_Y0));
    cds_thru i_cds_thru_263 (.src(io_south_d2d[1][0]), .dst(D2DS_X4_Y0));
    cds_thru i_cds_thru_264 (.src(io_south_d2d[0][19]), .dst(D2DS_X3_Y5));
    cds_thru i_cds_thru_265 (.src(io_south_d2d[0][18]), .dst(D2DS_X3_Y4));
    cds_thru i_cds_thru_266 (.src(io_south_d2d[0][17]), .dst(D2DS_X3_Y3));
    cds_thru i_cds_thru_267 (.src(io_south_d2d[0][16]), .dst(D2DS_X3_Y2));
    cds_thru i_cds_thru_268 (.src(io_south_d2d[0][15]), .dst(D2DS_X2_Y5));
    cds_thru i_cds_thru_269 (.src(io_south_d2d[0][14]), .dst(D2DS_X2_Y4));
    cds_thru i_cds_thru_270 (.src(io_south_d2d[0][13]), .dst(D2DS_X2_Y3));
    cds_thru i_cds_thru_271 (.src(io_south_d2d[0][12]), .dst(D2DS_X2_Y2));
    cds_thru i_cds_thru_272 (.src(io_south_d2d[0][11]), .dst(D2DS_X1_Y5));
    cds_thru i_cds_thru_273 (.src(io_south_d2d[0][10]), .dst(D2DS_X1_Y4));
    cds_thru i_cds_thru_274 (.src(io_south_d2d[0][9]), .dst(D2DS_X1_Y3));
    cds_thru i_cds_thru_275 (.src(io_south_d2d[0][8]), .dst(D2DS_X1_Y2));
    cds_thru i_cds_thru_276 (.src(io_south_d2d[0][7]), .dst(D2DS_X0_Y5));
    cds_thru i_cds_thru_277 (.src(io_south_d2d[0][6]), .dst(D2DS_X0_Y4));
    cds_thru i_cds_thru_278 (.src(io_south_d2d[0][5]), .dst(D2DS_X0_Y3));
    cds_thru i_cds_thru_279 (.src(io_south_d2d[0][4]), .dst(D2DS_X0_Y2));
    cds_thru i_cds_thru_280 (.src(io_south_d2d[0][3]), .dst(D2DS_X1_Y1));
    cds_thru i_cds_thru_281 (.src(io_south_d2d[0][2]), .dst(D2DS_X0_Y1));
    cds_thru i_cds_thru_282 (.src(io_south_d2d[0][1]), .dst(D2DS_X1_Y0));
    cds_thru i_cds_thru_283 (.src(io_south_d2d[0][0]), .dst(D2DS_X0_Y0));
    cds_thru i_cds_thru_284 (.src(io_uart_tx_o), .dst(CRBL_X1_Y4));
    cds_thru i_cds_thru_285 (.src(io_uart_rx_i), .dst(CRBL_X1_Y3));
    cds_thru i_cds_thru_286 (.src(io_uart_rts_no), .dst(CRBL_X1_Y2));
    cds_thru i_cds_thru_287 (.src(io_uart_cts_ni), .dst(CRBL_X1_Y1));
    cds_thru i_cds_thru_288 (.src(io_gpio[3]), .dst(CRBR_X3_Y2));
    cds_thru i_cds_thru_289 (.src(io_gpio[2]), .dst(CRBR_X3_Y3));
    cds_thru i_cds_thru_290 (.src(io_gpio[1]), .dst(CRBR_X3_Y4));
    cds_thru i_cds_thru_291 (.src(io_gpio[0]), .dst(CRBR_X3_Y5));
    cds_thru i_cds_thru_292 (.src(io_spim_sck_o), .dst(CRTL_X3_Y4));
    cds_thru i_cds_thru_293 (.src(io_spim_csb_o), .dst(CRTL_X4_Y4));
    cds_thru i_cds_thru_294 (.src(io_spim_sd[3]), .dst(CRTR_X3_Y3));
    cds_thru i_cds_thru_295 (.src(io_spim_sd[2]), .dst(CRTR_X2_Y3));
    cds_thru i_cds_thru_296 (.src(io_spim_sd[1]), .dst(CRTR_X1_Y3));
    cds_thru i_cds_thru_297 (.src(io_spim_sd[0]), .dst(CRTR_X0_Y3));
    cds_thru i_cds_thru_298 (.src(io_i2c_sda), .dst(CRTL_X2_Y1));
    cds_thru i_cds_thru_299 (.src(io_i2c_scl), .dst(CRTL_X2_Y0));
    cds_thru i_cds_thru_300 (.src(io_jtag_trst_ni), .dst(CRTR_X4_Y3));
    cds_thru i_cds_thru_301 (.src(io_jtag_tck_i), .dst(CRTR_X4_Y2));
    cds_thru i_cds_thru_302 (.src(io_jtag_tms_i), .dst(CRTR_X4_Y1));
    cds_thru i_cds_thru_303 (.src(io_jtag_tdi_i), .dst(CRTR_X4_Y0));
    cds_thru i_cds_thru_304 (.src(io_jtag_tdo_o), .dst(CRTR_X3_Y1));
//   hemaia i_hemaia (
//       .io_clk_i(CRTL_X1_Y3),
//       .io_rst_ni(CRTL_X1_Y1),
//       .io_pll_bypass_i(CRTL_X1_Y4),
//       .io_pll_en_i(CRTL_X3_Y1),
//       .io_pll_post_div_sel_i({CRTL_X4_Y1,CRTL_X4_Y2}),
//       .io_pll_lock_o(CRTL_X3_Y2),
//       .io_clk_obs_o(CRBL_X5_Y2),
//       .io_clk_periph_i(CRTL_X2_Y4),
//       .io_rst_periph_ni(CRTL_X1_Y0),
//       .io_test_mode_i(CRTL_X1_Y2),
//       .io_chip_id_i({
//         CRBL_X2_Y1,
//         CRBL_X3_Y1,
//         CRBL_X4_Y1,
//         CRBL_X5_Y1,
//         CRBR_X0_Y1,
//         CRBR_X1_Y1,
//         CRBR_X2_Y1,
//         CRBR_X3_Y1
//       }),
//       .io_boot_mode_i(CRBL_X3_Y4),
//       .io_east_test_being_requested_i(CRTR_X3_Y0),
//       .io_east_test_request_o(CRTR_X2_Y1),
//       .io_flow_control_east_rts_o(CRBR_X2_Y4),
//       .io_flow_control_east_cts_i(CRBR_X2_Y5),
//       .io_flow_control_east_rts_i(CRBR_X2_Y3),
//       .io_flow_control_east_cts_o(CRBR_X1_Y3),
//       .io_east_d2d({
//         D2DE_X3_Y10,
//         D2DE_X3_Y9,
//         D2DE_X3_Y8,
//         D2DE_X3_Y7,
//         D2DE_X2_Y10,
//         D2DE_X2_Y9,
//         D2DE_X2_Y8,
//         D2DE_X2_Y7,
//         D2DE_X1_Y10,
//         D2DE_X1_Y9,
//         D2DE_X1_Y8,
//         D2DE_X1_Y7,
//         D2DE_X0_Y10,
//         D2DE_X0_Y9,
//         D2DE_X0_Y8,
//         D2DE_X0_Y7,
//         D2DE_X4_Y10,
//         D2DE_X4_Y9,
//         D2DE_X5_Y9,
//         D2DE_X5_Y8,
//         D2DE_X4_Y8,
//         D2DE_X4_Y7,
//         D2DE_X4_Y3,
//         D2DE_X4_Y2,
//         D2DE_X3_Y6,
//         D2DE_X4_Y6,
//         D2DE_X4_Y4,
//         D2DE_X3_Y4,
//         D2DE_X2_Y6,
//         D2DE_X4_Y5,
//         D2DE_X3_Y5,
//         D2DE_X2_Y4,
//         D2DE_X1_Y6,
//         D2DE_X1_Y5,
//         D2DE_X1_Y4,
//         D2DE_X0_Y4,
//         D2DE_X0_Y6,
//         D2DE_X0_Y5,
//         D2DE_X5_Y5,
//         D2DE_X5_Y4,
//         D2DE_X3_Y3,
//         D2DE_X3_Y2,
//         D2DE_X3_Y1,
//         D2DE_X3_Y0,
//         D2DE_X2_Y3,
//         D2DE_X2_Y2,
//         D2DE_X2_Y1,
//         D2DE_X2_Y0,
//         D2DE_X1_Y3,
//         D2DE_X1_Y2,
//         D2DE_X1_Y1,
//         D2DE_X1_Y0,
//         D2DE_X0_Y3,
//         D2DE_X0_Y2,
//         D2DE_X0_Y1,
//         D2DE_X0_Y0,
//         D2DE_X4_Y1,
//         D2DE_X4_Y0,
//         D2DE_X5_Y1,
//         D2DE_X5_Y0
//       }),
//       .io_west_test_being_requested_i(CRTL_X2_Y2),
//       .io_west_test_request_o(CRTL_X2_Y3),
//       .io_flow_control_west_rts_o(CRBL_X2_Y2),
//       .io_flow_control_west_cts_i(CRBL_X2_Y4),
//       .io_flow_control_west_rts_i(CRBL_X3_Y3),
//       .io_flow_control_west_cts_o(CRBL_X2_Y3),
//       .io_west_d2d({
//         D2DW_X4_Y10,
//         D2DW_X4_Y9,
//         D2DW_X4_Y8,
//         D2DW_X4_Y7,
//         D2DW_X3_Y10,
//         D2DW_X3_Y9,
//         D2DW_X3_Y8,
//         D2DW_X3_Y7,
//         D2DW_X2_Y10,
//         D2DW_X2_Y9,
//         D2DW_X2_Y8,
//         D2DW_X2_Y7,
//         D2DW_X1_Y10,
//         D2DW_X1_Y9,
//         D2DW_X1_Y8,
//         D2DW_X1_Y7,
//         D2DW_X5_Y10,
//         D2DW_X5_Y9,
//         D2DW_X0_Y10,
//         D2DW_X0_Y9,
//         D2DW_X5_Y8,
//         D2DW_X5_Y7,
//         D2DW_X5_Y3,
//         D2DW_X5_Y2,
//         D2DW_X4_Y6,
//         D2DW_X5_Y6,
//         D2DW_X5_Y4,
//         D2DW_X4_Y4,
//         D2DW_X3_Y6,
//         D2DW_X5_Y5,
//         D2DW_X4_Y5,
//         D2DW_X3_Y4,
//         D2DW_X2_Y6,
//         D2DW_X2_Y5,
//         D2DW_X2_Y4,
//         D2DW_X1_Y4,
//         D2DW_X1_Y6,
//         D2DW_X1_Y5,
//         D2DW_X0_Y6,
//         D2DW_X0_Y5,
//         D2DW_X4_Y3,
//         D2DW_X4_Y2,
//         D2DW_X4_Y1,
//         D2DW_X4_Y0,
//         D2DW_X3_Y3,
//         D2DW_X3_Y2,
//         D2DW_X3_Y1,
//         D2DW_X3_Y0,
//         D2DW_X2_Y3,
//         D2DW_X2_Y2,
//         D2DW_X2_Y1,
//         D2DW_X2_Y0,
//         D2DW_X1_Y3,
//         D2DW_X1_Y2,
//         D2DW_X1_Y1,
//         D2DW_X1_Y0,
//         D2DW_X5_Y1,
//         D2DW_X5_Y0,
//         D2DW_X0_Y2,
//         D2DW_X0_Y1
//       }),
//       .io_north_test_being_requested_i(CRTL_X3_Y3),
//       .io_north_test_request_o(CRTL_X4_Y3),
//       .io_flow_control_north_rts_o(CRTR_X3_Y2),
//       .io_flow_control_north_cts_i(CRTR_X0_Y2),
//       .io_flow_control_north_rts_i(CRTR_X2_Y2),
//       .io_flow_control_north_cts_o(CRTR_X1_Y2),
//       .io_north_d2d({
//         D2DN_X10_Y4,
//         D2DN_X10_Y3,
//         D2DN_X10_Y2,
//         D2DN_X10_Y1,
//         D2DN_X9_Y4,
//         D2DN_X9_Y3,
//         D2DN_X9_Y2,
//         D2DN_X9_Y1,
//         D2DN_X8_Y4,
//         D2DN_X8_Y3,
//         D2DN_X8_Y2,
//         D2DN_X8_Y1,
//         D2DN_X7_Y4,
//         D2DN_X7_Y3,
//         D2DN_X7_Y2,
//         D2DN_X7_Y1,
//         D2DN_X10_Y0,
//         D2DN_X9_Y0,
//         D2DN_X10_Y5,
//         D2DN_X9_Y5,
//         D2DN_X6_Y4,
//         D2DN_X6_Y2,
//         D2DN_X6_Y0,
//         D2DN_X8_Y0,
//         D2DN_X6_Y3,
//         D2DN_X5_Y1,
//         D2DN_X6_Y1,
//         D2DN_X7_Y0,
//         D2DN_X5_Y4,
//         D2DN_X5_Y3,
//         D2DN_X5_Y0,
//         D2DN_X3_Y0,
//         D2DN_X4_Y2,
//         D2DN_X4_Y1,
//         D2DN_X4_Y0,
//         D2DN_X2_Y0,
//         D2DN_X4_Y4,
//         D2DN_X4_Y3,
//         D2DN_X6_Y5,
//         D2DN_X5_Y5,
//         D2DN_X3_Y4,
//         D2DN_X3_Y3,
//         D2DN_X3_Y2,
//         D2DN_X3_Y1,
//         D2DN_X2_Y4,
//         D2DN_X2_Y3,
//         D2DN_X2_Y2,
//         D2DN_X2_Y1,
//         D2DN_X1_Y4,
//         D2DN_X1_Y3,
//         D2DN_X1_Y2,
//         D2DN_X1_Y1,
//         D2DN_X0_Y4,
//         D2DN_X0_Y3,
//         D2DN_X0_Y2,
//         D2DN_X0_Y1,
//         D2DN_X1_Y0,
//         D2DN_X0_Y0,
//         D2DN_X2_Y5,
//         D2DN_X1_Y5
//       }),
//       .io_south_test_being_requested_i(CRBL_X4_Y2),
//       .io_south_test_request_o(CRBL_X3_Y2),
//       .io_flow_control_south_rts_o(CRBR_X0_Y2),
//       .io_flow_control_south_cts_i(CRBR_X2_Y2),
//       .io_flow_control_south_rts_i(CRBR_X1_Y2),
//       .io_flow_control_south_cts_o(CRBR_X0_Y3),
//       .io_south_d2d({
//         D2DS_X10_Y5,
//         D2DS_X10_Y4,
//         D2DS_X10_Y3,
//         D2DS_X10_Y2,
//         D2DS_X9_Y5,
//         D2DS_X9_Y4,
//         D2DS_X9_Y3,
//         D2DS_X9_Y2,
//         D2DS_X8_Y5,
//         D2DS_X8_Y4,
//         D2DS_X8_Y3,
//         D2DS_X8_Y2,
//         D2DS_X7_Y5,
//         D2DS_X7_Y4,
//         D2DS_X7_Y3,
//         D2DS_X7_Y2,
//         D2DS_X10_Y1,
//         D2DS_X9_Y1,
//         D2DS_X9_Y0,
//         D2DS_X8_Y0,
//         D2DS_X6_Y5,
//         D2DS_X6_Y3,
//         D2DS_X6_Y1,
//         D2DS_X8_Y1,
//         D2DS_X6_Y4,
//         D2DS_X5_Y2,
//         D2DS_X6_Y2,
//         D2DS_X7_Y1,
//         D2DS_X5_Y5,
//         D2DS_X5_Y4,
//         D2DS_X5_Y1,
//         D2DS_X3_Y1,
//         D2DS_X4_Y3,
//         D2DS_X4_Y2,
//         D2DS_X4_Y1,
//         D2DS_X2_Y1,
//         D2DS_X4_Y5,
//         D2DS_X4_Y4,
//         D2DS_X5_Y0,
//         D2DS_X4_Y0,
//         D2DS_X3_Y5,
//         D2DS_X3_Y4,
//         D2DS_X3_Y3,
//         D2DS_X3_Y2,
//         D2DS_X2_Y5,
//         D2DS_X2_Y4,
//         D2DS_X2_Y3,
//         D2DS_X2_Y2,
//         D2DS_X1_Y5,
//         D2DS_X1_Y4,
//         D2DS_X1_Y3,
//         D2DS_X1_Y2,
//         D2DS_X0_Y5,
//         D2DS_X0_Y4,
//         D2DS_X0_Y3,
//         D2DS_X0_Y2,
//         D2DS_X1_Y1,
//         D2DS_X0_Y1,
//         D2DS_X1_Y0,
//         D2DS_X0_Y0
//       }),
//       .io_uart_tx_o(CRBL_X1_Y4),
//       .io_uart_rx_i(CRBL_X1_Y3),
//       .io_uart_rts_no(CRBL_X1_Y2),
//       .io_uart_cts_ni(CRBL_X1_Y1),
//       .io_gpio({CRBR_X3_Y2, CRBR_X3_Y3, CRBR_X3_Y4, CRBR_X3_Y5}),
//       .io_spim_sck_o(CRTL_X3_Y4),
//       .io_spim_csb_o(CRTL_X4_Y4),
//       .io_spim_sd({CRTR_X3_Y3, CRTR_X2_Y3, CRTR_X1_Y3, CRTR_X0_Y3}),
//       .io_i2c_sda(CRTL_X2_Y1),
//       .io_i2c_scl(CRTL_X2_Y0),
//       .io_jtag_trst_ni(CRTR_X4_Y3),
//       .io_jtag_tck_i(CRTR_X4_Y2),
//       .io_jtag_tms_i(CRTR_X4_Y1),
//       .io_jtag_tdi_i(CRTR_X4_Y0),
//       .io_jtag_tdo_o(CRTR_X3_Y1)
//   );
endmodule
