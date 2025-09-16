// Copyright 2025 KU Leuven.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51

// Author: Yunhao Deng <yunhao.deng@kuleuven.be>

module hemaia (
    // Clocks, Boot, ChipId (14)
    inout logic        io_clk,
    inout logic        io_rst_n,
    inout logic        io_clk_periph,
    inout logic        io_rst_periph_n,
    inout logic        io_test_mode,
    inout logic [ 7:0] io_chip_id,
    inout logic        io_boot_mode,
% if multichip_cfg["single_chip"] is False:
    // East side (66)
    inout logic        io_east_test_being_requested,
    inout logic        io_east_test_request,
    inout logic        io_flow_control_east_rts_o,
    inout logic        io_flow_control_east_cts_i,
    inout logic        io_flow_control_east_rts_i,
    inout logic        io_flow_control_east_cts_o,
    inout logic [19:0] io_east_d2d                  [3],
    // West side (66)
    inout logic        io_west_test_being_requested,
    inout logic        io_west_test_request,
    inout logic        io_flow_control_west_rts,
    inout logic        io_flow_control_west_cts,
    inout logic        io_flow_control_west_rts,
    inout logic        io_flow_control_west_cts,
    inout logic [19:0] io_west_d2d                  [3],
    // North side (66)
    inout logic        io_north_test_being_requested,
    inout logic        io_north_test_request,
    inout logic        io_flow_control_north_rts,
    inout logic        io_flow_control_north_cts,
    inout logic        io_flow_control_north_rts,
    inout logic        io_flow_control_north_cts,
    inout logic [19:0] io_north_d2d                 [3],
    // South side (66)
    inout logic        io_south_test_being_requested,
    inout logic        io_south_test_request,
    inout logic        io_flow_control_south_rts,
    inout logic        io_flow_control_south_cts,
    inout logic        io_flow_control_south_rts,
    inout logic        io_flow_control_south_cts,
    inout logic [19:0] io_south_d2d                 [3],
% endif
    // `uart` Interface (4)
    inout logic        io_uart_tx,
    inout logic        io_uart_rx,
    inout logic        io_uart_rts_n,
    inout logic        io_uart_cts_n,
    // `gpio` Interface (8)
    inout logic [ 7:0] io_gpio,
    // `SPI Slave` for Debugging Purposes (6)
    inout logic        io_spis_sck,
    inout logic        io_spis_csb,
    inout logic [ 3:0] io_spis_sd,
    // SPI Master Interface (6)
    inout logic        io_spim_sck,
    inout logic        io_spim_csb,
    inout logic [ 3:0] io_spim_sd,
    // `jtag` Interface (5)
    inout logic        io_jtag_trst_n,
    inout logic        io_jtag_tck,
    inout logic        io_jtag_tms,
    inout logic        io_jtag_tdi,
    inout logic        io_jtag_tdo,
    // I2C Interface (2)
    inout logic        io_i2c_sda,
    inout logic        io_i2c_scl
);

  localparam int NumGpio = 8;
  localparam int D2DChannel = 3;
  localparam int D2DWidth = 20;

  logic clk_i;
  tc_digital_io clk_i_io (
      .data_i(1'b0),
      .data_o(clk_i),
      .io_direction_oe_ni(1'b1),
      .io_driving_strength_i(2'b00),
      .io_pullup_en_i(1'b0),
      .io_pulldown_en_i(1'b0),
      .io(io_clk)
  );

  logic rst_ni;
  tc_digital_io rst_ni_io (
      .data_i(1'b1),
      .data_o(rst_ni),
      .io_direction_oe_ni(1'b1),
      .io_driving_strength_i(2'b00),
      .io_pullup_en_i(1'b0),
      .io_pulldown_en_i(1'b0),
      .io(io_rst_n)
  );

  logic clk_periph_i;
  tc_digital_io clk_periph_i_io (
      .data_i(1'b0),
      .data_o(clk_periph_i),
      .io_direction_oe_ni(1'b1),
      .io_driving_strength_i(2'b00),
      .io_pullup_en_i(1'b0),
      .io_pulldown_en_i(1'b0),
      .io(io_clk_periph)
  );

  logic rst_periph_ni;
  tc_digital_io rst_periph_ni_io (
      .data_i(1'b1),
      .data_o(rst_periph_ni),
      .io_direction_oe_ni(1'b1),
      .io_driving_strength_i(2'b00),
      .io_pullup_en_i(1'b0),
      .io_pulldown_en_i(1'b0),
      .io(io_rst_periph_n)
  );

  logic test_mode_i;
  tc_digital_io test_mode_i_io (
      .data_i(1'b0),
      .data_o(test_mode_i),
      .io_direction_oe_ni(1'b1),
      .io_driving_strength_i(2'b00),
      .io_pullup_en_i(1'b0),
      .io_pulldown_en_i(1'b0),
      .io(io_test_mode)
  );

  logic [7:0] chip_id_i;
  tc_digital_io chip_id_i_io[7:0] (
      .data_i(1'b0),
      .data_o(chip_id_i),
      .io_direction_oe_ni(1'b1),
      .io_driving_strength_i(2'b00),
      .io_pullup_en_i(1'b0),
      .io_pulldown_en_i(1'b0),
      .io(io_chip_id)
  );

  logic [1:0] boot_mode_i;
  tc_digital_io boot_mode_i_io[1:0] (
      .data_i(1'b0),
      .data_o(boot_mode_i),
      .io_direction_oe_ni(1'b1),
      .io_driving_strength_i(2'b00),
      .io_pullup_en_i(1'b0),
      .io_pulldown_en_i(1'b0),
      .io(io_boot_mode)
  );

% if multichip_cfg["single_chip"] is False:
  // East side IOs
  logic east_test_being_requested_i;
  logic east_test_request_o;
  logic flow_control_east_rts_o;
  logic flow_control_east_cts_i;
  logic flow_control_east_rts_i;
  logic flow_control_east_cts_o;
  tc_digital_io east_test_being_requested_i_io (
      .data_i(1'b0),
      .data_o(east_test_being_requested_i),
      .io_direction_oe_ni(1'b1),
      .io_driving_strength_i(2'b00),
      .io_pullup_en_i(1'b0),
      .io_pulldown_en_i(1'b0),
      .io(io_east_test_being_requested)
  );
  tc_digital_io east_test_request_o_io (
      .data_i(east_test_request_o),
      .data_o(),
      .io_direction_oe_ni(1'b0),
      .io_driving_strength_i(2'b11),
      .io_pullup_en_i(1'b0),
      .io_pulldown_en_i(1'b0),
      .io(io_east_test_request)
  );
  tc_digital_io flow_control_east_rts_o_io (
      .data_i(flow_control_east_rts_o),
      .data_o(),
      .io_direction_oe_ni(1'b0),
      .io_driving_strength_i(2'b11),
      .io_pullup_en_i(1'b0),
      .io_pulldown_en_i(1'b0),
      .io(io_flow_control_east_rts_o)
  );
  tc_digital_io flow_control_east_cts_i_io (
      .data_i(1'b0),
      .data_o(flow_control_east_cts_i),
      .io_direction_oe_ni(1'b1),
      .io_driving_strength_i(2'b00),
      .io_pullup_en_i(1'b0),
      .io_pulldown_en_i(1'b0),
      .io(io_flow_control_east_cts_i)
  );
  tc_digital_io flow_control_east_rts_i_io (
      .data_i(1'b0),
      .data_o(flow_control_east_rts_i),
      .io_direction_oe_ni(1'b1),
      .io_driving_strength_i(2'b00),
      .io_pullup_en_i(1'b0),
      .io_pulldown_en_i(1'b0),
      .io(io_flow_control_east_rts_i)
  );
  tc_digital_io flow_control_east_cts_o_io (
      .data_i(flow_control_east_cts_o),
      .data_o(),
      .io_direction_oe_ni(1'b0),
      .io_driving_strength_i(2'b11),
      .io_pullup_en_i(1'b0),
      .io_pulldown_en_i(1'b0),
      .io(io_flow_control_east_cts_o)
  );

  generate
    for (genvar i = 0; i < D2DChannel; i++) begin : gen_east_d2d_io_cells
      tc_analog_io east_d2d_io[D2DWidth-1:0] (.io(io_east_d2d[i]));
    end
  endgenerate

  // West side IOs
  logic west_test_being_requested_i;
  logic west_test_request_o;
  logic flow_control_west_rts_o;
  logic flow_control_west_cts_i;
  logic flow_control_west_rts_i;
  logic flow_control_west_cts_o;
  tc_digital_io west_test_being_requested_i_io (
      .data_i(1'b0),
      .data_o(west_test_being_requested_i),
      .io_direction_oe_ni(1'b1),
      .io_driving_strength_i(2'b00),
      .io_pullup_en_i(1'b0),
      .io_pulldown_en_i(1'b0),
      .io(io_west_test_being_requested)
  );
  tc_digital_io west_test_request_o_io (
      .data_i(west_test_request_o),
      .data_o(),
      .io_direction_oe_ni(1'b0),
      .io_driving_strength_i(2'b11),
      .io_pullup_en_i(1'b0),
      .io_pulldown_en_i(1'b0),
      .io(io_west_test_request)
  );
  tc_digital_io flow_control_west_rts_o_io (
      .data_i(flow_control_west_rts_o),
      .data_o(),
      .io_direction_oe_ni(1'b0),
      .io_driving_strength_i(2'b11),
      .io_pullup_en_i(1'b0),
      .io_pulldown_en_i(1'b0),
      .io(io_flow_control_west_rts)
  );
  tc_digital_io flow_control_west_cts_i_io (
      .data_i(1'b0),
      .data_o(flow_control_west_cts_i),
      .io_direction_oe_ni(1'b1),
      .io_driving_strength_i(2'b00),
      .io_pullup_en_i(1'b0),
      .io_pulldown_en_i(1'b0),
      .io(io_flow_control_west_cts)
  );
  tc_digital_io flow_control_west_rts_i_io (
      .data_i(1'b0),
      .data_o(flow_control_west_rts_i),
      .io_direction_oe_ni(1'b1),
      .io_driving_strength_i(2'b00),
      .io_pullup_en_i(1'b0),
      .io_pulldown_en_i(1'b0),
      .io(io_flow_control_west_rts)
  );
  tc_digital_io flow_control_west_cts_o_io (
      .data_i(flow_control_west_cts_o),
      .data_o(),
      .io_direction_oe_ni(1'b0),
      .io_driving_strength_i(2'b11),
      .io_pullup_en_i(1'b0),
      .io_pulldown_en_i(1'b0),
      .io(io_flow_control_west_cts)
  );

  generate
    for (genvar i = 0; i < D2DChannel; i++) begin : gen_west_d2d_io_cells
      tc_analog_io west_d2d_io[D2DWidth-1:0] (.io(io_west_d2d[i]));
    end
  endgenerate

  // North side IOs
  logic north_test_being_requested_i;
  logic north_test_request_o;
  logic flow_control_north_rts_o;
  logic flow_control_north_cts_i;
  logic flow_control_north_rts_i;
  logic flow_control_north_cts_o;
  tc_digital_io north_test_being_requested_i_io (
      .data_i(1'b0),
      .data_o(north_test_being_requested_i),
      .io_direction_oe_ni(1'b1),
      .io_driving_strength_i(2'b00),
      .io_pullup_en_i(1'b0),
      .io_pulldown_en_i(1'b0),
      .io(io_north_test_being_requested)
  );
  tc_digital_io north_test_request_o_io (
      .data_i(north_test_request_o),
      .data_o(),
      .io_direction_oe_ni(1'b0),
      .io_driving_strength_i(2'b11),
      .io_pullup_en_i(1'b0),
      .io_pulldown_en_i(1'b0),
      .io(io_north_test_request)
  );
  tc_digital_io flow_control_north_rts_o_io (
      .data_i(flow_control_north_rts_o),
      .data_o(),
      .io_direction_oe_ni(1'b0),
      .io_driving_strength_i(2'b11),
      .io_pullup_en_i(1'b0),
      .io_pulldown_en_i(1'b0),
      .io(io_flow_control_north_rts)
  );
  tc_digital_io flow_control_north_cts_i_io (
      .data_i(1'b0),
      .data_o(flow_control_north_cts_i),
      .io_direction_oe_ni(1'b1),
      .io_driving_strength_i(2'b00),
      .io_pullup_en_i(1'b0),
      .io_pulldown_en_i(1'b0),
      .io(io_flow_control_north_cts)
  );
  tc_digital_io flow_control_north_rts_i_io (
      .data_i(1'b0),
      .data_o(flow_control_north_rts_i),
      .io_direction_oe_ni(1'b1),
      .io_driving_strength_i(2'b00),
      .io_pullup_en_i(1'b0),
      .io_pulldown_en_i(1'b0),
      .io(io_flow_control_north_rts)
  );
  tc_digital_io flow_control_north_cts_o_io (
      .data_i(flow_control_north_cts_o),
      .data_o(),
      .io_direction_oe_ni(1'b0),
      .io_driving_strength_i(2'b11),
      .io_pullup_en_i(1'b0),
      .io_pulldown_en_i(1'b0),
      .io(io_flow_control_north_cts)
  );

  generate
    for (genvar i = 0; i < D2DChannel; i++) begin : gen_north_d2d_io_cells
      tc_analog_io north_d2d_io[D2DWidth-1:0] (.io(io_north_d2d[i]));
    end
  endgenerate


  // South side IOs
  logic south_test_being_requested_i;
  logic south_test_request_o;
  logic flow_control_south_rts_o;
  logic flow_control_south_cts_i;
  logic flow_control_south_rts_i;
  logic flow_control_south_cts_o;
  tc_digital_io south_test_being_requested_i_io (
      .data_i(1'b0),
      .data_o(south_test_being_requested_i),
      .io_direction_oe_ni(1'b1),
      .io_driving_strength_i(2'b00),
      .io_pullup_en_i(1'b0),
      .io_pulldown_en_i(1'b0),
      .io(io_south_test_being_requested)
  );
  tc_digital_io south_test_request_o_io (
      .data_i(south_test_request_o),
      .data_o(),
      .io_direction_oe_ni(1'b0),
      .io_driving_strength_i(2'b11),
      .io_pullup_en_i(1'b0),
      .io_pulldown_en_i(1'b0),
      .io(io_south_test_request)
  );
  tc_digital_io flow_control_south_rts_o_io (
      .data_i(flow_control_south_rts_o),
      .data_o(),
      .io_direction_oe_ni(1'b0),
      .io_driving_strength_i(2'b11),
      .io_pullup_en_i(1'b0),
      .io_pulldown_en_i(1'b0),
      .io(io_flow_control_south_rts)
  );
  tc_digital_io flow_control_south_cts_i_io (
      .data_i(1'b0),
      .data_o(flow_control_south_cts_i),
      .io_direction_oe_ni(1'b1),
      .io_driving_strength_i(2'b00),
      .io_pullup_en_i(1'b0),
      .io_pulldown_en_i(1'b0),
      .io(io_flow_control_south_cts)
  );
  tc_digital_io flow_control_south_rts_i_io (
      .data_i(1'b0),
      .data_o(flow_control_south_rts_i),
      .io_direction_oe_ni(1'b1),
      .io_driving_strength_i(2'b00),
      .io_pullup_en_i(1'b0),
      .io_pulldown_en_i(1'b0),
      .io(io_flow_control_south_rts)
  );
  tc_digital_io flow_control_south_cts_o_io (
      .data_i(flow_control_south_cts_o),
      .data_o(),
      .io_direction_oe_ni(1'b0),
      .io_driving_strength_i(2'b11),
      .io_pullup_en_i(1'b0),
      .io_pulldown_en_i(1'b0),
      .io(io_flow_control_south_cts)
  );

  generate
    for (genvar i = 0; i < D2DChannel; i++) begin : gen_south_d2d_io_cells
      tc_analog_io south_d2d_io[D2DWidth-1:0] (.io(io_south_d2d[i]));
    end
  endgenerate
% endif

  logic uart_tx_o;
  logic uart_rx_i;
  logic uart_rts_no;
  logic uart_cts_ni;

  tc_digital_io uart_tx_o_io (
      .data_i(uart_tx_o),
      .data_o(),
      .io_direction_oe_ni(1'b0),
      .io_driving_strength_i(2'b11),
      .io_pullup_en_i(1'b0),
      .io_pulldown_en_i(1'b0),
      .io(io_uart_tx)
  );
  tc_digital_io uart_rx_i_io (
      .data_i(1'b0),
      .data_o(uart_rx_i),
      .io_direction_oe_ni(1'b1),
      .io_driving_strength_i(2'b00),
      .io_pullup_en_i(1'b0),
      .io_pulldown_en_i(1'b0),
      .io(io_uart_rx)
  );
  tc_digital_io uart_rts_no_io (
      .data_i(uart_rts_no),
      .data_o(),
      .io_direction_oe_ni(1'b0),
      .io_driving_strength_i(2'b11),
      .io_pullup_en_i(1'b0),
      .io_pulldown_en_i(1'b0),
      .io(io_uart_rts_n)
  );
  tc_digital_io uart_cts_ni_io (
      .data_i(1'b0),
      .data_o(uart_cts_ni),
      .io_direction_oe_ni(1'b1),
      .io_driving_strength_i(2'b00),
      .io_pullup_en_i(1'b0),
      .io_pulldown_en_i(1'b0),
      .io(io_uart_cts_n)
  );

  logic [NumGpio-1:0] gpio_d_i;
  logic [NumGpio-1:0] gpio_d_o;
  logic [NumGpio-1:0] gpio_oe_o;
  generate
    for (genvar i = 0; i < NumGpio; i++) begin : gen_gpio_io_cells
      tc_digital_io gpio_io (
          .data_i(gpio_d_o[i]),
          .data_o(gpio_d_i[i]),
          .io_direction_oe_ni(~gpio_oe_o[i]),
          .io_driving_strength_i(2'b11),
          .io_pullup_en_i(1'b0),
          .io_pulldown_en_i(1'b0),
          .io(io_gpio[i])
      );
    end
    for (genvar i = NumGpio; i < 8; i++) begin : gen_gpio_io_cells_nc
      assign gpio_d_i[i] = 1'b0;
    end
  endgenerate

  logic spis_sck_i;
  logic spis_csb_i;
  logic [3:0] spis_sd_o;
  logic [3:0] spis_sd_en_o;
  logic [3:0] spis_sd_i;
  tc_digital_io spis_sck_i_io (
      .data_i(1'b0),
      .data_o(spis_sck_i),
      .io_direction_oe_ni(1'b1),
      .io_driving_strength_i(2'b00),
      .io_pullup_en_i(1'b0),
      .io_pulldown_en_i(1'b0),
      .io(io_spis_sck)
  );
  tc_digital_io spis_csb_i_io (
      .data_i(1'b0),
      .data_o(spis_csb_i),
      .io_direction_oe_ni(1'b1),
      .io_driving_strength_i(2'b00),
      .io_pullup_en_i(1'b0),
      .io_pulldown_en_i(1'b0),
      .io(io_spis_csb)
  );
  tc_digital_io spis_sd_io[3:0] (
      .data_i(spis_sd_o),
      .data_o(spis_sd_i),
      .io_direction_oe_ni(~(spis_sd_en_o)),
      .io_driving_strength_i(2'b11),
      .io_pullup_en_i(1'b0),
      .io_pulldown_en_i(1'b0),
      .io(io_spis_sd)
  );

  logic spim_sck_o;
  logic spim_sck_en_o;
  logic spim_csb_o;
  logic spim_csb_en_o;
  logic [3:0] spim_sd_o;
  logic [3:0] spim_sd_en_o;
  logic [3:0] spim_sd_i;
  tc_digital_io spim_sck_o_io (
      .data_i(spim_sck_o),
      .data_o(),
      .io_direction_oe_ni(~spim_sck_en_o),
      .io_driving_strength_i(2'b11),
      .io_pullup_en_i(1'b0),
      .io_pulldown_en_i(1'b0),
      .io(io_spim_sck)
  );
  tc_digital_io spim_csb_o_io (
      .data_i(spim_csb_o),
      .data_o(),
      .io_direction_oe_ni(~spim_csb_en_o),
      .io_driving_strength_i(2'b11),
      .io_pullup_en_i(1'b0),
      .io_pulldown_en_i(1'b0),
      .io(io_spim_csb)
  );
  tc_digital_io spim_sd_io[3:0] (
      .data_i(spim_sd_o),
      .data_o(spim_sd_i),
      .io_direction_oe_ni(~(spim_sd_en_o)),
      .io_driving_strength_i(2'b11),
      .io_pullup_en_i(1'b0),
      .io_pulldown_en_i(1'b0),
      .io(io_spim_sd)
  );

  logic jtag_trst_ni;
  logic jtag_tck_i;
  logic jtag_tms_i;
  logic jtag_tdi_i;
  logic jtag_tdo_o;
  tc_digital_io jtag_trst_ni_io (
      .data_i(1'b0),
      .data_o(jtag_trst_ni),
      .io_direction_oe_ni(1'b1),
      .io_driving_strength_i(2'b00),
      .io_pullup_en_i(1'b0),
      .io_pulldown_en_i(1'b0),
      .io(io_jtag_trst_n)
  );
  tc_digital_io jtag_tck_i_io (
      .data_i(1'b0),
      .data_o(jtag_tck_i),
      .io_direction_oe_ni(1'b1),
      .io_driving_strength_i(2'b00),
      .io_pullup_en_i(1'b0),
      .io_pulldown_en_i(1'b0),
      .io(io_jtag_tck)
  );
  tc_digital_io jtag_tms_i_io (
      .data_i(1'b0),
      .data_o(jtag_tms_i),
      .io_direction_oe_ni(1'b1),
      .io_driving_strength_i(2'b00),
      .io_pullup_en_i(1'b0),
      .io_pulldown_en_i(1'b0),
      .io(io_jtag_tms)
  );
  tc_digital_io jtag_tdi_i_io (
      .data_i(1'b0),
      .data_o(jtag_tdi_i),
      .io_direction_oe_ni(1'b1),
      .io_driving_strength_i(2'b00),
      .io_pullup_en_i(1'b0),
      .io_pulldown_en_i(1'b0),
      .io(io_jtag_tdi)
  );
  tc_digital_io jtag_tdo_o_io (
      .data_i(jtag_tdo_o),
      .data_o(),
      .io_direction_oe_ni(1'b0),
      .io_driving_strength_i(2'b11),
      .io_pullup_en_i(1'b0),
      .io_pulldown_en_i(1'b0),
      .io(io_jtag_tdo)
  );

  logic i2c_sda_i;
  logic i2c_sda_o;
  logic i2c_sda_en_o;
  logic i2c_scl_i;
  logic i2c_scl_o;
  logic i2c_scl_en_o;
  tc_digital_io i2c_sda_io (
      .data_i(i2c_sda_o),
      .data_o(i2c_sda_i),
      .io_direction_oe_ni(~i2c_sda_en_o),
      .io_driving_strength_i(2'b11),
      .io_pullup_en_i(1'b0),
      .io_pulldown_en_i(1'b0),
      .io(io_i2c_sda)
  );
  tc_digital_io i2c_scl_io (
      .data_i(i2c_scl_o),
      .data_o(i2c_scl_i),
      .io_direction_oe_ni(~i2c_scl_en_o),
      .io_driving_strength_i(2'b11),
      .io_pullup_en_i(1'b0),
      .io_pulldown_en_i(1'b0),
      .io(io_i2c_scl)
  );

  occamy_chip i_occamy_chip (.*);

endmodule
