// Copyright 2025 KU Leuven.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51

// Author: Yunhao Deng <yunhao.deng@kuleuven.be>

module hemaia (
    // Clocks, Boot, ChipId (14)
    inout logic        io_clk_i,
    inout logic        io_rst_ni,
    inout logic        io_clk_periph_i,
    inout logic        io_rst_periph_ni,
    inout logic        io_test_mode_i,
    inout logic [ 7:0] io_chip_id_i,
    inout logic        io_boot_mode_i,
% if multichip_cfg["single_chip"] is False:
    // East side (66)
    inout logic        io_east_test_being_requested_i,
    inout logic        io_east_test_request_o,
    inout logic        io_flow_control_east_rts_o,
    inout logic        io_flow_control_east_cts_i,
    inout logic        io_flow_control_east_rts_i,
    inout logic        io_flow_control_east_cts_o,
    inout logic [19:0] io_east_d2d                  [3],
    // West side (66)
    inout logic        io_west_test_being_requested_i,
    inout logic        io_west_test_request_o,
    inout logic        io_flow_control_west_rts_o,
    inout logic        io_flow_control_west_cts_i,
    inout logic        io_flow_control_west_rts_i,
    inout logic        io_flow_control_west_cts_o,
    inout logic [19:0] io_west_d2d                  [3],
    // North side (66)
    inout logic        io_north_test_being_requested_i,
    inout logic        io_north_test_request_o,
    inout logic        io_flow_control_north_rts_o,
    inout logic        io_flow_control_north_cts_i,
    inout logic        io_flow_control_north_rts_i,
    inout logic        io_flow_control_north_cts_o,
    inout logic [19:0] io_north_d2d                 [3],
    // South side (66)
    inout logic        io_south_test_being_requested_i,
    inout logic        io_south_test_request_o,
    inout logic        io_flow_control_south_rts_o,
    inout logic        io_flow_control_south_cts_i,
    inout logic        io_flow_control_south_rts_i,
    inout logic        io_flow_control_south_cts_o,
    inout logic [19:0] io_south_d2d                 [3],
% endif
    // `uart` Interface (4)
    inout logic        io_uart_tx_o,
    inout logic        io_uart_rx_i,
    inout logic        io_uart_rts_no,
    inout logic        io_uart_cts_ni,
    // `gpio` Interface (8)
    inout logic [ 7:0] io_gpio,
    // `SPI Slave` for Debugging Purposes (6)
    inout logic        io_spis_sck_i,
    inout logic        io_spis_csb_i,
    inout logic [ 3:0] io_spis_sd,
    // SPI Master Interface (6)
    inout logic        io_spim_sck_o,
    inout logic        io_spim_csb_o,
    inout logic [ 3:0] io_spim_sd,
    // `jtag` Interface (5)
    inout logic        io_jtag_trst_ni,
    inout logic        io_jtag_tck_i,
    inout logic        io_jtag_tms_i,
    inout logic        io_jtag_tdi_i,
    inout logic        io_jtag_tdo_o,
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
      .io_driving_strength_i(4'h0),
      .io_pullup_en_i(1'b0),
      .io_pulldown_en_i(1'b0),
      .io(io_clk_i)
  );

  logic rst_ni;
  tc_digital_io rst_ni_io (
      .data_i(1'b1),
      .data_o(rst_ni),
      .io_direction_oe_ni(1'b1),
      .io_driving_strength_i(4'h0),
      .io_pullup_en_i(1'b0),
      .io_pulldown_en_i(1'b0),
      .io(io_rst_ni)
  );

  logic clk_periph_i;
  tc_digital_io clk_periph_i_io (
      .data_i(1'b0),
      .data_o(clk_periph_i),
      .io_direction_oe_ni(1'b1),
      .io_driving_strength_i(4'h0),
      .io_pullup_en_i(1'b0),
      .io_pulldown_en_i(1'b0),
      .io(io_clk_periph_i)
  );

  logic rst_periph_ni;
  tc_digital_io rst_periph_ni_io (
      .data_i(1'b1),
      .data_o(rst_periph_ni),
      .io_direction_oe_ni(1'b1),
      .io_driving_strength_i(4'h0),
      .io_pullup_en_i(1'b0),
      .io_pulldown_en_i(1'b0),
      .io(io_rst_periph_ni)
  );

  logic test_mode_i;
  tc_digital_io test_mode_i_io (
      .data_i(1'b0),
      .data_o(test_mode_i),
      .io_direction_oe_ni(1'b1),
      .io_driving_strength_i(4'h0),
      .io_pullup_en_i(1'b0),
      .io_pulldown_en_i(1'b0),
      .io(io_test_mode_i)
  );

  logic [7:0] chip_id_i;
  tc_digital_io chip_id_i_io[7:0] (
      .data_i(1'b0),
      .data_o(chip_id_i),
      .io_direction_oe_ni(1'b1),
      .io_driving_strength_i(4'h0),
      .io_pullup_en_i(1'b0),
      .io_pulldown_en_i(1'b0),
      .io(io_chip_id_i)
  );

  logic [1:0] boot_mode_i;
  tc_digital_io boot_mode_i_io[1:0] (
      .data_i(1'b0),
      .data_o(boot_mode_i),
      .io_direction_oe_ni(1'b1),
      .io_driving_strength_i(4'h0),
      .io_pullup_en_i(1'b0),
      .io_pulldown_en_i(1'b0),
      .io(io_boot_mode_i)
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
      .io_driving_strength_i(4'h0),
      .io_pullup_en_i(1'b0),
      .io_pulldown_en_i(1'b0),
      .io(io_east_test_being_requested_i)
  );
  tc_digital_io east_test_request_o_io (
      .data_i(east_test_request_o),
      .data_o(),
      .io_direction_oe_ni(1'b0),
      .io_driving_strength_i(4'hf),
      .io_pullup_en_i(1'b0),
      .io_pulldown_en_i(1'b0),
      .io(io_east_test_request_o)
  );
  tc_digital_io flow_control_east_rts_o_io (
      .data_i(flow_control_east_rts_o),
      .data_o(),
      .io_direction_oe_ni(1'b0),
      .io_driving_strength_i(4'hf),
      .io_pullup_en_i(1'b0),
      .io_pulldown_en_i(1'b0),
      .io(io_flow_control_east_rts_o)
  );
  tc_digital_io flow_control_east_cts_i_io (
      .data_i(1'b0),
      .data_o(flow_control_east_cts_i),
      .io_direction_oe_ni(1'b1),
      .io_driving_strength_i(4'h0),
      .io_pullup_en_i(1'b0),
      .io_pulldown_en_i(1'b0),
      .io(io_flow_control_east_cts_i)
  );
  tc_digital_io flow_control_east_rts_i_io (
      .data_i(1'b0),
      .data_o(flow_control_east_rts_i),
      .io_direction_oe_ni(1'b1),
      .io_driving_strength_i(4'h0),
      .io_pullup_en_i(1'b0),
      .io_pulldown_en_i(1'b0),
      .io(io_flow_control_east_rts_i)
  );
  tc_digital_io flow_control_east_cts_o_io (
      .data_i(flow_control_east_cts_o),
      .data_o(),
      .io_direction_oe_ni(1'b0),
      .io_driving_strength_i(4'hf),
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
      .io_driving_strength_i(4'h0),
      .io_pullup_en_i(1'b0),
      .io_pulldown_en_i(1'b0),
      .io(io_west_test_being_requested_i)
  );
  tc_digital_io west_test_request_o_io (
      .data_i(west_test_request_o),
      .data_o(),
      .io_direction_oe_ni(1'b0),
      .io_driving_strength_i(4'hf),
      .io_pullup_en_i(1'b0),
      .io_pulldown_en_i(1'b0),
      .io(io_west_test_request_o)
  );
  tc_digital_io flow_control_west_rts_o_io (
      .data_i(flow_control_west_rts_o),
      .data_o(),
      .io_direction_oe_ni(1'b0),
      .io_driving_strength_i(4'hf),
      .io_pullup_en_i(1'b0),
      .io_pulldown_en_i(1'b0),
      .io(io_flow_control_west_rts_o)
  );
  tc_digital_io flow_control_west_cts_i_io (
      .data_i(1'b0),
      .data_o(flow_control_west_cts_i),
      .io_direction_oe_ni(1'b1),
      .io_driving_strength_i(4'h0),
      .io_pullup_en_i(1'b0),
      .io_pulldown_en_i(1'b0),
      .io(io_flow_control_west_cts_i)
  );
  tc_digital_io flow_control_west_rts_i_io (
      .data_i(1'b0),
      .data_o(flow_control_west_rts_i),
      .io_direction_oe_ni(1'b1),
      .io_driving_strength_i(4'h0),
      .io_pullup_en_i(1'b0),
      .io_pulldown_en_i(1'b0),
      .io(io_flow_control_west_rts_i)
  );
  tc_digital_io flow_control_west_cts_o_io (
      .data_i(flow_control_west_cts_o),
      .data_o(),
      .io_direction_oe_ni(1'b0),
      .io_driving_strength_i(4'hf),
      .io_pullup_en_i(1'b0),
      .io_pulldown_en_i(1'b0),
      .io(io_flow_control_west_cts_o)
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
      .io_driving_strength_i(4'h0),
      .io_pullup_en_i(1'b0),
      .io_pulldown_en_i(1'b0),
      .io(io_north_test_being_requested_i)
  );
  tc_digital_io north_test_request_o_io (
      .data_i(north_test_request_o),
      .data_o(),
      .io_direction_oe_ni(1'b0),
      .io_driving_strength_i(4'hf),
      .io_pullup_en_i(1'b0),
      .io_pulldown_en_i(1'b0),
      .io(io_north_test_request_o)
  );
  tc_digital_io flow_control_north_rts_o_io (
      .data_i(flow_control_north_rts_o),
      .data_o(),
      .io_direction_oe_ni(1'b0),
      .io_driving_strength_i(4'hf),
      .io_pullup_en_i(1'b0),
      .io_pulldown_en_i(1'b0),
      .io(io_flow_control_north_rts_o)
  );
  tc_digital_io flow_control_north_cts_i_io (
      .data_i(1'b0),
      .data_o(flow_control_north_cts_i),
      .io_direction_oe_ni(1'b1),
      .io_driving_strength_i(4'h0),
      .io_pullup_en_i(1'b0),
      .io_pulldown_en_i(1'b0),
      .io(io_flow_control_north_cts_i)
  );
  tc_digital_io flow_control_north_rts_i_io (
      .data_i(1'b0),
      .data_o(flow_control_north_rts_i),
      .io_direction_oe_ni(1'b1),
      .io_driving_strength_i(4'h0),
      .io_pullup_en_i(1'b0),
      .io_pulldown_en_i(1'b0),
      .io(io_flow_control_north_rts_i)
  );
  tc_digital_io flow_control_north_cts_o_io (
      .data_i(flow_control_north_cts_o),
      .data_o(),
      .io_direction_oe_ni(1'b0),
      .io_driving_strength_i(4'hf),
      .io_pullup_en_i(1'b0),
      .io_pulldown_en_i(1'b0),
      .io(io_flow_control_north_cts_o)
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
      .io_driving_strength_i(4'h0),
      .io_pullup_en_i(1'b0),
      .io_pulldown_en_i(1'b0),
      .io(io_south_test_being_requested_i)
  );
  tc_digital_io south_test_request_o_io (
      .data_i(south_test_request_o),
      .data_o(),
      .io_direction_oe_ni(1'b0),
      .io_driving_strength_i(4'hf),
      .io_pullup_en_i(1'b0),
      .io_pulldown_en_i(1'b0),
      .io(io_south_test_request_o)
  );
  tc_digital_io flow_control_south_rts_o_io (
      .data_i(flow_control_south_rts_o),
      .data_o(),
      .io_direction_oe_ni(1'b0),
      .io_driving_strength_i(4'hf),
      .io_pullup_en_i(1'b0),
      .io_pulldown_en_i(1'b0),
      .io(io_flow_control_south_rts_o)
  );
  tc_digital_io flow_control_south_cts_i_io (
      .data_i(1'b0),
      .data_o(flow_control_south_cts_i),
      .io_direction_oe_ni(1'b1),
      .io_driving_strength_i(4'h0),
      .io_pullup_en_i(1'b0),
      .io_pulldown_en_i(1'b0),
      .io(io_flow_control_south_cts_i)
  );
  tc_digital_io flow_control_south_rts_i_io (
      .data_i(1'b0),
      .data_o(flow_control_south_rts_i),
      .io_direction_oe_ni(1'b1),
      .io_driving_strength_i(4'h0),
      .io_pullup_en_i(1'b0),
      .io_pulldown_en_i(1'b0),
      .io(io_flow_control_south_rts_i)
  );
  tc_digital_io flow_control_south_cts_o_io (
      .data_i(flow_control_south_cts_o),
      .data_o(),
      .io_direction_oe_ni(1'b0),
      .io_driving_strength_i(4'hf),
      .io_pullup_en_i(1'b0),
      .io_pulldown_en_i(1'b0),
      .io(io_flow_control_south_cts_o)
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
      .io_driving_strength_i(4'hf),
      .io_pullup_en_i(1'b0),
      .io_pulldown_en_i(1'b0),
      .io(io_uart_tx_o)
  );
  tc_digital_io uart_rx_i_io (
      .data_i(1'b0),
      .data_o(uart_rx_i),
      .io_direction_oe_ni(1'b1),
      .io_driving_strength_i(4'h0),
      .io_pullup_en_i(1'b0),
      .io_pulldown_en_i(1'b0),
      .io(io_uart_rx_i)
  );
  tc_digital_io uart_rts_no_io (
      .data_i(uart_rts_no),
      .data_o(),
      .io_direction_oe_ni(1'b0),
      .io_driving_strength_i(4'hf),
      .io_pullup_en_i(1'b0),
      .io_pulldown_en_i(1'b0),
      .io(io_uart_rts_no)
  );
  tc_digital_io uart_cts_ni_io (
      .data_i(1'b0),
      .data_o(uart_cts_ni),
      .io_direction_oe_ni(1'b1),
      .io_driving_strength_i(4'h0),
      .io_pullup_en_i(1'b0),
      .io_pulldown_en_i(1'b0),
      .io(io_uart_cts_ni)
  );

  logic [31:0] gpio_d_i;
  logic [31:0] gpio_d_o;
  logic [31:0] gpio_oe_o;
  generate
    for (genvar i = 0; i < NumGpio; i++) begin : gen_gpio_io_cells
      tc_digital_io gpio_io (
          .data_i(gpio_d_o[i]),
          .data_o(gpio_d_i[i]),
          .io_direction_oe_ni(~gpio_oe_o[i]),
          .io_driving_strength_i(4'hf),
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
      .io_driving_strength_i(4'h0),
      .io_pullup_en_i(1'b0),
      .io_pulldown_en_i(1'b0),
      .io(io_spis_sck_i)
  );
  tc_digital_io spis_csb_i_io (
      .data_i(1'b0),
      .data_o(spis_csb_i),
      .io_direction_oe_ni(1'b1),
      .io_driving_strength_i(4'h0),
      .io_pullup_en_i(1'b0),
      .io_pulldown_en_i(1'b0),
      .io(io_spis_csb_i)
  );
  tc_digital_io spis_sd_io[3:0] (
      .data_i(spis_sd_o),
      .data_o(spis_sd_i),
      .io_direction_oe_ni(~(spis_sd_en_o)),
      .io_driving_strength_i(4'hf),
      .io_pullup_en_i(1'b0),
      .io_pulldown_en_i(1'b0),
      .io(io_spis_sd)
  );

  logic spim_sck_o;
  logic spim_sck_en_o;
  logic [1:0] spim_csb_o;
  logic [1:0] spim_csb_en_o;
  logic [3:0] spim_sd_o;
  logic [3:0] spim_sd_en_o;
  logic [3:0] spim_sd_i;
  tc_digital_io spim_sck_o_io (
      .data_i(spim_sck_o),
      .data_o(),
      .io_direction_oe_ni(~spim_sck_en_o),
      .io_driving_strength_i(4'hf),
      .io_pullup_en_i(1'b0),
      .io_pulldown_en_i(1'b0),
      .io(io_spim_sck_o)
  );
  tc_digital_io spim_csb_o_io (
      .data_i(spim_csb_o[0]),
      .data_o(),
      .io_direction_oe_ni(~(spim_csb_en_o[0])),
      .io_driving_strength_i(4'hf),
      .io_pullup_en_i(1'b0),
      .io_pulldown_en_i(1'b0),
      .io(io_spim_csb_o)
  );
  tc_digital_io spim_sd_io[3:0] (
      .data_i(spim_sd_o),
      .data_o(spim_sd_i),
      .io_direction_oe_ni(~(spim_sd_en_o)),
      .io_driving_strength_i(4'hf),
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
      .io_driving_strength_i(4'h0),
      .io_pullup_en_i(1'b0),
      .io_pulldown_en_i(1'b0),
      .io(io_jtag_trst_ni)
  );
  tc_digital_io jtag_tck_i_io (
      .data_i(1'b0),
      .data_o(jtag_tck_i),
      .io_direction_oe_ni(1'b1),
      .io_driving_strength_i(4'h0),
      .io_pullup_en_i(1'b0),
      .io_pulldown_en_i(1'b0),
      .io(io_jtag_tck_i)
  );
  tc_digital_io jtag_tms_i_io (
      .data_i(1'b0),
      .data_o(jtag_tms_i),
      .io_direction_oe_ni(1'b1),
      .io_driving_strength_i(4'h0),
      .io_pullup_en_i(1'b0),
      .io_pulldown_en_i(1'b0),
      .io(io_jtag_tms_i)
  );
  tc_digital_io jtag_tdi_i_io (
      .data_i(1'b0),
      .data_o(jtag_tdi_i),
      .io_direction_oe_ni(1'b1),
      .io_driving_strength_i(4'h0),
      .io_pullup_en_i(1'b0),
      .io_pulldown_en_i(1'b0),
      .io(io_jtag_tdi_i)
  );
  tc_digital_io jtag_tdo_o_io (
      .data_i(jtag_tdo_o),
      .data_o(),
      .io_direction_oe_ni(1'b0),
      .io_driving_strength_i(4'hf),
      .io_pullup_en_i(1'b0),
      .io_pulldown_en_i(1'b0),
      .io(io_jtag_tdo_o)
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
      .io_driving_strength_i(4'hf),
      .io_pullup_en_i(1'b0),
      .io_pulldown_en_i(1'b0),
      .io(io_i2c_sda)
  );
  tc_digital_io i2c_scl_io (
      .data_i(i2c_scl_o),
      .data_o(i2c_scl_i),
      .io_direction_oe_ni(~i2c_scl_en_o),
      .io_driving_strength_i(4'hf),
      .io_pullup_en_i(1'b0),
      .io_pulldown_en_i(1'b0),
      .io(io_i2c_scl)
  );

  occamy_chip i_occamy_chip (
    .east_d2d_io         (io_east_d2d        ),
    .west_d2d_io         (io_west_d2d        ),
    .north_d2d_io        (io_north_d2d       ),
    .south_d2d_io        (io_south_d2d       ),
    .*
    );

endmodule
