// Copyright 2025 KU Leuven.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51

// Author: Yunhao Deng <yunhao.deng@kuleuven.be>
<% 
  spi_slave_present = any(periph["name"] == "spis" for periph in occamy_cfg["peripherals"]["axi_lite_peripherals"])
%>
<% 
  spi_master_present = any(periph["name"] == "spim" for periph in occamy_cfg["peripherals"]["axi_lite_narrow_peripherals"])
%>
<% 
  i2c_present = any(periph["name"] == "i2c" for periph in occamy_cfg["peripherals"]["axi_lite_narrow_peripherals"])
%>


module hemaia (
    // Clocks, Boot, ChipId (14)
    inout wire        io_clk_i,
    inout wire        io_rst_ni,
    inout wire        io_bypass_pll_division_i,
    inout wire        io_clk_obs_o,
    inout wire        io_clk_periph_i,
    inout wire        io_rst_periph_ni,
    inout wire        io_test_mode_i,
    inout wire [ 7:0] io_chip_id_i,
    inout wire        io_boot_mode_i,
% if multichip_cfg["single_chip"] is False:
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
% endif
    // `uart` Interface (4)
    inout wire        io_uart_tx_o,
    inout wire        io_uart_rx_i,
    inout wire        io_uart_rts_no,
    inout wire        io_uart_cts_ni,
    // `gpio` Interface (8)
    inout wire [ 3:0] io_gpio,
% if spi_slave_present: 
    // `SPI Slave` for Debugging Purposes (6)
    inout wire        io_spis_sck_i,
    inout wire        io_spis_csb_i,
    inout wire [ 3:0] io_spis_sd,
% endif
% if spi_master_present:
    // SPI Master Interface (6)
    inout wire        io_spim_sck_o,
    inout wire        io_spim_csb_o,
    inout wire [ 3:0] io_spim_sd,
% endif
% if i2c_present:
    // I2C Interface (2)
    inout wire        io_i2c_sda,
    inout wire        io_i2c_scl,
% endif
    // `jtag` Interface (5)
    inout wire        io_jtag_trst_ni,
    inout wire        io_jtag_tck_i,
    inout wire        io_jtag_tms_i,
    inout wire        io_jtag_tdi_i,
    inout wire        io_jtag_tdo_o
);

  localparam int NumGpio = 4;
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

  logic bypass_pll_division_i;
  tc_digital_io bypass_pll_division_i_io (
      .data_i(1'b0),
      .data_o(bypass_pll_division_i),
      .io_direction_oe_ni(1'b1),
      .io_driving_strength_i(4'h0),
      .io_pullup_en_i(1'b0),
      .io_pulldown_en_i(1'b0),
      .io(io_bypass_pll_division_i)
  );

  logic clk_obs_o;
  tc_digital_io #(
    .VerticalIO(1'b1)
  ) clk_obs_o_io (
      .data_i(clk_obs_o),
      .data_o(),
      .io_direction_oe_ni(1'b0),
      .io_driving_strength_i(4'hf),
      .io_pullup_en_i(1'b0),
      .io_pulldown_en_i(1'b0),
      .io(io_clk_obs_o)
  );

  logic clk_periph_i;
  tc_digital_io #(
    .VerticalIO(1'b1)
  ) clk_periph_i_io (
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
  tc_digital_io chip_id_i_io_0 (
      .data_i(1'b0),
      .data_o(chip_id_i[0]),
      .io_direction_oe_ni(1'b1),
      .io_driving_strength_i(4'h0),
      .io_pullup_en_i(1'b0),
      .io_pulldown_en_i(1'b0),
      .io(io_chip_id_i[0])
  );

  tc_digital_io #(
    .VerticalIO(1'b1)
  ) chip_id_i_io_1 (
      .data_i(1'b0),
      .data_o(chip_id_i[1]),
      .io_direction_oe_ni(1'b1),
      .io_driving_strength_i(4'h0),
      .io_pullup_en_i(1'b0),
      .io_pulldown_en_i(1'b0),
      .io(io_chip_id_i[1])
  );

  tc_digital_io #(
    .VerticalIO(1'b1)
  ) chip_id_i_io_2 (
      .data_i(1'b0),
      .data_o(chip_id_i[2]),
      .io_direction_oe_ni(1'b1),
      .io_driving_strength_i(4'h0),
      .io_pullup_en_i(1'b0),
      .io_pulldown_en_i(1'b0),
      .io(io_chip_id_i[2])
  );

  tc_digital_io #(
    .VerticalIO(1'b1)
  ) chip_id_i_io_3 (
      .data_i(1'b0),
      .data_o(chip_id_i[3]),
      .io_direction_oe_ni(1'b1),
      .io_driving_strength_i(4'h0),
      .io_pullup_en_i(1'b0),
      .io_pulldown_en_i(1'b0),
      .io(io_chip_id_i[3])
  );

  tc_digital_io #(
    .VerticalIO(1'b1)
  ) chip_id_i_io_4 (
      .data_i(1'b0),
      .data_o(chip_id_i[4]),
      .io_direction_oe_ni(1'b1),
      .io_driving_strength_i(4'h0),
      .io_pullup_en_i(1'b0),
      .io_pulldown_en_i(1'b0),
      .io(io_chip_id_i[4])
  );

  tc_digital_io #(
    .VerticalIO(1'b1)
  ) chip_id_i_io_5 (
      .data_i(1'b0),
      .data_o(chip_id_i[5]),
      .io_direction_oe_ni(1'b1),
      .io_driving_strength_i(4'h0),
      .io_pullup_en_i(1'b0),
      .io_pulldown_en_i(1'b0),
      .io(io_chip_id_i[5])
  );

  tc_digital_io #(
    .VerticalIO(1'b1)
  ) chip_id_i_io_6 (
      .data_i(1'b0),
      .data_o(chip_id_i[6]),
      .io_direction_oe_ni(1'b1),
      .io_driving_strength_i(4'h0),
      .io_pullup_en_i(1'b0),
      .io_pulldown_en_i(1'b0),
      .io(io_chip_id_i[6])
  );

  tc_digital_io #(
    .VerticalIO(1'b1)
  ) chip_id_i_io_7 (
      .data_i(1'b0),
      .data_o(chip_id_i[7]),
      .io_direction_oe_ni(1'b1),
      .io_driving_strength_i(4'h0),
      .io_pullup_en_i(1'b0),
      .io_pulldown_en_i(1'b0),
      .io(io_chip_id_i[7])
  );

  logic [1:0] boot_mode_i;
  assign boot_mode_i[1] = 1'b0;
  tc_digital_io boot_mode_i_io (
      .data_i(1'b0),
      .data_o(boot_mode_i[0]),
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
  tc_digital_io #(
    .VerticalIO(1'b1)
  ) flow_control_east_cts_o_io (
      .data_i(flow_control_east_cts_o),
      .data_o(),
      .io_direction_oe_ni(1'b0),
      .io_driving_strength_i(4'hf),
      .io_pullup_en_i(1'b0),
      .io_pulldown_en_i(1'b0),
      .io(io_flow_control_east_cts_o)
  );

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
  tc_digital_io #(
    .VerticalIO(1'b1)
  ) west_test_request_o_io (
      .data_i(west_test_request_o),
      .data_o(),
      .io_direction_oe_ni(1'b0),
      .io_driving_strength_i(4'hf),
      .io_pullup_en_i(1'b0),
      .io_pulldown_en_i(1'b0),
      .io(io_west_test_request_o)
  );
  tc_digital_io #(
    .VerticalIO(1'b1)
  ) flow_control_west_rts_o_io (
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

  // North side IOs
  logic north_test_being_requested_i;
  logic north_test_request_o;
  logic flow_control_north_rts_o;
  logic flow_control_north_cts_i;
  logic flow_control_north_rts_i;
  logic flow_control_north_cts_o;
  tc_digital_io #(
    .VerticalIO(1'b1)
  ) north_test_being_requested_i_io (
      .data_i(1'b0),
      .data_o(north_test_being_requested_i),
      .io_direction_oe_ni(1'b1),
      .io_driving_strength_i(4'h0),
      .io_pullup_en_i(1'b0),
      .io_pulldown_en_i(1'b0),
      .io(io_north_test_being_requested_i)
  );
  tc_digital_io #(
    .VerticalIO(1'b1)
  ) north_test_request_o_io (
      .data_i(north_test_request_o),
      .data_o(),
      .io_direction_oe_ni(1'b0),
      .io_driving_strength_i(4'hf),
      .io_pullup_en_i(1'b0),
      .io_pulldown_en_i(1'b0),
      .io(io_north_test_request_o)
  );
  tc_digital_io #(
    .VerticalIO(1'b1)
  ) flow_control_north_rts_o_io (
      .data_i(flow_control_north_rts_o),
      .data_o(),
      .io_direction_oe_ni(1'b0),
      .io_driving_strength_i(4'hf),
      .io_pullup_en_i(1'b0),
      .io_pulldown_en_i(1'b0),
      .io(io_flow_control_north_rts_o)
  );
  tc_digital_io #(
    .VerticalIO(1'b1)
  ) flow_control_north_cts_i_io (
      .data_i(1'b0),
      .data_o(flow_control_north_cts_i),
      .io_direction_oe_ni(1'b1),
      .io_driving_strength_i(4'h0),
      .io_pullup_en_i(1'b0),
      .io_pulldown_en_i(1'b0),
      .io(io_flow_control_north_cts_i)
  );
  tc_digital_io #(
    .VerticalIO(1'b1)
  ) flow_control_north_rts_i_io (
      .data_i(1'b0),
      .data_o(flow_control_north_rts_i),
      .io_direction_oe_ni(1'b1),
      .io_driving_strength_i(4'h0),
      .io_pullup_en_i(1'b0),
      .io_pulldown_en_i(1'b0),
      .io(io_flow_control_north_rts_i)
  );
  tc_digital_io #(
    .VerticalIO(1'b1)
  ) flow_control_north_cts_o_io (
      .data_i(flow_control_north_cts_o),
      .data_o(),
      .io_direction_oe_ni(1'b0),
      .io_driving_strength_i(4'hf),
      .io_pullup_en_i(1'b0),
      .io_pulldown_en_i(1'b0),
      .io(io_flow_control_north_cts_o)
  );

  // South side IOs
  logic south_test_being_requested_i;
  logic south_test_request_o;
  logic flow_control_south_rts_o;
  logic flow_control_south_cts_i;
  logic flow_control_south_rts_i;
  logic flow_control_south_cts_o;
  tc_digital_io #(
    .VerticalIO(1'b1)
  ) south_test_being_requested_i_io (
      .data_i(1'b0),
      .data_o(south_test_being_requested_i),
      .io_direction_oe_ni(1'b1),
      .io_driving_strength_i(4'h0),
      .io_pullup_en_i(1'b0),
      .io_pulldown_en_i(1'b0),
      .io(io_south_test_being_requested_i)
  );
  tc_digital_io #(
    .VerticalIO(1'b1)
  ) south_test_request_o_io (
      .data_i(south_test_request_o),
      .data_o(),
      .io_direction_oe_ni(1'b0),
      .io_driving_strength_i(4'hf),
      .io_pullup_en_i(1'b0),
      .io_pulldown_en_i(1'b0),
      .io(io_south_test_request_o)
  );
  tc_digital_io #(
    .VerticalIO(1'b1)
  ) flow_control_south_rts_o_io (
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
  tc_digital_io #(
    .VerticalIO(1'b1)
  ) flow_control_south_rts_i_io (
      .data_i(1'b0),
      .data_o(flow_control_south_rts_i),
      .io_direction_oe_ni(1'b1),
      .io_driving_strength_i(4'h0),
      .io_pullup_en_i(1'b0),
      .io_pulldown_en_i(1'b0),
      .io(io_flow_control_south_rts_i)
  );
  tc_digital_io #(
    .VerticalIO(1'b1)
  ) flow_control_south_cts_o_io (
      .data_i(flow_control_south_cts_o),
      .data_o(),
      .io_direction_oe_ni(1'b0),
      .io_driving_strength_i(4'hf),
      .io_pullup_en_i(1'b0),
      .io_pulldown_en_i(1'b0),
      .io(io_flow_control_south_cts_o)
  );
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
  tc_digital_io #(
    .VerticalIO(1'b1)
  ) uart_cts_ni_io (
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
    for (genvar i = NumGpio; i < 32; i++) begin : gen_gpio_io_cells_nc
      assign gpio_d_i[i] = 1'b0;
    end
  endgenerate

% if spi_slave_present:
  logic spis_sck_i;
  logic spis_csb_i;
  logic [3:0] spis_sd_o;
  logic [3:0] spis_sd_en_o;
  logic [3:0] spis_sd_i;
  tc_digital_io #(
    .VerticalIO(1'b1)
  ) spis_sck_i_io (
      .data_i(1'b0),
      .data_o(spis_sck_i),
      .io_direction_oe_ni(1'b1),
      .io_driving_strength_i(4'h0),
      .io_pullup_en_i(1'b0),
      .io_pulldown_en_i(1'b0),
      .io(io_spis_sck_i)
  );
  tc_digital_io #(
    .VerticalIO(1'b1)
  ) spis_csb_i_io (
      .data_i(1'b0),
      .data_o(spis_csb_i),
      .io_direction_oe_ni(1'b1),
      .io_driving_strength_i(4'h0),
      .io_pullup_en_i(1'b0),
      .io_pulldown_en_i(1'b0),
      .io(io_spis_csb_i)
  );
  tc_digital_io #(
    .VerticalIO(1'b1)
  ) spis_sd_io[3:0] (
      .data_i(spis_sd_o),
      .data_o(spis_sd_i),
      .io_direction_oe_ni(~(spis_sd_en_o)),
      .io_driving_strength_i(4'hf),
      .io_pullup_en_i(1'b0),
      .io_pulldown_en_i(1'b0),
      .io(io_spis_sd)
  );
% endif
% if spi_master_present:
  logic spim_sck_o;
  logic spim_sck_en_o;
  logic [1:0] spim_csb_o;
  logic [1:0] spim_csb_en_o;
  logic [3:0] spim_sd_o;
  logic [3:0] spim_sd_en_o;
  logic [3:0] spim_sd_i;
  tc_digital_io #(
    .VerticalIO(1'b1)
  ) spim_sck_o_io (
      .data_i(spim_sck_o),
      .data_o(),
      .io_direction_oe_ni(~spim_sck_en_o),
      .io_driving_strength_i(4'hf),
      .io_pullup_en_i(1'b0),
      .io_pulldown_en_i(1'b0),
      .io(io_spim_sck_o)
  );
  tc_digital_io #(
    .VerticalIO(1'b1)
  ) spim_csb_o_io (
      .data_i(spim_csb_o[0]),
      .data_o(),
      .io_direction_oe_ni(~(spim_csb_en_o[0])),
      .io_driving_strength_i(4'hf),
      .io_pullup_en_i(1'b0),
      .io_pulldown_en_i(1'b0),
      .io(io_spim_csb_o)
  );
  tc_digital_io #(
    .VerticalIO(1'b1)
  ) spim_sd_io[3:0] (
      .data_i(spim_sd_o),
      .data_o(spim_sd_i),
      .io_direction_oe_ni(~(spim_sd_en_o)),
      .io_driving_strength_i(4'hf),
      .io_pullup_en_i(1'b0),
      .io_pulldown_en_i(1'b0),
      .io(io_spim_sd)
  );
% endif
  logic jtag_trst_ni;
  logic jtag_tck_i;
  logic jtag_tms_i;
  logic jtag_tdi_i;
  logic jtag_tdo_o;
  tc_digital_io #(
    .VerticalIO(1'b1)
  ) jtag_trst_ni_io (
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

% if i2c_present:
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
% endif

  tc_digital_io_special_block special_blocks ();
  tc_digital_io_power_supply #(.VerticalIO(1'b0)) left_io_power_supply[3:0] ();
  tc_digital_io_power_supply #(.VerticalIO(1'b1)) top_io_power_supply[3:0] ();
  tc_digital_io_power_supply #(.VerticalIO(1'b0)) right_io_power_supply[2:0] ();
  tc_digital_io_power_supply #(.VerticalIO(1'b1)) bot_io_power_supply[3:0] ();

  tc_core_power_supply #(.VerticalIO(1'b0)) left_core_power_supply[1:0] ();
  tc_core_power_supply #(.VerticalIO(1'b1)) top_core_power_supply[1:0] ();
  tc_core_power_supply #(.VerticalIO(1'b0)) right_core_power_supply[1:0] ();
  tc_core_power_supply #(.VerticalIO(1'b1)) bot_core_power_supply[1:0] ();

  occamy_chip i_occamy_chip (
% if multichip_cfg["single_chip"] is False:
    .east_d2d_io         (io_east_d2d        ),
    .west_d2d_io         (io_west_d2d        ),
    .north_d2d_io        (io_north_d2d       ),
    .south_d2d_io        (io_south_d2d       ),
% endif
    .*
    );

endmodule
