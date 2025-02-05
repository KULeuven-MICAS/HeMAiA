// Copyright 2024 KU Leuven.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51
// Yunhao Deng <yunhao.deng@kuleuven.be>

<%
  x = range(0, 1)
  y = range(0, 1)
  if multichip_cfg["single_chip"] is False:
    x = range(multichip_cfg["testbench_cfg"]["upper_left_coordinate"][0], multichip_cfg["testbench_cfg"]["lower_right_coordinate"][0] + 1)
    y = range(multichip_cfg["testbench_cfg"]["upper_left_coordinate"][1], multichip_cfg["testbench_cfg"]["lower_right_coordinate"][1] + 1)
%>

`timescale 1ns / 1ps
`include "axi/typedef.svh"

module testharness
  import occamy_pkg::*;
();

  localparam RTCTCK = 30.518us;  // 32.768 kHz
  localparam CLKTCK = 1000ps;  // 1 GHz
  localparam SRAM_DEPTH = 16384;  // 16K Depp
  localparam SRAM_WIDTH = 64;  // 64 Bytes Wide

  logic clk_i;
  logic rst_ni;
  logic rtc_i;

  // Chip finish signal
  integer chip_finish[${max(x)}:${min(x)}][${max(y)}:${min(y)}];

  // Integer to save current time
  time current_time;

  // Generate reset and clock.
  initial begin
    rtc_i  = 0;
    clk_i  = 0;
    // Load the binaries
% for i in x:
%   for j in y:
    i_occamy_${i}_${j}.i_spm_wide_cut.i_mem.i_mem.i_tc_sram.load_data("app_chip_${i}_${j}.hex");
%   endfor
% endfor
    // Reset the chip
    foreach (chip_finish[i,j]) begin
      chip_finish[i][j] = 0;
    end
    rst_ni = 1;
    #0;
    current_time = $time / 1000;
    $display("Resetting the system at %tns", current_time);
    rst_ni = 0;
    #(10 + $urandom % 10);
    current_time = $time / 1000;
    $display("Reset released at %tns", current_time);
    rst_ni = 1;
  end

  always_comb begin
    automatic integer allFinished = 1;
    automatic integer allCorrect = 1;
    for (int i = ${min(x)}; i <= ${max(x)}; i = i + 1) begin
      for (int j = ${min(y)}; j <= ${max(y)}; j = j + 1) begin
        if (chip_finish[i][j] == 0) begin
          allFinished = 0;
        end
        if (chip_finish[i][j] == -1) begin
          allCorrect = 0;
        end
      end
    end

    if (allFinished == 1) begin
      if (allCorrect == 1) begin
        $display("All chips finished successfully at %tns", $time / 1000);
        $finish;
      end else begin
        $error("All chips finished with errors at %tns", $time / 1000);
      end
      $finish(-1);
    end
  end

  always #(CLKTCK / 2) begin
    clk_i = ~clk_i;
  end

  always #(RTCTCK / 2) begin
    rtc_i = ~rtc_i;
  end

  initial begin
    forever begin
      clk_i = 1;
      #(CLKTCK / 2);
      clk_i = 0;
      #(CLKTCK / 2);
    end
  end

  logic clk_periph_i, rst_periph_ni;
  assign clk_periph_i  = clk_i;
  assign rst_periph_ni = rst_ni;

  // Must be the frequency of i_uart0.clk_i in Hz
  localparam int unsigned UartDPIFreq = 1_000_000_000;

  // Instatiate Chips

% for i in x:
%   for j in y:

<%
    i_hex_string = "{:01x}".format(i)
    j_hex_string = "{:01x}".format(j)
%>
  /// Uart signals
  logic tx_${i}_${j}, rx_${i}_${j};

  /// Die2Die signals
  // East side
  logic chip_${i}_${j}_link_available_east;
  logic [597:0] chip_${i}_${j}_payload_from_east;
  logic chip_${i}_${j}_payload_from_east_valid;
  logic chip_${i}_${j}_payload_from_east_ready;
  logic [597:0] chip_${i}_${j}_payload_to_east;
  logic chip_${i}_${j}_payload_to_east_valid;
  logic chip_${i}_${j}_payload_to_east_ready;

  // West side
  logic chip_${i}_${j}_link_available_west;
  logic [597:0] chip_${i}_${j}_payload_from_west;
  logic chip_${i}_${j}_payload_from_west_valid;
  logic chip_${i}_${j}_payload_from_west_ready;
  logic [597:0] chip_${i}_${j}_payload_to_west;
  logic chip_${i}_${j}_payload_to_west_valid;
  logic chip_${i}_${j}_payload_to_west_ready;

  // North side
  logic chip_${i}_${j}_link_available_north;
  logic [597:0] chip_${i}_${j}_payload_from_north;
  logic chip_${i}_${j}_payload_from_north_valid;
  logic chip_${i}_${j}_payload_from_north_ready;
  logic [597:0] chip_${i}_${j}_payload_to_north;
  logic chip_${i}_${j}_payload_to_north_valid;
  logic chip_${i}_${j}_payload_to_north_ready;

  // South side
  logic chip_${i}_${j}_link_available_south;
  logic [597:0] chip_${i}_${j}_payload_from_south;
  logic chip_${i}_${j}_payload_from_south_valid;
  logic chip_${i}_${j}_payload_from_south_ready;
  logic [597:0] chip_${i}_${j}_payload_to_south;
  logic chip_${i}_${j}_payload_to_south_valid;
  logic chip_${i}_${j}_payload_to_south_ready;

  occamy_chip i_occamy_${i}_${j} (
      .clk_i,
      .rst_ni,
      .clk_periph_i,
      .rst_periph_ni,
      .rtc_i,
      .chip_id_i(8'h${i_hex_string}${j_hex_string}),
      .test_mode_i(1'b0),
      .boot_mode_i('0),
% if multichip_cfg['single_chip'] is False:
      .link_available_east_i(chip_${i}_${j}_link_available_east),
      .payload_from_east_i(chip_${i}_${j}_payload_from_east),
      .payload_from_east_valid_i(chip_${i}_${j}_payload_from_east_valid),
      .payload_from_east_ready_o(chip_${i}_${j}_payload_from_east_ready),
      .payload_to_east_o(chip_${i}_${j}_payload_to_east),
      .payload_to_east_valid_o(chip_${i}_${j}_payload_to_east_valid),
      .payload_to_east_ready_i(chip_${i}_${j}_payload_to_east_ready),
      .link_available_west_i(chip_${i}_${j}_link_available_west),
      .payload_from_west_i(chip_${i}_${j}_payload_from_west),
      .payload_from_west_valid_i(chip_${i}_${j}_payload_from_west_valid),
      .payload_from_west_ready_o(chip_${i}_${j}_payload_from_west_ready),
      .payload_to_west_o(chip_${i}_${j}_payload_to_west),
      .payload_to_west_valid_o(chip_${i}_${j}_payload_to_west_valid),
      .payload_to_west_ready_i(chip_${i}_${j}_payload_to_west_ready),
      .link_available_north_i(chip_${i}_${j}_link_available_north),
      .payload_from_north_i(chip_${i}_${j}_payload_from_north),
      .payload_from_north_valid_i(chip_${i}_${j}_payload_from_north_valid),
      .payload_from_north_ready_o(chip_${i}_${j}_payload_from_north_ready),
      .payload_to_north_o(chip_${i}_${j}_payload_to_north),
      .payload_to_north_valid_o(chip_${i}_${j}_payload_to_north_valid),
      .payload_to_north_ready_i(chip_${i}_${j}_payload_to_north_ready),
      .link_available_south_i(chip_${i}_${j}_link_available_south),
      .payload_from_south_i(chip_${i}_${j}_payload_from_south),
      .payload_from_south_valid_i(chip_${i}_${j}_payload_from_south_valid),
      .payload_from_south_ready_o(chip_${i}_${j}_payload_from_south_ready),
      .payload_to_south_o(chip_${i}_${j}_payload_to_south),
      .payload_to_south_valid_o(chip_${i}_${j}_payload_to_south_valid),
      .payload_to_south_ready_i(chip_${i}_${j}_payload_to_south_ready),
% endif
      .uart_tx_o(tx_${i}_${j}),
      .uart_rx_i(rx_${i}_${j}),
      .uart_rts_no(),
      .uart_cts_ni('0),
      .gpio_d_i('0),
      .gpio_d_o(),
      .gpio_oe_o(),
      .jtag_trst_ni('0),
      .jtag_tck_i('0),
      .jtag_tms_i('0),
      .jtag_tdi_i('0),
      .jtag_tdo_o(),
      .spis_sd_i('1),
      .spis_sd_en_o(),
      .spis_sd_o(),
      .spis_csb_i('1),
      .spis_sck_i('0)
  );

  uartdpi #(
      .BAUD('d20_000_000),
      // Frequency shouldn't matter since we are sending with the same clock.
      .FREQ(UartDPIFreq),
      .NAME("uart_${i}_${j}")
  ) i_uart_${i}_${j} (
      .clk_i (clk_i),
      .rst_ni(rst_ni),
      .tx_o  (rx_${i}_${j}),
      .rx_i  (tx_${i}_${j})
  );

  // Chip Status Monitor Block
  always @(i_occamy_${i}_${j}.i_spm_wide_cut.i_mem.i_mem.i_tc_sram.sram[SRAM_DEPTH-1][(SRAM_WIDTH*8-1)-:32]) begin
    if (i_occamy_${i}_${j}.i_spm_wide_cut.i_mem.i_mem.i_tc_sram.sram[SRAM_DEPTH-1][(SRAM_WIDTH*8-1)-:32] != 0) begin
      if (i_occamy_${i}_${j}.i_spm_wide_cut.i_mem.i_mem.i_tc_sram.sram[SRAM_DEPTH-1][(SRAM_WIDTH*8-1)-:32] == 32'd1) begin
        $display("Simulation of chip_${i}_${j} is finished at %tns", $time / 1000);
        chip_finish[${i}][${j}] = 1;
      end else begin
        $error("Simulation of chip_${i}_${j} is finished with errors %d at %tns",
               i_occamy_${i}_${j}.i_spm_wide_cut.i_mem.i_mem.i_tc_sram.sram[SRAM_DEPTH-1][(SRAM_WIDTH*8-1)-:32],
               $time / 1000);
        chip_finish[${i}][${j}] = -1;
      end
    end
  end

%   endfor
% endfor

// Connect the signals declared together
% for i in x:
%   for j in y:
  // Connect the east and west side of the chip
%     if i == min(x):
  assign chip_${i}_${j}_link_available_west = '0;
  assign chip_${i}_${j}_payload_from_west = '0;
  assign chip_${i}_${j}_payload_from_west_valid = '0;
  assign chip_${i}_${j}_payload_to_west_ready = '0;
  assign chip_${i}_${j}_link_available_east = '1;
  assign chip_${i}_${j}_payload_from_east = chip_${i+1}_${j}_payload_to_west;
  assign chip_${i}_${j}_payload_from_east_valid = chip_${i+1}_${j}_payload_to_west_valid;
  assign chip_${i}_${j}_payload_to_east_ready = chip_${i+1}_${j}_payload_from_west_ready;
%     elif i == max(x):
  assign chip_${i}_${j}_link_available_west = '1;
  assign chip_${i}_${j}_payload_from_west = chip_${i-1}_${j}_payload_to_east;
  assign chip_${i}_${j}_payload_from_west_valid = chip_${i-1}_${j}_payload_to_east_valid;
  assign chip_${i}_${j}_payload_to_west_ready = chip_${i-1}_${j}_payload_from_east_ready;
  assign chip_${i}_${j}_link_available_east = '0;
  assign chip_${i}_${j}_payload_from_east = '0;
  assign chip_${i}_${j}_payload_from_east_valid = '0;
  assign chip_${i}_${j}_payload_to_east_ready = '0;
%     else:
  assign chip_${i}_${j}_link_available_west = '1;
  assign chip_${i}_${j}_payload_from_west = chip_${i-1}_${j}_payload_to_east;
  assign chip_${i}_${j}_payload_from_west_valid = chip_${i-1}_${j}_payload_to_east_valid;
  assign chip_${i}_${j}_payload_to_west_ready = chip_${i-1}_${j}_payload_from_east_ready;
  assign chip_${i}_${j}_link_available_east = '1;
  assign chip_${i}_${j}_payload_from_east = chip_${i+1}_${j}_payload_to_west;
  assign chip_${i}_${j}_payload_from_east_valid = chip_${i+1}_${j}_payload_to_west_valid;
  assign chip_${i}_${j}_payload_to_east_ready = chip_${i+1}_${j}_payload_from_west_ready;
%     endif
  // Connect the north and south side of the chip
%     if j == min(y):
  assign chip_${i}_${j}_link_available_north = '0;
  assign chip_${i}_${j}_payload_from_north = '0;
  assign chip_${i}_${j}_payload_from_north_valid = '0;
  assign chip_${i}_${j}_payload_to_north_ready = '0;
  assign chip_${i}_${j}_link_available_south = '1;
  assign chip_${i}_${j}_payload_from_south = chip_${i}_${j+1}_payload_to_north;
  assign chip_${i}_${j}_payload_from_south_valid = chip_${i}_${j+1}_payload_to_north_valid;
  assign chip_${i}_${j}_payload_to_south_ready = chip_${i}_${j+1}_payload_from_north_ready;
%     elif j == max(y):
  assign chip_${i}_${j}_link_available_north = '1;
  assign chip_${i}_${j}_payload_from_north = chip_${i}_${j-1}_payload_to_south;
  assign chip_${i}_${j}_payload_from_north_valid = chip_${i}_${j-1}_payload_to_south_valid;
  assign chip_${i}_${j}_payload_to_north_ready = chip_${i}_${j-1}_payload_from_south_ready;
  assign chip_${i}_${j}_link_available_south = '0;
  assign chip_${i}_${j}_payload_from_south = '0;
  assign chip_${i}_${j}_payload_from_south_valid = '0;
  assign chip_${i}_${j}_payload_to_south_ready = '0;
%     else:
  assign chip_${i}_${j}_link_available_north = '1;
  assign chip_${i}_${j}_payload_from_north = chip_${i}_${j-1}_payload_to_south;
  assign chip_${i}_${j}_payload_from_north_valid = chip_${i}_${j-1}_payload_to_south_valid;
  assign chip_${i}_${j}_payload_to_north_ready = chip_${i}_${j-1}_payload_from_south_ready;
  assign chip_${i}_${j}_link_available_south = '1;
  assign chip_${i}_${j}_payload_from_south = chip_${i}_${j+1}_payload_to_north;
  assign chip_${i}_${j}_payload_from_south_valid = chip_${i}_${j+1}_payload_to_north_valid;
  assign chip_${i}_${j}_payload_to_south_ready = chip_${i}_${j+1}_payload_from_north_ready;
%     endif
%   endfor
% endfor
endmodule
