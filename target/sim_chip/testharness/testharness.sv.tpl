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

% if multichip_cfg['single_chip'] is False:
  /// Die2Die signals
  // East side
  logic chip_${i}_${j}_link_available_east;
  logic [207:0] chip_${i}_${j}_link_from_east [3];
  logic [2:0] chip_${i}_${j}_link_from_east_valid;
  logic [207:0] chip_${i}_${j}_link_to_east [3];
  logic [2:0] chip_${i}_${j}_link_to_east_valid;
  logic chip_${i}_${j}_link_from_east_rts;
  logic chip_${i}_${j}_link_from_east_cts;
  logic chip_${i}_${j}_link_to_east_rts;
  logic chip_${i}_${j}_link_to_east_cts;
  logic chip_${i}_${j}_link_east_tx_mode;
  
  // West side
  logic chip_${i}_${j}_link_available_west;
  logic [207:0] chip_${i}_${j}_link_from_west [3];
  logic [2:0] chip_${i}_${j}_link_from_west_valid;
  logic [207:0] chip_${i}_${j}_link_to_west [3];
  logic [2:0] chip_${i}_${j}_link_to_west_valid;
  logic chip_${i}_${j}_link_from_west_rts;
  logic chip_${i}_${j}_link_from_west_cts;
  logic chip_${i}_${j}_link_to_west_rts;
  logic chip_${i}_${j}_link_to_west_cts;
  logic chip_${i}_${j}_link_west_tx_mode;

  // North side
  logic chip_${i}_${j}_link_available_north;
  logic [207:0] chip_${i}_${j}_link_from_north [3];
  logic [2:0] chip_${i}_${j}_link_from_north_valid;
  logic [207:0] chip_${i}_${j}_link_to_north [3];
  logic [2:0] chip_${i}_${j}_link_to_north_valid;
  logic chip_${i}_${j}_link_from_north_rts;
  logic chip_${i}_${j}_link_from_north_cts;
  logic chip_${i}_${j}_link_to_north_rts;
  logic chip_${i}_${j}_link_to_north_cts;
  logic chip_${i}_${j}_link_north_tx_mode;
  // South side
  logic chip_${i}_${j}_link_available_south;
  logic [207:0] chip_${i}_${j}_link_from_south [3];
  logic [2:0] chip_${i}_${j}_link_from_south_valid;
  logic [207:0] chip_${i}_${j}_link_to_south [3];
  logic [2:0] chip_${i}_${j}_link_to_south_valid;
  logic chip_${i}_${j}_link_from_south_rts;
  logic chip_${i}_${j}_link_from_south_cts;
  logic chip_${i}_${j}_link_to_south_rts;
  logic chip_${i}_${j}_link_to_south_cts;
  logic chip_${i}_${j}_link_south_tx_mode;
% endif

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
      .link_east_tx_mode_o(chip_${i}_${j}_link_east_tx_mode),
      .link_from_east_i(chip_${i}_${j}_link_from_east),
      .link_from_east_valid_i(chip_${i}_${j}_link_from_east_valid),
      .link_to_east_o(chip_${i}_${j}_link_to_east),
      .link_to_east_valid_o(chip_${i}_${j}_link_to_east_valid),
      .flow_control_east_rts_o(chip_${i}_${j}_link_to_east_rts),
      .flow_control_east_cts_i(chip_${i}_${j}_link_to_east_cts),
      .flow_control_east_rts_i(chip_${i}_${j}_link_from_east_rts),
      .flow_control_east_cts_o(chip_${i}_${j}_link_from_east_cts),

      .link_available_west_i(chip_${i}_${j}_link_available_west),
      .link_west_tx_mode_o(chip_${i}_${j}_link_west_tx_mode),
      .link_from_west_i(chip_${i}_${j}_link_from_west),
      .link_from_west_valid_i(chip_${i}_${j}_link_from_west_valid),
      .link_to_west_o(chip_${i}_${j}_link_to_west),
      .link_to_west_valid_o(chip_${i}_${j}_link_to_west_valid),
      .flow_control_west_rts_o(chip_${i}_${j}_link_to_west_rts),
      .flow_control_west_cts_i(chip_${i}_${j}_link_to_west_cts),
      .flow_control_west_rts_i(chip_${i}_${j}_link_from_west_rts),
      .flow_control_west_cts_o(chip_${i}_${j}_link_from_west_cts),

      .link_available_north_i(chip_${i}_${j}_link_available_north),
      .link_north_tx_mode_o(chip_${i}_${j}_link_north_tx_mode),
      .link_from_north_i(chip_${i}_${j}_link_from_north),
      .link_from_north_valid_i(chip_${i}_${j}_link_from_north_valid),
      .link_to_north_o(chip_${i}_${j}_link_to_north),
      .link_to_north_valid_o(chip_${i}_${j}_link_to_north_valid),
      .flow_control_north_rts_o(chip_${i}_${j}_link_to_north_rts),
      .flow_control_north_cts_i(chip_${i}_${j}_link_to_north_cts),
      .flow_control_north_rts_i(chip_${i}_${j}_link_from_north_rts),
      .flow_control_north_cts_o(chip_${i}_${j}_link_from_north_cts),

      .link_available_south_i(chip_${i}_${j}_link_available_south),
      .link_south_tx_mode_o(chip_${i}_${j}_link_south_tx_mode),
      .link_from_south_i(chip_${i}_${j}_link_from_south),
      .link_from_south_valid_i(chip_${i}_${j}_link_from_south_valid),
      .link_to_south_o(chip_${i}_${j}_link_to_south),
      .link_to_south_valid_o(chip_${i}_${j}_link_to_south_valid),
      .flow_control_south_rts_o(chip_${i}_${j}_link_to_south_rts),
      .flow_control_south_cts_i(chip_${i}_${j}_link_to_south_cts),
      .flow_control_south_rts_i(chip_${i}_${j}_link_from_south_rts),
      .flow_control_south_cts_o(chip_${i}_${j}_link_from_south_cts),
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

% if multichip_cfg['single_chip'] is False:
// Connect the signals: control signals
% for i in x:
%   for j in y:
  // Connect the east and west side of the chip
%     if i == min(x):
  assign chip_${i}_${j}_link_available_west = '0;
  assign chip_${i}_${j}_link_to_west_cts = '0;
  assign chip_${i}_${j}_link_from_west_rts = '0;
  assign chip_${i}_${j}_link_from_west = '{default: '0};
  assign chip_${i}_${j}_link_from_west_valid = '0;
  assign chip_${i}_${j}_link_available_east = '1;
  assign chip_${i}_${j}_link_from_east_valid = chip_${i+1}_${j}_link_to_west_valid;
  assign chip_${i}_${j}_link_to_east_cts = chip_${i+1}_${j}_link_from_west_cts;
  assign chip_${i}_${j}_link_from_east_rts = chip_${i+1}_${j}_link_to_west_rts;
%     elif i == max(x):
  assign chip_${i}_${j}_link_available_west = '1;
  assign chip_${i}_${j}_link_from_west_valid = chip_${i-1}_${j}_link_to_east_valid;
  assign chip_${i}_${j}_link_to_west_cts = chip_${i-1}_${j}_link_from_east_cts;
  assign chip_${i}_${j}_link_from_west_rts = chip_${i-1}_${j}_link_to_east_rts;
  assign chip_${i}_${j}_link_available_east = '0;
  assign chip_${i}_${j}_link_from_east = '{default: '0};
  assign chip_${i}_${j}_link_from_east_valid = '0;
  assign chip_${i}_${j}_link_to_east_cts = '0;
  assign chip_${i}_${j}_link_from_east_rts = '0;
%     else:
  assign chip_${i}_${j}_link_available_west = '1;
  assign chip_${i}_${j}_link_from_west_valid = chip_${i-1}_${j}_link_to_east_valid;
  assign chip_${i}_${j}_link_to_west_cts = chip_${i-1}_${j}_link_from_east_cts;
  assign chip_${i}_${j}_link_from_west_rts = chip_${i-1}_${j}_link_to_east_rts;
  assign chip_${i}_${j}_link_available_east = '1;
  assign chip_${i}_${j}_link_from_east_valid = chip_${i+1}_${j}_link_to_west_valid;
  assign chip_${i}_${j}_link_to_east_cts = chip_${i+1}_${j}_link_from_west_cts;
  assign chip_${i}_${j}_link_from_east_rts = chip_${i+1}_${j}_link_to_west_rts;
%     endif
  // Connect the north and south side of the chip
%     if j == min(y):
  assign chip_${i}_${j}_link_available_north = '0;
  assign chip_${i}_${j}_link_to_north_cts = '0;
  assign chip_${i}_${j}_link_from_north_rts = '0;
  assign chip_${i}_${j}_link_from_north = '{default: '0};
  assign chip_${i}_${j}_link_from_north_valid = '0;
  assign chip_${i}_${j}_link_available_south = '1;
  assign chip_${i}_${j}_link_from_south_valid = chip_${i}_${j+1}_link_to_north_valid;
  assign chip_${i}_${j}_link_to_south_cts = chip_${i}_${j+1}_link_from_north_cts;
  assign chip_${i}_${j}_link_from_south_rts = chip_${i}_${j+1}_link_to_north_rts;
%     elif j == max(y):
  assign chip_${i}_${j}_link_available_north = '1;
  assign chip_${i}_${j}_link_from_north_valid = chip_${i}_${j-1}_link_to_south_valid;
  assign chip_${i}_${j}_link_to_north_cts = chip_${i}_${j-1}_link_from_south_cts;
  assign chip_${i}_${j}_link_from_north_rts = chip_${i}_${j-1}_link_to_south_rts;
  assign chip_${i}_${j}_link_available_south = '0;
  assign chip_${i}_${j}_link_from_south = '{default: '0};
  assign chip_${i}_${j}_link_from_south_valid = '0;
  assign chip_${i}_${j}_link_to_south_cts = '0;
  assign chip_${i}_${j}_link_from_south_rts = '0;
%     else:
  assign chip_${i}_${j}_link_available_north = '1;
  assign chip_${i}_${j}_link_from_north_valid = chip_${i}_${j-1}_link_to_south_valid;
  assign chip_${i}_${j}_link_to_north_cts = chip_${i}_${j-1}_link_from_south_cts;
  assign chip_${i}_${j}_link_from_north_rts = chip_${i}_${j-1}_link_to_south_rts;
  assign chip_${i}_${j}_link_available_south = '1;
  assign chip_${i}_${j}_link_from_south_valid = chip_${i}_${j+1}_link_to_north_valid;
  assign chip_${i}_${j}_link_to_south_cts = chip_${i}_${j+1}_link_from_north_cts;
  assign chip_${i}_${j}_link_from_south_rts = chip_${i}_${j+1}_link_to_north_rts;
%     endif
%   endfor
% endfor

// Connect the signals: half_duplex data signals
% for i in x:
%   for j in y:
%     if i != max(x) and j != max(y):
  half_duplex_bus_emulator #(
      .BusWidth(208),
      .ChannelNum(3)
  ) i_half_duplex_east_${i}_${j} (
      .port1_tx_mode_i(chip_${i}_${j}_link_east_tx_mode),
      .port2_tx_mode_i(chip_${i+1}_${j}_link_west_tx_mode),
      .port1_i(chip_${i}_${j}_link_to_east),
      .port2_i(chip_${i+1}_${j}_link_to_west),
      .port1_o(chip_${i+1}_${j}_link_from_west),
      .port2_o(chip_${i}_${j}_link_from_east)
  );
  half_duplex_bus_emulator #(
      .BusWidth(208),
      .ChannelNum(3)
  ) i_half_duplex_south_${i}_${j} (
      .port1_tx_mode_i(chip_${i}_${j}_link_south_tx_mode),
      .port2_tx_mode_i(chip_${i}_${j+1}_link_north_tx_mode),
      .port1_i(chip_${i}_${j}_link_to_south),
      .port2_i(chip_${i}_${j+1}_link_to_north),
      .port1_o(chip_${i}_${j+1}_link_from_north),
      .port2_o(chip_${i}_${j}_link_from_south)
  );
%     elif i == max(x) and j != max(y):
  half_duplex_bus_emulator #(
      .BusWidth(208),
      .ChannelNum(3)
  ) i_half_duplex_south_${i}_${j} (
      .port1_tx_mode_i(chip_${i}_${j}_link_south_tx_mode),
      .port2_tx_mode_i(chip_${i}_${j+1}_link_north_tx_mode),
      .port1_i(chip_${i}_${j}_link_to_south),
      .port2_i(chip_${i}_${j+1}_link_to_north),
      .port1_o(chip_${i}_${j+1}_link_from_north),
      .port2_o(chip_${i}_${j}_link_from_south)
  );
%     elif i != max(x) and j == max(y):
  half_duplex_bus_emulator #(
      .BusWidth(208),
      .ChannelNum(3)
  ) i_half_duplex_east_${i}_${j} (
      .port1_tx_mode_i(chip_${i}_${j}_link_east_tx_mode),
      .port2_tx_mode_i(chip_${i+1}_${j}_link_west_tx_mode),
      .port1_i(chip_${i}_${j}_link_to_east),
      .port2_i(chip_${i+1}_${j}_link_to_west),
      .port1_o(chip_${i+1}_${j}_link_from_west),
      .port2_o(chip_${i}_${j}_link_from_east)
  );
%     endif
%   endfor
% endfor
% endif
endmodule
