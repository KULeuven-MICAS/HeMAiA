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

  localparam RTCTCK = 30.518us;        // 32.768 kHz
  localparam CLKTCK = 2000ps;          // 500 MHz
  localparam D2DTCK = 400ps;           // 2.5 GHz
  localparam int  SRAM_BANK = 32;      // 32 Banks architecture
  localparam int  SRAM_DEPTH = ${int(mem_size/8/32)};
  localparam int  SRAM_WIDTH = 8;      // 8 Bytes Wide

  logic rtc_clk_i, core_clk_i, d2d_clk_i, rst_ni;

  // Chip finish signal
  integer chip_finish[${max(x)}:${min(x)}][${max(y)}:${min(y)}];

  // Integer to save current time
  time current_time;

  // Generate reset and clock.
  initial begin
    rtc_clk_i  = 0;
    core_clk_i  = 0;
    d2d_clk_i = 0;
    // Load the binaries
% for i in x:
%   for j in y:
%     for k in range(0, 32):
    i_occamy_${i}_${j}.i_hemaia_mem_system.i_hemaia_mem.gen_banks[${k}].i_data_mem.i_tc_sram.load_data("app_chip_${i}_${j}/bank_${k}.hex");
%     endfor
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

  always #(RTCTCK / 2) begin
    rtc_clk_i = ~rtc_clk_i;
  end
  
  always #(CLKTCK / 2) begin
    core_clk_i = ~core_clk_i;
  end

  always #(D2DTCK / 2) begin
    d2d_clk_i = ~d2d_clk_i;
  end

  // Must be the frequency of i_uart0.clk_i in Hz
  localparam int unsigned UartDPIFreq = 1_000_000_000;

  // Definition of tri_state bus
% for i in x:
%   for j in y:
  tri [19:0] chip_${i}_${j}_to_${i+1}_${j}_link[3];
  tri [19:0] chip_${i}_${j}_to_${i}_${j+1}_link[3];
%   endfor
% endfor

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
  logic chip_${i}_${j}_link_from_east_rts;
  logic chip_${i}_${j}_link_from_east_cts;
  logic chip_${i}_${j}_link_to_east_rts;
  logic chip_${i}_${j}_link_to_east_cts;
  logic chip_${i}_${j}_link_from_east_test_request;
  logic chip_${i}_${j}_link_to_east_test_request;
  // West side
  logic chip_${i}_${j}_link_from_west_rts;
  logic chip_${i}_${j}_link_from_west_cts;
  logic chip_${i}_${j}_link_to_west_rts;
  logic chip_${i}_${j}_link_to_west_cts;
  logic chip_${i}_${j}_link_from_west_test_request;
  logic chip_${i}_${j}_link_to_west_test_request;
  // North side
  logic chip_${i}_${j}_link_from_north_rts;
  logic chip_${i}_${j}_link_from_north_cts;
  logic chip_${i}_${j}_link_to_north_rts;
  logic chip_${i}_${j}_link_to_north_cts;
  logic chip_${i}_${j}_link_from_north_test_request;
  logic chip_${i}_${j}_link_to_north_test_request;
  // South side
  logic chip_${i}_${j}_link_from_south_rts;
  logic chip_${i}_${j}_link_from_south_cts;
  logic chip_${i}_${j}_link_to_south_rts;
  logic chip_${i}_${j}_link_to_south_cts;
  logic chip_${i}_${j}_link_from_south_test_request;
  logic chip_${i}_${j}_link_to_south_test_request;
% endif

  occamy_chip i_occamy_${i}_${j} (
      .clk_i(core_clk_i),
      .rst_ni,
      .clk_periph_i(core_clk_i),
      .rst_periph_ni(rst_ni),
      .rtc_i(rtc_clk_i),
      .chip_id_i(8'h${i_hex_string}${j_hex_string}),
      .test_mode_i(1'b0),
      .boot_mode_i('0),
% if multichip_cfg['single_chip'] is False:
      .d2d_clk_i(core_clk_i),

      .east_d2d_io(chip_${i}_${j}_to_${i+1}_${j}_link),
      .flow_control_east_rts_o(chip_${i}_${j}_link_to_east_rts),
      .flow_control_east_cts_i(chip_${i}_${j}_link_to_east_cts),
      .flow_control_east_rts_i(chip_${i}_${j}_link_from_east_rts),
      .flow_control_east_cts_o(chip_${i}_${j}_link_from_east_cts),
      .east_test_being_requested_i(chip_${i}_${j}_link_to_east_test_request),
      .east_test_request_o(chip_${i}_${j}_link_from_east_test_request),

%   if i > min(x):
      .west_d2d_io(chip_${i-1}_${j}_to_${i}_${j}_link),
%   else:
      .west_d2d_io(),
%   endif
      .flow_control_west_rts_o(chip_${i}_${j}_link_to_west_rts),
      .flow_control_west_cts_i(chip_${i}_${j}_link_to_west_cts),
      .flow_control_west_rts_i(chip_${i}_${j}_link_from_west_rts),
      .flow_control_west_cts_o(chip_${i}_${j}_link_from_west_cts),
      .west_test_being_requested_i(chip_${i}_${j}_link_to_west_test_request),
      .west_test_request_o(chip_${i}_${j}_link_from_west_test_request),

%   if j > min(y):
      .north_d2d_io(chip_${i}_${j-1}_to_${i}_${j}_link),
%   else:
      .north_d2d_io(),
%   endif
      .flow_control_north_rts_o(chip_${i}_${j}_link_to_north_rts),
      .flow_control_north_cts_i(chip_${i}_${j}_link_to_north_cts),
      .flow_control_north_rts_i(chip_${i}_${j}_link_from_north_rts),
      .flow_control_north_cts_o(chip_${i}_${j}_link_from_north_cts),
      .north_test_being_requested_i(chip_${i}_${j}_link_to_north_test_request),
      .north_test_request_o(chip_${i}_${j}_link_from_north_test_request),

      .south_d2d_io(chip_${i}_${j}_to_${i}_${j+1}_link),
      .flow_control_south_rts_o(chip_${i}_${j}_link_to_south_rts),
      .flow_control_south_cts_i(chip_${i}_${j}_link_to_south_cts),
      .flow_control_south_rts_i(chip_${i}_${j}_link_from_south_rts),
      .flow_control_south_cts_o(chip_${i}_${j}_link_from_south_cts),
      .south_test_being_requested_i(chip_${i}_${j}_link_to_south_test_request),
      .south_test_request_o(chip_${i}_${j}_link_from_south_test_request),
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
      .BAUD('d31250000),
      // Frequency shouldn't matter since we are sending with the same clock.
      .FREQ(UartDPIFreq),
      .NAME("uart_${i}_${j}")
  ) i_uart_${i}_${j} (
      .clk_i (core_clk_i),
      .rst_ni(rst_ni),
      .tx_o  (rx_${i}_${j}),
      .rx_i  (tx_${i}_${j})
  );

  // Chip Status Monitor Block
  always @(i_occamy_${i}_${j}.i_hemaia_mem_system.i_hemaia_mem.gen_banks[31].i_data_mem.i_tc_sram.sram[SRAM_DEPTH-1][(SRAM_WIDTH*8-1)-:32]) begin
    if (i_occamy_${i}_${j}.i_hemaia_mem_system.i_hemaia_mem.gen_banks[31].i_data_mem.i_tc_sram.sram[SRAM_DEPTH-1][(SRAM_WIDTH*8-1)-:32] != 0) begin
      if (i_occamy_${i}_${j}.i_hemaia_mem_system.i_hemaia_mem.gen_banks[31].i_data_mem.i_tc_sram.sram[SRAM_DEPTH-1][(SRAM_WIDTH*8-1)-:32] == 32'd1) begin
        $display("Simulation of chip_${i}_${j} is finished at %tns", $time / 1000);
        chip_finish[${i}][${j}] = 1;
      end else begin
        $error("Simulation of chip_${i}_${j} is finished with errors %d at %tns",
               i_occamy_${i}_${j}.i_hemaia_mem_system.i_hemaia_mem.gen_banks[31].i_data_mem.i_tc_sram.sram[SRAM_DEPTH-1][(SRAM_WIDTH*8-1)-:32],
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
  assign chip_${i}_${j}_link_to_west_cts = '0;
  assign chip_${i}_${j}_link_from_west_rts = '0;
  assign chip_${i}_${j}_link_to_west_test_request = '0;
  assign chip_${i}_${j}_link_to_east_cts = chip_${i+1}_${j}_link_from_west_cts;
  assign chip_${i}_${j}_link_from_east_rts = chip_${i+1}_${j}_link_to_west_rts;
  assign chip_${i}_${j}_link_to_east_test_request = chip_${i+1}_${j}_link_from_west_test_request;

%     elif i == max(x):
  assign chip_${i}_${j}_link_to_west_cts = chip_${i-1}_${j}_link_from_east_cts;
  assign chip_${i}_${j}_link_from_west_rts = chip_${i-1}_${j}_link_to_east_rts;
  assign chip_${i}_${j}_link_to_west_test_request = chip_${i-1}_${j}_link_from_east_test_request;
  assign chip_${i}_${j}_link_to_east_cts = '0;
  assign chip_${i}_${j}_link_from_east_rts = '0;
  assign chip_${i}_${j}_link_to_east_test_request = '0;
  
%     else:
  assign chip_${i}_${j}_link_to_west_cts = chip_${i-1}_${j}_link_from_east_cts;
  assign chip_${i}_${j}_link_from_west_rts = chip_${i-1}_${j}_link_to_east_rts;
  assign chip_${i}_${j}_link_to_west_test_request = chip_${i-1}_${j}_link_from_east_test_request;
  assign chip_${i}_${j}_link_to_east_cts = chip_${i+1}_${j}_link_from_west_cts;
  assign chip_${i}_${j}_link_from_east_rts = chip_${i+1}_${j}_link_to_west_rts;
  assign chip_${i}_${j}_link_to_east_test_request = chip_${i+1}_${j}_link_from_west_test_request;
%     endif
  // Connect the north and south side of the chip
%     if j == min(y):
  assign chip_${i}_${j}_link_to_north_cts = '0;
  assign chip_${i}_${j}_link_from_north_rts = '0;
  assign chip_${i}_${j}_link_to_north_test_request = '0;
  assign chip_${i}_${j}_link_to_south_cts = chip_${i}_${j+1}_link_from_north_cts;
  assign chip_${i}_${j}_link_from_south_rts = chip_${i}_${j+1}_link_to_north_rts;
  assign chip_${i}_${j}_link_to_south_test_request = chip_${i}_${j+1}_link_from_north_test_request;
%     elif j == max(y):
  assign chip_${i}_${j}_link_to_north_cts = chip_${i}_${j-1}_link_from_south_cts;
  assign chip_${i}_${j}_link_from_north_rts = chip_${i}_${j-1}_link_to_south_rts;
  assign chip_${i}_${j}_link_to_north_test_request = chip_${i}_${j-1}_link_from_south_test_request;
  assign chip_${i}_${j}_link_to_south_cts = '0;
  assign chip_${i}_${j}_link_from_south_rts = '0;
  assign chip_${i}_${j}_link_to_south_test_request = '0;
%     else:
  assign chip_${i}_${j}_link_to_north_cts = chip_${i}_${j-1}_link_from_south_cts;
  assign chip_${i}_${j}_link_from_north_rts = chip_${i}_${j-1}_link_to_south_rts;
  assign chip_${i}_${j}_link_to_north_test_request = chip_${i}_${j-1}_link_from_south_test_request;
  assign chip_${i}_${j}_link_to_south_cts = chip_${i}_${j+1}_link_from_north_cts;
  assign chip_${i}_${j}_link_from_south_rts = chip_${i}_${j+1}_link_to_north_rts;
  assign chip_${i}_${j}_link_to_south_test_request = chip_${i}_${j+1}_link_from_north_test_request;
%     endif
%   endfor
% endfor
% endif
endmodule
