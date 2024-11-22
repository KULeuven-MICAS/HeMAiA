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


  // Task to load binary file into memory array
  task automatic load_binary_to_hardware(input string filepath,
                                         ref logic [511:0] wide_sram_buffer[SRAM_DEPTH-1:0]);
    integer fd;
    integer i;
    reg [7:0] buffer[SRAM_WIDTH*SRAM_DEPTH];
    foreach (buffer[i]) buffer[i] = 0;  // Zero out the buffer

    // Open the binary file
    fd = $fopen(filepath, "rb");
    if (fd == 0) begin
      $display("Failed to open binary file: %s", filepath);
      $finish(-1);
    end

    // Read the binary data into the buffer
    $fread(buffer, fd);
    $fclose(fd);

    // Assemble bytes into 512-bit words
    for (i = 0; i < SRAM_DEPTH; i = i + 1) begin
      wide_sram_buffer[i] = {
        buffer[i*SRAM_WIDTH+63],
        buffer[i*SRAM_WIDTH+62],
        buffer[i*SRAM_WIDTH+61],
        buffer[i*SRAM_WIDTH+60],
        buffer[i*SRAM_WIDTH+59],
        buffer[i*SRAM_WIDTH+58],
        buffer[i*SRAM_WIDTH+57],
        buffer[i*SRAM_WIDTH+56],
        buffer[i*SRAM_WIDTH+55],
        buffer[i*SRAM_WIDTH+54],
        buffer[i*SRAM_WIDTH+53],
        buffer[i*SRAM_WIDTH+52],
        buffer[i*SRAM_WIDTH+51],
        buffer[i*SRAM_WIDTH+50],
        buffer[i*SRAM_WIDTH+49],
        buffer[i*SRAM_WIDTH+48],
        buffer[i*SRAM_WIDTH+47],
        buffer[i*SRAM_WIDTH+46],
        buffer[i*SRAM_WIDTH+45],
        buffer[i*SRAM_WIDTH+44],
        buffer[i*SRAM_WIDTH+43],
        buffer[i*SRAM_WIDTH+42],
        buffer[i*SRAM_WIDTH+41],
        buffer[i*SRAM_WIDTH+40],
        buffer[i*SRAM_WIDTH+39],
        buffer[i*SRAM_WIDTH+38],
        buffer[i*SRAM_WIDTH+37],
        buffer[i*SRAM_WIDTH+36],
        buffer[i*SRAM_WIDTH+35],
        buffer[i*SRAM_WIDTH+34],
        buffer[i*SRAM_WIDTH+33],
        buffer[i*SRAM_WIDTH+32],
        buffer[i*SRAM_WIDTH+31],
        buffer[i*SRAM_WIDTH+30],
        buffer[i*SRAM_WIDTH+29],
        buffer[i*SRAM_WIDTH+28],
        buffer[i*SRAM_WIDTH+27],
        buffer[i*SRAM_WIDTH+26],
        buffer[i*SRAM_WIDTH+25],
        buffer[i*SRAM_WIDTH+24],
        buffer[i*SRAM_WIDTH+23],
        buffer[i*SRAM_WIDTH+22],
        buffer[i*SRAM_WIDTH+21],
        buffer[i*SRAM_WIDTH+20],
        buffer[i*SRAM_WIDTH+19],
        buffer[i*SRAM_WIDTH+18],
        buffer[i*SRAM_WIDTH+17],
        buffer[i*SRAM_WIDTH+16],
        buffer[i*SRAM_WIDTH+15],
        buffer[i*SRAM_WIDTH+14],
        buffer[i*SRAM_WIDTH+13],
        buffer[i*SRAM_WIDTH+12],
        buffer[i*SRAM_WIDTH+11],
        buffer[i*SRAM_WIDTH+10],
        buffer[i*SRAM_WIDTH+9],
        buffer[i*SRAM_WIDTH+8],
        buffer[i*SRAM_WIDTH+7],
        buffer[i*SRAM_WIDTH+6],
        buffer[i*SRAM_WIDTH+5],
        buffer[i*SRAM_WIDTH+4],
        buffer[i*SRAM_WIDTH+3],
        buffer[i*SRAM_WIDTH+2],
        buffer[i*SRAM_WIDTH+1],
        buffer[i*SRAM_WIDTH+0]
      };
    end
    $display("Binary file '%s' loaded into memory.", filepath);
  endtask

  // Chip finish signal
  integer chip_finish[${max(x)}:${min(x)}][${max(y)}:${min(y)}];

  // Generate reset and clock.
  initial begin
    rtc_i  = 0;
    clk_i  = 0;

    // Reset the chip
    foreach (chip_finish[i,j]) begin
      chip_finish[i][j] = 0;
    end
    rst_ni = 1;
    #0;
    $display("Resetting the system at %tns", $time / 1000);
    rst_ni = 0;
    #(10 + $urandom % 10);
    $display("Reset released at %tns", $time / 1000);
    rst_ni = 1;
    // Load the binaries
% for i in x:
%   for j in y:
    i_occamy_${i}_${j}.i_spm_wide_cut.i_mem.i_mem.i_tc_sram.load_data("app_chip_${i}_${j}.hex");
%   endfor
% endfor
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

% for i in x:
%   for j in y:
  /// Uart signals
  logic tx_${i}_${j}, rx_${i}_${j};

  <%
    i_hex_string = "{:01x}".format(i)
    j_hex_string = "{:01x}".format(j)
  %>

  occamy_chip i_occamy_${i}_${j} (
      .clk_i,
      .rst_ni,
      .clk_periph_i,
      .rst_periph_ni,
      .rtc_i,
      .chip_id_i(8'h${i_hex_string}${j_hex_string}),
      .test_mode_i(1'b0),
      .boot_mode_i('0),
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

endmodule
