// Copyright 2020 ETH Zurich and University of Bologna.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51

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

  reg [511:0] wide_sram_buffer[SRAM_DEPTH-1:0];

  // Task to load binary file into memory array
  task automatic load_binary_to_buffer(input string filepath);
    integer fd;
    integer i;
    reg [7:0] buffer[SRAM_WIDTH*SRAM_DEPTH];
    foreach (buffer[i]) buffer[i] = 0;

    // Open the binary file
    fd = $fopen(filepath, "rb");
    if (fd == 0) begin
      $display("Failed to open binary file: %s", filepath);
      $finish;
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


  // Generate reset and clock.
  initial begin
    rtc_i  = 0;
    clk_i  = 0;

    // Reset the chip
    rst_ni = 1;
    #0;
    $display("Resetting the system at %tns", $time / 1000);
    rst_ni = 0;
    #(10 + $urandom % 10);
    $display("Reset released at %tns", $time / 1000);
    rst_ni = 1;

    // Initialize the memory
    load_binary_to_buffer("app_chip_0_0.bin");

    force i_occamy.i_spm_wide_cut.i_mem.i_tc_sram.sram = wide_sram_buffer;
    #0;
    release i_occamy.i_spm_wide_cut.i_mem.i_tc_sram.sram;

    // Places to load the binaries
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

  // Finish Block
  always @(i_occamy.i_spm_wide_cut.i_mem.i_tc_sram.sram[SRAM_DEPTH-1][(SRAM_WIDTH*8-1)-:32]) begin
    if (i_occamy.i_spm_wide_cut.i_mem.i_tc_sram.sram[SRAM_DEPTH-1][(SRAM_WIDTH*8-1)-:32] != 0) begin
      if (i_occamy.i_spm_wide_cut.i_mem.i_tc_sram.sram[SRAM_DEPTH-1][(SRAM_WIDTH*8-1)-:32] == 32'd1) begin
        $display("Simulation finished at %tns", $time / 1000);
        $finish;
      end else begin
        $error("Simulation finished with errors %d at %tns", i_occamy.i_spm_wide_cut.i_mem.i_tc_sram.sram[SRAM_DEPTH-1][(SRAM_WIDTH*8-1)-:32], $time / 1000);
      end
    end
  end

  logic clk_periph_i, rst_periph_ni;
  assign clk_periph_i  = clk_i;
  assign rst_periph_ni = rst_ni;

  /// Uart signals
  logic tx, rx;

  occamy_chip i_occamy (
      .clk_i,
      .rst_ni,
      .clk_periph_i,
      .rst_periph_ni,
      .rtc_i,
      .chip_id_i('0),
      .test_mode_i(1'b0),
      .boot_mode_i(0),
      .uart_tx_o(tx),
      .uart_rx_i(rx),
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
      .i2c_sda_io(),
      .i2c_scl_io(),
      .spim_sck_o(),
      .spim_csb_o(),
      .spim_sd_io(),
      .ext_irq_i('0)
  );

  // Must be the frequency of i_uart0.clk_i in Hz
  localparam int unsigned UartDPIFreq = 1_000_000_000;

  uartdpi #(
      .BAUD('d20_000_000),
      // Frequency shouldn't matter since we are sending with the same clock.
      .FREQ(UartDPIFreq),
      .NAME("uart0")
  ) i_uart0 (
      .clk_i (clk_i),
      .rst_ni(rst_ni),
      .tx_o  (rx),
      .rx_i  (tx)
  );

endmodule
