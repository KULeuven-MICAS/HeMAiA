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

  logic clk_i;
  logic rst_ni;
  logic rtc_i;

  // Generate reset and clock.
  initial begin
    rtc_i = 0;
    clk_i = 0;
    rst_ni = 0;
    $display("Initial block for reset signal started.");
    #100ns;
    $display("Reset released at %t", $time);
    rst_ni = 1;
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
      .test_mode_i(1'b0),
      .boot_mode_i('0),
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
