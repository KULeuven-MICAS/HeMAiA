// Copyright 2025 KU Leuven.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51
// Yunhao Deng <yunhao.deng@kuleuven.be>


`timescale 1ns / 1ps
`include "axi/typedef.svh"

module testharness
  import occamy_pkg::*;
();

  localparam RTCTCK = 30.518us;        // 32.768 kHz
  localparam CLKTCK = 1ns;             // 1 GHz
  localparam PRITCK = 1ns;             // 1 GHz
  localparam int  SRAM_BANK = 16;      // 16 Banks architecture
  localparam int  SRAM_DEPTH = 1024;
  localparam int  SRAM_WIDTH = 8;      // 8 Bytes Wide

  // Driver regs and IO nets for DUT inout ports
  // Inout/output ports on the DUT must connect to nets (wire/tri). We drive them via separate regs.
  logic rtc_clk_drv, mst_clk_drv, periph_clk_drv, rst_ni_drv;
  wire  rtc_clk_i,  mst_clk_i,  periph_clk_i,  rst_ni;

  logic pll_bypass_drv;
  logic pll_en_drv;
  logic [1:0] pll_post_div_sel_drv;
  wire pll_bypass_i;
  wire pll_en_i;
  wire [1:0] pll_post_div_sel_i;
  wire pll_lock_o;
  assign pll_bypass_i = pll_bypass_drv;
  assign pll_en_i = pll_en_drv;
  assign pll_post_div_sel_i = pll_post_div_sel_drv;

  // Some blocks also expect a separate peripheral reset; tie it to rst_ni unless overridden
  wire  rst_periph_ni;
  assign rtc_clk_i     = rtc_clk_drv;
  assign mst_clk_i     = mst_clk_drv;
  assign periph_clk_i  = periph_clk_drv;
  assign rst_ni        = rst_ni_drv;
  assign rst_periph_ni = rst_ni;



  // Chip finish signal
  integer chip_finish[1:0][1:0];

  // The task to reset the chips , init clocks, and load the binaries.
  task automatic init_and_load();
    // Integer to save current time
    time current_time;
    // Init the chip_finish flags
    foreach (chip_finish[i,j]) begin
      chip_finish[i][j] = 0;
    end
    // Init the clk pins
    rtc_clk_drv    = 0;
    mst_clk_drv    = 0;
    periph_clk_drv = 0;

    // Init the reset pin
    rst_ni_drv = 0;

    // Init the PLL control
    pll_bypass_drv = 1'b0;
    pll_en_drv = 1'b0;
    pll_post_div_sel_drv = 2'b00;

   // Load the binaries
    i_hemaia_0_0.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[0].i_data_mem.i_tc_sram.load_data("app_chip_0_0/bank_0.hex");
    i_hemaia_0_0.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[1].i_data_mem.i_tc_sram.load_data("app_chip_0_0/bank_1.hex");
    i_hemaia_0_0.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[2].i_data_mem.i_tc_sram.load_data("app_chip_0_0/bank_2.hex");
    i_hemaia_0_0.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[3].i_data_mem.i_tc_sram.load_data("app_chip_0_0/bank_3.hex");
    i_hemaia_0_0.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[4].i_data_mem.i_tc_sram.load_data("app_chip_0_0/bank_4.hex");
    i_hemaia_0_0.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[5].i_data_mem.i_tc_sram.load_data("app_chip_0_0/bank_5.hex");
    i_hemaia_0_0.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[6].i_data_mem.i_tc_sram.load_data("app_chip_0_0/bank_6.hex");
    i_hemaia_0_0.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[7].i_data_mem.i_tc_sram.load_data("app_chip_0_0/bank_7.hex");
    i_hemaia_0_0.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[8].i_data_mem.i_tc_sram.load_data("app_chip_0_0/bank_8.hex");
    i_hemaia_0_0.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[9].i_data_mem.i_tc_sram.load_data("app_chip_0_0/bank_9.hex");
    i_hemaia_0_0.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[10].i_data_mem.i_tc_sram.load_data("app_chip_0_0/bank_10.hex");
    i_hemaia_0_0.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[11].i_data_mem.i_tc_sram.load_data("app_chip_0_0/bank_11.hex");
    i_hemaia_0_0.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[12].i_data_mem.i_tc_sram.load_data("app_chip_0_0/bank_12.hex");
    i_hemaia_0_0.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[13].i_data_mem.i_tc_sram.load_data("app_chip_0_0/bank_13.hex");
    i_hemaia_0_0.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[14].i_data_mem.i_tc_sram.load_data("app_chip_0_0/bank_14.hex");
    i_hemaia_0_0.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[15].i_data_mem.i_tc_sram.load_data("app_chip_0_0/bank_15.hex");
    i_hemaia_0_1.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[0].i_data_mem.i_tc_sram.load_data("app_chip_0_1/bank_0.hex");
    i_hemaia_0_1.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[1].i_data_mem.i_tc_sram.load_data("app_chip_0_1/bank_1.hex");
    i_hemaia_0_1.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[2].i_data_mem.i_tc_sram.load_data("app_chip_0_1/bank_2.hex");
    i_hemaia_0_1.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[3].i_data_mem.i_tc_sram.load_data("app_chip_0_1/bank_3.hex");
    i_hemaia_0_1.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[4].i_data_mem.i_tc_sram.load_data("app_chip_0_1/bank_4.hex");
    i_hemaia_0_1.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[5].i_data_mem.i_tc_sram.load_data("app_chip_0_1/bank_5.hex");
    i_hemaia_0_1.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[6].i_data_mem.i_tc_sram.load_data("app_chip_0_1/bank_6.hex");
    i_hemaia_0_1.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[7].i_data_mem.i_tc_sram.load_data("app_chip_0_1/bank_7.hex");
    i_hemaia_0_1.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[8].i_data_mem.i_tc_sram.load_data("app_chip_0_1/bank_8.hex");
    i_hemaia_0_1.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[9].i_data_mem.i_tc_sram.load_data("app_chip_0_1/bank_9.hex");
    i_hemaia_0_1.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[10].i_data_mem.i_tc_sram.load_data("app_chip_0_1/bank_10.hex");
    i_hemaia_0_1.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[11].i_data_mem.i_tc_sram.load_data("app_chip_0_1/bank_11.hex");
    i_hemaia_0_1.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[12].i_data_mem.i_tc_sram.load_data("app_chip_0_1/bank_12.hex");
    i_hemaia_0_1.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[13].i_data_mem.i_tc_sram.load_data("app_chip_0_1/bank_13.hex");
    i_hemaia_0_1.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[14].i_data_mem.i_tc_sram.load_data("app_chip_0_1/bank_14.hex");
    i_hemaia_0_1.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[15].i_data_mem.i_tc_sram.load_data("app_chip_0_1/bank_15.hex");
    i_hemaia_1_0.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[0].i_data_mem.i_tc_sram.load_data("app_chip_1_0/bank_0.hex");
    i_hemaia_1_0.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[1].i_data_mem.i_tc_sram.load_data("app_chip_1_0/bank_1.hex");
    i_hemaia_1_0.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[2].i_data_mem.i_tc_sram.load_data("app_chip_1_0/bank_2.hex");
    i_hemaia_1_0.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[3].i_data_mem.i_tc_sram.load_data("app_chip_1_0/bank_3.hex");
    i_hemaia_1_0.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[4].i_data_mem.i_tc_sram.load_data("app_chip_1_0/bank_4.hex");
    i_hemaia_1_0.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[5].i_data_mem.i_tc_sram.load_data("app_chip_1_0/bank_5.hex");
    i_hemaia_1_0.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[6].i_data_mem.i_tc_sram.load_data("app_chip_1_0/bank_6.hex");
    i_hemaia_1_0.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[7].i_data_mem.i_tc_sram.load_data("app_chip_1_0/bank_7.hex");
    i_hemaia_1_0.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[8].i_data_mem.i_tc_sram.load_data("app_chip_1_0/bank_8.hex");
    i_hemaia_1_0.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[9].i_data_mem.i_tc_sram.load_data("app_chip_1_0/bank_9.hex");
    i_hemaia_1_0.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[10].i_data_mem.i_tc_sram.load_data("app_chip_1_0/bank_10.hex");
    i_hemaia_1_0.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[11].i_data_mem.i_tc_sram.load_data("app_chip_1_0/bank_11.hex");
    i_hemaia_1_0.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[12].i_data_mem.i_tc_sram.load_data("app_chip_1_0/bank_12.hex");
    i_hemaia_1_0.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[13].i_data_mem.i_tc_sram.load_data("app_chip_1_0/bank_13.hex");
    i_hemaia_1_0.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[14].i_data_mem.i_tc_sram.load_data("app_chip_1_0/bank_14.hex");
    i_hemaia_1_0.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[15].i_data_mem.i_tc_sram.load_data("app_chip_1_0/bank_15.hex");
    i_hemaia_1_1.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[0].i_data_mem.i_tc_sram.load_data("app_chip_1_1/bank_0.hex");
    i_hemaia_1_1.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[1].i_data_mem.i_tc_sram.load_data("app_chip_1_1/bank_1.hex");
    i_hemaia_1_1.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[2].i_data_mem.i_tc_sram.load_data("app_chip_1_1/bank_2.hex");
    i_hemaia_1_1.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[3].i_data_mem.i_tc_sram.load_data("app_chip_1_1/bank_3.hex");
    i_hemaia_1_1.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[4].i_data_mem.i_tc_sram.load_data("app_chip_1_1/bank_4.hex");
    i_hemaia_1_1.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[5].i_data_mem.i_tc_sram.load_data("app_chip_1_1/bank_5.hex");
    i_hemaia_1_1.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[6].i_data_mem.i_tc_sram.load_data("app_chip_1_1/bank_6.hex");
    i_hemaia_1_1.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[7].i_data_mem.i_tc_sram.load_data("app_chip_1_1/bank_7.hex");
    i_hemaia_1_1.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[8].i_data_mem.i_tc_sram.load_data("app_chip_1_1/bank_8.hex");
    i_hemaia_1_1.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[9].i_data_mem.i_tc_sram.load_data("app_chip_1_1/bank_9.hex");
    i_hemaia_1_1.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[10].i_data_mem.i_tc_sram.load_data("app_chip_1_1/bank_10.hex");
    i_hemaia_1_1.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[11].i_data_mem.i_tc_sram.load_data("app_chip_1_1/bank_11.hex");
    i_hemaia_1_1.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[12].i_data_mem.i_tc_sram.load_data("app_chip_1_1/bank_12.hex");
    i_hemaia_1_1.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[13].i_data_mem.i_tc_sram.load_data("app_chip_1_1/bank_13.hex");
    i_hemaia_1_1.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[14].i_data_mem.i_tc_sram.load_data("app_chip_1_1/bank_14.hex");
    i_hemaia_1_1.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[15].i_data_mem.i_tc_sram.load_data("app_chip_1_1/bank_15.hex");
    // Some random delay
    #(37.13us);
    rst_ni_drv = 1;
    current_time = $time / 1000;
    $display("Reset released at %tns", $time / 1000);
  endtask

  // Call the reusable init task from an initial block.
  initial begin
    init_and_load();
  end

  // Trigger the reusbale init task when reload_bin becomes high
  logic reload_bin = '0;
  always @(posedge reload_bin) begin
    init_and_load();
    reload_bin = '0;
  end

  always_comb begin
    automatic integer allFinished = 1;
    automatic integer allCorrect = 1;
    if (chip_finish[0][0] == 0) begin
        allFinished = 0;
    end
    if (chip_finish[0][0] == -1) begin
        allCorrect = 0;
    end
    if (chip_finish[0][1] == 0) begin
        allFinished = 0;
    end
    if (chip_finish[0][1] == -1) begin
        allCorrect = 0;
    end
    if (chip_finish[1][0] == 0) begin
        allFinished = 0;
    end
    if (chip_finish[1][0] == -1) begin
        allCorrect = 0;
    end
    if (chip_finish[1][1] == 0) begin
        allFinished = 0;
    end
    if (chip_finish[1][1] == -1) begin
        allCorrect = 0;
    end
    if (allFinished == 1) begin
      if (allCorrect == 1) begin
        $display("All chips finished successfully at %tns", $time / 1000);
        $finish;
      end else begin
        $error("All chips finished with errors at %tns", $time / 1000);
        $finish(-1);
      end
    end
  end

  always #(RTCTCK / 2.0) begin
    rtc_clk_drv = ~rtc_clk_drv;
  end

  always #(CLKTCK / 2.0) begin
    mst_clk_drv = ~mst_clk_drv;
  end

  always #(PRITCK / 2.0) begin
    periph_clk_drv = ~periph_clk_drv;
  end

  // Must be the frequency of i_uart0.clk_i in Hz
  localparam int unsigned UartDPIFreq = 1_000_000_000;

  // Definition of tri_state bus
  tri [2:0][19:0] chip_0_0_to_1_0_link;
  wire chip_0_0_to_1_0_link_rts;
  wire chip_0_0_to_1_0_link_cts;
  wire chip_1_0_to_0_0_link_rts;
  wire chip_1_0_to_0_0_link_cts;
  wire chip_0_0_to_1_0_link_test_request;
  wire chip_1_0_to_0_0_link_test_request;
  tri [2:0][19:0] chip_0_0_to_0_1_link;
  wire chip_0_0_to_0_1_link_rts;
  wire chip_0_0_to_0_1_link_cts;
  wire chip_0_1_to_0_0_link_rts;
  wire chip_0_1_to_0_0_link_cts;
  wire chip_0_0_to_0_1_link_test_request;
  wire chip_0_1_to_0_0_link_test_request;

  tri [2:0][19:0] chip_0_1_to_1_1_link;
  wire chip_0_1_to_1_1_link_rts;
  wire chip_0_1_to_1_1_link_cts;
  wire chip_1_1_to_0_1_link_rts;
  wire chip_1_1_to_0_1_link_cts;
  wire chip_0_1_to_1_1_link_test_request;
  wire chip_1_1_to_0_1_link_test_request;
  tri [2:0][19:0] chip_0_1_to_0_2_link;
  wire chip_0_1_to_0_2_link_rts;
  wire chip_0_1_to_0_2_link_cts;
  wire chip_0_2_to_0_1_link_rts;
  wire chip_0_2_to_0_1_link_cts;
  wire chip_0_1_to_0_2_link_test_request;
  wire chip_0_2_to_0_1_link_test_request;

  tri [2:0][19:0] chip_1_0_to_2_0_link;
  wire chip_1_0_to_2_0_link_rts;
  wire chip_1_0_to_2_0_link_cts;
  wire chip_2_0_to_1_0_link_rts;
  wire chip_2_0_to_1_0_link_cts;
  wire chip_1_0_to_2_0_link_test_request;
  wire chip_2_0_to_1_0_link_test_request;
  tri [2:0][19:0] chip_1_0_to_1_1_link;
  wire chip_1_0_to_1_1_link_rts;
  wire chip_1_0_to_1_1_link_cts;
  wire chip_1_1_to_1_0_link_rts;
  wire chip_1_1_to_1_0_link_cts;
  wire chip_1_0_to_1_1_link_test_request;
  wire chip_1_1_to_1_0_link_test_request;

  tri [2:0][19:0] chip_1_1_to_2_1_link;
  wire chip_1_1_to_2_1_link_rts;
  wire chip_1_1_to_2_1_link_cts;
  wire chip_2_1_to_1_1_link_rts;
  wire chip_2_1_to_1_1_link_cts;
  wire chip_1_1_to_2_1_link_test_request;
  wire chip_2_1_to_1_1_link_test_request;
  tri [2:0][19:0] chip_1_1_to_1_2_link;
  wire chip_1_1_to_1_2_link_rts;
  wire chip_1_1_to_1_2_link_cts;
  wire chip_1_2_to_1_1_link_rts;
  wire chip_1_2_to_1_1_link_cts;
  wire chip_1_1_to_1_2_link_test_request;
  wire chip_1_2_to_1_1_link_test_request;

  tri [2:0][19:0] chip_2_0_to_3_0_link;
  wire chip_2_0_to_3_0_link_rts;
  wire chip_2_0_to_3_0_link_cts;
  wire chip_3_0_to_2_0_link_rts;
  wire chip_3_0_to_2_0_link_cts;
  wire chip_2_0_to_3_0_link_test_request;
  wire chip_3_0_to_2_0_link_test_request;
  tri [2:0][19:0] chip_2_0_to_2_1_link;
  wire chip_2_0_to_2_1_link_rts;
  wire chip_2_0_to_2_1_link_cts;
  wire chip_2_1_to_2_0_link_rts;
  wire chip_2_1_to_2_0_link_cts;
  wire chip_2_0_to_2_1_link_test_request;
  wire chip_2_1_to_2_0_link_test_request;


  // Instatiate Chips
  wire const_zero;
  wire const_one;
  assign const_zero = 1'b0;
  assign const_one  = 1'b1;


  wire [7:0] chip_id_0_0;
  assign chip_id_0_0 = 8'h00;
  // Uart signals (must be nets when connected to DUT inout/output)
  wire tx_0_0, rx_0_0;

  hemaia i_hemaia_0_0 (
      .io_clk_i(mst_clk_i),
      .io_rst_ni(rst_ni),
      .io_pll_bypass_i(pll_bypass_i),
      .io_pll_en_i (pll_en_i),
      .io_pll_post_div_sel_i(pll_post_div_sel_i),
      .io_pll_lock_o(pll_lock_o),
      .io_clk_obs_o(),
      .io_clk_periph_i(periph_clk_i),
      .io_rst_periph_ni(rst_periph_ni),
      .io_test_mode_i(const_zero),
      .io_chip_id_i(chip_id_0_0),
      .io_boot_mode_i(const_zero),
      .io_east_d2d(chip_0_0_to_1_0_link),
      .io_flow_control_east_rts_o(chip_0_0_to_1_0_link_rts),
      .io_flow_control_east_cts_i(chip_0_0_to_1_0_link_cts),
      .io_flow_control_east_rts_i(chip_1_0_to_0_0_link_rts),
      .io_flow_control_east_cts_o(chip_1_0_to_0_0_link_cts),
      .io_east_test_being_requested_i(chip_1_0_to_0_0_link_test_request),
      .io_east_test_request_o(chip_0_0_to_1_0_link_test_request),
      .io_west_d2d(),
      .io_flow_control_west_rts_o(),
      .io_flow_control_west_cts_i(const_one),
      .io_flow_control_west_rts_i(const_zero),
      .io_flow_control_west_cts_o(),
      .io_west_test_being_requested_i(const_zero),
      .io_west_test_request_o(),
      .io_north_d2d(),
      .io_flow_control_north_rts_o(),
      .io_flow_control_north_cts_i(const_one),
      .io_flow_control_north_rts_i(const_zero),
      .io_flow_control_north_cts_o(),
      .io_north_test_being_requested_i(const_zero),
      .io_north_test_request_o(),
      .io_south_d2d(chip_0_0_to_0_1_link),
      .io_flow_control_south_rts_o(chip_0_0_to_0_1_link_rts),
      .io_flow_control_south_cts_i(chip_0_0_to_0_1_link_cts),
      .io_flow_control_south_rts_i(chip_0_1_to_0_0_link_rts),
      .io_flow_control_south_cts_o(chip_0_1_to_0_0_link_cts),
      .io_south_test_being_requested_i(chip_0_1_to_0_0_link_test_request),
      .io_south_test_request_o(chip_0_0_to_0_1_link_test_request),
      .io_uart_tx_o(tx_0_0),
      .io_uart_rx_i(rx_0_0),
      .io_uart_rts_no(),
      .io_uart_cts_ni(const_zero),
      .io_gpio(),
      .io_spim_sck_o(),
      .io_spim_csb_o(),
      .io_spim_sd(),
      .io_i2c_sda(),
      .io_i2c_scl(),
      .io_jtag_trst_ni(const_zero),
      .io_jtag_tck_i(const_zero),
      .io_jtag_tms_i(const_zero),
      .io_jtag_tdi_i(const_zero),
      .io_jtag_tdo_o()
  );

  uartdpi #(
      .BAUD('d31250000),
      // Frequency shouldn't matter since we are sending with the same clock.
      .FREQ(UartDPIFreq),
      .NAME("uart_0_0")
  ) i_uart_0_0 (
      .clk_i (periph_clk_i),
      .rst_ni(rst_ni),
      .tx_o  (rx_0_0),
      .rx_i  (tx_0_0)
  );


  wire [7:0] chip_id_0_1;
  assign chip_id_0_1 = 8'h01;
  // Uart signals (must be nets when connected to DUT inout/output)
  wire tx_0_1, rx_0_1;

  hemaia i_hemaia_0_1 (
      .io_clk_i(mst_clk_i),
      .io_rst_ni(rst_ni),
      .io_pll_bypass_i(pll_bypass_i),
      .io_pll_en_i (pll_en_i),
      .io_pll_post_div_sel_i(pll_post_div_sel_i),
      .io_pll_lock_o(pll_lock_o),
      .io_clk_obs_o(),
      .io_clk_periph_i(periph_clk_i),
      .io_rst_periph_ni(rst_periph_ni),
      .io_test_mode_i(const_zero),
      .io_chip_id_i(chip_id_0_1),
      .io_boot_mode_i(const_zero),
      .io_east_d2d(chip_0_1_to_1_1_link),
      .io_flow_control_east_rts_o(chip_0_1_to_1_1_link_rts),
      .io_flow_control_east_cts_i(chip_0_1_to_1_1_link_cts),
      .io_flow_control_east_rts_i(chip_1_1_to_0_1_link_rts),
      .io_flow_control_east_cts_o(chip_1_1_to_0_1_link_cts),
      .io_east_test_being_requested_i(chip_1_1_to_0_1_link_test_request),
      .io_east_test_request_o(chip_0_1_to_1_1_link_test_request),
      .io_west_d2d(),
      .io_flow_control_west_rts_o(),
      .io_flow_control_west_cts_i(const_one),
      .io_flow_control_west_rts_i(const_zero),
      .io_flow_control_west_cts_o(),
      .io_west_test_being_requested_i(const_zero),
      .io_west_test_request_o(),
      .io_north_d2d(chip_0_0_to_0_1_link),
      .io_flow_control_north_rts_o(chip_0_1_to_0_0_link_rts),
      .io_flow_control_north_cts_i(chip_0_1_to_0_0_link_cts),
      .io_flow_control_north_rts_i(chip_0_0_to_0_1_link_rts),
      .io_flow_control_north_cts_o(chip_0_0_to_0_1_link_cts),
      .io_north_test_being_requested_i(chip_0_0_to_0_1_link_test_request),
      .io_north_test_request_o(chip_0_1_to_0_0_link_test_request),
      .io_south_d2d(),
      .io_flow_control_south_rts_o(),
      .io_flow_control_south_cts_i(const_one),
      .io_flow_control_south_rts_i(const_zero),
      .io_flow_control_south_cts_o(),
      .io_south_test_being_requested_i(const_zero),
      .io_south_test_request_o(),
      .io_uart_tx_o(tx_0_1),
      .io_uart_rx_i(rx_0_1),
      .io_uart_rts_no(),
      .io_uart_cts_ni(const_zero),
      .io_gpio(),
      .io_spim_sck_o(),
      .io_spim_csb_o(),
      .io_spim_sd(),
      .io_i2c_sda(),
      .io_i2c_scl(),
      .io_jtag_trst_ni(const_zero),
      .io_jtag_tck_i(const_zero),
      .io_jtag_tms_i(const_zero),
      .io_jtag_tdi_i(const_zero),
      .io_jtag_tdo_o()
  );

  uartdpi #(
      .BAUD('d31250000),
      // Frequency shouldn't matter since we are sending with the same clock.
      .FREQ(UartDPIFreq),
      .NAME("uart_0_1")
  ) i_uart_0_1 (
      .clk_i (periph_clk_i),
      .rst_ni(rst_ni),
      .tx_o  (rx_0_1),
      .rx_i  (tx_0_1)
  );


  wire [7:0] chip_id_1_0;
  assign chip_id_1_0 = 8'h10;
  // Uart signals (must be nets when connected to DUT inout/output)
  wire tx_1_0, rx_1_0;

  hemaia i_hemaia_1_0 (
      .io_clk_i(mst_clk_i),
      .io_rst_ni(rst_ni),
      .io_pll_bypass_i(pll_bypass_i),
      .io_pll_en_i (pll_en_i),
      .io_pll_post_div_sel_i(pll_post_div_sel_i),
      .io_pll_lock_o(pll_lock_o),
      .io_clk_obs_o(),
      .io_clk_periph_i(periph_clk_i),
      .io_rst_periph_ni(rst_periph_ni),
      .io_test_mode_i(const_zero),
      .io_chip_id_i(chip_id_1_0),
      .io_boot_mode_i(const_zero),
      .io_east_d2d(chip_1_0_to_2_0_link),
      .io_flow_control_east_rts_o(chip_1_0_to_2_0_link_rts),
      .io_flow_control_east_cts_i(chip_1_0_to_2_0_link_cts),
      .io_flow_control_east_rts_i(chip_2_0_to_1_0_link_rts),
      .io_flow_control_east_cts_o(chip_2_0_to_1_0_link_cts),
      .io_east_test_being_requested_i(chip_2_0_to_1_0_link_test_request),
      .io_east_test_request_o(chip_1_0_to_2_0_link_test_request),
      .io_west_d2d(chip_0_0_to_1_0_link),
      .io_flow_control_west_rts_o(chip_1_0_to_0_0_link_rts),
      .io_flow_control_west_cts_i(chip_1_0_to_0_0_link_cts),
      .io_flow_control_west_rts_i(chip_0_0_to_1_0_link_rts),
      .io_flow_control_west_cts_o(chip_0_0_to_1_0_link_cts),
      .io_west_test_being_requested_i(chip_0_0_to_1_0_link_test_request),
      .io_west_test_request_o(chip_1_0_to_0_0_link_test_request),
      .io_north_d2d(),
      .io_flow_control_north_rts_o(),
      .io_flow_control_north_cts_i(const_one),
      .io_flow_control_north_rts_i(const_zero),
      .io_flow_control_north_cts_o(),
      .io_north_test_being_requested_i(const_zero),
      .io_north_test_request_o(),
      .io_south_d2d(chip_1_0_to_1_1_link),
      .io_flow_control_south_rts_o(chip_1_0_to_1_1_link_rts),
      .io_flow_control_south_cts_i(chip_1_0_to_1_1_link_cts),
      .io_flow_control_south_rts_i(chip_1_1_to_1_0_link_rts),
      .io_flow_control_south_cts_o(chip_1_1_to_1_0_link_cts),
      .io_south_test_being_requested_i(chip_1_1_to_1_0_link_test_request),
      .io_south_test_request_o(chip_1_0_to_1_1_link_test_request),
      .io_uart_tx_o(tx_1_0),
      .io_uart_rx_i(rx_1_0),
      .io_uart_rts_no(),
      .io_uart_cts_ni(const_zero),
      .io_gpio(),
      .io_spim_sck_o(),
      .io_spim_csb_o(),
      .io_spim_sd(),
      .io_i2c_sda(),
      .io_i2c_scl(),
      .io_jtag_trst_ni(const_zero),
      .io_jtag_tck_i(const_zero),
      .io_jtag_tms_i(const_zero),
      .io_jtag_tdi_i(const_zero),
      .io_jtag_tdo_o()
  );

  uartdpi #(
      .BAUD('d31250000),
      // Frequency shouldn't matter since we are sending with the same clock.
      .FREQ(UartDPIFreq),
      .NAME("uart_1_0")
  ) i_uart_1_0 (
      .clk_i (periph_clk_i),
      .rst_ni(rst_ni),
      .tx_o  (rx_1_0),
      .rx_i  (tx_1_0)
  );


  wire [7:0] chip_id_1_1;
  assign chip_id_1_1 = 8'h11;
  // Uart signals (must be nets when connected to DUT inout/output)
  wire tx_1_1, rx_1_1;

  hemaia i_hemaia_1_1 (
      .io_clk_i(mst_clk_i),
      .io_rst_ni(rst_ni),
      .io_pll_bypass_i(pll_bypass_i),
      .io_pll_en_i (pll_en_i),
      .io_pll_post_div_sel_i(pll_post_div_sel_i),
      .io_pll_lock_o(pll_lock_o),
      .io_clk_obs_o(),
      .io_clk_periph_i(periph_clk_i),
      .io_rst_periph_ni(rst_periph_ni),
      .io_test_mode_i(const_zero),
      .io_chip_id_i(chip_id_1_1),
      .io_boot_mode_i(const_zero),
      .io_east_d2d(),
      .io_flow_control_east_rts_o(),
      .io_flow_control_east_cts_i(const_one),
      .io_flow_control_east_rts_i(const_zero),
      .io_flow_control_east_cts_o(),
      .io_east_test_being_requested_i(const_zero),
      .io_east_test_request_o(),
      .io_west_d2d(chip_0_1_to_1_1_link),
      .io_flow_control_west_rts_o(chip_1_1_to_0_1_link_rts),
      .io_flow_control_west_cts_i(chip_1_1_to_0_1_link_cts),
      .io_flow_control_west_rts_i(chip_0_1_to_1_1_link_rts),
      .io_flow_control_west_cts_o(chip_0_1_to_1_1_link_cts),
      .io_west_test_being_requested_i(chip_0_1_to_1_1_link_test_request),
      .io_west_test_request_o(chip_1_1_to_0_1_link_test_request),
      .io_north_d2d(chip_1_0_to_1_1_link),
      .io_flow_control_north_rts_o(chip_1_1_to_1_0_link_rts),
      .io_flow_control_north_cts_i(chip_1_1_to_1_0_link_cts),
      .io_flow_control_north_rts_i(chip_1_0_to_1_1_link_rts),
      .io_flow_control_north_cts_o(chip_1_0_to_1_1_link_cts),
      .io_north_test_being_requested_i(chip_1_0_to_1_1_link_test_request),
      .io_north_test_request_o(chip_1_1_to_1_0_link_test_request),
      .io_south_d2d(),
      .io_flow_control_south_rts_o(),
      .io_flow_control_south_cts_i(const_one),
      .io_flow_control_south_rts_i(const_zero),
      .io_flow_control_south_cts_o(),
      .io_south_test_being_requested_i(const_zero),
      .io_south_test_request_o(),
      .io_uart_tx_o(tx_1_1),
      .io_uart_rx_i(rx_1_1),
      .io_uart_rts_no(),
      .io_uart_cts_ni(const_zero),
      .io_gpio(),
      .io_spim_sck_o(),
      .io_spim_csb_o(),
      .io_spim_sd(),
      .io_i2c_sda(),
      .io_i2c_scl(),
      .io_jtag_trst_ni(const_zero),
      .io_jtag_tck_i(const_zero),
      .io_jtag_tms_i(const_zero),
      .io_jtag_tdi_i(const_zero),
      .io_jtag_tdo_o()
  );

  uartdpi #(
      .BAUD('d31250000),
      // Frequency shouldn't matter since we are sending with the same clock.
      .FREQ(UartDPIFreq),
      .NAME("uart_1_1")
  ) i_uart_1_1 (
      .clk_i (periph_clk_i),
      .rst_ni(rst_ni),
      .tx_o  (rx_1_1),
      .rx_i  (tx_1_1)
  );


  wire [7:0] chip_id_2_0;
  assign chip_id_2_0 = 8'h20;
  hemaia_mem_chip #(
    .WideSRAMBankNum(16),
    .WideSRAMSize(67108864),
    .EnableEastPhy(1'b0),
    .EnableWestPhy(1'b1),
    .EnableNorthPhy(1'b0),
    .EnableSouthPhy(1'b0)
  ) i_hemaia_mem_2_0 (
      .clk_i(mst_clk_i),
      .rst_ni(rst_ni),
      .chip_id_i(chip_id_2_0),
      .east_d2d_io(),
      .flow_control_east_rts_o(),
      .flow_control_east_cts_i(const_zero),
      .flow_control_east_rts_i(const_zero),
      .flow_control_east_cts_o(),
      .east_test_being_requested_i(const_zero),
      .east_test_request_o(),
      .west_d2d_io(chip_1_0_to_2_0_link),
      .flow_control_west_rts_o(chip_2_0_to_1_0_link_rts),
      .flow_control_west_cts_i(chip_2_0_to_1_0_link_cts),
      .flow_control_west_rts_i(chip_1_0_to_2_0_link_rts),
      .flow_control_west_cts_o(chip_1_0_to_2_0_link_cts),
      .west_test_being_requested_i(chip_1_0_to_2_0_link_test_request),
      .west_test_request_o(chip_2_0_to_1_0_link_test_request),
      .north_d2d_io(),
      .flow_control_north_rts_o(),
      .flow_control_north_cts_i(const_zero),
      .flow_control_north_rts_i(const_zero),
      .flow_control_north_cts_o(),
      .north_test_being_requested_i(const_zero),
      .north_test_request_o(),
      .south_d2d_io(),
      .flow_control_south_rts_o(),
      .flow_control_south_cts_i(const_zero),
      .flow_control_south_rts_i(const_zero),
      .flow_control_south_cts_o(),
      .south_test_being_requested_i(const_zero),
      .south_test_request_o()
  );
endmodule
