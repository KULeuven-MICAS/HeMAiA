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
  localparam CLKTCK = 1ns;             // 1 GHz
  localparam PRITCK = 1ns;             // 1 GHz
  localparam int  SRAM_BANK = ${mem_bank};      // ${mem_bank} Banks architecture
  localparam int  SRAM_DEPTH = ${int(mem_size/8/mem_bank)};
  localparam int  SRAM_WIDTH = 8;      // 8 Bytes Wide

  // Driver regs and IO nets for DUT inout ports
  // Inout/output ports on the DUT must connect to nets (wire/tri). We drive them via separate regs.
  logic rtc_clk_drv, mst_clk_drv, periph_clk_drv, rst_ni_drv;
  wire  rtc_clk_i,  mst_clk_i,  periph_clk_i,  rst_ni;
  // Some blocks also expect a separate peripheral reset; tie it to rst_ni unless overridden
  wire  rst_periph_ni;
  assign rtc_clk_i     = rtc_clk_drv;
  assign mst_clk_i     = mst_clk_drv;
  assign periph_clk_i  = periph_clk_drv;
  assign rst_ni        = rst_ni_drv;
  assign rst_periph_ni = rst_ni;

  // Chip finish signal
  integer chip_finish[${max(x)}:${min(x)}][${max(y)}:${min(y)}];

  // Integer to save current time
  time current_time;

  // Generate reset and clock.
  initial begin
    // Init the chip_finish flags
    foreach (chip_finish[i,j]) begin
      chip_finish[i][j] = 0;
    end
    // Init the clk pins
  rtc_clk_drv    = 0;
  mst_clk_drv    = 0;
  periph_clk_drv = 0;
    // Init the reset pin
  rst_ni_drv = 1;
    #0;
  rst_ni_drv = 0;
   // Load the binaries
% for i in x:
%   for j in y:
%     for k in range(0, mem_bank):
%       if netlist is True:
    i_hemaia_${i}_${j}.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks_${k}__i_data_mem.i_tc_sram.gen_mem_gen_${int(mem_size/8/mem_bank)}x${64}_u_sram.load_data("app_chip_${i}_${j}/bank_${k}.hex");
%       elif mem_macro is True:
    i_hemaia_${i}_${j}.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[${k}].i_data_mem.i_tc_sram.gen_mem.gen_${int(mem_size/8/mem_bank)}x${64}.u_sram.load_data("app_chip_${i}_${j}/bank_${k}.hex");
%       else:
    i_hemaia_${i}_${j}.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[${k}].i_data_mem.i_tc_sram.load_data("app_chip_${i}_${j}/bank_${k}.hex");
%       endif
%     endfor
%   endfor
% endfor
    // Release the reset
    #(10 + $urandom % 10);
    current_time = $time / 1000;
    $display("Reset released at %tns", current_time);
    rst_ni_drv = 1;
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
    rtc_clk_drv = ~rtc_clk_drv;
  end
  
  always #(CLKTCK / 2) begin
    mst_clk_drv = ~mst_clk_drv;
  end

  always #(PRITCK / 2) begin
    periph_clk_drv = ~periph_clk_drv;
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
  wire const_zero;
  wire const_one;
  assign const_zero = 1'b0;
  assign const_one  = 1'b1;

% for i in x:
%   for j in y:
<%
    i_hex_string = "{:01x}".format(i)
    j_hex_string = "{:01x}".format(j)
%>
  /// Uart signals (must be nets when connected to DUT inout/output)
  wire tx_${i}_${j}, rx_${i}_${j};

% if multichip_cfg['single_chip'] is False:
  /// Die2Die signals
  // East side
  wire chip_${i}_${j}_link_from_east_rts;
  wire chip_${i}_${j}_link_from_east_cts;
  wire chip_${i}_${j}_link_to_east_rts;
  wire chip_${i}_${j}_link_to_east_cts;
  wire chip_${i}_${j}_link_from_east_test_request;
  wire chip_${i}_${j}_link_to_east_test_request;
  // West side
  wire chip_${i}_${j}_link_from_west_rts;
  wire chip_${i}_${j}_link_from_west_cts;
  wire chip_${i}_${j}_link_to_west_rts;
  wire chip_${i}_${j}_link_to_west_cts;
  wire chip_${i}_${j}_link_from_west_test_request;
  wire chip_${i}_${j}_link_to_west_test_request;
  // North side
  wire chip_${i}_${j}_link_from_north_rts;
  wire chip_${i}_${j}_link_from_north_cts;
  wire chip_${i}_${j}_link_to_north_rts;
  wire chip_${i}_${j}_link_to_north_cts;
  wire chip_${i}_${j}_link_from_north_test_request;
  wire chip_${i}_${j}_link_to_north_test_request;
  // South side
  wire chip_${i}_${j}_link_from_south_rts;
  wire chip_${i}_${j}_link_from_south_cts;
  wire chip_${i}_${j}_link_to_south_rts;
  wire chip_${i}_${j}_link_to_south_cts;
  wire chip_${i}_${j}_link_from_south_test_request;
  wire chip_${i}_${j}_link_to_south_test_request;
% endif

  wire [7:0] chip_id_${i}_${j};
  assign chip_id_${i}_${j} = 8'h${i_hex_string}${j_hex_string};

  hemaia i_hemaia_${i}_${j} (
      .io_clk_i(mst_clk_i),
      .io_rst_ni(rst_ni),
      .io_clk_periph_i(periph_clk_i),
      .io_rst_periph_ni(rst_periph_ni),
      .io_test_mode_i(const_zero),
      .io_chip_id_i(chip_id_${i}_${j}),
      .io_boot_mode_i(const_zero),
% if multichip_cfg['single_chip'] is False:
%   if i < max(x):
      .io_east_d2d(chip_${i}_${j}_to_${i+1}_${j}_link),
%   else:
      .io_east_d2d(),
%   endif
      .io_flow_control_east_rts_o(chip_${i}_${j}_link_to_east_rts),
      .io_flow_control_east_cts_i(chip_${i}_${j}_link_to_east_cts),
      .io_flow_control_east_rts_i(chip_${i}_${j}_link_from_east_rts),
      .io_flow_control_east_cts_o(chip_${i}_${j}_link_from_east_cts),
      .io_east_test_being_requested_i(chip_${i}_${j}_link_to_east_test_request),
      .io_east_test_request_o(chip_${i}_${j}_link_from_east_test_request),

%   if i > min(x):
      .io_west_d2d(chip_${i-1}_${j}_to_${i}_${j}_link),
%   else:
      .io_west_d2d(),
%   endif
      .io_flow_control_west_rts_o(chip_${i}_${j}_link_to_west_rts),
      .io_flow_control_west_cts_i(chip_${i}_${j}_link_to_west_cts),
      .io_flow_control_west_rts_i(chip_${i}_${j}_link_from_west_rts),
      .io_flow_control_west_cts_o(chip_${i}_${j}_link_from_west_cts),
      .io_west_test_being_requested_i(chip_${i}_${j}_link_to_west_test_request),
      .io_west_test_request_o(chip_${i}_${j}_link_from_west_test_request),

%   if j > min(y):
      .io_north_d2d(chip_${i}_${j-1}_to_${i}_${j}_link),
%   else:
      .io_north_d2d(),
%   endif
      .io_flow_control_north_rts_o(chip_${i}_${j}_link_to_north_rts),
      .io_flow_control_north_cts_i(chip_${i}_${j}_link_to_north_cts),
      .io_flow_control_north_rts_i(chip_${i}_${j}_link_from_north_rts),
      .io_flow_control_north_cts_o(chip_${i}_${j}_link_from_north_cts),
      .io_north_test_being_requested_i(chip_${i}_${j}_link_to_north_test_request),
      .io_north_test_request_o(chip_${i}_${j}_link_from_north_test_request),

%   if j < max(y):
      .io_south_d2d(chip_${i}_${j}_to_${i}_${j+1}_link),
%   else:
      .io_south_d2d(),
%   endif
      .io_flow_control_south_rts_o(chip_${i}_${j}_link_to_south_rts),
      .io_flow_control_south_cts_i(chip_${i}_${j}_link_to_south_cts),
      .io_flow_control_south_rts_i(chip_${i}_${j}_link_from_south_rts),
      .io_flow_control_south_cts_o(chip_${i}_${j}_link_from_south_cts),
      .io_south_test_being_requested_i(chip_${i}_${j}_link_to_south_test_request),
      .io_south_test_request_o(chip_${i}_${j}_link_from_south_test_request),
% endif
      .io_uart_tx_o(tx_${i}_${j}),
      .io_uart_rx_i(rx_${i}_${j}),
      .io_uart_rts_no(),
      .io_uart_cts_ni(const_zero),
      .io_gpio(),
      .io_spis_sck_i(),
      .io_spis_csb_i(),
      .io_spis_sd(),
      .io_spim_sck_o(),
      .io_spim_csb_o(),
      .io_spim_sd(),
      .io_jtag_trst_ni(const_zero),
      .io_jtag_tck_i(const_zero),
      .io_jtag_tms_i(const_zero),
      .io_jtag_tdi_i(const_zero),
      .io_jtag_tdo_o(),
      .io_i2c_sda(),
      .io_i2c_scl()
  );

  uartdpi #(
      .BAUD('d31250000),
      // Frequency shouldn't matter since we are sending with the same clock.
      .FREQ(UartDPIFreq),
      .NAME("uart_${i}_${j}")
  ) i_uart_${i}_${j} (
      .clk_i (periph_clk_i),
      .rst_ni(rst_ni),
      .tx_o  (rx_${i}_${j}),
      .rx_i  (tx_${i}_${j})
  );

% if mem_macro is False and netlist is False:
  // Chip Status Monitor Block
  always @(i_hemaia_${i}_${j}.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[${mem_bank-1}].i_data_mem.i_tc_sram.sram[SRAM_DEPTH-1][(SRAM_WIDTH*8-1)-:32]) begin
    if (i_hemaia_${i}_${j}.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[${mem_bank-1}].i_data_mem.i_tc_sram.sram[SRAM_DEPTH-1][(SRAM_WIDTH*8-1)-:32] != 0) begin
      if (i_hemaia_${i}_${j}.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[${mem_bank-1}].i_data_mem.i_tc_sram.sram[SRAM_DEPTH-1][(SRAM_WIDTH*8-1)-:32] == 32'd1) begin
        $display("Simulation of chip_${i}_${j} is finished at %tns", $time / 1000);
        chip_finish[${i}][${j}] = 1;
      end else begin
        $error("Simulation of chip_${i}_${j} is finished with errors %d at %tns",
               i_hemaia_${i}_${j}.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[${mem_bank-1}].i_data_mem.i_tc_sram.sram[SRAM_DEPTH-1][(SRAM_WIDTH*8-1)-:32],
               $time / 1000);
        chip_finish[${i}][${j}] = -1;
      end
    end
  end
% endif

%   endfor
% endfor

% if multichip_cfg['single_chip'] is False:
// Connect the control signals for chip ${i}_${j} to the adjacent chips
% for i in x:
%   for j in y:
  // Connect the west side of the chip
%     if i == min(x):
  assign chip_${i}_${j}_link_to_west_cts = '0;
  assign chip_${i}_${j}_link_from_west_rts = '0;
  assign chip_${i}_${j}_link_to_west_test_request = '0;
%     else:
  assign chip_${i}_${j}_link_to_west_cts = chip_${i-1}_${j}_link_from_east_cts;
  assign chip_${i}_${j}_link_from_west_rts = chip_${i-1}_${j}_link_to_east_rts;
  assign chip_${i}_${j}_link_to_west_test_request = chip_${i-1}_${j}_link_from_east_test_request;
%     endif
  // Connect the east side of the chip
%     if i == max(x):
  assign chip_${i}_${j}_link_to_east_cts = '0;
  assign chip_${i}_${j}_link_from_east_rts = '0;
  assign chip_${i}_${j}_link_to_east_test_request = '0;
%     else:
  assign chip_${i}_${j}_link_to_east_cts = chip_${i+1}_${j}_link_from_west_cts;
  assign chip_${i}_${j}_link_from_east_rts = chip_${i+1}_${j}_link_to_west_rts;
  assign chip_${i}_${j}_link_to_east_test_request = chip_${i+1}_${j}_link_from_west_test_request;
%     endif
  // Connect the north side of the chip
%     if j == min(y):
  assign chip_${i}_${j}_link_to_north_cts = '0;
  assign chip_${i}_${j}_link_from_north_rts = '0;
  assign chip_${i}_${j}_link_to_north_test_request = '0;
%     else:
  assign chip_${i}_${j}_link_to_north_cts = chip_${i}_${j-1}_link_from_south_cts;
  assign chip_${i}_${j}_link_from_north_rts = chip_${i}_${j-1}_link_to_south_rts;
  assign chip_${i}_${j}_link_to_north_test_request = chip_${i}_${j-1}_link_from_south_test_request;
%     endif
  // Connect the south side of the chip
%     if j == max(y):
  assign chip_${i}_${j}_link_to_south_cts = '0;
  assign chip_${i}_${j}_link_from_south_rts = '0;
  assign chip_${i}_${j}_link_to_south_test_request = '0;
%     else:
  assign chip_${i}_${j}_link_to_south_cts = chip_${i}_${j+1}_link_from_north_cts;
  assign chip_${i}_${j}_link_from_south_rts = chip_${i}_${j+1}_link_to_north_rts;
  assign chip_${i}_${j}_link_to_south_test_request = chip_${i}_${j+1}_link_from_north_test_request;
%     endif
%   endfor
% endfor
% endif
endmodule
