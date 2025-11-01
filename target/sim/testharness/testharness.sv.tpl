// Copyright 2025 KU Leuven.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51
// Yunhao Deng <yunhao.deng@kuleuven.be>

<%
from enum import Enum
from dataclasses import dataclass
from typing import Tuple

class ChipletType(Enum):
    COMPUTE = "compute"
    MEMORY = "memory"

@dataclass
class Chiplet:
    coordinate: Tuple[int, int]  # (x, y)
    type: ChipletType
    size: int = 0  # Only for memory chiplets

chip_coordinates = []

if multichip_cfg["single_chip"] is True:
    chip_coordinates.append(Chiplet(coordinate=(0, 0), type=ChipletType.COMPUTE))
else:
    for chip in multichip_cfg["testbench_cfg"]["hemaia_compute_chip"]:
        chip_coordinates.append(Chiplet(coordinate=(chip["coordinate"][0], chip["coordinate"][1]), type=ChipletType.COMPUTE))
    for memory in multichip_cfg["testbench_cfg"]["hemaia_mem_chip"]:
        chip_coordinates.append(Chiplet(coordinate=(memory["coordinate"][0], memory["coordinate"][1]), type=ChipletType.MEMORY, size=memory["mem_size"]))

max_compute_chiplet_x = max([chip.coordinate[0] for chip in chip_coordinates if chip.type == ChipletType.COMPUTE])
max_compute_chiplet_y = max([chip.coordinate[1] for chip in chip_coordinates if chip.type == ChipletType.COMPUTE])
min_compute_chiplet_x = min([chip.coordinate[0] for chip in chip_coordinates if chip.type == ChipletType.COMPUTE])
min_compute_chiplet_y = min([chip.coordinate[1] for chip in chip_coordinates if chip.type == ChipletType.COMPUTE])
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
  integer chip_finish[${max_compute_chiplet_x}:${min_compute_chiplet_x}][${max_compute_chiplet_y}:${min_compute_chiplet_y}];

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
% for chip in chip_coordinates:
%   if chip.type == ChipletType.COMPUTE:
%     for k in range(0, mem_bank):
%       if netlist is True:
    i_hemaia_${chip.coordinate[0]}_${chip.coordinate[1]}.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks_${k}__i_data_mem.i_tc_sram.gen_mem_gen_${int(mem_size/8/mem_bank)}x${64}_u_sram.load_data("app_chip_${chip.coordinate[0]}_${chip.coordinate[1]}/bank_${k}.hex");
%       elif mem_macro is True:
    i_hemaia_${chip.coordinate[0]}_${chip.coordinate[1]}.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[${k}].i_data_mem.i_tc_sram.gen_mem.gen_${int(mem_size/8/mem_bank)}x${64}.u_sram.load_data("app_chip_${chip.coordinate[0]}_${chip.coordinate[1]}/bank_${k}.hex");
%       else:
    i_hemaia_${chip.coordinate[0]}_${chip.coordinate[1]}.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[${k}].i_data_mem.i_tc_sram.load_data("app_chip_${chip.coordinate[0]}_${chip.coordinate[1]}/bank_${k}.hex");
%       endif
%     endfor
%   endif
% endfor
% for chip in chip_coordinates:
%   if chip.type == ChipletType.MEMORY:
%       for k in range(0, 32):
    i_hemaia_mem_${chip.coordinate[0]}_${chip.coordinate[1]}.i_hemaia_mem_system.i_hemaia_mem.gen_banks[${k}].i_data_mem.i_tc_sram.load_data("app_chip_${chip.coordinate[0]}_${chip.coordinate[1]}/bank_${k}.hex");
%       endfor
%   endif
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
% for chip in chip_coordinates:
%   if chip.type == ChipletType.COMPUTE:
    if (chip_finish[${chip.coordinate[0]}][${chip.coordinate[1]}] == 0) begin
        allFinished = 0;
    end
    if (chip_finish[${chip.coordinate[0]}][${chip.coordinate[1]}] == -1) begin
        allCorrect = 0;
    end
%   endif
% endfor
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
% for chip in chip_coordinates:
  tri [19:0] chip_${chip.coordinate[0]}_${chip.coordinate[1]}_to_${chip.coordinate[0]+1}_${chip.coordinate[1]}_link[3];
  wire chip_${chip.coordinate[0]}_${chip.coordinate[1]}_to_${chip.coordinate[0]+1}_${chip.coordinate[1]}_link_rts;
  wire chip_${chip.coordinate[0]}_${chip.coordinate[1]}_to_${chip.coordinate[0]+1}_${chip.coordinate[1]}_link_cts;
  wire chip_${chip.coordinate[0]+1}_${chip.coordinate[1]}_to_${chip.coordinate[0]}_${chip.coordinate[1]}_link_rts;
  wire chip_${chip.coordinate[0]+1}_${chip.coordinate[1]}_to_${chip.coordinate[0]}_${chip.coordinate[1]}_link_cts;
  wire chip_${chip.coordinate[0]}_${chip.coordinate[1]}_to_${chip.coordinate[0]+1}_${chip.coordinate[1]}_link_test_request;
  wire chip_${chip.coordinate[0]+1}_${chip.coordinate[1]}_to_${chip.coordinate[0]}_${chip.coordinate[1]}_link_test_request;
  tri [19:0] chip_${chip.coordinate[0]}_${chip.coordinate[1]}_to_${chip.coordinate[0]}_${chip.coordinate[1]+1}_link[3];
  wire chip_${chip.coordinate[0]}_${chip.coordinate[1]}_to_${chip.coordinate[0]}_${chip.coordinate[1]+1}_link_rts;
  wire chip_${chip.coordinate[0]}_${chip.coordinate[1]}_to_${chip.coordinate[0]}_${chip.coordinate[1]+1}_link_cts;
  wire chip_${chip.coordinate[0]}_${chip.coordinate[1]+1}_to_${chip.coordinate[0]}_${chip.coordinate[1]}_link_rts;
  wire chip_${chip.coordinate[0]}_${chip.coordinate[1]+1}_to_${chip.coordinate[0]}_${chip.coordinate[1]}_link_cts;
  wire chip_${chip.coordinate[0]}_${chip.coordinate[1]}_to_${chip.coordinate[0]}_${chip.coordinate[1]+1}_link_test_request;
  wire chip_${chip.coordinate[0]}_${chip.coordinate[1]+1}_to_${chip.coordinate[0]}_${chip.coordinate[1]}_link_test_request;

% endfor

  // Instatiate Chips
  wire const_zero;
  wire const_one;
  assign const_zero = 1'b0;
  assign const_one  = 1'b1;

% for chip in chip_coordinates:
<%
    chip_coordinate_0_str = "{:01x}".format(chip.coordinate[0])
    chip_coordinate_1_str = "{:01x}".format(chip.coordinate[1])
%>
  wire [7:0] chip_id_${chip.coordinate[0]}_${chip.coordinate[1]};
  assign chip_id_${chip.coordinate[0]}_${chip.coordinate[1]} = 8'h${chip_coordinate_0_str}${chip_coordinate_1_str};
% if chip.type == ChipletType.COMPUTE:
  // Uart signals (must be nets when connected to DUT inout/output)
  wire tx_${chip.coordinate[0]}_${chip.coordinate[1]}, rx_${chip.coordinate[0]}_${chip.coordinate[1]};

  hemaia i_hemaia_${chip.coordinate[0]}_${chip.coordinate[1]} (
      .io_clk_i(mst_clk_i),
      .io_rst_ni(rst_ni),
      .io_clk_periph_i(periph_clk_i),
      .io_rst_periph_ni(rst_periph_ni),
      .io_test_mode_i(const_zero),
      .io_chip_id_i(chip_id_${chip.coordinate[0]}_${chip.coordinate[1]}),
      .io_boot_mode_i(const_zero),
% if multichip_cfg['single_chip'] is False:
%   if any(neighborhood.coordinate == (chip.coordinate[0]+1, chip.coordinate[1]) for neighborhood in chip_coordinates):
      .io_east_d2d(chip_${chip.coordinate[0]}_${chip.coordinate[1]}_to_${chip.coordinate[0]+1}_${chip.coordinate[1]}_link),
      .io_flow_control_east_rts_o(chip_${chip.coordinate[0]}_${chip.coordinate[1]}_to_${chip.coordinate[0]+1}_${chip.coordinate[1]}_link_rts),
      .io_flow_control_east_cts_i(chip_${chip.coordinate[0]}_${chip.coordinate[1]}_to_${chip.coordinate[0]+1}_${chip.coordinate[1]}_link_cts),
      .io_flow_control_east_rts_i(chip_${chip.coordinate[0]+1}_${chip.coordinate[1]}_to_${chip.coordinate[0]}_${chip.coordinate[1]}_link_rts),
      .io_flow_control_east_cts_o(chip_${chip.coordinate[0]+1}_${chip.coordinate[1]}_to_${chip.coordinate[0]}_${chip.coordinate[1]}_link_cts),
      .io_east_test_being_requested_i(chip_${chip.coordinate[0]+1}_${chip.coordinate[1]}_to_${chip.coordinate[0]}_${chip.coordinate[1]}_link_test_request),
      .io_east_test_request_o(chip_${chip.coordinate[0]}_${chip.coordinate[1]}_to_${chip.coordinate[0]+1}_${chip.coordinate[1]}_link_test_request),
%   else:
      .io_east_d2d(),
      .io_flow_control_east_rts_o(),
      .io_flow_control_east_cts_i('0),
      .io_flow_control_east_rts_i('0),
      .io_flow_control_east_cts_o(),
      .io_east_test_being_requested_i('0),
      .io_east_test_request_o(),
%   endif
%   if any(neighborhood.coordinate == (chip.coordinate[0]-1, chip.coordinate[1]) for neighborhood in chip_coordinates):
      .io_west_d2d(chip_${chip.coordinate[0]-1}_${chip.coordinate[1]}_to_${chip.coordinate[0]}_${chip.coordinate[1]}_link),
      .io_flow_control_west_rts_o(chip_${chip.coordinate[0]}_${chip.coordinate[1]}_to_${chip.coordinate[0]-1}_${chip.coordinate[1]}_link_rts),
      .io_flow_control_west_cts_i(chip_${chip.coordinate[0]}_${chip.coordinate[1]}_to_${chip.coordinate[0]-1}_${chip.coordinate[1]}_link_cts),
      .io_flow_control_west_rts_i(chip_${chip.coordinate[0]-1}_${chip.coordinate[1]}_to_${chip.coordinate[0]}_${chip.coordinate[1]}_link_rts),
      .io_flow_control_west_cts_o(chip_${chip.coordinate[0]-1}_${chip.coordinate[1]}_to_${chip.coordinate[0]}_${chip.coordinate[1]}_link_cts),
      .io_west_test_being_requested_i(chip_${chip.coordinate[0]-1}_${chip.coordinate[1]}_to_${chip.coordinate[0]}_${chip.coordinate[1]}_link_test_request),
      .io_west_test_request_o(chip_${chip.coordinate[0]}_${chip.coordinate[1]}_to_${chip.coordinate[0]-1}_${chip.coordinate[1]}_link_test_request),
%   else:
      .io_west_d2d(),
      .io_flow_control_west_rts_o(),
      .io_flow_control_west_cts_i('0),
      .io_flow_control_west_rts_i('0),
      .io_flow_control_west_cts_o(),
      .io_west_test_being_requested_i('0),
      .io_west_test_request_o(),
%   endif
%   if any(neighborhood.coordinate == (chip.coordinate[0], chip.coordinate[1]-1) for neighborhood in chip_coordinates):
      .io_north_d2d(chip_${chip.coordinate[0]}_${chip.coordinate[1]-1}_to_${chip.coordinate[0]}_${chip.coordinate[1]}_link),
      .io_flow_control_north_rts_o(chip_${chip.coordinate[0]}_${chip.coordinate[1]}_to_${chip.coordinate[0]}_${chip.coordinate[1]-1}_link_rts),
      .io_flow_control_north_cts_i(chip_${chip.coordinate[0]}_${chip.coordinate[1]}_to_${chip.coordinate[0]}_${chip.coordinate[1]-1}_link_cts),
      .io_flow_control_north_rts_i(chip_${chip.coordinate[0]}_${chip.coordinate[1]-1}_to_${chip.coordinate[0]}_${chip.coordinate[1]}_link_rts),
      .io_flow_control_north_cts_o(chip_${chip.coordinate[0]}_${chip.coordinate[1]-1}_to_${chip.coordinate[0]}_${chip.coordinate[1]}_link_cts),
      .io_north_test_being_requested_i(chip_${chip.coordinate[0]}_${chip.coordinate[1]-1}_to_${chip.coordinate[0]}_${chip.coordinate[1]}_link_test_request),
      .io_north_test_request_o(chip_${chip.coordinate[0]}_${chip.coordinate[1]}_to_${chip.coordinate[0]}_${chip.coordinate[1]-1}_link_test_request),
%   else:
      .io_north_d2d(),
      .io_flow_control_north_rts_o(),
      .io_flow_control_north_cts_i('0),
      .io_flow_control_north_rts_i('0),
      .io_flow_control_north_cts_o(),
      .io_north_test_being_requested_i('0),
      .io_north_test_request_o(),
%   endif
%   if any(neighborhood.coordinate == (chip.coordinate[0], chip.coordinate[1]+1) for neighborhood in chip_coordinates):
      .io_south_d2d(chip_${chip.coordinate[0]}_${chip.coordinate[1]}_to_${chip.coordinate[0]}_${chip.coordinate[1]+1}_link),
      .io_flow_control_south_rts_o(chip_${chip.coordinate[0]}_${chip.coordinate[1]}_to_${chip.coordinate[0]}_${chip.coordinate[1]+1}_link_rts),
      .io_flow_control_south_cts_i(chip_${chip.coordinate[0]}_${chip.coordinate[1]}_to_${chip.coordinate[0]}_${chip.coordinate[1]+1}_link_cts),
      .io_flow_control_south_rts_i(chip_${chip.coordinate[0]}_${chip.coordinate[1]+1}_to_${chip.coordinate[0]}_${chip.coordinate[1]}_link_rts),
      .io_flow_control_south_cts_o(chip_${chip.coordinate[0]}_${chip.coordinate[1]+1}_to_${chip.coordinate[0]}_${chip.coordinate[1]}_link_cts),
      .io_south_test_being_requested_i(chip_${chip.coordinate[0]}_${chip.coordinate[1]+1}_to_${chip.coordinate[0]}_${chip.coordinate[1]}_link_test_request),
      .io_south_test_request_o(chip_${chip.coordinate[0]}_${chip.coordinate[1]}_to_${chip.coordinate[0]}_${chip.coordinate[1]+1}_link_test_request),
%   else:
      .io_south_d2d(),
      .io_flow_control_south_rts_o(),
      .io_flow_control_south_cts_i('0),
      .io_flow_control_south_rts_i('0),
      .io_flow_control_south_cts_o(),
      .io_south_test_being_requested_i('0),
      .io_south_test_request_o(),
%   endif
% endif
      .io_uart_tx_o(tx_${chip.coordinate[0]}_${chip.coordinate[1]}),
      .io_uart_rx_i(rx_${chip.coordinate[0]}_${chip.coordinate[1]}),
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
      .NAME("uart_${chip.coordinate[0]}_${chip.coordinate[1]}")
  ) i_uart_${chip.coordinate[0]}_${chip.coordinate[1]} (
      .clk_i (periph_clk_i),
      .rst_ni(rst_ni),
      .tx_o  (rx_${chip.coordinate[0]}_${chip.coordinate[1]}),
      .rx_i  (tx_${chip.coordinate[0]}_${chip.coordinate[1]})
  );

% if mem_macro is False and netlist is False:
  // Chip Status Monitor Block
  always @(i_hemaia_${chip.coordinate[0]}_${chip.coordinate[1]}.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[${mem_bank-1}].i_data_mem.i_tc_sram.sram[SRAM_DEPTH-1][(SRAM_WIDTH*8-1)-:32]) begin
    if (i_hemaia_${chip.coordinate[0]}_${chip.coordinate[1]}.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[${mem_bank-1}].i_data_mem.i_tc_sram.sram[SRAM_DEPTH-1][(SRAM_WIDTH*8-1)-:32] != 0) begin
      if (i_hemaia_${chip.coordinate[0]}_${chip.coordinate[1]}.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[${mem_bank-1}].i_data_mem.i_tc_sram.sram[SRAM_DEPTH-1][(SRAM_WIDTH*8-1)-:32] == 32'd1) begin
        $display("Simulation of chip_${chip.coordinate[0]}_${chip.coordinate[1]} is finished at %tns", $time / 1000);
        chip_finish[${chip.coordinate[0]}][${chip.coordinate[1]}] = 1;
      end else begin
        $error("Simulation of chip_${chip.coordinate[0]}_${chip.coordinate[1]} is finished with errors %d at %tns",
               i_hemaia_${chip.coordinate[0]}_${chip.coordinate[1]}.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[${mem_bank-1}].i_data_mem.i_tc_sram.sram[SRAM_DEPTH-1][(SRAM_WIDTH*8-1)-:32],
               $time / 1000);
        chip_finish[${chip.coordinate[0]}][${chip.coordinate[1]}] = -1;
      end
    end
  end
% endif
% elif chip.type == ChipletType.MEMORY:
  hemaia_mem_chip #(
    .MemBankNum(32),
    .MemSize(${chip.size}),
    .EnableEastPhy(1'b${
        '1' if any(neighborhood.coordinate == (chip.coordinate[0]+1, chip.coordinate[1]) for neighborhood in chip_coordinates) else '0'
    }),
    .EnableWestPhy(1'b${
        '1' if any(neighborhood.coordinate == (chip.coordinate[0]-1, chip.coordinate[1]) for neighborhood in chip_coordinates) else '0'
    }),
    .EnableNorthPhy(1'b${
        '1' if any(neighborhood.coordinate == (chip.coordinate[0], chip.coordinate[1]-1) for neighborhood in chip_coordinates) else '0'
    }),
    .EnableSouthPhy(1'b${
        '1' if any(neighborhood.coordinate == (chip.coordinate[0], chip.coordinate[1]+1) for neighborhood in chip_coordinates) else '0'
    })
  ) i_hemaia_mem_${chip.coordinate[0]}_${chip.coordinate[1]} (
      .clk_i(mst_clk_i),
      .rst_ni(rst_ni),
      .chip_id_i(chip_id_${chip.coordinate[0]}_${chip.coordinate[1]}),
%   if any(neighborhood.coordinate == (chip.coordinate[0]+1, chip.coordinate[1]) for neighborhood in chip_coordinates):
      .east_d2d_io(chip_${chip.coordinate[0]}_${chip.coordinate[1]}_to_${chip.coordinate[0]+1}_${chip.coordinate[1]}_link),
      .flow_control_east_rts_o(chip_${chip.coordinate[0]}_${chip.coordinate[1]}_to_${chip.coordinate[0]+1}_${chip.coordinate[1]}_link_rts),
      .flow_control_east_cts_i(chip_${chip.coordinate[0]}_${chip.coordinate[1]}_to_${chip.coordinate[0]+1}_${chip.coordinate[1]}_link_cts),
      .flow_control_east_rts_i(chip_${chip.coordinate[0]+1}_${chip.coordinate[1]}_to_${chip.coordinate[0]}_${chip.coordinate[1]}_link_rts),
      .flow_control_east_cts_o(chip_${chip.coordinate[0]+1}_${chip.coordinate[1]}_to_${chip.coordinate[0]}_${chip.coordinate[1]}_link_cts),
      .east_test_being_requested_i(chip_${chip.coordinate[0]+1}_${chip.coordinate[1]}_to_${chip.coordinate[0]}_${chip.coordinate[1]}_link_test_request),
      .east_test_request_o(chip_${chip.coordinate[0]}_${chip.coordinate[1]}_to_${chip.coordinate[0]+1}_${chip.coordinate[1]}_link_test_request),
%   else:
      .east_d2d_io(),
      .flow_control_east_rts_o(),
      .flow_control_east_cts_i('0),
      .flow_control_east_rts_i('0),
      .flow_control_east_cts_o(),
      .east_test_being_requested_i('0),
      .east_test_request_o(),
%   endif
%   if any(neighborhood.coordinate == (chip.coordinate[0]-1, chip.coordinate[1]) for neighborhood in chip_coordinates):
      .west_d2d_io(chip_${chip.coordinate[0]-1}_${chip.coordinate[1]}_to_${chip.coordinate[0]}_${chip.coordinate[1]}_link),
      .flow_control_west_rts_o(chip_${chip.coordinate[0]}_${chip.coordinate[1]}_to_${chip.coordinate[0]-1}_${chip.coordinate[1]}_link_rts),
      .flow_control_west_cts_i(chip_${chip.coordinate[0]}_${chip.coordinate[1]}_to_${chip.coordinate[0]-1}_${chip.coordinate[1]}_link_cts),
      .flow_control_west_rts_i(chip_${chip.coordinate[0]-1}_${chip.coordinate[1]}_to_${chip.coordinate[0]}_${chip.coordinate[1]}_link_rts),
      .flow_control_west_cts_o(chip_${chip.coordinate[0]-1}_${chip.coordinate[1]}_to_${chip.coordinate[0]}_${chip.coordinate[1]}_link_cts),
      .west_test_being_requested_i(chip_${chip.coordinate[0]-1}_${chip.coordinate[1]}_to_${chip.coordinate[0]}_${chip.coordinate[1]}_link_test_request),
      .west_test_request_o(chip_${chip.coordinate[0]}_${chip.coordinate[1]}_to_${chip.coordinate[0]-1}_${chip.coordinate[1]}_link_test_request),
%   else:
      .west_d2d_io(),
      .flow_control_west_rts_o(),
      .flow_control_west_cts_i('0),
      .flow_control_west_rts_i('0),
      .flow_control_west_cts_o(),
      .west_test_being_requested_i('0),
      .west_test_request_o(),
%   endif
%   if any(neighborhood.coordinate == (chip.coordinate[0], chip.coordinate[1]-1) for neighborhood in chip_coordinates):
      .north_d2d_io(chip_${chip.coordinate[0]}_${chip.coordinate[1]-1}_to_${chip.coordinate[0]}_${chip.coordinate[1]}_link),
      .flow_control_north_rts_o(chip_${chip.coordinate[0]}_${chip.coordinate[1]}_to_${chip.coordinate[0]}_${chip.coordinate[1]-1}_link_rts),
      .flow_control_north_cts_i(chip_${chip.coordinate[0]}_${chip.coordinate[1]}_to_${chip.coordinate[0]}_${chip.coordinate[1]-1}_link_cts),
      .flow_control_north_rts_i(chip_${chip.coordinate[0]}_${chip.coordinate[1]-1}_to_${chip.coordinate[0]}_${chip.coordinate[1]}_link_rts),
      .flow_control_north_cts_o(chip_${chip.coordinate[0]}_${chip.coordinate[1]-1}_to_${chip.coordinate[0]}_${chip.coordinate[1]}_link_cts),
      .north_test_being_requested_i(chip_${chip.coordinate[0]}_${chip.coordinate[1]-1}_to_${chip.coordinate[0]}_${chip.coordinate[1]}_link_test_request),
      .north_test_request_o(chip_${chip.coordinate[0]}_${chip.coordinate[1]}_to_${chip.coordinate[0]}_${chip.coordinate[1]-1}_link_test_request),
%   else:
      .north_d2d_io(),
      .flow_control_north_rts_o(),
      .flow_control_north_cts_i('0),
      .flow_control_north_rts_i('0),
      .flow_control_north_cts_o(),
      .north_test_being_requested_i('0),
      .north_test_request_o(),
%   endif
%   if any(neighborhood.coordinate == (chip.coordinate[0], chip.coordinate[1]+1) for neighborhood in chip_coordinates):
      .south_d2d_io(chip_${chip.coordinate[0]}_${chip.coordinate[1]}_to_${chip.coordinate[0]}_${chip.coordinate[1]+1}_link),
      .flow_control_south_rts_o(chip_${chip.coordinate[0]}_${chip.coordinate[1]}_to_${chip.coordinate[0]}_${chip.coordinate[1]+1}_link_rts),
      .flow_control_south_cts_i(chip_${chip.coordinate[0]}_${chip.coordinate[1]}_to_${chip.coordinate[0]}_${chip.coordinate[1]+1}_link_cts),
      .flow_control_south_rts_i(chip_${chip.coordinate[0]}_${chip.coordinate[1]+1}_to_${chip.coordinate[0]}_${chip.coordinate[1]}_link_rts),
      .flow_control_south_cts_o(chip_${chip.coordinate[0]}_${chip.coordinate[1]+1}_to_${chip.coordinate[0]}_${chip.coordinate[1]}_link_cts),
      .south_test_being_requested_i(chip_${chip.coordinate[0]}_${chip.coordinate[1]+1}_to_${chip.coordinate[0]}_${chip.coordinate[1]}_link_test_request),
      .south_test_request_o(chip_${chip.coordinate[0]}_${chip.coordinate[1]}_to_${chip.coordinate[0]}_${chip.coordinate[1]+1}_link_test_request)
%   else:
      .south_d2d_io(),
      .flow_control_south_rts_o(),
      .flow_control_south_cts_i('0),
      .flow_control_south_rts_i('0),
      .flow_control_south_cts_o(),
      .south_test_being_requested_i('0),
      .south_test_request_o()
%   endif
  );
% endif
% endfor
endmodule
