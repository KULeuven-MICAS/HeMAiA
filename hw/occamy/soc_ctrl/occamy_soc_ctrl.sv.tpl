// Copyright 2020 ETH Zurich and University of Bologna.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51

// Author: Florian Zaruba <zarubaf@iis.ee.ethz.ch>
`include "common_cells/registers.svh"

module occamy_soc_ctrl import occamy_soc_reg_pkg::*; #(
  parameter type reg_req_t = logic,
  parameter type reg_rsp_t = logic,
  parameter type chip_id_t = logic
) (
  input clk_i,
  input rst_ni,
  input chip_id_t chip_id_i,

  // Below Register interface can be changed
  input  reg_req_t reg_req_i,
  output reg_rsp_t reg_rsp_o,
  // To HW
  output occamy_soc_reg2hw_t reg2hw_o, // Write
  input  occamy_soc_hw2reg_t hw2reg_i,
  // Boot addr
  output logic [${addr_width - 1}:0] boot_addr_o
);

  occamy_soc_reg_top #(
    .reg_req_t ( reg_req_t ),
    .reg_rsp_t ( reg_rsp_t  )
  ) i_soc_ctrl (
    .clk_i     ( clk_i  ),
    .rst_ni    ( rst_ni ),
    .reg_req_i ( reg_req_i ),
    .reg_rsp_o ( reg_rsp_o ),
    .reg2hw    ( reg2hw_o ),
    .hw2reg    ( hw2reg_i ),
    .devmode_i ( 1'b1 )
  );
   // boot address
  logic [${addr_width-1}:0] boot_addr_d, boot_addr_q;
  logic [${addr_width-1}:0] boot_addr_init;
  logic [1:0] boot_mode;
  assign boot_mode = hw2reg_i.boot_mode.d;

  always_comb begin
    boot_addr_init = (boot_mode == 2'b00) ? {chip_id_i,${addr_width-occamy_cfg["hemaia_multichip"]["chip_id_width"]}'h${default_boot_addr}}:{chip_id_i,${addr_width-occamy_cfg["hemaia_multichip"]["chip_id_width"]}'h${backup_boot_addr}};
    boot_addr_d = (boot_mode == 2'b00) ? {chip_id_i,${addr_width-occamy_cfg["hemaia_multichip"]["chip_id_width"]}'h${default_boot_addr}}:{chip_id_i,${addr_width-occamy_cfg["hemaia_multichip"]["chip_id_width"]}'h${backup_boot_addr}};
    boot_addr_o = boot_addr_q;
  end

  `FF(boot_addr_q, boot_addr_d, boot_addr_init, clk_i, rst_ni)

endmodule