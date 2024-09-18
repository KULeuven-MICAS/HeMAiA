// Copyright 2020 ETH Zurich and University of Bologna.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51

// Author: Nils Wistoff <nwistoff@iis.ee.ethz.ch>
//
// AUTOMATICALLY GENERATED by genoccamy.py; edit the script instead.

module ${name}_chip
import ${name}_pkg::*;
 (
  input  logic        clk_i,
  input  logic        rst_ni,
  /// Peripheral clock
  input  logic        clk_periph_i,
  input  logic        rst_periph_ni,
  /// Real-time clock (for time keeping)
  input  logic        rtc_i,
  input  logic        test_mode_i,
  input  chip_id_t    chip_id_i,
  input  logic [1:0]  boot_mode_i,
  // `uart` Interface
  output logic        uart_tx_o,
  input  logic        uart_rx_i,
  output logic        uart_rts_no, 
  input  logic        uart_cts_ni, 
  // `gpio` Interface
  input  logic [31:0] gpio_d_i,
  output logic [31:0] gpio_d_o,
  output logic [31:0] gpio_oe_o,
  // `jtag` Interface
  input  logic        jtag_trst_ni,
  input  logic        jtag_tck_i,
  input  logic        jtag_tms_i,
  input  logic        jtag_tdi_i,
  output logic        jtag_tdo_o,
  // `i2c` Interface
  inout  logic        i2c_sda_io,
  inout  logic        i2c_scl_io,
  // `SPI Host` Interface
  output logic        spim_sck_o,
  output logic [1:0]  spim_csb_o,
  inout  logic [3:0]  spim_sd_io,

  input  logic [11:0] ext_irq_i
);

  /////////////////////////
  // SRAM as main memory //
  /////////////////////////

  <%
    ram_axi = soc_wide_xbar.out_spm_wide.copy(name="ram_axi").declare(context)
  %> \

  <% spm_wide_words = occamy_cfg["spm_wide"]["length"]//(soc_wide_xbar.out_spm_wide.dw//8) %>\

  typedef logic [${util.clog2(spm_wide_words) + util.clog2(soc_wide_xbar.out_spm_wide.dw//8)-1}:0] mem_wide_addr_t;
  typedef logic [${soc_wide_xbar.out_spm_wide.dw-1}:0] mem_wide_data_t;
  typedef logic [${soc_wide_xbar.out_spm_wide.dw//8-1}:0] mem_wide_strb_t;

  logic spm_wide_req, spm_wide_gnt, spm_wide_we, spm_wide_rvalid;
  mem_wide_addr_t spm_wide_addr;
  mem_wide_data_t spm_wide_wdata, spm_wide_rdata;
  mem_wide_strb_t spm_wide_strb;

    axi_to_mem_interleaved #(
    .axi_req_t (${soc_wide_xbar.out_spm_wide.req_type()}),
    .axi_resp_t (${soc_wide_xbar.out_spm_wide.rsp_type()}),
    .AddrWidth (${util.clog2(spm_wide_words) + util.clog2(soc_wide_xbar.out_spm_wide.dw//8)}),
    .DataWidth (${soc_wide_xbar.out_spm_wide.dw}),
    .IdWidth (${soc_wide_xbar.out_spm_wide.iw}),
    .NumBanks (1),
    .BufDepth (16)
  ) i_axi_to_mem (
    .clk_i (${soc_wide_xbar.out_spm_wide.clk}),
    .rst_ni (${soc_wide_xbar.out_spm_wide.rst}),
    .busy_o (),
    .axi_req_i (${ram_axi.req_name()}),
    .axi_resp_o (${ram_axi.rsp_name()}),
    .mem_req_o (spm_wide_req),
    .mem_gnt_i (spm_wide_gnt),
    .mem_addr_o (spm_wide_addr),
    .mem_wdata_o (spm_wide_wdata),
    .mem_strb_o (spm_wide_strb),
    .mem_atop_o (),
    .mem_we_o (spm_wide_we),
    .mem_rvalid_i (spm_wide_rvalid),
    .mem_rdata_i (spm_wide_rdata)
  );

  spm_1p_adv #(
    .NumWords (${spm_wide_words}),
    .DataWidth (${soc_wide_xbar.out_spm_wide.dw}),
    .ByteWidth (8),
    .EnableInputPipeline (1'b1),
    .EnableOutputPipeline (1'b1),
    .sram_cfg_t (sram_cfg_t)
  ) i_spm_wide_cut (
    .clk_i (${soc_wide_xbar.out_spm_wide.clk}),
    .rst_ni (${soc_wide_xbar.out_spm_wide.rst}),
    .valid_i (spm_wide_req),
    .ready_o (spm_wide_gnt),
    .we_i (spm_wide_we),
    .addr_i (spm_wide_addr[${util.clog2(spm_wide_words) + util.clog2(soc_wide_xbar.out_spm_wide.dw//8)-1}:${util.clog2(soc_wide_xbar.out_spm_wide.dw//8)}]),
    .wdata_i (spm_wide_wdata),
    .be_i (spm_wide_strb),
    .rdata_o (spm_wide_rdata),
    .rvalid_o (spm_wide_rvalid),
    .rerror_o (spm_wide_rerror_o),
    .sram_cfg_i ('0)
  );

  ///////////////////
  //   Boot ROM    //
  ///////////////////

  ${soc_axi_lite_narrow_periph_xbar.out_bootrom.req_type()} bootrom_axi_lite_req;
  ${soc_axi_lite_narrow_periph_xbar.out_bootrom.rsp_type()} bootrom_axi_lite_rsp;

  <% regbus_bootrom = soc_axi_lite_narrow_periph_xbar.out_bootrom.to_reg(context, "bootrom", fr="bootrom_axi_lite") %>

  bootrom i_bootrom (
    .clk_i(clk_i), 
    .rst_ni(rst_ni), 
    .req_i(bootrom_req.valid), 
    .addr_i(bootrom_req.addr), 
    .data_o(bootrom_rsp.rdata)
  );

  assign bootrom_rsp.ready   = '1;
  assign bootrom_rsp.error   = '0;

  ///////////////////
  // Chip ID Latch //
  ///////////////////

  // The latched chip_id
  chip_id_t chip_id;

  always_latch begin
    if (~rst_ni) begin
      chip_id <= chip_id_i;
    end
  end

  ///////////////////
  //  Occamy Top   //
  ///////////////////

  ${name}_top i_${name} (
    .chip_id_i       (chip_id),
    .bootrom_req_o   (bootrom_axi_lite_req),
    .bootrom_rsp_i   (bootrom_axi_lite_rsp),
    .ext_irq_i       (ext_irq_i),
    // RAM
    .spm_axi_wide_req_o(${ram_axi.req_name()}), 
    .spm_axi_wide_rsp_i(${ram_axi.rsp_name()}), 
    // Tie-off unused ports
    .chip_ctrl_req_o (), 
    .chip_ctrl_rsp_i ('0),
    .sram_cfgs_i ('0),
    .*
  );

endmodule
