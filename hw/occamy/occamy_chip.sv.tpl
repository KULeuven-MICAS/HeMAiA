// Copyright 2020 ETH Zurich and University of Bologna.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51

// Author: Nils Wistoff <nwistoff@iis.ee.ethz.ch>
//
// AUTOMATICALLY GENERATED by genoccamy.py; edit the script instead.

`include "axi_flat.sv"

module ${name}_chip
import ${name}_pkg::*;
 (
  input  logic        clk_i,
  input  logic        rst_ni,
  // Obs pins
  output logic [${obs_pin_width-1}:0]  obs_o [${nr_clusters_s1_quadrant}],
  /// Peripheral clock
  input  logic        clk_periph_i,
  input  logic        rst_periph_ni,
  /// Real-time clock (for time keeping)
  input  logic        rtc_i,
  input  logic        test_mode_i,
  input  logic [1:0]  chip_id_i,
  input  logic [1:0]  boot_mode_i,
  // Clk gen trim bits
  output logic [8:0]  clk_gen_trim_bits_o,
  // `uart` Interface
  output logic        uart_tx_o,
  input  logic        uart_rx_i,
  output logic        uart_rts_no, 
  input  logic        uart_cts_ni, 
  // `gpio` Interface
  input  logic [31:0] gpio_d_i,
  output logic [31:0] gpio_d_o,
  output logic [31:0] gpio_oe_o,
<% 
  spi_slave_present = any(periph["name"] == "spis" for periph in occamy_cfg["peripherals"]["axi_lite_peripherals"])
%>
% if spi_slave_present: 
  // `SPI Slave` for Debugging Purposes
  input  logic        spis_sck_i,
  input  logic        spis_csb_i,
  output logic [3:0]  spis_sd_o,
  output logic [3:0]  spis_sd_en_o,
  input  logic [3:0]  spis_sd_i,
% endif
  // `jtag` Interface
  input  logic        jtag_trst_ni,
  input  logic        jtag_tck_i,
  input  logic        jtag_tms_i,
  input  logic        jtag_tdi_i,
  output logic        jtag_tdo_o
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
    .rerror_o (),
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
    .addr_i(bootrom_req.addr[31:0]), 
    .data_o(bootrom_rsp.rdata)
  );

  assign bootrom_rsp.ready   = '1;
  assign bootrom_rsp.error   = '0;

  ///////////////////
  //  Occamy Top   //
  ///////////////////

  ${name}_top i_${name} (
    .clk_i              (clk_i),
    .rst_ni             (rst_ni),
    .obs_o              (obs_o),
    .sram_cfgs_i        ('0),
    .clk_periph_i       (clk_periph_i),
    .rst_periph_ni      (rst_periph_ni),
    .rtc_i              (rtc_i),
    .test_mode_i        (test_mode_i),
    .chip_id_i          (chip_id_i),
    .boot_mode_i        (boot_mode_i),
    .clk_gen_trim_bits_o (clk_gen_trim_bits_o),
    .uart_tx_o          (uart_tx_o),
    .uart_cts_ni        (uart_cts_ni),
    .uart_rts_no        (uart_rts_no),
    .uart_rx_i          (uart_rx_i),
    .gpio_d_i           (gpio_d_i),
    .gpio_d_o           (gpio_d_o),
    .gpio_oe_o          (gpio_oe_o),
    .jtag_trst_ni       (jtag_trst_ni),
    .jtag_tck_i         (jtag_tck_i),
    .jtag_tms_i         (jtag_tms_i),
    .jtag_tdi_i         (jtag_tdi_i),
    .jtag_tdo_o         (jtag_tdo_o),
    .i2c_sda_o          (),
    .i2c_sda_i          ('1),
    .i2c_sda_en_o       (),
    .i2c_scl_o          (),
    .i2c_scl_i          ('1),
    .i2c_scl_en_o       (),
    .spim_sck_o         (),
    .spim_sck_en_o      (),
    .spim_csb_o         (),
    .spim_csb_en_o      (),
    .spim_sd_o          (),
    .spim_sd_en_o       (),
    .spim_sd_i          ('1),
% if spi_slave_present: 
    .spis_sck_i         (spis_sck_i),
    .spis_csb_i         (spis_csb_i),
    .spis_sd_o          (spis_sd_o),
    .spis_sd_en_o       (spis_sd_en_o),
    .spis_sd_i          (spis_sd_i),
% endif    
    .bootrom_req_o      (bootrom_axi_lite_req),
    .bootrom_rsp_i      (bootrom_axi_lite_rsp),
    .spm_axi_wide_req_o (ram_axi_req),
    .spm_axi_wide_rsp_i (ram_axi_rsp),
    .chip_ctrl_req_o    (),
    .chip_ctrl_rsp_i    ('0),
    .ext_irq_i          ('0)
  );

endmodule
