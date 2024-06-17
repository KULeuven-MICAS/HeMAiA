// Copyright 2020 ETH Zurich and University of Bologna.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51

// Author: Florian Zaruba <zarubaf@iis.ee.ethz.ch>
// Author: Fabian Schuiki <fschuiki@iis.ee.ethz.ch>
//
// AUTOMATICALLY GENERATED by genoccamy.py; edit the script instead.

`include "common_cells/registers.svh"

module ${name}_top
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
  input  logic [1:0]  chip_id_i,
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

  /// Boot ROM
  output ${soc_axi_lite_narrow_periph_xbar.out_bootrom.req_type()} bootrom_req_o,
  input  ${soc_axi_lite_narrow_periph_xbar.out_bootrom.rsp_type()} bootrom_rsp_i,

  /// Wide SPM
  output ${soc_wide_xbar.out_spm_wide.req_type()} spm_axi_wide_req_o,
  input  ${soc_wide_xbar.out_spm_wide.rsp_type()} spm_axi_wide_rsp_i,

  /// Chip specific control registers
  output ${soc_axi_lite_narrow_periph_xbar.out_chip_ctrl.req_type()} chip_ctrl_req_o,
  input  ${soc_axi_lite_narrow_periph_xbar.out_chip_ctrl.rsp_type()} chip_ctrl_rsp_i,
  // "external interrupts from uncore - "programmable"
  input logic [12:0] ext_irq_i,

  /// SRAM configuration
  input sram_cfgs_t sram_cfgs_i
);

<%

  cuts_clint_cfg = occamy_cfg["cuts"]["periph_axi_lite_narrow_clint_cfg"]
  cuts_soc_ctrl_cfg = occamy_cfg["cuts"]["periph_axi_lite_narrow_soc_ctrl_cfg"]
  cuts_chip_ctrl_cfg = occamy_cfg["cuts"]["periph_axi_lite_narrow_chip_ctrl_cfg"]
  cuts_uart_cfg = occamy_cfg["cuts"]["periph_axi_lite_narrow_uart_cfg"]
  cuts_bootrom_cfg = occamy_cfg["cuts"]["periph_axi_lite_narrow_bootrom_cfg"]
  cuts_fll_system_cfg = occamy_cfg["cuts"]["periph_axi_lite_narrow_fll_system_cfg"]
  cuts_fll_periph_cfg = occamy_cfg["cuts"]["periph_axi_lite_narrow_fll_periph_cfg"]
  cuts_fll_hbm2e_cfg = occamy_cfg["cuts"]["periph_axi_lite_narrow_fll_hbm2e_cfg"]
  cuts_plic_cfg = occamy_cfg["cuts"]["periph_axi_lite_narrow_plic_cfg"]
  cuts_spim_cfg = occamy_cfg["cuts"]["periph_axi_lite_narrow_spim_cfg"]
  cuts_gpio_cfg = occamy_cfg["cuts"]["periph_axi_lite_narrow_gpio_cfg"]
  cuts_i2c_cfg = occamy_cfg["cuts"]["periph_axi_lite_narrow_i2c_cfg"]
  cuts_timer_cfg = occamy_cfg["cuts"]["periph_axi_lite_narrow_timer_cfg"]
%>

  ${name}_soc_reg_pkg::${name}_soc_reg2hw_t soc_ctrl_out;
  ${name}_soc_reg_pkg::${name}_soc_hw2reg_t soc_ctrl_in;
  logic [1:0] spm_narrow_rerror;
  logic [1:0] spm_wide_rerror;

  always_comb begin
    soc_ctrl_in = '0;
    soc_ctrl_in.boot_mode.d = boot_mode_i;
    soc_ctrl_in.chip_id.d = chip_id_i;
  end

  // Machine timer and machine software interrupt pending.
  logic [${cores-1}:0] mtip, msip;
  // Supervisor and machine-mode external interrupt pending.
  logic [1:0] eip;
  logic [0:0] debug_req;
  ${name}_interrupt_t irq;

  assign irq.ext_irq = ext_irq_i;

  //////////////////////////
  //   Peripheral Xbars   //
  //////////////////////////

  ${module}


  ///////////////////////////////
  //   Synchronous top level   //
  ///////////////////////////////

  // Peripheral Xbar connections
<%
  periph_axi_lite_soc2per = soc_narrow_xbar.out_periph.copy(name="periph_axi_lite_soc2per").declare(context)
  periph_axi_lite_per2soc = soc_narrow_xbar.in_periph.copy(name="periph_axi_lite_per2soc").declare(context)
  periph_regbus_soc2per = soc_narrow_xbar.out_axi_lite_narrow_periph.copy(name="periph_regbus_soc2per").declare(context)
%> \
  ${name}_soc i_${name}_soc (
    .clk_i,
    .rst_ni,
    .test_mode_i,
    .periph_axi_lite_req_o ( periph_axi_lite_soc2per_req ),
    .periph_axi_lite_rsp_i ( periph_axi_lite_soc2per_rsp ),
    .periph_axi_lite_req_i ( periph_axi_lite_per2soc_req ),
    .periph_axi_lite_rsp_o ( periph_axi_lite_per2soc_rsp ),
    .periph_axi_lite_narrow_req_o ( periph_regbus_soc2per_req ),
    .periph_axi_lite_narrow_rsp_i ( periph_regbus_soc2per_rsp ),
    .spm_axi_wide_req_o,
    .spm_axi_wide_rsp_i,
    .spm_narrow_rerror_o (spm_narrow_rerror),
    .spm_wide_rerror_o (spm_wide_rerror),
    .mtip_i ( mtip ),
    .msip_i ( msip ),
    .eip_i ( eip ),
    .debug_req_i ( debug_req ),
    .sram_cfgs_i
  );

  // Connect AXI-lite master
  <% periph_axi_lite_soc2per \
      .cdc(context, "clk_periph_i", "rst_periph_ni", "axi_lite_from_soc_cdc") \
      .to_axi_lite(context, "axi_to_axi_lite_periph", to=soc_periph_xbar.in_soc) %> \
  // Connect AXI-lite slave
  <% soc_periph_xbar.out_soc \
      .cdc(context, "clk_i", "rst_ni", "axi_lite_to_soc_cdc") \
      .to_axi(context, "axi_lite_to_axi_periph", to=periph_axi_lite_per2soc) %> \

  <% periph_regbus_soc2per \
    .cdc(context, "clk_periph_i", "rst_periph_ni", "periph_cdc") \
    .change_dw(context, 32, "axi_to_axi_lite_dw") \
    .to_axi_lite(context, "axi_to_axi_lite_regbus_periph", to=soc_axi_lite_narrow_periph_xbar.in_soc) %>


  ///////////
  // Debug //
  ///////////
  <% regbus_debug = soc_periph_xbar.out_debug.to_reg(context, "axi_lite_to_reg_debug") %>
  dm::hartinfo_t [0:0] hartinfo;
  assign hartinfo[0] = ariane_pkg::DebugHartInfo;

  logic          dmi_rst_n;
  dm::dmi_req_t  dmi_req;
  logic          dmi_req_valid;
  logic          dmi_req_ready;
  dm::dmi_resp_t dmi_resp;
  logic          dmi_resp_ready;
  logic          dmi_resp_valid;

  logic dbg_req;
  logic dbg_we;
  logic [${regbus_debug.aw-1}:0] dbg_addr;
  logic [${regbus_debug.dw-1}:0] dbg_wdata;
  logic [${regbus_debug.dw//8-1}:0] dbg_wstrb;
  logic [${regbus_debug.dw-1}:0] dbg_rdata;
  logic dbg_rvalid;

  reg_to_mem #(
    .AW(${regbus_debug.aw}),
    .DW(${regbus_debug.dw}),
    .req_t (${regbus_debug.req_type()}),
    .rsp_t (${regbus_debug.rsp_type()})
  ) i_reg_to_mem_dbg (
    .clk_i (${regbus_debug.clk}),
    .rst_ni (${regbus_debug.rst}),
    .reg_req_i (${regbus_debug.req_name()}),
    .reg_rsp_o (${regbus_debug.rsp_name()}),
    .req_o (dbg_req),
    .gnt_i (dbg_req),
    .we_o (dbg_we),
    .addr_o (dbg_addr),
    .wdata_o (dbg_wdata),
    .wstrb_o (dbg_wstrb),
    .rdata_i (dbg_rdata),
    .rvalid_i (dbg_rvalid),
    .rerror_i (1'b0)
  );

  `FFARN(dbg_rvalid, dbg_req, 1'b0, ${regbus_debug.clk}, ${regbus_debug.rst})

  logic        sba_req;
  logic [${regbus_debug.aw-1}:0] sba_addr;
  logic        sba_we;
  logic [${regbus_debug.dw-1}:0] sba_wdata;
  logic [${regbus_debug.dw//8-1}:0]  sba_strb;
  logic        sba_gnt;

  logic [${regbus_debug.dw-1}:0] sba_rdata;
  logic        sba_rvalid;
  logic        sba_err;

  logic [${regbus_debug.dw-1}:0] sba_addr_long;

  dm_top #(
    // .NrHarts (${cores}),
    .NrHarts (1),
    .BusWidth (${regbus_debug.dw}),
    .DmBaseAddress ('h0)
  ) i_dm_top (
    .clk_i (${regbus_debug.clk}),
    .rst_ni (${regbus_debug.rst}),
    .testmode_i (1'b0),
    .ndmreset_o (),
    .dmactive_o (),
    .debug_req_o (debug_req),
    .unavailable_i ('0),
    .hartinfo_i (hartinfo),
    .slave_req_i (dbg_req),
    .slave_we_i (dbg_we),
    .slave_addr_i ({${regbus_debug.dw-regbus_debug.aw}'b0, dbg_addr}),
    .slave_be_i (dbg_wstrb),
    .slave_wdata_i (dbg_wdata),
    .slave_rdata_o (dbg_rdata),
    .master_req_o (sba_req),
    .master_add_o (sba_addr_long),
    .master_we_o (sba_we),
    .master_wdata_o (sba_wdata),
    .master_be_o (sba_strb),
    .master_gnt_i (sba_gnt),
    .master_r_valid_i (sba_rvalid),
    .master_r_rdata_i (sba_rdata),
    .master_r_err_i (sba_err),
    .master_r_other_err_i (1'b0),
    .dmi_rst_ni (dmi_rst_n),
    .dmi_req_valid_i (dmi_req_valid),
    .dmi_req_ready_o (dmi_req_ready),
    .dmi_req_i (dmi_req),
    .dmi_resp_valid_o (dmi_resp_valid),
    .dmi_resp_ready_i (dmi_resp_ready),
    .dmi_resp_o (dmi_resp)
  );

  assign sba_addr = sba_addr_long[${regbus_debug.aw-1}:0];

  mem_to_axi_lite #(
    .MemAddrWidth (${regbus_debug.aw}),
    .AxiAddrWidth (${regbus_debug.aw}),
    .DataWidth (${regbus_debug.dw}),
    .MaxRequests (2),
    .AxiProt ('0),
    .axi_req_t (${soc_periph_xbar.in_debug.req_type()}),
    .axi_rsp_t (${soc_periph_xbar.in_debug.rsp_type()})
  ) i_mem_to_axi_lite (
    .clk_i (${regbus_debug.clk}),
    .rst_ni (${regbus_debug.rst}),
    .mem_req_i (sba_req),
    .mem_addr_i (sba_addr),
    .mem_we_i (sba_we),
    .mem_wdata_i (sba_wdata),
    .mem_be_i (sba_strb),
    .mem_gnt_o (sba_gnt),
    .mem_rsp_valid_o (sba_rvalid),
    .mem_rsp_rdata_o (sba_rdata),
    .mem_rsp_error_o (sba_err),
    .axi_req_o (${soc_periph_xbar.in_debug.req_name()}),
    .axi_rsp_i (${soc_periph_xbar.in_debug.rsp_name()})

  );

  dmi_jtag #(
    .IdcodeValue (${name}_pkg::IDCode)
  ) i_dmi_jtag (
    .clk_i (${regbus_debug.clk}),
    .rst_ni (${regbus_debug.rst}),
    .testmode_i (1'b0),
    .dmi_rst_no (dmi_rst_n),
    .dmi_req_o (dmi_req),
    .dmi_req_valid_o (dmi_req_valid),
    .dmi_req_ready_i (dmi_req_ready),
    .dmi_resp_i (dmi_resp),
    .dmi_resp_ready_o (dmi_resp_ready),
    .dmi_resp_valid_i (dmi_resp_valid),
    .tck_i (jtag_tck_i),
    .tms_i (jtag_tms_i),
    .trst_ni (jtag_trst_ni),
    .td_i (jtag_tdi_i),
    .td_o (jtag_tdo_o),
    .tdo_oe_o ()
  );


  ///////////////
  //   CLINT   //
  ///////////////

  <% regbus_clint = soc_axi_lite_narrow_periph_xbar.out_clint \
    .cut(context, cuts_clint_cfg, name="soc_axi_lite_narrow_periph_xbar_out_clint_cut") \
    .to_reg(context, "axi_lite_to_reg_clint") %>
  clint #(
    .reg_req_t ( ${regbus_clint.req_type()} ),
    .reg_rsp_t ( ${regbus_clint.rsp_type()} )
  ) i_clint (
    .clk_i (${regbus_clint.clk}),
    .rst_ni (${regbus_clint.rst}),
    .testmode_i (1'b0),
    .reg_req_i (${regbus_clint.req_name()}),
    .reg_rsp_o (${regbus_clint.rsp_name()}),
    .rtc_i (rtc_i),
    .timer_irq_o (mtip),
    .ipi_o (msip)
  );

  /////////////////////
  //   SOC CONTROL   //
  /////////////////////

  <% regbus_soc_ctrl = soc_axi_lite_narrow_periph_xbar.out_soc_ctrl \
    .cut(context, cuts_soc_ctrl_cfg, name="soc_axi_lite_narrow_periph_xbar_out_soc_ctrl_cut") \
    .to_reg(context, "axi_lite_to_reg_soc_ctrl") %>
  ${name}_soc_ctrl #(
    .reg_req_t ( ${regbus_soc_ctrl.req_type()} ),
    .reg_rsp_t ( ${regbus_soc_ctrl.rsp_type()} )
  ) i_soc_ctrl (
    .clk_i     ( ${regbus_soc_ctrl.clk} ),
    .rst_ni    ( ${regbus_soc_ctrl.rst} ),
    .reg_req_i ( ${regbus_soc_ctrl.req_name()} ),
    .reg_rsp_o ( ${regbus_soc_ctrl.rsp_name()} ),
    .reg2hw_o  ( soc_ctrl_out ),
    .hw2reg_i  ( soc_ctrl_in ),
    .event_ecc_rerror_narrow_i(spm_narrow_rerror),
    .event_ecc_rerror_wide_i(spm_wide_rerror),
    .intr_ecc_narrow_uncorrectable_o(irq.ecc_narrow_uncorrectable),
    .intr_ecc_narrow_correctable_o(irq.ecc_narrow_correctable),
    .intr_ecc_wide_uncorrectable_o(irq.ecc_wide_uncorrectable),
    .intr_ecc_wide_correctable_o(irq.ecc_wide_correctable)
  );

  //////////////////////
  //   CHIP CONTROL   //
  //////////////////////
  // Contains NDA and chip specific information.
  <% axi_lite_cut_chip_ctrl = soc_axi_lite_narrow_periph_xbar.out_chip_ctrl.cut(context, cuts_chip_ctrl_cfg, name="soc_axi_lite_narrow_periph_xbar_out_chip_ctrl_cut")  %>

  assign chip_ctrl_req_o = ${axi_lite_cut_chip_ctrl.req_name()};
  assign ${axi_lite_cut_chip_ctrl.rsp_name()} = chip_ctrl_rsp_i;

  //////////////
  //   UART   //
  //////////////

  <% regbus_uart = soc_axi_lite_narrow_periph_xbar.out_uart \
    .cut(context, cuts_uart_cfg, name="soc_axi_lite_narrow_periph_xbar_out_uart_cut") \
    .to_reg(context, "axi_lite_to_reg_uart") %>
  <% uart_apb = regbus_uart.to_apb(context, "uart_apb") %>
  apb_uart_wrap #(
    .apb_req_t (${uart_apb.req_type()} ),
    .apb_rsp_t (${uart_apb.rsp_type()} )
  ) i_uart (
    .clk_i (${uart_apb.clk}),
    .rst_ni (${uart_apb.rst}),
    .apb_req_i (${uart_apb.req_name()}),
    .apb_rsp_o (${uart_apb.rsp_name()}),
    .intr_o (irq.uart),
    .out1_no (  ),  // keep open
    .out2_no (  ),  // keep open
    .rts_no (uart_rts_no), 
    .dtr_no (  ),   // two-wire flow control
    .cts_ni (uart_cts_ni), 
    .dsr_ni (1'b0), // two-wire flow control
    .dcd_ni (1'b0), // two-wire flow control
    .rin_ni (1'b0),
    .sin_i (uart_rx_i),
    .sout_o (uart_tx_o)
  );

  /////////////
  //   ROM   //
  /////////////

  // This is very system specific, so we might be better off
  // placing it outside the top-level.
  <% axi_lite_cut_bootrom = soc_axi_lite_narrow_periph_xbar.out_bootrom.cut(context, cuts_bootrom_cfg, name="soc_axi_lite_narrow_periph_xbar_out_bootrom_cut")  %>
  assign bootrom_req_o = ${axi_lite_cut_bootrom.req_name()};
  assign ${axi_lite_cut_bootrom.rsp_name()} = bootrom_rsp_i;

  //////////////
  //   PLIC   //
  //////////////
  <% regbus_plic = soc_axi_lite_narrow_periph_xbar.out_plic \
    .cut(context, cuts_plic_cfg, name="soc_axi_lite_narrow_periph_xbar_out_plic_cut") \
    .to_reg(context, "axi_lite_to_reg_plic") %>
  rv_plic #(
    .reg_req_t (${regbus_plic.req_type()}),
    .reg_rsp_t (${regbus_plic.rsp_type()})
  ) i_rv_plic (
    .clk_i      ( ${regbus_plic.clk} ),
    .rst_ni     ( ${regbus_plic.rst} ),
    .reg_req_i  ( ${regbus_plic.req_name()} ),
    .reg_rsp_o  ( ${regbus_plic.rsp_name()} ),
    .intr_src_i ( irq ),
    .irq_o      ( eip ),
    .irq_id_o   ( ),
    .msip_o     ( )
  );

  assign irq.zero = 1'b0;

  //////////////////
  //   SPI Host   //
  //////////////////
  <% regbus_spim = soc_axi_lite_narrow_periph_xbar.out_spim \
    .cut(context, cuts_spim_cfg, name="soc_axi_lite_narrow_periph_xbar_out_spim_cut") \
    .to_reg(context, "axi_lite_to_reg_spim") %>

  logic [3:0] spim_sd_i, spim_sd_o, spim_sd_en_o;
  spi_host #(
    .reg_req_t (${regbus_spim.req_type()}),
    .reg_rsp_t (${regbus_spim.rsp_type()})
  ) i_spi_host (
    // TODO(zarubaf): Fix clock assignment
    .clk_i  (${regbus_spim.clk}),
    .rst_ni (${regbus_spim.rst}),
    .clk_core_i (${regbus_spim.clk}),
    .rst_core_ni (${regbus_spim.rst}),
    .reg_req_i (${regbus_spim.req_name()}),
    .reg_rsp_o (${regbus_spim.rsp_name()}),
    .cio_sck_o (spim_sck_o),
    .cio_sck_en_o (),
    .cio_csb_o (spim_csb_o),
    .cio_csb_en_o (),
    .cio_sd_o (spim_sd_o),
    .cio_sd_en_o (spim_sd_en_o),
    .cio_sd_i (spim_sd_i),
    .intr_error_o (irq.spim_error),
    .intr_spi_event_o (irq.spim_spi_event)
  );


  // Unidirectional - Bidirectional transform
  always_comb begin
    if (spim_sd_en_o > 0) begin             // Output Mode
      spim_sd_i = 4'b1;                     // Tie-off input
      spim_sd_io = spim_sd_o;
    end else begin                          // Input Mode
      spim_sd_i = spim_sd_io;
    end
  end

  //////////////
  //   GPIO   //
  //////////////
  <% regbus_gpio = soc_axi_lite_narrow_periph_xbar.out_gpio \
    .cut(context, cuts_gpio_cfg, name="soc_axi_lite_narrow_periph_xbar_out_gpio_cut") \
    .to_reg(context, "axi_lite_to_reg_gpio") %>
  gpio #(
    .reg_req_t (${regbus_gpio.req_type()}),
    .reg_rsp_t (${regbus_gpio.rsp_type()})
  ) i_gpio (
    .clk_i (${regbus_gpio.clk}),
    .rst_ni (${regbus_gpio.rst}),
    .reg_req_i (${regbus_gpio.req_name()}),
    .reg_rsp_o (${regbus_gpio.rsp_name()}),
    .cio_gpio_i (gpio_d_i),
    .cio_gpio_o (gpio_d_o),
    .cio_gpio_en_o (gpio_oe_o),
    .intr_gpio_o (irq.gpio)
  );

  /////////////
  //   I2C   //
  /////////////
  <% regbus_i2c = soc_axi_lite_narrow_periph_xbar.out_i2c \
    .cut(context, cuts_i2c_cfg, name="soc_axi_lite_narrow_periph_xbar_out_i2c_cut") \
    .to_reg(context, "axi_lite_to_reg_i2c") %>

  logic i2c_scl_i i2c_scl_o i2c_scl_en_o i2c_sda_i i2c_sda_o i2c_sda_en_o;
  
  i2c #(
    .reg_req_t (${regbus_i2c.req_type()}),
    .reg_rsp_t (${regbus_i2c.rsp_type()})
  ) i_i2c (
    .clk_i (${regbus_i2c.clk}),
    .rst_ni (${regbus_i2c.rst}),
    .reg_req_i (${regbus_i2c.req_name()}),
    .reg_rsp_o (${regbus_i2c.rsp_name()}),
    .cio_scl_i (i2c_scl_i),
    .cio_scl_o (i2c_scl_o),
    .cio_scl_en_o (i2c_scl_en_o),
    .cio_sda_i (i2c_sda_i),
    .cio_sda_o (i2c_sda_o),
    .cio_sda_en_o (i2c_sda_en_o),
    .intr_fmt_watermark_o (irq.i2c_fmt_watermark),
    .intr_rx_watermark_o (irq.i2c_rx_watermark),
    .intr_fmt_overflow_o (irq.i2c_fmt_overflow),
    .intr_rx_overflow_o (irq.i2c_rx_overflow),
    .intr_nak_o (irq.i2c_nak),
    .intr_scl_interference_o (irq.i2c_scl_interference),
    .intr_sda_interference_o (irq.i2c_sda_interference),
    .intr_stretch_timeout_o (irq.i2c_stretch_timeout),
    .intr_sda_unstable_o (irq.i2c_sda_unstable),
    .intr_trans_complete_o (irq.i2c_trans_complete),
    .intr_tx_empty_o (irq.i2c_tx_empty),
    .intr_tx_nonempty_o (irq.i2c_tx_nonempty),
    .intr_tx_overflow_o (irq.i2c_tx_overflow),
    .intr_acq_overflow_o (irq.i2c_acq_overflow),
    .intr_ack_stop_o (irq.i2c_ack_stop),
    .intr_host_timeout_o (irq.i2c_host_timeout)
  );

  // Unidirectional - Bidirectional transform
  always_comb begin
    if (i2c_sda_en_o) begin                 // Output Mode
      i2c_sda_i = 1'b1;                     // Tie-off input
      i2c_sda_io = i2c_sda_o ? 1'bZ : 1'b0; // Open-drain connection;
    end else begin                          // Input Mode
      i2c_sda_i = i2c_sda_io;
    end
  end

  always_comb begin
    if (i2c_scl_en_o) begin                 // Output Mode
      i2c_scl_i = 1'b1;                     // Tie-off input
      i2c_scl_io = i2c_scl_o ? 1'bZ : 1'b0; // Open-drain connection
    end else begin                          // Input Mode
      i2c_scl_i = i2c_scl_io;
    end
  end

  /////////////
  //  Timer  //
  /////////////
  <% regbus_timer = soc_axi_lite_narrow_periph_xbar.out_timer \
    .cut(context, cuts_timer_cfg, name="soc_axi_lite_narrow_periph_xbar_out_timer_cut") \
    .to_reg(context, "axi_lite_to_reg_timer") %>
  <% apb_timer_bus = regbus_timer.to_apb(context, "apb_timer") %>
  apb_timer #(
    .APB_ADDR_WIDTH (${apb_timer_bus.aw}),
    .TIMER_CNT (2)
  ) i_apb_timer (
    .HCLK (${apb_timer_bus.clk}),
    .HRESETn (${apb_timer_bus.rst}),
    .PADDR (apb_timer_req.paddr),
    .PWDATA (apb_timer_req.pwdata),
    .PWRITE (apb_timer_req.pwrite),
    .PSEL (apb_timer_req.psel),
    .PENABLE (apb_timer_req.penable),
    .PRDATA (apb_timer_rsp.prdata),
    .PREADY (apb_timer_rsp.pready),
    .PSLVERR (apb_timer_rsp.pslverr),
    .irq_o (irq.timer)
  );

endmodule
