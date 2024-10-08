// Copyright 2020 ETH Zurich and University of Bologna.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51

// Author: Florian Zaruba <zarubaf@iis.ee.ethz.ch>
// Author: Fabian Schuiki <fschuiki@iis.ee.ethz.ch>
//
// AUTOMATICALLY GENERATED by genoccamy.py; edit the script instead.

<%
  cuts_narrow_to_quad = occamy_cfg["cuts"]["narrow_to_quad"]
  cuts_quad_to_narrow = occamy_cfg["cuts"]["quad_to_narrow"]
  cuts_wide_to_quad = occamy_cfg["cuts"]["wide_to_quad"]
  cuts_quad_to_wide = occamy_cfg["cuts"]["quad_to_wide"]
  cuts_narrow_to_cva6 = occamy_cfg["cuts"]["narrow_to_cva6"]
  cuts_narrow_conv_to_spm_narrow_pre = occamy_cfg["cuts"]["narrow_conv_to_spm_narrow_pre"]
  cuts_narrow_conv_to_spm_narrow = occamy_cfg["cuts"]["narrow_conv_to_spm_narrow"]
  cuts_narrow_and_wide = occamy_cfg["cuts"]["narrow_and_wide"]
  cuts_wide_conv_to_spm_wide = occamy_cfg["cuts"]["wide_conv_to_spm_wide"]
  cuts_wide_to_wide_zero_mem = occamy_cfg["cuts"]["wide_to_wide_zero_mem"]
  cuts_wide_and_inter = occamy_cfg["cuts"]["wide_and_inter"]
  cuts_periph_axi_lite_narrow = occamy_cfg["cuts"]["periph_axi_lite_narrow"]
  cuts_periph_axi_lite = occamy_cfg["cuts"]["periph_axi_lite"]
  txns_wide_and_inter = occamy_cfg["txns"]["wide_and_inter"]
  txns_narrow_and_wide = occamy_cfg["txns"]["narrow_and_wide"]
  cuts_withing_atomic_adapter_narrow = occamy_cfg["cuts"]["atomic_adapter_narrow"]
  cuts_withing_atomic_adapter_narrow_wide = occamy_cfg["cuts"]["atomic_adapter_narrow_wide"]
  max_atomics_narrow = 8
  max_atomics_wide = 8
  max_trans_atop_filter_per = 4
  max_trans_atop_filter_ser = 32
  hbi_trunc_addr_width = 40
%>

`include "common_cells/registers.svh"
`include "register_interface/typedef.svh"
`include "axi/assign.svh"

module ${name}_soc
  import ${name}_pkg::*;
(
  input  logic        clk_i,
  input  logic        rst_ni,
  input  logic        test_mode_i,
  input  logic [${occamy_cfg["addr_width"]-1}:0] boot_addr_i,
  // Peripheral Ports (to AXI-lite Xbar)
  output  ${soc_narrow_xbar.out_periph.req_type()} periph_axi_lite_req_o,
  input   ${soc_narrow_xbar.out_periph.rsp_type()} periph_axi_lite_rsp_i,

  input   ${soc_narrow_xbar.in_periph.req_type()} periph_axi_lite_req_i,
  output  ${soc_narrow_xbar.in_periph.rsp_type()} periph_axi_lite_rsp_o,

  // Peripheral Ports (to Regbus Xbar)
  output  ${soc_narrow_xbar.out_axi_lite_narrow_periph.req_type()} periph_axi_lite_narrow_req_o,
  input   ${soc_narrow_xbar.out_axi_lite_narrow_periph.rsp_type()} periph_axi_lite_narrow_rsp_i,

  // To SPM
  output ${soc_wide_xbar.out_spm_wide.req_type()} spm_axi_wide_req_o,
  input  ${soc_wide_xbar.out_spm_wide.rsp_type()} spm_axi_wide_rsp_i,

  // SoC control register IO
  output logic [1:0] spm_narrow_rerror_o,
  output logic [1:0] spm_wide_rerror_o,

  // Interrupts and debug requests
  input  logic [${cores-1}:0] mtip_i,
  input  logic [${cores-1}:0] msip_i,
  input  logic [1:0] eip_i,
  input  logic [0:0] debug_req_i,

  /// SRAM configuration
  input sram_cfgs_t sram_cfgs_i
);


  ///////////////
  // Crossbars //
  ///////////////

  ${module}

  ///////////////////////////////////
  // Connections between crossbars //
  ///////////////////////////////////
  <%
    #// narrow xbar -> wide xbar & wide xbar -> narrow xbar
    soc_narrow_xbar.out_soc_wide \
      .atomic_adapter(context, max_trans=max_atomics_wide, user_as_id=1, user_id_msb=soc_narrow_xbar.out_soc_wide.uw-1, user_id_lsb=0, n_cuts= cuts_withing_atomic_adapter_narrow_wide, name="soc_narrow_wide_amo_adapter") \
      .cut(context, cuts_narrow_and_wide) \
      .change_iw(context, soc_wide_xbar.in_soc_narrow.iw, "soc_narrow_wide_iwc", max_txns_per_id=txns_narrow_and_wide) \
      .change_uw(context, soc_wide_xbar.in_soc_narrow.uw, "soc_narrow_wide_uwc") \
      .change_dw(context, soc_wide_xbar.in_soc_narrow.dw, "soc_narrow_wide_dw", to=soc_wide_xbar.in_soc_narrow)
    soc_wide_xbar.out_soc_narrow \
      .change_iw(context, soc_narrow_xbar.in_soc_wide.iw, "soc_wide_narrow_iwc", max_txns_per_id=txns_narrow_and_wide) \
      .change_uw(context, soc_narrow_xbar.in_soc_wide.uw, "soc_wide_narrow_uwc") \
      .change_dw(context, soc_narrow_xbar.in_soc_wide.dw, "soc_wide_narrow_dw") \
      .cut(context, cuts_narrow_and_wide, to=soc_narrow_xbar.in_soc_wide)
  %>\

  //////////
  // CVA6 //
  //////////
  <%
    cva6_mst = soc_narrow_xbar.__dict__["in_cva6"].copy(name="cva6_mst").declare(context)
    cva6_mst.cut(context, cuts_narrow_to_cva6, to=soc_narrow_xbar.__dict__["in_cva6"])
  %>\

  ${name}_cva6 i_${name}_cva6 (
    .clk_i (clk_i),
    .rst_ni (rst_ni),
    .irq_i (eip_i),
    .ipi_i (msip_i[0]),
    .time_irq_i (mtip_i[0]),
    .debug_req_i (debug_req_i[0]),
    .boot_addr_i (boot_addr_i),
    .axi_req_o (${cva6_mst.req_name()}),
    .axi_resp_i (${cva6_mst.rsp_name()}),
    .sram_cfg_i (sram_cfgs_i.cva6)
  );

  % for i in range(nr_s1_quadrants):
  //////////////////////
  // S1 Quadrant ${i} //
  //////////////////////
  <%
    #// Derived parameters
    nr_cores_s1_quadrant = nr_cores_quadrant
    lower_core = i * nr_cores_s1_quadrant + 1
    #// narrow xbar -> quad & quad -> narrow xbar
    narrow_in = soc_narrow_xbar.__dict__["out_s1_quadrant_{}".format(i)].cut(context, cuts_narrow_to_quad, name="narrow_in_{}".format(i))
    narrow_out = soc_narrow_xbar.__dict__["in_s1_quadrant_{}".format(i)].copy(name="narrow_out_{}".format(i)).declare(context)
    narrow_out.cut(context, cuts_quad_to_narrow, name="narrow_out_cut_{}".format(i), to=soc_narrow_xbar.__dict__["in_s1_quadrant_{}".format(i)])
    #// wide xbar -> quad & quad -> wide xbar
    wide_in = soc_wide_xbar.__dict__["out_quadrant_{}".format(i)].cut(context, cuts_wide_to_quad, name="wide_in_{}".format(i))
    #//wide_out = quadrant_pre_xbars[i].in_quadrant.copy(name="wide_out_{}".format(i)).declare(context)
    wide_out = soc_wide_xbar.__dict__["in_quadrant_{}".format(i)].copy(name="wide_out_{}".format(i)).declare(context)
    wide_out.cut(context, cuts_quad_to_wide, name="wide_out_cut_{}".format(i), to=soc_wide_xbar.__dict__["in_quadrant_{}".format(i)])
  %>\

  ${name}_quadrant_s1 i_${name}_quadrant_s1_${i} (
    .clk_i (clk_i),
    .rst_ni (rst_ni),
    .test_mode_i (test_mode_i),
    .boot_addr_i (boot_addr_i[31:0]),
    .chip_id_i (8'b0),  // Temporary solution as the Chip ID is not provided yet
    .meip_i ('0),
    .mtip_i (mtip_i[${lower_core + nr_cores_s1_quadrant - 1}:${lower_core}]),
    .msip_i (msip_i[${lower_core + nr_cores_s1_quadrant - 1}:${lower_core}]),
    .quadrant_narrow_out_req_o (${narrow_out.req_name()}),
    .quadrant_narrow_out_rsp_i (${narrow_out.rsp_name()}),
    .quadrant_narrow_in_req_i (${narrow_in.req_name()}),
    .quadrant_narrow_in_rsp_o (${narrow_in.rsp_name()}),
    .quadrant_wide_out_req_o (${wide_out.req_name()}),
    .quadrant_wide_out_rsp_i (${wide_out.rsp_name()}),
    .quadrant_wide_in_req_i (${wide_in.req_name()}),
    .quadrant_wide_in_rsp_o (${wide_in.rsp_name()}),
    .sram_cfg_i (sram_cfgs_i.quadrant)
  );

  % endfor

  ////////////////
  // SPM NARROW //
  ////////////////
  <% narrow_spm_mst = soc_narrow_xbar.out_spm_narrow \
                      .cut(context, cuts_narrow_conv_to_spm_narrow_pre) \
                      .atomic_adapter(context, max_trans=max_atomics_narrow, user_as_id=1, user_id_msb=soc_narrow_xbar.out_spm_narrow.uw-1, user_id_lsb=0, n_cuts= cuts_withing_atomic_adapter_narrow,name="spm_narrow_amo_adapter") \
                      .cut(context, cuts_narrow_conv_to_spm_narrow)
  %>\

  <% spm_narrow_words = occamy_cfg["spm_narrow"]["length"]//(soc_narrow_xbar.out_spm_narrow.dw//8) %>\

  typedef logic [${util.clog2(spm_narrow_words) + util.clog2(soc_narrow_xbar.out_spm_narrow.dw//8)-1}:0] mem_narrow_addr_t;
  typedef logic [${soc_narrow_xbar.out_spm_narrow.dw-1}:0] mem_narrow_data_t;
  typedef logic [${soc_narrow_xbar.out_spm_narrow.dw//8-1}:0] mem_narrow_strb_t;

  logic spm_narrow_req, spm_narrow_gnt, spm_narrow_we, spm_narrow_rvalid;
  mem_narrow_addr_t spm_narrow_addr;
  mem_narrow_data_t spm_narrow_wdata, spm_narrow_rdata;
  mem_narrow_strb_t spm_narrow_strb;

  axi_to_mem_interleaved #(
    .axi_req_t (${narrow_spm_mst.req_type()}),
    .axi_resp_t (${narrow_spm_mst.rsp_type()}),
    .AddrWidth (${util.clog2(spm_narrow_words) + util.clog2(narrow_spm_mst.dw//8)}),
    .DataWidth (${narrow_spm_mst.dw}),
    .IdWidth (${narrow_spm_mst.iw}),
    .NumBanks (1),
    .BufDepth (16)
  ) i_axi_to_wide_mem (
    .clk_i (${narrow_spm_mst.clk}),
    .rst_ni (${narrow_spm_mst.rst}),
    .busy_o (),
    .test_i (test_mode_i),
    .axi_req_i (${narrow_spm_mst.req_name()}),
    .axi_resp_o (${narrow_spm_mst.rsp_name()}),
    .mem_req_o (spm_narrow_req),
    .mem_gnt_i (spm_narrow_gnt),
    .mem_addr_o (spm_narrow_addr),
    .mem_wdata_o (spm_narrow_wdata),
    .mem_strb_o (spm_narrow_strb),
    .mem_atop_o (),
    .mem_we_o (spm_narrow_we),
    .mem_rvalid_i (spm_narrow_rvalid),
    .mem_rdata_i (spm_narrow_rdata)
  );

  spm_1p_adv #(
    .NumWords (${spm_narrow_words}),
    .DataWidth (${narrow_spm_mst.dw}),
    .ByteWidth (8),
    .EnableInputPipeline (1'b1),
    .EnableOutputPipeline (1'b1),
    .sram_cfg_t (sram_cfg_t)
  ) i_spm_narrow_cut (
    .clk_i (${narrow_spm_mst.clk}),
    .rst_ni (${narrow_spm_mst.rst}),
    .valid_i (spm_narrow_req),
    .ready_o (spm_narrow_gnt),
    .we_i (spm_narrow_we),
    .addr_i (spm_narrow_addr[${util.clog2(spm_narrow_words) + util.clog2(narrow_spm_mst.dw//8)-1}:${util.clog2(narrow_spm_mst.dw//8)}]),
    .wdata_i (spm_narrow_wdata),
    .be_i (spm_narrow_strb),
    .rdata_o (spm_narrow_rdata),
    .rvalid_o (spm_narrow_rvalid),
    .rerror_o (spm_narrow_rerror_o),
    .sram_cfg_i (sram_cfgs_i.spm_narrow)
  );

  //////////////
  // SPM WIDE //
  //////////////
  <% wide_spm_mst = soc_wide_xbar.out_spm_wide \
                    .cut(context, cuts_wide_conv_to_spm_wide)
  %>\

  assign spm_axi_wide_req_o = ${wide_spm_mst.req_name()};
  assign ${wide_spm_mst.rsp_name()} = spm_axi_wide_rsp_i;

  //////////////////////
  // WIDE ZERO MEMORY //
  //////////////////////
  <% wide_zero_mem_mst = soc_wide_xbar.out_wide_zero_mem \
                         .cut(context, cuts_wide_to_wide_zero_mem)
  %>\

  <% wide_zero_mem_words = occamy_cfg["wide_zero_mem"]["length"]//(soc_wide_xbar.out_wide_zero_mem.dw//8) %>\

  axi_zero_mem #(
    .axi_req_t (${wide_zero_mem_mst.req_type()}),
    .axi_resp_t (${wide_zero_mem_mst.rsp_type()}),
    .AddrWidth (${util.clog2(wide_zero_mem_words) + util.clog2(wide_zero_mem_mst.dw//8)}),
    .DataWidth (${wide_zero_mem_mst.dw}),
    .IdWidth (${wide_zero_mem_mst.iw}),
    .NumBanks (1),
    .BufDepth (4)
  ) i_axi_wide_zero_mem (
    .clk_i (${wide_zero_mem_mst.clk}),
    .rst_ni (${wide_zero_mem_mst.rst}),
    .busy_o (),
    .axi_req_i (${wide_zero_mem_mst.req_name()}),
    .axi_resp_o (${wide_zero_mem_mst.rsp_name()})
  );

  //////////////
  // SYS iDMA //
  //////////////

  <% out_sys_idma_cfg = soc_narrow_xbar.__dict__["out_sys_idma_cfg"] \
  .atomic_adapter(context, filter=True, max_trans=max_trans_atop_filter_per, name="out_sys_idma_cfg_noatop", inst_name="i_out_sys_idma_cfg_atop_filter") \
  %>\

  // .change_dw(context, 32, "out_sys_idma_cfg_dw") \

  <% in_sys_idma_mst  = soc_wide_xbar.__dict__["in_sys_idma_mst"] %>\

  // burst request
  typedef struct packed {
    logic [${wide_in.iw-1}:0] id;
    logic [${wide_in.aw-1}:0] src, dst;
    logic [${wide_in.aw-1}:0] num_bytes;
    axi_pkg::cache_t    cache_src, cache_dst;
    axi_pkg::burst_t    burst_src, burst_dst;
    logic               decouple_rw;
    logic               deburst;
    logic               serialize;
  } idma_burst_req_t;

  // local regbus definition
  `REG_BUS_TYPEDEF_ALL(idma_cfg_reg_a${wide_in.aw}_d64, logic [${wide_in.aw-1}:0], logic [63:0], logic [7:0])

  idma_burst_req_t idma_burst_req;
  logic idma_be_valid;
  logic idma_be_ready;
  logic idma_be_idle;
  logic idma_be_trans_complete;

  idma_cfg_reg_a${wide_in.aw}_d64_req_t idma_cfg_reg_req;
  idma_cfg_reg_a${wide_in.aw}_d64_rsp_t idma_cfg_reg_rsp;

  axi_to_reg #(
    .ADDR_WIDTH( ${out_sys_idma_cfg.aw}                 ),
    .DATA_WIDTH( ${out_sys_idma_cfg.dw}                 ),
    .ID_WIDTH  ( ${out_sys_idma_cfg.iw}                 ),
    .USER_WIDTH( ${out_sys_idma_cfg.uw}                 ),
    .axi_req_t ( ${out_sys_idma_cfg.req_type()}         ),
    .axi_rsp_t ( ${out_sys_idma_cfg.rsp_type()}         ),
    .reg_req_t ( idma_cfg_reg_a${wide_in.aw}_d64_req_t  ),
    .reg_rsp_t ( idma_cfg_reg_a${wide_in.aw}_d64_rsp_t  )
  ) i_axi_to_reg_sys_idma_cfg (
    .clk_i,
    .rst_ni,
    .testmode_i ( 1'b0                           ),
    .axi_req_i  ( ${out_sys_idma_cfg.req_name()} ),
    .axi_rsp_o  ( ${out_sys_idma_cfg.rsp_name()} ),
    .reg_req_o  ( idma_cfg_reg_req               ),
    .reg_rsp_i  ( idma_cfg_reg_rsp               )
  );

  idma_reg64_frontend #(
    .DmaAddrWidth    ( 'd64                                  ),
    .dma_regs_req_t  ( idma_cfg_reg_a${wide_in.aw}_d64_req_t ),
    .dma_regs_rsp_t  ( idma_cfg_reg_a${wide_in.aw}_d64_rsp_t ),
    .burst_req_t     ( idma_burst_req_t                      )
  ) i_idma_reg64_frontend_sys_idma (
    .clk_i,
    .rst_ni,
    .dma_ctrl_req_i   ( idma_cfg_reg_req       ),
    .dma_ctrl_rsp_o   ( idma_cfg_reg_rsp       ),
    .burst_req_o      ( idma_burst_req         ),
    .valid_o          ( idma_be_valid          ),
    .ready_i          ( idma_be_ready          ),
    .backend_idle_i   ( idma_be_idle           ),
    .trans_complete_i ( idma_be_trans_complete )
  );

  axi_dma_backend #(
    .DataWidth      ( ${in_sys_idma_mst.dw}         ),
    .AddrWidth      ( ${in_sys_idma_mst.aw}         ),
    .IdWidth        ( ${in_sys_idma_mst.iw}         ),
    .AxReqFifoDepth ( 'd64                          ),
    .TransFifoDepth ( 'd16                          ),
    .BufferDepth    ( 'd3                           ),
    .axi_req_t      ( ${in_sys_idma_mst.req_type()} ),
    .axi_res_t      ( ${in_sys_idma_mst.rsp_type()} ),
    .burst_req_t    ( idma_burst_req_t              ),
    .DmaIdWidth     ( 'd32                          ),
    .DmaTracing     ( 1'b1                          )
  ) i_axi_dma_backend_sys_idma (
    .clk_i,
    .rst_ni,
    .dma_id_i         ( 'd0                           ),
    .chip_id_i        ( '0                            ),
    .axi_dma_req_o    ( ${in_sys_idma_mst.req_name()} ),
    .axi_dma_res_i    ( ${in_sys_idma_mst.rsp_name()} ),
    .burst_req_i      ( idma_burst_req                ),
    .valid_i          ( idma_be_valid                 ),
    .ready_o          ( idma_be_ready                 ),
    .backend_idle_o   ( idma_be_idle                  ),
    .trans_complete_o ( idma_be_trans_complete        )
  );


  /////////////////
  // Peripherals //
  /////////////////
  <%
    periph_axi_lite_narrow_out = soc_narrow_xbar.__dict__["out_axi_lite_narrow_periph"] \
      .atomic_adapter(context, filter=True, max_trans=max_trans_atop_filter_per, name="periph_axi_lite_narrow_out_noatop", inst_name="i_periph_axi_lite_narrow_out_atop_filter") \
      .cut(context, cuts_periph_axi_lite_narrow, name="periph_axi_lite_narrow_out", inst_name="i_periph_axi_lite_narrow_out_cut")
    periph_axi_lite_out = soc_narrow_xbar.__dict__["out_periph"] \
      .atomic_adapter(context, filter=True, max_trans=max_trans_atop_filter_per, name="periph_axi_lite_out_noatop", inst_name="i_periph_axi_lite_out_atop_filter") \
      .cut(context, cuts_periph_axi_lite, name="periph_axi_lite_out", inst_name="i_periph_axi_lite_out_cut") \

    periph_axi_lite_in = soc_narrow_xbar.__dict__["in_periph"].copy(name="periph_axi_lite_in").declare(context)
    periph_axi_lite_in.cut(context, cuts_periph_axi_lite, to=soc_narrow_xbar.__dict__["in_periph"])
  %>\

  // Inputs
  assign ${periph_axi_lite_in.req_name()} = periph_axi_lite_req_i;
  assign periph_axi_lite_rsp_o = ${periph_axi_lite_in.rsp_name()};

  // Outputs
  assign periph_axi_lite_req_o = ${periph_axi_lite_out.req_name()};
  assign ${periph_axi_lite_out.rsp_name()} = periph_axi_lite_rsp_i;
  assign periph_axi_lite_narrow_req_o = ${periph_axi_lite_narrow_out.req_name()};
  assign ${periph_axi_lite_narrow_out.rsp_name()} = periph_axi_lite_narrow_rsp_i;

endmodule
