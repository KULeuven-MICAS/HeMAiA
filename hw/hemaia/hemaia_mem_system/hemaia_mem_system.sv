// Copyright 2025 KU Leuven.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51

// Author: Yunhao Deng <yunhao.deng@kuleuven.be>

`include "axi/assign.svh"
`include "axi/typedef.svh"
`include "common_cells/assertions.svh"

`include "mem_interface/typedef.svh"
`include "tcdm_interface/typedef.svh"

module hemaia_mem_system #(
    parameter type chip_id_t = logic,
    // AXI Wide Port
    parameter type axi_wide_master_req_t = logic,
    parameter type axi_wide_master_rsp_t = logic,
    parameter type axi_wide_slave_req_t = logic,
    parameter type axi_wide_slave_rsp_t = logic,
    // AXI Narrow Port
    parameter type axi_narrow_master_req_t = logic,
    parameter type axi_narrow_master_rsp_t = logic,
    parameter type axi_narrow_slave_req_t = logic,
    parameter type axi_narrow_slave_rsp_t = logic,
    // XMDA Defines
    parameter int unsigned ClusterAddressSpace = 48'h400000,
    parameter int unsigned MemBaseAddr = 32'h80000000,
    parameter int unsigned MemBankNum = 32,
    parameter int unsigned MemSize = 32'h100000
) (
    input  logic                   clk_i,
    input  logic                   rst_ni,
    input  chip_id_t               chip_id_i,
    // Wide axi ports
    input  axi_wide_master_req_t   axi_wide_master_req_i,
    output axi_wide_master_rsp_t   axi_wide_master_rsp_o,
    output axi_wide_slave_req_t    axi_wide_slave_req_o,
    input  axi_wide_slave_rsp_t    axi_wide_slave_rsp_i,
    // Narrow axi ports
    // The narrow ports are purely for the xdma to send/rcv
    // the control signals (xdma cfg, grant, finish)
    input  axi_narrow_master_req_t axi_narrow_master_req_i,
    output axi_narrow_master_rsp_t axi_narrow_master_rsp_o,
    output axi_narrow_slave_req_t  axi_narrow_slave_req_o,
    input  axi_narrow_slave_rsp_t  axi_narrow_slave_rsp_i
);

  localparam int unsigned AxiWideMasterDataWidth = $bits(axi_wide_master_req_i.w.data);
  localparam int unsigned AxiWideMasterIdWidth = $bits(axi_wide_master_req_i.aw.id);
  localparam int unsigned AxiWideMasterAddrWidth = $bits(axi_wide_master_req_i.aw.addr);
  localparam int unsigned AxiWideMasterUserWidth = $bits(axi_wide_master_req_i.aw.user);

  localparam int unsigned AxiNarrowMasterIdWidth = $bits(axi_narrow_master_req_i.aw.id);

  localparam int unsigned MemDepth = MemSize / 8 / MemBankNum;
  localparam int unsigned BanksPerSuperBank = AxiWideMasterDataWidth / 64;
  localparam int unsigned MemSuperBankNum = MemBankNum / BanksPerSuperBank;

  if (MemDepth * MemBankNum != MemSize / 8) begin : gen_mem_size_check
    initial begin
      $error("Memory depth and bank number mismatch");
      $finish;
    end
  end
  if ((MemBankNum & (MemBankNum - 1)) != 0) begin : gen_mem_bank_check
    initial begin
      $error("MemBankNum is not a power of two");
      $finish;
    end
  end


  localparam axi_pkg::xbar_cfg_t HeMAiAMemXbarCfg = '{
      NoSlvPorts: 1,
      NoMstPorts: 2,
      MaxMstTrans: 16,
      MaxSlvTrans: 16,
      FallThrough: 1'b0,
      LatencyMode: axi_pkg::CUT_ALL_PORTS,
      PipelineStages: 0,
      AxiIdWidthSlvPorts: AxiWideMasterIdWidth,
      AxiIdUsedSlvPorts: AxiWideMasterIdWidth,
      UniqueIds: 1'b0,
      AxiAddrWidth: AxiWideMasterAddrWidth,
      AxiDataWidth: AxiWideMasterDataWidth,
      NoAddrRules: 2
  };

  // Define the AXI type for the ports that before the xbar or after the xbar
  `AXI_TYPEDEF_ALL_CT(axi_in_pre_xbar, axi_in_pre_xbar_req_t, axi_in_pre_xbar_resp_t,
                      logic [AxiWideMasterAddrWidth-1:0], logic [AxiWideMasterIdWidth-1:0],
                      logic [AxiWideMasterDataWidth-1:0], logic [AxiWideMasterDataWidth/8-1:0],
                      logic [AxiWideMasterUserWidth-1:0])

  // The types after the xbar is one wire extra for the ID as there is two ports
  `AXI_TYPEDEF_ALL_CT(axi_in_post_xbar, axi_in_post_xbar_req_t, axi_in_post_xbar_resp_t,
                      logic [AxiWideMasterAddrWidth-1:0],
                      logic [AxiWideMasterIdWidth-1+$clog2(HeMAiAMemXbarCfg.NoSlvPorts):0],
                      logic [AxiWideMasterDataWidth-1:0], logic [AxiWideMasterDataWidth/8-1:0],
                      logic [AxiWideMasterUserWidth-1:0])

  typedef struct packed {
    int unsigned idx;
    logic [AxiWideMasterAddrWidth-1:0] start_addr;
    logic [AxiWideMasterAddrWidth-1:0] end_addr;
  } xbar_rule_t;

  axi_in_pre_xbar_req_t   [0:0] axi_pre_xbar_req;
  axi_in_pre_xbar_resp_t  [0:0] axi_pre_xbar_rsp;
  axi_in_post_xbar_req_t  [1:0] axi_post_xbar_req;
  axi_in_post_xbar_resp_t [1:0] axi_post_xbar_rsp;

  assign axi_pre_xbar_req[0]   = axi_wide_master_req_i;
  assign axi_wide_master_rsp_o = axi_pre_xbar_rsp[0];

  xbar_rule_t [1:0] mem_system_xbar_rule;
  // The 0nd port is for the XDMA virtual addresses
  // The 1st port is for the memory's direct access

  logic [AxiWideMasterAddrWidth-1:0] memory_start_address;
  logic [AxiWideMasterAddrWidth-1:0] memory_end_address;

  assign memory_start_address = {chip_id_i, 40'b0} + MemBaseAddr;
  assign memory_end_address   = {chip_id_i, 40'b0} + MemBaseAddr + MemSize;

  logic [AxiWideMasterAddrWidth-1:0] xdma_region_start_address;
  logic [AxiWideMasterAddrWidth-1:0] xdma_region_end_address;
  // The xdma data region is at [(end_addr - 16KB) - (end_addr - 12KB)] with the size of 4KB
  assign xdma_region_start_address = {chip_id_i, 40'b0} + 48'h1_0000_0000 - 'h4000;
  assign xdma_region_end_address = xdma_region_start_address + 'h1000;

  assign mem_system_xbar_rule = '{
          '{idx: 0, start_addr: xdma_region_start_address, end_addr: xdma_region_end_address},
          '{idx: 1, start_addr: memory_start_address, end_addr: memory_end_address}
      };

  // The XBar of the memory system
  axi_xbar #(
      .Cfg(HeMAiAMemXbarCfg),
      .ATOPs(0),
      .slv_aw_chan_t(axi_in_pre_xbar_aw_chan_t),
      .mst_aw_chan_t(axi_in_post_xbar_aw_chan_t),
      .w_chan_t(axi_in_pre_xbar_w_chan_t),
      .slv_b_chan_t(axi_in_pre_xbar_b_chan_t),
      .mst_b_chan_t(axi_in_post_xbar_b_chan_t),
      .slv_ar_chan_t(axi_in_pre_xbar_ar_chan_t),
      .mst_ar_chan_t(axi_in_post_xbar_ar_chan_t),
      .slv_r_chan_t(axi_in_pre_xbar_r_chan_t),
      .mst_r_chan_t(axi_in_post_xbar_r_chan_t),
      .slv_req_t(axi_in_pre_xbar_req_t),
      .slv_resp_t(axi_in_pre_xbar_resp_t),
      .mst_req_t(axi_in_post_xbar_req_t),
      .mst_resp_t(axi_in_post_xbar_resp_t),
      .rule_t(xbar_rule_t)
  ) i_mem_sys_xbar (
      .clk_i(clk_i),
      .rst_ni(rst_ni),
      .test_i(1'b0),
      .slv_ports_req_i(axi_pre_xbar_req),
      .slv_ports_resp_o(axi_pre_xbar_rsp),
      .mst_ports_req_o(axi_post_xbar_req),
      .mst_ports_resp_i(axi_post_xbar_rsp),
      .addr_map_i(mem_system_xbar_rule),
      .en_default_mst_port_i('0),
      .default_mst_port_i('0)
  );

  // TCDM memory banks
  // ----------------
  // Memory Subsystem
  // ----------------

  typedef logic [$clog2(MemDepth)-1:0] mem_addr_t;
  typedef logic [64-1:0] mem_data_t;
  typedef logic [AxiWideMasterDataWidth-1:0] wide_mem_data_t;
  typedef logic [8-1:0] mem_strb_t;
  typedef logic [AxiWideMasterDataWidth/8-1:0] wide_mem_strb_t;

  typedef logic [$clog2(MemSize)-1:0] tcdm_addr_t;

  `MEM_TYPEDEF_ALL(mem, mem_addr_t, mem_data_t, mem_strb_t, logic)
  `MEM_TYPEDEF_ALL(wide_mem, mem_addr_t, wide_mem_data_t, wide_mem_strb_t, logic)

  // TCDM definitions
  typedef struct packed {
    logic core_id;
    logic is_core;
  } tcdm_user_t;

  `TCDM_TYPEDEF_ALL(tcdm, tcdm_addr_t, mem_data_t, mem_strb_t, tcdm_user_t)
  `TCDM_TYPEDEF_ALL(wide_tcdm, tcdm_addr_t, wide_mem_data_t, wide_mem_strb_t, tcdm_user_t)

  logic      [MemBankNum-1:0] mem_cs;
  logic      [MemBankNum-1:0] mem_wen;
  mem_addr_t [MemBankNum-1:0] mem_add;
  mem_strb_t [MemBankNum-1:0] mem_be;
  mem_data_t [MemBankNum-1:0] mem_wdata;
  mem_data_t [MemBankNum-1:0] mem_rdata;

  typedef logic sram_cfg_t;
  typedef struct packed {sram_cfg_t tcdm;} sram_cfgs_t;

  snitch_data_mem #(
      .TCDMDepth      (MemDepth),
      .NarrowDataWidth(64),
      .NumTotalBanks  (MemBankNum),
      .tcdm_mem_addr_t(mem_addr_t),
      .strb_t         (mem_strb_t),
      .data_t         (mem_data_t),
      .sram_cfg_t     (sram_cfg_t),
      .sram_cfgs_t    (sram_cfgs_t)
  ) i_hemaia_mem (
      .clk_i      (clk_i),
      .rst_ni     (rst_ni),
      .sram_cfgs_i('0),
      .mem_cs_i   (mem_cs),
      .mem_add_i  (mem_add),
      .mem_wen_i  (mem_wen),

      .mem_be_i   (mem_be),
      .mem_wdata_i(mem_wdata),
      .mem_rdata_o(mem_rdata)
  );

  wide_tcdm_req_t [0:0] wide_mem_req;
  wide_tcdm_rsp_t [0:0] wide_mem_rsp;

  tcdm_req_t [BanksPerSuperBank*2-1:0] xdma_req;
  tcdm_rsp_t [BanksPerSuperBank*2-1:0] xdma_rsp;

  wide_mem_req_t [MemSuperBankNum-1:0] sb_req;
  wide_mem_rsp_t [MemSuperBankNum-1:0] sb_rsp;

  mem_req_t [MemBankNum-1:0] b_req;
  mem_rsp_t [MemBankNum-1:0] b_rsp;

  // The inteconnect to connect TCDM to Wide AXI
  snitch_tcdm_interconnect #(
      .NumInp(1),
      .NumOut(MemSuperBankNum),
      .tcdm_req_t(wide_tcdm_req_t),
      .tcdm_rsp_t(wide_tcdm_rsp_t),
      .mem_req_t(wide_mem_req_t),
      .mem_rsp_t(wide_mem_rsp_t),
      .user_t(logic),
      .MemAddrWidth($clog2(MemDepth)),
      .DataWidth(AxiWideMasterDataWidth),
      .MemoryResponseLatency(1)
  ) i_wide_interconnect (
      .clk_i,
      .rst_ni,
      .req_i(wide_mem_req),
      .rsp_o(wide_mem_rsp),
      .mem_req_o(sb_req),
      .mem_rsp_i(sb_rsp)
  );

  // The interconnect to connect TCDM to XDMA
  snitch_tcdm_interconnect #(
      .NumInp(BanksPerSuperBank * 2),
      .NumOut(MemBankNum),
      .tcdm_req_t(tcdm_req_t),
      .tcdm_rsp_t(tcdm_rsp_t),
      .mem_req_t(mem_req_t),
      .mem_rsp_t(mem_rsp_t),
      .user_t(logic),
      .MemAddrWidth($clog2(MemDepth)),
      .DataWidth(64),
      .MemoryResponseLatency(1)
  ) i_narrow_interconnect (
      .clk_i,
      .rst_ni,
      .req_i(xdma_req),
      .rsp_o(xdma_rsp),
      .mem_req_o(b_req),
      .mem_rsp_i(b_rsp)
  );

  for (genvar i = 0; i < MemSuperBankNum; i++) begin : gen_wide_narrow_mux
    mem_req_t [BanksPerSuperBank-1:0] mem_req;
    mem_rsp_t [BanksPerSuperBank-1:0] mem_rsp;

    for (genvar j = 0; j < BanksPerSuperBank; j++) begin : gen_mem_reqrsp
      assign mem_cs[i*BanksPerSuperBank+j] = mem_req[j].q_valid;
      assign mem_wen[i*BanksPerSuperBank+j] = mem_req[j].q.write;
      assign mem_add[i*BanksPerSuperBank+j] = mem_req[j].q.addr;
      assign mem_be[i*BanksPerSuperBank+j] = mem_req[j].q.strb;
      assign mem_wdata[i*BanksPerSuperBank+j] = mem_req[j].q.data;
      assign mem_rsp[j].p.data = mem_rdata[i*BanksPerSuperBank+j];
      assign mem_rsp[j].q_ready = mem_req[j].q_valid;
    end

    mem_wide_narrow_mux #(
        .NarrowDataWidth(64),
        .WideDataWidth(AxiWideMasterDataWidth),
        .mem_narrow_req_t(mem_req_t),
        .mem_narrow_rsp_t(mem_rsp_t),
        .mem_wide_req_t(wide_mem_req_t),
        .mem_wide_rsp_t(wide_mem_rsp_t)
    ) i_tcdm_mux (
        .clk_i,
        .rst_ni,
        .in_narrow_req_i(b_req[(i+1)*BanksPerSuperBank-1:i*BanksPerSuperBank]),
        .in_narrow_rsp_o(b_rsp[(i+1)*BanksPerSuperBank-1:i*BanksPerSuperBank]),
        .in_wide_req_i(sb_req[i]),
        .in_wide_rsp_o(sb_rsp[i]),
        .out_req_o(mem_req),
        .out_rsp_i(mem_rsp),
        .sel_wide_i(sb_req[i].q_valid)
    );
  end

  logic [AxiWideMasterAddrWidth-1:0] wide_mem_req_addr_nontrunc;
  assign wide_mem_req[0].q.addr = tcdm_addr_t'(wide_mem_req_addr_nontrunc);
  axi_to_mem_interleaved #(
      .axi_req_t(axi_in_post_xbar_req_t),
      .axi_resp_t(axi_in_post_xbar_resp_t),
      .AddrWidth(AxiWideMasterAddrWidth),
      .DataWidth(AxiWideMasterDataWidth),
      .IdWidth(AxiWideMasterIdWidth + $clog2(HeMAiAMemXbarCfg.NoSlvPorts)),
      .NumBanks(1),
      .BufDepth(2)
  ) i_axi_to_mem_hemaia (
      .clk_i,
      .rst_ni,
      .busy_o(),
      .test_i(1'b0),
      .axi_req_i(axi_post_xbar_req[1]),
      .axi_resp_o(axi_post_xbar_rsp[1]),
      .mem_req_o(wide_mem_req[0].q_valid),
      .mem_gnt_i(wide_mem_rsp[0].q_ready),
      .mem_addr_o(wide_mem_req_addr_nontrunc),
      .mem_wdata_o(wide_mem_req[0].q.data),
      .mem_strb_o(wide_mem_req[0].q.strb),
      .mem_atop_o(  /* The main memory does not support ATOP */),
      .mem_we_o(wide_mem_req[0].q.write),
      .mem_rvalid_i(wide_mem_rsp[0].p_valid),
      .mem_rdata_i(wide_mem_rsp[0].p.data)
  );

  hemaia_xdma_wrapper #(
      .tcdm_req_t         (tcdm_req_t),
      .tcdm_rsp_t         (tcdm_rsp_t),
      // Wide ports
      .wide_slv_id_t      (logic [AxiWideMasterIdWidth+$clog2(HeMAiAMemXbarCfg.NoSlvPorts)-1:0]),
      .wide_out_req_t     (axi_wide_slave_req_t),
      .wide_out_resp_t    (axi_wide_slave_rsp_t),
      .wide_in_req_t      (axi_in_post_xbar_req_t),
      .wide_in_resp_t     (axi_in_post_xbar_resp_t),
      // Narrow ports
      .narrow_slv_id_t    (logic [AxiNarrowMasterIdWidth-1:0]),
      .narrow_out_req_t   (axi_narrow_slave_req_t),
      .narrow_out_resp_t  (axi_narrow_slave_rsp_t),
      .narrow_in_req_t    (axi_narrow_master_req_t),
      .narrow_in_resp_t   (axi_narrow_master_rsp_t),
      // TCDM
      .TCDMNumPorts       (BanksPerSuperBank * 2),
      .TCDMAddrWidth      ($clog2(MemSize)),
      .ClusterAddressSpace(ClusterAddressSpace)
  ) i_hemaia_xdma_wrapper (
      .clk_i                 (clk_i),
      .rst_ni                (rst_ni),
      .cluster_base_addr_i   ({chip_id_i, 40'b0} + MemBaseAddr),
      .tcdm_req_o            (xdma_req),
      .tcdm_rsp_i            (xdma_rsp),
      .csr_req_bits_data_i   ('0),
      .csr_req_bits_addr_i   ('0),
      .csr_req_bits_write_i  (1'b0),
      .csr_req_valid_i       (1'b0),
      .csr_req_ready_o       (),
      .csr_rsp_bits_data_o   (),
      .csr_rsp_valid_o       (),
      .csr_rsp_ready_i       (1'b1),
      // Wide ports
      .xdma_wide_out_req_o   (axi_wide_slave_req_o),
      .xdma_wide_out_resp_i  (axi_wide_slave_rsp_i),
      .xdma_wide_in_req_i    (axi_post_xbar_req[0]),
      .xdma_wide_in_resp_o   (axi_post_xbar_rsp[0]),
      // Narrow ports
      .xdma_narrow_out_req_o (axi_narrow_slave_req_o),
      .xdma_narrow_out_resp_i(axi_narrow_slave_rsp_i),
      .xdma_narrow_in_req_i  (axi_narrow_master_req_i),
      .xdma_narrow_in_resp_o (axi_narrow_master_rsp_o)
  );
endmodule
