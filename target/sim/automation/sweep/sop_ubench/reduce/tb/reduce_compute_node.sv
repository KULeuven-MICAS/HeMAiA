// Copyright 2026 KU Leuven.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51
//
// reduce_compute_node: one compute chiplet of tb_reduce, from cluster 0's xDMA endpoint to the
// D2D link's single 512-bit local port (soc2router / router2soc).
//
// It rebuilds, stage for stage, the path the real chip has between cluster 0's xDMA and the
// link, with the same IPs and the same parameters:
//
//   cluster (clk_cl, 500 MHz)       snitch_cluster.sv: the DMA (wide) xbar and the cluster
//                                   (narrow) xbar, CUT_ALL_PORTS, and the RegisterExtWide /
//                                   RegisterExtNarrow axi_cuts on the cluster ports
//   cluster <-> quadrant            occamy_quad.sv: id remaps + axi_cdc (LogDepth 3) x 4
//   quadrant (clk_fab, 1 GHz)       occamy_quad.sv: quad wide / narrow xbars, their cuts and id
//                                   remaps; occamy_quad_ctrl.sv: the soc_to_quad and
//                                   quad_to_soc xbars
//   SoC (clk_fab)                   occamy_soc.sv: the quad cuts (cfg narrow_to_quad 3,
//                                   quad_to_narrow 3, wide_to_quad 3, quad_to_wide 3), the SoC
//                                   narrow and wide xbars, the narrow -> wide path (atomics
//                                   adapter + cut + id remap + 64 -> 512 dw converter), the
//                                   wide -> narrow path (id remap + 512 -> 64 dw converter +
//                                   cut), and soc_to_router / router_to_soc (1 cut each)
//
// For the software arms and the hardware-scheduler edges it also has
//   - the cluster iDMA (snitch_cc.sv: idma_inst64_top) on the DMA xbar's SDMAMst port, its
//     instruction interface brought out (the testbench issues the DM core's DMA instructions);
//   - the TCDM's wide DMA port behind axi_to_mem_interleaved on the DMA xbar's TCDMDMA port
//     (the TCDM model is in the testbench);
//   - the cores' stores, as an AXI master on the cluster xbar's CoreReq port;
//   - occamy_quad_ctrl.sv's quadrant AXI-Lite xbar with the BINGO manager's two chiplet ports
//     (to_remote_chiplet_axi_lite in, CHIPLET_DONE_QUEUE mailbox out) and the conversions that
//     join it to the quad_to_soc / soc_to_quad xbars (the manager itself is a testbench model).
//
// Everything else on those chips (the snitch cores, the cluster's TCDM interconnect and other
// engines, cluster 1, CVA6, the L3 SPMs, the peripherals, the memory-system and sys-iDMA ports)
// is absent: their xbar ports are tied off, and an assertion fires if any request is ever
// routed to one of them.

`include "axi/typedef.svh"
`include "axi/assign.svh"

module reduce_compute_node
  import reduce_pkg::*;
(
  input  logic        clk_cl_i,
  input  logic        clk_fab_i,
  input  logic        rst_ni,
  input  chip_id_t    chip_id_i,
  // xDMA CSR port (driven by the testbench where the DM core's CSR path would)
  input  logic [31:0] csr_addr_i,
  input  logic        csr_write_i,
  input  logic [31:0] csr_wdata_i,
  input  logic        csr_valid_i,
  output logic        csr_ready_o,
  output logic [31:0] csr_rdata_o,
  output logic        csr_rsp_valid_o,
  input  logic        csr_rsp_ready_i,
  // xDMA TCDM ports (the TCDM model lives in the testbench)
  output tcdm_req_t [TcdmPorts-1:0] tcdm_req_o,
  input  tcdm_rsp_t [TcdmPorts-1:0] tcdm_rsp_i,
  // The D2D link's local port
  output axi_a48_d512_i4_u1_req_t  soc2router_req_o,
  input  axi_a48_d512_i4_u1_resp_t soc2router_rsp_i,
  input  axi_a48_d512_i4_u1_req_t  router2soc_req_i,
  output axi_a48_d512_i4_u1_resp_t router2soc_rsp_o,
  // Core stores (snrt_xchip_writew of the software arms): an AXI master on the cluster xbar's
  // CoreReq port, where the cores' reqrsp_to_axi enters
  input  axi_a48_d64_i2_u3_req_t   core_req_i,
  output axi_a48_d64_i2_u3_resp_t  core_rsp_o,
  // The cluster iDMA (idma_inst64_top): its frontend takes the DM core's DMA instructions
  input  acc_req_t                 idma_acc_req_i,
  input  logic                     idma_acc_req_valid_i,
  output logic                     idma_acc_req_ready_o,
  output acc_resp_t                idma_acc_rsp_o,
  output logic                     idma_acc_rsp_valid_o,
  input  logic                     idma_acc_rsp_ready_i,
  output logic                     idma_busy_o,
  // The TCDM's wide (512 b) DMA port, behind axi_to_mem_interleaved (the TCDM model is outside)
  output logic                     dma_mem_req_o,
  input  logic                     dma_mem_gnt_i,
  output addr_t                    dma_mem_addr_o,
  output logic [511:0]             dma_mem_wdata_o,
  output logic [63:0]              dma_mem_strb_o,
  output logic                     dma_mem_we_o,
  input  logic                     dma_mem_rvalid_i,
  input  logic [511:0]             dma_mem_rdata_i,
  // The BINGO manager's chiplet ports (the manager itself is modelled in the testbench):
  // to_remote_chiplet_axi_lite (dep-set writes out) and from_remote_axi_lite (its mailbox)
  input  axi_lite_a48_d64_req_t    sched_tx_req_i,
  output axi_lite_a48_d64_rsp_t    sched_tx_rsp_o,
  output axi_lite_a48_d64_req_t    sched_rx_req_o,
  input  axi_lite_a48_d64_rsp_t    sched_rx_rsp_i
);

  // ==========================================================================================
  // Helper: an axi_multicut of the SoC's types
  // ==========================================================================================
`define RD_MULTICUT(__name, __n, __T, __clk, __sreq, __srsp, __mreq, __mrsp) \
  axi_multicut #(                                                             \
    .NoCuts    (__n),                                                         \
    .aw_chan_t (__T``_aw_chan_t),                                             \
    .w_chan_t  (__T``_w_chan_t),                                              \
    .b_chan_t  (__T``_b_chan_t),                                              \
    .ar_chan_t (__T``_ar_chan_t),                                             \
    .r_chan_t  (__T``_r_chan_t),                                              \
    .axi_req_t (__T``_req_t),                                                 \
    .axi_resp_t(__T``_resp_t)                                                 \
  ) __name (                                                                  \
    .clk_i     (__clk),                                                       \
    .rst_ni    (rst_ni),                                                      \
    .slv_req_i (__sreq),                                                      \
    .slv_resp_o(__srsp),                                                      \
    .mst_req_o (__mreq),                                                      \
    .mst_resp_i(__mrsp)                                                       \
  );

  // A request on a port that is tied off on the real chip's side (or that only other engines
  // use) means the testbench routes something the real chip would not: stop loudly.
`define RD_NEVER(__lbl, __clk, __req)                                                   \
  __lbl : assert property (@(posedge __clk) disable iff (!rst_ni)                       \
      !(__req.aw_valid || __req.ar_valid || __req.w_valid))                             \
    else $fatal(1, "[reduce_compute_node %02x] request on an unused port (%m)", chip_id_i);

  // ==========================================================================================
  // Cluster 0 (clk_cl): snitch_cluster.sv as generated for snax_split_cluster
  // ==========================================================================================
  addr_t cluster_base;
  assign cluster_base = {chip_id_i, ClusterBaseOffset};

  // snitch_cluster.sv address windows
  addr_t tcdm_start, tcdm_end, periph_end, mmio_start, mmio_data_end, mmio_end;
  assign tcdm_start    = cluster_base;                       // & TCDMMask (base is aligned)
  assign tcdm_end      = cluster_base + TcdmBytes;
  assign periph_end    = tcdm_end + ClusterPeriphBytes;
  assign mmio_start    = cluster_base + ClusterAddressSpace - MMIOKiB * 1024;
  assign mmio_data_end = mmio_start + (MMIOKiB / 4) * 1024;
  assign mmio_end      = cluster_base + ClusterAddressSpace;

  // ---- Cluster ports (cluster side of the CDCs) ----
  axi_a48_d64_i2_u3_req_t   cl_narrow_in_req;    // into the cluster
  axi_a48_d64_i2_u3_resp_t  cl_narrow_in_rsp;
  axi_a48_d64_i4_u3_req_t   cl_narrow_out_req;   // out of the cluster
  axi_a48_d64_i4_u3_resp_t  cl_narrow_out_rsp;
  axi_a48_d512_i1_u1_req_t  cl_wide_in_req;
  axi_a48_d512_i1_u1_resp_t cl_wide_in_rsp;
  axi_a48_d512_i3_u1_req_t  cl_wide_out_req;
  axi_a48_d512_i3_u1_resp_t cl_wide_out_rsp;

  // ---- xDMA endpoint ----
  // The wrapper's port types are the cluster's, crossed exactly as in
  // snax_split_cluster_wrapper.sv: its wide OUT is a master on the DMA xbar (wide_in type),
  // its wide IN a slave of it (wide_out type); the same for narrow.
  axi_a48_d512_i1_u1_req_t  xdma_wide_out_req;
  axi_a48_d512_i1_u1_resp_t xdma_wide_out_rsp;
  axi_a48_d512_i3_u1_req_t  xdma_wide_in_req;
  axi_a48_d512_i3_u1_resp_t xdma_wide_in_rsp;
  axi_a48_d64_i2_u3_req_t   xdma_narrow_out_req;
  axi_a48_d64_i2_u3_resp_t  xdma_narrow_out_rsp;
  axi_a48_d64_i4_u3_req_t   xdma_narrow_in_req;
  axi_a48_d64_i4_u3_resp_t  xdma_narrow_in_rsp;

  snax_split_cluster_xdma_wrapper #(
    .tcdm_req_t          (tcdm_req_t),
    .tcdm_rsp_t          (tcdm_rsp_t),
    .wide_slv_id_t       (logic [2:0]),                 // snax_split_cluster_pkg::wide_out_id_t
    .wide_out_req_t      (axi_a48_d512_i1_u1_req_t),    // wide_in_req_t
    .wide_out_resp_t     (axi_a48_d512_i1_u1_resp_t),
    .wide_in_req_t       (axi_a48_d512_i3_u1_req_t),    // wide_out_req_t
    .wide_in_resp_t      (axi_a48_d512_i3_u1_resp_t),
    .narrow_slv_id_t     (logic [3:0]),                 // narrow_out_id_t
    .narrow_out_req_t    (axi_a48_d64_i2_u3_req_t),     // narrow_in_req_t
    .narrow_out_resp_t   (axi_a48_d64_i2_u3_resp_t),
    .narrow_in_req_t     (axi_a48_d64_i4_u3_req_t),     // narrow_out_req_t
    .narrow_in_resp_t    (axi_a48_d64_i4_u3_resp_t),
    .TCDMDataWidth       (64),
    .TCDMNumPorts        (TcdmPorts),
    .TCDMAddrWidth       (TcdmAddrWidth),
    .ClusterBaseAddr     (48'h1000_0000),
    .ClusterAddressSpace (ClusterAddressSpace),
    .MMIOSize            (MMIOKiB)
  ) i_xdma (
    .clk_i                 (clk_cl_i),
    .rst_ni                (rst_ni),
    .cluster_base_addr_i   (cluster_base),
    .tcdm_req_o            (tcdm_req_o),
    .tcdm_rsp_i            (tcdm_rsp_i),
    .csr_req_bits_addr_i   (csr_addr_i),
    .csr_req_bits_write_i  (csr_write_i),
    .csr_req_bits_data_i   (csr_wdata_i),
    .csr_req_bits_strb_i   ('1),
    .csr_req_valid_i       (csr_valid_i),
    .csr_req_ready_o       (csr_ready_o),
    .csr_rsp_bits_data_o   (csr_rdata_o),
    .csr_rsp_valid_o       (csr_rsp_valid_o),
    .csr_rsp_ready_i       (csr_rsp_ready_i),
    .xdma_wide_out_req_o   (xdma_wide_out_req),
    .xdma_wide_out_resp_i  (xdma_wide_out_rsp),
    .xdma_wide_in_req_i    (xdma_wide_in_req),
    .xdma_wide_in_resp_o   (xdma_wide_in_rsp),
    .xdma_narrow_out_req_o (xdma_narrow_out_req),
    .xdma_narrow_out_resp_i(xdma_narrow_out_rsp),
    .xdma_narrow_in_req_i  (xdma_narrow_in_req),
    .xdma_narrow_in_resp_o (xdma_narrow_in_rsp)
  );

  // ---- DMA (wide) xbar: snitch_cluster.sv i_axi_dma_xbar ----
  // slaves: SDMAMst 0, SoCDMAIn 1, ICache 2, WideXDMAIn 3; masters: TCDMDMA 0, SoCDMAOut 1,
  // WideXDMAOut 2 (snitch_pkg). Default SoCDMAOut.
  localparam axi_pkg::xbar_cfg_t DmaXbarCfg = '{
    NoSlvPorts:         4,
    NoMstPorts:         3,
    MaxMstTrans:        32,    // WideMaxMstTrans
    MaxSlvTrans:        32,    // WideMaxSlvTrans
    FallThrough:        1'b0,
    LatencyMode:        axi_pkg::CUT_ALL_PORTS,
    PipelineStages:     0,
    AxiIdWidthSlvPorts: 1,
    AxiIdUsedSlvPorts:  1,
    UniqueIds:          1'b0,
    AxiAddrWidth:       48,
    AxiDataWidth:       512,
    NoAddrRules:        2
  };
  axi_a48_d512_i1_u1_req_t  [3:0] dma_xbar_in_req;
  axi_a48_d512_i1_u1_resp_t [3:0] dma_xbar_in_rsp;
  axi_a48_d512_i3_u1_req_t  [2:0] dma_xbar_out_req;
  axi_a48_d512_i3_u1_resp_t [2:0] dma_xbar_out_rsp;
  xbar_rule_48_t [1:0] dma_xbar_rule;
  assign dma_xbar_rule = '{
    '{idx: 32'd0, start_addr: tcdm_start, end_addr: tcdm_end},          // TCDMDMA
    '{idx: 32'd2, start_addr: mmio_start, end_addr: mmio_data_end}      // WideXDMAOut
  };

  axi_xbar #(
    .Cfg          (DmaXbarCfg),
    .ATOPs        (1'b0),
    .slv_aw_chan_t(axi_a48_d512_i1_u1_aw_chan_t),
    .mst_aw_chan_t(axi_a48_d512_i3_u1_aw_chan_t),
    .w_chan_t     (axi_a48_d512_i1_u1_w_chan_t),
    .slv_b_chan_t (axi_a48_d512_i1_u1_b_chan_t),
    .mst_b_chan_t (axi_a48_d512_i3_u1_b_chan_t),
    .slv_ar_chan_t(axi_a48_d512_i1_u1_ar_chan_t),
    .mst_ar_chan_t(axi_a48_d512_i3_u1_ar_chan_t),
    .slv_r_chan_t (axi_a48_d512_i1_u1_r_chan_t),
    .mst_r_chan_t (axi_a48_d512_i3_u1_r_chan_t),
    .slv_req_t    (axi_a48_d512_i1_u1_req_t),
    .slv_resp_t   (axi_a48_d512_i1_u1_resp_t),
    .mst_req_t    (axi_a48_d512_i3_u1_req_t),
    .mst_resp_t   (axi_a48_d512_i3_u1_resp_t),
    .rule_t       (xbar_rule_48_t)
  ) i_dma_xbar (
    .clk_i                (clk_cl_i),
    .rst_ni               (rst_ni),
    .test_i               (1'b0),
    .slv_ports_req_i      (dma_xbar_in_req),
    .slv_ports_resp_o     (dma_xbar_in_rsp),
    .mst_ports_req_o      (dma_xbar_out_req),
    .mst_ports_resp_i     (dma_xbar_out_rsp),
    .addr_map_i           (dma_xbar_rule),
    .en_default_mst_port_i('1),
    .default_mst_port_i   ({4{2'd1}})
  );

  assign dma_xbar_in_req[2] = '0;                    // ICache (absent)
  assign dma_xbar_in_req[3] = xdma_wide_out_req;     // WideXDMAIn
  assign xdma_wide_out_rsp  = dma_xbar_in_rsp[3];
  assign xdma_wide_in_req   = dma_xbar_out_req[2];   // WideXDMAOut
  assign dma_xbar_out_rsp[2] = xdma_wide_in_rsp;

  // ---- SDMAMst: the cluster iDMA (snitch_cc.sv gen_dma: idma_inst64_top) ----
  // DMAAxiReqFifoDepth 16, DMAReqFifoDepth 8 (snax_split_cluster cfg), one channel; the DMA
  // trace is off (SNITCH_NO_DMA_TRACE).
  axi_a48_d512_i1_u1_req_t  [0:0] idma_axi_req;
  axi_a48_d512_i1_u1_resp_t [0:0] idma_axi_rsp;
  logic                     [0:0] idma_busy;
  idma_inst64_top #(
    .AxiAddrWidth   (48),
    .AxiDataWidth   (512),
    .AxiIdWidth     (1),
    .AxiUserWidth   (1),
    .NumAxInFlight  (16),
    .DMAReqFifoDepth(8),
    .NumChannels    (1),
    .DMATracing     (0),
    .axi_ar_chan_t  (axi_a48_d512_i1_u1_ar_chan_t),
    .axi_aw_chan_t  (axi_a48_d512_i1_u1_aw_chan_t),
    .axi_req_t      (axi_a48_d512_i1_u1_req_t),
    .axi_res_t      (axi_a48_d512_i1_u1_resp_t),
    .acc_req_t      (acc_req_t),
    .acc_res_t      (acc_resp_t),
    .dma_events_t   (dma_events_t)
  ) i_idma (
    .clk_i          (clk_cl_i),
    .rst_ni         (rst_ni),
    .testmode_i     (1'b0),
    .axi_req_o      (idma_axi_req),
    .axi_res_i      (idma_axi_rsp),
    .busy_o         (idma_busy),
    .acc_req_i      (idma_acc_req_i),
    .acc_req_valid_i(idma_acc_req_valid_i),
    .acc_req_ready_o(idma_acc_req_ready_o),
    .acc_res_o      (idma_acc_rsp_o),
    .acc_res_valid_o(idma_acc_rsp_valid_o),
    .acc_res_ready_i(idma_acc_rsp_ready_i),
    .hart_id_i      (32'd3),
    .events_o       ()
  );
  assign idma_busy_o        = idma_busy[0];
  assign dma_xbar_in_req[0] = idma_axi_req[0];
  assign idma_axi_rsp[0]    = dma_xbar_in_rsp[0];

  // ---- TCDMDMA: axi_to_mem_interleaved into the TCDM's wide port (snitch_cluster.sv
  //      i_axi_to_mem_dma; NumBanks 1, BufDepth = MemoryMacroLatency + 1) ----
  axi_to_mem_interleaved #(
    .axi_req_t (axi_a48_d512_i3_u1_req_t),
    .axi_resp_t(axi_a48_d512_i3_u1_resp_t),
    .AddrWidth (48),
    .DataWidth (512),
    .IdWidth   (3),
    .NumBanks  (1),
    .BufDepth  (2)
  ) i_axi_to_mem_dma (
    .clk_i       (clk_cl_i),
    .rst_ni      (rst_ni),
    .test_i      (1'b0),
    .busy_o      (),
    .axi_req_i   (dma_xbar_out_req[0]),
    .axi_resp_o  (dma_xbar_out_rsp[0]),
    .mem_req_o   (dma_mem_req_o),
    .mem_gnt_i   (dma_mem_gnt_i),
    .mem_addr_o  (dma_mem_addr_o),
    .mem_wdata_o (dma_mem_wdata_o),
    .mem_strb_o  (dma_mem_strb_o),
    .mem_atop_o  (),
    .mem_we_o    (dma_mem_we_o),
    .mem_rvalid_i(dma_mem_rvalid_i),
    .mem_rdata_i (dma_mem_rdata_i)
  );

  // RegisterExtWide = 1: i_cut_ext_wide_out / i_cut_ext_wide_in
  axi_cut #(
    .Bypass(1'b0),
    .aw_chan_t(axi_a48_d512_i3_u1_aw_chan_t), .w_chan_t(axi_a48_d512_i3_u1_w_chan_t),
    .b_chan_t(axi_a48_d512_i3_u1_b_chan_t), .ar_chan_t(axi_a48_d512_i3_u1_ar_chan_t),
    .r_chan_t(axi_a48_d512_i3_u1_r_chan_t), .axi_req_t(axi_a48_d512_i3_u1_req_t),
    .axi_resp_t(axi_a48_d512_i3_u1_resp_t)
  ) i_cut_ext_wide_out (
    .clk_i(clk_cl_i), .rst_ni(rst_ni),
    .slv_req_i(dma_xbar_out_req[1]), .slv_resp_o(dma_xbar_out_rsp[1]),
    .mst_req_o(cl_wide_out_req), .mst_resp_i(cl_wide_out_rsp)
  );
  axi_cut #(
    .Bypass(1'b0),
    .aw_chan_t(axi_a48_d512_i1_u1_aw_chan_t), .w_chan_t(axi_a48_d512_i1_u1_w_chan_t),
    .b_chan_t(axi_a48_d512_i1_u1_b_chan_t), .ar_chan_t(axi_a48_d512_i1_u1_ar_chan_t),
    .r_chan_t(axi_a48_d512_i1_u1_r_chan_t), .axi_req_t(axi_a48_d512_i1_u1_req_t),
    .axi_resp_t(axi_a48_d512_i1_u1_resp_t)
  ) i_cut_ext_wide_in (
    .clk_i(clk_cl_i), .rst_ni(rst_ni),
    .slv_req_i(cl_wide_in_req), .slv_resp_o(cl_wide_in_rsp),
    .mst_req_o(dma_xbar_in_req[1]), .mst_resp_i(dma_xbar_in_rsp[1])
  );

  // ---- Cluster (narrow) xbar: snitch_cluster.sv i_cluster_xbar ----
  // slaves: CoreReq 0, AXISoC 1, PTW 2, NarrowXDMAIn 3; masters: TCDM 0,
  // ClusterPeripherals 1, SoC 2, NarrowXDMAOut 3. Default SoC.
  localparam axi_pkg::xbar_cfg_t ClusterXbarCfg = '{
    NoSlvPorts:         4,
    NoMstPorts:         4,
    MaxMstTrans:        4,     // NarrowMaxMstTrans
    MaxSlvTrans:        4,     // NarrowMaxSlvTrans
    FallThrough:        1'b0,
    LatencyMode:        axi_pkg::CUT_ALL_PORTS,
    PipelineStages:     0,
    AxiIdWidthSlvPorts: 2,
    AxiIdUsedSlvPorts:  2,
    UniqueIds:          1'b0,
    AxiAddrWidth:       48,
    AxiDataWidth:       64,
    NoAddrRules:        3
  };
  axi_a48_d64_i2_u3_req_t  [3:0] cl_xbar_in_req;
  axi_a48_d64_i2_u3_resp_t [3:0] cl_xbar_in_rsp;
  axi_a48_d64_i4_u3_req_t  [3:0] cl_xbar_out_req;
  axi_a48_d64_i4_u3_resp_t [3:0] cl_xbar_out_rsp;
  xbar_rule_48_t [2:0] cl_xbar_rule;
  assign cl_xbar_rule = '{
    '{idx: 32'd0, start_addr: tcdm_start, end_addr: tcdm_end},          // TCDM
    '{idx: 32'd1, start_addr: tcdm_end, end_addr: periph_end},          // ClusterPeripherals
    '{idx: 32'd3, start_addr: mmio_data_end, end_addr: mmio_end}        // NarrowXDMAOut
  };

  axi_xbar #(
    .Cfg          (ClusterXbarCfg),
    .slv_aw_chan_t(axi_a48_d64_i2_u3_aw_chan_t),
    .mst_aw_chan_t(axi_a48_d64_i4_u3_aw_chan_t),
    .w_chan_t     (axi_a48_d64_i2_u3_w_chan_t),
    .slv_b_chan_t (axi_a48_d64_i2_u3_b_chan_t),
    .mst_b_chan_t (axi_a48_d64_i4_u3_b_chan_t),
    .slv_ar_chan_t(axi_a48_d64_i2_u3_ar_chan_t),
    .mst_ar_chan_t(axi_a48_d64_i4_u3_ar_chan_t),
    .slv_r_chan_t (axi_a48_d64_i2_u3_r_chan_t),
    .mst_r_chan_t (axi_a48_d64_i4_u3_r_chan_t),
    .slv_req_t    (axi_a48_d64_i2_u3_req_t),
    .slv_resp_t   (axi_a48_d64_i2_u3_resp_t),
    .mst_req_t    (axi_a48_d64_i4_u3_req_t),
    .mst_resp_t   (axi_a48_d64_i4_u3_resp_t),
    .rule_t       (xbar_rule_48_t)
  ) i_cluster_xbar (
    .clk_i                (clk_cl_i),
    .rst_ni               (rst_ni),
    .test_i               (1'b0),
    .slv_ports_req_i      (cl_xbar_in_req),
    .slv_ports_resp_o     (cl_xbar_in_rsp),
    .mst_ports_req_o      (cl_xbar_out_req),
    .mst_ports_resp_i     (cl_xbar_out_rsp),
    .addr_map_i           (cl_xbar_rule),
    .en_default_mst_port_i('1),
    .default_mst_port_i   ({4{2'd2}})
  );

  assign cl_xbar_in_req[0] = core_req_i;             // CoreReq: the cores' stores
  assign core_rsp_o        = cl_xbar_in_rsp[0];
  assign cl_xbar_in_req[2] = '0;                     // PTW (absent)
  assign cl_xbar_in_req[3] = xdma_narrow_out_req;    // NarrowXDMAIn
  assign xdma_narrow_out_rsp = cl_xbar_in_rsp[3];
  assign xdma_narrow_in_req  = cl_xbar_out_req[3];   // NarrowXDMAOut
  assign cl_xbar_out_rsp[3]  = xdma_narrow_in_rsp;
  assign cl_xbar_out_rsp[0]  = '0;                   // TCDM (axi_to_tcdm; unused here)
  assign cl_xbar_out_rsp[1]  = '0;                   // cluster peripherals (unused here)
  `RD_NEVER(a_never_cl_tcdm, clk_cl_i, cl_xbar_out_req[0])
  `RD_NEVER(a_never_cl_periph, clk_cl_i, cl_xbar_out_req[1])

  // RegisterExtNarrow = 1: i_cut_ext_narrow_slv / i_cut_ext_narrow_mst
  axi_cut #(
    .Bypass(1'b0),
    .aw_chan_t(axi_a48_d64_i2_u3_aw_chan_t), .w_chan_t(axi_a48_d64_i2_u3_w_chan_t),
    .b_chan_t(axi_a48_d64_i2_u3_b_chan_t), .ar_chan_t(axi_a48_d64_i2_u3_ar_chan_t),
    .r_chan_t(axi_a48_d64_i2_u3_r_chan_t), .axi_req_t(axi_a48_d64_i2_u3_req_t),
    .axi_resp_t(axi_a48_d64_i2_u3_resp_t)
  ) i_cut_ext_narrow_slv (
    .clk_i(clk_cl_i), .rst_ni(rst_ni),
    .slv_req_i(cl_narrow_in_req), .slv_resp_o(cl_narrow_in_rsp),
    .mst_req_o(cl_xbar_in_req[1]), .mst_resp_i(cl_xbar_in_rsp[1])
  );
  axi_cut #(
    .Bypass(1'b0),
    .aw_chan_t(axi_a48_d64_i4_u3_aw_chan_t), .w_chan_t(axi_a48_d64_i4_u3_w_chan_t),
    .b_chan_t(axi_a48_d64_i4_u3_b_chan_t), .ar_chan_t(axi_a48_d64_i4_u3_ar_chan_t),
    .r_chan_t(axi_a48_d64_i4_u3_r_chan_t), .axi_req_t(axi_a48_d64_i4_u3_req_t),
    .axi_resp_t(axi_a48_d64_i4_u3_resp_t)
  ) i_cut_ext_narrow_mst (
    .clk_i(clk_cl_i), .rst_ni(rst_ni),
    .slv_req_i(cl_xbar_out_req[2]), .slv_resp_o(cl_xbar_out_rsp[2]),
    .mst_req_o(cl_narrow_out_req), .mst_resp_i(cl_narrow_out_rsp)
  );

  // ==========================================================================================
  // Quadrant (clk_fab): occamy_quad.sv, cluster 0
  // ==========================================================================================
  // Quad xbar port indices (occamy_pkg)
  localparam int unsigned QW_SOC = 0, QW_CL0 = 1, QW_CL1 = 2;     // quad_wide_xbar
  localparam int unsigned QN_SOC = 0, QN_CL0 = 1, QN_CL1 = 2;     // quad_narrow_xbar

  axi_a48_d512_i3_u1_req_t  [2:0] quad_wide_in_req;
  axi_a48_d512_i3_u1_resp_t [2:0] quad_wide_in_rsp;
  axi_a48_d512_i5_u1_req_t  [2:0] quad_wide_out_req;
  axi_a48_d512_i5_u1_resp_t [2:0] quad_wide_out_rsp;
  axi_a48_d64_i4_u3_req_t   [2:0] quad_narrow_in_req;
  axi_a48_d64_i4_u3_resp_t  [2:0] quad_narrow_in_rsp;
  axi_a48_d64_i6_u3_req_t   [2:0] quad_narrow_out_req;
  axi_a48_d64_i6_u3_resp_t  [2:0] quad_narrow_out_rsp;

  // ---- cluster 0 narrow in: i_narrow_in_iwc_0 (6 -> 2) + i_narrow_cluster_in_cdc_0 ----
  axi_a48_d64_i2_u3_req_t  narrow_in_iwc_0_req;
  axi_a48_d64_i2_u3_resp_t narrow_in_iwc_0_rsp;
  axi_id_remap #(
    .AxiSlvPortIdWidth(6), .AxiSlvPortMaxUniqIds(4), .AxiMaxTxnsPerId(4), .AxiMstPortIdWidth(2),
    .slv_req_t(axi_a48_d64_i6_u3_req_t), .slv_resp_t(axi_a48_d64_i6_u3_resp_t),
    .mst_req_t(axi_a48_d64_i2_u3_req_t), .mst_resp_t(axi_a48_d64_i2_u3_resp_t)
  ) i_narrow_in_iwc_0 (
    .clk_i(clk_fab_i), .rst_ni(rst_ni),
    .slv_req_i(quad_narrow_out_req[QN_CL0]), .slv_resp_o(quad_narrow_out_rsp[QN_CL0]),
    .mst_req_o(narrow_in_iwc_0_req), .mst_resp_i(narrow_in_iwc_0_rsp)
  );
  axi_cdc #(
    .aw_chan_t(axi_a48_d64_i2_u3_aw_chan_t), .w_chan_t(axi_a48_d64_i2_u3_w_chan_t),
    .b_chan_t(axi_a48_d64_i2_u3_b_chan_t), .ar_chan_t(axi_a48_d64_i2_u3_ar_chan_t),
    .r_chan_t(axi_a48_d64_i2_u3_r_chan_t), .axi_req_t(axi_a48_d64_i2_u3_req_t),
    .axi_resp_t(axi_a48_d64_i2_u3_resp_t), .LogDepth(3)
  ) i_narrow_cluster_in_cdc_0 (
    .src_clk_i(clk_fab_i), .src_rst_ni(rst_ni),
    .src_req_i(narrow_in_iwc_0_req), .src_resp_o(narrow_in_iwc_0_rsp),
    .dst_clk_i(clk_cl_i), .dst_rst_ni(rst_ni),
    .dst_req_o(cl_narrow_in_req), .dst_resp_i(cl_narrow_in_rsp)
  );

  // ---- cluster 0 narrow out: i_narrow_cluster_out_cdc_0 ----
  axi_cdc #(
    .aw_chan_t(axi_a48_d64_i4_u3_aw_chan_t), .w_chan_t(axi_a48_d64_i4_u3_w_chan_t),
    .b_chan_t(axi_a48_d64_i4_u3_b_chan_t), .ar_chan_t(axi_a48_d64_i4_u3_ar_chan_t),
    .r_chan_t(axi_a48_d64_i4_u3_r_chan_t), .axi_req_t(axi_a48_d64_i4_u3_req_t),
    .axi_resp_t(axi_a48_d64_i4_u3_resp_t), .LogDepth(3)
  ) i_narrow_cluster_out_cdc_0 (
    .src_clk_i(clk_cl_i), .src_rst_ni(rst_ni),
    .src_req_i(cl_narrow_out_req), .src_resp_o(cl_narrow_out_rsp),
    .dst_clk_i(clk_fab_i), .dst_rst_ni(rst_ni),
    .dst_req_o(quad_narrow_in_req[QN_CL0]), .dst_resp_i(quad_narrow_in_rsp[QN_CL0])
  );

  // ---- cluster 0 wide in: i_wide_in_iwc_0 (5 -> 1) + i_wide_cluster_in_cdc_0 ----
  axi_a48_d512_i1_u1_req_t  wide_in_iwc_0_req;
  axi_a48_d512_i1_u1_resp_t wide_in_iwc_0_rsp;
  axi_id_remap #(
    .AxiSlvPortIdWidth(5), .AxiSlvPortMaxUniqIds(2), .AxiMaxTxnsPerId(32), .AxiMstPortIdWidth(1),
    .slv_req_t(axi_a48_d512_i5_u1_req_t), .slv_resp_t(axi_a48_d512_i5_u1_resp_t),
    .mst_req_t(axi_a48_d512_i1_u1_req_t), .mst_resp_t(axi_a48_d512_i1_u1_resp_t)
  ) i_wide_in_iwc_0 (
    .clk_i(clk_fab_i), .rst_ni(rst_ni),
    .slv_req_i(quad_wide_out_req[QW_CL0]), .slv_resp_o(quad_wide_out_rsp[QW_CL0]),
    .mst_req_o(wide_in_iwc_0_req), .mst_resp_i(wide_in_iwc_0_rsp)
  );
  axi_cdc #(
    .aw_chan_t(axi_a48_d512_i1_u1_aw_chan_t), .w_chan_t(axi_a48_d512_i1_u1_w_chan_t),
    .b_chan_t(axi_a48_d512_i1_u1_b_chan_t), .ar_chan_t(axi_a48_d512_i1_u1_ar_chan_t),
    .r_chan_t(axi_a48_d512_i1_u1_r_chan_t), .axi_req_t(axi_a48_d512_i1_u1_req_t),
    .axi_resp_t(axi_a48_d512_i1_u1_resp_t), .LogDepth(3)
  ) i_wide_cluster_in_cdc_0 (
    .src_clk_i(clk_fab_i), .src_rst_ni(rst_ni),
    .src_req_i(wide_in_iwc_0_req), .src_resp_o(wide_in_iwc_0_rsp),
    .dst_clk_i(clk_cl_i), .dst_rst_ni(rst_ni),
    .dst_req_o(cl_wide_in_req), .dst_resp_i(cl_wide_in_rsp)
  );

  // ---- cluster 0 wide out: i_wide_cluster_out_cdc_0 ----
  axi_cdc #(
    .aw_chan_t(axi_a48_d512_i3_u1_aw_chan_t), .w_chan_t(axi_a48_d512_i3_u1_w_chan_t),
    .b_chan_t(axi_a48_d512_i3_u1_b_chan_t), .ar_chan_t(axi_a48_d512_i3_u1_ar_chan_t),
    .r_chan_t(axi_a48_d512_i3_u1_r_chan_t), .axi_req_t(axi_a48_d512_i3_u1_req_t),
    .axi_resp_t(axi_a48_d512_i3_u1_resp_t), .LogDepth(3)
  ) i_wide_cluster_out_cdc_0 (
    .src_clk_i(clk_cl_i), .src_rst_ni(rst_ni),
    .src_req_i(cl_wide_out_req), .src_resp_o(cl_wide_out_rsp),
    .dst_clk_i(clk_fab_i), .dst_rst_ni(rst_ni),
    .dst_req_o(quad_wide_in_req[QW_CL0]), .dst_resp_i(quad_wide_in_rsp[QW_CL0])
  );

  // Cluster 1 is absent
  assign quad_wide_in_req[QW_CL1]    = '0;
  assign quad_wide_out_rsp[QW_CL1]   = '0;
  assign quad_narrow_in_req[QN_CL1]  = '0;
  assign quad_narrow_out_rsp[QN_CL1] = '0;
  `RD_NEVER(a_never_quad_wide_cl1, clk_fab_i, quad_wide_out_req[QW_CL1])
  `RD_NEVER(a_never_quad_narrow_cl1, clk_fab_i, quad_narrow_out_req[QN_CL1])

  // ---- quad_wide_xbar ----
  localparam axi_pkg::xbar_cfg_t QuadWideXbarCfg = '{
    NoSlvPorts: 3, NoMstPorts: 3, MaxSlvTrans: 32, MaxMstTrans: 32, FallThrough: 1'b0,
    LatencyMode: axi_pkg::CUT_ALL_PORTS, PipelineStages: 0, AxiIdWidthSlvPorts: 3,
    AxiIdUsedSlvPorts: 3, UniqueIds: 1'b0, AxiAddrWidth: 48, AxiDataWidth: 512, NoAddrRules: 9
  };
  xbar_rule_48_t [8:0] QuadWideXbarAddrmap;
  assign QuadWideXbarAddrmap = '{
    '{ idx: 0, start_addr: {chip_id_i,40'h00000000}, end_addr: {chip_id_i,40'h10000000 }},
    '{ idx: 0, start_addr: {chip_id_i,40'h10100000}, end_addr: {chip_id_i,40'h10110000 }},
    '{ idx: 0, start_addr: {chip_id_i,40'h103fd000}, end_addr: {chip_id_i,40'h10400000 }},
    '{ idx: 0, start_addr: {chip_id_i,40'h10500000}, end_addr: {chip_id_i,40'h10510000 }},
    '{ idx: 0, start_addr: {chip_id_i,40'h107fd000}, end_addr: {chip_id_i,40'h1200000000 }},
    '{ idx: 1, start_addr: {chip_id_i,40'h10000000}, end_addr: {chip_id_i,40'h10100000 }},
    '{ idx: 1, start_addr: {chip_id_i,40'h10120000}, end_addr: {chip_id_i,40'h103fd000 }},
    '{ idx: 2, start_addr: {chip_id_i,40'h10400000}, end_addr: {chip_id_i,40'h10500000 }},
    '{ idx: 2, start_addr: {chip_id_i,40'h10520000}, end_addr: {chip_id_i,40'h107fd000 }}
  };
  axi_xbar #(
    .Cfg          (QuadWideXbarCfg),
    .Connectivity (9'b011101110),
    .ATOPs        (1'b0),
    .slv_aw_chan_t(axi_a48_d512_i3_u1_aw_chan_t),
    .mst_aw_chan_t(axi_a48_d512_i5_u1_aw_chan_t),
    .w_chan_t     (axi_a48_d512_i3_u1_w_chan_t),
    .slv_b_chan_t (axi_a48_d512_i3_u1_b_chan_t),
    .mst_b_chan_t (axi_a48_d512_i5_u1_b_chan_t),
    .slv_ar_chan_t(axi_a48_d512_i3_u1_ar_chan_t),
    .mst_ar_chan_t(axi_a48_d512_i5_u1_ar_chan_t),
    .slv_r_chan_t (axi_a48_d512_i3_u1_r_chan_t),
    .mst_r_chan_t (axi_a48_d512_i5_u1_r_chan_t),
    .slv_req_t    (axi_a48_d512_i3_u1_req_t),
    .slv_resp_t   (axi_a48_d512_i3_u1_resp_t),
    .mst_req_t    (axi_a48_d512_i5_u1_req_t),
    .mst_resp_t   (axi_a48_d512_i5_u1_resp_t),
    .rule_t       (xbar_rule_48_t)
  ) i_quad_wide_xbar (
    .clk_i(clk_fab_i), .rst_ni(rst_ni), .test_i(1'b0),
    .slv_ports_req_i(quad_wide_in_req), .slv_ports_resp_o(quad_wide_in_rsp),
    .mst_ports_req_o(quad_wide_out_req), .mst_ports_resp_i(quad_wide_out_rsp),
    .addr_map_i(QuadWideXbarAddrmap), .en_default_mst_port_i('1), .default_mst_port_i('0)
  );

  // ---- quad_narrow_xbar ----
  localparam axi_pkg::xbar_cfg_t QuadNarrowXbarCfg = '{
    NoSlvPorts: 3, NoMstPorts: 3, MaxSlvTrans: 8, MaxMstTrans: 8, FallThrough: 1'b0,
    LatencyMode: axi_pkg::CUT_ALL_PORTS, PipelineStages: 0, AxiIdWidthSlvPorts: 4,
    AxiIdUsedSlvPorts: 4, UniqueIds: 1'b0, AxiAddrWidth: 48, AxiDataWidth: 64, NoAddrRules: 10
  };
  xbar_rule_48_t [9:0] QuadNarrowXbarAddrmap;
  assign QuadNarrowXbarAddrmap = '{
    '{ idx: 0, start_addr: {chip_id_i,40'h00000000}, end_addr: {chip_id_i,40'h10000000 }},
    '{ idx: 0, start_addr: {chip_id_i,40'h70000000}, end_addr: {chip_id_i,40'h1200000000 }},
    '{ idx: 1, start_addr: {chip_id_i,40'h10000000}, end_addr: {chip_id_i,40'h10100000 }},
    '{ idx: 1, start_addr: {chip_id_i,40'h10100000}, end_addr: {chip_id_i,40'h10110000 }},
    '{ idx: 1, start_addr: {chip_id_i,40'h10120000}, end_addr: {chip_id_i,40'h103fd000 }},
    '{ idx: 1, start_addr: {chip_id_i,40'h103fd000}, end_addr: {chip_id_i,40'h10400000 }},
    '{ idx: 2, start_addr: {chip_id_i,40'h10400000}, end_addr: {chip_id_i,40'h10500000 }},
    '{ idx: 2, start_addr: {chip_id_i,40'h10500000}, end_addr: {chip_id_i,40'h10510000 }},
    '{ idx: 2, start_addr: {chip_id_i,40'h10520000}, end_addr: {chip_id_i,40'h107fd000 }},
    '{ idx: 2, start_addr: {chip_id_i,40'h107fd000}, end_addr: {chip_id_i,40'h10800000 }}
  };
  axi_xbar #(
    .Cfg          (QuadNarrowXbarCfg),
    .Connectivity (9'b011101111),
    .ATOPs        (1'b1),
    .slv_aw_chan_t(axi_a48_d64_i4_u3_aw_chan_t),
    .mst_aw_chan_t(axi_a48_d64_i6_u3_aw_chan_t),
    .w_chan_t     (axi_a48_d64_i4_u3_w_chan_t),
    .slv_b_chan_t (axi_a48_d64_i4_u3_b_chan_t),
    .mst_b_chan_t (axi_a48_d64_i6_u3_b_chan_t),
    .slv_ar_chan_t(axi_a48_d64_i4_u3_ar_chan_t),
    .mst_ar_chan_t(axi_a48_d64_i6_u3_ar_chan_t),
    .slv_r_chan_t (axi_a48_d64_i4_u3_r_chan_t),
    .mst_r_chan_t (axi_a48_d64_i6_u3_r_chan_t),
    .slv_req_t    (axi_a48_d64_i4_u3_req_t),
    .slv_resp_t   (axi_a48_d64_i4_u3_resp_t),
    .mst_req_t    (axi_a48_d64_i6_u3_req_t),
    .mst_resp_t   (axi_a48_d64_i6_u3_resp_t),
    .rule_t       (xbar_rule_48_t)
  ) i_quad_narrow_xbar (
    .clk_i(clk_fab_i), .rst_ni(rst_ni), .test_i(1'b0),
    .slv_ports_req_i(quad_narrow_in_req), .slv_ports_resp_o(quad_narrow_in_rsp),
    .mst_ports_req_o(quad_narrow_out_req), .mst_ports_resp_i(quad_narrow_out_rsp),
    .addr_map_i(QuadNarrowXbarAddrmap), .en_default_mst_port_i('1), .default_mst_port_i('0)
  );

  // ---- Narrow in + IW converter: i_narrow_cluster_in_ctrl_cut + i_narrow_cluster_in_iwc ----
  axi_a48_d64_i7_u3_req_t  narrow_cluster_in_ctrl_req, narrow_cluster_in_ctrl_cut_req;
  axi_a48_d64_i7_u3_resp_t narrow_cluster_in_ctrl_rsp, narrow_cluster_in_ctrl_cut_rsp;
  `RD_MULTICUT(i_narrow_cluster_in_ctrl_cut, 1, axi_a48_d64_i7_u3, clk_fab_i,
               narrow_cluster_in_ctrl_req, narrow_cluster_in_ctrl_rsp,
               narrow_cluster_in_ctrl_cut_req, narrow_cluster_in_ctrl_cut_rsp)
  axi_id_remap #(
    .AxiSlvPortIdWidth(7), .AxiSlvPortMaxUniqIds(16), .AxiMaxTxnsPerId(4), .AxiMstPortIdWidth(4),
    .slv_req_t(axi_a48_d64_i7_u3_req_t), .slv_resp_t(axi_a48_d64_i7_u3_resp_t),
    .mst_req_t(axi_a48_d64_i4_u3_req_t), .mst_resp_t(axi_a48_d64_i4_u3_resp_t)
  ) i_narrow_cluster_in_iwc (
    .clk_i(clk_fab_i), .rst_ni(rst_ni),
    .slv_req_i(narrow_cluster_in_ctrl_cut_req), .slv_resp_o(narrow_cluster_in_ctrl_cut_rsp),
    .mst_req_o(quad_narrow_in_req[QN_SOC]), .mst_resp_i(quad_narrow_in_rsp[QN_SOC])
  );

  // ---- Narrow out + IW converter: i_narrow_cluster_out_iwc + i_narrow_cluster_out_ctrl ----
  axi_a48_d64_i3_u3_req_t  narrow_cluster_out_iwc_req, narrow_cluster_out_ctrl_req;
  axi_a48_d64_i3_u3_resp_t narrow_cluster_out_iwc_rsp, narrow_cluster_out_ctrl_rsp;
  axi_id_remap #(
    .AxiSlvPortIdWidth(6), .AxiSlvPortMaxUniqIds(8), .AxiMaxTxnsPerId(4), .AxiMstPortIdWidth(3),
    .slv_req_t(axi_a48_d64_i6_u3_req_t), .slv_resp_t(axi_a48_d64_i6_u3_resp_t),
    .mst_req_t(axi_a48_d64_i3_u3_req_t), .mst_resp_t(axi_a48_d64_i3_u3_resp_t)
  ) i_narrow_cluster_out_iwc (
    .clk_i(clk_fab_i), .rst_ni(rst_ni),
    .slv_req_i(quad_narrow_out_req[QN_SOC]), .slv_resp_o(quad_narrow_out_rsp[QN_SOC]),
    .mst_req_o(narrow_cluster_out_iwc_req), .mst_resp_i(narrow_cluster_out_iwc_rsp)
  );
  `RD_MULTICUT(i_narrow_cluster_out_ctrl, 1, axi_a48_d64_i3_u3, clk_fab_i,
               narrow_cluster_out_iwc_req, narrow_cluster_out_iwc_rsp,
               narrow_cluster_out_ctrl_req, narrow_cluster_out_ctrl_rsp)

  // ---- Wide out: i_quad_wide_xbar_out_cut + i_wide_cluster_out_iwc + i_wide_cluster_out_iwc_cut
  axi_a48_d512_i5_u1_req_t  quad_wide_xbar_out_cut_req;
  axi_a48_d512_i5_u1_resp_t quad_wide_xbar_out_cut_rsp;
  axi_a48_d512_i4_u1_req_t  wide_cluster_out_iwc_req, quadrant_wide_out_req;
  axi_a48_d512_i4_u1_resp_t wide_cluster_out_iwc_rsp, quadrant_wide_out_rsp;
  `RD_MULTICUT(i_quad_wide_xbar_out_cut, 1, axi_a48_d512_i5_u1, clk_fab_i,
               quad_wide_out_req[QW_SOC], quad_wide_out_rsp[QW_SOC],
               quad_wide_xbar_out_cut_req, quad_wide_xbar_out_cut_rsp)
  axi_id_remap #(
    .AxiSlvPortIdWidth(5), .AxiSlvPortMaxUniqIds(16), .AxiMaxTxnsPerId(32), .AxiMstPortIdWidth(4),
    .slv_req_t(axi_a48_d512_i5_u1_req_t), .slv_resp_t(axi_a48_d512_i5_u1_resp_t),
    .mst_req_t(axi_a48_d512_i4_u1_req_t), .mst_resp_t(axi_a48_d512_i4_u1_resp_t)
  ) i_wide_cluster_out_iwc (
    .clk_i(clk_fab_i), .rst_ni(rst_ni),
    .slv_req_i(quad_wide_xbar_out_cut_req), .slv_resp_o(quad_wide_xbar_out_cut_rsp),
    .mst_req_o(wide_cluster_out_iwc_req), .mst_resp_i(wide_cluster_out_iwc_rsp)
  );
  `RD_MULTICUT(i_wide_cluster_out_iwc_cut, 1, axi_a48_d512_i4_u1, clk_fab_i,
               wide_cluster_out_iwc_req, wide_cluster_out_iwc_rsp,
               quadrant_wide_out_req, quadrant_wide_out_rsp)

  // ---- Wide in: i_wide_cluster_in_iwc_cut + i_wide_cluster_in_iwc ----
  axi_a48_d512_i7_u1_req_t  quadrant_wide_in_req, wide_cluster_in_iwc_cut_req;
  axi_a48_d512_i7_u1_resp_t quadrant_wide_in_rsp, wide_cluster_in_iwc_cut_rsp;
  `RD_MULTICUT(i_wide_cluster_in_iwc_cut, 1, axi_a48_d512_i7_u1, clk_fab_i,
               quadrant_wide_in_req, quadrant_wide_in_rsp,
               wide_cluster_in_iwc_cut_req, wide_cluster_in_iwc_cut_rsp)
  axi_id_remap #(
    .AxiSlvPortIdWidth(7), .AxiSlvPortMaxUniqIds(8), .AxiMaxTxnsPerId(4), .AxiMstPortIdWidth(3),
    .slv_req_t(axi_a48_d512_i7_u1_req_t), .slv_resp_t(axi_a48_d512_i7_u1_resp_t),
    .mst_req_t(axi_a48_d512_i3_u1_req_t), .mst_resp_t(axi_a48_d512_i3_u1_resp_t)
  ) i_wide_cluster_in_iwc (
    .clk_i(clk_fab_i), .rst_ni(rst_ni),
    .slv_req_i(wide_cluster_in_iwc_cut_req), .slv_resp_o(wide_cluster_in_iwc_cut_rsp),
    .mst_req_o(quad_wide_in_req[QW_SOC]), .mst_resp_i(quad_wide_in_rsp[QW_SOC])
  );

  // ==========================================================================================
  // Quadrant controller (clk_fab): occamy_quad_ctrl.sv, the two narrow xbars only
  // ==========================================================================================
  axi_a48_d64_i7_u3_req_t  quadrant_narrow_in_req;    // SoC -> quad (soc_in)
  axi_a48_d64_i7_u3_resp_t quadrant_narrow_in_rsp;
  axi_a48_d64_i4_u3_req_t  quadrant_narrow_out_req;   // quad -> SoC (soc_out)
  axi_a48_d64_i4_u3_resp_t quadrant_narrow_out_rsp;

  // soc_to_quad: 1 slave (SOC_NARROW), 3 masters (CLUSTERS 0, QUAD_AXI_LITE 1, _NARROW 2)
  localparam axi_pkg::xbar_cfg_t QuadCtrlSocToQuadXbarCfg = '{
    NoSlvPorts: 1, NoMstPorts: 3, MaxSlvTrans: 32, MaxMstTrans: 32, FallThrough: 1'b0,
    LatencyMode: axi_pkg::CUT_SLV_PORTS, PipelineStages: 0, AxiIdWidthSlvPorts: 7,
    AxiIdUsedSlvPorts: 7, UniqueIds: 1'b0, AxiAddrWidth: 48, AxiDataWidth: 64, NoAddrRules: 9
  };
  xbar_rule_48_t [8:0] QuadCtrlSocToQuadXbarAddrmap;
  assign QuadCtrlSocToQuadXbarAddrmap = '{
    '{ idx: 0, start_addr: {chip_id_i,40'h10000000}, end_addr: {chip_id_i,40'h10100000 }},
    '{ idx: 0, start_addr: {chip_id_i,40'h10400000}, end_addr: {chip_id_i,40'h10500000 }},
    '{ idx: 0, start_addr: {chip_id_i,40'h10100000}, end_addr: {chip_id_i,40'h10110000 }},
    '{ idx: 0, start_addr: {chip_id_i,40'h10500000}, end_addr: {chip_id_i,40'h10510000 }},
    '{ idx: 1, start_addr: {chip_id_i,40'h0a000000}, end_addr: {chip_id_i,40'h0a001000 }},
    '{ idx: 1, start_addr: {chip_id_i,40'h0a001000}, end_addr: {chip_id_i,40'h0a002000 }},
    '{ idx: 2, start_addr: {chip_id_i,40'h0b000000}, end_addr: {chip_id_i,40'h0b001000 }},
    '{ idx: 2, start_addr: {chip_id_i,40'h0b001000}, end_addr: {chip_id_i,40'h0b002000 }},
    '{ idx: 2, start_addr: {chip_id_i,40'h0b002000}, end_addr: {chip_id_i,40'h0b003000 }}
  };
  axi_a48_d64_i7_u3_req_t  [0:0] s2q_in_req;
  axi_a48_d64_i7_u3_resp_t [0:0] s2q_in_rsp;
  axi_a48_d64_i7_u3_req_t  [2:0] s2q_out_req;
  axi_a48_d64_i7_u3_resp_t [2:0] s2q_out_rsp;
  axi_xbar #(
    .Cfg          (QuadCtrlSocToQuadXbarCfg),
    .Connectivity (3'b111),
    .ATOPs        (1'b1),
    .slv_aw_chan_t(axi_a48_d64_i7_u3_aw_chan_t),
    .mst_aw_chan_t(axi_a48_d64_i7_u3_aw_chan_t),
    .w_chan_t     (axi_a48_d64_i7_u3_w_chan_t),
    .slv_b_chan_t (axi_a48_d64_i7_u3_b_chan_t),
    .mst_b_chan_t (axi_a48_d64_i7_u3_b_chan_t),
    .slv_ar_chan_t(axi_a48_d64_i7_u3_ar_chan_t),
    .mst_ar_chan_t(axi_a48_d64_i7_u3_ar_chan_t),
    .slv_r_chan_t (axi_a48_d64_i7_u3_r_chan_t),
    .mst_r_chan_t (axi_a48_d64_i7_u3_r_chan_t),
    .slv_req_t    (axi_a48_d64_i7_u3_req_t),
    .slv_resp_t   (axi_a48_d64_i7_u3_resp_t),
    .mst_req_t    (axi_a48_d64_i7_u3_req_t),
    .mst_resp_t   (axi_a48_d64_i7_u3_resp_t),
    .rule_t       (xbar_rule_48_t)
  ) i_quad_ctrl_soc_to_quad_xbar (
    .clk_i(clk_fab_i), .rst_ni(rst_ni), .test_i(1'b0),
    .slv_ports_req_i(s2q_in_req), .slv_ports_resp_o(s2q_in_rsp),
    .mst_ports_req_o(s2q_out_req), .mst_ports_resp_i(s2q_out_rsp),
    .addr_map_i(QuadCtrlSocToQuadXbarAddrmap), .en_default_mst_port_i('1),
    .default_mst_port_i('0)
  );
  assign s2q_in_req[0]          = quadrant_narrow_in_req;
  assign quadrant_narrow_in_rsp = s2q_in_rsp[0];
  assign narrow_cluster_in_ctrl_req = s2q_out_req[0];
  assign s2q_out_rsp[0]             = narrow_cluster_in_ctrl_rsp;
  assign s2q_out_rsp[2] = '0;
  `RD_NEVER(a_never_s2q_lite_narrow, clk_fab_i, s2q_out_req[2])

  // quad_to_soc: 2 slaves (CLUSTERS 0, QUAD_AXI_LITE 1), 3 masters (SOC_NARROW 0, ...)
  localparam axi_pkg::xbar_cfg_t QuadCtrlQuadToSocXbarCfg = '{
    NoSlvPorts: 2, NoMstPorts: 3, MaxSlvTrans: 32, MaxMstTrans: 32, FallThrough: 1'b0,
    LatencyMode: axi_pkg::CUT_MST_PORTS, PipelineStages: 0, AxiIdWidthSlvPorts: 3,
    AxiIdUsedSlvPorts: 3, UniqueIds: 1'b0, AxiAddrWidth: 48, AxiDataWidth: 64, NoAddrRules: 5
  };
  xbar_rule_48_t [4:0] QuadCtrlQuadToSocXbarAddrmap;
  assign QuadCtrlQuadToSocXbarAddrmap = '{
    '{ idx: 1, start_addr: {chip_id_i,40'h0a000000}, end_addr: {chip_id_i,40'h0a001000 }},
    '{ idx: 1, start_addr: {chip_id_i,40'h0a001000}, end_addr: {chip_id_i,40'h0a002000 }},
    '{ idx: 2, start_addr: {chip_id_i,40'h0b000000}, end_addr: {chip_id_i,40'h0b001000 }},
    '{ idx: 2, start_addr: {chip_id_i,40'h0b001000}, end_addr: {chip_id_i,40'h0b002000 }},
    '{ idx: 2, start_addr: {chip_id_i,40'h0b002000}, end_addr: {chip_id_i,40'h0b003000 }}
  };
  axi_a48_d64_i3_u3_req_t  [1:0] q2s_in_req;
  axi_a48_d64_i3_u3_resp_t [1:0] q2s_in_rsp;
  axi_a48_d64_i4_u3_req_t  [2:0] q2s_out_req;
  axi_a48_d64_i4_u3_resp_t [2:0] q2s_out_rsp;
  axi_xbar #(
    .Cfg          (QuadCtrlQuadToSocXbarCfg),
    .Connectivity (6'b111111),
    .ATOPs        (1'b1),
    .slv_aw_chan_t(axi_a48_d64_i3_u3_aw_chan_t),
    .mst_aw_chan_t(axi_a48_d64_i4_u3_aw_chan_t),
    .w_chan_t     (axi_a48_d64_i3_u3_w_chan_t),
    .slv_b_chan_t (axi_a48_d64_i3_u3_b_chan_t),
    .mst_b_chan_t (axi_a48_d64_i4_u3_b_chan_t),
    .slv_ar_chan_t(axi_a48_d64_i3_u3_ar_chan_t),
    .mst_ar_chan_t(axi_a48_d64_i4_u3_ar_chan_t),
    .slv_r_chan_t (axi_a48_d64_i3_u3_r_chan_t),
    .mst_r_chan_t (axi_a48_d64_i4_u3_r_chan_t),
    .slv_req_t    (axi_a48_d64_i3_u3_req_t),
    .slv_resp_t   (axi_a48_d64_i3_u3_resp_t),
    .mst_req_t    (axi_a48_d64_i4_u3_req_t),
    .mst_resp_t   (axi_a48_d64_i4_u3_resp_t),
    .rule_t       (xbar_rule_48_t)
  ) i_quad_ctrl_quad_to_soc_xbar (
    .clk_i(clk_fab_i), .rst_ni(rst_ni), .test_i(1'b0),
    .slv_ports_req_i(q2s_in_req), .slv_ports_resp_o(q2s_in_rsp),
    .mst_ports_req_o(q2s_out_req), .mst_ports_resp_i(q2s_out_rsp),
    .addr_map_i(QuadCtrlQuadToSocXbarAddrmap), .en_default_mst_port_i('1),
    .default_mst_port_i('0)
  );
  assign q2s_in_req[0] = narrow_cluster_out_ctrl_req;
  assign narrow_cluster_out_ctrl_rsp = q2s_in_rsp[0];
  assign quadrant_narrow_out_req = q2s_out_req[0];
  assign q2s_out_rsp[0] = quadrant_narrow_out_rsp;
  assign q2s_out_rsp[1] = '0;
  assign q2s_out_rsp[2] = '0;
  `RD_NEVER(a_never_q2s_lite, clk_fab_i, q2s_out_req[1])
  `RD_NEVER(a_never_q2s_lite_narrow, clk_fab_i, q2s_out_req[2])

  // ---- The quadrant AXI-Lite xbar and the BINGO manager's ports (occamy_quad_ctrl.sv) ----
  // quad_axi_lite_xbar: 5 slaves (SOC_TO_QUAD 0, QUAD_TO_SOC 1, BINGO_READ_LOCAL_TASK 2,
  // BINGO_WRITE_REMOTE_DONE 3, BINGO_WRITE_PM 4), 3 masters (QUAD_TO_SOC 0,
  // BINGO_CHIPLET_DONE_QUEUE 1, BINGO_HOST_READY_DONE_INTF 2), CUT_ALL_PORTS.
  // A remote dep set leaves the manager on WRITE_REMOTE_DONE, goes out on QUAD_TO_SOC
  // (axi_lite_to_axi) into quad_to_soc's QUAD_AXI_LITE slave port; one arriving from another
  // chip leaves soc_to_quad on QUAD_AXI_LITE (atop filter + axi_to_axi_lite), enters on
  // SOC_TO_QUAD and is routed to the manager's CHIPLET_DONE_QUEUE mailbox.
  localparam axi_pkg::xbar_cfg_t QuadAxiLiteXbarCfg = '{
    NoSlvPorts: 5, NoMstPorts: 3, MaxSlvTrans: 32, MaxMstTrans: 32, FallThrough: 1'b0,
    LatencyMode: axi_pkg::CUT_ALL_PORTS, PipelineStages: 0, AxiIdWidthSlvPorts: 0,
    AxiIdUsedSlvPorts: 0, UniqueIds: 1'b0, AxiAddrWidth: 48, AxiDataWidth: 64, NoAddrRules: 2
  };
  xbar_rule_48_t [1:0] QuadAxiLiteXbarAddrmap;
  assign QuadAxiLiteXbarAddrmap = '{
    '{ idx: 1, start_addr: {chip_id_i,40'h0a000000}, end_addr: {chip_id_i,40'h0a001000} },
    '{ idx: 2, start_addr: {chip_id_i,40'h0a001000}, end_addr: {chip_id_i,40'h0a002000} }
  };
  axi_lite_a48_d64_req_t [4:0] qlite_in_req;
  axi_lite_a48_d64_rsp_t [4:0] qlite_in_rsp;
  axi_lite_a48_d64_req_t [2:0] qlite_out_req;
  axi_lite_a48_d64_rsp_t [2:0] qlite_out_rsp;
  axi_lite_xbar #(
    .Cfg       (QuadAxiLiteXbarCfg),
    .aw_chan_t (axi_lite_a48_d64_aw_chan_t),
    .w_chan_t  (axi_lite_a48_d64_w_chan_t),
    .b_chan_t  (axi_lite_a48_d64_b_chan_t),
    .ar_chan_t (axi_lite_a48_d64_ar_chan_t),
    .r_chan_t  (axi_lite_a48_d64_r_chan_t),
    .axi_req_t (axi_lite_a48_d64_req_t),
    .axi_resp_t(axi_lite_a48_d64_rsp_t),
    .rule_t    (xbar_rule_48_t)
  ) i_quad_axi_lite_xbar (
    .clk_i(clk_fab_i), .rst_ni(rst_ni), .test_i(1'b0),
    .slv_ports_req_i(qlite_in_req), .slv_ports_resp_o(qlite_in_rsp),
    .mst_ports_req_o(qlite_out_req), .mst_ports_resp_i(qlite_out_rsp),
    .addr_map_i(QuadAxiLiteXbarAddrmap), .en_default_mst_port_i('1), .default_mst_port_i('0)
  );
  assign qlite_in_req[1] = '0;
  assign qlite_in_req[2] = '0;
  assign qlite_in_req[3] = sched_tx_req_i;
  assign sched_tx_rsp_o  = qlite_in_rsp[3];
  assign qlite_in_req[4] = '0;
  assign sched_rx_req_o   = qlite_out_req[1];
  assign qlite_out_rsp[1] = sched_rx_rsp_i;
  assign qlite_out_rsp[2] = '0;
  a_never_qlite_host : assert property (@(posedge clk_fab_i) disable iff (!rst_ni)
      !(qlite_out_req[2].aw_valid || qlite_out_req[2].ar_valid))
    else $fatal(1, "[reduce_compute_node %02x] request on an unused port (%m)", chip_id_i);

  // out QUAD_TO_SOC -> quad_to_soc xbar slave QUAD_AXI_LITE (i_quad_axi_lite_to_quad_to_soc_xbar_pc)
  axi_lite_to_axi #(
    .AxiDataWidth(64),
    .req_lite_t  (axi_lite_a48_d64_req_t),
    .resp_lite_t (axi_lite_a48_d64_rsp_t),
    .axi_req_t   (axi_a48_d64_i3_u3_req_t),
    .axi_resp_t  (axi_a48_d64_i3_u3_resp_t)
  ) i_quad_axi_lite_to_quad_to_soc_xbar_pc (
    .slv_req_lite_i (qlite_out_req[0]),
    .slv_resp_lite_o(qlite_out_rsp[0]),
    .slv_aw_cache_i (axi_pkg::CACHE_MODIFIABLE),
    .slv_ar_cache_i (axi_pkg::CACHE_MODIFIABLE),
    .mst_req_o      (q2s_in_req[1]),
    .mst_resp_i     (q2s_in_rsp[1])
  );

  // soc_to_quad xbar master QUAD_AXI_LITE -> in SOC_TO_QUAD (i_soc_to_quad_axi_lite_noatop +
  // i_soc_to_quad_to_axi_lite_pc)
  axi_a48_d64_i7_u3_req_t  s2q_lite_noatop_req;
  axi_a48_d64_i7_u3_resp_t s2q_lite_noatop_rsp;
  axi_atop_filter #(
    .AxiIdWidth     (7),
    .AxiMaxWriteTxns(4),
    .axi_req_t      (axi_a48_d64_i7_u3_req_t),
    .axi_resp_t     (axi_a48_d64_i7_u3_resp_t)
  ) i_soc_to_quad_axi_lite_noatop (
    .clk_i(clk_fab_i), .rst_ni(rst_ni),
    .slv_req_i(s2q_out_req[1]), .slv_resp_o(s2q_out_rsp[1]),
    .mst_req_o(s2q_lite_noatop_req), .mst_resp_i(s2q_lite_noatop_rsp)
  );
  axi_to_axi_lite #(
    .AxiAddrWidth   (48),
    .AxiDataWidth   (64),
    .AxiIdWidth     (7),
    .AxiUserWidth   (3),
    .AxiMaxWriteTxns(4),
    .AxiMaxReadTxns (4),
    .FallThrough    (0),
    .full_req_t     (axi_a48_d64_i7_u3_req_t),
    .full_resp_t    (axi_a48_d64_i7_u3_resp_t),
    .lite_req_t     (axi_lite_a48_d64_req_t),
    .lite_resp_t    (axi_lite_a48_d64_rsp_t)
  ) i_soc_to_quad_to_axi_lite_pc (
    .clk_i(clk_fab_i), .rst_ni(rst_ni), .test_i(1'b0),
    .slv_req_i(s2q_lite_noatop_req), .slv_resp_o(s2q_lite_noatop_rsp),
    .mst_req_o(qlite_in_req[0]), .mst_resp_i(qlite_in_rsp[0])
  );

  // ==========================================================================================
  // SoC (clk_fab): occamy_soc.sv
  // ==========================================================================================
  localparam int unsigned SW_MULTICHIP = 0, SW_SOC_NARROW = 1, SW_SYS_IDMA = 2, SW_HEMAIA_MEM = 3,
                          SW_QUAD = 4;                                   // wide xbar inputs
  localparam int unsigned SWO_MULTICHIP = 0, SWO_SOC_NARROW = 1, SWO_HEMAIA_MEM = 2,
                          SWO_ZERO_MEM = 3, SWO_QUAD = 4;                // wide xbar outputs
  localparam int unsigned SN_SOC_WIDE = 0, SN_QUAD = 1, SN_CVA6 = 2, SN_PERIPH = 3,
                          SN_HEMAIA_MEM = 4;                             // narrow xbar inputs
  localparam int unsigned SNO_SOC_WIDE = 0, SNO_QUAD = 1;                // narrow xbar outputs 0..6

  axi_a48_d512_i4_u1_req_t  [4:0] soc_wide_xbar_in_req;
  axi_a48_d512_i4_u1_resp_t [4:0] soc_wide_xbar_in_rsp;
  axi_a48_d512_i7_u1_req_t  [4:0] soc_wide_xbar_out_req;
  axi_a48_d512_i7_u1_resp_t [4:0] soc_wide_xbar_out_rsp;
  axi_a48_d64_i4_u3_req_t   [4:0] soc_narrow_xbar_in_req;
  axi_a48_d64_i4_u3_resp_t  [4:0] soc_narrow_xbar_in_rsp;
  axi_a48_d64_i7_u3_req_t   [6:0] soc_narrow_xbar_out_req;
  axi_a48_d64_i7_u3_resp_t  [6:0] soc_narrow_xbar_out_rsp;

  // ---- soc_wide_xbar ----
  localparam axi_pkg::xbar_cfg_t SocWideXbarCfg = '{
    NoSlvPorts: 5, NoMstPorts: 5, MaxSlvTrans: 64, MaxMstTrans: 64, FallThrough: 1'b0,
    LatencyMode: axi_pkg::CUT_ALL_PORTS, PipelineStages: 0, AxiIdWidthSlvPorts: 4,
    AxiIdUsedSlvPorts: 4, UniqueIds: 1'b0, AxiAddrWidth: 48, AxiDataWidth: 512, NoAddrRules: 14
  };
  xbar_rule_48_t [13:0] SocWideXbarAddrmap;
  assign SocWideXbarAddrmap = '{
    '{ idx: 1, start_addr: {chip_id_i,40'h00000000}, end_addr: {chip_id_i,40'h10000000 }},
    '{ idx: 1, start_addr: {chip_id_i,40'h10100000}, end_addr: {chip_id_i,40'h10110000 }},
    '{ idx: 1, start_addr: {chip_id_i,40'h103fd000}, end_addr: {chip_id_i,40'h10400000 }},
    '{ idx: 1, start_addr: {chip_id_i,40'h10500000}, end_addr: {chip_id_i,40'h10510000 }},
    '{ idx: 1, start_addr: {chip_id_i,40'h107fd000}, end_addr: {chip_id_i,40'h70020000 }},
    '{ idx: 1, start_addr: {chip_id_i,40'hffffb000}, end_addr: {chip_id_i,40'hffffc000 }},
    '{ idx: 1, start_addr: {chip_id_i,40'hffffd000}, end_addr: {chip_id_i,40'h100000000 }},
    '{ idx: 2, start_addr: {chip_id_i,40'h80000000}, end_addr: {chip_id_i,40'h80800000 }},
    '{ idx: 2, start_addr: {chip_id_i,40'hffffc000}, end_addr: {chip_id_i,40'hffffd000 }},
    '{ idx: 3, start_addr: {chip_id_i,40'h1000000000}, end_addr: {chip_id_i,40'h1200000000 }},
    '{ idx: 4, start_addr: {chip_id_i,40'h10000000}, end_addr: {chip_id_i,40'h10100000 }},
    '{ idx: 4, start_addr: {chip_id_i,40'h10120000}, end_addr: {chip_id_i,40'h103fd000 }},
    '{ idx: 4, start_addr: {chip_id_i,40'h10400000}, end_addr: {chip_id_i,40'h10500000 }},
    '{ idx: 4, start_addr: {chip_id_i,40'h10520000}, end_addr: {chip_id_i,40'h107fd000 }}
  };
  axi_xbar #(
    .Cfg          (SocWideXbarCfg),
    .Connectivity (25'b0111111011111111110111110),
    .ATOPs        (1'b0),
    .slv_aw_chan_t(axi_a48_d512_i4_u1_aw_chan_t),
    .mst_aw_chan_t(axi_a48_d512_i7_u1_aw_chan_t),
    .w_chan_t     (axi_a48_d512_i4_u1_w_chan_t),
    .slv_b_chan_t (axi_a48_d512_i4_u1_b_chan_t),
    .mst_b_chan_t (axi_a48_d512_i7_u1_b_chan_t),
    .slv_ar_chan_t(axi_a48_d512_i4_u1_ar_chan_t),
    .mst_ar_chan_t(axi_a48_d512_i7_u1_ar_chan_t),
    .slv_r_chan_t (axi_a48_d512_i4_u1_r_chan_t),
    .mst_r_chan_t (axi_a48_d512_i7_u1_r_chan_t),
    .slv_req_t    (axi_a48_d512_i4_u1_req_t),
    .slv_resp_t   (axi_a48_d512_i4_u1_resp_t),
    .mst_req_t    (axi_a48_d512_i7_u1_req_t),
    .mst_resp_t   (axi_a48_d512_i7_u1_resp_t),
    .rule_t       (xbar_rule_48_t)
  ) i_soc_wide_xbar (
    .clk_i(clk_fab_i), .rst_ni(rst_ni), .test_i(1'b0),
    .slv_ports_req_i(soc_wide_xbar_in_req), .slv_ports_resp_o(soc_wide_xbar_in_rsp),
    .mst_ports_req_o(soc_wide_xbar_out_req), .mst_ports_resp_i(soc_wide_xbar_out_rsp),
    .addr_map_i(SocWideXbarAddrmap), .en_default_mst_port_i('1), .default_mst_port_i('0)
  );
  assign soc_wide_xbar_in_req[SW_SYS_IDMA]   = '0;     // sys iDMA (absent)
  assign soc_wide_xbar_in_req[SW_HEMAIA_MEM] = '0;     // L3 memory system (absent)
  assign soc_wide_xbar_out_rsp[SWO_HEMAIA_MEM] = '0;
  assign soc_wide_xbar_out_rsp[SWO_ZERO_MEM]   = '0;
  `RD_NEVER(a_never_sw_mem, clk_fab_i, soc_wide_xbar_out_req[SWO_HEMAIA_MEM])
  `RD_NEVER(a_never_sw_zero, clk_fab_i, soc_wide_xbar_out_req[SWO_ZERO_MEM])

  // ---- soc_narrow_xbar ----
  localparam axi_pkg::xbar_cfg_t SocNarrowXbarCfg = '{
    NoSlvPorts: 5, NoMstPorts: 7, MaxSlvTrans: 32, MaxMstTrans: 32, FallThrough: 1'b0,
    LatencyMode: axi_pkg::CUT_ALL_PORTS, PipelineStages: 0, AxiIdWidthSlvPorts: 4,
    AxiIdUsedSlvPorts: 4, UniqueIds: 1'b0, AxiAddrWidth: 48, AxiDataWidth: 64, NoAddrRules: 12
  };
  xbar_rule_48_t [11:0] SocNarrowXbarAddrmap;
  assign SocNarrowXbarAddrmap = '{
    '{ idx: 0, start_addr: {chip_id_i,40'h80000000}, end_addr: {chip_id_i,40'h80800000 }},
    '{ idx: 0, start_addr: {chip_id_i,40'hffffc000}, end_addr: {chip_id_i,40'hffffd000 }},
    '{ idx: 0, start_addr: {chip_id_i,40'h1000000000}, end_addr: {chip_id_i,40'h1200000000 }},
    '{ idx: 1, start_addr: {chip_id_i,40'h0a000000}, end_addr: {chip_id_i,40'h0b003000 }},
    '{ idx: 1, start_addr: {chip_id_i,40'h10000000}, end_addr: {chip_id_i,40'h10800000 }},
    '{ idx: 2, start_addr: {chip_id_i,40'h00000000}, end_addr: {chip_id_i,40'h00002000 }},
    '{ idx: 3, start_addr: {chip_id_i,40'h70000000}, end_addr: {chip_id_i,40'h70020000 }},
    '{ idx: 4, start_addr: {chip_id_i,40'h05000000}, end_addr: {chip_id_i,40'h05010000 }},
    '{ idx: 5, start_addr: {chip_id_i,40'h01000000}, end_addr: {chip_id_i,40'h04100000 }},
    '{ idx: 5, start_addr: {chip_id_i,40'h0c000000}, end_addr: {chip_id_i,40'h10000000 }},
    '{ idx: 6, start_addr: {chip_id_i,40'hffffd000}, end_addr: {chip_id_i,40'h100000000 }},
    '{ idx: 6, start_addr: {chip_id_i,40'hffffb000}, end_addr: {chip_id_i,40'hffffc000 }}
  };
  axi_xbar #(
    .Cfg          (SocNarrowXbarCfg),
    .Connectivity (35'b01111111111011111111111111011111110),
    .ATOPs        (1'b1),
    .slv_aw_chan_t(axi_a48_d64_i4_u3_aw_chan_t),
    .mst_aw_chan_t(axi_a48_d64_i7_u3_aw_chan_t),
    .w_chan_t     (axi_a48_d64_i4_u3_w_chan_t),
    .slv_b_chan_t (axi_a48_d64_i4_u3_b_chan_t),
    .mst_b_chan_t (axi_a48_d64_i7_u3_b_chan_t),
    .slv_ar_chan_t(axi_a48_d64_i4_u3_ar_chan_t),
    .mst_ar_chan_t(axi_a48_d64_i7_u3_ar_chan_t),
    .slv_r_chan_t (axi_a48_d64_i4_u3_r_chan_t),
    .mst_r_chan_t (axi_a48_d64_i7_u3_r_chan_t),
    .slv_req_t    (axi_a48_d64_i4_u3_req_t),
    .slv_resp_t   (axi_a48_d64_i4_u3_resp_t),
    .mst_req_t    (axi_a48_d64_i7_u3_req_t),
    .mst_resp_t   (axi_a48_d64_i7_u3_resp_t),
    .rule_t       (xbar_rule_48_t)
  ) i_soc_narrow_xbar (
    .clk_i(clk_fab_i), .rst_ni(rst_ni), .test_i(1'b0),
    .slv_ports_req_i(soc_narrow_xbar_in_req), .slv_ports_resp_o(soc_narrow_xbar_in_rsp),
    .mst_ports_req_o(soc_narrow_xbar_out_req), .mst_ports_resp_i(soc_narrow_xbar_out_rsp),
    .addr_map_i(SocNarrowXbarAddrmap), .en_default_mst_port_i('1), .default_mst_port_i('0)
  );
  assign soc_narrow_xbar_in_req[SN_CVA6]       = '0;   // host (absent)
  assign soc_narrow_xbar_in_req[SN_PERIPH]     = '0;   // peripherals (absent)
  assign soc_narrow_xbar_in_req[SN_HEMAIA_MEM] = '0;   // memory system (absent)
  for (genvar k = 2; k < 7; k++) begin : gen_soc_narrow_unused
    assign soc_narrow_xbar_out_rsp[k] = '0;            // periph, SPMs, sys-iDMA cfg, ...
    `RD_NEVER(a_never_sn_out, clk_fab_i, soc_narrow_xbar_out_req[k])
  end

  // ---- Quad <-> SoC cuts (cfg cuts: narrow_to_quad, quad_to_narrow, wide_to_quad,
  //      quad_to_wide = 3) ----
  `RD_MULTICUT(i_narrow_in, 3, axi_a48_d64_i7_u3, clk_fab_i,
               soc_narrow_xbar_out_req[SNO_QUAD], soc_narrow_xbar_out_rsp[SNO_QUAD],
               quadrant_narrow_in_req, quadrant_narrow_in_rsp)
  `RD_MULTICUT(i_narrow_out_cut, 3, axi_a48_d64_i4_u3, clk_fab_i,
               quadrant_narrow_out_req, quadrant_narrow_out_rsp,
               soc_narrow_xbar_in_req[SN_QUAD], soc_narrow_xbar_in_rsp[SN_QUAD])
  `RD_MULTICUT(i_wide_in, 3, axi_a48_d512_i7_u1, clk_fab_i,
               soc_wide_xbar_out_req[SWO_QUAD], soc_wide_xbar_out_rsp[SWO_QUAD],
               quadrant_wide_in_req, quadrant_wide_in_rsp)
  `RD_MULTICUT(i_wide_out_cut, 3, axi_a48_d512_i4_u1, clk_fab_i,
               quadrant_wide_out_req, quadrant_wide_out_rsp,
               soc_wide_xbar_in_req[SW_QUAD], soc_wide_xbar_in_rsp[SW_QUAD])

  // ---- Narrow -> wide: amo adapter + cut + id remap + user conversion + 64 -> 512 dw ----
  axi_a48_d64_i7_u3_req_t  soc_narrow_wide_amo_adapter_req, soc_narrow_wide_amo_adapter_cut_req;
  axi_a48_d64_i7_u3_resp_t soc_narrow_wide_amo_adapter_rsp, soc_narrow_wide_amo_adapter_cut_rsp;
  axi_riscv_atomics_structs #(
    .AxiAddrWidth   (48),
    .AxiDataWidth   (64),
    .AxiIdWidth     (7),
    .AxiUserWidth   (3),
    .AxiMaxReadTxns (8),
    .AxiMaxWriteTxns(8),
    .AxiUserAsId    (1),
    .AxiUserIdMsb   (2),
    .AxiUserIdLsb   (0),
    .RiscvWordWidth (64),
    .NAxiCuts       (1),     // cfg cuts.atomic_adapter_narrow_wide
    .axi_req_t      (axi_a48_d64_i7_u3_req_t),
    .axi_rsp_t      (axi_a48_d64_i7_u3_resp_t)
  ) i_soc_narrow_wide_amo_adapter (
    .clk_i        (clk_fab_i),
    .rst_ni       (rst_ni),
    .axi_slv_req_i(soc_narrow_xbar_out_req[SNO_SOC_WIDE]),
    .axi_slv_rsp_o(soc_narrow_xbar_out_rsp[SNO_SOC_WIDE]),
    .axi_mst_req_o(soc_narrow_wide_amo_adapter_req),
    .axi_mst_rsp_i(soc_narrow_wide_amo_adapter_rsp)
  );
  `RD_MULTICUT(i_soc_narrow_wide_amo_adapter_cut, 1, axi_a48_d64_i7_u3, clk_fab_i,
               soc_narrow_wide_amo_adapter_req, soc_narrow_wide_amo_adapter_rsp,
               soc_narrow_wide_amo_adapter_cut_req, soc_narrow_wide_amo_adapter_cut_rsp)
  axi_a48_d64_i4_u3_req_t  soc_narrow_wide_iwc_req;
  axi_a48_d64_i4_u3_resp_t soc_narrow_wide_iwc_rsp;
  axi_id_remap #(
    .AxiSlvPortIdWidth(7), .AxiSlvPortMaxUniqIds(16), .AxiMaxTxnsPerId(16), .AxiMstPortIdWidth(4),
    .slv_req_t(axi_a48_d64_i7_u3_req_t), .slv_resp_t(axi_a48_d64_i7_u3_resp_t),
    .mst_req_t(axi_a48_d64_i4_u3_req_t), .mst_resp_t(axi_a48_d64_i4_u3_resp_t)
  ) i_soc_narrow_wide_iwc (
    .clk_i(clk_fab_i), .rst_ni(rst_ni),
    .slv_req_i(soc_narrow_wide_amo_adapter_cut_req),
    .slv_resp_o(soc_narrow_wide_amo_adapter_cut_rsp),
    .mst_req_o(soc_narrow_wide_iwc_req), .mst_resp_i(soc_narrow_wide_iwc_rsp)
  );
  axi_a48_d64_i4_u1_req_t  soc_narrow_wide_uwc_req;
  axi_a48_d64_i4_u1_resp_t soc_narrow_wide_uwc_rsp;
  `AXI_ASSIGN_REQ_STRUCT(soc_narrow_wide_uwc_req, soc_narrow_wide_iwc_req)
  `AXI_ASSIGN_RESP_STRUCT(soc_narrow_wide_iwc_rsp, soc_narrow_wide_uwc_rsp)
  axi_dw_converter #(
    .AxiMaxReads        (8),
    .AxiSlvPortDataWidth(64),
    .AxiMstPortDataWidth(512),
    .AxiAddrWidth       (48),
    .AxiIdWidth         (4),
    .aw_chan_t          (axi_a48_d512_i4_u1_aw_chan_t),
    .mst_w_chan_t       (axi_a48_d512_i4_u1_w_chan_t),
    .slv_w_chan_t       (axi_a48_d64_i4_u1_w_chan_t),
    .b_chan_t           (axi_a48_d512_i4_u1_b_chan_t),
    .ar_chan_t          (axi_a48_d512_i4_u1_ar_chan_t),
    .mst_r_chan_t       (axi_a48_d512_i4_u1_r_chan_t),
    .slv_r_chan_t       (axi_a48_d64_i4_u1_r_chan_t),
    .axi_mst_req_t      (axi_a48_d512_i4_u1_req_t),
    .axi_mst_resp_t     (axi_a48_d512_i4_u1_resp_t),
    .axi_slv_req_t      (axi_a48_d64_i4_u1_req_t),
    .axi_slv_resp_t     (axi_a48_d64_i4_u1_resp_t)
  ) i_soc_narrow_wide_dw (
    .clk_i(clk_fab_i), .rst_ni(rst_ni),
    .slv_req_i(soc_narrow_wide_uwc_req), .slv_resp_o(soc_narrow_wide_uwc_rsp),
    .mst_req_o(soc_wide_xbar_in_req[SW_SOC_NARROW]),
    .mst_resp_i(soc_wide_xbar_in_rsp[SW_SOC_NARROW])
  );

  // ---- Wide -> narrow: id remap + user conversion + 512 -> 64 dw + cut ----
  axi_a48_d512_i4_u1_req_t  soc_wide_narrow_iwc_req;
  axi_a48_d512_i4_u1_resp_t soc_wide_narrow_iwc_rsp;
  axi_id_remap #(
    .AxiSlvPortIdWidth(7), .AxiSlvPortMaxUniqIds(16), .AxiMaxTxnsPerId(16), .AxiMstPortIdWidth(4),
    .slv_req_t(axi_a48_d512_i7_u1_req_t), .slv_resp_t(axi_a48_d512_i7_u1_resp_t),
    .mst_req_t(axi_a48_d512_i4_u1_req_t), .mst_resp_t(axi_a48_d512_i4_u1_resp_t)
  ) i_soc_wide_narrow_iwc (
    .clk_i(clk_fab_i), .rst_ni(rst_ni),
    .slv_req_i(soc_wide_xbar_out_req[SWO_SOC_NARROW]),
    .slv_resp_o(soc_wide_xbar_out_rsp[SWO_SOC_NARROW]),
    .mst_req_o(soc_wide_narrow_iwc_req), .mst_resp_i(soc_wide_narrow_iwc_rsp)
  );
  axi_a48_d512_i4_u3_req_t  soc_wide_narrow_uwc_req;
  axi_a48_d512_i4_u3_resp_t soc_wide_narrow_uwc_rsp;
  `AXI_ASSIGN_REQ_STRUCT(soc_wide_narrow_uwc_req, soc_wide_narrow_iwc_req)
  `AXI_ASSIGN_RESP_STRUCT(soc_wide_narrow_iwc_rsp, soc_wide_narrow_uwc_rsp)
  axi_a48_d64_i4_u3_req_t  soc_wide_narrow_dw_req;
  axi_a48_d64_i4_u3_resp_t soc_wide_narrow_dw_rsp;
  axi_dw_converter #(
    .AxiMaxReads        (8),
    .AxiSlvPortDataWidth(512),
    .AxiMstPortDataWidth(64),
    .AxiAddrWidth       (48),
    .AxiIdWidth         (4),
    .aw_chan_t          (axi_a48_d64_i4_u3_aw_chan_t),
    .mst_w_chan_t       (axi_a48_d64_i4_u3_w_chan_t),
    .slv_w_chan_t       (axi_a48_d512_i4_u3_w_chan_t),
    .b_chan_t           (axi_a48_d64_i4_u3_b_chan_t),
    .ar_chan_t          (axi_a48_d64_i4_u3_ar_chan_t),
    .mst_r_chan_t       (axi_a48_d64_i4_u3_r_chan_t),
    .slv_r_chan_t       (axi_a48_d512_i4_u3_r_chan_t),
    .axi_mst_req_t      (axi_a48_d64_i4_u3_req_t),
    .axi_mst_resp_t     (axi_a48_d64_i4_u3_resp_t),
    .axi_slv_req_t      (axi_a48_d512_i4_u3_req_t),
    .axi_slv_resp_t     (axi_a48_d512_i4_u3_resp_t)
  ) i_soc_wide_narrow_dw (
    .clk_i(clk_fab_i), .rst_ni(rst_ni),
    .slv_req_i(soc_wide_narrow_uwc_req), .slv_resp_o(soc_wide_narrow_uwc_rsp),
    .mst_req_o(soc_wide_narrow_dw_req), .mst_resp_i(soc_wide_narrow_dw_rsp)
  );
  `RD_MULTICUT(i_soc_wide_narrow_dw_cut, 1, axi_a48_d64_i4_u3, clk_fab_i,
               soc_wide_narrow_dw_req, soc_wide_narrow_dw_rsp,
               soc_narrow_xbar_in_req[SN_SOC_WIDE], soc_narrow_xbar_in_rsp[SN_SOC_WIDE])

  // ---- SoC <-> router: i_soc2router_iwc + i_soc2router_cut, i_router2soc_cut ----
  axi_a48_d512_i4_u1_req_t  soc2router_iwc_req;
  axi_a48_d512_i4_u1_resp_t soc2router_iwc_rsp;
  axi_id_remap #(
    .AxiSlvPortIdWidth(7), .AxiSlvPortMaxUniqIds(16), .AxiMaxTxnsPerId(16), .AxiMstPortIdWidth(4),
    .slv_req_t(axi_a48_d512_i7_u1_req_t), .slv_resp_t(axi_a48_d512_i7_u1_resp_t),
    .mst_req_t(axi_a48_d512_i4_u1_req_t), .mst_resp_t(axi_a48_d512_i4_u1_resp_t)
  ) i_soc2router_iwc (
    .clk_i(clk_fab_i), .rst_ni(rst_ni),
    .slv_req_i(soc_wide_xbar_out_req[SWO_MULTICHIP]),
    .slv_resp_o(soc_wide_xbar_out_rsp[SWO_MULTICHIP]),
    .mst_req_o(soc2router_iwc_req), .mst_resp_i(soc2router_iwc_rsp)
  );
  `RD_MULTICUT(i_soc2router_cut, 1, axi_a48_d512_i4_u1, clk_fab_i,
               soc2router_iwc_req, soc2router_iwc_rsp, soc2router_req_o, soc2router_rsp_i)
  `RD_MULTICUT(i_router2soc_cut, 1, axi_a48_d512_i4_u1, clk_fab_i,
               router2soc_req_i, router2soc_rsp_o,
               soc_wide_xbar_in_req[SW_MULTICHIP], soc_wide_xbar_in_rsp[SW_MULTICHIP])

`undef RD_MULTICUT
`undef RD_NEVER

endmodule
