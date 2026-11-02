// Copyright 2026 KU Leuven.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51
//
// reduce_pkg: the types and constants of the endpoint chain-reduce testbench (tb_reduce).
//
// The AXI types are the SoC's own (occamy_pkg's AXI_TYPEDEF_ALL_CT names, generated for
// hemaia_16c4m_8MBL3_2cluster), declared here so the testbench does not have to compile
// occamy_pkg and everything it imports. The TCDM and cluster AXI types mirror
// snax_split_cluster_pkg (the HeMAiA cluster that owns the xDMA). The xbar configurations are
// copied from occamy_pkg / snitch_cluster.sv; every value carries its origin.

`include "axi/typedef.svh"
`include "tcdm_interface/typedef.svh"

package reduce_pkg;

  // ------------------------------------------------------------------------------------------
  // Platform: hemaia_16c4m_8MBL3_2cluster, 5 x 4 package, chip id = x << 4 | y
  // ------------------------------------------------------------------------------------------
  localparam int unsigned GridX = 5;
  localparam int unsigned GridY = 4;
  localparam int unsigned NumNodes = GridX * GridY;   // 20: 16 compute + 4 memory (x = 2)
  localparam int unsigned MemColumn = 2;
  localparam int unsigned NumCompute = 16;

  typedef logic [7:0] chip_id_t;
  typedef logic [47:0] addr_t;

  function automatic bit is_compute(input int unsigned x);
    return x != MemColumn;
  endfunction

  // Compute index 0..15 of compute chip (x, y): column-major over the compute columns.
  function automatic int unsigned cidx_of_xy(input int unsigned x, input int unsigned y);
    return ((x < MemColumn) ? x : x - 1) * GridY + y;
  endfunction

  function automatic int unsigned cidx_of_chip(input chip_id_t c);
    return cidx_of_xy(c[7:4], c[3:0]);
  endfunction

  function automatic chip_id_t chip_of_cidx(input int unsigned ci);
    int unsigned col, x;
    col = ci / GridY;
    x   = (col < MemColumn) ? col : col + 1;
    return chip_id_t'((x << 4) | (ci % GridY));
  endfunction

  // The snake (data order of the chain; P = 2, 4, 8, 16 are its prefixes). README.
  localparam chip_id_t Snake[NumCompute] = '{
      8'h00, 8'h01, 8'h02, 8'h03,
      8'h13, 8'h12, 8'h11, 8'h10,
      8'h30, 8'h31, 8'h32, 8'h33,
      8'h43, 8'h42, 8'h41, 8'h40
  };

  // ------------------------------------------------------------------------------------------
  // Clocks (cfg hemaia_multichip.sim_clock: master 5 GHz, core /5, cluster /10, PHY /1)
  // ------------------------------------------------------------------------------------------
  localparam realtime PhyHalf     = 0.1ns;   // 5 GHz
  localparam realtime FabricHalf  = 0.5ns;   // 1 GHz: SoC fabric, link digital side
  localparam realtime ClusterHalf = 1.0ns;   // 500 MHz: xDMA endpoint, TCDM
  localparam realtime ClusterPeriod = 2.0ns;

  // ------------------------------------------------------------------------------------------
  // Address map (occamy_pkg / snax_split_cluster cfg)
  // ------------------------------------------------------------------------------------------
  localparam logic [39:0] ClusterBaseOffset   = 40'h00_1000_0000;
  localparam addr_t       ClusterAddressSpace = 48'h40_0000;    // 4 MiB per cluster
  localparam int unsigned TcdmBytes           = 1024 * 1024;    // 1 MiB (tcdm.size 1024)
  localparam int unsigned ClusterPeriphBytes  = 64 * 1024;      // cluster_periph_size 64
  localparam int unsigned MMIOKiB             = 16;             // xDMA MMIO window

  function automatic addr_t cluster_base(input chip_id_t c);
    return {c, ClusterBaseOffset};
  endfunction

  // ------------------------------------------------------------------------------------------
  // SoC AXI types (occamy_pkg)
  // ------------------------------------------------------------------------------------------
  // 512-bit wide network
  `AXI_TYPEDEF_ALL_CT(axi_a48_d512_i1_u1, axi_a48_d512_i1_u1_req_t, axi_a48_d512_i1_u1_resp_t,
                      logic [47:0], logic [0:0], logic [511:0], logic [63:0], logic [0:0])
  `AXI_TYPEDEF_ALL_CT(axi_a48_d512_i3_u1, axi_a48_d512_i3_u1_req_t, axi_a48_d512_i3_u1_resp_t,
                      logic [47:0], logic [2:0], logic [511:0], logic [63:0], logic [0:0])
  `AXI_TYPEDEF_ALL_CT(axi_a48_d512_i4_u1, axi_a48_d512_i4_u1_req_t, axi_a48_d512_i4_u1_resp_t,
                      logic [47:0], logic [3:0], logic [511:0], logic [63:0], logic [0:0])
  `AXI_TYPEDEF_ALL_CT(axi_a48_d512_i5_u1, axi_a48_d512_i5_u1_req_t, axi_a48_d512_i5_u1_resp_t,
                      logic [47:0], logic [4:0], logic [511:0], logic [63:0], logic [0:0])
  `AXI_TYPEDEF_ALL_CT(axi_a48_d512_i7_u1, axi_a48_d512_i7_u1_req_t, axi_a48_d512_i7_u1_resp_t,
                      logic [47:0], logic [6:0], logic [511:0], logic [63:0], logic [0:0])
  `AXI_TYPEDEF_ALL_CT(axi_a48_d512_i4_u3, axi_a48_d512_i4_u3_req_t, axi_a48_d512_i4_u3_resp_t,
                      logic [47:0], logic [3:0], logic [511:0], logic [63:0], logic [2:0])
  // 64-bit narrow network
  `AXI_TYPEDEF_ALL_CT(axi_a48_d64_i2_u3, axi_a48_d64_i2_u3_req_t, axi_a48_d64_i2_u3_resp_t,
                      logic [47:0], logic [1:0], logic [63:0], logic [7:0], logic [2:0])
  `AXI_TYPEDEF_ALL_CT(axi_a48_d64_i3_u3, axi_a48_d64_i3_u3_req_t, axi_a48_d64_i3_u3_resp_t,
                      logic [47:0], logic [2:0], logic [63:0], logic [7:0], logic [2:0])
  `AXI_TYPEDEF_ALL_CT(axi_a48_d64_i4_u3, axi_a48_d64_i4_u3_req_t, axi_a48_d64_i4_u3_resp_t,
                      logic [47:0], logic [3:0], logic [63:0], logic [7:0], logic [2:0])
  `AXI_TYPEDEF_ALL_CT(axi_a48_d64_i6_u3, axi_a48_d64_i6_u3_req_t, axi_a48_d64_i6_u3_resp_t,
                      logic [47:0], logic [5:0], logic [63:0], logic [7:0], logic [2:0])
  `AXI_TYPEDEF_ALL_CT(axi_a48_d64_i7_u3, axi_a48_d64_i7_u3_req_t, axi_a48_d64_i7_u3_resp_t,
                      logic [47:0], logic [6:0], logic [63:0], logic [7:0], logic [2:0])
  `AXI_TYPEDEF_ALL_CT(axi_a48_d64_i4_u1, axi_a48_d64_i4_u1_req_t, axi_a48_d64_i4_u1_resp_t,
                      logic [47:0], logic [3:0], logic [63:0], logic [7:0], logic [0:0])
  // D2D link control port
  `AXI_LITE_TYPEDEF_ALL_CT(axi_lite_a48_d32, axi_lite_a48_d32_req_t, axi_lite_a48_d32_rsp_t,
                           logic [47:0], logic [31:0], logic [3:0])
  // Quadrant AXI-Lite (the BINGO manager's host side)
  `AXI_LITE_TYPEDEF_ALL_CT(axi_lite_a48_d64, axi_lite_a48_d64_req_t, axi_lite_a48_d64_rsp_t,
                           logic [47:0], logic [63:0], logic [7:0])

  // Snitch accelerator (offload) interface, as snax_split_cluster_pkg: the iDMA's frontend
  typedef struct packed {
    logic [31:0] addr;          // snitch_pkg::acc_addr_e (DMA_SS = 2)
    logic [4:0]  id;
    logic [31:0] data_op;       // the instruction
    logic [63:0] data_arga;
    logic [63:0] data_argb;
    logic [47:0] data_argc;
  } acc_req_t;
  typedef struct packed {
    logic [4:0]  id;
    logic        error;
    logic [63:0] data;
  } acc_resp_t;
  // snitch_cluster.sv dma_events_t
  typedef struct packed {
    logic aw_stall, ar_stall, r_stall, w_stall, buf_w_stall, buf_r_stall;
    logic aw_valid, aw_ready, aw_done, aw_bw;
    logic ar_valid, ar_ready, ar_done, ar_bw;
    logic r_valid, r_ready, r_done, r_bw;
    logic w_valid, w_ready, w_done, w_bw;
    logic b_valid, b_ready, b_done;
    logic dma_busy;
    axi_pkg::len_t aw_len, ar_len;
    axi_pkg::size_t aw_size, ar_size;
    logic [6:0] num_bytes_written;    // $clog2(WideDataWidth/8):0
  } dma_events_t;

  // occamy_pkg::xbar_rule_48_t
  typedef struct packed {
    logic [31:0] idx;
    logic [47:0] start_addr;
    logic [47:0] end_addr;
  } xbar_rule_48_t;

  // ------------------------------------------------------------------------------------------
  // Cluster (snax_split_cluster_pkg)
  // ------------------------------------------------------------------------------------------
  // narrow_in = i2 (into the cluster), narrow_out = i4, wide_in = i1, wide_out = i3.
  localparam int unsigned TcdmPorts     = 16;    // 8 reader + 8 writer (xDMA)
  localparam int unsigned TcdmAddrWidth = 20;    // 1 MiB
  localparam int unsigned TcdmWords     = TcdmBytes / 8;
  localparam int unsigned TcdmBanks     = 32;    // tcdm.banks

  typedef logic [TcdmAddrWidth-1:0] tcdm_addr_t;
  typedef logic [63:0]              tcdm_data_t;
  typedef logic [7:0]               tcdm_strb_t;
  typedef struct packed {
    logic [1:0] core_id;        // idx_width(NrCores = 4)
    bit         is_core;
    logic [1:0] tcdm_priority;
  } tcdm_user_t;
  `TCDM_TYPEDEF_ALL(tcdm, tcdm_addr_t, tcdm_data_t, tcdm_strb_t, tcdm_user_t)

  // ------------------------------------------------------------------------------------------
  // xDMA CSR map of snax_split_cluster (target/sw/device/runtime/snax/xdma/snax_xdma_addr.h).
  // Plain 0-based register indices; cross-checked against the generated ReqRspManager bounds
  // ("csr write address overflow ... Max: 71", "csr read ... Max allowed address is 79").
  // ------------------------------------------------------------------------------------------
  localparam int unsigned CSR_SRC_ADDR        = 0;
  localparam int unsigned CSR_DST_ADDR        = 2;    // 16 slots x 2
  localparam int unsigned CSR_SRC_SPAT_STRIDE = 34;
  localparam int unsigned CSR_SRC_TEMP_BOUND  = 35;   // +0..4
  localparam int unsigned CSR_SRC_TEMP_STRIDE = 40;   // +0..4
  localparam int unsigned CSR_SRC_ENAB_CHAN   = 45;
  localparam int unsigned CSR_SRC_ENABLE      = 46;
  localparam int unsigned CSR_SRC_EXT         = 47;   // +0..2
  localparam int unsigned CSR_DST_SPAT_STRIDE = 50;
  localparam int unsigned CSR_DST_TEMP_BOUND  = 51;   // +0..4
  localparam int unsigned CSR_DST_TEMP_STRIDE = 56;   // +0..4
  localparam int unsigned CSR_DST_ENAB_CHAN   = 61;
  localparam int unsigned CSR_DST_ENAB_BYTE   = 62;
  localparam int unsigned CSR_DST_ENABLE      = 63;
  localparam int unsigned CSR_DST_EXT         = 64;
  localparam int unsigned CSR_DST_JCT_ENABLE  = 65;
  localparam int unsigned CSR_DST_JCT_CSR     = 66;   // Elementwise 66; SoftmaxStats 67-68; TopK 69; Bitwise 70
  localparam int unsigned CSR_START           = 71;
  localparam int unsigned CSR_COMMIT_LOCAL    = 72;
  localparam int unsigned CSR_COMMIT_REMOTE   = 73;
  localparam int unsigned CSR_FINISH_LOCAL    = 74;
  localparam int unsigned CSR_FINISH_REMOTE   = 75;
  localparam int unsigned CSR_PERF_TASK       = 76;
  localparam int unsigned CSR_PERF_READER     = 77;
  localparam int unsigned CSR_PERF_WRITER     = 78;
  localparam int unsigned CSR_JCT_STATUS      = 79;
  localparam int unsigned XdmaDstJctNum       = 4;
  localparam int unsigned JctElementwise      = 0;    // WRITER_JCT_ELEMENTWISEJUNCTION
  // ElementwiseJunction CSR(0): [3:0] op ADD = 0, [6:4] fmt FP16 = 0 (snax-coll-reduce.c)
  localparam logic [31:0] JctCsrAddFp16       = 32'h0000_0000;

  // ------------------------------------------------------------------------------------------
  // D2D link registers (hemaia_d2d_link_peripheral.h)
  // ------------------------------------------------------------------------------------------
  localparam logic [31:0] D2D_AVAILABILITY    = 32'h008;
  localparam logic [31:0] D2D_MULTICAST_FENCE = 32'h00c;
  localparam logic [31:0] D2D_PHY_DDR_MODE    = 32'h178;
  localparam logic [31:0] D2D_TX_TURNAROUND   = 32'h17c;
  localparam logic [31:0] D2D_RX_YIELD_WINDOW = 32'h180;

  // Ports that face a chip (HEMAIA_D2D_PORT_TABLE): bit 0 east, 1 west, 2 north, 3 south.
  function automatic logic [3:0] d2d_links_of_xy(input int unsigned x, input int unsigned y);
    return {(y < GridY - 1), (y > 0), (x > 0), (x < GridX - 1)};
  endfunction

endpackage
