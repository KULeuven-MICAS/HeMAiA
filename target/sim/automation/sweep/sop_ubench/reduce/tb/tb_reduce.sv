// Copyright 2026 KU Leuven.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51
//
// tb_reduce: the endpoint reduce on the 5 x 4 package of hemaia_16c4m_8MBL3_2cluster, hardware
// only (see ../README.md).
//
//   - 16 compute nodes (x = 0, 1, 3, 4; y = 0..3): cluster 0's xDMA endpoint
//     (snax_split_cluster_xdma_wrapper), its iDMA (idma_inst64_top), the TCDM's wide DMA port
//     (axi_to_mem_interleaved), and the chip's path from them to the D2D link
//     (reduce_compute_node); a TCDM model and a SIMD model (reduce_simd_model) per node;
//   - 20 hemaia_d2d_link instances, PHYs wired to their neighbours as the testharness's
//     io_wrapper does; the 4 memory nodes (x = 2) are router-only (local ports tied off);
//   - clocks: endpoints + TCDM 500 MHz, fabric and link digital side 1 GHz, PHY 5 GHz;
//   - the link bring-up the legacy host does at boot (hemaia_d2d_link_initialize_grid and
//     hemaia_d2d_link_ddr_on_grid, HEMAIA_D2D_DDR), written through each link's AXI-Lite port;
//   - a model of the BINGO manager per node (its measured internal latencies) behind real
//     CSR clock crossings, sending real dep-set writes over the links (trees);
//   - the arms of snax-coll-reduce.c, replayed: the xDMA through its CSRs, the iDMA through its
//     instruction interface, the SIMD through the model, flags as real stores over the links.
//
// Plusargs
//   +ARM=<arm>        chain | two | binomial | pull | swchain | swtree
//   +BYTES=<n>        payload per chip, FP16 (multiple of 256, <= 256 KiB)
//   +PLIST=<p,p,..>   participant counts, prefixes of the snake (default 2,4,8,16)
//   +VERBOSE=<n>      1: per-chip details of every run
//   +TIMEOUT_US=<n>   simulation-time watchdog (default 20000 us)
//
// Every run prints one machine-readable [REDUCE] line (see run_point's report). Latency
// `lat_prog_cc` runs from the first programming action of the reduction (the first CSR write,
// DMA instruction or store any chip issues) to the result complete at the collector; `lat_cc`
// from the first START write (xDMA arms) or first DMA copy instruction (software arms). Both in
// 500 MHz cluster cycles.

`timescale 1ns / 1ps

module tb_reduce
  import reduce_pkg::*;
#(
  /// 1: the TCDM model arbitrates 32 word-interleaved banks (one access per bank per cycle);
  /// 0: every narrow access is granted at once.
  parameter bit TcdmBanked = 1'b1,
  /// hemaia_d2d_link FramerTxWBufBeats of every link. 64 is the RTL default, which occamy_chip and
  /// hemaia_mem_chip use; 0 (the framer before 2026-10-13) is a diagnostic only.
  parameter int unsigned LinkTxWBufBeats = 64,
  /// Software cost of one hand-off in the software arms, in cluster cycles, paid by the waiting
  /// side after the flag has landed (the archived full-package runs: ~250 per tree level).
  parameter int unsigned SwHandoffCc = 250,
  /// BINGO manager internal latencies, in its 1 GHz clock cycles, measured on its RTL
  /// (sched/tb_sched_latency.sv, sched/sched_latency.log):
  ///   done CSR at the manager -> the dep-set AXI-Lite AW leaving it (remote edge)
  parameter int unsigned SchedTxCc = 2,
  ///   the dep-set write arriving in its mailbox -> the dependent task in the ready queue
  parameter int unsigned SchedRxReadyCc = 3,
  ///   done CSR -> the dependent task in the ready queue (local edge)
  parameter int unsigned SchedLocalReadyCc = 3,
  ///   ready queue -> the waiting core's CSR read answered
  parameter int unsigned SchedReadyRspCc = 1
) ();

  localparam realtime FabricPeriod = 2 * FabricHalf;
  // ==========================================================================================
  // Clocks and reset
  // ==========================================================================================
  logic clk_phy, clk_fab, clk_cl;
  logic rst_n;

  initial begin
    clk_phy = 1'b0;
    forever #(PhyHalf) clk_phy = ~clk_phy;
  end
  initial begin
    clk_fab = 1'b0;
    forever #(FabricHalf) clk_fab = ~clk_fab;
  end
  initial begin
    clk_cl = 1'b0;
    forever #(ClusterHalf) clk_cl = ~clk_cl;
  end
  initial begin
    rst_n = 1'b0;
    #(101ns);
    rst_n = 1'b1;
  end

  // ==========================================================================================
  // Signals
  // ==========================================================================================
  // xDMA CSR ports, one per compute node (index = compute index, see reduce_pkg)
  logic [31:0] csr_addr      [NumCompute];
  logic [31:0] csr_wdata     [NumCompute];
  logic        csr_write     [NumCompute];
  logic        csr_valid     [NumCompute];
  logic        csr_req_ready [NumCompute];
  logic [31:0] csr_rdata     [NumCompute];
  logic        csr_rsp_valid [NumCompute];
  logic        csr_rsp_ready [NumCompute];

  // TCDM: per node 24 narrow ports (0..15 the xDMA, 16..19 SIMD reads, 20..23 SIMD writes) and
  // the wide DMA port
  localparam int unsigned NarrowPorts = TcdmPorts + 8;
  tcdm_req_t [TcdmPorts-1:0]   xdma_tcdm_req [NumCompute];
  tcdm_rsp_t [TcdmPorts-1:0]   xdma_tcdm_rsp [NumCompute];
  tcdm_req_t [7:0]             simd_req      [NumCompute];
  tcdm_rsp_t [7:0]             simd_rsp      [NumCompute];
  tcdm_req_t [NarrowPorts-1:0] nreq          [NumCompute];
  tcdm_rsp_t [NarrowPorts-1:0] nrsp          [NumCompute];
  logic                        wreq          [NumCompute];
  addr_t                       waddr         [NumCompute];
  logic [511:0]                wwdata        [NumCompute];
  logic [63:0]                 wstrb         [NumCompute];
  logic                        wwe           [NumCompute];
  logic                        wrvalid_q     [NumCompute];
  logic [511:0]                wrdata_q      [NumCompute];

  // the cores' stores, the iDMA's instruction interface, the BINGO manager's ports
  axi_a48_d64_i2_u3_req_t  core_req [NumCompute];
  axi_a48_d64_i2_u3_resp_t core_rsp [NumCompute];
  acc_req_t                acc_req       [NumCompute];
  logic                    acc_req_valid [NumCompute];
  logic                    acc_req_ready [NumCompute];
  acc_resp_t               acc_rsp       [NumCompute];
  logic                    acc_rsp_valid [NumCompute];
  logic                    idma_busy     [NumCompute];
  axi_lite_a48_d64_req_t   stx_req [NumCompute];
  axi_lite_a48_d64_rsp_t   stx_rsp [NumCompute];
  axi_lite_a48_d64_req_t   srx_req [NumCompute];
  axi_lite_a48_d64_rsp_t   srx_rsp [NumCompute];

  // D2D link control ports, one per node (index = x * GridY + y)
  axi_lite_a48_d32_req_t lite_req [NumNodes];
  axi_lite_a48_d32_rsp_t lite_rsp [NumNodes];

  // PHY IO and side-band, one per node and side
  wire [NumNodes-1:0][2:0][19:0] east_io, west_io, north_io, south_io;
  wire [NumNodes-1:0] e_rts_o, e_cts_i, e_rts_i, e_cts_o, e_treq_o, e_tbr_i;
  wire [NumNodes-1:0] w_rts_o, w_cts_i, w_rts_i, w_cts_o, w_treq_o, w_tbr_i;
  wire [NumNodes-1:0] n_rts_o, n_cts_i, n_rts_i, n_cts_o, n_treq_o, n_tbr_i;
  wire [NumNodes-1:0] s_rts_o, s_cts_i, s_rts_i, s_cts_o, s_treq_o, s_tbr_i;

  // per-hop probes (written by gen_compute, cleared by ctr_clear)
  realtime     wout_first_t[NumCompute], wout_last_t[NumCompute];
  realtime     win_first_t [NumCompute], win_last_t [NumCompute];
  int unsigned wout_n[NumCompute], win_n[NumCompute];
  int unsigned naw_out[NumCompute], naw_in[NumCompute];
  bit          mon_on;

  // ==========================================================================================
  // The package: 20 nodes, each a hemaia_d2d_link; compute nodes add the chip behind it
  // ==========================================================================================
  for (genvar gx = 0; gx < GridX; gx++) begin : gen_x
    for (genvar gy = 0; gy < GridY; gy++) begin : gen_y
      localparam int unsigned N = gx * GridY + gy;
      localparam chip_id_t ChipId = chip_id_t'((gx << 4) | gy);

      if (gx != MemColumn) begin : gen_compute
        localparam int unsigned CI = cidx_of_xy(gx, gy);

        axi_a48_d512_i4_u1_req_t  soc2router_req, router2soc_req;
        axi_a48_d512_i4_u1_resp_t soc2router_rsp, router2soc_rsp;

        reduce_compute_node i_node (
          .clk_cl_i        (clk_cl),
          .clk_fab_i       (clk_fab),
          .rst_ni          (rst_n),
          .chip_id_i       (ChipId),
          .csr_addr_i      (csr_addr[CI]),
          .csr_write_i     (csr_write[CI]),
          .csr_wdata_i     (csr_wdata[CI]),
          .csr_valid_i     (csr_valid[CI]),
          .csr_ready_o     (csr_req_ready[CI]),
          .csr_rdata_o     (csr_rdata[CI]),
          .csr_rsp_valid_o (csr_rsp_valid[CI]),
          .csr_rsp_ready_i (csr_rsp_ready[CI]),
          .tcdm_req_o      (xdma_tcdm_req[CI]),
          .tcdm_rsp_i      (xdma_tcdm_rsp[CI]),
          .soc2router_req_o(soc2router_req),
          .soc2router_rsp_i(soc2router_rsp),
          .router2soc_req_i(router2soc_req),
          .router2soc_rsp_o(router2soc_rsp),
          .core_req_i          (core_req[CI]),
          .core_rsp_o          (core_rsp[CI]),
          .idma_acc_req_i      (acc_req[CI]),
          .idma_acc_req_valid_i(acc_req_valid[CI]),
          .idma_acc_req_ready_o(acc_req_ready[CI]),
          .idma_acc_rsp_o      (acc_rsp[CI]),
          .idma_acc_rsp_valid_o(acc_rsp_valid[CI]),
          .idma_acc_rsp_ready_i(1'b1),
          .idma_busy_o         (idma_busy[CI]),
          .dma_mem_req_o       (wreq[CI]),
          .dma_mem_gnt_i       (wreq[CI]),        // the wide port wins its superbank unconditionally
          .dma_mem_addr_o      (waddr[CI]),
          .dma_mem_wdata_o     (wwdata[CI]),
          .dma_mem_strb_o      (wstrb[CI]),
          .dma_mem_we_o        (wwe[CI]),
          .dma_mem_rvalid_i    (wrvalid_q[CI]),
          .dma_mem_rdata_i     (wrdata_q[CI]),
          .sched_tx_req_i      (stx_req[CI]),
          .sched_tx_rsp_o      (stx_rsp[CI]),
          .sched_rx_req_o      (srx_req[CI]),
          .sched_rx_rsp_i      (srx_rsp[CI])
        );

        // Per-hop probes (VERBOSE): the xDMA's first and last W beat out and in, and its
        // narrow AW count each way.
        always @(posedge clk_cl) begin
          if (rst_n && mon_on) begin
            if (i_node.xdma_wide_out_req.w_valid && i_node.xdma_wide_out_rsp.w_ready) begin
              if (wout_n[CI] == 0) wout_first_t[CI] <= $realtime;
              wout_last_t[CI] <= $realtime;
              wout_n[CI]      <= wout_n[CI] + 1;
            end
            if (i_node.xdma_wide_in_req.w_valid && i_node.xdma_wide_in_rsp.w_ready) begin
              if (win_n[CI] == 0) win_first_t[CI] <= $realtime;
              win_last_t[CI] <= $realtime;
              win_n[CI]      <= win_n[CI] + 1;
            end
            if (i_node.xdma_narrow_out_req.aw_valid && i_node.xdma_narrow_out_rsp.aw_ready)
              naw_out[CI] <= naw_out[CI] + 1;
            if (i_node.xdma_narrow_in_req.aw_valid && i_node.xdma_narrow_in_rsp.aw_ready)
              naw_in[CI] <= naw_in[CI] + 1;
          end
        end

        // Per-hop probes (VERBOSE): the xDMA's first and last W beat out (wide out, toward the
        // next chip) and in (wide in, from the previous chip), and its narrow AW count each way
        // (cfg, grant, finish messages).
        always @(posedge clk_cl) begin
          if (rst_n && mon_on) begin
            if (i_node.xdma_wide_out_req.w_valid && i_node.xdma_wide_out_rsp.w_ready) begin
              if (wout_n[CI] == 0) wout_first_t[CI] <= $realtime;
              wout_last_t[CI] <= $realtime;
              wout_n[CI]      <= wout_n[CI] + 1;
            end
            if (i_node.xdma_wide_in_req.w_valid && i_node.xdma_wide_in_rsp.w_ready) begin
              if (win_n[CI] == 0) win_first_t[CI] <= $realtime;
              win_last_t[CI] <= $realtime;
              win_n[CI]      <= win_n[CI] + 1;
            end
            if (i_node.xdma_narrow_out_req.aw_valid && i_node.xdma_narrow_out_rsp.aw_ready)
              naw_out[CI] <= naw_out[CI] + 1;
            if (i_node.xdma_narrow_in_req.aw_valid && i_node.xdma_narrow_in_rsp.aw_ready)
              naw_in[CI] <= naw_in[CI] + 1;
          end
        end

        // occamy_chip.sv i_d2d_link: all four PHYs present (EAST_IO .. SOUTH_IO = 1)
        hemaia_d2d_link #(
          .SendFifoDepth (4),
          .RecvFifoDepth (8),
          .FramerTxWBufBeats(LinkTxWBufBeats),
          .chip_id_t     (chip_id_t),
          .axi_req_t     (axi_a48_d512_i4_u1_req_t),
          .axi_rsp_t     (axi_a48_d512_i4_u1_resp_t),
          .axi_lite_req_t(axi_lite_a48_d32_req_t),
          .axi_lite_rsp_t(axi_lite_a48_d32_rsp_t),
          .aw_chan_t     (axi_a48_d512_i4_u1_aw_chan_t),
          .ar_chan_t     (axi_a48_d512_i4_u1_ar_chan_t),
          .r_chan_t      (axi_a48_d512_i4_u1_r_chan_t),
          .w_chan_t      (axi_a48_d512_i4_u1_w_chan_t),
          .b_chan_t      (axi_a48_d512_i4_u1_b_chan_t),
          .EnableEastPhy (1),
          .EnableWestPhy (1),
          .EnableNorthPhy(1),
          .EnableSouthPhy(1)
        ) i_link (
          .chip_id_i                   (ChipId),
          .control_clk_i               (clk_cl),       // clk_periph_i: 500 MHz
          .control_rst_ni              (rst_n),
          .digital_clk_i               (clk_fab),      // clk_host
          .east_phy_tx_clk_i           (clk_phy),
          .west_phy_tx_clk_i           (clk_phy),
          .north_phy_tx_clk_i          (clk_phy),
          .south_phy_tx_clk_i          (clk_phy),
          .rst_ni                      (rst_n),
          .axi_lite_req_i              (lite_req[N]),
          .axi_lite_rsp_o              (lite_rsp[N]),
          .to_remote_req_i             (soc2router_req),
          .to_remote_rsp_o             (soc2router_rsp),
          .from_remote_req_o           (router2soc_req),
          .from_remote_rsp_i           (router2soc_rsp),
          .east_test_being_requested_i (e_tbr_i[N]),
          .east_test_request_o         (e_treq_o[N]),
          .flow_control_east_rts_o     (e_rts_o[N]),
          .flow_control_east_cts_i     (e_cts_i[N]),
          .flow_control_east_rts_i     (e_rts_i[N]),
          .flow_control_east_cts_o     (e_cts_o[N]),
          .east_phy_io                 (east_io[N]),
          .west_test_being_requested_i (w_tbr_i[N]),
          .west_test_request_o         (w_treq_o[N]),
          .flow_control_west_rts_o     (w_rts_o[N]),
          .flow_control_west_cts_i     (w_cts_i[N]),
          .flow_control_west_rts_i     (w_rts_i[N]),
          .flow_control_west_cts_o     (w_cts_o[N]),
          .west_phy_io                 (west_io[N]),
          .north_test_being_requested_i(n_tbr_i[N]),
          .north_test_request_o        (n_treq_o[N]),
          .flow_control_north_rts_o    (n_rts_o[N]),
          .flow_control_north_cts_i    (n_cts_i[N]),
          .flow_control_north_rts_i    (n_rts_i[N]),
          .flow_control_north_cts_o    (n_cts_o[N]),
          .north_phy_io                (north_io[N]),
          .south_test_being_requested_i(s_tbr_i[N]),
          .south_test_request_o        (s_treq_o[N]),
          .flow_control_south_rts_o    (s_rts_o[N]),
          .flow_control_south_cts_i    (s_cts_i[N]),
          .flow_control_south_rts_i    (s_rts_i[N]),
          .flow_control_south_cts_o    (s_cts_o[N]),
          .south_phy_io                (south_io[N])
        );

      end else begin : gen_memory
        // hemaia_mem_chip.sv i_d2d_link: a PHY on every side that faces a chip, NumSysIdma = 4
        // local ports. Router only here: nothing is behind the local ports.
        localparam int unsigned NumLocal = 4;
        axi_a48_d512_i4_u1_req_t  [NumLocal-1:0] mem_to_remote_req, mem_from_remote_req;
        axi_a48_d512_i4_u1_resp_t [NumLocal-1:0] mem_to_remote_rsp, mem_from_remote_rsp;
        assign mem_to_remote_req   = '0;
        assign mem_from_remote_rsp = '0;

        hemaia_d2d_link #(
          .SendFifoDepth (4),
          .RecvFifoDepth (8),
          .NumLocalPorts (NumLocal),
          .FramerTxWBufBeats(LinkTxWBufBeats),
          .chip_id_t     (chip_id_t),
          .axi_req_t     (axi_a48_d512_i4_u1_req_t),
          .axi_rsp_t     (axi_a48_d512_i4_u1_resp_t),
          .axi_lite_req_t(axi_lite_a48_d32_req_t),
          .axi_lite_rsp_t(axi_lite_a48_d32_rsp_t),
          .aw_chan_t     (axi_a48_d512_i4_u1_aw_chan_t),
          .ar_chan_t     (axi_a48_d512_i4_u1_ar_chan_t),
          .r_chan_t      (axi_a48_d512_i4_u1_r_chan_t),
          .w_chan_t      (axi_a48_d512_i4_u1_w_chan_t),
          .b_chan_t      (axi_a48_d512_i4_u1_b_chan_t),
          .EnableEastPhy (gx < GridX - 1),
          .EnableWestPhy (gx > 0),
          .EnableNorthPhy(gy > 0),
          .EnableSouthPhy(gy < GridY - 1)
        ) i_link (
          .chip_id_i                   (ChipId),
          .control_clk_i               (clk_cl),
          .control_rst_ni              (rst_n),
          .digital_clk_i               (clk_fab),      // clk_host (same_memchip_speed)
          .east_phy_tx_clk_i           (clk_phy),
          .west_phy_tx_clk_i           (clk_phy),
          .north_phy_tx_clk_i          (clk_phy),
          .south_phy_tx_clk_i          (clk_phy),
          .rst_ni                      (rst_n),
          .axi_lite_req_i              (lite_req[N]),
          .axi_lite_rsp_o              (lite_rsp[N]),
          .to_remote_req_i             (mem_to_remote_req),
          .to_remote_rsp_o             (mem_to_remote_rsp),
          .from_remote_req_o           (mem_from_remote_req),
          .from_remote_rsp_i           (mem_from_remote_rsp),
          .east_test_being_requested_i (e_tbr_i[N]),
          .east_test_request_o         (e_treq_o[N]),
          .flow_control_east_rts_o     (e_rts_o[N]),
          .flow_control_east_cts_i     (e_cts_i[N]),
          .flow_control_east_rts_i     (e_rts_i[N]),
          .flow_control_east_cts_o     (e_cts_o[N]),
          .east_phy_io                 (east_io[N]),
          .west_test_being_requested_i (w_tbr_i[N]),
          .west_test_request_o         (w_treq_o[N]),
          .flow_control_west_rts_o     (w_rts_o[N]),
          .flow_control_west_cts_i     (w_cts_i[N]),
          .flow_control_west_rts_i     (w_rts_i[N]),
          .flow_control_west_cts_o     (w_cts_o[N]),
          .west_phy_io                 (west_io[N]),
          .north_test_being_requested_i(n_tbr_i[N]),
          .north_test_request_o        (n_treq_o[N]),
          .flow_control_north_rts_o    (n_rts_o[N]),
          .flow_control_north_cts_i    (n_cts_i[N]),
          .flow_control_north_rts_i    (n_rts_i[N]),
          .flow_control_north_cts_o    (n_cts_o[N]),
          .north_phy_io                (north_io[N]),
          .south_test_being_requested_i(s_tbr_i[N]),
          .south_test_request_o        (s_treq_o[N]),
          .flow_control_south_rts_o    (s_rts_o[N]),
          .flow_control_south_cts_i    (s_cts_i[N]),
          .flow_control_south_rts_i    (s_rts_i[N]),
          .flow_control_south_cts_o    (s_cts_o[N]),
          .south_phy_io                (south_io[N])
        );

        for (genvar k = 0; k < NumLocal; k++) begin : gen_never
          a_mem_never_local : assert property (@(posedge clk_fab) disable iff (!rst_n)
              !(mem_from_remote_req[k].aw_valid || mem_from_remote_req[k].ar_valid))
            else $fatal(1, "[tb_reduce] a request was delivered to memory node %02x", ChipId);
        end
      end
    end
  end

  // ---- Links between neighbours (target/sim/testharness/io_wrapper.sv) ----
  for (genvar gx = 0; gx < GridX; gx++) begin : gen_wire_x
    for (genvar gy = 0; gy < GridY; gy++) begin : gen_wire_y
      localparam int unsigned N = gx * GridY + gy;
      if (gx < GridX - 1) begin : gen_east
        localparam int unsigned E = (gx + 1) * GridY + gy;
        assign e_cts_i[N] = w_cts_o[E];
        assign e_rts_i[N] = w_rts_o[E];
        assign e_tbr_i[N] = w_treq_o[E];
        assign w_cts_i[E] = e_cts_o[N];
        assign w_rts_i[E] = e_rts_o[N];
        assign w_tbr_i[E] = e_treq_o[N];
        for (genvar c = 0; c < 3; c++) begin : gen_c
          for (genvar b = 0; b < 20; b++) begin : gen_b
            tran i_bit (east_io[N][c][b], west_io[E][c][b]);
          end
        end
      end else begin : gen_no_east
        assign e_cts_i[N] = 1'b0;
        assign e_rts_i[N] = 1'b0;
        assign e_tbr_i[N] = 1'b0;
      end
      if (gx == 0) begin : gen_no_west
        assign w_cts_i[N] = 1'b0;
        assign w_rts_i[N] = 1'b0;
        assign w_tbr_i[N] = 1'b0;
      end
      if (gy < GridY - 1) begin : gen_south
        localparam int unsigned S = gx * GridY + gy + 1;
        assign s_cts_i[N] = n_cts_o[S];
        assign s_rts_i[N] = n_rts_o[S];
        assign s_tbr_i[N] = n_treq_o[S];
        assign n_cts_i[S] = s_cts_o[N];
        assign n_rts_i[S] = s_rts_o[N];
        assign n_tbr_i[S] = s_treq_o[N];
        for (genvar c = 0; c < 3; c++) begin : gen_c
          for (genvar b = 0; b < 20; b++) begin : gen_b
            tran i_bit (south_io[N][c][b], north_io[S][c][b]);
          end
        end
      end else begin : gen_no_south
        assign s_cts_i[N] = 1'b0;
        assign s_rts_i[N] = 1'b0;
        assign s_tbr_i[N] = 1'b0;
      end
      if (gy == 0) begin : gen_no_north
        assign n_cts_i[N] = 1'b0;
        assign n_rts_i[N] = 1'b0;
        assign n_tbr_i[N] = 1'b0;
      end
    end
  end

  // ==========================================================================================
  // TCDM model, one per compute node. One-cycle read latency (MemoryMacroLatency = 1 with
  // RegisterTCDMCuts = 0), responses cannot be back-pressured. 32 word-interleaved 64-bit
  // banks, one access per bank per cycle: the wide DMA port (axi_to_mem_interleaved, one 512 b
  // super-bank access) wins its 8 banks unconditionally, as in snitch_cluster.sv; the narrow
  // ports (xDMA, SIMD) share the rest in a priority order that rotates every cycle.
  // `lw_t` keeps the time of every word's last write: flags and result words "land" then.
  // ==========================================================================================
  logic [63:0] mem  [NumCompute][TcdmWords];
  realtime     lw_t [NumCompute][TcdmWords];

  logic       [NarrowPorts-1:0] tcdm_gnt   [NumCompute];
  logic       [NarrowPorts-1:0] rsp_valid_q[NumCompute];
  tcdm_data_t [NarrowPorts-1:0] rsp_data_q [NumCompute];
  int unsigned rot_q;

  always_ff @(posedge clk_cl or negedge rst_n) begin
    if (!rst_n) rot_q <= 0;
    else rot_q <= (rot_q + 1) % NarrowPorts;
  end

  for (genvar c = 0; c < NumCompute; c++) begin : gen_tcdm
    assign nreq[c]          = {simd_req[c], xdma_tcdm_req[c]};
    assign xdma_tcdm_rsp[c] = nrsp[c][TcdmPorts-1:0];
    assign simd_rsp[c]      = nrsp[c][NarrowPorts-1:TcdmPorts];

    always_comb begin
      automatic logic [TcdmBanks-1:0] taken = '0;
      tcdm_gnt[c] = '0;
      if (wreq[c]) taken[waddr[c][7:6]*8+:8] = 8'hFF;
      for (int unsigned k = 0; k < NarrowPorts; k++) begin
        automatic int unsigned p = (k + rot_q) % NarrowPorts;
        automatic int unsigned b = (nreq[c][p].q.addr >> 3) % TcdmBanks;
        if (nreq[c][p].q_valid) begin
          if (!TcdmBanked) begin
            tcdm_gnt[c][p] = 1'b1;
          end else if (!taken[b]) begin
            tcdm_gnt[c][p] = 1'b1;
            taken[b] = 1'b1;
          end
        end
      end
    end

    for (genvar p = 0; p < NarrowPorts; p++) begin : gen_port
      assign nrsp[c][p].q_ready = tcdm_gnt[c][p];
      assign nrsp[c][p].p_valid = rsp_valid_q[c][p];
      assign nrsp[c][p].p.data  = rsp_data_q[c][p];
    end

    // plain `always`: the testbench tasks also write `mem` and `lw_t` (seeding, sentinels)
    always @(posedge clk_cl or negedge rst_n) begin
      if (!rst_n) begin
        rsp_valid_q[c] <= '0;
        wrvalid_q[c]   <= 1'b0;
      end else begin
        for (int unsigned p = 0; p < NarrowPorts; p++) begin
          rsp_valid_q[c][p] <= 1'b0;
          if (nreq[c][p].q_valid && tcdm_gnt[c][p]) begin
            automatic int unsigned w = nreq[c][p].q.addr >> 3;
            if (nreq[c][p].q.write) begin
              for (int unsigned bb = 0; bb < 8; bb++) begin
                if (nreq[c][p].q.strb[bb]) mem[c][w][bb*8+:8] <= nreq[c][p].q.data[bb*8+:8];
              end
              lw_t[c][w] <= $realtime;
            end else begin
              rsp_data_q[c][p]  <= mem[c][w];
              rsp_valid_q[c][p] <= 1'b1;
            end
          end
        end
        // the wide port answers every granted request, writes included (axi_to_mem pops its
        // response FIFO on every mem_rvalid)
        wrvalid_q[c] <= wreq[c];
        if (wreq[c]) begin
          automatic int unsigned w0 = (waddr[c][TcdmAddrWidth-1:0] >> 6) << 3;
          for (int unsigned i = 0; i < 8; i++) begin
            if (wwe[c]) begin
              for (int unsigned bb = 0; bb < 8; bb++) begin
                if (wstrb[c][8*i+bb]) mem[c][w0+i][bb*8+:8] <= wwdata[c][64*i+8*bb+:8];
              end
              if (wstrb[c][8*i+:8] != 8'h00) lw_t[c][w0+i] <= $realtime;
            end else begin
              wrdata_q[c][64*i+:64] <= mem[c][w0+i];
            end
          end
        end
      end
    end
  end

  // ==========================================================================================
  // SIMD model, one per compute node
  // ==========================================================================================
  logic                     simd_start [NumCompute];
  logic [TcdmAddrWidth-1:0] simd_a     [NumCompute];
  logic [TcdmAddrWidth-1:0] simd_b     [NumCompute];
  logic [TcdmAddrWidth-1:0] simd_o     [NumCompute];
  int unsigned              simd_n     [NumCompute];
  logic                     simd_busy  [NumCompute];
  logic                     simd_err   [NumCompute];

  for (genvar c = 0; c < NumCompute; c++) begin : gen_simd
    reduce_simd_model i_simd (
      .clk_i  (clk_cl),
      .rst_ni (rst_n),
      .start_i(simd_start[c]),
      .a_i    (simd_a[c]),
      .b_i    (simd_b[c]),
      .o_i    (simd_o[c]),
      .bytes_i(simd_n[c]),
      .busy_o (simd_busy[c]),
      .err_o  (simd_err[c]),
      .req_o  (simd_req[c]),
      .rsp_i  (simd_rsp[c])
    );
  end

  // ==========================================================================================
  // Instrumentation: SRAM accesses (one per 64-bit word a bank serves: a narrow handshake is
  // one, a wide read eight, a wide write the words its strobes touch), per-chip first/last
  // access. One always block for the whole array, so simultaneous accesses are all counted.
  // ==========================================================================================
  longint unsigned sram_rd [NumCompute];
  longint unsigned sram_wr [NumCompute];
  realtime first_acc_t [NumCompute];
  realtime last_acc_t  [NumCompute];
  realtime any_acc_t;

  always @(posedge clk_cl) begin
    if (rst_n && mon_on) begin
      for (int unsigned c = 0; c < NumCompute; c++) begin
        automatic int unsigned nrd = 0;
        automatic int unsigned nwr = 0;
        for (int unsigned p = 0; p < NarrowPorts; p++) begin
          if (nreq[c][p].q_valid && tcdm_gnt[c][p]) begin
            if (nreq[c][p].q.write) nwr++;
            else nrd++;
          end
        end
        if (wreq[c]) begin
          if (wwe[c]) begin
            for (int unsigned i = 0; i < 8; i++) if (wstrb[c][8*i+:8] != 8'h00) nwr++;
          end else begin
            nrd += 8;
          end
        end
        if (nrd + nwr > 0) begin
          if (sram_rd[c] + sram_wr[c] == 0) first_acc_t[c] <= $realtime;
          last_acc_t[c] <= $realtime;
          any_acc_t     <= $realtime;
        end
        sram_rd[c] <= sram_rd[c] + nrd;
        sram_wr[c] <= sram_wr[c] + nwr;
      end
    end
  end

  // ==========================================================================================
  // The BINGO manager, per compute node: a model with the manager's measured internal
  // latencies, behind the real core <-> manager CSR clock crossings (occamy_quad.sv:
  // cdc_fifo_gray, LOG_DEPTH 1, SYNC_STAGES 2, cluster clock <-> host clock), sending its
  // dep-set writes as real AXI-Lite writes through the quadrant and the links to the target
  // chip's CHIPLET_DONE_QUEUE mailbox ({chip, 0x0a00_0000}).
  //
  // The core writes `done` (code: bit 31; bit 0 a local set, bit 1 a remote set to chip
  // [23:16]) and then reads its next task (bit 30), blocking until the manager answers. The
  // manager answers once `need` dependencies are set: a local set SchedLocalReadyCc after the
  // done, a remote one SchedRxReadyCc after its write arrives, and then SchedReadyRspCc later.
  // ==========================================================================================
  logic [31:0] sq_src_data [NumCompute];
  logic        sq_src_valid[NumCompute], sq_src_ready[NumCompute];
  logic [31:0] sq_dst_data [NumCompute];
  logic        sq_dst_valid[NumCompute];
  logic [31:0] sp_src_data [NumCompute];
  logic        sp_src_valid[NumCompute], sp_src_ready[NumCompute];
  logic [31:0] sp_dst_data [NumCompute];
  logic        sp_dst_valid[NumCompute];

  realtime     dep_q    [NumCompute][$];
  int unsigned need_q   [NumCompute];
  bit          read_pend[NumCompute];
  bit          rsp_arm  [NumCompute];
  realtime     rsp_at   [NumCompute];
  chip_id_t    tx_tgt_q [NumCompute][$];
  realtime     tx_at_q  [NumCompute][$];
  int unsigned edges_sent;

  for (genvar c = 0; c < NumCompute; c++) begin : gen_sched
    cdc_fifo_gray #(
      .T          (logic [31:0]),
      .LOG_DEPTH  (1),
      .SYNC_STAGES(2)
    ) i_csr_req_cdc (
      .src_rst_ni (rst_n),
      .src_clk_i  (clk_cl),
      .src_data_i (sq_src_data[c]),
      .src_valid_i(sq_src_valid[c]),
      .src_ready_o(sq_src_ready[c]),
      .dst_rst_ni (rst_n),
      .dst_clk_i  (clk_fab),
      .dst_data_o (sq_dst_data[c]),
      .dst_valid_o(sq_dst_valid[c]),
      .dst_ready_i(1'b1)
    );
    cdc_fifo_gray #(
      .T          (logic [31:0]),
      .LOG_DEPTH  (1),
      .SYNC_STAGES(2)
    ) i_csr_rsp_cdc (
      .src_rst_ni (rst_n),
      .src_clk_i  (clk_fab),
      .src_data_i (sp_src_data[c]),
      .src_valid_i(sp_src_valid[c]),
      .src_ready_o(sp_src_ready[c]),
      .dst_rst_ni (rst_n),
      .dst_clk_i  (clk_cl),
      .dst_data_o (sp_dst_data[c]),
      .dst_valid_o(sp_dst_valid[c]),
      .dst_ready_i(1'b1)
    );

    // the manager
    always @(posedge clk_fab or negedge rst_n) begin
      if (!rst_n) begin
        sp_src_valid[c] <= 1'b0;
        sp_src_data[c]  <= '0;
      end else begin
        if (sq_dst_valid[c]) begin
          if (sq_dst_data[c][31]) begin
            if (sq_dst_data[c][0]) dep_q[c].push_back($realtime + SchedLocalReadyCc * FabricPeriod);
            if (sq_dst_data[c][1]) begin
              tx_tgt_q[c].push_back(chip_id_t'(sq_dst_data[c][23:16]));
              tx_at_q[c].push_back($realtime + SchedTxCc * FabricPeriod);
            end
          end else if (sq_dst_data[c][30]) begin
            read_pend[c] = 1'b1;
          end
        end
        if (read_pend[c] && !rsp_arm[c]) begin
          automatic int unsigned have = 0;
          foreach (dep_q[c][i]) if (dep_q[c][i] <= $realtime + 0.01) have++;
          if (have >= need_q[c]) begin
            automatic int unsigned take = need_q[c];
            for (int i = dep_q[c].size() - 1; i >= 0 && take > 0; i--) begin
              if (dep_q[c][i] <= $realtime + 0.01) begin
                dep_q[c].delete(i);
                take--;
              end
            end
            rsp_at[c]    = $realtime + SchedReadyRspCc * FabricPeriod;
            rsp_arm[c]   = 1'b1;
            read_pend[c] = 1'b0;
          end
        end
        if (sp_src_valid[c] && sp_src_ready[c]) begin
          sp_src_valid[c] <= 1'b0;
        end else if (!sp_src_valid[c] && rsp_arm[c] && $realtime + 0.01 >= rsp_at[c]) begin
          sp_src_valid[c] <= 1'b1;
          sp_src_data[c]  <= 32'h1;
          rsp_arm[c] = 1'b0;
        end
      end
    end

    // its to_remote_chiplet_axi_lite master: one dep-set write at a time
    int unsigned tx_state;
    always @(posedge clk_fab or negedge rst_n) begin
      if (!rst_n) begin
        stx_req[c] <= '0;
        tx_state   <= 0;
      end else begin
        case (tx_state)
          0: if (tx_at_q[c].size() > 0 && tx_at_q[c][0] <= $realtime + 0.01) begin
               stx_req[c].aw.addr  <= {tx_tgt_q[c][0], 40'h00_0a00_0000};
               stx_req[c].aw.prot  <= '0;
               stx_req[c].aw_valid <= 1'b1;
               stx_req[c].w.data   <= 64'(chip_of_cidx(c));
               stx_req[c].w.strb   <= '1;
               stx_req[c].w_valid  <= 1'b1;
               stx_req[c].b_ready  <= 1'b1;
               if (mon_on) begin
                 edges_sent++;
                 $display("   [EDGE] tx %02x -> %02x at %0t", chip_of_cidx(c), tx_tgt_q[c][0],
                          $realtime);
               end
               tx_state <= 1;
             end
          1: begin
               automatic bit aw_open = stx_req[c].aw_valid && !stx_rsp[c].aw_ready;
               automatic bit w_open  = stx_req[c].w_valid && !stx_rsp[c].w_ready;
               if (stx_req[c].aw_valid && stx_rsp[c].aw_ready) stx_req[c].aw_valid <= 1'b0;
               if (stx_req[c].w_valid && stx_rsp[c].w_ready) stx_req[c].w_valid <= 1'b0;
               if (!aw_open && !w_open) tx_state <= 2;
             end
          2: if (stx_rsp[c].b_valid) begin
               void'(tx_at_q[c].pop_front());
               void'(tx_tgt_q[c].pop_front());
               tx_state <= 0;
             end
          default: tx_state <= 0;
        endcase
      end
    end

    // its from_remote_axi_lite mailbox: always ready; a write has arrived with its AW and W
    bit rx_aw_q, rx_w_q;
    always @(posedge clk_fab or negedge rst_n) begin
      if (!rst_n) begin
        srx_rsp[c] <= '0;
        rx_aw_q    <= 1'b0;
        rx_w_q     <= 1'b0;
      end else begin
        automatic bit aw = rx_aw_q || (srx_req[c].aw_valid && srx_rsp[c].aw_ready);
        automatic bit w  = rx_w_q || (srx_req[c].w_valid && srx_rsp[c].w_ready);
        srx_rsp[c].aw_ready <= 1'b1;
        srx_rsp[c].w_ready  <= 1'b1;
        srx_rsp[c].ar_ready <= 1'b0;
        if (srx_rsp[c].b_valid && srx_req[c].b_ready) srx_rsp[c].b_valid <= 1'b0;
        if (aw && w) begin
          dep_q[c].push_back($realtime + SchedRxReadyCc * FabricPeriod);
          srx_rsp[c].b_valid <= 1'b1;
          srx_rsp[c].b.resp  <= axi_pkg::RESP_OKAY;
          if (mon_on) $display("   [EDGE] rx at %02x at %0t", chip_of_cidx(c), $realtime);
          aw = 1'b0;
          w  = 1'b0;
        end
        rx_aw_q <= aw;
        rx_w_q  <= w;
      end
    end
  end

  task automatic sched_done(input int unsigned c, input bit local_set, input bit remote_set,
                            input chip_id_t tgt);
    @(negedge clk_cl);
    sq_src_data[c]  = {1'b1, 1'b0, 6'b0, tgt, 14'b0, remote_set, local_set};
    sq_src_valid[c] = 1'b1;
    do @(posedge clk_cl); while (!sq_src_ready[c]);
    @(negedge clk_cl);
    sq_src_valid[c] = 1'b0;
  endtask

  task automatic sched_read(input int unsigned c, input int unsigned need);
    need_q[c] = need;
    @(negedge clk_cl);
    sq_src_data[c]  = 32'h4000_0000;
    sq_src_valid[c] = 1'b1;
    do @(posedge clk_cl); while (!sq_src_ready[c]);
    @(negedge clk_cl);
    sq_src_valid[c] = 1'b0;
    do @(posedge clk_cl); while (!sp_dst_valid[c]);
  endtask

  // ==========================================================================================
  // Bookkeeping of a run
  // ==========================================================================================
  int unsigned errors = 0;
  realtime     t_first_act, t_first_start;   // the first programming action / START / copy
  int unsigned handoffs;

  task automatic fail(input string msg);
    errors++;
    $error("[FAIL] %s", msg);
  endtask

  function automatic void note_act(input realtime t);
    if (mon_on && t < t_first_act) t_first_act = t;
  endfunction

  function automatic void note_start(input realtime t);
    if (mon_on && t < t_first_start) t_first_start = t;
  endfunction

  task automatic ctr_clear();
    for (int unsigned c = 0; c < NumCompute; c++) begin
      sram_rd[c]      = 0;
      sram_wr[c]      = 0;
      first_acc_t[c]  = 0;
      last_acc_t[c]   = 0;
      wout_first_t[c] = 0;
      wout_last_t[c]  = 0;
      win_first_t[c]  = 0;
      win_last_t[c]   = 0;
      wout_n[c]       = 0;
      win_n[c]        = 0;
      naw_out[c]      = 0;
      naw_in[c]       = 0;
      dep_q[c].delete();
      read_pend[c]    = 1'b0;
      rsp_arm[c]      = 1'b0;
    end
    any_acc_t     = $realtime;
    t_first_act   = 1.0e18;
    t_first_start = 1.0e18;
    handoffs      = 0;
    edges_sent    = 0;
  endtask

  // ==========================================================================================
  // CSR tasks (the standalone bench's csr_wr / csr_rd / csr_wr64); each records its handshake
  // time per endpoint, so endpoints can be programmed concurrently.
  // ==========================================================================================
  localparam int unsigned SpinLimit = 200000;
  realtime csr_hs_t [NumCompute];

  task automatic csr_wr(input int unsigned ep, input int unsigned idx, input logic [31:0] d);
    int unsigned guard;
    @(negedge clk_cl);
    csr_addr[ep]  = idx;
    csr_wdata[ep] = d;
    csr_write[ep] = 1'b1;
    csr_valid[ep] = 1'b1;
    guard         = 0;
    forever begin
      @(posedge clk_cl);
      if (csr_req_ready[ep]) begin
        csr_hs_t[ep] = $realtime;
        note_act($realtime);
        break;
      end
      guard++;
      if (guard > SpinLimit) begin
        fail($sformatf("csr_wr(ep=%0d, idx=%0d) never accepted", ep, idx));
        break;
      end
    end
    @(negedge clk_cl);
    csr_valid[ep] = 1'b0;
  endtask

  // The read response is combinational (ReqRspManager answers in the cycle the request fires).
  task automatic csr_rd(input int unsigned ep, input int unsigned idx, output logic [31:0] d);
    int unsigned guard;
    @(negedge clk_cl);
    csr_addr[ep]  = idx;
    csr_wdata[ep] = '0;
    csr_write[ep] = 1'b0;
    csr_valid[ep] = 1'b1;
    d             = '0;
    guard         = 0;
    forever begin
      @(posedge clk_cl);
      if (csr_req_ready[ep]) begin
        csr_hs_t[ep] = $realtime;
        note_act($realtime);
        if (csr_rsp_valid[ep]) d = csr_rdata[ep];
        else fail($sformatf("csr_rd(ep=%0d, idx=%0d): request fired with no response", ep, idx));
        break;
      end
      guard++;
      if (guard > SpinLimit) begin
        fail($sformatf("csr_rd(ep=%0d, idx=%0d) never accepted", ep, idx));
        break;
      end
    end
    @(negedge clk_cl);
    csr_valid[ep] = 1'b0;
  endtask

  task automatic csr_wr64(input int unsigned ep, input int unsigned idx, input addr_t a);
    csr_wr(ep, idx, a[31:0]);
    csr_wr(ep, idx + 1, {16'h0, a[47:32]});
  endtask
  // ==========================================================================================
  // AXI-Lite tasks for the D2D link control ports (the host's register writes)
  // ==========================================================================================
  task automatic lite_wr(input int unsigned n, input logic [31:0] off, input logic [31:0] d);
    bit aw_done, w_done;
    int unsigned guard;
    @(negedge clk_cl);
    lite_req[n].aw.addr  = {16'h0, off};
    lite_req[n].aw.prot  = '0;
    lite_req[n].aw_valid = 1'b1;
    lite_req[n].w.data   = d;
    lite_req[n].w.strb   = '1;
    lite_req[n].w_valid  = 1'b1;
    lite_req[n].b_ready  = 1'b1;
    aw_done = 1'b0;
    w_done  = 1'b0;
    guard   = 0;
    while (!(aw_done && w_done)) begin
      @(posedge clk_cl);
      if (lite_req[n].aw_valid && lite_rsp[n].aw_ready) aw_done = 1'b1;
      if (lite_req[n].w_valid && lite_rsp[n].w_ready) w_done = 1'b1;
      @(negedge clk_cl);
      if (aw_done) lite_req[n].aw_valid = 1'b0;
      if (w_done) lite_req[n].w_valid = 1'b0;
      if (++guard > 1000) begin
        fail($sformatf("lite_wr(node %0d, 0x%03x) not accepted", n, off));
        return;
      end
    end
    guard = 0;
    forever begin
      @(posedge clk_cl);
      if (lite_rsp[n].b_valid) break;
      if (++guard > 1000) begin
        fail($sformatf("lite_wr(node %0d, 0x%03x) no B", n, off));
        break;
      end
    end
  endtask

  task automatic lite_rd(input int unsigned n, input logic [31:0] off, output logic [31:0] d);
    int unsigned guard;
    @(negedge clk_cl);
    lite_req[n].ar.addr  = {16'h0, off};
    lite_req[n].ar.prot  = '0;
    lite_req[n].ar_valid = 1'b1;
    lite_req[n].r_ready  = 1'b1;
    guard = 0;
    forever begin
      @(posedge clk_cl);
      if (lite_rsp[n].ar_ready) break;
      if (++guard > 1000) begin
        fail($sformatf("lite_rd(node %0d, 0x%03x) not accepted", n, off));
        return;
      end
    end
    @(negedge clk_cl);
    lite_req[n].ar_valid = 1'b0;
    guard = 0;
    forever begin
      @(posedge clk_cl);
      if (lite_rsp[n].r_valid) begin
        d = lite_rsp[n].r.data;
        break;
      end
      if (++guard > 1000) begin
        fail($sformatf("lite_rd(node %0d, 0x%03x) no R", n, off));
        break;
      end
    end
  endtask

  // The legacy host's link bring-up, as its end state (target/sw/shared/runtime/
  // hemaia_d2d_link.h: hemaia_d2d_link_initialize_grid, then hemaia_d2d_link_ddr_on_grid for
  // HEMAIA_D2D_DDR). Compute chips: TX turnaround silence 0, RX yield window 0
  // (HEMAIA_D2D_LINK_RX_YIELD_WINDOW default), availability = the ports facing a chip,
  // PHY DDR mode on all four sides. Memory chips (programmed by the configurer chip over the
  // link): availability, multicast fence = availability (transit only), RX yield 0, DDR on.
  // The memory chips keep their reset turnaround silence: the host never writes it there.
  // same_memchip_speed is set, so no port gets the slow-memchip yield / clock settings.
  task automatic link_bringup();
    logic [31:0] rb;
    for (int unsigned x = 0; x < GridX; x++) begin
      for (int unsigned y = 0; y < GridY; y++) begin
        automatic int unsigned n = x * GridY + y;
        automatic logic [3:0] links = d2d_links_of_xy(x, y);
        if (is_compute(x)) begin
          lite_wr(n, D2D_TX_TURNAROUND, 32'h0);
          lite_wr(n, D2D_RX_YIELD_WINDOW, 32'h0);
          lite_wr(n, D2D_AVAILABILITY, {28'h0, links});
        end else begin
          lite_wr(n, D2D_AVAILABILITY, {28'h0, links});
          lite_wr(n, D2D_MULTICAST_FENCE, {28'h0, links});
          lite_wr(n, D2D_RX_YIELD_WINDOW, 32'h0);
        end
      end
    end
    // DDR: every node, before any D2D traffic
    for (int unsigned n = 0; n < NumNodes; n++) lite_wr(n, D2D_PHY_DDR_MODE, 32'h0000_000F);
    repeat (100) @(posedge clk_cl);
    for (int unsigned n = 0; n < NumNodes; n++) begin
      lite_rd(n, D2D_PHY_DDR_MODE, rb);
      if (rb[3:0] !== 4'hF) fail($sformatf("node %0d: DDR mode reads back 0x%0x", n, rb));
    end
  endtask

  // ==========================================================================================
  // Payload and memory map. FP16 small integers, distinct per chip and per element (and per
  // beat), so every sum over a set of chips is exact in FP16 and a replayed beat or a wrong
  // operand cannot match. The buffers are snax-coll-reduce.c's (IN = TCDM + 0x08000, OUT =
  // + 0x28000, TMP = + 0x48000, 128 KiB apart); above 128 KiB they move to N apart.
  // ==========================================================================================
  localparam int unsigned InOff   = 32'h0_8000;
  localparam int unsigned CVal    = 32'h0_1040;          // C_VAL: the word every flag copies (k)
  localparam int unsigned CCts    = 32'h0_1D00;          // C_CTS: clear-to-send slots, by level
  localparam int unsigned CSync   = 32'h0_2100;          // C_SYNC: data flags, by step
  localparam logic [63:0] Sentinel = {2{32'hDEAD_BEEF}};

  function automatic int unsigned buf_stride(input int unsigned bytes);
    return (bytes > 32'h2_0000) ? bytes : 32'h2_0000;
  endfunction
  function automatic int unsigned out_off(input int unsigned bytes);
    return InOff + buf_stride(bytes);
  endfunction
  function automatic int unsigned tmp_off(input int unsigned bytes);
    return InOff + 2 * buf_stride(bytes);
  endfunction

  function automatic int unsigned val(input int unsigned s, input int unsigned i);
    return ((i >> 5) * 3 + i + 7 * s) % 32;
  endfunction

  function automatic logic [15:0] fp16_of(input int unsigned n);
    int unsigned e;
    if (n == 0) return 16'h0000;
    e = 0;
    while ((n >> (e + 1)) != 0) e++;
    return {1'b0, 5'(e + 15), 10'((n << (10 - e)) & 32'h3FF)};
  endfunction

  function automatic logic [63:0] word_of(input int unsigned s, input int unsigned w);
    return {fp16_of(val(s, 4 * w + 3)), fp16_of(val(s, 4 * w + 2)),
            fp16_of(val(s, 4 * w + 1)), fp16_of(val(s, 4 * w))};
  endfunction

  // the sum over snake positions lo .. hi-1
  function automatic logic [63:0] golden_word(input int unsigned lo, input int unsigned hi,
                                              input int unsigned w);
    logic [63:0] g;
    for (int unsigned l = 0; l < 4; l++) begin
      automatic int unsigned sum = 0;
      for (int unsigned s = lo; s < hi; s++) sum += val(s, 4 * w + l);
      g[16*l+:16] = fp16_of(sum);
    end
    return g;
  endfunction

  // the sum over an explicit set of snake positions (bit s set)
  function automatic logic [63:0] golden_set(input logic [15:0] set, input int unsigned w);
    logic [63:0] g;
    for (int unsigned l = 0; l < 4; l++) begin
      automatic int unsigned sum = 0;
      for (int unsigned s = 0; s < NumCompute; s++) if (set[s]) sum += val(s, 4 * w + l);
      g[16*l+:16] = fp16_of(sum);
    end
    return g;
  endfunction

  function automatic int unsigned ci_of_pos(input int unsigned s);
    return cidx_of_chip(Snake[s]);
  endfunction

  function automatic addr_t rem(input int unsigned s, input int unsigned off);
    return cluster_base(Snake[s]) + off;
  endfunction

  // Compare `bytes` at chip position s, offset off, with the sum over the position set.
  task automatic check_buf(input int unsigned s, input int unsigned off, input int unsigned bytes,
                           input logic [15:0] set, input string what, inout int unsigned bad);
    automatic int unsigned c = ci_of_pos(s);
    automatic int unsigned nb = 0;
    for (int unsigned w = 0; w < bytes / 8; w++) begin
      automatic logic [63:0] got = mem[c][(off >> 3) + w];
      automatic logic [63:0] exp = golden_set(set, w);
      if (got !== exp) begin
        if (nb < 3) $display("   [check] %s word %0d: got %016x want %016x", what, w, got, exp);
        nb++;
      end
    end
    if (nb != 0) fail($sformatf("%s: %0d of %0d words wrong", what, nb, bytes / 8));
    bad += nb;
  endtask

  function automatic logic [15:0] range_set(input int unsigned lo, input int unsigned hi);
    logic [15:0] m = '0;
    for (int unsigned s = lo; s < hi; s++) m[s] = 1'b1;
    return m;
  endfunction

  // untimed set-up of a point: every chip's partial in IN, OUT and TMP filled with the
  // sentinel, C_VAL = k (the point's flag value)
  task automatic prepare(input int unsigned bytes, input int unsigned k);
    for (int unsigned s = 0; s < NumCompute; s++) begin
      automatic int unsigned c = ci_of_pos(s);
      for (int unsigned w = 0; w < bytes / 8; w++) begin
        mem[c][(InOff >> 3) + w]           = word_of(s, w);
        mem[c][(out_off(bytes) >> 3) + w]  = Sentinel;
        mem[c][(tmp_off(bytes) >> 3) + w]  = Sentinel;
      end
      mem[c][CVal >> 3] = {32'h0, k};
    end
  endtask

  // ==========================================================================================
  // The cores: stores over the links, the iDMA's instructions, the SIMD, flags
  // ==========================================================================================
  // A 32-bit store (snrt_xchip_writew): a narrow AXI write into the cluster xbar's CoreReq port;
  // to its own chip, a plain store into its TCDM.
  task automatic core_store(input int unsigned s_from, input int unsigned s_to,
                            input int unsigned off, input logic [31:0] d);
    automatic int unsigned c = ci_of_pos(s_from);
    automatic addr_t a = rem(s_to, off);
    bit aw_done, w_done;
    if (s_from == s_to) begin
      @(posedge clk_cl);
      note_act($realtime);
      mem[c][off >> 3][32*off[2]+:32] = d;
      lw_t[c][off >> 3] = $realtime;
      return;
    end
    @(negedge clk_cl);
    core_req[c].aw         = '0;
    core_req[c].aw.addr    = a;
    core_req[c].aw.size    = 3'd2;
    core_req[c].aw.burst   = axi_pkg::BURST_INCR;
    core_req[c].aw.cache   = axi_pkg::CACHE_MODIFIABLE;
    core_req[c].aw_valid   = 1'b1;
    core_req[c].w          = '0;
    core_req[c].w.data     = {2{d}};
    core_req[c].w.strb     = 8'h0F << a[2:0];
    core_req[c].w.last     = 1'b1;
    core_req[c].w_valid    = 1'b1;
    core_req[c].b_ready    = 1'b1;
    aw_done = 1'b0;
    w_done  = 1'b0;
    while (!(aw_done && w_done)) begin
      @(posedge clk_cl);
      if (core_req[c].aw_valid && core_rsp[c].aw_ready) begin
        aw_done = 1'b1;
        note_act($realtime);
      end
      if (core_req[c].w_valid && core_rsp[c].w_ready) w_done = 1'b1;
      @(negedge clk_cl);
      if (aw_done) core_req[c].aw_valid = 1'b0;
      if (w_done) core_req[c].w_valid = 1'b0;
    end
  endtask

  // iDMA instructions (idma_inst64_snitch_pkg), as snRuntime's dma.h encodes them
  localparam logic [31:0] OpDmSrc   = 32'h0000_002B;
  localparam logic [31:0] OpDmDst   = 32'h0200_002B;
  localparam logic [31:0] OpDmCpyI  = 32'h0400_002B;   // config 0
  localparam logic [31:0] OpDmStatI = 32'h0820_002B;   // status 2 = busy

  int unsigned acc_cnt  [NumCompute];
  logic [63:0] acc_last [NumCompute];
  for (genvar c = 0; c < NumCompute; c++) begin : gen_acc_rsp
    always @(posedge clk_cl) begin
      if (rst_n && acc_rsp_valid[c]) begin
        acc_last[c] <= acc_rsp[c].data;
        acc_cnt[c]  <= acc_cnt[c] + 1;
      end
    end
  end

  task automatic acc_cmd(input int unsigned c, input logic [31:0] op, input logic [63:0] a,
                         input logic [63:0] b, input bit want_rsp, output logic [63:0] rd);
    int unsigned cnt0;
    @(negedge clk_cl);
    acc_req[c]           = '0;
    acc_req[c].addr      = 32'd2;            // snitch_pkg::DMA_SS
    acc_req[c].data_op   = op;
    acc_req[c].data_arga = a;
    acc_req[c].data_argb = b;
    acc_req_valid[c]     = 1'b1;
    cnt0 = acc_cnt[c];
    do @(posedge clk_cl); while (!acc_req_ready[c]);
    note_act($realtime);
    if (op == OpDmCpyI) note_start($realtime);
    @(negedge clk_cl);
    acc_req_valid[c] = 1'b0;
    rd = '0;
    if (want_rsp) begin
      while (acc_cnt[c] == cnt0) @(posedge clk_cl);
      rd = acc_last[c];
    end
  endtask

  // snrt_dma_start_1d_wideptr: dmsrc, dmdst, dmcpyi
  task automatic dma(input int unsigned c, input addr_t dst, input addr_t src,
                     input int unsigned n);
    logic [63:0] rd;
    acc_cmd(c, OpDmSrc, {32'h0, src[31:0]}, {48'h0, src[47:32]}, 1'b0, rd);
    acc_cmd(c, OpDmDst, {32'h0, dst[31:0]}, {48'h0, dst[47:32]}, 1'b0, rd);
    acc_cmd(c, OpDmCpyI, 64'(n), '0, 1'b1, rd);
  endtask

  // snrt_dma_wait_all: dmstati busy until it reads 0
  task automatic dma_wait_all(input int unsigned c);
    logic [63:0] rd;
    forever begin
      acc_cmd(c, OpDmStatI, '0, '0, 1'b1, rd);
      if (rd == 0) break;
    end
  endtask

  // out = a + b on the SIMD (snax-coll-reduce.c simd_add), blocking until done
  task automatic simd_add(input int unsigned c, input int unsigned o, input int unsigned a,
                          input int unsigned b, input int unsigned n);
    @(negedge clk_cl);
    simd_a[c]     = TcdmAddrWidth'(a);
    simd_b[c]     = TcdmAddrWidth'(b);
    simd_o[c]     = TcdmAddrWidth'(o);
    simd_n[c]     = n;
    simd_start[c] = 1'b1;
    @(posedge clk_cl);
    note_act($realtime);
    @(negedge clk_cl);
    simd_start[c] = 1'b0;
    do @(posedge clk_cl); while (simd_busy[c]);
    if (simd_err[c]) fail($sformatf("chip %02x: SIMD operand not a small FP16 integer",
                                    chip_of_cidx(c)));
  endtask

  // wait_eq on a flag word in this chip's TCDM, then the software cost of the hand-off
  task automatic wait_flag(input int unsigned s, input int unsigned off, input logic [31:0] k,
                           output realtime t_land);
    automatic int unsigned c = ci_of_pos(s);
    automatic int unsigned guard = 0;
    while (mem[c][off >> 3][32*off[2]+:32] !== k) begin
      @(posedge clk_cl);
      if (++guard > 4000000) begin
        fail($sformatf("chip %02x: flag at 0x%05x never reached %0d", Snake[s], off, k));
        break;
      end
    end
    t_land = lw_t[c][off >> 3];
    handoffs++;
    while ($realtime < t_land + SwHandoffCc * ClusterPeriod - 0.01) @(posedge clk_cl);
  endtask

  // ==========================================================================================
  // One ChainGather issued at position `cs`, the CSR sequence of snax-coll-reduce.c's
  // xdma_chain_fold(): reader = its `src`, writer chain in data order ending at its `dst`;
  // ElementwiseJunction, ADD, FP16. Returns once the finish counter reached its target and
  // the first and the last word of `dst` hold data (the app's sentinel poll), after the app's
  // clean-up; t_end is the result complete.
  // ==========================================================================================
  typedef struct {
    int unsigned level;
    int unsigned pos;
    realtime     t_dep;
    realtime     t_prog;
    realtime     t_start;
    realtime     t_end;
  } lvl_rec_t;
  lvl_rec_t lvl_recs[$];

  task automatic xdma_fold(input int unsigned cs, input int unsigned src_off,
                           input addr_t chain[16], input int unsigned n,
                           input int unsigned dst_off, input int unsigned bytes,
                           output realtime t_prog, output realtime t_start,
                           output realtime t_fin, output realtime t_end, output bit ok);
    automatic int unsigned coll  = ci_of_pos(cs);
    automatic int unsigned beats = bytes / 64;
    automatic int unsigned wf    = dst_off >> 3;
    automatic int unsigned wl    = (dst_off + bytes - 8) >> 3;
    logic [31:0] r, cl0, cr0, cl1, cr1, fin, want, jct_st;
    int unsigned guard;
    bit          is_local;

    ok = 1'b1;
    // the app's sentinel in the first and the last word of dst
    @(posedge clk_cl);
    mem[coll][wf] = Sentinel;
    mem[coll][wl] = Sentinel;
    // xdma_disable_all_extensions(), then every junction disabled (read-modify-write)
    csr_wr(coll, CSR_SRC_ENABLE, 32'd0);
    t_prog = csr_hs_t[coll];
    csr_wr(coll, CSR_DST_ENABLE, 32'd0);
    for (int unsigned j = 0; j < XdmaDstJctNum; j++) begin
      csr_rd(coll, CSR_DST_JCT_ENABLE, r);
      csr_wr(coll, CSR_DST_JCT_ENABLE, r & ~(32'd1 << j));
    end
    // xdma_chain_gather_1d_full_address(): xdma_multicast_nd_full_address() ...
    csr_wr64(coll, CSR_SRC_ADDR, rem(cs, src_off));
    for (int unsigned k = 0; k < 16; k++) csr_wr64(coll, CSR_DST_ADDR + 2 * k, (k < n) ? chain[k] : '0);
    csr_wr(coll, CSR_SRC_SPAT_STRIDE, 32'd8);
    csr_wr(coll, CSR_DST_SPAT_STRIDE, 32'd8);
    for (int unsigned d = 0; d < 5; d++) begin
      csr_wr(coll, CSR_SRC_TEMP_BOUND + d, (d == 0) ? beats : 32'd1);
      csr_wr(coll, CSR_SRC_TEMP_STRIDE + d, (d == 0) ? 32'd64 : 32'd0);
    end
    for (int unsigned d = 0; d < 5; d++) begin
      csr_wr(coll, CSR_DST_TEMP_BOUND + d, (d == 0) ? beats : 32'd1);
      csr_wr(coll, CSR_DST_TEMP_STRIDE + d, (d == 0) ? 32'd64 : 32'd0);
    end
    csr_wr(coll, CSR_SRC_ENAB_CHAN, 32'hFFFF_FFFF);
    csr_wr(coll, CSR_DST_ENAB_CHAN, 32'hFFFF_FFFF);
    csr_wr(coll, CSR_DST_ENAB_BYTE, 32'hFFFF_FFFF);
    // ... then xdma_enable_dst_junction(ELEMENTWISE, {ADD | FP16})
    csr_rd(coll, CSR_DST_JCT_ENABLE, r);
    csr_wr(coll, CSR_DST_JCT_ENABLE, r | (32'd1 << JctElementwise));
    csr_wr(coll, CSR_DST_JCT_CSR + 0, JctCsrAddFp16);
    // the app reads the remote counters before starting (standing-finish check)
    csr_rd(coll, CSR_FINISH_REMOTE, fin);
    csr_rd(coll, CSR_COMMIT_REMOTE, r);
    if (fin > r) fail($sformatf("chip %02x: standing xDMA finish %0d > commit %0d", Snake[cs], fin, r));
    // xdma_start()
    csr_rd(coll, CSR_COMMIT_LOCAL, cl0);
    csr_rd(coll, CSR_COMMIT_REMOTE, cr0);
    csr_wr(coll, CSR_START, 32'd1);
    t_start = csr_hs_t[coll];
    note_start(t_start);
    guard = 0;
    forever begin
      csr_rd(coll, CSR_COMMIT_LOCAL, cl1);
      csr_rd(coll, CSR_COMMIT_REMOTE, cr1);
      if (cl1 !== cl0 || cr1 !== cr0) break;
      if (++guard > 2000) begin
        fail($sformatf("chip %02x: task never committed", Snake[cs]));
        ok = 1'b0;
        break;
      end
    end
    is_local = (cl1 !== cl0);
    want     = is_local ? cl1 : cr1;
    // xdma_wait_task()
    t_fin = 0;
    guard = 0;
    while (ok) begin
      csr_rd(coll, is_local ? CSR_FINISH_LOCAL : CSR_FINISH_REMOTE, fin);
      if (fin >= want) begin
        t_fin = csr_hs_t[coll];
        break;
      end
      if (++guard > 1000000) begin
        csr_rd(coll, CSR_JCT_STATUS, jct_st);
        fail($sformatf("chip %02x: finish counter stuck at %0d, want %0d (JCT_STATUS=0x%08x)",
                       Snake[cs], fin, want, jct_st));
        ok = 1'b0;
      end
    end
    // the first and the last word landed
    guard = 0;
    while (ok && (mem[coll][wf] === Sentinel || mem[coll][wl] === Sentinel)) begin
      @(posedge clk_cl);
      if (++guard > 200000) begin
        fail($sformatf("chip %02x: result words never landed", Snake[cs]));
        ok = 1'b0;
      end
    end
    t_end = t_fin;
    if (lw_t[coll][wf] + ClusterPeriod > t_end) t_end = lw_t[coll][wf] + ClusterPeriod;
    if (lw_t[coll][wl] + ClusterPeriod > t_end) t_end = lw_t[coll][wl] + ClusterPeriod;
    csr_rd(coll, CSR_JCT_STATUS, jct_st);
    if (jct_st[1:0] != 2'b00) fail($sformatf("chip %02x: JCT_STATUS=0x%08x", Snake[cs], jct_st));
    // the app's clean-up: disable the junction, clear slots 1..15
    csr_rd(coll, CSR_DST_JCT_ENABLE, r);
    csr_wr(coll, CSR_DST_JCT_ENABLE, r & ~(32'd1 << JctElementwise));
    for (int unsigned k = 1; k < 16; k++) csr_wr64(coll, CSR_DST_ADDR + 2 * k, '0);
  endtask

  // ==========================================================================================
  // The arms
  // ==========================================================================================
  // The binomial tree on snake positions (snax-coll-reduce.c tree_takes)
  function automatic int unsigned tree_takes(input int unsigned p, input int unsigned P);
    int unsigned n = 0;
    for (int unsigned b = 1; b < P; b <<= 1) begin
      if ((p & (2 * b - 1)) == b) break;
      if ((p & (2 * b - 1)) == 0 && p + b < P) n++;
    end
    return n;
  endfunction

  // the group root of positions first..last: the member fewest mesh hops from the root
  function automatic int unsigned group_root(input int unsigned first, input int unsigned last);
    automatic int rx = Snake[0][7:4], ry = Snake[0][3:0];
    automatic int unsigned best = first, best_d = 255;
    for (int unsigned p = first; p <= last; p++) begin
      automatic int dx = int'(Snake[p][7:4]) - rx;
      automatic int dy = int'(Snake[p][3:0]) - ry;
      automatic int unsigned d = ((dx < 0) ? -dx : dx) + ((dy < 0) ? -dy : dy);
      if (d < best_d) begin
        best_d = d;
        best   = p;
      end
    end
    return best;
  endfunction

  realtime t_result;        // the reduction's result complete at the collector
  int unsigned bad_words;
  bit          run_ok;

  // ---- chain: one ChainGather at position 0 over positions P-1 .. 0 ----
  task automatic arm_chain(input int unsigned P, input int unsigned bytes);
    addr_t   chain[16];
    realtime tp, ts, tf, te;
    bit      ok;
    for (int unsigned k = 0; k + 1 < P; k++) chain[k] = rem(P - 1 - k, InOff);
    chain[P-1] = rem(0, out_off(bytes));
    xdma_fold(0, InOff, chain, P, out_off(bytes), bytes, tp, ts, tf, te, ok);
    lvl_recs.push_back('{level: 0, pos: 0, t_dep: tp, t_prog: tp, t_start: ts, t_end: te});
    if (!ok) run_ok = 1'b0;
    t_result = te;
    check_buf(0, out_off(bytes), bytes, range_set(0, P), "chain result", bad_words);
  endtask

  // ---- two: groups of 4 consecutive positions into their group roots, then the root over
  //      the group roots (snax-coll-reduce.c xdma_two_level_reduce), synchronised by BINGO
  //      edges instead of software flags ----
  localparam int unsigned Group = 4;
  task automatic arm_two(input int unsigned P, input int unsigned bytes);
    automatic int unsigned K = (P + Group - 1) / Group;
    realtime grp_end[16];
    if (K == 1) begin
      arm_chain(P, bytes);
      return;
    end
    for (int unsigned g = 0; g < K; g++) begin
      fork
        automatic int unsigned gg = g;
        begin
          automatic int unsigned first = gg * Group;
          automatic int unsigned last  = ((first + Group < P) ? first + Group : P) - 1;
          automatic int unsigned gr    = group_root(first, last);
          automatic int unsigned sum   = (gr == 0) ? tmp_off(bytes) : out_off(bytes);
          addr_t   chain[16];
          int unsigned n;
          realtime tp, ts, tf, te;
          bit      ok;
          n = 0;
          for (int d = int'(last - first); d >= 1; d--) begin
            if (gr + d <= last) chain[n++] = rem(gr + d, InOff);
            if (int'(gr) >= int'(first) + d) chain[n++] = rem(gr - d, InOff);
          end
          chain[n++] = rem(gr, sum);
          xdma_fold(gr, InOff, chain, n, sum, bytes, tp, ts, tf, te, ok);
          if (!ok) run_ok = 1'b0;
          grp_end[gg] = te;
          lvl_recs.push_back('{level: 0, pos: gr, t_dep: tp, t_prog: tp, t_start: ts, t_end: te});
          check_buf(gr, sum, bytes, range_set(first, last + 1),
                    $sformatf("two P=%0d group %0d at %02x", P, gg, Snake[gr]), bad_words);
          // the BINGO edge: a group root's done sets the root's level-2 dependency
          if (gr != 0) sched_done(ci_of_pos(gr), 1'b0, 1'b1, Snake[0]);
          else sched_done(ci_of_pos(0), 1'b1, 1'b0, '0);
        end
      join_none
    end
    // the root: its level-2 task waits for its own group (local) and K-1 remote sets
    begin
      addr_t   chain[16];
      int unsigned n;
      realtime tp, ts, tf, te, tdep;
      bit      ok;
      #1ps;
      sched_read(ci_of_pos(0), K);
      tdep = 0;
      for (int unsigned g = 0; g < K; g++) if (grp_end[g] > tdep) tdep = grp_end[g];
      n = 0;
      for (int unsigned j = K - 1; j >= 1; j--) begin
        automatic int unsigned f = j * Group;
        automatic int unsigned l = ((f + Group < P) ? f + Group : P) - 1;
        chain[n++] = rem(group_root(f, l), (l > f) ? out_off(bytes) : InOff);
      end
      chain[n++] = rem(0, out_off(bytes));
      xdma_fold(0, tmp_off(bytes), chain, n, out_off(bytes), bytes, tp, ts, tf, te, ok);
      if (!ok) run_ok = 1'b0;
      lvl_recs.push_back('{level: 1, pos: 0, t_dep: tdep, t_prog: tp, t_start: ts, t_end: te});
      t_result = te;
      check_buf(0, out_off(bytes), bytes, range_set(0, P), $sformatf("two P=%0d result", P),
                bad_words);
    end
    wait fork;
  endtask

  // ---- binomial: log2 P levels of two-chip ChainGathers (snax-coll-reduce.c
  //      xdma_tree_reduce), synchronised by BINGO edges ----
  task automatic arm_binomial(input int unsigned P, input int unsigned bytes);
    realtime last_end[16];
    for (int unsigned s = 0; s < 16; s++) last_end[s] = 0;
    for (int unsigned p = 0; p < P; p++) begin
      fork
        automatic int unsigned pp = p;
        begin
          automatic int unsigned R = tree_takes(pp, P);
          automatic int unsigned acc = InOff;
          automatic int unsigned m = 0;
          for (int unsigned b = 1, l = 0; b < P; b <<= 1, l++) begin
            if ((pp & (2 * b - 1)) == b) break;
            if ((pp & (2 * b - 1)) == 0 && pp + b < P) begin
              automatic int unsigned giver = pp + b;
              automatic bit gtook = tree_takes(giver, P) > 0;
              automatic int unsigned need = (gtook ? 1 : 0) + ((m > 0) ? 1 : 0);
              automatic int unsigned dst, give;
              addr_t   chain[16];
              realtime tp, ts, tf, te, tdep;
              bit      ok;
              m++;
              dst  = ((R - m) & 1) ? tmp_off(bytes) : out_off(bytes);
              give = gtook ? out_off(bytes) : InOff;
              if (need > 0) sched_read(ci_of_pos(pp), need);
              tdep = (m > 1) ? last_end[pp] : 0;
              if (gtook) begin
                wait (last_end[giver] > 0);
                if (last_end[giver] > tdep) tdep = last_end[giver];
              end
              chain[0] = rem(giver, give);
              chain[1] = rem(pp, dst);
              xdma_fold(pp, acc, chain, 2, dst, bytes, tp, ts, tf, te, ok);
              if (!ok) run_ok = 1'b0;
              if (tdep == 0) tdep = tp;
              lvl_recs.push_back('{level: l, pos: pp, t_dep: tdep, t_prog: tp, t_start: ts,
                                   t_end: te});
              check_buf(pp, dst, bytes, range_set(pp, (pp + 2 * b < P) ? pp + 2 * b : P),
                        $sformatf("binomial P=%0d level %0d at %02x", P, l, Snake[pp]), bad_words);
              acc = dst;
              if (m == R) last_end[pp] = te;
              // the BINGO edge: the next take here (local), or the taker of this sum (remote)
              if (m < R) begin
                last_end[pp] = te;
                sched_done(ci_of_pos(pp), 1'b1, 1'b0, '0);
              end else if (pp != 0) begin
                sched_done(ci_of_pos(pp), 1'b0, 1'b1, Snake[pp - (pp & -pp)]);
              end else begin
                t_result = te;
              end
            end
          end
        end
      join_none
    end
    wait fork;
    check_buf(0, out_off(bytes), bytes, range_set(0, P), $sformatf("binomial P=%0d result", P),
              bad_words);
  endtask

  // ---- pull: the root reads every partial (one source in flight) and folds it on its SIMD,
  //      over two half-size slots of TMP (snax-coll-reduce.c fold_all) ----
  task automatic arm_pull(input int unsigned P, input int unsigned bytes);
    automatic int unsigned c = ci_of_pos(0);
    automatic int unsigned H = bytes / 2;
    automatic int unsigned T = 2 * (P - 1);
    dma(c, rem(0, tmp_off(bytes)), rem(1, InOff), H);
    for (int unsigned t = 0; t < T; t++) begin
      automatic int unsigned h = t & 1, src = 1 + t / 2;
      dma_wait_all(c);
      if (t + 1 < T) begin
        automatic int unsigned h1 = (t + 1) & 1, src1 = 1 + (t + 1) / 2;
        dma(c, rem(0, tmp_off(bytes) + h1 * H), rem(src1, InOff + h1 * H), H);
      end
      simd_add(c, out_off(bytes) + h * H, ((src == 1) ? InOff : out_off(bytes)) + h * H,
               tmp_off(bytes) + h * H, H);
    end
    t_result = $realtime;
    check_buf(0, out_off(bytes), bytes, range_set(0, P), $sformatf("pull P=%0d result", P),
              bad_words);
  endtask

  // ---- swchain: receive, add, forward, pipelined in 4 chunks ----
  localparam int unsigned ChainChunks = 4;
  task automatic arm_swchain(input int unsigned P, input int unsigned bytes, input logic [31:0] k);
    automatic int unsigned Q = bytes / ChainChunks;
    for (int unsigned p = 0; p < P; p++) begin
      fork
        automatic int unsigned me = p;
        begin
          automatic int unsigned c = ci_of_pos(me);
          automatic bit far = (me == P - 1);
          for (int unsigned q = 0; q < ChainChunks; q++) begin
            if (!far) begin
              realtime tl;
              wait_flag(me, CSync + 64 * q, k, tl);
              simd_add(c, out_off(bytes) + q * Q, InOff + q * Q, tmp_off(bytes) + q * Q, Q);
              if (me == 0 && q == ChainChunks - 1) t_result = $realtime;
            end
            if (me != 0) begin
              dma(c, rem(me - 1, tmp_off(bytes) + q * Q),
                  rem(me, (far ? InOff : out_off(bytes)) + q * Q), Q);
              dma(c, rem(me - 1, CSync + 64 * q), rem(me, CVal), 4);
            end
          end
          dma_wait_all(c);
        end
      join_none
    end
    wait fork;
    for (int unsigned p = 0; p + 1 < P; p++)
      check_buf(p, out_off(bytes), bytes, range_set(p, P),
                $sformatf("swchain P=%0d sum at %02x", P, Snake[p]), bad_words);
  endtask

  // ---- swtree: binomial tree, each receive behind a clear-to-send and a flag ----
  task automatic arm_swtree(input int unsigned P, input int unsigned bytes, input logic [31:0] k);
    automatic int unsigned L = 0;
    while ((1 << L) < P) L++;
    for (int unsigned p = 0; p < P; p++) begin
      fork
        automatic int unsigned me = p;
        begin
          automatic int unsigned c = ci_of_pos(me);
          automatic bit have = 1'b0;
          for (int unsigned l = 0; l < L; l++) begin
            automatic int unsigned b = 1 << l;
            if ((me & (2 * b - 1)) == b) begin
              automatic int unsigned to = me - b;
              realtime tl;
              if (l > 0) wait_flag(me, CCts + 64 * l, k, tl);
              dma(c, rem(to, tmp_off(bytes)), rem(me, have ? out_off(bytes) : InOff), bytes);
              dma(c, rem(to, CSync + 64 * l), rem(me, CVal), 4);
              dma_wait_all(c);
              break;
            end
            if ((me & (2 * b - 1)) == 0 && me + b < P) begin
              realtime tl, ta;
              if (l > 0) core_store(me, me + b, CCts + 64 * l, k);
              wait_flag(me, CSync + 64 * l, k, tl);
              ta = $realtime;
              simd_add(c, out_off(bytes), have ? out_off(bytes) : InOff, tmp_off(bytes), bytes);
              have = 1'b1;
              // the level's "dependency" is its data flag landing; sync = flag -> add start
              lvl_recs.push_back('{level: l, pos: me, t_dep: tl, t_prog: ta, t_start: ta,
                                   t_end: $realtime});
              check_buf(me, out_off(bytes), bytes,
                        range_set(me, (me + 2 * b < P) ? me + 2 * b : P),
                        $sformatf("swtree P=%0d level %0d at %02x", P, l, Snake[me]), bad_words);
              if (me == 0 && l == L - 1) t_result = $realtime;
            end
          end
        end
      join_none
    end
    wait fork;
  endtask

  // ==========================================================================================
  // One point: set-up (untimed), the arm, quiescence, checks, the report
  // ==========================================================================================
  int unsigned point_k = 0;

  task automatic run_point(input string arm, input int unsigned P, input int unsigned bytes,
                           input int unsigned verbose);
    longint unsigned tot, mx, v;
    int unsigned busiest, guard, nlev;
    realtime     lvl_start[16], lvl_end[16], lvl_sync[16];
    string       lv, same;
    point_k++;
    prepare(bytes, point_k);
    lvl_recs.delete();
    bad_words = 0;
    run_ok    = 1'b1;
    t_result  = 0;
    repeat (20) @(posedge clk_cl);
    ctr_clear();
    mon_on = 1'b1;
    case (arm)
      "chain":    arm_chain(P, bytes);
      "two":      arm_two(P, bytes);
      "binomial": arm_binomial(P, bytes);
      "pull":     arm_pull(P, bytes);
      "swchain":  arm_swchain(P, bytes, point_k);
      "swtree":   arm_swtree(P, bytes, point_k);
      default:    fail($sformatf("unknown arm '%s'", arm));
    endcase
    // quiescence: no TCDM access anywhere for 256 cycles and no DMA busy
    guard = 0;
    forever begin
      automatic bit idle = ($realtime - any_acc_t >= 256 * ClusterPeriod);
      for (int unsigned c = 0; c < NumCompute; c++) if (idma_busy[c]) idle = 1'b0;
      if (idle) break;
      @(posedge clk_cl);
      if (++guard > 2000000) begin
        fail("no quiescence");
        break;
      end
    end
    mon_on = 1'b0;

    tot = 0;
    mx = 0;
    busiest = 0;
    for (int unsigned c = 0; c < NumCompute; c++) begin
      v = sram_rd[c] + sram_wr[c];
      tot += v;
      if (v > mx) begin
        mx = v;
        busiest = c;
      end
    end
    // per level: first programming, last completion, the longest edge (programming after the
    // last dependency completed)
    nlev = 0;
    foreach (lvl_recs[i]) if (lvl_recs[i].level + 1 > nlev) nlev = lvl_recs[i].level + 1;
    lv = "";
    for (int unsigned l = 0; l < nlev; l++) begin
      lvl_start[l] = 1.0e18;
      lvl_end[l]   = 0;
      lvl_sync[l]  = 0;
      foreach (lvl_recs[i]) begin
        if (lvl_recs[i].level == l) begin
          if (lvl_recs[i].t_prog > 0 && lvl_recs[i].t_prog < lvl_start[l]) lvl_start[l] = lvl_recs[i].t_prog;
          if (lvl_recs[i].t_end > lvl_end[l]) lvl_end[l] = lvl_recs[i].t_end;
          if (lvl_recs[i].t_prog > 0 && lvl_recs[i].t_prog - lvl_recs[i].t_dep > lvl_sync[l])
            lvl_sync[l] = lvl_recs[i].t_prog - lvl_recs[i].t_dep;
        end
      end
      if (lvl_start[l] > 1.0e17) lvl_start[l] = t_first_act;
      lv = {lv, $sformatf("%s%0d:%0d:%0d:%0d", (l == 0) ? "" : ";", l,
                          int'((lvl_start[l] - t_first_act) / ClusterPeriod),
                          int'((lvl_end[l] - t_first_act) / ClusterPeriod),
                          int'(lvl_sync[l] / ClusterPeriod))};
    end
    if (verbose >= 1) begin
      foreach (lvl_recs[i])
        $display("   [LEVEL] arm=%s P=%0d bytes=%0d lvl=%0d chip=%02x dep_cc=%0d prog_cc=%0d start_cc=%0d end_cc=%0d",
                 arm, P, bytes, lvl_recs[i].level, Snake[lvl_recs[i].pos],
                 int'((lvl_recs[i].t_dep - t_first_act) / ClusterPeriod),
                 int'((lvl_recs[i].t_prog - t_first_act) / ClusterPeriod),
                 int'((lvl_recs[i].t_start - t_first_act) / ClusterPeriod),
                 int'((lvl_recs[i].t_end - t_first_act) / ClusterPeriod));
      for (int unsigned s = 0; s < P; s++) begin
        automatic int unsigned c = ci_of_pos(s);
        $display("   [chip] s=%0d chip=%02x rd=%0d wr=%0d tcdm_first=%0d tcdm_last=%0d w_in=%0d [%0d..%0d] w_out=%0d [%0d..%0d] narrow_aw out=%0d in=%0d",
                 s, Snake[s], sram_rd[c], sram_wr[c],
                 int'((first_acc_t[c] - t_first_act) / ClusterPeriod),
                 int'((last_acc_t[c] - t_first_act) / ClusterPeriod), win_n[c],
                 (win_n[c] == 0) ? 0 : int'((win_first_t[c] - t_first_act) / ClusterPeriod),
                 (win_n[c] == 0) ? 0 : int'((win_last_t[c] - t_first_act) / ClusterPeriod),
                 wout_n[c],
                 (wout_n[c] == 0) ? 0 : int'((wout_first_t[c] - t_first_act) / ClusterPeriod),
                 (wout_n[c] == 0) ? 0 : int'((wout_last_t[c] - t_first_act) / ClusterPeriod),
                 naw_out[c], naw_in[c]);
      end
    end
    same = ((arm == "two" && P <= Group) || (arm == "binomial" && P == 2)) ? "chain" : "-";
    $display("[REDUCE] arm=%s P=%0d bytes=%0d lat_prog_cc=%0d lat_cc=%0d sram_total=%0d sram_max=%0d busiest=%02x check=%s levels=%0d lvl=%s edges=%0d handoffs=%0d same_as=%s",
             arm, P, bytes, int'((t_result - t_first_act) / ClusterPeriod),
             int'((t_result - t_first_start) / ClusterPeriod), tot, mx, chip_of_cidx(busiest),
             (run_ok && bad_words == 0 && t_result > 0) ? "PASS" : "FAIL", nlev, lv, edges_sent,
             handoffs, same);
  endtask

  // ==========================================================================================
  // Main
  // ==========================================================================================
  int unsigned plist[$];

  task automatic parse_plist(input string s);
    automatic int unsigned acc = 0;
    automatic bit have = 1'b0;
    plist.delete();
    for (int i = 0; i <= s.len(); i++) begin
      if (i < s.len() && s[i] >= "0" && s[i] <= "9") begin
        acc  = acc * 10 + (s[i] - "0");
        have = 1'b1;
      end else if (have) begin
        plist.push_back(acc);
        acc  = 0;
        have = 1'b0;
      end
    end
  endtask

  initial begin : proc_main
    string arm, pl;
    int unsigned bytes, verbose;

    for (int unsigned c = 0; c < NumCompute; c++) begin
      csr_addr[c]      = '0;
      csr_wdata[c]     = '0;
      csr_write[c]     = 1'b0;
      csr_valid[c]     = 1'b0;
      csr_rsp_ready[c] = 1'b1;
      sram_rd[c]       = 0;
      sram_wr[c]       = 0;
      wout_n[c]        = 0;
      win_n[c]         = 0;
      naw_out[c]       = 0;
      naw_in[c]        = 0;
      core_req[c]      = '0;
      acc_req[c]       = '0;
      acc_req_valid[c] = 1'b0;
      acc_cnt[c]       = 0;
      simd_start[c]    = 1'b0;
      simd_a[c]        = '0;
      simd_b[c]        = '0;
      simd_o[c]        = '0;
      simd_n[c]        = 0;
      sq_src_valid[c]  = 1'b0;
      sq_src_data[c]   = '0;
      need_q[c]        = 0;
      read_pend[c]     = 1'b0;
      rsp_arm[c]       = 1'b0;
    end
    for (int unsigned n = 0; n < NumNodes; n++) lite_req[n] = '0;
    mon_on = 1'b0;

    if (!$value$plusargs("ARM=%s", arm)) arm = "chain";
    if (!$value$plusargs("BYTES=%d", bytes)) bytes = 131072;
    if (!$value$plusargs("PLIST=%s", pl)) pl = "2,4,8,16";
    if (!$value$plusargs("VERBOSE=%d", verbose)) verbose = 0;
    parse_plist(pl);

    if (bytes % 256 != 0 || bytes == 0 || bytes > 262144)
      $fatal(1, "[tb_reduce] BYTES=%0d: must be a non-zero multiple of 256, at most 256 KiB", bytes);
    foreach (plist[i]) begin
      if (plist[i] < 2 || plist[i] > NumCompute || (plist[i] & (plist[i] - 1)) != 0)
        $fatal(1, "[tb_reduce] P=%0d: must be 2, 4, 8 or 16", plist[i]);
    end

    $display("[tb_reduce] arm=%s bytes=%0d P=%p TcdmBanked=%0d LinkTxWBufBeats=%0d SwHandoffCc=%0d Sched(tx,rx,local,rsp)=(%0d,%0d,%0d,%0d)",
             arm, bytes, plist, TcdmBanked, LinkTxWBufBeats, SwHandoffCc, SchedTxCc,
             SchedRxReadyCc, SchedLocalReadyCc, SchedReadyRspCc);

    @(posedge rst_n);
    repeat (20) @(posedge clk_cl);
    link_bringup();
    $display("[tb_reduce] links up (DDR) at %0t", $realtime);
    repeat (200) @(posedge clk_cl);

    foreach (plist[i]) begin
      run_point(arm, plist[i], bytes, verbose);
      repeat (100) @(posedge clk_cl);
    end

    if (errors == 0) $display("[tb_reduce] RESULT: PASS");
    else $display("[tb_reduce] RESULT: FAIL (%0d errors)", errors);
    $finish;
  end

  initial begin : proc_timeout
    int unsigned timeout_us;
    if (!$value$plusargs("TIMEOUT_US=%d", timeout_us)) timeout_us = 20000;
    #(timeout_us * 1us);
    $display("[tb_reduce] RESULT: FAIL (simulation-time watchdog, %0d us)", timeout_us);
    for (int unsigned c = 0; c < NumCompute; c++)
      $display("   [state] chip %02x: acc_cnt=%0d acc_req_valid=%0b ready=%0b idma_busy=%0b simd_busy=%0b dep_q=%0d read_pend=%0b",
               chip_of_cidx(c), acc_cnt[c], acc_req_valid[c], acc_req_ready[c], idma_busy[c],
               simd_busy[c], dep_q[c].size(), read_pend[c]);
    $finish;
  end

endmodule
