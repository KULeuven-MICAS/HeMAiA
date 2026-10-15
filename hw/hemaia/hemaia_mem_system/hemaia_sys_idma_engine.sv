// Copyright 2026 KU Leuven.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51
//
// Fanchen Kong <fanchen.kong@kuleuven.be>

// One memory chip push engine: an iDMA backend fed by two frontends.
//   * idma_reg64_1d: a transfer is launched by a read of next_id (offsets below 0x800).
//   * hemaia_sys_idma_queue: transfers posted with writes alone, run back to back (offsets
//     0x800 and up).
// A granted request is held until the backend takes it (the arbiter's LockIn), as both frontends
// need. The backend's read and write AXI ports are separate; the chip joins and routes them.

module hemaia_sys_idma_engine #(
    parameter int unsigned DataWidth = 512,
    parameter int unsigned AddrWidth = 48,
    parameter int unsigned AxiIdWidth = 4,
    parameter int unsigned TFLenWidth = 48,
    parameter int unsigned QueueDepth = 32,
    parameter type reg_req_t = logic,
    parameter type reg_rsp_t = logic,
    parameter type idma_req_t = logic,
    parameter type idma_rsp_t = logic,
    parameter type axi_req_t = logic,
    parameter type axi_rsp_t = logic,
    parameter type read_meta_channel_t = logic,
    parameter type write_meta_channel_t = logic
) (
    input  logic     clk_i,
    input  logic     rst_ni,
    // the engine's register window
    input  reg_req_t cfg_req_i,
    output reg_rsp_t cfg_rsp_o,
    // the backend's ports
    output axi_req_t axi_read_req_o,
    input  axi_rsp_t axi_read_rsp_i,
    output axi_req_t axi_write_req_o,
    input  axi_rsp_t axi_write_rsp_i
);

  idma_req_t            idma_req;
  logic                 idma_req_valid, idma_req_ready;
  idma_rsp_t            idma_rsp;
  logic                 idma_rsp_valid;
  idma_pkg::idma_busy_t idma_busy;
  logic [31:0]          next_id, done_id;

  reg_req_t  [1:0] win_req;
  reg_rsp_t  [1:0] win_rsp;
  idma_req_t [1:0] src_req;  // 0: register frontend, 1: queue
  logic      [1:0] src_valid, src_ready;

  reg_demux #(
      .NoPorts(2),
      .req_t  (reg_req_t),
      .rsp_t  (reg_rsp_t)
  ) i_win_demux (
      .clk_i      (clk_i),
      .rst_ni     (rst_ni),
      .in_select_i(cfg_req_i.addr[11]),
      .in_req_i   (cfg_req_i),
      .in_rsp_o   (cfg_rsp_o),
      .out_req_o  (win_req),
      .out_rsp_i  (win_rsp)
  );

  idma_reg64_1d #(
      .NumRegs   (32'd1),
      .NumStreams(32'd1),
      .reg_req_t (reg_req_t),
      .reg_rsp_t (reg_rsp_t),
      .dma_req_t (idma_req_t)
  ) i_idma_reg64_1d (
      .clk_i         (clk_i),
      .rst_ni        (rst_ni),
      .dma_ctrl_req_i(win_req[0]),
      .dma_ctrl_rsp_o(win_rsp[0]),
      .dma_req_o     (src_req[0]),
      .req_valid_o   (src_valid[0]),
      .req_ready_i   (src_ready[0]),
      .next_id_i     (next_id),
      .stream_idx_o  (/* NC */),
      .done_id_i     (done_id),
      .busy_i        (idma_busy),
      .midend_busy_i (1'b0)
  );

  hemaia_sys_idma_queue #(
      .Depth    (QueueDepth),
      .reg_req_t(reg_req_t),
      .reg_rsp_t(reg_rsp_t),
      .dma_req_t(idma_req_t)
  ) i_queue (
      .clk_i    (clk_i),
      .rst_ni   (rst_ni),
      .reg_req_i(win_req[1]),
      .reg_rsp_o(win_rsp[1]),
      .opt_i    (src_req[0]),
      .req_o    (src_req[1]),
      .valid_o  (src_valid[1]),
      .ready_i  (src_ready[1])
  );

  rr_arb_tree #(
      .NumIn    (2),
      .DataType (idma_req_t),
      .ExtPrio  (1'b0),
      .AxiVldRdy(1'b1),
      .LockIn   (1'b1)
  ) i_frontend_arb (
      .clk_i  (clk_i),
      .rst_ni (rst_ni),
      .flush_i(1'b0),
      .rr_i   ('0),
      .req_i  (src_valid),
      .gnt_o  (src_ready),
      .data_i (src_req),
      .gnt_i  (idma_req_ready),
      .req_o  (idma_req_valid),
      .data_o (idma_req),
      .idx_o  (/* NC */)
  );

  idma_transfer_id_gen #(
      .IdWidth(32'd32)
  ) i_transfer_id_gen (
      .clk_i      (clk_i),
      .rst_ni     (rst_ni),
      .issue_i    (idma_req_valid & idma_req_ready),
      .retire_i   (idma_rsp_valid),  // backend rsp_ready tied to 1
      .next_o     (next_id),
      .completed_o(done_id)
  );

  idma_backend_rw_axi #(
      .DataWidth           (DataWidth),
      .AddrWidth           (AddrWidth),
      .UserWidth           (1),
      .AxiIdWidth          (AxiIdWidth),
      .NumAxInFlight       (32'd64),
      .BufferDepth         (32'd3),
      .TFLenWidth          (TFLenWidth),
      .MemSysDepth         (32'd16),
      .CombinedShifter     (1'b1),
      .RAWCouplingAvail    (1'b1),
      .MaskInvalidData     (1'b1),
      .HardwareLegalizer   (1'b1),
      .RejectZeroTransfers (1'b1),
      .ErrorCap            (idma_pkg::NO_ERROR_HANDLING),
      .idma_req_t          (idma_req_t),
      .idma_rsp_t          (idma_rsp_t),
      .idma_eh_req_t       (idma_pkg::idma_eh_req_t),
      .idma_busy_t         (idma_pkg::idma_busy_t),
      .axi_req_t           (axi_req_t),
      .axi_rsp_t           (axi_rsp_t),
      .read_meta_channel_t (read_meta_channel_t),
      .write_meta_channel_t(write_meta_channel_t)
  ) i_idma_backend_rw_axi (
      .clk_i          (clk_i),
      .rst_ni         (rst_ni),
      .testmode_i     (1'b0),
      .idma_req_i     (idma_req),
      .req_valid_i    (idma_req_valid),
      .req_ready_o    (idma_req_ready),
      .idma_rsp_o     (idma_rsp),
      .rsp_valid_o    (idma_rsp_valid),
      .rsp_ready_i    (1'b1),
      .idma_eh_req_i  ('0),
      .eh_req_valid_i (1'b0),
      .eh_req_ready_o (/* NC */),
      .axi_read_req_o (axi_read_req_o),
      .axi_read_rsp_i (axi_read_rsp_i),
      .axi_write_req_o(axi_write_req_o),
      .axi_write_rsp_i(axi_write_rsp_i),
      .busy_o         (idma_busy)
  );

endmodule
