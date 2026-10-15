// Copyright 2026 KU Leuven.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51
//
// Fanchen Kong <fanchen.kong@kuleuven.be>

// A transfer queue in front of a memory chip push engine (an iDMA backend). The host posts a
// transfer with writes alone -- source, destination, length, then a doorbell -- and goes on; the
// queue hands its transfers to the backend back to back. Launching through idma_reg64_1d takes a
// read of next_id instead, which a host on another chip has to wait for across the link.
//
// Register window (offsets in the engine's window; the register frontend keeps the offsets below
// 0x800):
//   +0x800 / +0x804  source address, low / high word
//   +0x808 / +0x80c  destination address, low / high word
//   +0x810 / +0x814  length in bytes, low / high word
//   +0x818           doorbell: any write queues the transfer the three registers above hold. While
//                    the queue is full the write is held (its response waits for a free entry).
//   +0x81c           control: writes are ignored (software that enables the queue before using it
//                    writes 1 here); reads 1
//   +0x820           read: transfers waiting in the queue
//   +0x824           read: transfers handed to the backend since reset
//   +0x828           read: the queue's depth
// A queued transfer takes its options (protocols, bursts, decoupling, length limits) from opt_i:
// the register frontend's request, whose options follow its conf register.

module hemaia_sys_idma_queue #(
    parameter int unsigned Depth = 32,
    parameter type reg_req_t = logic,
    parameter type reg_rsp_t = logic,
    parameter type dma_req_t = logic
) (
    input  logic     clk_i,
    input  logic     rst_ni,
    // the queue's register window
    input  reg_req_t reg_req_i,
    output reg_rsp_t reg_rsp_o,
    // the options of every queued transfer
    input  dma_req_t opt_i,
    // to the backend
    output dma_req_t req_o,
    output logic     valid_o,
    input  logic     ready_i
);

  localparam logic [11:0] OffSrcLo = 12'h800, OffSrcHi = 12'h804;
  localparam logic [11:0] OffDstLo = 12'h808, OffDstHi = 12'h80c;
  localparam logic [11:0] OffLenLo = 12'h810, OffLenHi = 12'h814;
  localparam logic [11:0] OffDoorbell = 12'h818, OffCtrl = 12'h81c;
  localparam logic [11:0] OffCount = 12'h820, OffIssued = 12'h824, OffDepth = 12'h828;
  localparam int unsigned CountWidth = (Depth > 1) ? $clog2(Depth) : 1;

  if (Depth < 1) begin : gen_bad_depth
    $fatal(1, "hemaia_sys_idma_queue: Depth must be at least 1, got %0d", Depth);
  end

  typedef struct packed {
    logic [63:0] src;
    logic [63:0] dst;
    logic [63:0] len;
  } entry_t;

  entry_t staged_q, staged_d;
  entry_t head;
  logic full, empty, push, pop;
  logic [CountWidth-1:0] usage;
  logic [31:0] issued_q;

  logic [11:0] off;
  logic access, write;
  assign off = reg_req_i.addr[11:0];
  assign access = reg_req_i.valid;
  assign write = reg_req_i.write;

  // a 32-bit register word written byte by byte, as the strobe says
  function automatic logic [31:0] merge(input logic [31:0] old, input logic [31:0] wdata,
                                        input logic [3:0] wstrb);
    logic [31:0] w = old;
    for (int unsigned b = 0; b < 4; b++) if (wstrb[b]) w[8*b+:8] = wdata[8*b+:8];
    return w;
  endfunction

  always_comb begin
    staged_d = staged_q;
    if (access && write) begin
      case (off)
        OffSrcLo: staged_d.src[31:0] = merge(staged_q.src[31:0], reg_req_i.wdata, reg_req_i.wstrb);
        OffSrcHi: staged_d.src[63:32] = merge(staged_q.src[63:32], reg_req_i.wdata, reg_req_i.wstrb);
        OffDstLo: staged_d.dst[31:0] = merge(staged_q.dst[31:0], reg_req_i.wdata, reg_req_i.wstrb);
        OffDstHi: staged_d.dst[63:32] = merge(staged_q.dst[63:32], reg_req_i.wdata, reg_req_i.wstrb);
        OffLenLo: staged_d.len[31:0] = merge(staged_q.len[31:0], reg_req_i.wdata, reg_req_i.wstrb);
        OffLenHi: staged_d.len[63:32] = merge(staged_q.len[63:32], reg_req_i.wdata, reg_req_i.wstrb);
        default: ;
      endcase
    end
  end

  // a doorbell write completes the cycle its transfer enters the queue
  assign push = access && write && (off == OffDoorbell) && !full;

  always_comb begin
    reg_rsp_o = '0;
    reg_rsp_o.ready = !(access && write && (off == OffDoorbell) && full);
    reg_rsp_o.error = 1'b0;
    if (access && !write) begin
      case (off)
        OffSrcLo: reg_rsp_o.rdata = staged_q.src[31:0];
        OffSrcHi: reg_rsp_o.rdata = staged_q.src[63:32];
        OffDstLo: reg_rsp_o.rdata = staged_q.dst[31:0];
        OffDstHi: reg_rsp_o.rdata = staged_q.dst[63:32];
        OffLenLo: reg_rsp_o.rdata = staged_q.len[31:0];
        OffLenHi: reg_rsp_o.rdata = staged_q.len[63:32];
        OffCtrl: reg_rsp_o.rdata = 32'd1;
        OffCount: reg_rsp_o.rdata = full ? 32'(Depth) : 32'(usage);
        OffIssued: reg_rsp_o.rdata = issued_q;
        OffDepth: reg_rsp_o.rdata = 32'(Depth);
        default: reg_rsp_o.rdata = '0;
      endcase
    end
  end

  fifo_v3 #(
      .FALL_THROUGH(1'b0),
      .DEPTH       (Depth),
      .dtype       (entry_t)
  ) i_fifo (
      .clk_i     (clk_i),
      .rst_ni    (rst_ni),
      .flush_i   (1'b0),
      .testmode_i(1'b0),
      .full_o    (full),
      .empty_o   (empty),
      .usage_o   (usage),
      .data_i    (staged_q),
      .push_i    (push),
      .data_o    (head),
      .pop_i     (pop)
  );

  always_comb begin
    req_o = opt_i;
    req_o.src_addr = head.src[$bits(req_o.src_addr)-1:0];
    req_o.dst_addr = head.dst[$bits(req_o.dst_addr)-1:0];
    req_o.length = head.len[$bits(req_o.length)-1:0];
  end
  assign valid_o = !empty;
  assign pop = valid_o && ready_i;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      staged_q <= '0;
      issued_q <= '0;
    end else begin
      staged_q <= staged_d;
      if (pop) issued_q <= issued_q + 32'd1;
    end
  end

endmodule
