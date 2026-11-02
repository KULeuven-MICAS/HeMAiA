// Copyright 2026 KU Leuven.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51
//
// reduce_simd_model: a behavioural model of the cluster SIMD's two-operand FP16 ADD
// (snax_split_cluster_simd, StreamElementwise in PAIRS mode, as snax-coll-reduce.c's simd_add
// programs it), driving real TCDM ports so its accesses are counted and contend with the DMA.
//
// Rate, from the RTL (snax_split_cluster.hjson: reader num_channel 4 x 64 b = one 32 B beat per
// cycle; SimdStreamElementwise.scala: "one beat per cycle ... the stream alternates B, x, B, x,
// and each pair emits one beat"): one 32 B input beat per cycle, alternating the two operands,
// one 32 B output beat (16 FP16 sums) per pair, i.e. 8 FP16 results per cycle. Reads use four
// 64-bit ports (one beat), writes four more; a beat waits until all four words are granted.
// Not modelled: the SIMD's programming (CSR writes by its core), its stream pipeline fill, its
// reader/writer buffer depths (writes keep up here).
//
// The arithmetic: every lane is an FP16 small integer (the testbench's payload); a lane that is
// not an exact integer below 2048 flags `err_o` (a wrong or unwritten operand cannot pass).

module reduce_simd_model
  import reduce_pkg::*;
(
  input  logic                   clk_i,
  input  logic                   rst_ni,
  input  logic                   start_i,      // pulse; operands latched
  input  logic [TcdmAddrWidth-1:0] a_i,         // TCDM byte addresses
  input  logic [TcdmAddrWidth-1:0] b_i,
  input  logic [TcdmAddrWidth-1:0] o_i,
  input  int unsigned            bytes_i,
  output logic                   busy_o,
  output logic                   err_o,
  output tcdm_req_t [7:0]        req_o,        // 0..3 read, 4..7 write
  input  tcdm_rsp_t [7:0]        rsp_i
);

  function automatic int fp16_int(input logic [15:0] h);
    int unsigned e, m;
    if (h == 16'h0000) return 0;
    if (h[15]) return -1;
    e = h[14:10];
    m = {1'b1, h[9:0]};
    if (e < 15 || e > 25) return -1;
    if ((m & ((1 << (25 - e)) - 1)) != 0) return -1;   // not an integer
    return int'(m >> (25 - e));
  endfunction

  function automatic logic [15:0] fp16_of(input int unsigned n);
    int unsigned e;
    if (n == 0) return 16'h0000;
    e = 0;
    while ((n >> (e + 1)) != 0) e++;
    return {1'b0, 5'(e + 15), 10'((n << (10 - e)) & 32'h3FF)};
  endfunction

  logic [TcdmAddrWidth-1:0] a_q, b_q, o_q;
  int unsigned nbeats_q;          // output beats
  int unsigned rd_beat_q;         // next input beat: 2 * index + operand
  logic [3:0]  rd_pend_q;         // words of rd_beat_q still to be granted
  int unsigned rd_beat_inflight[4];
  logic [3:0]  rd_inflight_q;
  logic [63:0] beat_a [4];        // operand A of the current pair
  logic [63:0] beat_buf [4];      // words of the beat being collected
  logic [3:0]  got_q;
  int unsigned got_beat_q;
  // write queue of finished output beats
  logic [63:0] wq_data [$];
  int unsigned wq_idx [$];
  logic [3:0]  wr_pend_q;
  logic        wr_cur_valid_q;    // the head of the write queue, registered (the output stage)
  int unsigned wr_cur_idx_q;
  logic [63:0] wr_cur_data_q [4];
  int unsigned wr_done_q;
  logic        busy_q, err_q;

  assign busy_o = busy_q;
  assign err_o  = err_q;

  // requests
  always_comb begin
    req_o = '0;
    for (int k = 0; k < 4; k++) begin
      req_o[k].q.addr  = (rd_beat_q[0] ? b_q : a_q) + TcdmAddrWidth'((rd_beat_q >> 1) * 32 + k * 8);
      req_o[k].q.write = 1'b0;
      req_o[k].q.strb  = '1;
      req_o[k].q_valid = busy_q && (rd_beat_q < 2 * nbeats_q) && rd_pend_q[k];
    end
    for (int k = 0; k < 4; k++) begin
      req_o[4+k].q.addr  = o_q + TcdmAddrWidth'(wr_cur_idx_q * 32 + k * 8);
      req_o[4+k].q.write = 1'b1;
      req_o[4+k].q.data  = wr_cur_data_q[k];
      req_o[4+k].q.strb  = '1;
      req_o[4+k].q_valid = busy_q && wr_cur_valid_q && wr_pend_q[k];
    end
  end

  always @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      busy_q <= 1'b0;
      err_q  <= 1'b0;
    end else begin
      if (start_i && !busy_q) begin
        a_q        <= a_i;
        b_q        <= b_i;
        o_q        <= o_i;
        nbeats_q   <= bytes_i / 32;
        rd_beat_q  <= 0;
        rd_pend_q  <= 4'hF;
        rd_inflight_q <= '0;
        got_q      <= '0;
        got_beat_q <= 0;
        wr_pend_q  <= 4'hF;
        wr_done_q  <= 0;
        wr_cur_valid_q <= 1'b0;
        wq_data.delete();
        wq_idx.delete();
        busy_q     <= (bytes_i >= 32);
      end else if (busy_q) begin
        automatic logic [3:0] pend = rd_pend_q;
        automatic logic [3:0] got = got_q;
        automatic logic [3:0] wpend = wr_pend_q;
        // read responses (one cycle after the grant)
        for (int k = 0; k < 4; k++) begin
          if (rd_inflight_q[k] && rsp_i[k].p_valid) begin
            beat_buf[k] = rsp_i[k].p.data;
            got[k] = 1'b1;
          end
        end
        rd_inflight_q <= '0;
        if (got == 4'hF) begin
          if (got_beat_q[0] == 1'b0) begin
            for (int k = 0; k < 4; k++) beat_a[k] = beat_buf[k];
          end else begin
            for (int k = 0; k < 4; k++) begin
              automatic logic [63:0] w;
              for (int l = 0; l < 4; l++) begin
                automatic int x = fp16_int(beat_a[k][16*l+:16]);
                automatic int y = fp16_int(beat_buf[k][16*l+:16]);
                if (x < 0 || y < 0 || x + y >= 2048) err_q <= 1'b1;
                w[16*l+:16] = fp16_of((x < 0 || y < 0) ? 0 : x + y);
              end
              wq_data.push_back(w);
            end
            wq_idx.push_back(got_beat_q >> 1);
          end
          got = '0;
          got_beat_q <= got_beat_q + 1;
        end
        got_q <= got;
        // read grants
        for (int k = 0; k < 4; k++) begin
          if (req_o[k].q_valid && rsp_i[k].q_ready) begin
            pend[k] = 1'b0;
            rd_inflight_q[k] <= 1'b1;
          end
        end
        if (pend == 4'h0 && rd_beat_q < 2 * nbeats_q) begin
          rd_beat_q <= rd_beat_q + 1;
          pend = 4'hF;
        end
        rd_pend_q <= pend;
        // write grants: the registered head beat; when all four words are in, take the next
        begin : wr_stage
          automatic bit fin = 1'b0;
          if (wr_cur_valid_q) begin
            for (int k = 0; k < 4; k++)
              if (req_o[4+k].q_valid && rsp_i[4+k].q_ready) wpend[k] = 1'b0;
            if (wpend == 4'h0) begin
              fin = 1'b1;
              wr_done_q <= wr_done_q + 1;
              wpend = 4'hF;
              if (wr_done_q + 1 == nbeats_q) busy_q <= 1'b0;
            end
          end
          if (!wr_cur_valid_q || fin) begin
          if (wq_idx.size() > 0) begin
            wr_cur_valid_q <= 1'b1;
            wr_cur_idx_q   <= wq_idx.pop_front();
            for (int k = 0; k < 4; k++) wr_cur_data_q[k] <= wq_data.pop_front();
            wpend = 4'hF;
            end else begin
              wr_cur_valid_q <= 1'b0;
            end
          end
        end
        wr_pend_q <= wpend;
      end
    end
  end

endmodule
