// Copyright 2026 KU Leuven.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51
//
// Fanchen Kong <fanchen.kong@kuleuven.be>

// Simulated HBM behind the memory chip (simulation only).
//
// The memchip stands in for an FPGA, and the FPGA has HBM. This model is that HBM as
// its AXI port sees it: an AXI4 slave holding GiB of data, with the latency and the
// channel parallelism of a real HBM controller.
//
// STORAGE lives in C++ (hemaia_hbm_dpi.cc): files are mmap()ed copy-on-write and the
// rest is sparse, so a 16 GiB HBM costs host memory only for what is touched. The
// testharness loads it through load_manifest() -- see README.md.
//
// TIMING is a first-order HBM model, evaluated in picoseconds of simulated time so it
// does not depend on the memchip clock:
//
//   * The HBM is `num_channels` pseudo-channels, each with its own data bus and
//     `num_banks` banks. An address picks its channel by `interleave_bytes`
//     interleaving, then bank and row (row-bank-column inside the channel).
//   * Every beat of a burst is scheduled on its channel when the burst is accepted
//     (reads) or when its data arrives (writes), first come first served:
//       command = max(arrival + t_ctrl/2, bank free), pushed past a refresh;
//       + t_rp + t_rcd for a row conflict, + t_rcd for a closed row;
//       data    = max(command + t_cl (t_cwl), channel data bus free), for one beat
//                 time (bus bytes / channel bandwidth);
//       done    = data end + t_ctrl/2.
//     Bursts on different channels, and on different banks of one channel, overlap;
//     bursts on one channel share its data bus.
//   * Refresh blocks a channel for t_rfc every t_refi (staggered across channels) and
//     closes its rows.
//
// The AXI side is a controller that reorders across IDs, never within one: responses
// leave in order of completion, but a burst waits for every older burst with the same
// ID. A read burst, once started, is not interleaved with another. `max_reads` and
// `max_writes` bursts in flight, then back-pressure. An access outside
// [base, base + size) is answered with SLVERR and touches nothing. No atomics (the
// memchip xbar has none to send).
//
// Plusargs: +hbm_trace prints every burst. A statistics summary prints at the end of
// the simulation if the HBM saw any traffic.

module hemaia_hbm_model import hemaia_hbm_pkg::*; #(
  parameter hbm_cfg_t    Cfg            = DefaultHbmCfg,
  parameter int unsigned AddrWidth      = 48,
  parameter int unsigned DataWidth      = 512,
  parameter int unsigned IdWidth        = 6,
  // Address bits below the chip id. The bits above are ignored.
  parameter int unsigned LocalAddrWidth = 40,
  parameter type         axi_req_t      = logic,
  parameter type         axi_rsp_t      = logic
) (
  input  logic     clk_i,
  input  logic     rst_ni,
  input  axi_req_t axi_req_i,
  output axi_rsp_t axi_rsp_o
);

  timeunit 1ns;
  timeprecision 1ps;

  localparam int unsigned StrbWidth = DataWidth / 8;
  localparam int unsigned NumCh     = Cfg.num_channels;
  localparam int unsigned NumBanks  = Cfg.num_banks;
  localparam int unsigned MaxBeats  = 256;
  // One bus beat on one pseudo-channel.
  localparam longint      TBeatPs   = (longint'(StrbWidth) * 1000000 + Cfg.channel_mbps - 1) /
                                      Cfg.channel_mbps;
  localparam longint      TReqPs    = Cfg.t_ctrl_ps / 2;
  localparam longint      TRspPs    = Cfg.t_ctrl_ps - TReqPs;
  localparam longint unsigned ChannelBytes = Cfg.size / NumCh;

  typedef logic [IdWidth-1:0]   id_t;
  typedef logic [AddrWidth-1:0] addr_t;
  typedef logic [DataWidth-1:0] data_t;

  // ---------------------------------------------------------------------------------
  // Storage (hemaia_hbm_dpi.cc)
  // ---------------------------------------------------------------------------------

  import "DPI-C" function chandle hemaia_hbm_create(input string name,
                                                    input longint unsigned size);
  import "DPI-C" function void hemaia_hbm_clear(input chandle h);
  import "DPI-C" function int hemaia_hbm_load_file(input chandle h, input string path,
                                                   input longint unsigned off);
  import "DPI-C" function int hemaia_hbm_load_manifest(input chandle h, input string path,
                                                       input int chip_id);
  import "DPI-C" function void hemaia_hbm_read(input chandle h, input longint unsigned off,
                                               input int unsigned len,
                                               output bit [511:0] data);
  import "DPI-C" function void hemaia_hbm_write(input chandle h, input longint unsigned off,
                                                input int unsigned len,
                                                input bit [511:0] data,
                                                input bit [63:0] strb);
  import "DPI-C" function int hemaia_hbm_dump(input chandle h, input string path,
                                              input longint unsigned off,
                                              input longint unsigned len);
  import "DPI-C" function longint unsigned hemaia_hbm_sparse_bytes(input chandle h);

  string  inst_name;
  chandle store;

  initial inst_name = $sformatf("%m");

  function automatic chandle get_store();
    if (store == null) store = hemaia_hbm_create(inst_name, Cfg.size);
    return store;
  endfunction

  // Loader interface, called hierarchically by the testharness (load_binary.sv).

  // Forget every byte: unmap all files, drop all written pages.
  function automatic void clear();
    hemaia_hbm_clear(get_store());
  endfunction

  // Map one binary file at `offset` bytes above the HBM base.
  function automatic void load_file(string path, longint unsigned offset);
    if (hemaia_hbm_load_file(get_store(), path, offset) != 0)
      $fatal(1, "[HBM %s] failed to load %s", inst_name, path);
  endfunction

  // Load every file `manifest` lists for the memchip `chip_id`. A missing manifest
  // leaves the HBM as it is.
  function automatic void load_manifest(string manifest, int chip_id);
    int n = hemaia_hbm_load_manifest(get_store(), manifest, chip_id);
    if (n == -1)
      $display("[HBM %s] no %s; HBM starts zeroed", inst_name, manifest);
    else if (n < 0)
      $fatal(1, "[HBM %s] failed to load %s", inst_name, manifest);
    else
      $display("[HBM %s] loaded %0d file(s) from %s", inst_name, n, manifest);
  endfunction

  // Write `len` bytes starting `offset` bytes above the HBM base to `path`.
  function automatic void dump(string path, longint unsigned offset, longint unsigned len);
    if (hemaia_hbm_dump(get_store(), path, offset, len) != 0)
      $error("[HBM %s] failed to dump to %s", inst_name, path);
  endfunction

  // ---------------------------------------------------------------------------------
  // Timing model
  // ---------------------------------------------------------------------------------

  longint          bus_free  [NumCh];            // data bus free from (ps)
  longint          bank_free [NumCh][NumBanks];  // bank takes a column command from (ps)
  longint          open_row  [NumCh][NumBanks];  // -1: closed
  longint          ref_epoch [NumCh];            // last refresh interval seen

  // Statistics.
  longint unsigned st_rd_bursts, st_rd_beats, st_wr_bursts, st_wr_beats, st_err_bursts;
  longint unsigned st_row_hit, st_row_closed, st_row_conflict;
  longint unsigned st_rd_lat_sum, st_rd_lat_max, st_wr_lat_sum, st_wr_lat_max;
  longint unsigned st_ref_stall;
  longint unsigned st_ch_beats [NumCh];
  longint          st_first_ps, st_last_ps;

  function automatic longint now_ps();
    return longint'($realtime / 1ps);
  endfunction

  function automatic longint max2(longint a, longint b);
    return (a > b) ? a : b;
  endfunction

  // Channel, bank and row of the byte `off` bytes above the base.
  function automatic void decode(input longint unsigned off, output int unsigned ch,
                                 output int unsigned bank, output longint row);
    longint unsigned in_ch;
    if (Cfg.interleave_bytes == 0) begin
      ch    = off / ChannelBytes;
      in_ch = off % ChannelBytes;
    end else begin
      longint unsigned chunk = off / Cfg.interleave_bytes;
      ch    = chunk % NumCh;
      in_ch = (chunk / NumCh) * Cfg.interleave_bytes + off % Cfg.interleave_bytes;
    end
    bank = (in_ch / Cfg.row_bytes) % NumBanks;
    row  = in_ch / (longint'(Cfg.row_bytes) * NumBanks);
  endfunction

  // Earliest time >= t at which channel `ch` is not refreshing. Crossing a refresh
  // closes every row of the channel.
  function automatic longint refresh(int unsigned ch, longint t);
    longint phase, epoch, start;
    if (Cfg.t_refi_ps == 0) return t;
    phase = longint'(ch) * (Cfg.t_refi_ps / NumCh);
    if (t < phase) return t;
    epoch = (t - phase) / Cfg.t_refi_ps;
    start = phase + epoch * Cfg.t_refi_ps;
    if (t < start + Cfg.t_rfc_ps) begin
      st_ref_stall += start + Cfg.t_rfc_ps - t;
      t = start + Cfg.t_rfc_ps;
    end
    if (epoch != ref_epoch[ch]) begin
      ref_epoch[ch] = epoch;
      for (int b = 0; b < NumBanks; b++) open_row[ch][b] = -1;
    end
    return t;
  endfunction

  // Schedule one bus beat at `off`, arriving at `t_arrive`. Returns when it is done:
  // read data available, or write committed and acknowledged.
  function automatic longint schedule(longint unsigned off, bit is_write, longint t_arrive);
    int unsigned ch, bank;
    longint row, t, t_data;
    decode(off, ch, bank, row);
    t = max2(t_arrive + TReqPs, bank_free[ch][bank]);
    t = refresh(ch, t);
    if (open_row[ch][bank] == row) begin
      st_row_hit++;
    end else begin
      if (open_row[ch][bank] < 0) st_row_closed++;
      else begin
        st_row_conflict++;
        t += Cfg.t_rp_ps;
      end
      t += Cfg.t_rcd_ps;
      open_row[ch][bank] = row;
    end
    t_data = max2(t + (is_write ? Cfg.t_cwl_ps : Cfg.t_cl_ps), bus_free[ch]);
    bus_free[ch]        = t_data + TBeatPs;
    bank_free[ch][bank] = t + TBeatPs;
    st_ch_beats[ch]++;
    return t_data + TBeatPs + TRspPs;
  endfunction

  // ---------------------------------------------------------------------------------
  // AXI
  // ---------------------------------------------------------------------------------

  typedef struct {
    longint unsigned seq;        // acceptance order, across reads and writes
    id_t             id;
    addr_t           addr;
    axi_pkg::len_t   len;
    axi_pkg::size_t  size;
    axi_pkg::burst_t burst;
    bit              err;        // outside the HBM
    longint          t_accept;
    int unsigned     beats;      // read: beats sent; write: beats received
    longint          t_ready [MaxBeats];  // read: beat i can go out
    longint          t_done;     // write: B can go out once all beats arrived
  } txn_t;

  txn_t            rd_q [$];     // in acceptance order
  txn_t            wr_q [$];     // in acceptance order
  longint unsigned seq_cnt;
  bit              r_busy;       // a read burst is part-way out
  longint unsigned r_seq, b_seq; // burst on R / B
  bit              trace;

  // Registered outputs.
  logic            aw_ready_q, w_ready_q, ar_ready_q;
  logic            r_valid_q, r_last_q, b_valid_q;
  id_t             r_id_q, b_id_q;
  axi_pkg::resp_t  r_resp_q, b_resp_q;
  data_t           r_data_q;

  always_comb begin
    axi_rsp_o          = '0;
    axi_rsp_o.aw_ready = aw_ready_q;
    axi_rsp_o.w_ready  = w_ready_q;
    axi_rsp_o.ar_ready = ar_ready_q;
    axi_rsp_o.b_valid  = b_valid_q;
    axi_rsp_o.b.id     = b_id_q;
    axi_rsp_o.b.resp   = b_resp_q;
    axi_rsp_o.r_valid  = r_valid_q;
    axi_rsp_o.r.id     = r_id_q;
    axi_rsp_o.r.data   = r_data_q;
    axi_rsp_o.r.resp   = r_resp_q;
    axi_rsp_o.r.last   = r_last_q;
  end

  // Byte offset above the base of the bus line holding beat `i` of a burst, or -1
  // outside the HBM.
  function automatic longint line_off(addr_t addr, axi_pkg::len_t len, axi_pkg::size_t size,
                                      axi_pkg::burst_t burst, int unsigned i);
    axi_pkg::largest_addr_t a = axi_pkg::beat_addr(addr, size, len, burst, i);
    longint unsigned local_a  = a[LocalAddrWidth-1:0];
    longint unsigned line     = (local_a / StrbWidth) * StrbWidth;
    if (line < Cfg.base || line - Cfg.base >= Cfg.size) return -1;
    return line - Cfg.base;
  endfunction

  function automatic longint beat_off(int unsigned i_txn, bit is_write, int unsigned beat);
    if (is_write)
      return line_off(wr_q[i_txn].addr, wr_q[i_txn].len, wr_q[i_txn].size, wr_q[i_txn].burst, beat);
    return line_off(rd_q[i_txn].addr, rd_q[i_txn].len, rd_q[i_txn].size, rd_q[i_txn].burst, beat);
  endfunction

  function automatic txn_t new_txn(id_t id, addr_t addr, axi_pkg::len_t len,
                                   axi_pkg::size_t size, axi_pkg::burst_t burst, longint now);
    txn_t t;
    t.seq      = seq_cnt++;
    t.id       = id;
    t.addr     = addr;
    t.len      = len;
    t.size     = size;
    t.burst    = burst;
    t.t_accept = now;
    t.beats    = 0;
    t.t_done   = now;
    t.err      = 1'b0;
    for (int unsigned i = 0; i <= len; i++)
      if (line_off(addr, len, size, burst, i) < 0) t.err = 1'b1;
    if (st_first_ps < 0) st_first_ps = now;
    if (t.err) begin
      st_err_bursts++;
      $warning("[HBM %s] burst at 0x%0h (len %0d) is outside [0x%0h, 0x%0h): SLVERR",
               inst_name, addr, len, Cfg.base, Cfg.base + Cfg.size);
    end
    return t;
  endfunction

  function automatic void accept_read(longint now);
    txn_t t = new_txn(axi_req_i.ar.id, axi_req_i.ar.addr, axi_req_i.ar.len,
                      axi_req_i.ar.size, axi_req_i.ar.burst, now);
    for (int unsigned i = 0; i <= t.len; i++) begin
      t.t_ready[i] = t.err ? now + Cfg.t_ctrl_ps :
                     schedule(line_off(t.addr, t.len, t.size, t.burst, i), 1'b0, now);
      // Beats leave in order.
      if (i > 0) t.t_ready[i] = max2(t.t_ready[i], t.t_ready[i-1]);
    end
    rd_q.push_back(t);
    st_rd_bursts++;
    if (trace)
      $display("[HBM %s] %0t AR id=%0h addr=0x%0h len=%0d size=%0d -> first beat in %0d ps",
               inst_name, $realtime, t.id, t.addr, t.len, t.size, t.t_ready[0] - now);
  endfunction

  function automatic void accept_write(longint now);
    if (axi_req_i.aw.atop != '0)
      $error("[HBM %s] atomic 0x%0h at 0x%0h is not supported", inst_name,
             axi_req_i.aw.atop, axi_req_i.aw.addr);
    wr_q.push_back(new_txn(axi_req_i.aw.id, axi_req_i.aw.addr, axi_req_i.aw.len,
                           axi_req_i.aw.size, axi_req_i.aw.burst, now));
    st_wr_bursts++;
    if (trace)
      $display("[HBM %s] %0t AW id=%0h addr=0x%0h len=%0d size=%0d", inst_name, $realtime,
               axi_req_i.aw.id, axi_req_i.aw.addr, axi_req_i.aw.len, axi_req_i.aw.size);
  endfunction

  // The oldest write burst still missing data, or -1. W beats follow AW order.
  function automatic int wr_awaiting_data();
    foreach (wr_q[i]) if (wr_q[i].beats <= wr_q[i].len) return i;
    return -1;
  endfunction

  function automatic void accept_write_beat(longint now);
    int     i = wr_awaiting_data();
    int unsigned beat;
    longint off;
    bit [511:0] data = '0;
    bit [63:0]  strb = '0;
    beat = wr_q[i].beats;
    if (axi_req_i.w.last != (beat == wr_q[i].len))
      $error("[HBM %s] W last=%0b on beat %0d of a %0d-beat burst", inst_name,
             axi_req_i.w.last, beat, wr_q[i].len + 1);
    if (wr_q[i].err) begin
      wr_q[i].t_done = max2(wr_q[i].t_done, now + Cfg.t_ctrl_ps);
    end else begin
      off = beat_off(i, 1'b1, beat);
      data[DataWidth-1:0] = axi_req_i.w.data;
      strb[StrbWidth-1:0] = axi_req_i.w.strb;
      hemaia_hbm_write(store, off, StrbWidth, data, strb);
      wr_q[i].t_done = max2(wr_q[i].t_done, schedule(off, 1'b1, now));
    end
    wr_q[i].beats++;
    st_wr_beats++;
  endfunction

  // Can burst `i` of `q` answer, as far as ordering goes? Only if no older burst with
  // the same ID is still in `q`.
  function automatic bit head_of_id(ref txn_t q [$], input int i);
    for (int j = 0; j < i; j++) if (q[j].id == q[i].id) return 1'b0;
    return 1'b1;
  endfunction

  function automatic int find_seq(ref txn_t q [$], input longint unsigned seq);
    foreach (q[i]) if (q[i].seq == seq) return i;
    return -1;
  endfunction

  function automatic void retire_read_beat(longint now);
    int i = find_seq(rd_q, r_seq);
    rd_q[i].beats++;
    st_rd_beats++;
    if (rd_q[i].beats > rd_q[i].len) begin
      st_rd_lat_sum += now - rd_q[i].t_accept;
      if (now - rd_q[i].t_accept > st_rd_lat_max) st_rd_lat_max = now - rd_q[i].t_accept;
      if (trace)
        $display("[HBM %s] %0t R  id=%0h addr=0x%0h done after %0d ps", inst_name, $realtime,
                 rd_q[i].id, rd_q[i].addr, now - rd_q[i].t_accept);
      rd_q.delete(i);
      r_busy = 1'b0;
    end
    st_last_ps = now;
  endfunction

  function automatic void retire_write(longint now);
    int i = find_seq(wr_q, b_seq);
    st_wr_lat_sum += now - wr_q[i].t_accept;
    if (now - wr_q[i].t_accept > st_wr_lat_max) st_wr_lat_max = now - wr_q[i].t_accept;
    if (trace)
      $display("[HBM %s] %0t B  id=%0h addr=0x%0h done after %0d ps", inst_name, $realtime,
               wr_q[i].id, wr_q[i].addr, now - wr_q[i].t_accept);
    wr_q.delete(i);
    st_last_ps = now;
  endfunction

  // Put the next read beat on R, if one is ready.
  task automatic present_read(longint now);
    int i = -1;
    bit [511:0] line = '0;
    longint off;
    if (r_busy) begin
      i = find_seq(rd_q, r_seq);
      if (rd_q[i].t_ready[rd_q[i].beats] > now) i = -1;
    end else begin
      // Whichever burst is ready first, among those allowed to answer.
      foreach (rd_q[j])
        if (rd_q[j].t_ready[0] <= now && head_of_id(rd_q, j) &&
            (i < 0 || rd_q[j].t_ready[0] < rd_q[i].t_ready[0]))
          i = j;
      if (i >= 0) begin
        r_busy = 1'b1;
        r_seq  = rd_q[i].seq;
      end
    end
    if (i < 0) begin
      r_valid_q <= 1'b0;
      return;
    end
    if (!rd_q[i].err) begin
      off = beat_off(i, 1'b0, rd_q[i].beats);
      hemaia_hbm_read(store, off, StrbWidth, line);
    end
    r_valid_q <= 1'b1;
    r_id_q    <= rd_q[i].id;
    r_data_q  <= line[DataWidth-1:0];
    r_resp_q  <= rd_q[i].err ? axi_pkg::RESP_SLVERR : axi_pkg::RESP_OKAY;
    r_last_q  <= (rd_q[i].beats == rd_q[i].len);
  endtask

  // Put the next write response on B, if one is ready.
  task automatic present_b(longint now);
    int i = -1;
    foreach (wr_q[j])
      if (wr_q[j].beats > wr_q[j].len && wr_q[j].t_done <= now && head_of_id(wr_q, j) &&
          (i < 0 || wr_q[j].t_done < wr_q[i].t_done))
        i = j;
    if (i < 0) begin
      b_valid_q <= 1'b0;
      return;
    end
    b_seq     = wr_q[i].seq;
    b_valid_q <= 1'b1;
    b_id_q    <= wr_q[i].id;
    b_resp_q  <= wr_q[i].err ? axi_pkg::RESP_SLVERR : axi_pkg::RESP_OKAY;
  endtask

  // AXI handshakes are sampled at the clock edge and every output is registered, so
  // the model is race-free against the flops that drive it.
  always @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      rd_q.delete();
      wr_q.delete();
      r_busy     = 1'b0;
      aw_ready_q <= 1'b0;
      w_ready_q  <= 1'b0;
      ar_ready_q <= 1'b0;
      r_valid_q  <= 1'b0;
      r_last_q   <= 1'b0;
      r_id_q     <= '0;
      r_resp_q   <= '0;
      r_data_q   <= '0;
      b_valid_q  <= 1'b0;
      b_id_q     <= '0;
      b_resp_q   <= '0;
      for (int c = 0; c < NumCh; c++) begin
        bus_free[c]  = 0;
        ref_epoch[c] = -1;
        for (int b = 0; b < NumBanks; b++) begin
          bank_free[c][b] = 0;
          open_row[c][b]  = -1;
        end
      end
    end else begin
      longint now;
      bit     r_free, b_free;
      now    = now_ps();
      r_free = !r_valid_q || axi_req_i.r_ready;
      b_free = !b_valid_q || axi_req_i.b_ready;
      // Handshakes of the cycle that just ended. W before AW: a beat accepted now was
      // granted for a burst that was already waiting.
      if (axi_req_i.ar_valid && ar_ready_q) accept_read(now);
      if (axi_req_i.w_valid  && w_ready_q)  accept_write_beat(now);
      if (axi_req_i.aw_valid && aw_ready_q) accept_write(now);
      if (r_valid_q && axi_req_i.r_ready)   retire_read_beat(now);
      if (b_valid_q && axi_req_i.b_ready)   retire_write(now);
      // Outputs for the next cycle.
      if (r_free) present_read(now);
      if (b_free) present_b(now);
      ar_ready_q <= rd_q.size() < Cfg.max_reads;
      aw_ready_q <= wr_q.size() < Cfg.max_writes;
      w_ready_q  <= wr_awaiting_data() >= 0;
    end
  end

  // ---------------------------------------------------------------------------------
  // Setup, checks, statistics
  // ---------------------------------------------------------------------------------

  initial begin
    trace       = $test$plusargs("hbm_trace");
    seq_cnt     = 0;
    st_first_ps = -1;
    st_last_ps  = 0;
    void'(get_store());
    if (DataWidth > 512 || DataWidth < 8 || (DataWidth & (DataWidth - 1)) != 0)
      $fatal(1, "[HBM %s] DataWidth %0d: must be a power of two <= 512", inst_name, DataWidth);
    if ($bits(axi_req_i.w.data) != DataWidth || $bits(axi_req_i.ar.id) != IdWidth ||
        $bits(axi_req_i.ar.addr) != AddrWidth)
      $fatal(1, "[HBM %s] AXI types do not match DataWidth/IdWidth/AddrWidth", inst_name);
    if (Cfg.base % StrbWidth != 0 || Cfg.size % StrbWidth != 0 || Cfg.size == 0)
      $fatal(1, "[HBM %s] base and size must be non-zero multiples of the bus width",
             inst_name);
    if (Cfg.base + Cfg.size > (64'd1 << LocalAddrWidth))
      $fatal(1, "[HBM %s] [0x%0h, 0x%0h) does not fit %0d local address bits", inst_name,
             Cfg.base, Cfg.base + Cfg.size, LocalAddrWidth);
    if (NumCh == 0 || NumBanks == 0 || Cfg.row_bytes == 0 || Cfg.channel_mbps == 0 ||
        Cfg.max_reads == 0 || Cfg.max_writes == 0)
      $fatal(1, "[HBM %s] channels, banks, row size, bandwidth and queues must be non-zero",
             inst_name);
    if (Cfg.interleave_bytes == 0 && Cfg.size % NumCh != 0)
      $fatal(1, "[HBM %s] size must split evenly over %0d channels", inst_name, NumCh);
    $display("[HBM %s] %0d MiB at 0x%0h, %0d pseudo-channels x %0d banks, %0d B interleave,",
             inst_name, Cfg.size >> 20, Cfg.base, NumCh, NumBanks, Cfg.interleave_bytes);
    $display("[HBM %s] %0d MB/s per channel, row hit %0d ps / closed %0d ps / conflict %0d ps",
             inst_name, Cfg.channel_mbps, Cfg.t_ctrl_ps + Cfg.t_cl_ps + TBeatPs,
             Cfg.t_ctrl_ps + Cfg.t_rcd_ps + Cfg.t_cl_ps + TBeatPs,
             Cfg.t_ctrl_ps + Cfg.t_rp_ps + Cfg.t_rcd_ps + Cfg.t_cl_ps + TBeatPs);
  end

  final begin
    // Assigned, not initialised: an initialiser in a static block runs at time 0.
    longint unsigned beats, rows, ch_min, ch_max;
    longint          window;
    beats  = st_rd_beats + st_wr_beats;
    rows   = st_row_hit + st_row_closed + st_row_conflict;
    ch_min = '1;
    ch_max = 0;
    window = st_last_ps - st_first_ps;
    if (st_rd_bursts + st_wr_bursts != 0) begin
      for (int c = 0; c < NumCh; c++) begin
        if (st_ch_beats[c] < ch_min) ch_min = st_ch_beats[c];
        if (st_ch_beats[c] > ch_max) ch_max = st_ch_beats[c];
      end
      $display("[HBM %s] ---- statistics ----", inst_name);
      $display("[HBM %s] reads : %0d bursts, %0d beats, latency avg %0d ps max %0d ps",
               inst_name, st_rd_bursts, st_rd_beats,
               st_rd_bursts ? st_rd_lat_sum / st_rd_bursts : 0, st_rd_lat_max);
      $display("[HBM %s] writes: %0d bursts, %0d beats, latency avg %0d ps max %0d ps",
               inst_name, st_wr_bursts, st_wr_beats,
               st_wr_bursts ? st_wr_lat_sum / st_wr_bursts : 0, st_wr_lat_max);
      $display("[HBM %s] rows  : %0d hit, %0d closed, %0d conflict (%0d%% hit)", inst_name,
               st_row_hit, st_row_closed, st_row_conflict,
               rows ? st_row_hit * 100 / rows : 0);
      $display("[HBM %s] beats per channel: min %0d max %0d; refresh stalls %0d ps; %0d SLVERR",
               inst_name, ch_min, ch_max, st_ref_stall, st_err_bursts);
      if (window > 0)
        $display("[HBM %s] %0d bytes in %0d ns = %0d MB/s", inst_name, beats * StrbWidth,
                 window / 1000, beats * StrbWidth * 1000000 / window);
      $display("[HBM %s] host memory for written pages: %0d KiB", inst_name,
               hemaia_hbm_sparse_bytes(store) >> 10);
    end
  end

endmodule
