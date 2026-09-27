// Copyright 2026 KU Leuven.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51
//
// Fanchen Kong <fanchen.kong@kuleuven.be>

// Unit test of hemaia_hbm_model. Run with run.sh, which also builds the data files.
//
//   1. Loading: files from a manifest land at their offsets, entries for another
//      memchip do not, bytes past the end of a file read zero.
//   2. Copy-on-write: a write over a loaded file is read back (run.sh checks the file
//      on disk did not change).
//   3. SLVERR outside [base, base + size), and nothing is touched.
//   4. Latency: row closed / hit / conflict, and a stall behind refresh.
//   5. Ordering: reordered across IDs, in order within one ID.
//   6. Channels: the same traffic spread over 16 channels vs. on one channel.
//   7. Random traffic (pulp axi_rand_master), every read beat checked against a
//      shadow copy of what was written (the checker at the bottom).
//
// Prints "HBM TEST PASSED" or "HBM TEST FAILED".

`include "axi/typedef.svh"
`include "axi/assign.svh"

module tb_hemaia_hbm_model;
  import hemaia_hbm_pkg::*;

  localparam time CyclTime = 12ns;  // the memchip's clk_host with same_memchip_speed
  localparam time ApplTime = 2ns;
  localparam time TestTime = 10ns;

  localparam int unsigned AW = 48;
  localparam int unsigned DW = 512;
  localparam int unsigned IW = 6;
  localparam int unsigned UW = 1;

  localparam logic [7:0]      ChipId   = 8'h20;
  localparam longint unsigned ChipBase = longint'(ChipId) << 40;
  localparam longint unsigned HbmBase  = DefaultHbmCfg.base;

  // Slow channels (64 ns per beat, 5x slower than the AXI port), no refresh: makes the
  // difference between one channel and many visible.
  localparam hbm_cfg_t SlowCfg = '{
    base:             64'h1_0000_0000,
    size:             64'h4_0000_0000,
    num_channels:     16,
    num_banks:        16,
    row_bytes:        1024,
    interleave_bytes: 4096,
    channel_mbps:     1000,
    t_ctrl_ps:        90000,
    t_cl_ps:          14000,
    t_cwl_ps:         7000,
    t_rcd_ps:         14000,
    t_rp_ps:          14000,
    t_refi_ps:        0,
    t_rfc_ps:         0,
    max_reads:        64,
    max_writes:       64
  };

  `AXI_TYPEDEF_ALL(axi, logic [AW-1:0], logic [IW-1:0], logic [DW-1:0], logic [DW/8-1:0],
                   logic [UW-1:0])

  typedef axi_test::axi_driver #(
    .AW(AW), .DW(DW), .IW(IW), .UW(UW), .TA(ApplTime), .TT(TestTime)
  ) drv_t;
  // The driver's own specialisations: an equal-valued but differently-typed parameter
  // list would be a different class.
  typedef drv_t::ax_beat_t ax_beat_t;
  typedef drv_t::w_beat_t  w_beat_t;
  typedef drv_t::b_beat_t  b_beat_t;
  typedef drv_t::r_beat_t  r_beat_t;

  typedef axi_test::axi_rand_master #(
    .AW(AW), .DW(DW), .IW(IW), .UW(UW), .TA(ApplTime), .TT(TestTime),
    .MAX_READ_TXNS(32), .MAX_WRITE_TXNS(32),
    .AX_MIN_WAIT_CYCLES(0), .AX_MAX_WAIT_CYCLES(3),
    .W_MIN_WAIT_CYCLES(0), .W_MAX_WAIT_CYCLES(2),
    .RESP_MIN_WAIT_CYCLES(0), .RESP_MAX_WAIT_CYCLES(2),
    .AXI_MAX_BURST_LEN(0), .TRAFFIC_SHAPING(0),
    .AXI_EXCLS(1'b0), .AXI_ATOPS(1'b0),
    .AXI_BURST_FIXED(1'b1), .AXI_BURST_INCR(1'b1), .AXI_BURST_WRAP(1'b1)
  ) rand_master_t;

  logic clk, rst_n;

  clk_rst_gen #(
    .ClkPeriod    ( CyclTime ),
    .RstClkCycles ( 5        )
  ) i_clk_rst_gen (
    .clk_o  ( clk   ),
    .rst_no ( rst_n )
  );

  // Two DUTs: the default HBM, and the slow-channel one for test 6.
  AXI_BUS_DV #(.AXI_ADDR_WIDTH(AW), .AXI_DATA_WIDTH(DW), .AXI_ID_WIDTH(IW),
               .AXI_USER_WIDTH(UW)) dv (clk), dv_slow (clk);
  axi_req_t  req, req_slow;
  axi_resp_t rsp, rsp_slow;
  `AXI_ASSIGN_TO_REQ(req, dv)
  `AXI_ASSIGN_FROM_RESP(dv, rsp)
  `AXI_ASSIGN_TO_REQ(req_slow, dv_slow)
  `AXI_ASSIGN_FROM_RESP(dv_slow, rsp_slow)

  hemaia_hbm_model #(
    .Cfg       ( DefaultHbmCfg ),
    .AddrWidth ( AW            ),
    .DataWidth ( DW            ),
    .IdWidth   ( IW            ),
    .axi_req_t ( axi_req_t     ),
    .axi_rsp_t ( axi_resp_t    )
  ) i_dut (
    .clk_i     ( clk   ),
    .rst_ni    ( rst_n ),
    .axi_req_i ( req   ),
    .axi_rsp_o ( rsp   )
  );

  hemaia_hbm_model #(
    .Cfg       ( SlowCfg    ),
    .AddrWidth ( AW         ),
    .DataWidth ( DW         ),
    .IdWidth   ( IW         ),
    .axi_req_t ( axi_req_t  ),
    .axi_rsp_t ( axi_resp_t )
  ) i_dut_slow (
    .clk_i     ( clk      ),
    .rst_ni    ( rst_n    ),
    .axi_req_i ( req_slow ),
    .axi_rsp_o ( rsp_slow )
  );

  drv_t        drv      = new(dv);
  drv_t        drv_slow = new(dv_slow);
  int unsigned errors = 0;
  int unsigned sb_checked = 0, sb_skipped = 0;  // read beats the checker compared / skipped

  // ---------------------------------------------------------------------------------
  // Helpers
  // ---------------------------------------------------------------------------------

  function automatic logic [AW-1:0] hbm_addr(longint unsigned off);
    return AW'(ChipBase | (HbmBase + off));
  endfunction

  function automatic void check(bit cond, string what);
    if (!cond) begin
      errors++;
      $error("CHECK FAILED: %s", what);
    end
  endfunction

  // 64-bit word `i` of a pattern file written by run.sh.
  function automatic logic [63:0] pattern(logic [15:0] tag, longint unsigned i);
    return {tag, 48'(i)};
  endfunction

  task automatic read_line(drv_t d, logic [AW-1:0] addr, logic [IW-1:0] id,
                           output logic [DW-1:0] data, output axi_pkg::resp_t resp,
                           output time lat);
    ax_beat_t ar = new;
    r_beat_t  r;
    time      t0;
    ar.ax_id    = id;
    ar.ax_addr  = addr;
    ar.ax_len   = 0;
    ar.ax_size  = $clog2(DW/8);
    ar.ax_burst = axi_pkg::BURST_INCR;
    d.send_ar(ar);
    t0 = $time;
    d.recv_r(r);
    lat  = $time - t0;
    data = r.r_data;
    resp = r.r_resp;
  endtask

  task automatic write_line(drv_t d, logic [AW-1:0] addr, logic [DW-1:0] data,
                            logic [DW/8-1:0] strb, output axi_pkg::resp_t resp);
    ax_beat_t aw = new;
    w_beat_t  w  = new;
    b_beat_t  b;
    aw.ax_id    = 0;
    aw.ax_addr  = addr;
    aw.ax_len   = 0;
    aw.ax_size  = $clog2(DW/8);
    aw.ax_burst = axi_pkg::BURST_INCR;
    w.w_data    = data;
    w.w_strb    = strb;
    w.w_last    = 1'b1;
    fork
      d.send_aw(aw);
      d.send_w(w);
    join
    d.recv_b(b);
    resp = b.b_resp;
  endtask

  // Check the line at `off` holds pattern words `first`.. of `tag`.
  task automatic check_pattern(longint unsigned off, logic [15:0] tag, longint unsigned first,
                               string what);
    logic [DW-1:0]  data;
    axi_pkg::resp_t resp;
    time            lat;
    read_line(drv, hbm_addr(off), 0, data, resp, lat);
    check(resp == axi_pkg::RESP_OKAY, {what, ": OKAY"});
    for (int k = 0; k < DW / 64; k++)
      check(data[64*k+:64] == pattern(tag, first + k),
            $sformatf("%s: word %0d = %h, expected %h", what, k, data[64*k+:64],
                      pattern(tag, first + k)));
  endtask

  // Wait until simulated time sits `lo`..`hi` ps into a refresh interval of the default
  // config's channel 0 (whose refresh starts at every multiple of t_refi).
  task automatic align_refresh(longint lo, longint hi);
    longint p;
    do begin
      @(posedge clk);
      p = longint'($realtime / 1ps) % DefaultHbmCfg.t_refi_ps;
    end while (p < lo || p > hi);
    #ApplTime;
  endtask

  // ---------------------------------------------------------------------------------
  // Tests
  // ---------------------------------------------------------------------------------

  task automatic test_loading();
    logic [DW-1:0]  data;
    axi_pkg::resp_t resp;
    time            lat;
    $display("== 1. loading");
    check_pattern(64'h0,              16'hA5A5, 0,    "a.bin first line");
    check_pattern(64'hFFC0,           16'hA5A5, 8184, "a.bin last line");
    check_pattern(64'h2_0000_0000,    16'hB0B0, 0,    "b.bin at 8 GiB");
    check_pattern(64'h1000_2000,      16'hD0D0, 0,    "d.bin (chip=0x20)");
    read_line(drv, hbm_addr(64'h1000_0000), 0, data, resp, lat);
    check(data == '0, "c.bin (chip=0x21) must not load into chip 0x20");
    // e.bin is 100 bytes: bytes 0..99 are 0xEE ^ index, the rest of the line is zero.
    read_line(drv, hbm_addr(64'h3000_0040), 0, data, resp, lat);
    for (int k = 0; k < DW / 8; k++)
      check(data[8*k+:8] == ((k < 100) ? (8'hEE ^ 8'(k)) : 8'h00),
            $sformatf("e.bin byte %0d = %h", k, data[8*k+:8]));
    read_line(drv, hbm_addr(64'h3000_0080), 0, data, resp, lat);
    for (int k = 0; k < 36; k++)
      check(data[8*k+:8] == (8'hEE ^ 8'(k + 64)), $sformatf("e.bin byte %0d", k + 64));
    check(data[DW-1:36*8] == '0, "e.bin tail past end of file is zero");
    // Never-written, unloaded bytes read zero.
    read_line(drv, hbm_addr(64'h1_2345_6780), 0, data, resp, lat);
    check(data == '0 && resp == axi_pkg::RESP_OKAY, "unwritten line reads zero");
  endtask

  task automatic test_cow();
    logic [DW-1:0]   data;
    logic [DW/8-1:0] strb;
    axi_pkg::resp_t  resp;
    time             lat;
    $display("== 2. copy-on-write over a loaded file");
    // Overwrite bytes 8..15 of the line at 0x40 (pattern word 9).
    data = '0;
    data[127:64] = 64'hC0FF_EE00_DEAD_BEEF;
    strb = '0;
    strb[15:8] = '1;
    write_line(drv, hbm_addr(64'h40), data, strb, resp);
    check(resp == axi_pkg::RESP_OKAY, "COW write OKAY");
    read_line(drv, hbm_addr(64'h40), 0, data, resp, lat);
    check(data[63:0]   == pattern(16'hA5A5, 8), "COW: word 8 kept");
    check(data[127:64] == 64'hC0FF_EE00_DEAD_BEEF, "COW: word 9 written");
    check(data[191:128] == pattern(16'hA5A5, 10), "COW: word 10 kept");
    // A sparse write far away, and the last line of the HBM.
    data = {8{64'h1234_5678_9ABC_DEF0}};
    write_line(drv, hbm_addr(DefaultHbmCfg.size - 64), data, '1, resp);
    check(resp == axi_pkg::RESP_OKAY, "write to the last line OKAY");
    read_line(drv, hbm_addr(DefaultHbmCfg.size - 64), 0, data, resp, lat);
    check(data == {8{64'h1234_5678_9ABC_DEF0}}, "last line reads back");
  endtask

  task automatic test_errors();
    logic [DW-1:0]  data;
    axi_pkg::resp_t resp;
    time            lat;
    ax_beat_t       ar = new;
    r_beat_t        r;
    $display("== 3. outside the HBM");
    read_line(drv, hbm_addr(DefaultHbmCfg.size), 0, data, resp, lat);
    check(resp == axi_pkg::RESP_SLVERR, "read at base + size: SLVERR");
    read_line(drv, AW'(ChipBase | 64'h8000_0000), 0, data, resp, lat);
    check(resp == axi_pkg::RESP_SLVERR, "read below base: SLVERR");
    write_line(drv, hbm_addr(DefaultHbmCfg.size), '1, '1, resp);
    check(resp == axi_pkg::RESP_SLVERR, "write at base + size: SLVERR");
    // A two-beat burst whose second beat is past the end fails as a whole.
    ar.ax_addr  = hbm_addr(DefaultHbmCfg.size - 64);
    ar.ax_len   = 1;
    ar.ax_size  = $clog2(DW/8);
    ar.ax_burst = axi_pkg::BURST_INCR;
    drv.send_ar(ar);
    drv.recv_r(r);
    check(r.r_resp == axi_pkg::RESP_SLVERR && !r.r_last, "straddling burst beat 0 SLVERR");
    drv.recv_r(r);
    check(r.r_resp == axi_pkg::RESP_SLVERR && r.r_last, "straddling burst beat 1 SLVERR");
    // The rejected write did not land.
    read_line(drv, hbm_addr(DefaultHbmCfg.size - 64), 0, data, resp, lat);
    check(data == {8{64'h1234_5678_9ABC_DEF0}}, "SLVERR write touched nothing");
  endtask

  // Expected latency of a single beat, as the TB sees it: the model presents the beat on
  // the first edge at or after its ready time, and the handshake is one edge later.
  function automatic bit lat_ok(time lat, longint exp_ps);
    longint l = longint'(lat / 1ps);
    return l >= exp_ps + longint'(CyclTime / 1ps) && l < exp_ps + 2 * longint'(CyclTime / 1ps);
  endfunction

  task automatic test_latency();
    logic [DW-1:0]  data;
    axi_pkg::resp_t resp;
    time            lat;
    longint         beat_ps = 64 * 1000000 / DefaultHbmCfg.channel_mbps;
    longint         hit     = DefaultHbmCfg.t_ctrl_ps + DefaultHbmCfg.t_cl_ps + beat_ps;
    longint         closed  = hit + DefaultHbmCfg.t_rcd_ps;
    longint         miss    = closed + DefaultHbmCfg.t_rp_ps;
    // Channel 0, bank 5 (untouched so far), row 2; then row 3 of the same bank.
    longint unsigned off_a = 64'h12_0400, off_b = 64'h1A_0400;
    $display("== 4. latency (expected: hit %0d, closed %0d, conflict %0d ps)", hit, closed,
             miss);
    align_refresh(1_000_000, 1_500_000);
    read_line(drv, hbm_addr(off_a), 0, data, resp, lat);
    $display("   closed row   %0t", lat);
    check(lat_ok(lat, closed), $sformatf("closed-row latency %0t", lat));
    read_line(drv, hbm_addr(off_a + 64), 0, data, resp, lat);
    $display("   row hit      %0t", lat);
    check(lat_ok(lat, hit), $sformatf("row-hit latency %0t", lat));
    read_line(drv, hbm_addr(off_b), 0, data, resp, lat);
    $display("   row conflict %0t", lat);
    check(lat_ok(lat, miss), $sformatf("row-conflict latency %0t", lat));
    // Refresh: channel 0 is busy for t_rfc from each multiple of t_refi, then its rows
    // are closed. Issue 40 ns into one.
    align_refresh(30_000, 40_000);
    read_line(drv, hbm_addr(off_a), 0, data, resp, lat);
    $display("   in refresh   %0t", lat);
    check(longint'(lat / 1ps) >= DefaultHbmCfg.t_rfc_ps - 60_000 + closed - DefaultHbmCfg.t_ctrl_ps / 2,
          $sformatf("read behind refresh waited only %0t", lat));
  endtask

  task automatic test_ordering();
    ax_beat_t ar1 = new, ar2 = new;
    r_beat_t  r1, r2;
    $display("== 5. ordering");
    // Channel 0 is refreshing; channel 16 (offset 64 KiB) is not.
    ar1.ax_addr  = hbm_addr(64'h10_0000);
    ar2.ax_addr  = hbm_addr(64'h1_0000);
    ar1.ax_size  = $clog2(DW/8);
    ar2.ax_size  = $clog2(DW/8);
    ar1.ax_burst = axi_pkg::BURST_INCR;
    ar2.ax_burst = axi_pkg::BURST_INCR;
    // Different IDs: the fast one overtakes.
    ar1.ax_id = 1;
    ar2.ax_id = 2;
    align_refresh(20_000, 30_000);
    drv.send_ar(ar1);
    drv.send_ar(ar2);
    drv.recv_r(r1);
    drv.recv_r(r2);
    check(r1.r_id == 2 && r2.r_id == 1,
          $sformatf("different IDs: expected 2 before 1, got %0d then %0d", r1.r_id, r2.r_id));
    // Same ID: in order, even though the second is ready first.
    ar1.ax_id = 3;
    ar2.ax_id = 3;
    ar1.ax_addr = hbm_addr(64'h0);            // a.bin, channel 0 (refreshing)
    ar2.ax_addr = hbm_addr(64'h1000_2000);    // d.bin, channel 2 (not refreshing)
    align_refresh(20_000, 30_000);
    drv.send_ar(ar1);
    drv.send_ar(ar2);
    drv.recv_r(r1);
    drv.recv_r(r2);
    check(r1.r_data[63:0] == pattern(16'hA5A5, 0) && r2.r_data[63:0] == pattern(16'hD0D0, 0),
          "same ID: responses in request order");
  endtask

  // 16 bursts of 16 beats: all on channel 0, or one per channel. Returns the time from
  // the first AR to the last R beat.
  task automatic run_bursts(bit spread, output time t);
    time t0;
    t0 = $time;
    fork
      for (int k = 0; k < 16; k++) begin
        automatic ax_beat_t ar = new;
        // Channel k: offset k * 4 KiB. Channel 0 only: offset k * 16 channels * 4 KiB.
        ar.ax_addr  = hbm_addr(spread ? k * 4096 : k * 16 * 4096);
        ar.ax_id    = k;
        ar.ax_len   = 15;
        ar.ax_size  = $clog2(DW/8);
        ar.ax_burst = axi_pkg::BURST_INCR;
        drv_slow.send_ar(ar);
      end
      for (int k = 0; k < 16 * 16; k++) begin
        automatic r_beat_t r;
        drv_slow.recv_r(r);
      end
    join
    t = $time - t0;
  endtask

  task automatic test_channels();
    time t_one, t_spread;
    $display("== 6. channel parallelism (slow channels: 64 ns per beat)");
    run_bursts(1'b0, t_one);
    run_bursts(1'b1, t_spread);
    $display("   256 beats on 1 channel: %0t, on 16 channels: %0t", t_one, t_spread);
    // One channel is bound by its bus: 256 x 64 ns. Sixteen are bound by the AXI port
    // (256 beats at one per cycle) plus the first burst, which streams at channel rate.
    check(t_one >= 256 * 64ns, "one channel serialises its beats");
    check(t_spread * 3 < t_one, "sixteen channels overlap");
  endtask

  // The two regions random traffic runs in. Nothing else writes there.
  localparam logic [AW-1:0] Rand0Lo = AW'(ChipBase | 64'h1_0010_0000);
  localparam logic [AW-1:0] Rand0Hi = AW'(ChipBase | 64'h1_0014_0000);
  localparam logic [AW-1:0] Rand1Lo = AW'(ChipBase | 64'h3_4560_0000);
  localparam logic [AW-1:0] Rand1Hi = AW'(ChipBase | 64'h3_4564_0000);

  task automatic test_random();
    rand_master_t master = new(dv);
    $display("== 7. random traffic");
    master.add_memory_region(Rand0Lo, Rand0Hi, axi_pkg::DEVICE_NONBUFFERABLE);
    master.add_memory_region(Rand1Lo, Rand1Hi, axi_pkg::DEVICE_NONBUFFERABLE);
    master.reset();
    master.run(2000, 2000);
    $display("   %0d read beats checked, %0d skipped (a write to the line was in flight)",
             sb_checked, sb_skipped);
    check(sb_checked > 1000, "random traffic checked enough read beats");
  endtask

  initial begin
    drv.reset_master();
    drv_slow.reset_master();
    #1ns;
    i_dut.clear();
    i_dut.load_manifest("hbm/manifest.txt", ChipId);
    @(posedge rst_n);
    repeat (2) @(posedge clk);
    test_loading();
    test_cow();
    test_errors();
    test_latency();
    test_ordering();
    test_channels();
    test_random();
    repeat (10) @(posedge clk);
    i_dut.dump("dump_a.bin", 0, 256);
    if (errors == 0) $display("HBM TEST PASSED");
    else $display("HBM TEST FAILED: %0d check(s)", errors);
    $finish;
  end

  // ---------------------------------------------------------------------------------
  // Checker for the default DUT
  // ---------------------------------------------------------------------------------
  //
  // The model commits a write beat when it accepts it. So a read beat of a line nobody
  // wrote between the read's AR and its R beat has exactly one right answer: the line
  // as the last W beats left it. Reads that overlap a write are skipped (either answer
  // is legal AXI). Also checks R last and that every B and R matches an open burst.

  typedef struct {
    logic [AW-1:0]   addr;
    axi_pkg::len_t   len;
    axi_pkg::size_t  size;
    axi_pkg::burst_t burst;
    time             t;
    int unsigned     beat;
  } burst_t;

  burst_t          sb_aw [$];
  burst_t          sb_ar [logic [IW-1:0]][$];
  int unsigned     sb_writes [logic [IW-1:0]];
  logic [DW-1:0]   shadow [logic [AW-1:0]];
  time             shadow_t [logic [AW-1:0]];

  function automatic logic [AW-1:0] sb_line(burst_t b);
    return AW'(axi_pkg::aligned_addr(axi_pkg::beat_addr(b.addr, b.size, b.len, b.burst, b.beat),
                                     $clog2(DW/8)));
  endfunction

  function automatic bit in_rand(logic [AW-1:0] a);
    return (a >= Rand0Lo && a < Rand0Hi) || (a >= Rand1Lo && a < Rand1Hi);
  endfunction

  always @(posedge clk) if (rst_n) begin
    if (dv.aw_valid && dv.aw_ready) begin
      sb_aw.push_back('{dv.aw_addr, dv.aw_len, dv.aw_size, dv.aw_burst, $time, 0});
      if (!sb_writes.exists(dv.aw_id)) sb_writes[dv.aw_id] = 0;
      sb_writes[dv.aw_id]++;
    end
    if (dv.ar_valid && dv.ar_ready)
      sb_ar[dv.ar_id].push_back('{dv.ar_addr, dv.ar_len, dv.ar_size, dv.ar_burst, $time, 0});
    if (dv.w_valid && dv.w_ready) begin
      logic [AW-1:0] line;
      logic [DW-1:0] data;
      if (sb_aw.size() == 0) check(0, "W beat without a burst");
      else begin
        line = sb_line(sb_aw[0]);
        data = shadow.exists(line) ? shadow[line] : '0;
        for (int k = 0; k < DW / 8; k++) if (dv.w_strb[k]) data[8*k+:8] = dv.w_data[8*k+:8];
        shadow[line]   = data;
        shadow_t[line] = $time;
        sb_aw[0].beat++;
        if (sb_aw[0].beat > sb_aw[0].len) void'(sb_aw.pop_front());
      end
    end
    if (dv.b_valid && dv.b_ready) begin
      if (!sb_writes.exists(dv.b_id) || sb_writes[dv.b_id] == 0)
        check(0, $sformatf("B for id %0h with no write open", dv.b_id));
      else sb_writes[dv.b_id]--;
    end
    if (dv.r_valid && dv.r_ready) begin
      logic [AW-1:0] line;
      if (!sb_ar.exists(dv.r_id) || sb_ar[dv.r_id].size() == 0) begin
        check(0, $sformatf("R for id %0h with no read open", dv.r_id));
      end else begin
        line = sb_line(sb_ar[dv.r_id][0]);
        check(dv.r_last == (sb_ar[dv.r_id][0].beat == sb_ar[dv.r_id][0].len),
              $sformatf("R last on beat %0d of %0d", sb_ar[dv.r_id][0].beat,
                        sb_ar[dv.r_id][0].len + 1));
        if (in_rand(line) && dv.r_resp == axi_pkg::RESP_OKAY) begin
          if (shadow_t.exists(line) && shadow_t[line] >= sb_ar[dv.r_id][0].t) begin
            sb_skipped++;
          end else begin
            sb_checked++;
            check(dv.r_data == (shadow.exists(line) ? shadow[line] : '0),
                  $sformatf("random read of line 0x%0h: got %h, expected %h", line, dv.r_data,
                            shadow.exists(line) ? shadow[line] : '0));
          end
        end
        sb_ar[dv.r_id][0].beat++;
        if (sb_ar[dv.r_id][0].beat > sb_ar[dv.r_id][0].len) void'(sb_ar[dv.r_id].pop_front());
      end
    end
  end

  initial begin
    #20ms;
    $display("HBM TEST FAILED: timeout");
    $finish;
  end

endmodule
