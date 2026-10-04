// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Simulation-only observers, bound into the design so no generated file is edited.
//
// They exist for two reasons at once:
//
//  1. THEY MAKE THE SIGNALS VISIBLE. `+vcs+fsdbon` dumps scalars at any depth but not the
//     struct-typed AXI ports, so an AXI handshake cannot be read out of the waveform
//     directly. Each probe flattens the handshakes it watches into plain `logic`, which the
//     same auto-dump then captures.
//
//  2. THEY COUNT. A census printed at `final` does not depend on post-processing a waveform,
//     and answers "how often" rather than "here is one instance of it".
//
// Both are pure monitors: no driven outputs, nothing in the DUT's cone.
//
// Probes that answered their question have been removed. What is left is the set that earns
// its place on any run: what dispatch costs, what an instruction refill costs, whether the
// array is denied an operand, and whether the system is still alive.
// ---------------------------------------------------------------------------------------
// Probe output goes to a FILE, not stdout.
//
// The runs worth debugging are the ones that HANG, and a hung run never reaches $finish, so
// `final` blocks never print. Worse, the simulator's stdout is a pipe into the sweep runner,
// so anything already displayed sits in libc's buffer, unreadable, for as long as the hang
// lasts. A file handle with an explicit $fflush after every line is readable from outside
// while the simulation is still stuck, which is the only time it matters.
//
// One shared descriptor for every probe instance: a package variable is static, so the first
// caller opens the file and everyone else appends to the same handle.
// ---------------------------------------------------------------------------------------
package snax_probe_pkg;
  int unsigned probe_fd = 0;
  // Liveness is SYSTEM-wide: every snax_live_probe instance stamps the time of the last
  // transaction it saw here, and an instance only counts idle cycles while nobody anywhere
  // moved data. A decode layer streams weights HBM -> cluster L1 for milliseconds without
  // touching L3; a per-memory watchdog killed that as a hang.
  longint unsigned live_last_ps = 0;
  function automatic int unsigned fd();
    if (probe_fd == 0) probe_fd = $fopen("snax_probes.log", "w");
    return probe_fd;
  endfunction
endpackage

// ---------------------------------------------------------------------------------------
// bingo_hw_manager_top: what does the task-descriptor fetch actually cost?
//
// The manager walks the descriptor list in order over an AXI-Lite master, and a core cannot
// be granted a task that has not been fetched. This measures the three things that makes it
// a floor on dispatch: the AR-to-R round trip, how many reads are actually in flight against
// the MaxOutstanding budget, and how much of the run the task queue is empty.
//
// AXI-Lite has no IDs and its responses are in order, so matching R beats to AR beats is a
// FIFO of issue timestamps.
// ---------------------------------------------------------------------------------------
module snax_taskq_probe (
    input logic clk_i,
    input logic rst_ni,
    input logic ar_valid,
    input logic ar_ready,
    input logic r_valid,
    input logic r_ready,
    input logic q_empty,
    input logic q_pop
);

  // Flattened for the waveform.
  int  outstanding;
  int  last_latency;

  longint now, cyc, empty_cyc, pops;
  longint ar_beats, r_beats, lat_sum;
  int     lat_max, outstanding_max;
  longint ts_q[$];
  longint t0;
  // wedge detection: an AR held without ready, or responses that stop arriving while
  // transactions are still outstanding. Both mean the fetch path itself is stuck.
  int     ar_stall, idle_r, wedge_rpt;

  always @(posedge clk_i) begin
    if (!rst_ni) begin
      now = 0; cyc = 0; empty_cyc = 0; pops = 0;
      ar_beats = 0; r_beats = 0; lat_sum = 0; lat_max = 0;
      outstanding = 0; outstanding_max = 0; last_latency = 0;
      ar_stall = 0; idle_r = 0; wedge_rpt = 0;
      ts_q.delete();
    end else begin
      now = now + 1;
      cyc = cyc + 1;
      if (q_empty) empty_cyc = empty_cyc + 1;
      if (q_pop)   pops      = pops + 1;
      if (ar_valid && ar_ready) begin
        ts_q.push_back(now);
        ar_beats    = ar_beats + 1;
        outstanding = outstanding + 1;
        if (outstanding > outstanding_max) outstanding_max = outstanding;
      end
      if (r_valid && r_ready) begin
        r_beats = r_beats + 1;
        if (ts_q.size() > 0) begin
          t0           = ts_q.pop_front();
          last_latency = int'(now - t0);
          lat_sum      = lat_sum + last_latency;
          if (last_latency > lat_max) lat_max = last_latency;
        end
        if (outstanding > 0) outstanding = outstanding - 1;
      end
      if (ar_valid && !ar_ready)       ar_stall = ar_stall + 1; else ar_stall = 0;
      if (outstanding > 0 && !r_valid) idle_r   = idle_r   + 1; else idle_r   = 0;
      if ((ar_stall == 20000 || idle_r == 20000) && wedge_rpt < 6) begin
        wedge_rpt = wedge_rpt + 1;
        $fflush(snax_probe_pkg::fd()); $fdisplay(snax_probe_pkg::fd(), "[TASKQ WEDGE] %m @%0t ar_stall=%0d idle_r=%0d outstanding=%0d ar_beats=%0d r_beats=%0d q_empty=%0d ar_v/r=%0d/%0d r_v/r=%0d/%0d",
                 $time, ar_stall, idle_r, outstanding, ar_beats, r_beats, q_empty,
                 ar_valid, ar_ready, r_valid, r_ready);
      end
    end
  end

  task automatic report(input string when);
    if (ar_beats > 0) begin
      $fflush(snax_probe_pkg::fd()); $fdisplay(snax_probe_pkg::fd(), "[TASKQ %s] %m beats ar=%0d r=%0d | latency avg=%0.1f max=%0d cc | outstanding max=%0d",
               when, ar_beats, r_beats,
               (r_beats > 0) ? real'(lat_sum) / real'(r_beats) : 0.0, lat_max,
               outstanding_max);
      $fflush(snax_probe_pkg::fd()); $fdisplay(snax_probe_pkg::fd(), "[TASKQ %s] %m queue empty %0d of %0d cc (%0.1f%%) | descriptors popped=%0d",
               when, empty_cyc, cyc,
               (cyc > 0) ? 100.0 * real'(empty_cyc) / real'(cyc) : 0.0, pops);
    end
  endtask

  always @(posedge clk_i) if (rst_ni && cyc > 0 && (cyc % 20000) == 0) report("tick");

  final report("final");

endmodule

bind bingo_hw_manager_top snax_taskq_probe u_snax_taskq_probe (
    .clk_i   (clk_i),
    .rst_ni  (rst_ni),
    .ar_valid(task_queue_axi_lite_req_o.ar_valid),
    .ar_ready(task_queue_axi_lite_resp_i.ar_ready),
    .r_valid (task_queue_axi_lite_resp_i.r_valid),
    .r_ready (task_queue_axi_lite_req_o.r_ready),
    .q_empty (task_queue_mbox_empty),
    .q_pop   (task_queue_mbox_pop)
);

// ---------------------------------------------------------------------------------------
// snitch_hive: what does an ICACHE REFILL actually wait for?
//
// The refill master shares the cluster's wide crossbar with the DMA engines, and the quadrant
// then funnels every cluster onto one wide master. A cold line therefore queues behind whatever
// bulk transfer is in flight, which is why a refill can cost orders of magnitude more than the
// same instructions cost warm.
//
// The question this probe exists to settle has two candidate answers and they want opposite
// fixes:
//   ar_stall high  -> the refill cannot even ISSUE; it is losing arbitration on the cluster
//                     wide xbar. Fix = move icache refills to the NARROW master.
//   ar_stall low, latency high -> it issues promptly and the DATA is late; the wait is
//                     downstream in the quadrant funnel or L3. Fix = a second-level icache
//                     at the quadrant.
// Both counters are kept so the answer is a ratio, not an impression.
// ---------------------------------------------------------------------------------------
module snax_icache_probe (
    input logic clk_i,
    input logic rst_ni,
    input logic ar_valid,
    input logic ar_ready,
    input logic r_valid,
    input logic r_ready,
    input logic r_last
);

  // Flattened for the waveform.
  int  outstanding;
  int  last_latency;

  longint now, cyc, ar_beats, ar_stall, r_beats, lat_sum, refills;
  int     lat_max, lat_min, outstanding_max;
  longint ts_q[$];
  longint t0;

  always @(posedge clk_i) begin
    if (!rst_ni) begin
      now = 0; cyc = 0; ar_beats = 0; ar_stall = 0; r_beats = 0;
      lat_sum = 0; lat_max = 0; lat_min = 1 << 30; refills = 0;
      outstanding = 0; outstanding_max = 0; last_latency = 0;
      ts_q.delete();
    end else begin
      now = now + 1;
      cyc = cyc + 1;
      if (ar_valid && !ar_ready) ar_stall = ar_stall + 1;
      if (ar_valid && ar_ready) begin
        ts_q.push_back(now);
        ar_beats    = ar_beats + 1;
        outstanding = outstanding + 1;
        if (outstanding > outstanding_max) outstanding_max = outstanding;
      end
      if (r_valid && r_ready) begin
        r_beats = r_beats + 1;
        if (r_last) begin
          refills = refills + 1;
          if (ts_q.size() > 0) begin
            t0           = ts_q.pop_front();
            last_latency = int'(now - t0);
            lat_sum      = lat_sum + last_latency;
            if (last_latency > lat_max) lat_max = last_latency;
            if (last_latency < lat_min) lat_min = last_latency;
          end
          if (outstanding > 0) outstanding = outstanding - 1;
        end
      end
    end
  end

  task automatic report(input string when);
    if (ar_beats > 0) begin
      $fflush(snax_probe_pkg::fd()); $fdisplay(snax_probe_pkg::fd(), "[ICACHE %s] %m refills=%0d ar=%0d rbeats=%0d | AR stalled %0d of %0d valid cc (%0.1f%%)",
               when, refills, ar_beats, r_beats, ar_stall, ar_stall + ar_beats,
               (ar_stall + ar_beats > 0) ? 100.0 * real'(ar_stall) / real'(ar_stall + ar_beats) : 0.0);
      $fflush(snax_probe_pkg::fd()); $fdisplay(snax_probe_pkg::fd(), "[ICACHE %s] %m   AR->RLAST latency avg=%0.1f min=%0d max=%0d cc | outstanding max=%0d",
               when, (refills > 0) ? real'(lat_sum) / real'(refills) : 0.0,
               (lat_min == (1 << 30)) ? 0 : lat_min, lat_max, outstanding_max);
    end
  endtask

  always @(posedge clk_i) if (rst_ni && cyc > 0 && (cyc % 20000) == 0) report("tick");

  final report("final");

endmodule

bind snitch_hive snax_icache_probe u_snax_icache_probe (
    .clk_i   (clk_i),
    .rst_ni  (rst_ni),
    .ar_valid(axi_req_o.ar_valid),
    .ar_ready(axi_rsp_i.ar_ready),
    .r_valid (axi_rsp_i.r_valid),
    .r_ready (axi_req_o.r_ready),
    .r_last  (axi_rsp_i.r.last)
);

// ---------------------------------------------------------------------------------------
// snitch_hive: does an instruction REFILL come back poisoned?
//
// A refill beat carrying X (or an error response) is cached by the icache, so the core
// traps on that line from then on -- with mtvec in the bootrom, every trap restarts the
// program and the run looks like a hang. Main memory is zeroed before the binary loads, so
// an X here has a writer (or a broken read path) behind it; the refill address says where.
// Refills of one hive return in order, so the address is the head of an AR FIFO.
// ---------------------------------------------------------------------------------------
module snax_refill_xguard #(
    parameter int unsigned AW = 48,
    parameter int unsigned DW = 512
) (
    input logic          clk_i,
    input logic          rst_ni,
    input logic          ar_valid,
    input logic          ar_ready,
    input logic [AW-1:0] ar_addr,
    input logic          r_valid,
    input logic          r_ready,
    input logic          r_last,
    input logic [1:0]    r_resp,
    input logic [DW-1:0] r_data
);
  logic [AW-1:0] addr_q[$];
  int unsigned   beat, bad;

  always @(posedge clk_i) begin
    if (!rst_ni) begin
      addr_q.delete(); beat = 0; bad = 0;
    end else begin
      if (ar_valid && ar_ready) addr_q.push_back(ar_addr);
      if (r_valid && r_ready) begin
        // An all-zero beat decodes as an illegal instruction just like X does, so it is
        // reported too (a whole zero beat of code is never real).
        if (($isunknown(r_data) || r_resp != 2'b00 || r_data == '0) && bad < 32) begin
          bad = bad + 1;
          $fdisplay(snax_probe_pkg::fd(),
                    "[REFILL-X] %m @%0t line 0x%0h beat %0d resp %0d x=%0d zero=%0d",
                    $time, (addr_q.size() > 0) ? addr_q[0] : '1, beat, r_resp,
                    $isunknown(r_data), r_data == '0);
          $fflush(snax_probe_pkg::fd());
        end
        beat = beat + 1;
        if (r_last) begin
          beat = 0;
          if (addr_q.size() > 0) void'(addr_q.pop_front());
        end
      end
    end
  end
endmodule

bind snitch_hive snax_refill_xguard #(
    .AW($bits(axi_req_o.ar.addr)), .DW($bits(axi_rsp_i.r.data))
) u_snax_refill_xguard (
    .clk_i   (clk_i),
    .rst_ni  (rst_ni),
    .ar_valid(axi_req_o.ar_valid),
    .ar_ready(axi_rsp_i.ar_ready),
    .ar_addr (axi_req_o.ar.addr),
    .r_valid (axi_rsp_i.r_valid),
    .r_ready (axi_req_o.r_ready),
    .r_last  (axi_rsp_i.r.last),
    .r_resp  (axi_rsp_i.r.resp),
    .r_data  (axi_rsp_i.r.data)
);

// ---------------------------------------------------------------------------------------
// axi_to_mem_split: who puts X INTO main memory, and is it read back?
//
// Watches the memory side, where every request carries its own address. A write reports
// X only under its strobe (unwritten lanes are legitimately X on the bus). Reads are matched
// to their response through a per-port FIFO (the port answers every request in order, write
// or read). +L3G_LO=<hex> +L3G_HI=<hex> additionally reports every write whose low 32
// address bits fall in [LO, HI) -- set it to the device .text to catch whatever overwrites
// code. The AW side gives the writer's AXI id: the head of the AW FIFO is the burst the
// current W beat belongs to.
// ---------------------------------------------------------------------------------------
module snax_mem_xguard #(
    parameter int unsigned NP  = 2,
    parameter int unsigned AW  = 48,
    parameter int unsigned DW  = 512,
    parameter int unsigned AAW = 48,
    parameter int unsigned IW  = 8
) (
    input logic                     clk_i,
    input logic                     rst_ni,
    input logic [NP-1:0]            req,
    input logic [NP-1:0]            gnt,
    input logic [NP-1:0][AW-1:0]    addr,
    input logic [NP-1:0][DW-1:0]    wdata,
    input logic [NP-1:0][DW/8-1:0]  strb,
    input logic [NP-1:0]            we,
    input logic [NP-1:0]            rvalid,
    input logic [NP-1:0][DW-1:0]    rdata,
    input logic                     aw_valid,
    input logic                     aw_ready,
    input logic [AAW-1:0]           aw_addr,
    input logic [IW-1:0]            aw_id,
    input logic                     w_valid,
    input logic                     w_ready,
    input logic                     w_last
);
  typedef struct {logic [AW-1:0] a; logic w;} ent_t;
  ent_t          q[NP][$];
  logic [AAW-1:0] awa_q[$];
  logic [IW-1:0]  awi_q[$];
  longint unsigned lo, hi;
  bit             range_on;
  int unsigned    n_xw, n_xr, n_rg, cap_xw, cap_xr, cap_rg;
  ent_t           e;
  bit             xw;

  initial begin
    range_on = $value$plusargs("L3G_LO=%h", lo) && $value$plusargs("L3G_HI=%h", hi);
  end

  always @(posedge clk_i) begin
    if (!rst_ni) begin
      for (int p = 0; p < NP; p++) q[p].delete();
      awa_q.delete(); awi_q.delete();
      n_xw = 0; n_xr = 0; n_rg = 0; cap_xw = 0; cap_xr = 0; cap_rg = 0;
    end else begin
      for (int p = 0; p < NP; p++) begin
        if (rvalid[p] && q[p].size() > 0) begin
          e = q[p].pop_front();
          // X anywhere, or all zeros inside the guarded range (code is never a zero beat).
          if (!e.w && ($isunknown(rdata[p]) ||
                       (range_on && rdata[p] == '0 && longint'(e.a[31:0]) >= lo &&
                        longint'(e.a[31:0]) < hi))) begin
            n_xr = n_xr + 1;
            if (cap_xr < 32) begin
              cap_xr = cap_xr + 1;
              $fdisplay(snax_probe_pkg::fd(), "[L3-XREAD] %m @%0t port %0d addr 0x%0h x=%0d", $time,
                        p, e.a, $isunknown(rdata[p]));
            end
          end
        end
        if (req[p] && gnt[p]) begin
          q[p].push_back('{a: addr[p], w: we[p]});
          if (we[p]) begin
            xw = $isunknown(strb[p]);
            for (int b = 0; b < DW / 8; b++)
              if (strb[p][b] === 1'b1 && $isunknown(wdata[p][8*b+:8])) xw = 1;
            if (xw) begin
              n_xw = n_xw + 1;
              if (cap_xw < 32) begin
                cap_xw = cap_xw + 1;
                $fdisplay(snax_probe_pkg::fd(),
                          "[L3-XWRITE] %m @%0t port %0d addr 0x%0h strb 0x%0h | AW head id 0x%0h addr 0x%0h",
                          $time, p, addr[p], strb[p], (awi_q.size() > 0) ? awi_q[0] : '1,
                          (awa_q.size() > 0) ? awa_q[0] : '1);
              end
            end
            if (range_on && longint'(addr[p][31:0]) >= lo && longint'(addr[p][31:0]) < hi) begin
              n_rg = n_rg + 1;
              if (cap_rg < 64) begin
                cap_rg = cap_rg + 1;
                $fdisplay(snax_probe_pkg::fd(),
                          "[L3-GUARD] %m @%0t port %0d addr 0x%0h strb 0x%0h x=%0d | AW head id 0x%0h addr 0x%0h",
                          $time, p, addr[p], strb[p], xw, (awi_q.size() > 0) ? awi_q[0] : '1,
                          (awa_q.size() > 0) ? awa_q[0] : '1);
              end
            end
          end
        end
      end
      // AW after the W bookkeeping: a W beat in the same cycle as its own AW is rare and
      // only costs the id in the report.
      if (w_valid && w_ready && w_last && awa_q.size() > 0) begin
        void'(awa_q.pop_front()); void'(awi_q.pop_front());
      end
      if (aw_valid && aw_ready) begin awa_q.push_back(aw_addr); awi_q.push_back(aw_id); end
      if ((n_xw + n_xr + n_rg) > 0 && ($time % 64'd10_000_000) == 0) $fflush(snax_probe_pkg::fd());
    end
  end

  final if (n_xw + n_xr + n_rg > 0)
    $fdisplay(snax_probe_pkg::fd(), "[L3-X final] %m x-writes=%0d x-reads=%0d guarded-writes=%0d",
              n_xw, n_xr, n_rg);
endmodule

bind axi_to_mem_split snax_mem_xguard #(
    .NP (NumMemPorts), .AW(AddrWidth), .DW(MemDataWidth),
    .AAW($bits(axi_req_i.aw.addr)), .IW($bits(axi_req_i.aw.id))
) u_snax_mem_xguard (
    .clk_i   (clk_i),
    .rst_ni  (rst_ni),
    .req     (mem_req_o),
    .gnt     (mem_gnt_i),
    .addr    (mem_addr_o),
    .wdata   (mem_wdata_o),
    .strb    (mem_strb_o),
    .we      (mem_we_o),
    .rvalid  (mem_rvalid_i),
    .rdata   (mem_rdata_i),
    .aw_valid(axi_req_i.aw_valid),
    .aw_ready(axi_resp_o.aw_ready),
    .aw_addr (axi_req_i.aw.addr),
    .aw_id   (axi_req_i.aw.id),
    .w_valid (axi_req_i.w_valid),
    .w_ready (axi_resp_o.w_ready),
    .w_last  (axi_req_i.w.last)
);


// ---------------------------------------------------------------------------------------
// VersaCore SpatialArray: IS EVERY ACCUMULATION ACTUALLY ADDED?
//
// `accAddExtIn` marks the pass that is the FIRST of an output block -- the pass that takes
// external C instead of the running sum -- and `accAddExtInInput` is its input-side twin.
// Counting C handshakes against accAddExtIn assertions per dispatch says directly whether a
// block was denied its C, which is the failure that shows up as an output short by exactly
// one accumulation while the memory traffic looks correct. A dispatch boundary is a gap in
// computeFire.
// ---------------------------------------------------------------------------------------
module snax_array_probe #(
    parameter int unsigned IdleGap = 200,
    parameter int unsigned MaxRpt  = 40
) (
    input logic clk_i,
    input logic rst_ni,
    input logic in_c_valid,
    input logic in_c_ready,
    input logic out_d_valid,
    input logic out_d_ready,
    input logic acc_add_ext_in,
    input logic acc_add_ext_in_input,
    input logic compute_fire,
    input logic cstate_is_busy
);

  longint cyc, idle;
  longint c_fire, d_fire, add_asserts, addin_asserts, cfire_cnt;
  // per-dispatch tallies, flushed when computeFire has been quiet for IdleGap cycles
  longint d_c_fire, d_d_fire, d_add, d_addin, d_cf;
  int     disp, rpt;

  always @(posedge clk_i) begin
    if (!rst_ni) begin
      cyc=0; idle=0; c_fire=0; d_fire=0; add_asserts=0; addin_asserts=0; cfire_cnt=0;
      d_c_fire=0; d_d_fire=0; d_add=0; d_addin=0; d_cf=0; disp=0; rpt=0;
    end else begin
      cyc++;
      if (in_c_valid && in_c_ready)   begin c_fire++;  d_c_fire++;  end
      if (out_d_valid && out_d_ready) begin d_fire++;  d_d_fire++;  end
      if (acc_add_ext_in)             begin add_asserts++;   d_add++;   end
      if (acc_add_ext_in_input)       begin addin_asserts++; d_addin++; end
      if (compute_fire)               begin cfire_cnt++; d_cf++; end

      if (compute_fire) idle = 0;
      else              idle = idle + 1;

      // dispatch boundary
      if (idle == IdleGap && d_cf > 0) begin
        disp++;
        if (rpt < MaxRpt) begin
          rpt++;
          $fflush(snax_probe_pkg::fd());
          $fdisplay(snax_probe_pkg::fd(),
            "[ARRAY] %m dispatch %0d @%0t computeFire=%0d C_fire=%0d D_fire=%0d accAddExtIn_cyc=%0d accAddExtInInput_cyc=%0d",
            disp, $time, d_cf, d_c_fire, d_d_fire, d_add, d_addin);
        end
        d_c_fire=0; d_d_fire=0; d_add=0; d_addin=0; d_cf=0;
      end
    end
  end

  task automatic report(input string when);
    $fflush(snax_probe_pkg::fd());
    $fdisplay(snax_probe_pkg::fd(),
      "[ARRAY %s] %m dispatches=%0d computeFire=%0d C_fire=%0d D_fire=%0d accAddExtIn_cyc=%0d accAddExtInInput_cyc=%0d",
      when, disp, cfire_cnt, c_fire, d_fire, add_asserts, addin_asserts);
  endtask
  always @(posedge clk_i) if (rst_ni && cyc > 0 && (cyc % 200000) == 0) report("tick");
  final report("final");

endmodule

bind SpatialArray snax_array_probe u_snax_array_probe (
    .clk_i (clock), .rst_ni (~reset),
    .in_c_valid           (io_array_data_in_c_valid),
    .in_c_ready           (io_array_data_in_c_ready),
    .out_d_valid          (io_array_data_out_d_valid),
    .out_d_ready          (io_array_data_out_d_ready),
    .acc_add_ext_in       (io_ctrl_accAddExtIn),
    .acc_add_ext_in_input (io_ctrl_accAddExtInInput),
    .compute_fire         (io_ctrl_computeFire),
    .cstate_is_busy       (io_ctrl_cstate_is_busy)
);

// ---------------------------------------------------------------------------------------
// Main memory: IS THE SYSTEM STILL RUNNING?
//
// Turns a hang into a run that reports. Every phase of a device workload ends with the host
// reading results back, so main memory going completely silent for a long stretch means
// nothing is asking any more -- which is the difference between "slow" and "stopped". Without
// this a wedged run occupies a simulator until someone notices.
//
// The threshold is far above any real round trip here, so it cannot fire on congestion. On
// expiry it prints the traffic census and finishes, so the `final` blocks of every other
// probe run too -- a stalled run then yields the same data a healthy one does.
// ---------------------------------------------------------------------------------------
module snax_live_probe #(parameter int unsigned IdleTh = 200000) (
  input logic clk_i,
  input logic rst_ni,
  input logic aw_valid, input logic aw_ready,
  input logic ar_valid, input logic ar_ready
);
  longint aw_all, ar_all, cyc, idle_cyc;
  longint unsigned seen_last_ps;

  always @(posedge clk_i) begin
    if (!rst_ni) begin
      aw_all <= 0; ar_all <= 0; cyc <= 0; idle_cyc <= 0; seen_last_ps <= 0;
    end else begin
      cyc <= cyc + 1;
      if (aw_valid && aw_ready) aw_all <= aw_all + 1;
      if (ar_valid && ar_ready) ar_all <= ar_all + 1;
      // Any accepted transaction is proof of life -- here or at any other bound memory
      // (snax_probe_pkg::live_last_ps); only a total absence everywhere counts.
      if ((aw_valid && aw_ready) || (ar_valid && ar_ready)) begin
        idle_cyc <= 0;
        snax_probe_pkg::live_last_ps = $time;
      end else if (snax_probe_pkg::live_last_ps != seen_last_ps) begin
        idle_cyc <= 0;
        seen_last_ps <= snax_probe_pkg::live_last_ps;
      end else idle_cyc <= idle_cyc + 1;
    end
  end

  task automatic report(input string when);
    $fflush(snax_probe_pkg::fd());
    $fdisplay(snax_probe_pkg::fd(), "[LIVE %s] %m aw=%0d ar=%0d cyc=%0d", when, aw_all, ar_all, cyc);
  endtask

  always @(posedge clk_i) if (rst_ni && cyc > 0 && (cyc % IdleTh) == 0) report("tick");

  always @(posedge clk_i) begin
    // `> 0` so the guard cannot trip before the workload has started at all.
    if (rst_ni && idle_cyc == IdleTh && (aw_all + ar_all) > 0) begin
      $fdisplay(snax_probe_pkg::fd(),
                "[LIVE WATCHDOG] %m no main-memory traffic for %0d cc -- system stopped.", idle_cyc);
      report("watchdog");
      $fflush(snax_probe_pkg::fd());
      $finish;
    end
  end
  always @(posedge clk_i) if (rst_ni && (cyc % 2000) == 0) $fflush(snax_probe_pkg::fd());

  final report("final");
endmodule

bind axi_to_mem_split snax_live_probe u_snax_live_probe (
    .clk_i (clk_i), .rst_ni (rst_ni),
    .aw_valid (axi_req_i.aw_valid), .aw_ready (axi_resp_o.aw_ready),
    .ar_valid (axi_req_i.ar_valid), .ar_ready (axi_resp_o.ar_ready)
);

// The memory chiplet's simulated HBM is main memory too: weight streaming lives here. It has
// one port per push engine besides the xbar's, and a handshake on any of them counts.
module snax_live_probe_ports #(
  parameter int unsigned NumPorts = 1,
  parameter type         req_t    = logic,
  parameter type         rsp_t    = logic
) (
  input logic                  clk_i,
  input logic                  rst_ni,
  input req_t [NumPorts-1:0]   req_i,
  input rsp_t [NumPorts-1:0]   rsp_i
);
  logic aw_fire, ar_fire;
  always_comb begin
    aw_fire = 1'b0;
    ar_fire = 1'b0;
    for (int unsigned k = 0; k < NumPorts; k++) begin
      aw_fire |= req_i[k].aw_valid & rsp_i[k].aw_ready;
      ar_fire |= req_i[k].ar_valid & rsp_i[k].ar_ready;
    end
  end
  snax_live_probe u_snax_live_probe (
    .clk_i, .rst_ni,
    .aw_valid (aw_fire), .aw_ready (1'b1),
    .ar_valid (ar_fire), .ar_ready (1'b1)
  );
endmodule

bind hemaia_hbm_model snax_live_probe_ports #(
    .NumPorts (NumPorts), .req_t (axi_req_t), .rsp_t (axi_rsp_t)
) u_snax_live_probe (
    .clk_i (clk_i), .rst_ni (rst_ni), .req_i (axi_req_i), .rsp_i (axi_rsp_o)
);

// ---------------------------------------------------------------------------------------
// D2D link: WHAT IS STUCK WHEN THE SYSTEM STOPS?
//
// A run the LIVE watchdog ends ("no main-memory traffic") is a deadlock somewhere, and in a
// multi-chip run the D2D network is the prime suspect: each framer's receive side is ONE
// in-order payload stream that carries remote requests (AW, W, AR) AND the R responses to
// this chip's own reads, and an incoming AR waits for a free read slot (the R remapper)
// with everything behind it. These probes print, at the end of the run, every stream that
// has been holding a payload it cannot hand on, with what it is waiting for -- so the
// blocked cycle can be read off the log. Header tags (hemaia_d2d_link_pkg): 0 idle, 1 AW,
// 2 W, 3 AR, 4 R. Only stalls older than StallTh cycles are reported.
// ---------------------------------------------------------------------------------------
module snax_d2d_framer_probe #(parameter int unsigned HW = 3, parameter int unsigned StallTh = 2000) (
  input logic clk_i, input logic rst_ni,
  // receive side: the payload stream from the router
  input logic in_valid, input logic in_ready, input logic [HW-1:0] in_hdr,
  input logic map_valid,
  input logic aw_v, input logic aw_r, input logic w_v, input logic w_r,
  input logic ar_v, input logic ar_r,
  input logic r_v, input logic r_r,                 // R to this chip's own requester
  input logic srv_r_v, input logic srv_r_r, input logic srv_r_last,   // memory's R to a remote read
  // send side
  input logic out_valid, input logic out_ready, input logic [HW-1:0] out_hdr,
  input logic r_allowed,
  input logic req_ar_v, input logic req_ar_r, input logic own_r_last
);
  longint in_stall, out_stall, served, own;
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      in_stall <= 0; out_stall <= 0; served <= 0; own <= 0;
    end else begin
      in_stall  <= (in_valid && !in_ready) ? in_stall + 1 : 0;
      out_stall <= (out_valid && !out_ready) ? out_stall + 1 : 0;
      served <= served + (ar_v && ar_r) - (srv_r_v && srv_r_r && srv_r_last);
      own    <= own + (req_ar_v && req_ar_r) - (r_v && r_r && own_r_last);
    end
  end
  final begin
    if (in_stall > StallTh || out_stall > StallTh)
      $fdisplay(snax_probe_pkg::fd(),
                "[D2D-FRAMER] %m | RX head hdr=%0d stalled %0d cc (map_valid=%0d aw %0d/%0d w %0d/%0d ar %0d/%0d r %0d/%0d) | TX hdr=%0d stalled %0d cc | remote reads served in flight %0d, own reads in flight %0d (new allowed %0d), own AR waiting %0d",
                in_hdr, in_stall, map_valid, aw_v, aw_r, w_v, w_r, ar_v, ar_r, r_v, r_r,
                out_hdr, out_stall, served, own, r_allowed, req_ar_v && !req_ar_r);
  end
endmodule

bind hemaia_d2d_link_framer snax_d2d_framer_probe #(.HW($bits(payload_in_i.hdr))) u_snax_d2d_framer_probe (
  .clk_i(clk_i), .rst_ni(rst_ni),
  .in_valid(payload_in_valid_i), .in_ready(payload_in_ready_o), .in_hdr(payload_in_i.hdr),
  .map_valid(mapped_axi_id_valid),
  .aw_v(axi_out_req_o.aw_valid), .aw_r(axi_out_rsp_i.aw_ready),
  .w_v(axi_out_req_o.w_valid), .w_r(axi_out_rsp_i.w_ready),
  .ar_v(axi_out_req_o.ar_valid), .ar_r(axi_out_rsp_i.ar_ready),
  .r_v(axi_in_rsp_o.r_valid), .r_r(axi_in_req_i.r_ready),
  .srv_r_v(axi_out_rsp_i.r_valid), .srv_r_r(axi_out_req_o.r_ready), .srv_r_last(axi_out_rsp_i.r.last),
  .out_valid(payload_out_valid_o), .out_ready(payload_out_ready_i), .out_hdr(payload_out_o.hdr),
  .r_allowed(r_allowed),
  .req_ar_v(axi_in_req_i.ar_valid), .req_ar_r(axi_in_rsp_o.ar_ready), .own_r_last(axi_in_rsp_o.r.last)
);

module snax_d2d_dl_probe #(parameter int unsigned SW = 4, parameter int unsigned StallTh = 2000) (
  input logic clk_i, input logic rst_ni, input logic [SW-1:0] state,
  input logic rts_o, input logic cts_o, input logic rts_i, input logic cts_i,
  input logic txv, input logic txr, input logic rxv, input logic rxr,
  input logic tx_empty, input logic rx_full
);
  longint tx_stall, rx_stall;
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin tx_stall <= 0; rx_stall <= 0; end
    else begin
      tx_stall <= (txv && !txr) ? tx_stall + 1 : 0;
      rx_stall <= (rxv && !rxr) ? rx_stall + 1 : 0;
    end
  end
  final begin
    if (tx_stall > StallTh || rx_stall > StallTh)
      $fdisplay(snax_probe_pkg::fd(),
                "[D2D-LINK] %m | state=%0d rts_o=%0d cts_o=%0d rts_i=%0d cts_i=%0d | to-remote stalled %0d cc, from-remote stalled %0d cc | tx_empty=%0d rx_near_full=%0d",
                state, rts_o, cts_o, rts_i, cts_i, tx_stall, rx_stall, tx_empty, rx_full);
  end
endmodule

bind hemaia_d2d_link_data_link snax_d2d_dl_probe #(.SW($bits(flow_control_state))) u_snax_d2d_dl_probe (
  .clk_i(clk_i), .rst_ni(rst_ni), .state(flow_control_state),
  .rts_o(rts_o), .cts_o(cts_o), .rts_i(rts_i), .cts_i(cts_i),
  .txv(payload_to_remote_valid_i), .txr(payload_to_remote_ready_o),
  .rxv(payload_from_remote_valid_o), .rxr(payload_from_remote_ready_i),
  .tx_empty(tx_buffer_empty_i), .rx_full(rx_buffer_near_full)
);

module snax_d2d_router_probe #(parameter int unsigned NP = 5, parameter int unsigned StallTh = 2000) (
  input logic clk_i, input logic rst_ni,
  input logic [NP-1:0] iv, input logic [NP-1:0] ir, input logic [NP-1:0] ov, input logic [NP-1:0] orr
);
  longint is[NP], os[NP];
  always_ff @(posedge clk_i or negedge rst_ni) begin
    for (int p = 0; p < NP; p++) begin
      if (!rst_ni) begin is[p] <= 0; os[p] <= 0; end
      else begin
        is[p] <= (iv[p] && !ir[p]) ? is[p] + 1 : 0;
        os[p] <= (ov[p] && !orr[p]) ? os[p] + 1 : 0;
      end
    end
  end
  final begin
    for (int p = 0; p < NP; p++)
      if (is[p] > StallTh || os[p] > StallTh)
        $fdisplay(snax_probe_pkg::fd(),
                  "[D2D-ROUTER] %m port %0d (locals first, then E W N S) | in stalled %0d cc, out stalled %0d cc",
                  p, is[p], os[p]);
  end
endmodule

bind hemaia_d2d_link_router snax_d2d_router_probe #(.NP(NumPorts)) u_snax_d2d_router_probe (
  .clk_i(clk_i), .rst_ni(rst_ni), .iv(payload_in_valid), .ir(payload_in_ready),
  .ov(payload_out_valid), .orr(payload_out_ready)
);

// The write-aware mux: per output a WRITE LOCK (an AW holds the output for its W beats, and
// only the lock owner's writes may use it; a read or response waits while the owner has
// write data pending). A stalled stream that no link is refusing is waiting on one of these
// locks, so the dump names, for every stalled input, its head flit (tag, source and
// destination chip, output mask), and for every locked output its owner and count.
// router_fabric_data_t = {payload_network_t payload, logic [NP-1:0] dir}; payload_network_t =
// {axi_ch, hdr (3), dst_chip_id (8), src_chip_id (8)} -- packed from the LSB: dir, src, dst, hdr.
module snax_d2d_mux_probe #(
  parameter int unsigned NP = 5, parameter int unsigned EW = 64,
  parameter int unsigned PIW = 3, parameter int unsigned LCW = 5,
  parameter int unsigned StallTh = 2000
) (
  input logic clk_i, input logic rst_ni,
  input logic [NP*EW-1:0] data, input logic [NP-1:0] iv, input logic [NP-1:0] ir,
  input logic [NP-1:0] lock_active, input logic [NP*PIW-1:0] lock_owner,
  input logic [NP*LCW-1:0] lock_count, input logic [NP-1:0] ov, input logic [NP-1:0] orr
);
  longint is[NP];
  always_ff @(posedge clk_i or negedge rst_ni) begin
    for (int p = 0; p < NP; p++) begin
      if (!rst_ni) is[p] <= 0;
      else is[p] <= (iv[p] && !ir[p]) ? is[p] + 1 : 0;
    end
  end
  final begin
    automatic bit any = 0;
    for (int p = 0; p < NP; p++) if (is[p] > StallTh) any = 1;
    if (any) begin
      for (int p = 0; p < NP; p++) begin
        automatic logic [EW-1:0] e = data[p*EW +: EW];
        if (is[p] > StallTh)
          $fdisplay(snax_probe_pkg::fd(),
                    "[D2D-MUX] %m in %0d stalled %0d cc: hdr=%0d src=%02h dst=%02h to-outputs=%b",
                    p, is[p], e[NP + 16 +: 3], e[NP +: 8], e[NP + 8 +: 8], e[NP-1:0]);
      end
      for (int o = 0; o < NP; o++)
        if (lock_active[o])
          $fdisplay(snax_probe_pkg::fd(),
                    "[D2D-MUX] %m out %0d LOCKED by in %0d (count %0d) | out valid %0d ready %0d",
                    o, lock_owner[o*PIW +: PIW], lock_count[o*LCW +: LCW], ov[o], orr[o]);
    end
  end
endmodule

bind hemaia_d2d_link_write_aware_router_mux snax_d2d_mux_probe #(
  .NP(NumPorts), .EW($bits(input_data_i) / NumPorts), .PIW(PortIdxWidth), .LCW(LockCountWidth)
) u_snax_d2d_mux_probe (
  .clk_i(clk_i), .rst_ni(rst_ni), .data(input_data_i), .iv(input_valid_i), .ir(input_ready_o),
  .lock_active(lock_active_q), .lock_owner(lock_owner_q), .lock_count(lock_count_q),
  .ov(output_valid_q), .orr(output_ready_i)
);

// ---------------------------------------------------------------------------------------------
// snax_idma_queue_model: a TEST stand-in for a descriptor-queue frontend (what idma_desc64 would
// give) on the memory chiplet's push engines. The host posts a transfer as WRITES into its
// engine's register window and never reads anything back:
//   +0x800 src   +0x808 dst   +0x810 length   +0x818 any write queues it
//   +0x81c write 1: switch the engine over (the register frontend must be idle then)
// From the switch on, this model -- not idma_reg64_1d -- drives the engine's backend, back to
// back from its queue, each request FetchLat cycles after its doorbell landed (the descriptor
// fetch a real frontend would do from local memory). The options come from the register
// frontend's request at the switch (its conf register). Testbench only; nothing changes for a
// program that never writes +0x81c.
// ---------------------------------------------------------------------------------------------
`include "idma/typedef.svh"

module snax_idma_queue_model #(
    parameter int unsigned N = 4,
    parameter int unsigned FetchLat = 16
) (
    input logic clk_i,
    input logic rst_ni,
    // the engines' register buses (REG_BUS layout: addr, write, wdata, wstrb, valid / rdata,
    // error, ready)
    input logic [N-1:0][85:0] cfg_req_i,
    input logic [N-1:0][33:0] cfg_rsp_i
);
  `IDMA_TYPEDEF_OPTIONS_T(q_options_t, logic [3:0])
  `IDMA_TYPEDEF_REQ_T(q_req_t, logic [47:0], logic [47:0], q_options_t, logic [0:0])

  typedef struct {
    logic [47:0] src, dst, len;
    longint      t;
  } desc_t;

  longint cyc;
  always_ff @(posedge clk_i or negedge rst_ni)
    if (!rst_ni) cyc <= 0; else cyc <= cyc + 1;

  for (genvar k = 0; k < N; k++) begin : gen_q
    logic        wr, en, head_valid;
    logic [11:0] off;
    logic [31:0] wdata;
    logic [63:0] src, dst, len;
    q_req_t      tmpl, head;
    desc_t       q[$];
    int unsigned n_queued, n_issued, max_depth;

    assign wr    = cfg_req_i[k][0] && cfg_req_i[k][37] && cfg_rsp_i[k][0];
    assign off   = cfg_req_i[k][49:38];
    assign wdata = cfg_req_i[k][36:5];

    always_ff @(posedge clk_i or negedge rst_ni) begin
      if (!rst_ni) begin
        en <= 1'b0;
        q.delete();
        n_queued <= 0;
        n_issued <= 0;
        max_depth <= 0;
      end else begin
        if (wr) begin
          case (off)
            12'h800: src[31:0]  <= wdata;
            12'h804: src[63:32] <= wdata;
            12'h808: dst[31:0]  <= wdata;
            12'h80c: dst[63:32] <= wdata;
            12'h810: len[31:0]  <= wdata;
            12'h814: len[63:32] <= wdata;
            12'h818: begin
              q.push_back('{src[47:0], dst[47:0], len[47:0], cyc});
              n_queued <= n_queued + 1;
              if (q.size() + 1 > max_depth) max_depth <= q.size() + 1;
            end
            12'h81c: if (wdata[0] && !en) begin
              en   <= 1'b1;
              tmpl <= gen_sys_idma[k].idma_req;
            end
            default: ;
          endcase
        end
        if (head_valid && gen_sys_idma[k].idma_req_ready) begin
          void'(q.pop_front());
          n_issued <= n_issued + 1;
        end
      end
    end

    always @* begin
      head = tmpl;
      head_valid = 1'b0;
      if (en && q.size() > 0 && cyc >= q[0].t + FetchLat) begin
        head_valid    = 1'b1;
        head.length   = q[0].len;
        head.src_addr = q[0].src;
        head.dst_addr = q[0].dst;
      end
    end

    initial begin
      wait (en === 1'b1);
      $fdisplay(snax_probe_pkg::fd(), "[IDMA-Q] %m: engine %0d switched to the queue model @%0t",
                k, $time);
      force gen_sys_idma[k].idma_req = head;
      force gen_sys_idma[k].idma_req_valid = head_valid;
    end

    final if (en)
      $fdisplay(snax_probe_pkg::fd(), "[IDMA-Q] %m: engine %0d queued %0d issued %0d, deepest queue %0d",
                k, n_queued, n_issued, max_depth);
  end
endmodule

bind hemaia_mem_chip snax_idma_queue_model #(.N(NumSysIdma)) u_snax_idma_queue_model (
  .clk_i(clk_host), .rst_ni(rst_host_n), .cfg_req_i(engine_cfg_req), .cfg_rsp_i(engine_cfg_rsp)
);

// ---------------------------------------------------------------------------------------
// D2D EVENT TRACE IN A TIME WINDOW: who holds a link, and where a read waits.
//
// The probes above report a stall only if it is still there at the end of the run. A read
// that waits long and then completes leaves nothing behind, so these log events while they
// happen, inside a window given in ns: +D2DW_LO=<ns> +D2DW_HI=<ns>. Without both plusargs
// they are silent. Every line starts with the tag, the time in ns and the instance path, so
// one chip's link can be cut out of the log afterwards.
//   [D2DW-LINK]  a data link's flow-control state change, with rts/cts and the buffers
//   [D2DW-RATE]  per data link and microsecond: payload handshakes and stalled cycles, each way
//   [D2DW-STALL] a mux input that waited >= 32 cycles: its head flit (hdr, src, dst, outputs)
//   [D2DW-LOCK]  a mux output locked by a write (owner input) and released
//   [D2DW-AR]    a read request (hdr 3) handed through a mux
//   [D2DW-READ]  a framer: a remote read served (AR in / last R out), or its own read (AR / last R)
// Header tags: 0 idle, 1 AW, 2 W, 3 AR, 4 R.
// ---------------------------------------------------------------------------------------
package snax_d2dw_pkg;
  bit init = 0, cfg_on = 0;
  longint unsigned lo = 0, hi = 0;
  function automatic bit active(realtime now_ns);
    if (!init) begin
      init = 1;
      cfg_on = $value$plusargs("D2DW_LO=%d", lo) && $value$plusargs("D2DW_HI=%d", hi);
    end
    return cfg_on && now_ns >= lo && now_ns <= hi;
  endfunction
endpackage

module snax_d2dw_dl_probe #(parameter int unsigned SW = 4) (
  input logic clk_i, input logic rst_ni, input logic [SW-1:0] state,
  input logic rts_o, input logic cts_o, input logic rts_i, input logic cts_i,
  input logic txv, input logic txr, input logic rxv, input logic rxr,
  input logic tx_empty, input logic rx_full
);
  timeunit 1ns; timeprecision 1ps;
  logic [SW-1:0] st_q;
  longint unsigned tx_hs, rx_hs, tx_st, rx_st, cyc;
  realtime bucket;
  always @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      st_q <= '0; tx_hs <= 0; rx_hs <= 0; tx_st <= 0; rx_st <= 0; cyc <= 0; bucket <= 0;
    end else begin
      st_q <= state;
      if (snax_d2dw_pkg::active($realtime)) begin
        if (state != st_q)
          $fdisplay(snax_probe_pkg::fd(),
                    "[D2DW-LINK] %0.1f %m state %0d->%0d rts_o %0d cts_o %0d rts_i %0d cts_i %0d tx_empty %0d rx_full %0d txv %0d rxv %0d",
                    $realtime, st_q, state, rts_o, cts_o, rts_i, cts_i, tx_empty, rx_full, txv, rxv);
        if ($realtime - bucket >= 1000.0) begin
          $fdisplay(snax_probe_pkg::fd(),
                    "[D2DW-RATE] %0.1f %m cycles %0d tx_hs %0d tx_stall %0d rx_hs %0d rx_stall %0d",
                    $realtime, cyc, tx_hs, tx_st, rx_hs, rx_st);
          bucket <= $realtime; tx_hs <= 0; rx_hs <= 0; tx_st <= 0; rx_st <= 0; cyc <= 0;
        end else begin
          cyc <= cyc + 1;
          tx_hs <= tx_hs + (txv && txr); tx_st <= tx_st + (txv && !txr);
          rx_hs <= rx_hs + (rxv && rxr); rx_st <= rx_st + (rxv && !rxr);
        end
      end else bucket <= $realtime;
    end
  end
endmodule

bind hemaia_d2d_link_data_link snax_d2dw_dl_probe #(.SW($bits(flow_control_state))) u_snax_d2dw_dl_probe (
  .clk_i(clk_i), .rst_ni(rst_ni), .state(flow_control_state),
  .rts_o(rts_o), .cts_o(cts_o), .rts_i(rts_i), .cts_i(cts_i),
  .txv(payload_to_remote_valid_i), .txr(payload_to_remote_ready_o),
  .rxv(payload_from_remote_valid_o), .rxr(payload_from_remote_ready_i),
  .tx_empty(tx_buffer_empty_i), .rx_full(rx_buffer_near_full)
);

module snax_d2dw_mux_probe #(
  parameter int unsigned NP = 5, parameter int unsigned EW = 64,
  parameter int unsigned PIW = 3, parameter int unsigned LCW = 5
) (
  input logic clk_i, input logic rst_ni,
  input logic [NP*EW-1:0] data, input logic [NP-1:0] iv, input logic [NP-1:0] ir,
  input logic [NP-1:0] lock_active, input logic [NP*PIW-1:0] lock_owner
);
  timeunit 1ns; timeprecision 1ps;
  longint unsigned st[NP];
  realtime st_t[NP];
  logic [NP-1:0] lock_q;
  logic [NP*PIW-1:0] owner_q;
  always @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      for (int p = 0; p < NP; p++) begin st[p] <= 0; st_t[p] <= 0; end
      lock_q <= '0; owner_q <= '0;
    end else begin
      lock_q <= lock_active; owner_q <= lock_owner;
      for (int p = 0; p < NP; p++) begin
        automatic logic [EW-1:0] e = data[p*EW +: EW];
        if (iv[p] && !ir[p]) begin
          if (st[p] == 0) st_t[p] <= $realtime;
          st[p] <= st[p] + 1;
        end else begin
          if (st[p] >= 32 && snax_d2dw_pkg::active($realtime))
            $fdisplay(snax_probe_pkg::fd(),
                      "[D2DW-STALL] %0.1f %m in %0d waited %0d cc from %0.1f: hdr=%0d src=%02h dst=%02h to-outputs=%b",
                      $realtime, p, st[p], st_t[p], e[NP + 16 +: 3], e[NP +: 8], e[NP + 8 +: 8], e[NP-1:0]);
          st[p] <= 0;
        end
        if (iv[p] && ir[p] && e[NP + 16 +: 3] == 3'd3 && snax_d2dw_pkg::active($realtime))
          $fdisplay(snax_probe_pkg::fd(), "[D2DW-AR] %0.1f %m in %0d src=%02h dst=%02h to-outputs=%b",
                    $realtime, p, e[NP +: 8], e[NP + 8 +: 8], e[NP-1:0]);
      end
      if (snax_d2dw_pkg::active($realtime))
        for (int o = 0; o < NP; o++)
          if (lock_active[o] != lock_q[o] ||
              (lock_active[o] && lock_owner[o*PIW +: PIW] != owner_q[o*PIW +: PIW]))
            $fdisplay(snax_probe_pkg::fd(), "[D2DW-LOCK] %0.1f %m out %0d %s by in %0d",
                      $realtime, o, lock_active[o] ? "locked" : "released",
                      lock_active[o] ? lock_owner[o*PIW +: PIW] : owner_q[o*PIW +: PIW]);
    end
  end
endmodule

bind hemaia_d2d_link_write_aware_router_mux snax_d2dw_mux_probe #(
  .NP(NumPorts), .EW($bits(input_data_i) / NumPorts), .PIW(PortIdxWidth), .LCW(LockCountWidth)
) u_snax_d2dw_mux_probe (
  .clk_i(clk_i), .rst_ni(rst_ni), .data(input_data_i), .iv(input_valid_i), .ir(input_ready_o),
  .lock_active(lock_active_q), .lock_owner(lock_owner_q)
);

module snax_d2dw_framer_probe #(parameter int unsigned AW = 48) (
  input logic clk_i, input logic rst_ni,
  input logic srv_ar_v, input logic srv_ar_r, input logic [AW-1:0] srv_ar_addr, input logic [7:0] srv_ar_len,
  input logic srv_r_v, input logic srv_r_r, input logic srv_r_last,
  input logic own_ar_v, input logic own_ar_r, input logic [AW-1:0] own_ar_addr, input logic [7:0] own_ar_len,
  input logic own_r_v, input logic own_r_r, input logic own_r_last
);
  timeunit 1ns; timeprecision 1ps;
  always @(posedge clk_i) begin
    if (rst_ni && snax_d2dw_pkg::active($realtime)) begin
      if (srv_ar_v && srv_ar_r)
        $fdisplay(snax_probe_pkg::fd(), "[D2DW-READ] %0.1f %m served AR addr=%012h len=%0d",
                  $realtime, srv_ar_addr, srv_ar_len);
      if (srv_r_v && srv_r_r && srv_r_last)
        $fdisplay(snax_probe_pkg::fd(), "[D2DW-READ] %0.1f %m served R last", $realtime);
      if (own_ar_v && own_ar_r)
        $fdisplay(snax_probe_pkg::fd(), "[D2DW-READ] %0.1f %m own AR addr=%012h len=%0d",
                  $realtime, own_ar_addr, own_ar_len);
      if (own_r_v && own_r_r && own_r_last)
        $fdisplay(snax_probe_pkg::fd(), "[D2DW-READ] %0.1f %m own R last", $realtime);
    end
  end
endmodule

bind hemaia_d2d_link_framer snax_d2dw_framer_probe #(.AW($bits(axi_out_req_o.ar.addr))) u_snax_d2dw_framer_probe (
  .clk_i(clk_i), .rst_ni(rst_ni),
  .srv_ar_v(axi_out_req_o.ar_valid), .srv_ar_r(axi_out_rsp_i.ar_ready),
  .srv_ar_addr(axi_out_req_o.ar.addr), .srv_ar_len(axi_out_req_o.ar.len),
  .srv_r_v(axi_out_rsp_i.r_valid), .srv_r_r(axi_out_req_o.r_ready), .srv_r_last(axi_out_rsp_i.r.last),
  .own_ar_v(axi_in_req_i.ar_valid), .own_ar_r(axi_in_rsp_o.ar_ready),
  .own_ar_addr(axi_in_req_i.ar.addr), .own_ar_len(axi_in_req_i.ar.len),
  .own_r_v(axi_in_rsp_o.r_valid), .own_r_r(axi_in_req_i.r_ready), .own_r_last(axi_in_rsp_o.r.last)
);
