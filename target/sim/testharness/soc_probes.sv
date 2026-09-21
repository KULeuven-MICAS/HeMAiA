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

  always @(posedge clk_i) begin
    if (!rst_ni) begin
      aw_all <= 0; ar_all <= 0; cyc <= 0; idle_cyc <= 0;
    end else begin
      cyc <= cyc + 1;
      if (aw_valid && aw_ready) aw_all <= aw_all + 1;
      if (ar_valid && ar_ready) ar_all <= ar_all + 1;
      // Any accepted transaction is proof of life; only a total absence of both counts.
      if ((aw_valid && aw_ready) || (ar_valid && ar_ready)) idle_cyc <= 0;
      else idle_cyc <= idle_cyc + 1;
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
