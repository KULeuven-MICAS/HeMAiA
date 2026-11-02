// Copyright 2026 KU Leuven.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51
//
// tb_sched_latency: the BINGO hardware manager's internal latency on a dependency edge, measured
// on its own RTL with its own unit-test harness (bingo_hw_manager test/
// tb_bingo_hw_manager_harness.svh, the same revision HeMAiA locks: 90bd3ee), configured as
// occamy_quad_ctrl instantiates it in its structure (2 clusters; 4 cores per cluster where HeMAiA has 5 = 4 + host: 5 does not fit the harness's 64-bit descriptor;
// CSR ready/done interface). The descriptor field widths are the harness defaults (12-bit task
// ids, 32 CERF groups, 4-bit tags, a 64-bit descriptor through the AXI-Lite slave task queue)
// instead of HeMAiA's (32-bit ids, 128 groups, 5-bit tags, 128 bit through the fetch master):
// they change how wide the fields are and how tasks enter, not the dependency path measured.
// Stimulus and probes: stim_sched_latency.svh.

`define TB_STIMULUS_FILE "stim_sched_latency.svh"
`define TB_NUM_CHIPLET 2
`define TB_NUM_CLUSTERS_PER_CHIPLET 2
`define TB_NUM_CORES_PER_CLUSTER 4
module tb_sched_latency;
  `include "tb_bingo_hw_manager_harness.svh"
endmodule
