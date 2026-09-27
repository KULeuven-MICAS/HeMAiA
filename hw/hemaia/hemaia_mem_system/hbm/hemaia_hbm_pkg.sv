// Copyright 2026 KU Leuven.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51
//
// Fanchen Kong <fanchen.kong@kuleuven.be>

// Configuration of hemaia_hbm_model, the simulated HBM behind the memory chip.
package hemaia_hbm_pkg;

  // Every time is in picoseconds, so the model's latencies do not depend on the
  // memchip clock (which the testharness runs at 1/20 or 1x the compute clock, and
  // the memchip itself divides again).
  typedef struct packed {
    // Address map. `base` is chip-local: the model ignores the chip-id bits above.
    longint unsigned base;
    longint unsigned size;              // capacity in bytes
    // Organisation.
    int unsigned     num_channels;      // pseudo-channels: independent command + data buses
    int unsigned     num_banks;         // banks per pseudo-channel
    int unsigned     row_bytes;         // row (page) size of one pseudo-channel
    int unsigned     interleave_bytes;  // consecutive bytes per channel before moving to
                                        // the next; 0 = one contiguous range per channel
    int unsigned     channel_mbps;      // peak data rate of one pseudo-channel, MB/s
    // Timing.
    int unsigned     t_ctrl_ps;         // controller + PHY + fabric, request and response
                                        // path together: the part of every access that is
                                        // not DRAM
    int unsigned     t_cl_ps;           // column read to data
    int unsigned     t_cwl_ps;          // column write to data
    int unsigned     t_rcd_ps;          // activate to column
    int unsigned     t_rp_ps;           // precharge (closing the open row)
    int unsigned     t_refi_ps;         // refresh interval per channel; 0 disables refresh
    int unsigned     t_rfc_ps;          // refresh duration, channel blocked
    // Controller queues.
    int unsigned     max_reads;         // read bursts in flight before AR back-pressure
    int unsigned     max_writes;        // write bursts in flight before AW back-pressure
  } hbm_cfg_t;

  // HBM2e as an AMD FPGA exposes it: two stacks of 16 pseudo-channels (64 bit at
  // 3.2 Gb/s = 25.6 GB/s each, 819 GB/s in total), 16 GiB.
  //
  // The latency is calibrated to what was measured on an FPGA HBM, not to the DRAM
  // datasheet: Shuhai (Wang et al., FCCM 2020) reports 106.7 ns for a row hit,
  // 122.2 ns for a closed row and 137.8 ns for a row conflict. t_ctrl + t_cl + one
  // beat gives 106.5 ns, + t_rcd 120.5 ns, + t_rp 134.5 ns.
  //
  // A 4 KiB interleave keeps every AXI burst (which never crosses 4 KiB) on one
  // pseudo-channel; bandwidth comes from bursts in flight on different channels.
  //
  // Only the fallback for an instance given no cfg (hemaia_mem_chip's default, the unit
  // test). The chip testharness always passes the platform cfg's `hbm` object, whose
  // defaults are in docs/schema/hemaia_hbm.schema.json -- keep the two equal.
  localparam hbm_cfg_t DefaultHbmCfg = '{
    base:             64'h1_0000_0000,
    size:             64'h4_0000_0000,
    num_channels:     32,
    num_banks:        16,
    row_bytes:        1024,
    interleave_bytes: 4096,
    channel_mbps:     25600,
    t_ctrl_ps:        90000,
    t_cl_ps:          14000,
    t_cwl_ps:         7000,
    t_rcd_ps:         14000,
    t_rp_ps:          14000,
    t_refi_ps:        3900000,
    t_rfc_ps:         350000,
    max_reads:        64,
    max_writes:       64
  };

endpackage
