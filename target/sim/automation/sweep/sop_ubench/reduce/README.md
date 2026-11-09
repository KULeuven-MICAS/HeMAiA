# Endpoint chain reduce: a standalone, hardware-only testbench

A reduction across P compute chiplets, measured on the hardware alone: the xDMA ChainGather (one chain, a two-stage
chain, a binomial tree of chains) against the three software reductions, and the nonlinear fold operators. Feeds the
paper's `exp/eval2_reduce` (Section VI-C, Figs. 9 and 10).

## Questions

1. **Fig. 10, chain or tree.** Where does a tree of chains overtake the single chain? The payload sweep puts the
   crossovers inside P <= 16.
2. **Fig. 9, chain against software.** The chain reduce against the three software reductions (root pull + SIMD fold,
   software chain, software tree): latency and SRAM accesses against P.

## Design

- Pure hardware: no CVA6 host, no boot, no device software. The testbench programs every xDMA through its CSRs and
  times the hardware alone.
- Endpoints plus the real D2D links and routers: one xDMA endpoint (cluster 0's `snax_xdma_cluster_xdma_wrapper` with
  its TCDM) per compute chiplet, joined by `hemaia_d2d_link` instances; no Snitch cores, no clusters' other engines.
- The software baselines (Fig. 9) are replayed: the iDMA transfer sequence the software issues, the SIMD fold time,
  and a fixed software hand-off time.
- Tree levels (Fig. 10) are synchronised by a simulated hardware-scheduler edge: the dep-set message as one narrow
  write over the real links, plus the scheduler's own latency.
- P <= 16: the package's 16 compute chiplets.

## Platform modelled (`hemaia_16c4m_8MBL3_2cluster`)

- 5 x 4 package; compute chiplets at x = 0, 1, 3, 4, y = 0..3; memory chiplets at x = 2, which only route here (their
  link instance forwards transit traffic; no endpoint behind it). Chip ID = x << 4 | y.
- Clocks: the cluster (xDMA endpoint, TCDM) at 500 MHz; the SoC fabric and the link's digital side at 1 GHz; the PHY
  at 5 GHz, links in DDR.
- Per compute chiplet, the path the real SoC builds between the cluster and the link: narrow (64 b) and wide (512 b)
  AXI, the narrow-to-wide converters, the SoC wide crossbar with the link as the remote port, and the cfg's pipeline
  cuts (`narrow_to_quad` 3, `quad_to_narrow` 3, `wide_to_quad` 3, `quad_to_wide` 3, `soc_to_router` 1,
  `router_to_soc` 1, ...). What is simplified is listed under Simplifications.
- TCDM 1 MiB per cluster; only cluster 0 of each chiplet takes part.

Snake (data order of the chain; P = 2, 4, 8, 16 are its prefixes):

```
0x00 0x01 0x02 0x03 | 0x13 0x12 0x11 0x10 | 0x30 0x31 0x32 0x33 | 0x43 0x42 0x41 0x40
```

Consecutive chips are neighbours, except 0x10 -> 0x30, which crosses the memory chiplet 0x20.

## Arms (`+ARM=`)

| arm | figure | what the testbench does |
|---|---|---|
| `chain` | 9, 10 | one ChainGather over the P chips, issued at the collector (snake index 0); the stream runs from snake index P-1 to 0, folding (FP16 add) at every hop |
| `two` | 10 | two-stage chain: groups of 4 consecutive snake chips gather at once, then one chain over the group collectors; P = 8 is 4-2, P = 16 is 4-4 |
| `binomial` | 10 | log2 P levels of two-chip ChainGathers |
| `pull` | 9 | the root reads every partial (iDMA sequence, one source in flight) and folds it on the SIMD |
| `swchain` | 9 | software chain: receive, add, forward, pipelined in chunks |
| `swtree` | 9 | software binomial tree with clear-to-send and flags |
| `rt_sched`, `rt_flag` | - | a round trip from chip 0x00 to the participants and back: the scheduler edge against hand-placed flags (below) |

Tree levels and software hand-offs pay their sync as simulated traffic over the real links: a hardware-scheduler edge
for `two` and `binomial`, a software flag for the software arms. Each arm follows `snax-coll-reduce.c` (the arm's role
in that app), with the partials already in place at the start: the app's GO / READY round trip and its grid barriers
are not part of any arm.

`+OP=add|smx|topk` (default `add`) selects the collector's writer junction for `chain` and `two`: `smx` =
SoftmaxStatsJunction (junction 1, CSRs nValid = 32, scale = 1.0; a partial is groups of a key beat m[0..31] and a
value beat l[0..31], FP16), `topk` = TopKJunction (junction 2, k = 8, keyPol = 0; one list of 8 FP32 values + 8
payloads per beat). The software arms take `+OP` too: the sum's payload (every movement and flag still checked), each
fold lasting at least the merge's cycles on one cluster, measured on the cluster RTL by snax_cluster
`sw/apps/snax-sw-merge` (`reduce_pkg` MergeBytes / MergeSmxCc (SIMD datapath) / MergeTopkCc (four cores)),
piecewise linear in between. Goldens: top-k exact (values distinct over all chips), smx m* exact and l* within 2 % of
sum l_k exp(m_k - m*).

## Sweep and metrics

- Payload per chip: 4, 8, 16, 32, 64, 128, 256 KiB (FP16); P: 2, 4, 8, 16; every arm at every point, one repeat.
- `lat_prog_cc`, the latency the figures use: from the first programming action of the reduction (the first CSR
  write, DMA instruction or store any chip issues) to the result complete at the collector. `lat_cc`: from the first
  START write (xDMA arms) or the first DMA copy instruction (software arms). The result is complete when the
  collector's finish counter has reached its target and the first and the last result word have landed (xDMA arms),
  or when the collector's last SIMD add has written its last beat (software arms). 500 MHz cluster cycles.
- SRAM accesses: TCDM port handshakes (one 64-bit word each) of all 16 chips from START to quiescence, total and at
  the busiest chip (`sram_max`).
- Check: the result at the collector and every intermediate sum (each tree level's, each software-chain stage's,
  each software-tree level's) against the exact sum over its chips (small integers, exact in FP16), over the whole
  buffer.
- Buffers: IN = TCDM + 0x08000, OUT = + 0x28000, TMP = + 0x48000 (the app's) up to 128 KiB; at 256 KiB they are
  256 KiB apart (OUT + 0x48000, TMP + 0x88000), which still fits the 1 MiB TCDM.

## Running

| file | what |
|---|---|
| `tb/reduce_pkg.sv` | platform constants, the snake, the SoC's AXI types (occamy_pkg names), the cluster TCDM types, the xDMA CSR map of `snax_split_cluster`, the D2D link register offsets, the merge costs |
| `tb/reduce_compute_node.sv` | one compute chip: cluster 0's xDMA endpoint and the chip's path from it to the link's local port |
| `tb/tb_reduce.sv` | the 5 x 4 package: 20 `hemaia_d2d_link`, 16 compute nodes, the TCDM models, clocks, link bring-up, CSR tasks, every arm, checks and counters |
| `tb/reduce_simd_model.sv` | the cluster SIMD's two-operand FP16 ADD as a model on real TCDM ports (software arms) |
| `run.sh` | VCS build and run (`build`, `smoke`, `validate`, `run <tag> <plusargs>`, `table`, `sweep`, `parse`) |
| `parse.py` | `results/logs/*.log` (or the logs given) -> `results/reduce.csv` |
| `parse_ops.py` | the `+OP` logs -> `results/reduce_ops.csv` |
| `summary.py` | `results/reduce.csv` -> `results/summary.md`: one payload's points and the fastest arm per (P, payload) |
| `sanity_plot.py` | a quick look at `results/reduce.csv` -> `results/sanity.png` (not a paper figure) |
| `sched/` | the BINGO manager latency measurement: `tb_sched_latency.sv`, `stim_sched_latency.svh`, `run_sched.sh`, `sched_latency.log` |

```bash
./run.sh build      # ~1 min: work/simv
./run.sh smoke      # chain, P = 2, 4 KiB          -> logs/smoke.log      (~3 s)
./run.sh validate   # chain, P = 2..16, 128 KiB    -> logs/validate.log
./run.sh run p8_16k +BYTES=16384 +PLIST=8 +VERBOSE=1
./run.sh sweep      # every arm x payload, all P, JOBS (default 4) at a time -> results/logs/, then parse.py
```

`run.sh` compiles only what `tb_reduce` needs: `bender script flist-plus` restricted to the packages it instantiates
(common_cells, axi, register_interface, tech_cells_generic, hemaia_d2d_link, axi_riscv_atomics, xdma_axi_adapter,
idma and their dependencies), plus, from snitch_cluster, only `reqrsp_pkg` and the generated
`snax_split_cluster_xdma` (blocks + wrapper). Flags are target/sim's VCS flags (`-override_timescale=1ns/1ps`,
`+nospecify`, `-O4`, ...). It reads the generated xDMA under
`deps/snitch_cluster/target/snitch_cluster/generated/snax_split_cluster_xdma/` as it is; if the Chisel changes,
regenerate it with the cluster flow. VCS runtime licences are scarce, so every simulation queues for one
(`+vcs+lic+wait`). Knobs (environment, see `run.sh`): `TCDM_BANKED`, `LINK_TXWBUF`, `XDMA_WHOLEBURST`, `XDMA_PAGE`,
`XDMA_ADAPTER_DIR` (a local xdma_axi_adapter checkout whose `src/` replaces the pinned one), `PHY_BEHAVIORAL`, and
`WORK` / `LOGS` for side-by-side builds.

Every run prints one line per point:
`[REDUCE] arm=chain P=.. bytes=.. lat_cc=.. lat_prog_cc=.. fin_cc=.. first_cc=.. last_cc=.. task_cc=.. sram_total=.. sram_max=.. busiest=.. check=..`;
`+VERBOSE=1` adds, per participant, its TCDM accesses, the first/last W beat entering and leaving its xDMA (`w_in`,
`w_out`) and its narrow AW count each way.

## What is modelled

Per compute chip, the path between cluster 0's xDMA and the link's 512-bit local port, stage for stage with the RTL's
IPs and parameters (cfg `hemaia_16c4m_8MBL3_2cluster`):

```
cluster (500 MHz)  xDMA (snax_split_cluster_xdma_wrapper: Chisel frontend + xdma_axi_adapter_top)
                   DMA xbar (wide, 4x3, CUT_ALL_PORTS) / cluster xbar (narrow, 4x4, CUT_ALL_PORTS)
                   RegisterExtWide / RegisterExtNarrow axi_cuts on the cluster ports
CDC                id remaps + 4 axi_cdc (LogDepth 3), as occamy_quad cluster 0
quadrant (1 GHz)   quad wide xbar (3x3) + out cut + id remap + cut; in cut + id remap
                   quad narrow xbar (3x3) + id remaps + cuts; quad_ctrl soc_to_quad (CUT_SLV_PORTS)
                   and quad_to_soc (CUT_MST_PORTS) xbars
SoC (1 GHz)        narrow_to_quad / quad_to_narrow / wide_to_quad / quad_to_wide cuts (3 each)
                   SoC narrow xbar (5x7) and SoC wide xbar (5x5), CUT_ALL_PORTS, chip-ID address maps
                   narrow -> wide: atomics adapter (1 cut) + cut + id remap 7->4 + 64->512 dw converter
                   wide -> narrow: id remap 7->4 + 512->64 dw converter + cut
                   soc_to_router: id remap 7->4 + 1 cut; router_to_soc: 1 cut
link (1 GHz / PHY 5 GHz)  hemaia_d2d_link as occamy_chip i_d2d_link (Send 4, Recv 8, 4 PHYs)
```

Addresses carry the chip ID in bits [47:40]; every xbar decodes them with the chip's own map, so a remote address
leaves through the default ports (cluster -> quad -> SoC -> link) exactly as on the chip, and an incoming xDMA data
write (to the remote cluster's data MMIO, `0x103F_C000`) or control write (cfg / grant / finish MMIO, `0x103F_D000..`)
takes the wide or the narrow path in.

Memory nodes (x = 2) are `hemaia_d2d_link` instances as `hemaia_mem_chip` has them (a PHY on each side that faces a
chip, 4 local ports), with the local ports tied off: they forward transit traffic only (an assertion fails the run if
anything is delivered to one). PHY IO is joined by `tran` per bit, flow control and test-request side bands by
`assign`, as the testharness's `io_wrapper` does; compute chips have all four PHYs (as `occamy_chip`), the unused sides
unconnected with their inputs tied low.

Clocks: PHY 5 GHz, fabric and link digital side 1 GHz, cluster 500 MHz, edge aligned (the cfg's master 5 GHz with
core /5, cluster /10, PHY /1). Link bring-up: the end state of the host's `hemaia_d2d_link_initialize_grid` +
`hemaia_d2d_link_ddr_on_grid` (`-DHEMAIA_D2D_DDR=1`), written through each link's AXI-Lite port before any traffic:

| node | TX turnaround silence | RX yield window | availability | multicast fence | PHY DDR mode |
|---|---|---|---|---|---|
| compute | 0 (all sides) | 0 | ports facing a chip | reset (0x1F) | 0xF |
| memory | reset (4) | 0 | ports facing a chip | ports facing a chip (transit only) | 0xF |

(`same_memchip_speed` is set, so no port gets the slow-memchip yield/clock settings; the DDR mode is read back on every
node.) No link training: the host does none at boot.

The `chain` arm programs the collector (snake index 0) with the CSR sequence of `snax-coll-reduce.c`'s
`xdma_chain_fold()`: extensions off, every junction disabled (read-modify-write),
`xdma_chain_gather_1d_full_address(IN, [S[P-1].IN, ..., S[1].IN, S[0].OUT], P, N, ELEMENTWISE, 0 = ADD | FP16)` (all
16 slots, 5 dims each side, unused dims 1/0), the remote-counter check, `xdma_start()`, `xdma_wait_task()`, then the
app's clean-up. Partials are FP16 small integers distinct per chip, element and beat.

### Trees: `two`, `binomial`

The ChainGathers are the app's (`xdma_chain_fold`, the same CSR sequence as `chain`, including its sentinel poll and
its clean-up); the synchronisation between them is a simulated BINGO hardware-scheduler edge instead of the app's
software flags.

- `two` (`xdma_two_level_reduce`): groups of 4 consecutive snake positions; each group's root is the member nearest
  the collector (P = 8: 0x00, 0x10; P = 16: 0x00, 0x10, 0x30, 0x40); all groups gather at once (data order farthest
  member first, sums into OUT, the collector's own into TMP); then the collector gathers the group roots' sums into
  OUT. P = 2 and 4 are one group: the plain chain (reported with `same_as=chain`).
- `binomial` (`xdma_tree_reduce`): log2 P levels; at level l position p (low l+1 bits clear) takes p + 2^l's sum with
  a two-chip ChainGather through the giver's writer junction; sums alternate TMP / OUT. P = 2 is the plain chain.

**The edge.** When a gather completes at its collector, that chip's DM core writes `done` to its BINGO manager; the
manager sends the dep-set message as one 64-bit AXI-Lite write to the next level's issuing chip; that chip's core,
blocked reading its ready queue, gets its next task and starts programming it. Modelled, per compute node:

| piece | how |
|---|---|
| core <-> manager CSR crossing, both ways | the real `cdc_fifo_gray` pair of `occamy_quad.sv` (LOG_DEPTH 1, SYNC_STAGES 2), cluster clock <-> 1 GHz host clock |
| the manager's internal latency | a model with the latencies measured on its RTL (below) |
| the dep-set write | a real AXI-Lite write from the manager's `to_remote_chiplet_axi_lite` port through the quadrant AXI-Lite xbar, `axi_lite_to_axi`, quad_to_soc, the SoC, the links, the target's SoC and soc_to_quad, `axi_atop_filter`, `axi_to_axi_lite`, the quadrant AXI-Lite xbar into the target manager's `CHIPLET_DONE_QUEUE` mailbox at {chip, 0x0a00_0000} (not on the data path) |

A remote edge pays: the done CSR crossing, 2 manager cycles to the write, the write's real path, 3 cycles mailbox ->
ready queue, 1 cycle to the core's read, the response crossing. A local edge (the same chip's next task) pays the
crossings and 3 + 1 cycles. The next level's programming starts when the core has its task. The app's clean-up after a
fold (junction off, 30 slot writes, ~33 cluster cycles) runs before the core writes `done`, as in the app.

**BINGO manager internal latency.** `sched/tb_sched_latency.sv` runs the manager's own RTL (bingo_hw_manager 90bd3ee)
in its own unit-test harness with a stimulus of one remote and one local edge (`sched/run_sched.sh`, log
`sched/sched_latency.log`):

| edge | cycles of the manager's clock |
|---|---|
| remote, sender: done CSR at the manager -> dep-set AW out | 2 |
| remote, receiver: write in the mailbox -> dependent task in the ready queue -> core's CSR read answered | 3 + 1 = 4 |
| local: done CSR -> ready queue -> CSR read answered | 3 + 1 = 4 |

The manager runs at the 1 GHz host clock in HeMAiA, so 2 cycles = 1 cluster cycle and 4 cycles = 2 cluster cycles.
The harness is configured with 2 clusters of 4 cores (HeMAiA: 5 = 4 + host, which does not fit the harness's 64-bit
descriptor) and the harness's descriptor widths (12-bit ids, 32 CERF groups, 4-bit tags, slave task queue) instead of
HeMAiA's (32-bit ids, 128 groups, 5-bit tags, fetch master): field widths and the task-entry path, not the dependency
path.

### Software baselines: `pull`, `swchain`, `swtree`

The app's sequences, with real data movement and real flags:

- **Data movement: the real cluster iDMA** (`idma_inst64_top`, DMAAxiReqFifoDepth 16, DMAReqFifoDepth 8) on the
  cluster DMA xbar's SDMAMst port; the testbench issues the DM core's instructions (`dmsrc`, `dmdst`, `dmcpyi`, and
  `dmstati 2` polling for `snrt_dma_wait_all`) at its instruction interface, one per cycle. Remote reads (pull) and
  writes go over the links; the TCDM side is the cluster's `axi_to_mem_interleaved` into the TCDM's wide port, which
  wins its super-bank over the narrow ports as in `snitch_cluster.sv`. A flag behind data is a 4-byte iDMA copy of
  C_VAL after the data, as `dma_flag`.
- **SIMD fold: a model** (`tb/reduce_simd_model.sv`) on real TCDM ports (its reads and writes are counted and contend
  with the DMA): one 32 B input beat per cycle (reader: 4 channels x 64 b), the two operands alternating, one 32 B
  output beat (16 FP16 sums) per pair: **8 FP16 results per cycle, n bytes in n / 16 cycles**, the RTL's rate
  (`SimdStreamElementwise.scala`: "one beat per cycle ... each pair emits one beat"). Not modelled: the SIMD's
  programming by its core and its pipeline fill.
- **Software sync**: every control flag is a real 32-bit store over the links (a narrow AXI write from the cluster
  xbar's CoreReq port, `snrt_xchip_writew`), every data flag a real iDMA write behind its data; the waiting side sees a
  flag when it lands in its TCDM and then pays `SwHandoffCc` = 250 cluster cycles (a testbench parameter). Hand-offs:
  `swchain` one per chunk per hop, `swtree` the data flag per level plus the clear-to-send at levels > 0; `pull` has
  none (its per-step overheads are only what is modelled: the DMA instructions and `dmstati` polling).
- `pull` (`fold_all`): the root pulls each partial in two halves into two TMP slots, one source in flight
  (`dma_wait` before the next), the add of one half overlapping the pull of the next.
- `swchain`: 4 chunks; chip i waits for chunk q's flag from i+1, adds its IN, pushes the sum chunk and its flag to
  i-1; the far end pushes its IN.
- `swtree`: binomial on snake positions; a receiver at level l > 0 first sends the sender a clear-to-send (TMP is
  reused), then waits for the data flag and adds into OUT.

### Round trips: `rt_flag`, `rt_sched`

The software counterpart of a scheduler edge: out from chip 00 to the participants and back, timed on chip 00 in
500 MHz cluster cycles. Participants: the rectangle of compute chips from 00 (P = 2: 1x2, 4: 2x2, 8: 2x4, 16: all,
7 hops to chip 43); one-to-one goes to the far corner, one-to-all to every participant, each of which answers chip 00.

- `rt_flag`: a 32-bit store (`snrt_xchip_writew`) into the consumer's TCDM, polled there (one iteration per cycle,
  the most favourable loop); chip 00 scans its answer slots in order. `+BCAST=1` sends chip 00's one-to-all flag as
  one broadcast store (chip id 0xFF).
- `rt_sched`: the BINGO manager model: done CSR, dep-set write over the links (one-to-all: one broadcast write), the
  consumer's ready-queue CSR read.
- Both: a core acts `+TURN_CC` (default 4) cycles after it learns of its dependency.

### Simplifications

1. **Cluster.** Only the xDMA, the iDMA and the 16 TCDM ports. The snitch cores, the SIMD and VersaCore engines, the
   icache, `axi_to_tcdm` and the cluster peripherals are absent; their xbar ports are tied off (assertions fire if
   anything is routed to them).
2. **TCDM.** A behavioural model of the 1 MiB TCDM: 1-cycle read latency (RegisterTCDMCuts = 0), 32 word-interleaved
   64-bit banks, one access per bank per cycle, conflicting ports served in a priority order that rotates every
   cycle. The real `snitch_tcdm_interconnect` + `snitch_tcdm_arb_level` (per-bank round robin with urgency levels) is
   not instantiated, so the order in which conflicting ports win may differ. `TCDM_BANKED=0` grants every access at
   once.
3. **CSR path.** The testbench drives the xDMA's CSR port directly, one CSR per cluster cycle; the DM core's
   CSR/offload path (register_offload_req/rsp cuts, instruction issue) is absent. `lat_cc` starts at the START write,
   so only `lat_prog_cc` depends on this.
4. **Cluster 1, quadrant controller.** Cluster 1 is absent (its quad xbar ports tied off). Of `occamy_quad_ctrl` only
   the two narrow xbars the xDMA's traffic crosses are kept (the AXI-Lite muxes, mailboxes and BINGO manager are
   absent; their ports tied off).
5. **SoC.** CVA6, the L3 memory system, the narrow SPM, the peripherals, the sys-iDMA and the zero memory are absent;
   their SoC xbar ports are tied off. The xbars keep their real port counts, connectivity and address maps.
6. **No host or host traffic.** No boot, no polling cores, no other chip traffic: the links and SoC xbars carry only
   the reduction.
7. **Link control.** The register writes the host makes are made directly on each link's AXI-Lite port (the host
   reaches the memory chips' registers over the D2D links); only the end state is reproduced. The memory nodes'
   control port runs on the 500 MHz clock instead of their 1 GHz host clock (control registers only; no data-path
   effect).
8. **Clocks and reset.** Ideal edge-aligned clocks instead of the `hemaia_clk_rst_controller` dividers; one reset for
   all domains.
9. **No IO pads.** The D2D pins are joined directly; the pad cells are zero-delay assigns in RTL simulation anyway.
10. **Memory chips.** Link only; no HBM, SRAM or push engines behind it (they only forward here).
11. **Cores.** No Snitch core is simulated. The testbench issues what the DM core issues: xDMA CSR accesses (one per
    cluster cycle), iDMA instructions at the iDMA's instruction interface (one per cycle), stores as AXI writes at the
    cluster xbar's CoreReq port (without the core's LSU and `reqrsp_to_axi`), polls at one per one or two cycles. A
    real core issues them slower. `lat_prog_cc` includes the xDMA programming at this rate (~152 cycles per
    ChainGather).
12. **BINGO manager.** Not instantiated: a model with its measured internal latencies behind the real CSR clock
    crossings, sending real dep-set writes. Its task fetch, CERF, power management and descriptor handling are absent.
13. **SIMD.** A model at the RTL's rate (8 FP16 results per cycle), on real TCDM ports; its programming and pipeline
    fill are not modelled.
14. **Software sync.** A fixed `SwHandoffCc` = 250 cluster cycles per hand-off, counted from the flag's landing (work
    the waiting core is still doing when the flag lands overlaps it): the option most favourable to the software arms.

## Results

| file | what | logs |
|---|---|---|
| `results/reduce.csv` | the sweep (`parse.py`): every arm x payload x P, 168 points, all PASS; adapter 7d58489 (`XDMA_ADAPTER_DIR`: no whole-burst buffer, 512 B wide-path bursts) | `logs_new/` |
| `results/reduce_wholeburst_2026-11-01.csv` | the same sweep on the adapter Bender.lock pins (a307833: a remote data burst leaves only once whole) | `results/logs/` |
| `results/reduce_ops.csv` | the `+OP` points (`parse_ops.py`), adapter 7d58489; the paper's `exp/eval2_reduce/reduce_ops.csv` adds the sum arms | `logs_new/`, `logs_swop/` |
| `results/rt_sync.csv` | the round trips | `logs_rt/` |
| (logs only) | the chain's validation and the `+OP` chain and two-stage points on the pinned adapter | `logs/validate.log`, `logs_op/` |
| `results/summary.md`, `results/sanity.png` | `summary.py`, `sanity_plot.py` on `results/reduce.csv` | |

`results/reduce.csv` columns: one row per (arm, P, bytes): `lat_prog_cc`, `lat_cc`, `sram_total`, `sram_max`,
`busiest`, `check`, `levels`, per level `lvl<l>_start_cc`, `lvl<l>_end_cc`, `lvl<l>_sync_cc` (trees and swtree),
`sync_cc` (their sum over l >= 1), `edges` (BINGO dep-set writes), `handoffs` (software flags waited for), `same_as`
(`chain` where the arm is the plain chain at that P), `log`.

**The chain** (128 KiB, `lat_cc`): 2,190 / 2,454 / 2,998 / 4,104 cycles at P = 2 / 4 / 8 / 16, 137 cycles per extra
chip; with the pinned adapter 2,368 / 2,868 / 3,880 / 5,922, 254 per chip. SRAM accesses are exactly 16,384 words x
(P + 1): every chip reads its partial once, the collector writes the result once. With the pinned adapter a hop at
P = 16 costs ~88 cycles of control (the cfg frames reaching every participant and the grant cascading back to the head
before any data moves), 95 cycles of transport (W out of chip s+1 to W into chip s; 104 across the memory chiplet,
0x30 -> 0x20 -> 0x10) and 70 cycles inside the middle xDMA (W in to W out): the adapter holds a remote burst until
its 64 beats are in, and the link's framer holds a write burst until it is whole (`FramerTxWBufBeats` 64, the RTL's;
`LINK_TXWBUF=0` gives 189 cycles a chip, `logs_txw0/validate.log`). The TCDM bank model costs this arm nothing
(`TCDM_BANKED=0` gives identical cycles, `logs_ideal/validate.log`).

**Validation** against the full-package simulation of the same chain (128 KiB, P = 2..16: 3,493 / 3,930 / 4,788 /
6,394 cycles, 207 a chip, intercept 3,101, measured with a link framer that forwards a write burst as it comes): with
that framer (`LINK_TXWBUF=0`) the testbench gives 189 a chip. The full-package intercept is the root core's software
time (the ~85 CSR accesses through its accelerator CSR path, the start, finish and sentinel polls, the cluster
barrier), which the testbench does not run; its SRAM counts are 0.5-0.7 % higher in total (the cores' own accesses).

**Chain or tree (Fig. 10, `lat_prog_cc`).** At P = 16 the two-stage chain beats the chain up to 32 KiB (2,290 vs
2,720) and ties it at 64 KiB (3,314 vs 3,232); at P = 8 it wins at 4 and 8 KiB (1,213 vs 1,230). The binomial tree is
never the fastest: behind the two-stage chain at every P >= 8 and payload, and behind the chain at P = 4. A BINGO edge
costs 105-132 cluster cycles from the giving level's completion to the next level's first programming write: ~33 are
the app's clean-up after the fold, ~2 + 4 scheduler cycles (1 GHz) and the CSR crossings, the rest the dep-set write's
path (~35 cluster cycles over two links).

**Chain against software (Fig. 9, 128 KiB, P = 16, `lat_prog_cc`).** Sum: chain 4,256 against the software tree
43,370 (10.2x), the software chain 55,046 and the root pull 130,438. Softmax statistics: chain 22,544 against the
software tree 92,814 (4.1x). Top-8: chain 4,286 against the software tree 364,042 (84.9x).

**The operators.** Top-k runs within 1.4 % of the sum on every chain and two-stage point both sweeps hold (4 to
128 KiB). SoftmaxStatsJunction costs
~19 cycles per 32-row group against 2 for the sum: it is a single-buffered FSM that waits for both operands' key and
value beats, issues 32 rows at rowsPerCycle = 4 (8 cycles), drains its pipeline (delta*scale, exp lookup, FMA: ~6),
emits 2 beats, and only then accepts the next group. With the pinned adapter each middle hop forwards a burst only
once the operator has produced all of it (bursts are cut at 4 KiB pages), so smx pays min(payload, 4 KiB) / 128 B x
~16.9 cycles more per hop. Long smx chains raise JCT_STATUS bit 1, the junction's starvation watchdog at 4,096
cycles, which the testbench counts as an error but which is not a data error.

**Round trips** (`rt_sync.csv`, TURN_CC = 4, cluster cycles):

| P (hops) | sched 1-to-1 / 1-to-all | flag 1-to-1 | flag 1-to-all, broadcast | flag 1-to-all, unicast |
|---|---|---|---|---|
| 2 (1) | 77 / 77 | 85 | 85 | 85 |
| 4 (2) | 95 / 95 | 101 | 101 | 105 |
| 8 (4) | 129 / 129 | 135 | 136 | 191 |
| 16 (7) | 179 / 179 | 187 | 187 | 302 |

Flags polled in a tight loop come within 4-10 % of the scheduler on the same links; without the routers' duplication
a one-to-all flag takes 1.7x at P = 16. The full-package scheduler arm (`sop_ubench/sync`, arm D) also pays the cores'
dispatch loop and probe kernel, so compare the two arms here with each other, not with arm D.
