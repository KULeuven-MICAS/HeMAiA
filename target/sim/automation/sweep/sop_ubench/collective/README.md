# E2 reduce: in-fabric xDMA chains against software reductions

One button: `./run.sh all` simulates the reduce, then draws the figure.
- `./run.sh parse` re-parses.
- `./run.sh plot` redraws (`plot_coll.py --log` for log y axes against P).

## Question

The paper claims the xDMA ChainGather speeds up the reduce a distributed LLM layer needs: an
in-transit fold, in which the root's partial travels a path of chips and every chip's writer
junction folds that chip's partial into the passing data.

The alternative is what software does without it: the iDMA moves the data, the SIMD adds, and every
hand-off between chips is synchronised in software. Emulating a chain or a tree in software pays a
synchronisation per hop. Two sub-experiments:
- **R1, latency and SRAM accesses against P:** the xDMA chain against the three software reductions.
- **R2, chain or tree:** the xDMA chain against trees of xDMA chains, the binomial tree and the
  two-level tree, checked against a model fitted to the chain (the paper's `exp/eval2_reduce`).

Broadcast, all-gather and all-reduce were part of E2 until 21 October 2026; their apps are gone
and their earlier rounds stay in `results/collective.csv`, skipped by `plot_coll.py`.

## Why offload_legacy

Under `offload_bingo_hw`, a baseline's cross-chip dependencies were carried by the BINGO hardware
scheduler. Under `offload_legacy` every chip runs one device binary and nothing schedules it, so
every hand-off in a baseline is software: point-to-point stores and local polling, the primitive
the software chip barrier (`chip_sync.h`) is built from.

## Platform

The level-2 platform, `hemaia_16c4m_8MBL3_2cluster` (8 MiB L3 per compute chip, 1 MiB TCDM per
cluster; the results so far were measured on `hemaia_16c4m_16MBL3_2cluster`, 16 MiB L3 and
512 KiB TCDM), on its standing simv: a 5 x 4 package, 16 compute chiplets with 2 split clusters
each, the memory column x = 2 in the middle (it only forwards). Only cluster 0 of each chip takes
part; cluster 1 waits in its own L1 until the end, reading one L3 word every ~32 K cycles: the
testbench's LIVE watchdog stops a simulation after 200 K cycles without main-memory traffic
anywhere, and a root folding 15 partials L1 to L1 runs longer than that.
Cluster clock 500 MHz; the legacy host switches the D2D links to DDR for this app, as the level-2
runs do.

P = 2, 4, 8, 16 is a prefix of a snake that walks the compute columns, down one and up the next:

```
0x00 0x01 0x02 0x03 | 0x13 0x12 0x11 0x10 | 0x30 0x31 0x32 0x33 | 0x43 0x42 0x41 0x40
```

P = 2 is a pair, 4 one column, 8 two columns, 16 all. Consecutive chips are D2D neighbours, except
0x10 -> 0x30, which crosses the memory chiplet 0x20.

## Implementations

128 KiB of FP16 per chip, in L1. The sum lands in the root's (0x00's) L1.

| arm | what runs | in |
|---|---|---|
| `xdma` | one ChainGather by the root: its partial travels the snake out to S[P-1] and back, and every writer junction adds that chip's partial (FP16) | R1, R2 |
| `pull` | the root pulls every partial and folds it on its SIMD, pipelined over two 64 KiB slots | R1 |
| `chain` | software chain: receive, add, forward, pipelined in 4 chunks | R1 |
| `tree` | software binomial tree: log2 P levels, each receive behind a clear-to-send and a flag | R1 |
| `xtree` | binomial tree of xDMA chains: log2 P levels, each a two-chip ChainGather through the giver's writer junction, behind a software flag | R2 |
| `x2l` | two-level tree of xDMA chains: groups of 4 snake chips (one compute column; `group` in `params.hjson`) each fold into the member nearest the root with one ChainGather, all at once; then the root runs one ChainGather over the group roots. Software flags member -> group root -> root. P <= 4 is the plain chain, 8 is 4-2, 16 is 4-4; the group roots of P = 16 are 0x00, 0x10, 0x30, 0x40 | R2 |

**Why the chain wins at this size.** Every tree level streams the whole 128 KiB again (about 3.7 K
cycles a level, of which the software sync is about 250), while the chain streams it once and pays
about 207 cycles per extra chip. The model T(L, g) = L T_pass + (L - 1) s + L h (g - 1) puts the
two-level tree ahead of the chain only from P ~ 26 at 128 KiB (`exp/eval2_reduce/plot_model.py`).

## Reads, one source at a time

Measured on RTL: one iDMA that issues pulls back to back to chips on different routes deadlocks the
D2D link. Its reads share one AXI ID, so the responses come back out of order, and the response at
the head of the receiver's single RX stream waits for one queued behind it. The LIVE watchdog then
stops the simulation. This hit a root pulling the parts of 7 chips back to back (P = 8). So the root
fold keeps one source in flight.

## Software sync and hand-offs

- A data hand-off's flag goes out on the iDMA right behind the data, into the same memory, so a
  flag that has landed proves the data has. A remote write's B response is generated at the sender,
  so a flag the core stored after its DMA completed would prove nothing.
- Control signals (GO, READY, clear-to-send, a tree level's flag) are core stores to the other chip
  (`snrt_xchip_writew`). An xDMA fold returns only once the first and the last word of its result
  have landed, so a tree level may flag its sum right after its fold.
- A receive buffer reused within a point (the software tree) also needs a clear-to-send from the
  receiver before the sender may push into it.
- Every flag carries the point's sequence number k, the same on every chip, so a flag left by an
  earlier point never matches and nothing is ever cleared.

## Measurement

- **A point.** A software grid barrier over all 16 chips (the comm-buffer barrier of `chip_sync.h`,
  untimed). The root stamps t0 and writes an SRAM mark, then sends GO to every participant. Each
  participant answers READY and runs its part. An arm that reads other chips' memory from the root
  (the xDMA chain, the root fold) waits for every READY first. The root waits for every READY,
  stamps t1 and writes a second mark. A second grid barrier closes the point; the check and the
  report follow, untimed.
- **Empty points** (GO and READY alone) per P; net latency and net SRAM accesses subtract their
  median. This is conservative for the xDMA chain: the arms that start from the root pay the READY
  round trip, those whose participants act (the trees) start on GO.
- **SRAM accesses** are measured by the testbench's `snax_sram_probe` (`soc_probes.sv`), bound to
  every `snitch_data_mem` (both clusters' L1 and the L3 of every compute chip). The root's mark
  store (`0x5A11_0000 | slot` into its L1) makes every probe print its counters to
  `bin/snax_probes.log`. The memory chiplets are not counted.
- **Repeats.** R = 2 per (arm, P); the figure keeps the faster one (the first pays a cold
  instruction cache), latency and SRAM from the same repeat. The CSV has both.
- **Checks.** The last repeat of every (arm, P) is checked at the root against a numpy golden:
  about 2,000 sampled 32-bit words, one every 64 x ceil(span / 128 KiB) + 4 bytes, so the word
  position inside the 64 B beat rotates. The inputs are small integers, so every fold order is exact
  in FP16.

## Files

- **Device app:** `target/sw/device/apps/snax/snax-coll-reduce`: `src/snax-coll-reduce.c`, and
  `data/params.hjson` (payload, P list, repeats, chain chunks, group, impls) with `data/datagen.py`
  -> `data.h`: every chip's partial in snake order and the sampled sums. Host:
  `target/sw/host/apps/offload_legacy/multi_chip`.
- `run.sh`: one runner through `common/launch.sh` (watchdog, sim slots); the run lands in
  `runs/$COLL_TAG_reduce`.
- `run_staged.sh`: the same build and staging, then `sim_staged.sh` simulates without waiting in
  the shared 3-slot queue (a waiting runner holds the slot lock, so nothing behind it can start).
  `sim_staged.sh` links every Snitch trace to /dev/null except the root DM core's: tracing is
  compiled into the simv, and a polling DM core writes ~200 B per cycle.
- **Rounds.** `e2d_*`: the D2D links in DDR (the legacy host switches them, `-DHEMAIA_D2D_DDR`), the
  figure's round. `e2l_*`: the first round, links left in SDR (the define did not reach the host
  build); kept for reference, plotted with `plot_coll.py --tag e2l`.
- **Adding or completing arms.** The reduce is one simulation of the whole platform, 2-5 h;
  `run_staged.sh` stops it after `COLL_TIMEOUT` (default 5h). The points it finished stay (verdict
  TIMEOUT, kept if no check failed). To run some arms alone, set `P` and `impls` in `params.hjson`
  (only those arms run), run under a tag that starts with the round's, and restore the file. The
  two-level tree:

  ```
  # params.hjson: P: [8, 16], impls: ["x2l"]
  COLL_TAG=e2d_x2l ./run_staged.sh
  ```

  Every run whose name starts with the round's tag joins its figure: per point the run parsed last,
  and in it the faster repeat.
- `parse_coll.py`: the root's `[COLL]` UART lines + the SRAM marks + the root's `Check` lines ->
  `results/collective.csv`.
- `plot_coll.py`: `figs/fig_e2_collectives` (1.1 the reduce against P; 1.2 every arm at P = 16,
  each over the xDMA chain) and `results/collective_summary.csv`.
- `old/`: the dropped gather primitive's runs (`gather_runs.tgz`); the 4 KiB BINGO round (results,
  figures, and the source of the deleted `offload_bingo_hw/multi_chip/workloads/collective`).
