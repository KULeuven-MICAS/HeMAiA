# E2 collectives: in-fabric mechanisms against iDMA collectives with software sync

One button: `./run.sh all` simulates the three primitives (broadcast, reduce, all-reduce), then
draws the figure.
- `./run.sh run reduce broadcast` runs a subset.
- `./run.sh parse` re-parses.
- `./run.sh plot` redraws (`plot_coll.py --log` for log y axes against P).

## Question

The paper claims two mechanisms speed up the collectives a distributed LLM layer needs:
- **xDMA ChainGather:** an in-transit fold. The root's partial travels a path of chips, and every
  chip's writer junction folds that chip's partial into the passing data.
- **D2D router broadcast** (0xFF multicast) with landed flags.

The alternative is what software does without them: the iDMA moves the data, and every hand-off
between chips is synchronised in software. Emulating a chain or a tree in software pays a
synchronisation per hop. How large is the gap, in latency and SRAM accesses, as P grows?

## Why offload_legacy

Under `offload_bingo_hw`, a baseline's cross-chip dependencies were carried by the BINGO hardware
scheduler. Under `offload_legacy` every chip runs one device binary and nothing schedules it, so
every hand-off in a baseline is software: point-to-point stores and local polling, the primitive
the software chip barrier (`chip_sync.h`) is built from.

## Platform

The level-2 platform, `hemaia_16c4m_8MBL3_2cluster` (8 MiB L3 per compute chip, 1 MiB TCDM per
cluster; the results so far were measured on `hemaia_16c4m_16MBL3_2cluster`, 16 MiB L3 and
512 KiB TCDM), on its standing simv: a 5 x 4 package,
16 compute chiplets with 2 split clusters each, the memory column x = 2 in the middle (it only
forwards). Only cluster 0 of each chip takes part; cluster 1 waits in its own L1 until the end,
reading one L3 word every ~32 K cycles in reduce builds only: the testbench's LIVE watchdog stops a
simulation after 200 K cycles without main-memory traffic anywhere, and a root folding 15 partials
L1 to L1 runs longer than that. In the other builds that read stopped every iDMA streaming L3 out to
another chip (the all-gather ring and recursive doubling hung at P = 16, before all-gather was
dropped), so they do not read.
Cluster clock 500 MHz; the legacy host switches the D2D links to DDR for these apps, as the
level-2 runs do.

P = 2, 4, 8, 16 is a prefix of a snake that walks the compute columns, down one and up the next:

```
0x00 0x01 0x02 0x03 | 0x13 0x12 0x11 0x10 | 0x30 0x31 0x32 0x33 | 0x43 0x42 0x41 0x40
```

P = 2 is a pair, 4 one column, 8 two columns, 16 all. Consecutive chips are D2D neighbours, except
0x10 -> 0x30, which crosses the memory chiplet 0x20.

**The broadcast range is set in software.** Before each P, every chip writes its own D2D
multicast-fence register: deliver here if this chip is a participant, forward toward a neighbour
inside the participants' bounding box. A 0xFF multicast then reaches the P participants and no
other chip. Before this, it reached all 16 chips at every P, so at small P the multicast wrote more
SRAM than the pulls.

## Primitives and implementations

128 KiB of FP16 per chip, in L1 (all-reduce's all-to-all arm receives into L3).

| primitive | in fabric | iDMA, software sync |
|---|---|---|
| broadcast | `d2d`: D2D router broadcast into every participant's L1, then a landed flag | `pull`: every destination pulls |
| reduce | `xdma`: one ChainGather by the root, FP16 add in every writer junction; `xtree`: the binomial tree, each level a two-chip ChainGather through the giver's junction behind a software flag | `pull`: the root pulls every partial and folds on its SIMD, pipelined over two 64 KiB slots; `chain`: receive, add, forward, pipelined in 4 chunks; `tree`: binomial tree, log2 P levels, a clear-to-send per reused receive |
| all-reduce | `xdma`: the ChainGather into the root, then one D2D broadcast of the sum; `xtree`: the xDMA tree into the root, then the same broadcast | `a2a`: every chip pushes its partial into every other chip's L3 and folds the P - 1 it receives; `ring`: reduce-scatter then all-gather, 2(P - 1) steps of 128 KiB / P; `rd`: recursive doubling, a clear-to-send per level |

**The ring cycle.** For P <= 8 the ring arms follow the snake prefix, closed by one step back to
0x00. At P = 16 that step would be four hops across the memory column, against the snake's own
step across it, so the ring takes a cycle of single hops and two crossings in different rows:
`0x00 0x01 0x02 0x03 0x13 0x33 0x43 0x42 0x41 0x40 0x30 0x31 0x32 0x12 0x11 0x10`
(`ring: snake` in `params.hjson` keeps the snake).

**Gather and all-gather are not part of E2.** Gather has no in-fabric arm: there is nothing to
fold, and the xDMA chain could only concatenate by OR-ing zero-padded P x 128 KiB buffers at every
hop. All-gather was dropped with it: the paper's mechanisms are broadcast, reduce and
synchronisation, and all-reduce exercises the first two together. Its runs and app are archived
outside the repo (`isca_archive/2026-10-07_allgather`).

## Reads, one source at a time

Measured on RTL: one iDMA that issues pulls back to back to chips on different routes deadlocks the
D2D link. Its reads share one AXI ID, so the responses come back out of order, and the response at the
head of the receiver's single RX stream waits for one queued behind it. The LIVE watchdog then stops the
simulation. This hit a root pulling the parts of 7 chips back to back (P = 8) and all-to-all pulls at
P = 4.
So every pull arm keeps one source in flight, and the all-to-all "load all" arm of all-reduce pushes
(`a2a`) instead of pulling. Pushing is also the faster baseline.

## Software sync and hand-offs

- A data hand-off's flag goes out on the iDMA right behind the data, into the same memory (L1 data,
  L1 flag; L3 data, L3 flag), so a flag that has landed proves the data has. A remote write's B
  response is generated at the sender, so a flag the core stored after its DMA completed would
  prove nothing.
- Control signals (GO, READY, DONE, clear-to-send) are core stores to the other chip
  (`snrt_xchip_writew`).
- A receive buffer reused within a point (the tree reduce, recursive-doubling all-reduce) also
  needs a clear-to-send from the receiver before the sender may push into it.
- Every flag carries the point's sequence number k, the same on every chip, so a flag left by an
  earlier point never matches and nothing is ever cleared.

## Measurement

- **A point.** A software grid barrier over all 16 chips (the comm-buffer barrier of `chip_sync.h`,
  untimed). The root stamps t0 and writes an SRAM mark, then sends GO to every participant. Each
  participant answers READY and runs its part. An arm that reads or writes other chips' memory from
  the root (the xDMA chain, the root's multicast, root pulls) waits for every READY first. The root
  waits for every DONE (the primitives whose result lands on every participant) and every READY,
  stamps t1 and writes a second mark. A second grid barrier closes the point; the check and the
  report follow, untimed.
- **Empty points** (GO and READY alone) per P; net latency and net SRAM accesses subtract their
  median. This is conservative for the in-fabric arms: the arms that start from the root pay the
  READY round trip, those whose participants act start on GO.
- **SRAM accesses** are measured by the testbench's `snax_sram_probe` (`soc_probes.sv`), bound to
  every `snitch_data_mem` (both clusters' L1 and the L3 of every compute chip). The root's mark
  store (`0x5A11_0000 | slot` into its L1) makes every probe print its counters to
  `bin/snax_probes.log`. The memory chiplets are not counted.
- **Repeats.** R = 2 per (arm, P); the figure keeps the faster one (the first pays a cold
  instruction cache), latency and SRAM from the same repeat. The CSV has both.
- **Checks.** The last repeat of every (arm, P) is checked against a numpy golden: about 2,000
  sampled 32-bit words per result, one every 64 x ceil(span / 128 KiB) + 4 bytes, so the word
  position inside the 64 B beat rotates. The inputs are small integers, so every fold order is exact
  in FP16. For all-reduce every participant checks.

## Files

- **Device app:** `target/sw/device/apps/snax/snax-coll/src/snax-coll.c`, built once per
  primitive by `snax-coll-{broadcast,reduce,allreduce}` (`COLL_PRIM`; the source still has the
  dropped gather and all-gather primitives). Host:
  `target/sw/host/apps/offload_legacy/multi_chip`.
- **Data:** `snax-coll/data/params.hjson` (payload, P list, repeats, chain chunks) and
  `datagen.py` -> `data.h`: every chip's partial in snake order and the sampled sums.
- `run.sh`: one runner per primitive through `common/launch.sh` (watchdog, sim slots); runs land in
  `runs/$COLL_TAG_<primitive>`.
- `run_staged.sh [primitive ...]`: the same build and staging, then `sim_staged.sh` simulates without
  waiting in the shared 3-slot queue (a waiting runner holds the slot lock, so nothing behind it can
  start). `sim_staged.sh` links every Snitch trace to /dev/null except the root DM core's: tracing is
  compiled into the simv, and a polling DM core writes ~200 B per cycle.
- **Rounds.** `e2d_*`: the D2D links in DDR (the legacy host switches them, `-DHEMAIA_D2D_DDR`), the
  figure's round. `e2l_*`: the first round, links left in SDR (the define did not reach the host
  build); kept for reference, plotted with `plot_coll.py --tag e2l`.
- **Completing a run.** A primitive is one simulation of the whole platform, 2-5 h; `run_staged.sh`
  stops it after `COLL_TIMEOUT` (default 5h). The points it finished stay (verdict TIMEOUT, kept
  if no check failed). The missing ones run alone: in `params.hjson` set `P: [16]`, `repeats: 1`
  and `impls: [...]` (only those arms are compiled in), then
  `COLL_TAG=e2d_p16 COLL_TIMEOUT=8h ./run_staged.sh allreduce`, and restore the file. Every run
  whose name starts with the round's tag joins its figure: per point the run parsed last, and in it
  the faster repeat.
- `parse_coll.py`: the root's `[COLL]` UART lines + the SRAM marks + every UART's `Check` lines ->
  `results/collective.csv`.
- `plot_coll.py`: `figs/fig_e2_collectives` (1.1 broadcast, 1.2 reduce, 1.3 all-reduce against P;
  1.4 the three at P = 16, each arm over the in-fabric one) and
  `results/collective_summary.csv`.
- `old/`: the dropped gather primitive's runs (`gather_runs.tgz`); the 4 KiB BINGO round (results, figures, and the source of the deleted
  `offload_bingo_hw/multi_chip/workloads/collective`).
