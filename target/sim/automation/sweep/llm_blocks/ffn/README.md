# L2 FFN: a real LLM FFN on 16 compute chiplets + 4 memory chiplets, with an ablation

One button: `./run.sh all` fetches the weights and runs every variant.
- `./run.sh run dsv2_l0/all dsv2_l0/base` runs a subset.
- `./run.sh parse` re-parses into `results/ffn.csv`.

## Question

The paper's three mechanisms, each against its conventional counterpart on the same hardware and
the same compute:
- **broadcast**: one D2D router multicast into every cluster's L1 and a landed flag, against
  every cluster pulling the root's copy with its iDMA;
- **reduce**: the xDMA ChainGather folding in transit, against the root pulling every partial and
  folding it on its SIMD;
- **sync**: the BINGO hardware scheduler resolving every cross-chip dependency per edge, against
  barriers: the vanilla HeMAiA chip barrier at every cross-chip sync point.

How much does each buy inside a real LLM layer whose weights stream from the memory chiplets,
alone and together?

## Workload

DeepSeek-V2-Lite's dense FFN (layer 0): `x[2048] -> gate, up (2048 -> 10,944) -> SwiGLU -> down
(10,944 -> 2048)`, one token.
- **Weights are the checkpoint's own.** `fetch_weights.py` caches gate, up, down and the RMSNorm
  gain in front of them (BF16, pinned commit) in `cache/`.
  - Quantised INT4 symmetric per output column, 32.06 MiB in all.
  - The gain is NOT folded into gate/up rows, unlike the dsv2 golden. On this checkpoint the gain
    spans 30x across channels. Folded, a few rows set every column's scale: cosine to the float
    FFN drops from 0.97 to 0.67, and the INT4 zero fraction doubles (22% to 44%).
- **The token is a seeded normal draw** (as the dsv2 golden draws hidden states). Static INT8 scales
  are calibrated on 8 more draws.
- **Device vs the float FFN:** cosine 0.973, max |err| / max |ref| = 0.084 (INT4 weights, INT8
  activations).

## Platform

`hemaia_16c4m_8MBL3_2cluster`: a 5 x 4 grid whose middle column is memory, 2 split clusters per
compute chiplet.

```
C00 C10 M20 C30 C40
C01 C11 M21 C31 C41      each memory chiplet feeds the four compute chips of its row,
C02 C12 M22 C32 C42      one push engine each (engines 0..3)
C03 C13 M23 C33 C43
```

## Mapping (tensor parallel, `workloads/ffn/common/ffn_mc.py`)

| step | where | what |
|---|---|---|
| broadcast | root -> every cluster | x8 (4 KiB, a_row) into every cluster's L1 |
| gate, up | every cluster | its 352 or 320 intermediate columns of both, one INT4 GEMV, dequantised |
| SwiGLU | every cluster | silu(gate) * up, quantised to INT8 |
| down | every cluster | its ROWS of down: a partial y over all 2048 outputs (FP16) |
| local fold | each chip | cluster 1 pushes its partial to cluster 0, whose SIMD adds them |
| reduce | 16 chips -> root | the chips' partials folded in the fold order |
| dequant | root | y16 = y_raw (.) s_dn |

- **Column split:** 342 pairs of 32 columns over 32 clusters. The 22 leftover pairs go one per
  memory link in turn, so each link carries 1,376 or 1,344 columns.
- **Fold order:** a Hamiltonian cycle through the root with two memory-column crossings:
  `00 10 11 12 32 31 30 40 41 42 43 33 13 03 02 01`.
  - Both reduce implementations fold in this order, so one golden checks both, byte for byte.
- **Symmetric L1:** the chain names other chips' buffers by the root's own address. Every
  cluster's collective buffers are therefore one allocation, `AAAA_ffn_sym`, at offset 0 of the
  static arena. The generator refuses a build where that does not hold.

## Variants (the ablation)

Every combination of the three mechanisms: what is broadcast is the token x (INT8, 2 KiB); what is reduced is every chip's partial y (FP16, 4 KiB).

| variant | rung | broadcast | reduce | sync |
|---|---|---|---|---|
| `base` | R0 | iDMA pulls | root gathers | chip barriers |
| `bcast` | R1 | multicast | root | chip barriers |
| `bcast_reduce` | R2 | multicast | ChainGather | chip barriers |
| `all` | R3 | multicast | ChainGather | hardware, per edge |
| `reduce_sync` | R3 without broadcast | pulls | ChainGather | hardware |
| `bcast_sync` | R3 without reduce | multicast | root | hardware |
| `reduce` | reduce alone | pulls | ChainGather | chip barriers |
| `sync` | sync alone | pulls | root | hardware |

**Barrier sync, the baseline** (`sync: barrier`, `barrier_mode: bcast`). Inside the window no
task depends on another chip's. The vanilla HeMAiA chip barrier (`snrt_hw_chip_global_barrier`,
`chip_sync.h`) stands in for those dependencies: one barrier task on every chip's cluster-0 DM
core, which announces its arrival with one 0xFF broadcast store and polls the others' locally.
Every chip waits for every chip, which is what the per-edge hardware sync removes. Barriers sit
after the start anchor, before a pull broadcast of a value made inside the window, and before the
reduce; a multicast needs none (its landed flag says the data is there). Barriers are chained per
chip, so every chip runs them in program order.
**Hardware sync** (`sync: edge`): every cross-chip dependency is a BINGO hardware edge, so a task
waits for exactly its producers.

Weights stream plain (`weight_crest: false`): CREST is not part of the paper, and the paper's
cluster has no decompressor. The first round, a CREST ladder on the 16 MiB-L3 platform, is in
`results/round1/` (its runs and variants: `isca_archive/2026-10-07_l2_round1`).

## Measurement

- **Anchors.** Two DM-core tasks on chip 0x00 stamp mcycle and store an SRAM-probe mark:
  - start: after every chip's setup;
  - end: after y16.
- **What is inside the window.** The weight prefetchers start after the start anchor, so the whole
  weight stream counts. `anchor_reduce: true` adds a third anchor once every chip's partial is
  ready. It is off by default, because it makes the root's pulls wait for the last chip.
- **SRAM accesses** come from the testbench probe (see `sweep/sop_ubench/collective/README.md`): the
  difference of the two anchors' snapshots.
- **Checks.** The root compares y16, y_raw and one chip's local partial with the golden, byte for
  byte.

## Round 1 results (the CREST ladder on the 16 MiB-L3 platform; `results/round1/`)

| variant | rung | latency, us | vs R0 | x on every cluster | last down GEMV | reduce after last partial | SRAM, M (L1 / L3) |
|---|---|---|---|---|---|---|---|
| `base_barrier` | R0 | 280.6 | 1.00x | 9.9 | 248.5 | 28.1 | 4.39 / 8.82 |
| `crest_barrier` | R1 | 232.5 | 1.21x | 9.8 | 199.8 | 28.8 | 7.71 / 7.13 |
| `full_barrier` | R2 | 232.1 | 1.21x | 6.2 | 212.2 | 15.4 | 7.71 / 7.14 |
| `full` | R3 | 202.0 | 1.39x | 2.7 | 187.7 | 10.4 | 7.71 / 7.09 |
| `crest` | R3 - collectives | 217.4 | 1.29x | 5.3 | 188.5 | 24.8 | 7.71 / 7.11 |
| `coll` | R3 - CREST | 262.9 | 1.07x | 2.7 | 248.2 | 10.4 | 4.39 / 8.81 |
| `base` | R0 + per-edge sync | 272.0 | 1.03x | 5.1 | 245.5 | 22.4 | 4.39 / 8.82 |

- **CREST** is the lever at one token: 272.0 -> 217.4 us with per-edge sync; the stream (first
  chunk to last down GEMV) 231 -> 176 us for 1.23x fewer bytes. The memory links deliver about
  18 GB/s each.
- **Collectives**: x by multicast (2.7 us vs 5.1-5.3), the reduce in transit (10.4 us vs
  22.4-24.8). 9.1 us on plain weights, 15.4 us with CREST: fixed costs at both ends of the stream.
- **Per-edge sync vs barriers**: the barriers' own latency is 8-10 us (the start release delays x
  by 3.5-4.8 us, the barrier before the reduce adds 4.0-5.7 us). The rest is the stream: the
  barrier releases every prefetcher at once and the memory chips' push arbitration reorders, so
  the slowest row changes (`full`: row 0, chip 40, 187.7 us; `full_barrier`: row 1, chips 01/11,
  212.2 us). Report the two parts separately.
- **SRAM**: CREST costs 12 % more accesses (the in-place expansion in L1), the collectives none at
  these payloads.
- **The fold order**: the FFN's partials (raw D-port outputs) sum to the same bytes in any order,
  so the chain and root reduces pass one golden; the golden still follows each arm's order
  (`ffn_mc.py`), which matters for MoE and FA.

Figure: `results/ffn_ablation.pdf` (`plot_l2.py`, all three workloads).

## Files

- `fetch_weights.py`: the weights into `cache/` (uses `sweep/common/weights/fetch_weights.py`'s HTTP helpers).
- `run.sh`: the runs, through `common/launch.sh` (watchdog, sim slots).
- `parse_ffn.py`: UART SYNC lines + SRAM marks -> `results/ffn.csv`; `trace_ffn.py`: every
  task's span from the core traces, and the phase milestones.
- `plot_l2.py`: the level-2 figures (FFN, MoE, FA) and `results/l2_summary.json`.
- **Workloads:** `target/sw/host/apps/offload_bingo_hw/multi_chip/workloads/ffn/`.
  - `common/ffn_mc.py`: the driver.
  - `common/ffn_data.py`: weights, pack, scales, golden.
  - `common/ffn.mk`: the build.
  - `dsv2_l0/<variant>/params.hjson`: one directory per variant.
