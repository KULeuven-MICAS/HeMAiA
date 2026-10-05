# DeepSeek-V2-Lite on sixteen chiplets

DeepSeek-V2-Lite layer 1, one token of decode, INT4 weights, on a 4 × 4 package of twelve compute chiplets
and four memory chiplets. It is the scaling point after [`../six_chiplet`](../six_chiplet), and the
platform for its ablation study: the same stage structure, the same driver
([`../common/dsv2_mc.py`](../common/dsv2_mc.py)) and the same settings wherever the platform allows.

## The platform

`target/rtl/cfg/hemaia_sixteenchiplet_16MBL3_2cluster.hjson`. Each compute chiplet has two clusters and
16 MiB of L3, and each memory chiplet has its HBM. The L3 and the memory chiplets are configured as in
`hemaia_sixchiplet_16MBL3_2cluster_l3b64.hjson`.

```
            x=0      x=1      x=2      x=3
    y=0    C 00  -  C 10  -  M 20  -  C 30
    y=1    M 01  -  C 11  -  C 21  -  C 31
    y=2    C 02  -  C 12  -  C 22  -  M 32
    y=3    C 03  -  M 13  -  C 23  -  C 33
```

Every memory chiplet sits on the package edge and feeds the three compute chiplets beside it, each over
its own link with a push engine of its own. So every compute chiplet has exactly one memory chiplet
beside it, and its weights never cross another compute chiplet's link:

| memory chiplet | feeds |
|---|---|
| `0x20` | `0x10`, `0x21`, `0x30` |
| `0x01` | `0x00`, `0x02`, `0x11` |
| `0x32` | `0x22`, `0x31`, `0x33` |
| `0x13` | `0x03`, `0x12`, `0x23` |

## The mapping, and how it differs from six_chiplet

Tensor parallel, as on six chiplets: every weight is split by output columns over the clusters, in whole
32-column pairs, and every link streams its compute chiplet's share. Three things differ, all forced by
the platform:

- **The per-head work runs on eight of the twelve chiplets.** This covers W_Q, the RoPE, W_UK and W_UV.
  Sixteen heads do not divide over 24 clusters, so the driver puts one head on each of 16 clusters: two
  head chiplets per memory chiplet (`0x00`, `0x02`, `0x10`, `0x21`, `0x22`, `0x31`, `0x03`, `0x12`).
  The params key `head_chips` overrides the choice.
- **The routed experts are split over every cluster** (`expert_map: tp8`). The six-chiplet `colgroup`
  map pairs exactly two columns of compute chiplets.
- **No expert's all-gather is deferred** (`expert_gather_lag: 0`). Each gather collects from twelve
  chiplets, and keeping two in flight needs more dependency tags than the manager has.
- **Every all-gather runs in two levels** (`gather_tree: 3`, with `gather_chain` and
  `prune_implied`). Each row of three compute chiplets gathers onto its first chiplet, and every
  destination then reads the four rows' ranges. A flat gather has every collector wait on one task
  per chiplet. Once two gathers overlap, that needs 24 dependency tags in one cell, against the
  manager's 16; even 32 tags (5-bit tags) do not fit. With two levels, a collector waits on one
  task per row, and the whole layer fits 16 tags.
- **104 KiB weight chunks, and the static L1 plan capped at 470,000 bytes** (`w_chunk_bytes`,
  `l1_capacity`). With 24 clusters the runtime's own L1 tables grow. At six_chiplet's 112 KiB
  chunks, one cluster's static plan left too little for them: chip `0x22`'s cluster 0 failed an L1
  allocation at boot.

## What the platform needs from the D2D link

Stage 8 needs a fix in `hw/hemaia/hemaia_d2d_link` (`network_framing/hemaia_d2d_link_framer.sv`;
uncommitted at the time of writing). Received read requests wait in a queue of their own, not at the head of the
framer's receive stream.

Without the fix, two chips that read each other's L3 at once can deadlock:
- each holds a read request at its receive head;
- that request waits for its own interconnect, which takes no new read while its responses cannot
  leave;
- those responses sit behind the other chip's waiting request.

Here this happened between two row leaders of the all-gather tree. The six-chiplet platform never
filled the interconnect's read limit, so it never hit it. `make sim-framer-mutual-reads-tb` in that
repository reproduces it: the old framer times out, the fixed one passes.

## Stages

One directory per stage, each building on the one before, as in `../six_chiplet`. `params.hjson` is the
same in every directory, except stages 3 and 4, which build W_DKV where they can check it.

| directory | adds | checked |
|---|---|---|
| `stage1_rmsnorm` | the token's norm and INT8 pack, on every compute chiplet | x8, every chiplet |
| `stage2_wq` | W_Q, by head | q16, every head cluster |
| `stage3_wdkv` | W_DKV, on the latent's cluster | kv16 |
| `stage4_latent_rope_wuk` | the latent's norm, RoPE, W_UK (absorbed) | cn, c8, kpe8, the rotated rows, q~ |
| `stage5_attention` | the cache append and MLA attention, on one cluster | q8, the attention's output |
| `stage6_wuv_wo` | W_UV (absorbed), W_O, the residual | o per head, attn and h per cluster |
| `stage7_router` | the post-attention norm and the router, on every chiplet | hn, logits, probabilities, the route |
| `stage8_experts` | the shared and six routed experts, the combine | the layer's output, every cluster |
| `full_layer` | stages 1–8, the layer's settings | as stage 8 |

## Running a stage

The same as `../six_chiplet`: build the RTL for the cfg above once, then run a stage's workload, e.g.
`WORKLOAD=dsv2/sixteen_chiplet/stage1_rmsnorm` with `HOST_APP_TYPE=offload_bingo_hw`,
`CHIP_TYPE=multi_chip` and `DEV_APP=snax-bingo-offload`.

## Status

The platform's broadcast region is verified on RTL: `host_only/multi_chip/d2d_broadcast_region`
passes. Every compute chiplet receives every broadcast, and no memory chiplet consumes one.

On RTL (VCS), every check byte-exact. All stages were run with the params in this directory, on RTL with the D2D framer fix
(5 October):

| stage | checks |
|---|---|
| `stage1_rmsnorm` | 12 / 12 |
| `stage2_wq` | 28 / 28 |
| `stage3_wdkv` | 29 / 29 |
| `stage4_latent_rope_wuk` | 64 / 64 |
| `stage5_attention` | 67 / 67 |
| `stage6_wuv_wo` | 131 / 131 |
| `stage7_router` | 146 / 146 |
| `stage8_experts` (the whole layer) | **200 / 200** |

All stages through `stage8_experts` generate, and they pass the compiler's dependency-tag fit and its
hang check.

## The layer, measured (stage 8, 5 October)

Measured from the first task to the last compute task, checks excluded, by the same Gantt analysis
as the six-chiplet numbers:

| | |
|---|---|
| **layer** | **428.0 µs** |
| against one chiplet (1705.8 µs) | 3.99× |
| against the best six-chiplet layer (508.5 µs) | 1.19× |
| weight pushes, per compute chiplet | 2.0–3.6 MiB at 27.8–28.5 GB/s; the link pushes 75–135 µs of the 428 |
| GEMM busy, per cluster | 10–33% |

**What bounds it.** At sixteen chiplets the layer is bound by synchronization and collectives, not by
the weight stream:
- the route is decided at about 210 µs, as on six chiplets — attention, the query gathers, `h`'s
  gather and the router do not shrink with the number of links;
- the experts take from about 224 µs to the end. Every routed slot is gathered over twelve chips in
  two levels, six times in series, and the shared expert's gather alone takes about 45 µs.

That makes the platform a good one for the ablation: the barrier and multicast arms should show here
what they cost.
