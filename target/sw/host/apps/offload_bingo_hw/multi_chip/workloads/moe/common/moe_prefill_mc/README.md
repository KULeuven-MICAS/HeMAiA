# moe_prefill_mc: the MoE prefill block

`MBuild` builds DeepSeek-V2-Lite's layer-1 MoE over a prompt of T tokens on the 16 + 4 chiplet platform: every GEMM
on VersaCore's (8, 8, 16) shape (array shape 3) with INT4 weights in the parity layout (the router's INT8). It is
the driver of the ablation workloads (`../../dsv2_l1_prefill_ablation/<rung>`) and the MoE block of the whole layer-1
prefill (`../../../dsv2/common/dsv2_layer_prefill_mc.py`, `MoePart`). The platform, the weight rings and the stamps
are `chiplet_mc.py`'s; the data and the golden `../moe_prefill_data.py`'s.

| module | what it holds |
|---|---|
| `__init__.py` | `MBuild` (the mixins below over `ChipletBuild`), its `build()` in phase order, `run()` |
| `params.py` | every switch, parsed and checked |
| `placement.py` | the experts on the chips, their columns on a chip's clusters, the chunks and slices |
| `memory.py` | the data, the golden, the HBM images of the weights, the L3 areas, every cluster's L1 buffers |
| `tasks.py` | a weight chunk's load and release, one GEMM, the HBM casts, the residual add, the args phases |
| `x_route.py` | the window's start, X from HBM, the route, the shared MLP's first pass over X's row blocks |
| `gate_up.py` | the shared MLP's later passes, the experts' gate\|up, the quantise, the A8 exchange |
| `down.py` | the down projection, the reduce into the root, the stores of Y |
| `checks.py` | the checks and `finish()` |
| `common.py` | the constants and helpers `moe_decode_mc` shares |

## X and the route

X and the router's weights sit in the HBM of the root's memory chip. The root's host queues that memory chip's push
engine (`hbm_bcasts`, `ChipletBuild.hbm_cast`): the router's weights behind one flag, then X a tile of 8 tokens at a
time, each behind a flag of its own, multicast into every chip's L3, so a cluster starts on a tile as soon as it lands.
Every chip's weight rings start once X has landed: the multicast has the half-duplex D2D links to itself.

Every cluster routes its own run of 8-token tiles (`rt_tiles`, at most 32 rows, in one row block of X): it copies the
tiles from the chip's L3 as soon as they land, then the router GEMM, its column factors, the softmax and the top 6 on
its cores, and puts the tiles' entries and a flag into the root's landing area. Per row block of X, the root waits for
the block's flags and sends the block's table on to every chip (one multicast source: the first after X's last tile has
landed on the root); every cluster fetches each block's table and appends its tokens to the row lists of its chip's
experts (`moe_dispatch`). The uploads, the root's table and the row lists are made behind each cluster's copies of the
first row blocks.

## The placement

Static: 4 experts a chip, most padded rows first onto the chip with the least work so far, each expert's intermediate
columns split over the chip's two clusters by 16-column blocks (cluster 0 an even number), moved until the clusters'
work is even. The shared MLP: each chip an even run of its column blocks, split the same way. Every expert's row count
(its tokens, padded to whole 8-row tiles) is fixed here, from the generator's own route; `moe_dispatch` checks the
device's route against it. A chip's expert slots run in `slot_order`.

## Per cluster

In the order the arrays run it:

| step | |
|---|---|
| shared gate\|up | per row block of X (copied from the chip's L3 into a row buffer at a padded row pitch): one GEMM per weight chunk, its dequantisation and SwiGLU, quantised into the shared A8 right after each block; the first chunk over every block as X arrives, chunk j after expert slot `shared_after[j - 1]` |
| experts gate\|up | per expert slot: its rows of X gathered from L3 by the row lists (`idma_gather_rows`), in row blocks of at most `expert_rows`, one GEMM per weight chunk, dequantisation and SwiGLU |
| quantise | every row times its route weight (the shared MLP's: 1) into the chip's row-major A8 of the MLP at the cluster's columns; then each cluster copies its columns into the other cluster's A8, one slot later |
| down | per output slice of the cluster's half of the outputs: the shared MLP's GEMM over the chip's shared columns writes the slice, dequantised; then each expert's GEMM over all its columns, dequantised, its rows added into the rows of their tokens (`simd_scatter_add_rows`) |
| reduce | one ChainGather a slice through every chip's same cluster into the root's cluster (`chain_path`), after the previous gather of either cluster, the two clusters' order alternating slice by slice; the last slice's two run at once, once every push has landed; the root stores the slice |

Each weight chunk's scale tile is copied from its slab into a small buffer right after its load, so the slab is free
once the chunk's GEMMs are done. In a whole layer, one chip adds the residual into its partial of every slice
(`resid`). Y lies over X's source and landing in the root's L3.

## The ablation

`x_cast`, `reduce` and `sync` take the paper's three mechanisms off one at a time on the same mapping (`params.py`):
X, the router's weights and the route table by multicast or one copy a chip; ChainGathers or the root's pulls and adds;
BINGO hardware edges or chip barriers, after every kernel (`barrier_at kernel`) or before each slice's reduce (`cross`).

## Checks

The root's host compares, byte for byte after the end stamp: the route table, the root cluster 1's row lists, the root
cluster 0's A8 of its first expert slot and of the shared MLP (row 0), and three windows of Y.

Timing: the start stamp after every chip's setup; the end stamp after the last store.

## Parameters

Beside the platform's and the graph's switches (`chiplet_mc.py`: `bcast bcast_l1`, `reduce`, `sync`, the rings, ...):

| switch | default | |
|---|---|---|
| `x_cast` | `multicast` | `unicast`: X, the router's weights and the table one copy a chip |
| `barrier_at` | `kernel` | with `sync barrier`: `kernel` or `cross` |
| `args_phases` | false | the block alone: args phases as a layer has them, one boundary before the down projection |
| `blocks` | 3 | gate blocks a gate\|up chunk |
| `slice` | 128 | output columns a down slice |
| `x_blocks` | 4 | row blocks of X |
| `expert_rows` | 0 (off) | the most rows a gate\|up pass of an expert slot takes |
| `shared_after` | none | the slot after which each later shared chunk's pass runs |
| `slot_order` | the placement's | a chip's slots as ranks from the fewest padded rows |
| `chain_path` | the fold cycle | the chips every ChainGather passes, hex ids, the root last |
| `weights`, `seed`, `n_calib`, `n_cal_act`, `inter_e`, `inter_sh` | | the data (`moe_prefill_data.py`) |
