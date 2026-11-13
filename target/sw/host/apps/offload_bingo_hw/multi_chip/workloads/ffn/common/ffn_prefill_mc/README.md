# ffn_prefill_mc: the FFN prefill block

The shared driver of the FFN prefill workloads (`workloads/ffn/<model>_prefill/<variant>/`): one LLM's dense SwiGLU FFN
over a prompt of T tokens, on the (16, 4, 16) GEMM shape with INT4 weights, tensor-parallel over every cluster of the
16 + 4 chiplet platform. The platform, the weight rings, the stamps and the setup are `chiplet_mc.py`'s; the data and
the golden are `ffn_data.py`'s. `ffn_decode_mc` runs it for one token; the whole-layer drivers
(`dsv2/common/dsv2_layer_prefill_mc.py`, `llama3/full_layer_prefill/common/llama3_layer_mc.py`) run it inside a layer.
The stage apps (`<model>_prefill/stage*/main_bingo.py`) are standalone programs of its stages.

| module | what it holds |
|---|---|
| `__init__.py` | `PBuild`, composed of the mixins below on `chiplet_mc.ChipletBuild`; `run()` |
| `params.py` | every switch, parsed and checked (`PBuild.__init__`) |
| `memory.py` | the data, the plan (column split, gate\|up chunks, down slices), the HBM images, the L1 buffers |
| `x.py` | X: its landing area, its multicast from HBM, its fetches; Y_raw's places in L3 |
| `gate_up.py` | the weight loads, gate\|up, the post-processing into the chip's A8 |
| `passes.py` | the window and the passes (sync edge): the down slices, the ChainGathers, the stores of Y_raw |
| `kernel.py` | params `sync barrier`: the passes kernel by kernel and the reduce as one kernel |
| `checks.py` | the data checks, the buffer-hazard proof, `finish()` |

## The mapping

Cluster cl takes the intermediate columns J_cl (whole column blocks), the same columns of gate and of up. The tokens
run in passes of `rows` (T / rows passes). In a pass, every cluster holds the pass's X in L1, and each gate|up weight
chunk meets all of the pass's rows in one GEMM as it streams in (the single pass: each weight is loaded once a pass). A
chip's two clusters then split the down projection by output columns: each takes the chip's columns (K) for its half of
the outputs, so a slice of the down output holds the chip's whole partial and the chips' partials are reduced by a
ChainGather into the root. Per pass:

```
X           the pass's rows (INT8, A layout) into every cluster's L1
gate|up     per weight chunk: GEMM X . [gate | up] chunk -> G (D layout, FP16), dequantised (x the chunk's column
            scales), SwiGLU into A16 (D layout)
A8          A16 relaid out D -> A on the xDMA, quantised to INT8 into the chip's A8, and the cluster's columns of it
            copied into the other cluster's A8
down        per weight chunk (a slice of the cluster's outputs): GEMM A8 . down chunk -> the chip's partial of the
            slice, into an output slot of the cluster's exchange buffer
reduce      per slice (or group of slices): one ChainGather through every chip, each hop's xDMA adding its partial
            in transit, into the root's fold buffer; the root stores it to L3
```

**X comes from HBM.** Every pass's X sits in the HBM of the root's memory chip; the root's host posts into that memory
chip's push engine (the one that also pushes the root's weights), which multicasts X a tile at a time into every chip's
L3, each tile behind a flag (`ChipletBuild.hbm_cast`). Every pass's X is multicast at the start, ahead of the root's
weight rings in that engine's queue. Every cluster copies a pass's X from its own L3 into L1 once the pass's flags have
landed.

**The weights.** Each cluster's gate|up and down chunks sit in its memory chip's HBM; the memory chip pushes them into
the chip's L3 ring and the DM core loads each into an L1 slab behind its landed flag. The first pass loads them from the
ring; a later pass copies each from the slot it landed in (the ring holds every chunk of a cluster), or with `restream`
loads them from the ring again (weights larger than the ring).

**The program order.** The compiler orders every device core's program (mini_compiler `passes/bingo_dfg_schedule.py`)
and splits the GEMMs, SIMD and xDMA tasks into configure / start / wait parts where its schedule gains; the edges this
driver adds are every ordering the data needs. `check_hazards` proves every reused L1 buffer slot ordered.

**Timing.** The start stamp after every chip's setup (the scales loaded); the end stamp after the root's last stores.
X's multicast and the weight stream are inside the window.

## Parameters

Beyond the platform's and the graph's switches (`chiplet_mc.py`: `num_chiplets`, `num_clusters`, `static_l1`,
`weight_ring`, `w_chunk_bytes`, `w_buffers`, `wbits`, `bcast` (only `bcast_l1`), `reduce`, `sync`, ...):

| switch | default | what it sets |
|---|---|---|
| `tokens`, `rows` | -, 32 | the prompt, and the rows a pass (a multiple of 16 dividing `tokens`) |
| `model`, `seed`, `weights`, `inter` | dsv2-lite-l0, 1 | the FFN's data (`ffn_data.FFNData`) |
| `blocks`, `first_blocks` | 3, 0 | 16-column gate blocks a gate\|up chunk; the first chunk's width (0: `blocks`) |
| `dn_cols` | 512 | output columns a down chunk (whole 32-column pairs) |
| `root_cols` | 0 | the root chip's clusters' intermediate columns (one number, or one a cluster; 0: an even split) |
| `dn_balance` | false | each cluster's share of the outputs set by the array steps |
| `x_cast` | multicast | X from HBM by one multicast, or `unicast` a copy a chip (the ablation) |
| `reduce` | chain | ChainGathers, or `root` (sync barrier): the root pulls every partial and folds it on its SIMD |
| `sync` | edge | every cross-chip dependency a BINGO edge, or `barrier`: a chip barrier after every kernel |
| `deq_per_row` | false | the dequantise one task a row tile |
| `x_blocks` | 1 | the first pass's X fetched in this many row blocks, the first gate\|up chunk one GEMM a block |
| `post_blocks` | 1 | A16's reshape and quantise in this many row blocks |
| `split_last` | false | the last gate\|up chunk one GEMM a post block, its post-processing right behind each |
| `split_first_dn` | false | the first down slice one GEMM a post block, each behind its block's A8 rows |
| `restream` | false | every pass streams its weights from the memory chip again |
| `chain_alt` | false | cluster 1's ChainGathers run the chips in reverse order |
| `l3_overlay` | false | Y_raw of a pass stored over that pass's X buffers in L3 |
| `y_scatter` | false | Y_raw stored row tile by row tile into the L3 of the tile's owner chip |
| `out_slots` | 2 | output slots of each cluster's exchange buffer (slice j into slot j mod out_slots) |
| `chain_group` | 1 | consecutive down slices one ChainGather carries |
| `start_per_chip` | false | a chip's scale loads chained, the start waiting for one task a chip |
| `gather_relay_only` | false | an over-budget dep-tag cell of the gathers relays its edges instead of ordering them |
