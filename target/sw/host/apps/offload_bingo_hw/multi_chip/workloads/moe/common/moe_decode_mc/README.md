# moe_decode_mc: the MoE decode block

`DBuild` builds DeepSeek-V2-Lite's layer-1 MoE for ONE token (a decode step) on the 16 + 4 chiplet platform, every
weight multiply on VersaCore's one-token GEMV: the router on the device, a static placement the device's route is
checked against, the weight rings, the A8 exchange between a chip's two clusters, the down projection by output
slices and the ChainGather reduce into the root. It is the MoE block of the whole layer-1 decode
(`../../../dsv2/common/dsv2_decode_mc.py`). The data and the golden are `../moe_prefill_data.py`'s at one token; the
route table's and the row lists' bytes and the constants are `../moe_prefill_mc/common.py`'s.

| module | what it holds |
|---|---|
| `__init__.py` | `DBuild` (the mixins below over `ChipletBuild`), `run()` |
| `params.py` | the switches |
| `placement.py` | the pieces of the token's experts and of the shared MLP on the chips, their columns on the clusters |
| `memory.py` | the data, the golden, the HBM images (split at the route), the L3 areas, every cluster's L1 buffers |
| `build.py` | the task graph, phase by phase |
| `checks.py` | the checks |
| `common.py` | the a_row, the spread of an A16 row, the padded-row stream multiply, `Piece` |

## Only the token's experts are streamed

The 6 experts of the token's route (the generator's own, which the device's is checked against) and the shared MLP
are the only weights streamed, and no expert runs a pad row.

## The placement

Every MLP is spread over many chips so that each memory chip streams the same share and every cluster works:

| MLP | |
|---|---|
| shared | a run of its column blocks on every chip |
| a routed expert | its column blocks in `routed_split` runs (pieces), each on another chip: piece i of slot s on chip (s * routed_split + i) mod 16 in the platform's order |

Every chip holds the same number of intermediate columns and streams the same bytes; a memory chip feeds the four chips
of its row, two over each of its links, so every memory chip and every link carries the same share (the root's link one
chunk more: the router's). Every chip streams its shared run while the route is not known yet.

## The clusters

Each piece's columns are split over the chip's two clusters by 16-column blocks, cluster 0 an even number, moved two
at a time until the two clusters hold the same number. Per step:

| step | |
|---|---|
| X | the token's INT8 row in a_row (below), multicast from the HBM of the root's memory chip into every chip's L3 behind one flag; every cluster copies it into L1 |
| route | on the root's cluster 0: one INT8 GEMV of the router's chunk (W_r in the B layout, then its factors' row) into the logits' row, times the factors into row 0 of the softmax's 8 rows (rows 1 to 7 zeroed before the window), the softmax, the top 6 into the route table's entry, which the root multicasts to every chip's L3 behind a flag; every cluster turns it into its chip's row lists (`moe_dispatch`, which fails when a held expert's count is not the placement's) |
| gate\|up | the chunks of a cluster's run of a piece, at most `blocks` column blocks each, as even as whole blocks allow; per chunk one INT4 GEMV of X into the chunk's part of the piece's G row (each block's gate columns, then its up columns), times the chunk's factor row in place; after the piece's last chunk SiLU over the G row, then each block's SiLU'd gate times its up into the piece's A16 row |
| quantise | the A16 row spread on the xDMA into the quantiser's input (atom t at 256 (t / 2) + 16 (t % 2)), then `simd_quant_rows` of that one row times the route weight (the device's row lists; the shared MLP's 1) into the cluster's run of the piece's A8 row in a_row |
| exchange | the cluster's run of the A8 row copied into the other cluster's |
| down | per output slice and piece: one INT4 GEMV of the piece's A8 row into the slice's FP16 row; the shared piece's into the chip's partial, times the factor row in place; a routed piece's into a row buffer, times its factor row, then added into the partial; the shared piece's down runs right after its exchange, ahead of the routed pieces |
| reduce | one ChainGather per output slice of each cluster through every chip's same cluster into the root's fold row (`chain_path`), one at a time but the last slice's two, which run at once once every push has landed |
| y | each root cluster's fold row is its half of y in column order, written to the HBM of the root's memory chip once both root clusters' last chains are done |

Every chunk's factors stream with it as one FP16 row. Every SIMD task is a 32-byte-beat stream kernel; there is no GEMM
(no `gemm_full` may follow a GEMV on a GEMM core), which the build checks.

## Route before the routed pushes

Each cluster posts its ring's runs to its memory chip's push engine. Each ring's first n chunks are those no route
decides (the router's chunk, the shared MLP's gate|up and down); their runs start once X has landed. Chunk n starts a
run of its own (`WeightRings.boundary`), whose post depends on both of the chip's `moe_dispatch` tasks: no routed byte
leaves a memory chip before the device's route has reached that chip and passed its check, while the shared MLP's
stream covers the route's latency. n is the fewer such chunks of the chip's two rings; the HBM image of each ring has a
64-byte gap before chunk n.

## The X layout

A producer delivers X into `md_xland`, then `md_xtflag = 1`: the token's INT8 row in a_row, 2 x hidden bytes, value c
at byte (c // 4) * 8 + c % 4; the GEMV reads the first 4 bytes of each 8-byte word, so the other 4 are free (zero). The
layer's glue makes it from the FP16 row with an xDMA spread of the row's 8-byte atoms 16 bytes apart into an FP16
buffer zeroed once, then `simd_fp16_to_int8` at the static scale s_h.

## Checks

The root's host compares, byte for byte: the route table's entry on the root, the root cluster 1's row lists, the root
cluster 0's A8 rows (a_row) of its shared piece and of its first routed piece (both clusters' columns), and y in HBM.
The golden is the device's arithmetic: each chip's partial over an output slice is the shared MLP's down over the chip's
shared columns, then each of its routed pieces' down over its columns added in piece order; the partials are folded in
the chain's order.

## Parameters

`tokens` 1; `routed_split` (default 8); `slice` (default H / 2: one slice a cluster); `blocks` (column blocks a
gate\|up chunk, at most; default 3); `w_chunk_bytes`; `w_buffers`; `weight_ring`; `chain_path`; `inter_e` / `inter_sh`
(the experts' and the shared MLP's first columns); and the platform's and the graph's switches (`chiplet_mc.py`). The
prefill's own switches are refused.
