# fa_decode_ks_mc: the key-split MLA decode step

DeepSeek-V2-Lite's layer-1 MLA attention as ONE DECODE STEP with the attention SPLIT BY KEY (flash decoding) and every
weight multiply on VersaCore's one-token GEMV ((1, 4, 32), `__snax_bingo_kernel_gemv`, INT4 in the paired layout), on
the 16 + 4 chiplet platform. It is the attention block of the whole-layer decode
(`../../../dsv2/common/dsv2_decode_mc.py`, `AttnKsPart`), which provides the window, x, y's destination and the checks.

The token at position P (params `cache`, default 1,024) attends to keys 0 .. P: the P cached keys and its own. The data
and the scales are `fa_decode_data.FaDecodeData`'s (a fresh causal prompt of P + 1 tokens, the token its last;
rope_perm; the mask as the score matmul's INT32 C); the golden is the arithmetic of the steps below (`KsGolden`,
`golden.py`).

## The modules

| module | what it holds |
|---|---|
| `__init__.py` | `FaDecodeKs`, composed from the mixins below; the sizes re-exported for the layer |
| `consts.py` | the sizes (tiles, rows, chunks, the FA softmax's arena and beats), the constants' layouts, `a_row` |
| `golden.py` | `KsGolden`: the device's arithmetic of the step |
| `params.py` | `__init__`: the parameters and the mapping (heads, tiles, units, W_KVA's chunks a chip) |
| `memory.py` | `mem_handles`: the data, the staging in L3 and HBM, the ring and its loads, the exchange buffers |
| `ops.py` | the task makers the steps share (GEMV, SIMD passes, xDMA reshapes, copies, flagged puts and waits) |
| `build.py` | `build()`: the steps in order; the preloads; the hooks a layer provides (`x_multicast`, `stage_store`) |
| `query.py` | x into every cluster's L1, W_KVA's columns, every head's query row, the query tile |
| `new_key.py` | the new key on the root's cluster 1 and its append (`kv_append`) |
| `sweep.py` | pass 1, m*, pass 2 |
| `output.py` | the partials' chain, o~, W_UV, W_O by columns |
| `report.py` | the D2D traffic and the stream floor (`link_report`) |

## The mapping

Chip i (chips[i], i = 0 .. 15) owns head i, the cached key tiles i, i + 16, .. (64 keys each: one tile a chip at
P = 1,024) and W_O's columns of its two clusters; the root C00 (chips[0]) also owns the new key and its tile. Each
chip's memory chip pushes ONE ring holding all of the chip's chunks in the order they are needed (the push order is the
ring order).

The 16 heads are the 16 FP16 lanes of one query tile, so the attention is one key split over the chips: two passes,
the lanewise maximum m* exchanged between them, the partials folded into the root and normalised there.

## The steps

| step | what runs |
|---|---|
| x | the layer's input norm multicasts the token's INT8 row in a_row (value c at byte (c // 4) * 8 + c % 4, the other bytes zero; 4 KiB) into every chip's L3 (`fk_xland`) behind one flag (`fk_xtflag`); every cluster fetches it into L1 |
| new key | W_KVA's 18 chunks of 32 columns: chunk n on chip n's cluster 1 (n < 16), chunks 16 and 17 (RoPE) on one chip a memory link of the memory chips other than the root's, each the chip nearest to its memory chip; a GEMV each, its FP16 row put into C00's L3 behind its flag. C00's cluster 1 waits for the 18, dequantises the row, RMSNorm of c (simd_rmsnorm, one row), RoPE of k_pe at P, then makes the piece of 16 keys whose row 0 is the token and the rest zero: its key rows (the row in row 0 of a zeroed FP16 A tile [16, 576], quantised in two parts, c at s_c and k_pe at s_kpe) are the first 16 keys of the new tile's K; its value columns (c~ in row 0 of a zeroed [16, 512] tile, transposed, the A layout, quantised) the first four columns of each 16-row block of the new tile's V (one 2-D copy) |
| q | on every chip's cluster 0, head h: W_q's six chunks of the head (the two RoPE chunks first, then the four q_nope chunks), a GEMV each into the head's raw row; q_pe dequantised and rotated at P (rope_perm's halves: x cos + the other half x sin); q_nope spread into a_row, dequantised and quantised at s_qn in one SIMD pass (SimdMulI8 over the spread factors); the absorb q~ = q_nope W_UK,h, one GEMV (K 128, N 512), dequantised. The head's FP16 row [q~ \| q_pe] (1,152 B) is put into C00's L3, row h, behind its flag |
| query | C00's cluster 0 waits for the 16 rows, converts the FP16 rows [16, 576] to the A layout (xDMA) and quantises them in two parts (q~ at s_qt, q_pe at s_qpe): Q8, the query tile of 16 lanes, lane h head h, multicast into every chip's cluster-0 exchange buffer; C00's cluster 1 copies it from its cluster 0 |
| pass 1 | every chip's cluster 0 over its cached tiles (K from the ring) and C00's cluster 1 over the new tile (K made on the device, its keys after P masked by the QK's C): S^T = K Q^T and the FA softmax with its own seed, the shard's lanewise maximum; C00 takes the maximum of its two clusters' (the softmax over the two beats) |
| m* | one ChainGather of the chips' maxima through every chip with the Elementwise junction's FP16 MAX into C00, after every chip's last ring load; m* is multicast into every chip's pass-2 arena (cluster 0 only; C00's cluster 1 copies it) |
| pass 2 | the softmax seeded at m* and PV (V from the ring, or the new tile's), O^T leaving the D port as FP16; each cluster's partial (O^T, then its row sums) at its chip's OFF_LOC; C00 adds its two clusters' |
| reduce | one ChainGather through every chip (chiplet_mc's fold order) with the Elementwise junction's FP16 ADD into C00; C00 normalises (o~ = O c / l), transposes ([16, 512], a head a row), spreads every head's row into a_row, quantises at s_ot and multicasts the 16 heads' rows (16 KiB) into every chip's L3 |
| W_UV | every chip's cluster 0 fetches its head's row and runs o_h = o~_h W_UV,h, one GEMV (K 512, N 128), spread into a_row, dequantised and quantised at s_o in one pass: o8_h, put into C00's L3 at its head's place behind its flag |
| W_O | C00 waits for the 16 and multicasts every head's o8 (one a_row of 4 KiB) into every chip's L3; every cluster runs W_O BY COLUMNS: its 64 columns of y over the full depth of 2,048, two GEMVs of 32 columns, one exact INT32 sum each, and puts its 64 FP16 values into C00's L3 row (`fk_yrow`) behind its flag; C00 waits for the 32 and the layer takes the row (`stage_store`), y before W_O's per-column dequantisation |
| append | C00's cluster 1 writes the new piece into its cache shard in HBM at the new tile's slot: the key rows (a 1-D copy), then the value columns (32 runs of 256 B, one 16-row block apart); the layer makes it at its end (`kv_append`) |

## The split at P = 1,024

1,025 keys: 16 whole cached tiles of 64 and the token's. One cached tile a chip on its cluster 0 and the new tile on
C00's cluster 1: every attention cluster runs one tile, and the pass-1 maxima and the partials of the 16 chips meet
once each (a chain of 16 entries, BINGO_XDMA_CHAIN_MAX). Splitting a tile's PV by latent halves over a chip's two
clusters would repeat its QK and its key stream on both; 32-key tiles would add a chip fold of 16 KiB partials on every
chip; either halves only the array passes of one tile's matmuls, whose dispatch does not shrink with them. More cached
tiles (params `cache` a multiple of 1,024) go to the same clusters, two passes over each shard.

## The cache

Chip i's shard in its memory chip's HBM (`fk_kv_k<chip>`): its tiles in order, each [K tile, the A layout of [64, 576] |
V tile, the A layout of V^T [512, 64]], the layouts the attention reads; after them on C00 the slot of tile P / 64,
which holds the appended piece. Each tile streams through the chip's ring like a weight chunk, K and V one chunk each:
no chip's cache crosses another memory chip's links, and no multicast carries it.

## The orders

The compiler orders every device core's program (mini_compiler passes/bingo_dfg_schedule.py), each cluster's ring
loads in the ring's order; a ring load blocks its DM core until its chunk lands. A chip's ring order: W_KVA's chunk(s)
(cluster 1), W_q x 6 and W_UK (cluster 0), its cached K tiles, then its V tiles, W_UV, cluster 1's two W_O chunks,
cluster 0's two W_O chunks: cluster 0's last W_O chunk is the chip's last push. Each run of the ring is posted by the DM
core of the cluster that reads its first chunk (WeightRings post_chunk, in ring order).

## The collectives

Every two multicasts and every multicast / chain pair are ordered by the graph's dependency and flag edges (the layer's
check_collectives proves it): x, the query, m*, o~ and the o8 row are each made on C00 after every chip's part of the
step before has landed there. Both chains (m*'s and the partials') run after every chip's last ring load, so after every
push of every memory chip has landed: no ChainGather runs beside a memory-chip push. Every write into HBM (the append)
follows every collective and every chip's last ring load. The gathers (the W_KVA rows, the query rows, the o8 rows, the
y columns) are unicast iDMA puts behind flags.

## In a layer

The layer subclass provides `setup` and `end_window` (the window), `x_multicast(go)` (the task the x fetches are
ordered after: the layer's input norm puts x into `fk_xland` on every chip behind `fk_xtflag`, the a_row above, INT8 at
the block's s_x, 4,096 B), `stage_store(yw)` (y from the root's L3 row `fk_yrow` after `yw`) and the checks; it may make
the append at its own end (`kv_append(deps)`). What a layer reads: `d` (FaDecodeData), `gold` (y_raw and the stages'
values), `_chains`, `_y_stores`, `append`, the sizes (`XB`, `P`, ...).

## Parameters

The block's section of the layer's params.hjson (`attn`):

| key | what it sets |
|---|---|
| `cache` | P, the cached positions (a multiple of 64 x 16); default 1,024 |
| `n_calib` | the dequantisation's calibration positions; default 32 |
| `checks` | `y` (default): the layer's checks of the attention (m*, o~, a partial, the appended piece, the query tile with `stop attn`); `stages`: also a chip's query row, the root's o8 row and the new tile's K, for locating a failing step |
| `kv_cache`, `weights` | the cache of hidden states and the checkpoint's weights, paths under the repository |
| `w_chunk_bytes` | the ring's slot (one K tile, 36,864) |
| `weight_ring` | `slots`, `batch`, `head`, `batch_head`, `queue` (chiplet_mc's) |

and the platform's and the graph's switches of chiplet_mc (static_l1, l1_capacity, l1_order, compact_id_maps,
gid_lists, per_chip_image, desc_in_image, fast_host_init, d2d_ddr, d2d_rx_yield). The step runs one token (`tokens`
1), INT4 weights (`wbits` 4), and refuses `bcast`, `reduce` or `sync` other than bcast_l1, chain and edge.
