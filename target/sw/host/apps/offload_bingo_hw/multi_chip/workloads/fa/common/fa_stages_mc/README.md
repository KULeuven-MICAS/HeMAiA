# fa_stages_mc: the whole FA prefill block (FaStages)

The DSv2-Lite MLA fresh prefill (`../fa_prefill_mc`, FaBuild) built whole, stages 0 to 8 in one window:
fa/dsv2_l1_prefill/full_n and t1024, fa/dsv2_l1_prefill/stage01234 (params stop q8: up to every cluster's query tiles),
and the attention of the DSv2 layer prefill (dsv2/common/dsv2_layer_prefill_mc.py). Each stage alone is its own app,
fa/dsv2_l1_prefill/stage<N>/main_bingo.py, built on `../../dsv2_l1_prefill/common/stage_common.py`.

| module | what it holds |
|---|---|
| `params.py` | the block builder's switches |
| `inputs.py` | the keys' flags beside FaBuild's image, and the copies before the window |
| `ring.py` | the order of the root memory chip's engine queue (x_pf_split) |
| `x.py` | stage 0: x from HBM into every chip's L3; a cluster's rows of a chunk |
| `keys.py` | stage 1: the keys by columns |
| `owner.py` | stages 2 and 3: the owners finish the keys, the root multicasts them |
| `queries.py` | stage 4: a chunk's query tiles; stage 5: its sweep |
| `build.py` | build(): the whole block |
| `order.py` | keys_by_chunk: the order of the collectives and the key tasks, and its checks |
| `checks.py` | the checks' waits for the owners' puts; finish() |

## The stages

    stage 0  x from HBM: the prompt's rows sit in the HBM of the root's memory chip; that memory chip's push engine
             multicasts them a tile of 16 tokens (32 KiB) at a time into every chip's L3, each tile behind a landed flag
             of its own, queued by the root's host (ChipletBuild.hbm_cast: its engine posts and a wait on the landed
             flags); the root gets its copy from the multicast like every other chip. The root's ring shares that
             engine: its posts queue behind the multicast's, or with x_pf_split its W_KVA runs ahead of it.
    stage 1  the keys by columns: each cluster copies its chip's W_KVA columns into L1 first, then each of its tiles of
             x (cluster c: the tiles t with t mod 2 = c) from the chip's L3 into L1 behind the tile's flag, 70 % by the
             iDMA and 30 % by the xDMA (two different wide ports into L3); a GEMM a tile; each slice into the tile
             owner's L3 behind a flag; every owner waits for its tile's 16 slices. kv_balance (INT4): the two chips with
             two pairs of W_KVA's columns hand their second pair to helper chips for all but the first of
             kv_balance_ways runs of tiles (kv_helpers).
    stage 2  the owner finishes a key tile behind its stage 1 waits: c on the tile's cluster 0, its dequantisation and
             RMSNorm row_major -> A (INT8) into the key rows; cluster 1 builds c transposed itself and RMSNorm
             col_major -> B (INT8) writes the value columns, and dequantises and rotates k_pe; each cluster puts its
             part into the root's L3 behind a flag of its own.
    stage 3  the keys to every chip: the root multicasts a chunk's pieces at a time once both their flags are in;
             every other chip waits for every chunk's keys (full_kv_all).
    stage 4  per chunk and cluster its rows of x into L1, then q = x W_q, q~ = q_nope W_UK (a query tile at a time) and
             RoPE on q_pe into the query tiles Q8 (stop q8: the window ends here).
    stage 5  per chunk and cluster the sweep (QK with the causal masks, the online softmax, PV, the normalisation
             quantising o~), each softmax arena's geometry built on the xDMA core.
    6 - 8    W_UV, W_O and the reduce chains into the root (y_roots: a collector chip a W_O block), each block's sum
             stored into the collector's L3.

The checks are FaBuild's: the keys' pieces, y's windows in each collector's L3 (stop q8: the query tiles of the chip in
the middle of the fold order).

## Parameters (params.hjson)

FaBuild's (`../fa_prefill_mc/README.md`), and:

| switch | what it does |
|---|---|
| x_pf_split | the root's W_KVA ring runs ahead of x's multicast in its memory chip's engine queue, the rest after it |
| kv_balance, kv_balance_ways | helper chips for the two-pair chips' second pair (stage 1) |
| full_kv_all | every chip waits for every chunk's keys at the end of stage 3 (default); false: each sweep waits for its own |
| keys_by_chunk | each chunk's keys just ahead of its queries; the later chunks' keys after the window |
| x_rest_early | the rest of x right after chunk 0's key multicast (keys_by_chunk) |
| kx_mid_double | the later chunks' key tiles inside the window double-buffered (keys_by_chunk) |
| kc_xs_late | a helper's tiles after the window take their own copy of the helped pair (keys_by_chunk) |
| kx_per_chunk | every chunk's key tiles in buffers and copies of their own, behind the chunk before's queries (keys_by_chunk) |
| kx_after_chains | every key transfer ordered with every reduce chain (kx_per_chunk, y_roots, yr_store_tile) |
| chains_after_sweep | the root clusters' chains of chunk i - 1 behind their sweep of chunk i |
| yr_store_tile | a collector's store of chunk i's y behind its load of chunk i + 1's key tile t's values (y_roots) |
| yr_rel1 | on the collectors of chunk i - 1, chunk i's ring chunks are released by cluster 1's read (y_roots) |
| s4_q8_pairs | the query tiles quantised tile by tile, q~ then q_pe |
| relay_prefer | {consumer role: [roles]}: the cores dep-tag relays go to first |

The builder takes no `stage`: a params file that names one is refused.
