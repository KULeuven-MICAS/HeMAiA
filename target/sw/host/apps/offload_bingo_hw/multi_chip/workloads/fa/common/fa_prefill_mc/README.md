# fa_prefill_mc: the FA prefill block (FaBuild)

DeepSeek-V2-Lite's layer-1 MLA attention for a prefill of `tokens` tokens over a fresh prompt (no cache: the keys are
the prompt's own, causal), head parallel over the 16 + 4 chiplet platform. The platform, the stamps and the weight rings'
posts are `util/sim/multichip/chiplet_mc.py`'s; the data and the golden `../fa_prefill_data.py`'s; the sweep over the keys
`libs/blocks/fa_sweep.py`'s. FaBuild has the block's pieces; `../fa_stages_mc` (FaStages) builds the whole block from
them.

| module | what it holds |
|---|---|
| `consts.py` | the shapes, and `check`: the mapping's arithmetic |
| `params.py` | the parameters: the shapes, the head of each chip and the fold order, every switch |
| `memory.py` | the data, every input staged in L3 and HBM, the L1 buffers, where y lands |
| `ring.py` | the weight ring and its loads; `FreshTileSource`, the sweep's key and value tiles from L3 |
| `ops.py` | one task of one cluster: GEMM, SIMD, xDMA, copy |
| `inputs.py` | a cluster's wait for the keys' pieces before its sweep |
| `post.py` | W_UV on the sweep's o~ and W_O into the exchange buffer |
| `reduce.py` | the reduce chains and the stores of y |
| `checks.py` | the checks |

## The mapping

THE SPLIT. Chip k computes head h(k) (16 compute chips, 16 heads, a memory chip's row of four chips holding four
consecutive heads); cluster c of every chip takes half of each chunk's tokens (`row_swap`: the other half in every odd
chunk, so both clusters sweep as many causal tiles), so a cluster owns `chunk` / 2 query rows of ONE head: `chunk` / 32
query tiles of 16 rows (one SIMD beat of FP16 lanes). The 16 heads share the one latent of the keys (MLA, absorbed), so
every chip sweeps all of it and nothing is exchanged inside the attention.

THE KEYS. Each 16-token tile's keys are made once (`../fa_stages_mc`, stages 1 to 3): W_KVA by columns over the chips (each
chip streams its slice of W_KVA's granules and makes them for every tile, putting each slice into the tile's owner's L3),
the owner finishes the tile on its two clusters (dequantisation, RMSNorm of c into the key rows and the value columns,
RoPE of k_pe) and puts the 17 KB piece [key rows | value columns] into the root's L3; the root multicasts a chunk's
pieces at a time into every other chip's L3. A chunk's sweep runs over the key tiles up to the chunk's end, one wait on
the pieces' flags and then a tile at a time from the chip's own L3, the causal masks as the score matmul's INT32 C.

THE CHUNKS. The `tokens` / `chunk` chunks run one after the other. Per chunk and cluster:

    x          the cluster's rows (INT8, A layout) from the chip's L3 into L1
    q          x . W_q[:, head] in three GEMMs of 64 columns (D layout, FP16)
    q_nope     the first 128 columns reshaped D -> A, dequantised and quantised in one SIMD task: the absorb's A operand
    q~         q_nope . W_UK,head a query tile at a time (512 columns), reshaped, dequantised and quantised: Q8's first
               128 A blocks
    q_pe       the last 64 columns dequantised and rotated (one simd_rope over [x | cos | xswap | sin]; the rotary
               columns in half order, so the swap moves two halves), reshaped, quantised: Q8's last 16 blocks
    sweep      fa_sweep: S^T = K Q^T with the causal masks, the online softmax, O^T += V^T P^T per tile of `bc` keys and
               query tile, then o~ = O / l quantised straight into W_UV's A operand (INT8, A layout)
    o          o~ . W_UV,head a query tile at a time, reshaped, dequantised and quantised (post_tiles: a query tile at a
               time behind its own GEMM)
    y_h        o . W_O[head rows, :] in two blocks of 1024 columns, FP16, into the exchange buffer (wo_split_rows: GEMMs
               of two row tiles)
    reduce     per W_O block and cluster index c one ChainGather folding the 16 heads' blocks into the block's collector
               (the root, or y_roots in turn), right behind which the collector stores the block's sum into its L3, over
               x (cluster 0's in chunk i's landing area, cluster 1's in its source)

Chunk i - 1's chains are made right behind chunk i's queries on the root (`../fa_stages_mc` chains_after_sweep: behind its
sweep), so they run under chunk i; chunk i's W_O waits for them (one exchange buffer). The last chunk's two clusters'
chains of a W_O block run at the same time once every chip's ring has landed its last chunk; every other chain runs
after the one before unless chain_concurrent.

THE RING. A chip's two clusters read the SAME stream of chunks: one ring a chip, pushed once by the memory chip's engine,
each run posted by the cluster that first asks for its first chunk. It holds the chip's W_KVA slice, then per token
chunk W_q (3), W_UK, W_UV and W_O (2). Cluster 1 loads each chunk first; cluster 0's load follows it and is the chunk's
last reader, which the post that overwrites the slot follows. A cluster takes the chunks into `w_buffers` (3) L1 slabs;
the sweep's K tiles go into slab 0 (kv_slabs split_kb: the odd ones into a one-tile buffer of their own) and its V tiles
into slabs 1 and 2 alternately.

THE PROGRAM ORDER. The compiler orders every device core's program (mini_compiler passes/bingo_dfg_schedule.py) and
splits the GEMMs, the FlashAttention matmuls, the SIMD and xDMA tasks into configure / start / wait parts where its
schedule gains; it also skips the CSRs a task's predecessor on its core left in place (CSR reuse).

THE MECHANISMS. x and the keys reach L1 behind landed flags (bcast_l1), the partials fold in ChainGathers (chain), every
cross-chip dependency is a BINGO edge (sync edge): the only ones the block builds.

## Parameters (params.hjson)

The platform (num_chiplets, num_clusters), the graph's switches every block shares (static_l1, l1_capacity, l1_order,
compact_id_maps, gid_lists, per_chip_image, desc_in_image, fast_host_init, d2d_ddr, d2d_rx_yield), weight_ring, and:

| switch | what it does |
|---|---|
| tokens, chunk, bc | the prompt, a chunk's tokens, a key tile's keys |
| w_chunk_bytes, w_buffers, wbits | a ring chunk's bytes, the L1 slabs (3), the weights' width (INT8, or INT4 paired) |
| n_calib, kv_cache, weights | the calibration tokens, the hidden states and the checkpoint (paths under the repository) |
| stop | full, or q8: the chunks up to their query tiles |
| kv_slabs | split, or split_kb |
| row_swap | the clusters take the other halves in every odd chunk |
| skip_masked | no key tile whose every score is masked |
| post_tiles | o's relayout and quantisation a query tile at a time |
| wo_split_rows, wo_row_pipe | W_O as GEMMs of two row tiles; each waiting for its own two tiles' o8 only |
| y_roots | the chips that collect and store y, one a W_O block in turn |
| chain_concurrent | the two clusters' chains at the same time (default) |
| checks | y, or stages (also the query tiles and o~ of the chip in the middle of the fold order) |
