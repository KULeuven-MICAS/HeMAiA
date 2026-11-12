# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""The shared driver of the fa PREFILL workloads (workloads/fa/dsv2_l1_prefill*/<variant>/):
DeepSeek-V2-Lite's layer-1 MLA attention for a prefill of `tokens` tokens over the REAL 8K-token
latent cache, head parallel over the 16 + 4 chiplet platform. The platform, the stamps, the
barriers and the synchronisation arm are chiplet_mc.py's; the data and the golden are
fa_prefill_data.py's; the sweep over the cache is libs/blocks/fa_sweep.py's.

A FRESH PROMPT (params fresh, alone or as a block of a whole layer): no cache; the
keys are the prompt's own and causal. Each chip makes the latent of the 16-token tiles its input norm gave
it (stage_kv, cluster 0; tile q on chip q mod 16, T / 256 tiles a chip): kv = x8 . W_KVA in nine GEMMs of 64
columns a tile (W_KVA through the ring once, ahead of the chunks' weights), dequantised; c~ = RMSNorm of its
first 512 columns, into the key rows (A layout, quantised) and, transposed, the value columns; k_pe = RoPE of
the last 64, into the key rows. The piece [key rows 9 KiB | value columns 8 KiB] goes into the root's L3
behind a landed flag; the root multicasts a chunk's pieces at a time into every other chip's L3
(stage_kv_bcast): one source, as x, because 16 chips multicasting at once lock the D2D routers' outputs in a
cycle. A chunk's sweep runs over the key tiles up to the chunk's end (FreshTileSource: one wait on the landed
flags, then a tile at a time from the chip's own L3), the causal masks added to the scores (fa_sweep's mask).

THE SPLIT. Chip k computes head h(k) (16 compute chips, 16 heads, a memory chip's row of four chips
holding four consecutive heads); cluster c of every chip takes half of each chunk's tokens, so a
cluster owns `chunk` / 2 query rows of ONE head: `chunk` / 32 query tiles of 16 rows (BR, one SIMD
beat of FP16 lanes). The 16 heads share the one latent cache (MLA, absorbed), so every chip sweeps all
of it and nothing is exchanged inside the attention.

THE CHUNKS. The `tokens` / `chunk` chunks run one after the other and each sweeps the whole cache
again: a chunked prefill, where a serving system runs the other layers between two chunks. Per chunk
and cluster:

    x          the chunk's rows (INT8, A layout) from the root's L3 into every chip's L3 and the
               cluster's rows into its L1
    q          x . W_q[:, head] in three GEMMs of 64 columns (D layout, FP16), dequantised
    q_nope     the first 128 columns reshaped D -> A (FP16) and quantised: the absorb's A operand
    q~         q_nope . W_UK,head (one GEMM, 512 columns), dequantised, reshaped, quantised: Q8's
               first 128 A blocks, a query tile at a time
    q_pe       the last 64 columns rotated (RoPE: x cos + swap(x) sin_signed, three elementwise
               stages and a pair swap on the iDMA), reshaped, quantised: Q8's last 16 blocks
    sweep      fa_sweep over the cache: S^T = K Q^T, the online softmax, O^T += V^T P^T per tile of
               `bc` keys and query tile, then o~ = O / l transposed to one query a row
    o          o~ reshaped row -> A and quantised, . W_UV,head, dequantised, reshaped, quantised
    y_h        o . W_O[head rows, :] in two GEMMs of 1024 columns, FP16, into the exchange buffer
    reduce     the 16 heads' y_h of cluster index c added into the root's cluster c (chain: one
               ChainGather in the fold order; root: the root pulls and adds), stored in the root's L3

THE RING. A chip's two clusters read the SAME stream of chunks: one ring a chip, pushed once by the
memory chip's engine under the chip's prefetcher. Per token chunk it is W_q (3), W_UK, then the cache
tile by tile K0 V0 K1 V1 ..., then W_UV and W_O (2). Cluster 1 loads each chunk without releasing its
slot; cluster 0's load releases it and follows cluster 1's. The ring's chunks are not all one size; a
cluster takes them into `w_buffers` L1 slabs in ring order, in turn, or (kv_slabs split, three slabs) the
K tiles always into one slab and the V tiles into the other two (fa_sweep's tile source).

THE PROGRAM ORDER. The compiler orders every device core's program (mini_compiler passes/bingo_dfg_schedule.py)
and splits the GEMMs, the FlashAttention matmuls, the SIMD and xDMA tasks into configure / start / wait parts where its
schedule gains; it also skips the CSRs a task's predecessor on its core left in place (CSR reuse).

THE MECHANISMS (params): bcast_l1 | pull (x), chain | root (the partials), edge | barrier (every
cross-chip dependency: a BINGO edge, or the chip barrier before and after each chunk's reduce, a
full fence). The ladder base -> bcast -> bcast_reduce -> all adds one each.

PARAMETERS (params.hjson): the platform (num_chiplets, num_clusters), tokens, chunk, keys, bc,
w_chunk_bytes and w_buffers (the ring chunk and the L1 slabs), weight_ring,
bcast, reduce, sync, barrier_mode, barrier_fence, async_exec, static_l1, l1_capacity, d2d_ddr, and
n_calib (calibration tokens), kv_cache and weights (paths under the repository), root_slice (the
root's pull slice), compact_id_maps, gid_lists, relay_per_gather (each dependency-tag relay of
an xDMA task serves that one task; default true). The orders and buffers (each explained where it is read):
prefetch_after_x, l3_overlay, kv_slabs, y_halves, y_block_chains, chain_concurrent,
wo_split_rows, o_land; fresh (a fresh prompt: keys = tokens, a multiple of 256, kv_slabs split), kv_clusters
(fresh: a chip's n-th tile makes its keys on cluster n mod 2 instead of cluster 0), kv_cols (fresh: W_KVA by columns
over the chips, the rows traded, stage_kv_cols).
rope_fused (default false): q_pe's RoPE in one SIMD task (simd_rope) instead of three. rope_perm (default false,
with rope_fused and fresh): the rotary columns in half order, so RoPE's swap moves two halves instead of FP16 pairs.
dq_fused (default false): q_nope's, q~'s and o's dequantisations folded into the quantises after them.
mask_c (default false, fresh): the causal mask as the score matmul's INT32 C instead of a SIMD add.
x_resident (default false): every chip's clusters copy their x rows from the chip's own copy of x, with no multicast
from the root (the layer before leaves x on every chip). x_hbm (default false; the stage workloads, fa_stages_mc.py): x
in the HBM of the root's memory chip, every chip's L3 copy filled by a multicast inside the window. wbits (8 | 4,
default 8): the weights INT8 or paired INT4.
q_tiles (default false): a query tile's sweep follows its
own Q8 quantises only. o_tiles (default false): W_UV a query tile at a time, each behind that tile's relayout and
quantise.
The checks: checks (y | stages), y_windows (3 | 4) (fresh adds the keys' pieces in the root's L3). For
bisecting a failure: stop (full | q8 | sweep | post | x), x_root, x_put_cluster, x_ring.
"""

import dataclasses
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.abspath(__file__).split("/target/sw/host/")[0], "util", "sim", "multichip"))
sys.path.insert(0, os.path.join(os.path.abspath(__file__).split("/target/sw/host/")[0], "util", "sim", "common"))
from driver import repo_root, run_driver  # noqa: E402
REPO = repo_root(__file__)
sys.path.insert(0, f"{REPO}/target/sw/host/runtime/libbingo/mini_compiler")
sys.path.insert(0, HERE)

import _bingo_paths  # noqa: F401,E402
from bingo_kernel_args import (  # noqa: E402
    INT4_B_LAYOUT_PAIRED, SnaxBingoKernelGemmFullArgs, SnaxBingoKernelIdma1dCopyArgs, SnaxBingoKernelIdma2dCopyArgs, SnaxBingoKernelIdmaFetchFlaggedArgs,
    SnaxBingoKernelIdmaPairwiseSwapArgs, SnaxBingoKernelIdmaPutFlaggedArgs, SnaxBingoKernelIdmaRingLoadArgs,
    SnaxBingoKernelSimdAddF16Args, SnaxBingoKernelSimdFp16ToInt8Args, SnaxBingoKernelSimdMulF16Args,
    SnaxBingoKernelSimdMulI8Args,
    SnaxBingoKernelSimdRmsnormArgs, SnaxBingoKernelSimdRopeArgs, SnaxBingoKernelWaitFlagsArgs, SnaxBingoKernelXdmaChainGatherArgs,
    SnaxBingoKernelXdmaTranspose2dArgs, simd_beats, simd_fp16_lanes)
from bingo_mem_handle import BingoMemSymbol  # noqa: E402
from libs.blocks import FaSweepCfg, WeightRings, fa_sweep, fa_sweep_alloc  # noqa: E402
from libs.comm.nest import convert_args, d_to_a_args  # noqa: E402

import chiplet_mc as cm  # noqa: E402
import fa_prefill_data  # noqa: E402

BLK = 16                      # a GEMM column block (meshCol)
BR = 16                       # query rows a tile: the SIMD's FP16 lanes
D_QK, D_V, KV_RANK, ROPE, Q_NOPE, Q_HEAD, HIDDEN = 576, 512, 512, 64, 128, 192, 2048
TILE16 = 16 * 16 * 2          # one D-layout FP16 block
EW_F16_ADD = cm.EW_F16_ADD
STRIPE = [0, 1, 3, 4]         # the compute columns of a row
KB, VB = 16 * D_QK, D_V * 16  # a fresh prompt's 16-token piece: its key rows, its value columns (INT8)
KVP = KB + VB


def check(p):
    """The mapping's arithmetic; raises ValueError where the parameters cannot be laid out."""
    tokens, chunk, bc = (int(p[k]) for k in ("tokens", "chunk", "bc"))
    fresh = bool(p.get("fresh", False))
    keys = tokens if fresh else int(p["keys"])
    rows = chunk // 2
    if tokens % chunk or chunk % 2 or rows % BR:
        raise ValueError(f"tokens {tokens} / chunk {chunk}: a cluster takes chunk / 2 rows, a whole number of "
                         f"{BR}-row query tiles")
    if keys % bc or bc % 16 or keys // bc < 4:
        raise ValueError(f"keys {keys}: not whole tiles of bc {bc} (a multiple of 16), at least four")
    if fresh and (chunk % bc or int(p.get("keys", keys)) != keys):
        raise ValueError(f"fresh: a chunk of {chunk} is whole key tiles of bc {bc} (a chunk's two row blocks sweep "
                         f"every tile up to its end), and the keys are the {tokens} tokens")
    wq_block = HIDDEN * 64 * int(p.get("wbits", 8)) // 8
    if int(p["w_chunk_bytes"]) < max(bc * D_QK, D_V * bc, wq_block):
        raise ValueError(f"w_chunk_bytes {p['w_chunk_bytes']}: a K tile is {bc * D_QK} B, a W_q block "
                         f"{wq_block} B")
    return dict(chunks=tokens // chunk, query_tiles=rows // BR, tiles=keys // bc, keys=keys,
                qk_passes=(bc // 16) * (D_QK // 4), pv_passes=(D_V // 16) * (bc // 4))


class FreshTileSource:
    """fa_sweep's tiles for a fresh prompt: key and value tile j of chunk i from the chip's own L3, where stage_kv's
    multicasts leave the 16-token pieces [key rows | value columns] side by side, behind one wait a sweep on the
    flags of the pieces it reads. A key tile is its pieces' key rows one after the other (a 2D copy); a value tile is
    the A layout of [512, bc], each piece's 256-byte runs one 16-row block apart (a 3D copy). K tiles in slab 0 (split)
    or slabs 0 and 3 alternately (split4), or slab 0 and a one-tile buffer alternately (split_kb), V tiles in slabs 1
    and 2 alternately."""

    def __init__(self, fb, cl, i, nt):
        self.fb, self.cl, self.i, self.nt = fb, cl, i, nt
        self.wait = None

    def fetch(self, kind, j):
        fb = self.fb
        k, c = fb.chip(self.cl), fb.local(self.cl)
        if self.wait is None:
            self.wait = fb.kv_wait(self.cl, self.i)
        n = fb.bc // 16
        src = BingoMemSymbol(f"{fb.TAG}_kv", offset=j * n * KVP + (KB if kind == "V" else 0))
        if kind == "K" and fb.kv_slabs == "split_kb" and j % 2:
            # split_kb: the odd key tiles in a buffer of one key tile of their own, free once the last QK of the
            # odd tile before has read it
            si, slab = ("kb", self.cl), fb.kbuf(self.cl)
            war = list(fb._kb_reader.get(self.cl, []))
            args = SnaxBingoKernelIdma2dCopyArgs(src, slab, KB, KVP, KB, n)
        elif kind == "K":
            si, slab, war = fb.ls[self.cl].take_at(3 if fb.kv_slabs == "split4" and j % 2 else 0, fb.bc * D_QK)
            args = SnaxBingoKernelIdma2dCopyArgs(src, slab, KB, KVP, KB, n)
        else:
            si, slab, war = fb.ls[self.cl].take_at(1 + j % 2, D_V * fb.bc)
            args = SnaxBingoKernelIdma2dCopyArgs(src, slab, 256, 256, 16 * fb.bc, D_V // 16, outer=n, src_outer=KVP,
                                                 dst_outer=256)
        ld = fb.node(k, c, fb.roles["dm"], f"Ld_{kind}{j}_i{self.i}", "__snax_bingo_kernel_idma_2d_copy", args,
                     war + [self.wait])
        return ld, slab, si

    def release(self, token, task):
        if isinstance(token, tuple) and token[0] == "kb":
            self.fb._kb_reader[token[1]] = list(task) if isinstance(task, (list, tuple)) else [task]
        else:
            self.fb.ls[self.cl].release(token, task)


class TileSource:
    """fa_sweep's tile stream of one cluster: K and V tile j of token chunk i are ring chunks
    base + 2 j and base + 2 j + 1, through the cluster's slabs in turn."""

    def __init__(self, fb, cl, i):
        self.fb, self.cl, self.i = fb, cl, i

    def fetch(self, kind, j):
        fb = self.fb
        n = fb.chunk_index(self.i, "kv") + 2 * j + (kind == "V")
        # kv_slabs split: every K tile in slab 0, V tiles in slabs 1 and 2 in turn, so K_{j+1} loads under
        # PV(j-1) and V_{j+1} under QK(j+1) (in turn, K_{j+2} would wait for PV(j), right before QK(j+2))
        slab = None
        if fb.kv_slabs in ("split", "split4"):
            slab = (3 if fb.kv_slabs == "split4" and j % 2 else 0) if kind == "K" else 1 + j % 2
        ld, si, slab = fb.load(self.cl, n, f"Ld_{kind}{j}_i{self.i}", slab=slab)
        return ld, slab, si

    def release(self, token, task):
        self.fb.ls[self.cl].release(token, task)


class FaBuild(cm.ChipletBuild):
    """The MLA prefill block (the module's docstring). run_driver makes the program in four steps: __init__ (the
    parameters), mem_handles() (the data and the memory, no task), build() (the task graph: every task and edge) and
    finish() (the compile into the C header and the data image, in which the compiler orders every core's tasks)."""
    TAG = "fp"

    def kbuf(self, cl):
        """kv_slabs split_kb: cluster cl's buffer for its odd key tiles (one key tile)."""
        if cl not in self._kbuf:
            self._kbuf[cl] = self.cx(cl).l1(f"{self.TAG}_kbuf", self.bc * D_QK)
        return self._kbuf[cl]

    def __init__(self, args, p, plat, hw, shared=None):
        """The parameters; no data, memory or task yet. ChipletBuild.__init__ first: the platform's and the graph's
        switches, the chips, each one's memory chip and its push engine there, the empty DFG, the data staging (self.st)
        and the stamp buffer (a layer's block shares the layer's). Then the block's: the chunks, their query and key
        tiles (check), every switch with its checks, the head of each chip and the fold order, the data's inputs and
        the sizes of a chunk's x and of a cluster's partial y."""
        super().__init__(args, p, plat, hw, shared=shared)
        # x_device (a block of a whole layer): the root's x source is made on the device,
        # chunk i's tasks in x_ready[i]: staged as zeros, and each chunk's multicast waits for them
        self.x_device = bool(p.get("x_device", False))
        self.x_ready = {}
        m = check(p)
        self.NCH, self.NQ, self.NT = m["chunks"], m["query_tiles"], m["tiles"]
        self.C = int(p["chunk"])
        self.ROWS = self.C // self.NC                  # query rows a cluster
        # row_swap: in every odd chunk the clusters take the other halves (causal: the late half sweeps more tiles)
        self.row_swap = bool(p.get("row_swap", False))
        # ot8_fused: the sweep's normalisation quantises o~ into W_UV's A operand (simd_mla_normalise_q8), so no
        # transpose, no row-major -> A conversion and no separate quantisation sit between the sweep and W_UV
        self.ot8_fused = bool(p.get("ot8_fused", False))
        # post_tiles (o_tiles, dq_fused): o_h's relayout and quantisation a query tile at a time, each behind its own
        # W_UV GEMM, so they run under the next tiles' GEMMs and W_O starts behind the last tile's
        self.post_tiles = bool(p.get("post_tiles", False))
        # sm_batch: a key tile's softmaxes of the cluster's query tiles as one SIMD task (simd_fa_softmax_n)
        self.sm_batch = bool(p.get("sm_batch", False))
        # sm_prologue: each sweep's softmax arenas get their geometry from PROLOGUE nodes at the sweep's head
        self.sm_prologue = bool(p.get("sm_prologue", False))
        # sm_geom_xdma: each softmax arena's geometry built on the cluster's xDMA core at the sweep's head (fa_build_shapes)
        # instead of on the SIMD core; the arena's first softmax then runs GEOM_PRIMED (replaces sm_prologue)
        self.sm_geom = "xdma" if bool(p.get("sm_geom_xdma", False)) else None
        if self.post_tiles and not (bool(p.get("o_tiles", False)) and bool(p.get("dq_fused", False))):
            raise ValueError("params post_tiles needs o_tiles and dq_fused")
        if self.ot8_fused and not bool(p.get("o_tiles", False)):
            raise ValueError("params ot8_fused needs o_tiles (W_UV a query tile at a time)")
        self.keys, self.bc = m["keys"], int(p["bc"])
        # fresh: a fresh prompt (the module's docstring); chip chips[q % 16] makes the keys of tokens 16 q .. 16 q + 15,
        # the 16-row tiles its input norm gives it, T / 256 a chip 16 tiles apart (a layer hands each tile in:
        # kv_src {tile: (x8 handle, [ready tasks])}; the block alone loads them from the staged x, fresh_tiles)
        self.fresh = bool(p.get("fresh", False))
        self.kv_src, self.kv_put = {}, {}
        # params wo_cols: W_O by output columns instead of by heads: every head's o8 into the root's L3 (the chunk's x
        # landing area holds them, dead by then), one multicast of them from the root, and each cluster's two GEMMs make
        # its chip's 128 columns of y for its rows over the full depth, stored straight into the root's y: no reduce
        self.wo_cols = bool(p.get("wo_cols", False))
        # params o_land (wo_cols): each chunk's o8 lands in an area of its own in the root's L3 (fp_oland, two chunks'
        # worth, by the chunk's parity) instead of the chunk's x landing area, so no put of it waits for the root's
        # fetches of x: a chip's DM core then holds no edge from the root's every chunk at once
        self.o_land = bool(p.get("o_land", False))
        # params wo_prefetch (wo_cols): each cluster loads its chunk's two W_O blocks right after its o8 put, so its
        # in-order DM core streams them during the o8's round trip through the root instead of after its gather
        self.wo_prefetch = bool(p.get("wo_prefetch", False))
        if self.wo_prefetch and not self.wo_cols:
            raise ValueError("params wo_prefetch: needs wo_cols")
        if self.o_land and not self.wo_cols:
            raise ValueError("params o_land: needs wo_cols")
        if self.fresh:
            nch = len(self.chips)
            if self.T % (16 * nch):
                raise ValueError(f"fresh: {self.T} tokens; each of the {nch} chips makes whole 16-token tiles")
            # tile q on chip chips[q % 16]: the first chunks' keys come from every chip's first tile
            self.kv_tiles = {k: [q for q in range(self.T // 16) if self.chips[q % nch] == k] for k in self.chips}
        # params kv_clusters (fresh): a chip's n-th tile makes its keys on cluster n mod NC (kv_cluster), so with more
        # than one tile a chip both clusters make keys; the layer puts the tile's input norm there too
        self.kv_clusters = bool(p.get("kv_clusters", False))
        if self.kv_clusters and not self.fresh:
            raise ValueError("params kv_clusters: needs fresh")
        # params owner_bcast (fresh, a layer's x): each tile's owner multicasts its x8 rows and its keys' piece to every
        # chip itself, each with a landed flag of its own, instead of putting them into the root's L3 for the root to
        # multicast a chunk at a time: the root's DM core then runs no chip's collectives before its own chunk 0, and
        # a cluster's x and key waits count the tiles' flags. It HANGS the D2D fabric: sixteen chips multicasting at once
        # deadlock (prefill_tiny/layer12a3, 25 Oct: every chip's SoC crossbar stalled before the first tile landed)
        self.owner_bcast = bool(p.get("owner_bcast", False))
        if self.owner_bcast and not (self.fresh and self.x_device and self.bcast_impl == "bcast_l1"):
            raise ValueError("params owner_bcast: needs fresh, a layer's x (x_device) and bcast bcast_l1")
        # params wbits (8 | 4, default 8): the projection weights INT8 or INT4 in the paired layout (half the bytes
        # pushed, half the L1 slab); the base class's default of 4 is the FFN's and the MoE's
        self.wbits = int(p.get("wbits", 8))
        if self.wbits not in (8, 4):
            raise ValueError(f"params wbits={self.wbits}: 8 or 4")
        # params kv_cols (fresh, the block alone): W_KVA by columns instead of by tokens. Chip j streams only its
        # 16-column granules [36 j / 16, 36 (j + 1) / 16) of W_KVA's 36 (two or three: one ring chunk) and makes them
        # for every token, its cluster c the tiles q = c mod 2, a GEMM a tile; each tile's slice goes into the L3 of
        # the tile's owner behind a flag of its own, and the owner runs the keys' glue on the gathered rows as before
        self.kv_cols = bool(p.get("kv_cols", False))
        if self.kv_cols and not (self.fresh and not self.x_device and not self.owner_bcast):
            raise ValueError("params kv_cols: needs fresh, the block's staged x (not x_device), not owner_bcast")
        if self.kv_cols:
            # INT4 streams whole pairs of granules (the paired layout's blocks of 32 columns)
            pg = 2 if self.wbits == 4 else 1
            ng, nch = D_QK // 16 // pg, len(self.chips)
            self.kvg = {k: (pg * (j * ng // nch), pg * ((j + 1) * ng // nch)) for j, k in enumerate(self.chips)}
            self.kv_owner = {q: k for k in self.chips for q in self.kv_tiles[k]}
            self.KVD = 16 * D_QK * 2                     # a tile's kv, D layout [16, 576] FP16
            self.KVR_TILES = max(len(t) for t in self.kv_tiles.values())
        if simd_fp16_lanes() != BR:
            raise ValueError(f"the SIMD has {simd_fp16_lanes()} FP16 lanes; the softmax takes one query "
                             f"a lane and the tiles are {BR} rows")
        # params rope_fused: q_pe's RoPE as one simd_rope task over [x | cos | xswap | sin] instead of two multiplies
        # and an add; the same roundings (RNE(RNE(x cos) + RNE(swap(x) sin))), so the same bits
        self.rope_fused = bool(p.get("rope_fused", False))
        # params rope_perm (with rope_fused, fresh): the rotary columns ordered as the pairs' first elements, then their
        # second (fa_prefill_data.py), so the swap is the two 32-column halves exchanged: whole 1 KB runs of each row
        # tile's D layout on the iDMA, instead of the pairwise swap's 2-byte rows
        self.rope_perm = bool(p.get("rope_perm", False))
        # params dq_fused: each dequantisation that a quantise follows (q_nope, q~, o) folded into that quantise: the
        # D -> A reshape moves the raw FP16 values, then one SIMD task multiplies them by an A-layout copy of the
        # per-column scales and quantises (SimdMulI8, the same roundings as the two tasks)
        self.dq_fused = bool(p.get("dq_fused", False))
        if self.rope_perm and not (self.rope_fused and self.fresh):
            raise ValueError("params rope_perm: needs rope_fused and fresh")
        # params kv_rope_fused (fresh, rope_perm): the keys' k_pe RoPE as one simd_rope, as q_pe's; kv_rope_first: the
        # owner's SIMD core takes the keys' RoPE right after their dequantisation (stage_kv_on)
        self.kv_rope_fused = bool(p.get("kv_rope_fused", False))
        self.kv_rope_first = bool(p.get("kv_rope_first", False))
        if self.kv_rope_fused and not self.rope_perm:
            raise ValueError("params kv_rope_fused: needs rope_perm")
        # params mask_c (fresh): each masked score tile's mask is its QK matmul's C, an INT32 bias added before the D
        # port's conversion (fa_sweep's mask_c), instead of a SIMD add between the matmul and the softmax
        self.mask_c = bool(p.get("mask_c", False))
        if self.mask_c and not self.fresh:
            raise ValueError("params mask_c: needs fresh")
        self.stop = str(p.get("stop", "full"))
        if self.stop not in ("full", "q8", "sweep", "post", "x"):
            raise ValueError(f"params stop={self.stop!r}: full, q8, sweep, post or x")
        # x_root: how the root's own clusters read their x rows: flagged (behind the landed flag, as every
        # chip does), copy (a plain copy from the root's source), poll2d (the flagged fetch's two DMAs as two
        # tasks), flagged_zero (the polled word zeroed first). x_put_cluster: the root cluster whose DM core
        # multicasts x. x_ring (stop x): the chunks' W_q and W_UK still stream into L1, no GEMM reading them.
        self.x_root = str(p.get("x_root", "flagged"))
        if self.x_root not in ("flagged", "copy", "poll2d", "flagged_zero"):
            raise ValueError(f"params x_root={self.x_root!r}: flagged, copy, poll2d or flagged_zero")
        self.x_put_cluster = int(p.get("x_put_cluster", 0))
        # x_resident: every chip holds the prompt's x in its L3 (the block's staged x, which fresh_tiles already reads
        # there): each cluster copies its rows of a chunk from its own chip's copy, and the root multicasts no x. A
        # mapping of the layer around the block: the block before leaves x on every chip (an all-gather) instead of
        # in the root
        self.x_resident = bool(p.get("x_resident", False))
        if self.x_resident and (self.x_device or self.owner_bcast):
            raise ValueError("params x_resident: x is staged on every chip (not x_device, not owner_bcast)")
        # x_hbm (the staged builder, fa_stages_mc.py): the prompt's rows sit in the HBM of the root's memory chip, and
        # every chip's copy of them in L3 (fp_xsrc) starts as zeros for a multicast inside the window to fill, a tile of
        # 16 tokens at a time behind its own flag (fp_xtflag)
        self.x_hbm = bool(p.get("x_hbm", False))
        if self.x_hbm and (self.x_resident or self.x_device or self.owner_bcast or not (self.fresh and self.kv_cols)):
            raise ValueError("params x_hbm: needs fresh and kv_cols, not x_resident, x_device or owner_bcast")
        self.x_ring = bool(p.get("x_ring", True))
        # q_tiles: a query tile's first score matmul follows that tile's two Q8 quantises only, not every projection
        # task of the cluster; o_tiles: o~'s relayout, its quantise and W_UV a query tile at a time, each tile's W_UV
        # behind its own transpose's chain (one load of W_UV for both)
        self.q_tiles = bool(p.get("q_tiles", False))
        self.o_tiles = bool(p.get("o_tiles", False))
        # l3_overlay: the root's y lives in its x buffers (y of chunk i, cluster 0 in the landing buffer of
        # chunk i, cluster 1 in the source of chunk i): every chip's image is the same, and an 8 MiB L3 cannot
        # hold y beside both x buffers, the ring and the task tables. A chunk's y is stored after its reduce,
        # which follows every use of that chunk's x.
        self.l3_overlay = bool(p.get("l3_overlay", False))
        # y_halves: a cluster's partial y goes through an exchange buffer of half its size, one W_O block
        # at a time: W_O block 0, its ChainGather, W_O block 1 behind that gather, its ChainGather
        self.y_halves = bool(p.get("y_halves", False))
        if self.y_halves and (self.reduce_impl != "chain" or self.sync == "barrier"):
            raise ValueError("y_halves needs reduce: chain and sync: edge")
        # y_store_halves (with y_halves): each half's gather lands in a buffer of its own, stored right after it,
        # and the second half's gather waits for the first half's store, so the root holds half of y at a time
        self.y_store_halves = bool(p.get("y_store_halves", False))
        # chunk_pipeline (y_halves): chunk i's front -- x, projections, sweep, post -- is made right after chunk i-1's
        # W_O block 0 and its reduce, before chunk i-1's block 1: the in-order GEMM core runs it while the root's xDMA
        # reduces block 0 (block 1 waits for that reduce anyway: one half-size exchange buffer)
        self.chunk_pipeline = bool(p.get("chunk_pipeline", False))
        if self.chunk_pipeline and not (self.y_halves and not self.y_store_halves and self.stop == "full"):
            raise ValueError("chunk_pipeline needs y_halves, no y_store_halves, stop full")
        if self.y_store_halves and not (self.y_halves or bool(p.get("y_block_chains", False))):
            raise ValueError("y_store_halves needs y_halves or y_block_chains")
        # chain_concurrent (default true): cluster 0's and cluster 1's ChainGathers run at the same time
        # through every chip. False: every ChainGather after the previous one of either cluster, so a
        # chain's finish never meets the other cluster's chain data in a chip's D2D framer (the MoE's
        # deadlock with a memory chip's push through a hop chip).
        self.chain_concurrent = bool(p.get("chain_concurrent", True))
        # kv_slabs (rr | split | split4): the sweep's K and V tiles through the W slabs in turn, or (split, three
        # slabs) K tiles in slab 0 and V tiles in slabs 1 and 2 alternately, or (split4, four slabs) K tiles in slabs 0
        # and 3 alternately as well, so K_{j+1} waits for QK(j-1), not QK(j), and loads under QK(j)
        self.kv_slabs = str(p.get("kv_slabs", "rr"))
        nwb = int(p.get("w_buffers", 3))
        if (self.kv_slabs not in ("rr", "split", "split4", "split_kb") or (self.kv_slabs in ("split", "split_kb") and nwb != 3)
                or (self.kv_slabs == "split4" and nwb != 4)):
            raise ValueError(f"params kv_slabs={self.kv_slabs!r}: rr, split or split_kb with w_buffers 3, or split4 with "
                             f"w_buffers 4")
        # split_kb (fresh): the odd key tiles in a one-tile buffer of each cluster, not a fourth ring slab
        self._kbuf, self._kb_reader = {}, {}
        # wo_split_rows: W_O as GEMMs of two 16-row tiles each instead of one of all the cluster's rows
        self.wo_split_rows = bool(p.get("wo_split_rows", False))
        # wo_row_pipe (post_tiles, wo_split_rows): each W_O GEMM of two 16-row tiles waits for those two tiles' o8 only,
        # not for the last tile's, so W_O starts while the last tiles are still relaid and quantised
        self.wo_row_pipe = bool(p.get("wo_row_pipe", False))
        if self.wo_row_pipe and not (self.post_tiles and self.wo_split_rows):
            raise ValueError("params wo_row_pipe: needs post_tiles and wo_split_rows")
        self._o8_tiles = {}
        # prefetch_after_x: the memory chips' pushes start after the multicast of x (fa_stages_mc)
        self.prefetch_after_x = bool(p.get("prefetch_after_x", False))
        # y_block_chains: a ChainGather per W_O block (the whole partial y in the exchange buffer)
        self.y_block_chains = bool(p.get("y_block_chains", False))
        if self.y_block_chains and (self.y_halves or self.reduce_impl != "chain" or self.sync == "barrier"):
            raise ValueError("y_block_chains: reduce chain, sync edge, no y_halves")
        # reduce_early (y_block_chains): chunk i-1's chains for root cluster c are made right after that cluster's q side
        # of chunk i, so on the root's in-order xDMA core they run during chunk i's sweep instead of behind its o side,
        # and chunk i's W_O (which waits for them: one exchange buffer) no longer waits for chunk i's front. With
        # y_store_halves each W_O block's sum lands in a half-size buffer, stored right after its chain
        self.reduce_early = bool(p.get("reduce_early", False))
        if self.reduce_early and not (self.y_block_chains and self.stop == "full"):
            raise ValueError("reduce_early: needs y_block_chains and stop full")
        # y_roots (y_block_chains, y_store_halves): the chips that collect and store y, one a W_O block in turn (chunk i's
        # block h: y_roots[(2 i + h) % len]), so the root is not every chain's collector and store; each chain folds the
        # cycle from its own collector (default: the root collects every block)
        self.y_roots = [int(str(k), 16) for k in p.get("y_roots", [])]
        if self.y_roots and not (self.y_block_chains and self.y_store_halves):
            raise ValueError("y_roots: needs y_block_chains and y_store_halves (a collector a W_O block)")
        # xch_per_block (y_block_chains): W_O block r of chunk i waits only for chunk i-1's chains of block r, which
        # read the half of the exchange buffer block r writes, instead of for both blocks' chains
        self.xch_per_block = bool(p.get("xch_per_block", False))
        # y_chain_l3 (y_block_chains, y_store_halves): each chain's last hop writes its block of y straight into the
        # collector's L3, where the store would have put it: no landing buffer in L1, no store
        self.y_chain_l3 = bool(p.get("y_chain_l3", False))
        if (self.xch_per_block or self.y_chain_l3) and not (self.y_block_chains and self.y_store_halves):
            raise ValueError("xch_per_block, y_chain_l3: need y_block_chains and y_store_halves")
        self._sym_blk = {}
        self._wo_blk = {}
        self._last_chain = None
        self._chains = []                           # every reduce chain, in creation order
        # last_pair_chains (with serial chains): the last chunk's two clusters' chains of a W_O block run at the same
        # time, once every chip's prefetcher has pushed its last chunk (no memory-chip push then transits a hop chip,
        # the third party of the concurrent-chain deadlock); the blocks stay in series
        self.last_pair_chains = bool(p.get("last_pair_chains", False))
        # chain_gate_prefetch: every reduce chain waits for every chip's prefetcher (all its pushes done), so with
        # chain_concurrent no memory-chip push can transit a hop chip while two chains run
        self.chain_gate_prefetch = bool(p.get("chain_gate_prefetch", False))
        # y_dst_shared (y_block_chains, y_store_halves): a chunk's two W_O blocks' chains (but the last chunk's) land in
        # one half-size buffer on each root cluster, block 1's chain behind block 0's store, so a root cluster's sweep keeps
        # its L1 while the previous chunk's chains run. With chains_after_sweep the root clusters' chains of a chunk are
        # made together, block 0 of every cluster first, and each block-0 store goes behind the root cluster's load of the
        # values of key tile dst_store_tile in its DM core's program, not behind the whole sweep's loads
        self.y_dst_shared = bool(p.get("y_dst_shared", False))
        self.dst_store_tile = int(p.get("dst_store_tile", 2))
        if self.y_dst_shared and not (self.y_block_chains and self.y_store_halves):
            raise ValueError("y_dst_shared: needs y_block_chains and y_store_halves")
        if self.y_dst_shared and self.y_chain_l3:
            raise ValueError("y_dst_shared: not with y_chain_l3 (its block-0 store frees the buffer)")
        self._pair_prev = None
        self.checks_kind = str(p.get("checks", "y"))
        # head h(k): a memory chip's row holds four consecutive heads, nearest the memory chip last
        by_row = sorted(self.chips, key=lambda k: (k & 15, k >> 4))
        self.head = {k: i for i, k in enumerate(by_row)}
        if len(self.head) != 16:
            raise ValueError(f"{len(self.head)} compute chips: the layer has 16 heads, one a chip")
        # the fold order of the 16 partials: the chips' order from the base class, as heads
        self.fold_heads = [self.head[k] for k in self.fold_order()]
        # the data's inputs: the cache of hidden states, the checkpoint's weights, the calibration tokens
        self.kv_path = os.path.join(REPO, p.get("kv_cache", "target/sim/automation/sweep/llm_blocks/fa/cache/"
                                                            "dsv2_l1_kv_8193.npz"))
        self.w_cache = os.path.join(REPO, p.get("weights", "target/sim/automation/sweep/common/weights/cache"))
        self.n_calib = int(p.get("n_calib", 32))
        self.skip_masked = bool(p.get("skip_masked", False))
        self.XB = self.C * HIDDEN                       # a chunk's x, INT8
        self.YB = self.ROWS * HIDDEN * 2                # a cluster's partial y of a chunk, FP16
        self.SL = int(p.get("root_slice", 16 * 1024))

    def mem_handles(self):
        """The data and the memory; no task yet: the data and its golden (fa_prefill_data.FaPrefillData) and their
        report; _stage (every input staged in L3 and HBM: x, the cache or the fresh prompt's key pieces and their
        flags, W_KVA, the weights and the dequantisation constants of each head, y); the load streams' L1 slabs
        (make_streams); one weight ring a chip (_make_rings); the L1 buffers (_buffers: the exchange buffer, the
        constants, the keys' constants)."""
        d = self.d = fa_prefill_data.FaPrefillData(self.kv_path, self.w_cache, tokens=self.T, keys=self.keys,
                                                   bc=self.bc, n_calib=self.n_calib, fresh=self.fresh,
                                                   chunk=self.C, rope_perm=self.rope_perm, mask_c=self.mask_c,
                                                   wbits=self.wbits, skip_masked=self.skip_masked)
        self.lay = d.layout
        for line in d.report(self.fold_heads):
            print(f"[fp] {line}")
        print(f"[fp] {self.T} tokens in {self.NCH} chunks of {self.C}: {self.NQ} query tiles of {BR} rows a "
              f"cluster, {self.NT} K/V tiles of {self.bc} keys; fold order "
              f"{[f'{k:02x}' for k in self.order]}")
        self._stage()
        self.make_streams()
        self._make_rings()
        self._buffers()

    # ---- the ring's plan: per token chunk ------------------------------------------------------
    PER_CHUNK_HEAD, PER_CHUNK_TAIL = 4, 3          # W_q x 3 and W_UK; W_UV and W_O x 2

    def kv_in_ring(self):
        """Ring chunks of K and V a token chunk takes: the cache's tiles (a fresh prompt's come from L3)."""
        return 0 if self.fresh else 2 * self.NT

    def kva_in_ring(self):
        """Ring chunks ahead of the token chunks: W_KVA's blocks (fresh; kv_cols: the chip's slice)."""
        return (1 if self.kv_cols else D_QK // 64) if self.fresh else 0

    def per_chunk(self):
        """Ring chunks a token chunk takes: a run that stops early takes only what it uses."""
        if self.stop == "x":
            return self.PER_CHUNK_HEAD if self.x_ring else 0
        if self.stop == "q8":
            return self.PER_CHUNK_HEAD
        return self.PER_CHUNK_HEAD + self.kv_in_ring() + (0 if self.stop == "sweep" else self.PER_CHUNK_TAIL)

    def chunk_index(self, i, part):
        """The ring index of token chunk i's first chunk of `part`: head (W_q, W_UK), kv, tail."""
        base = self.kva_in_ring() + i * self.per_chunk()
        return base + {"head": 0, "kv": self.PER_CHUNK_HEAD,
                       "tail": self.PER_CHUNK_HEAD + self.kv_in_ring()}[part]

    def kv_cluster(self, k, q):
        """The cluster of chip k that makes tile q's keys (params kv_clusters), cluster 0 otherwise."""
        return self.kv_tiles[k].index(q) % self.NC if self.kv_clusters else 0

    def kv_cls(self, k):
        """The clusters of chip k that make keys, in order."""
        return sorted({self.kv_cluster(k, q) for q in self.kv_tiles[k]})

    def flag_dep(self, t, k, c):
        """Whether task t enters a flag wait on chip k cluster c as a dependency, else as an ordering edge (the flag
        proves the landing): a task of that chip, and with params kv_clusters of that cluster too (the chip's tiles run
        on both its clusters, and the other cluster's puts would crowd the waiting core's dependency tags)."""
        return t.assigned_chiplet_id == k and (not self.kv_clusters or t.assigned_cluster_id == c)

    # ---- data -----------------------------------------------------------------------------------
    def _stage(self):
        st, d, T, lay = self.st, self.d, self.TAG, self.lay
        mesh = d.mesh
        # the prompt's rows, A layout per chunk: the root's source and every chip's landing
        self.xsrc = np.concatenate([d.x_chunk_a(i * self.C, self.C) for i in range(self.NCH)]).astype(np.int8)
        self.h_xsrc = (st.put_zeros(f"{T}_xsrc", "int8_t", self.xsrc.size, preload=True) if self.x_device or self.x_hbm
                       else st.put(f"{T}_xsrc", "int8_t", self.xsrc))
        if self.x_hbm:
            m = self.mem[self.ROOT]
            self.h_xhbm = st.put_hbm(f"{T}_xhbm", self.xsrc, mem_chip=(m >> 4, m & 15))
            st.put_zeros(f"{T}_xtflag", "uint8_t", (self.T // 16) * 64)
        st.put_zeros(f"{T}_xland", "int8_t", self.NCH * self.XB, preload=True)
        st.put_zeros(f"{T}_xflag", "uint8_t", self.NCH * 64)
        if self.owner_bcast:                                # a tile's x8 rows / its keys' piece landed (every chip)
            st.put_zeros(f"{T}_xtflag", "uint8_t", (self.T // 16) * 64)
            st.put_zeros(f"{T}_kvtflag", "uint8_t", (self.T // 16) * 64)
        if self.x_root == "flagged_zero":
            st.put_zeros(f"{T}_zero64", "uint8_t", 64, preload=True)
        if self.l3_overlay:
            if self.NC != 2 or self.XB != self.YB:
                raise ValueError("l3_overlay needs two clusters and a chunk's x the size of a cluster's y")
        else:
            st.put_zeros(f"{T}_y", "uint8_t", self.NCH * self.NC * self.YB, preload=True)
        # RoPE tables, per chunk and cluster: cos then sin_signed, D layout [rows, 64]
        rope = []
        for i in range(self.NCH):
            for c in range(self.NC):
                cs, sn = d.rope_tables(self.r0(i, c), self.ROWS)
                rope += [cs, sn]
        self.h_rope = st.put(f"{T}_rope", "uint16_t", np.concatenate(rope).astype(np.float16).view(np.uint16))
        self.ROPE_B = self.ROWS * ROPE * 2
        self.h_kv, self.h_w, self.h_c, self.h_wkv, self.h_kc = {}, {}, {}, {}, {}
        if self.fresh:
            # W_KVA, once in each memory chip's HBM; the 16-token pieces and their landed flags in every chip's L3
            wkv = np.concatenate(d.kva_chunks()).astype(np.int8)
            for m in sorted(set(self.mem.values())):
                self.h_wkv[m] = st.put_hbm(f"{T}_wkv_m{m:02x}", wkv, mem_chip=(m >> 4, m & 15))
            st.put_zeros(f"{T}_kv", "int8_t", (self.T // 16) * KVP, preload=True)
            st.put_zeros(f"{T}_kvflag", "uint8_t", self.NCH * 64)          # a chunk's pieces landed (multicast)
            st.put_zeros(f"{T}_kvnflag", "uint8_t", (self.T // 16) * 64)   # a piece landed on the root
            if self.kv_cols:
                # each chip's W_KVA slice in its memory chip's HBM; per owned tile the gathered kv and a flag a chip
                self.h_wkvc = {k: st.put_hbm(f"{T}_wkvc_k{k:02x}", d.kva_cols(16 * self.kvg[k][0], 16 * self.kvg[k][1]),
                                             mem_chip=(self.mem[k] >> 4, self.mem[k] & 15)) for k in self.chips}
                st.put_zeros(f"{T}_kvraw", "uint8_t", self.KVR_TILES * self.KVD)
                st.put_zeros(f"{T}_kvrflag", "uint8_t", self.KVR_TILES * len(self.chips) * 64)
            # per chip: W_KVA's dequantisation (D layout [16, 576]), then each of its tiles' RoPE tables (cos,
            # sin_signed), KC_TILE bytes a tile
            kva = d.scale_tile(d.kva_scale()).view(np.uint8)
            for k in self.chips:
                tabs = []
                for q in self.kv_tiles[k]:
                    cs, sn = d.rope_tables(16 * q, 16)
                    tabs += [np.asarray(t, dtype=np.float16).view(np.uint8) for t in (cs, sn)]
                img = np.concatenate([kva] + tabs)
                self.h_kc[k] = st.put_hbm(f"{T}_kc_k{k:02x}", img.astype(np.uint8),
                                          mem_chip=(self.mem[k] >> 4, self.mem[k] & 15))
            self.KC_ROPE, self.KC_TILE = kva.size, 2 * 16 * ROPE * 2
            self.KC_BYTES = kva.size + len(self.kv_tiles[self.ROOT]) * self.KC_TILE
        else:
            # the cache, once in each memory chip's HBM, tile by tile K0 V0 K1 V1 ...
            tiles = d.cache_tiles()
            cache = np.concatenate([np.concatenate([k, v]) for k, v in tiles]).astype(np.int8)
            self.kv_bytes = [(k.size, v.size) for k, v in tiles]
            for m in sorted(set(self.mem.values())):
                self.h_kv[m] = st.put_hbm(f"{T}_kv_m{m:02x}", cache, mem_chip=(m >> 4, m & 15))
        # per head: the weights as the ring streams them, and the dequantisation constants
        self.w_off = {}
        for k in self.chips:
            h = self.head[k]
            loc = (self.mem[k] >> 4, self.mem[k] & 15)
            chunks = d.weight_chunks(h, wo_block=h if self.wo_cols else None)
            self.w_off[k], off = [], 0
            for nm, b in chunks:
                self.w_off[k].append((nm, off, b.size))
                off += b.size
            self.h_w[k] = st.put_hbm(f"{T}_w_k{k:02x}", np.concatenate([b for _, b in chunks]), mem_chip=loc)
            # W_q's per-column scales, one copy per 16-row tile of the cluster's rows: dq{r} multiplies all
            # of them element by element
            sq = np.concatenate([np.tile(d.scale_tile(d.q_scale(h)[64 * r:64 * (r + 1)]), self.ROWS // 16)
                                 for r in range(3)])
            sqt, suv = d.scale_tile(d.qt_scale(h)), d.scale_tile(d.uv_scale(h))
            if self.dq_fused:
                # q_nope's 128 scales and o's in the A layout of the cluster's rows, q~'s of one query tile, where the
                # fused tasks multiply the reshaped values; q_pe's stay D (RoPE reads them dequantised)
                qs = d.q_scale(h)
                sq = np.concatenate([d.scale_tile_a(qs[:Q_NOPE], self.ROWS).view(np.uint8),
                                     np.tile(d.scale_tile(qs[Q_NOPE:]), self.ROWS // 16).view(np.uint8)])
                sqt, suv = d.scale_tile_a(d.qt_scale(h), 16), d.scale_tile_a(d.uv_scale(h), self.ROWS)
            masks = d.mask_bias_patterns() if self.mask_c else [m.astype(np.float16) for m in d.mask_patterns]
            const = np.concatenate([sq.view(np.uint8), sqt.view(np.uint8), suv.view(np.uint8)]
                                   + ([np.concatenate([m.reshape(-1) for m in masks]).view(np.uint8)]
                                      if self.fresh else []))
            self.h_c[k] = st.put_hbm(f"{T}_c_k{k:02x}", const.astype(np.uint8), mem_chip=loc)
        if self.o_land:
            st.put_zeros(f"{T}_oland", "int8_t", 2 * self.NC * len(self.chips) * self.ROWS * Q_NOPE, preload=True)
        if self.wo_cols:
            st.put_zeros(f"{T}_oflag", "uint8_t", self.NCH * self.NC * len(self.chips) * 64)   # a head's o8 landed (root)
            st.put_zeros(f"{T}_omflag", "uint8_t", self.NCH * 64)                             # the root's multicast landed
            st.put_zeros(f"{T}_yflag", "uint8_t", self.NCH * self.N_CL * 64)                 # a cluster's y landed
            st.put_zeros(f"{T}_ydum", "uint8_t", self.N_CL * 64)                             # the flag puts' 64 bytes
        self.SQ_B = (self.ROWS // 16) * 2048           # one W_q block's scales: 16 rows x 64 FP16 a row tile
        self.C_SQ, self.C_QT = 0, 3 * self.SQ_B
        self.C_UV = self.C_QT + 16384
        # fresh: the causal masks after the scales, FP16 [bc, 16] each (fa_prefill_data.py tile_plan), or under mask_c
        # the score matmul's INT32 [bc, 16] biases
        self.C_MASK = self.C_UV + (self.ROWS * Q_NOPE * 2 if self.dq_fused else 4096)
        self.MASK_B = self.bc * BR * (4 if self.mask_c else 2)
        self.C_BYTES = self.C_MASK + (len(d.mask_patterns) * self.MASK_B if self.fresh else 0)

    def _make_rings(self):
        """One ring a chip, shared by its two clusters: the per-chip schedule is the same for both."""
        p, st, T = self.p, self.st, self.TAG
        wr = p.get("weight_ring", {})
        self.n_slots = int(wr.get("slots", 16))
        self.ring_batch = int(wr.get("batch", 4))
        self.ring_trailer = False
        self.ring_head = int(wr.get("head", 0))
        self.ring_batch_head = int(wr.get("batch_head", 0))
        self.ring_queued = int(bool(wr.get("queued", False)))
        seq = np.zeros((1024, 8), dtype=np.uint64)
        seq[:, 0] = np.arange(1024, dtype=np.uint64)
        mems = sorted(set(self.mem.values()))
        self.h_seq = {m: st.put_hbm(f"{T}_ring_seq_m{m:02x}", seq.view(np.int8).reshape(-1),
                                    mem_chip=(m >> 4, m & 15)) for m in mems}
        self.h_flags = st.put_zeros(f"{T}_ring_flags", "uint8_t", self.n_slots * 64)
        self.h_rel = st.put_zeros(f"{T}_ring_release", "uint8_t", self.n_slots * 64)
        self.h_rec = st.put_zeros(f"{T}_rec_l3", "uint8_t", 128)
        self.h_wslots = st.put_zeros(f"{T}_wring_slots", "uint8_t", self.n_slots * self.slot_bytes, preload=True)
        self.rings, self.prefetch, self.ring_take, self.ring_pos = {}, {}, {}, {}
        from bingo_kernel_args import HostBingoKernelWeightPrefetchArgs
        for k in self.chips:
            self.rings[k] = WeightRings(
                slots=self.h_wslots, flags=self.h_flags, releases=self.h_rel, n_rings=1, n_slots=self.n_slots,
                slot_bytes=self.slot_bytes, batch=self.ring_batch, head=self.ring_head,
                batch_head=self.ring_batch_head)
            self.ring_take[k] = []
            self.ring_pos[k] = {}
            self.prefetch[k] = self.cx_chip(k).host_node(
                f"weight_prefetch_k{k:02x}", "__host_bingo_kernel_weight_prefetch",
                HostBingoKernelWeightPrefetchArgs(
                    self.ring_schedule(k, self.rings[k]), self.rings[k].slots, self.h_flags, self.h_rel,
                    self.h_seq[self.mem[k]], self.h_rec, 1, 1, self.n_slots, self.slot_bytes, self.mem[k],
                    batch=self.ring_batch, policy=int(wr.get("policy", 1)), engine=self.engine[k],
                    queued=self.ring_queued, head=self.ring_head, batch_head=self.ring_batch_head))
        self.ld_node = {}
        # (chip, ring chunk) whose slot cluster 1's read releases, after cluster 0's; every other chunk's slot is
        # released by cluster 0's read
        self.ring_rel1 = set()

    def _src(self, k, n):
        """(HBM address, bytes) of chip k's ring chunk n."""
        if n < self.kva_in_ring() and self.kv_cols:  # the chip's W_KVA slice (fresh, kv_cols)
            gr0, gr1 = self.kvg[k]
            return self.h_wkvc[k].address, HIDDEN * 16 * (gr1 - gr0) * self.wbits // 8
        if n < self.kva_in_ring():                  # W_KVA's blocks (fresh)
            nb = HIDDEN * 64 * self.wbits // 8
            return self.h_wkv[self.mem[k]].address + n * nb, nb
        r = (n - self.kva_in_ring()) % self.per_chunk()
        if r < 4:                                   # W_q x 3, W_UK
            nm, off, size = self.w_off[k][r]
            return self.h_w[k].address + off, size
        if r < 4 + self.kv_in_ring():               # K, V
            j, v = divmod(r - 4, 2)
            off = sum(a + b for a, b in self.kv_bytes[:j]) + (self.kv_bytes[j][0] if v else 0)
            return self.h_kv[self.mem[k]].address + off, self.kv_bytes[j][v]
        nm, off, size = self.w_off[k][4 + r - 4 - self.kv_in_ring()]      # W_UV, W_O x 2
        return self.h_w[k].address + off, size

    # ---- buffers --------------------------------------------------------------------------------
    def _buffers(self):
        d, R = self.d, self.ROOT
        mesh = d.mesh
        self.xch_bytes = self.YB // 2 if self.y_halves else self.YB
        self.xch_zero = False
        self.xch = {(k, c): self.cx_chip(k).at(c).l1(f"AAAA_{self.TAG}_xch", self.xch_bytes)
                    for k in self.chips for c in range(self.NC)}
        self.sweep_cfg = FaSweepCfg(bc=self.bc, nt=self.NT, nq=self.NQ, br=BR, k_s=d.k_s, k_o=d.k_o,
                                    a_exp=d.a_exp, a_n_f32bits=d.a_n_bits)
        self.consts = {cl: self.cx(cl).l1(f"{self.TAG}_consts", self.C_BYTES)
                       for cl in range(self.N_CL)}
        # fresh: W_KVA's dequantisation and the chip's tiles' RoPE tables, on each cluster that makes keys (stage_kv)
        self.kvc = ({(k, c): self.cx_chip(k).at(c).l1(f"{self.TAG}_kvc", self.KC_BYTES)
                     for k in self.chips for c in self.kv_cls(k)} if self.fresh else {})
        self._bufs, self._dst, self._slot, self._root_wait = {}, {}, {}, {}

    def y_root(self, i, half=None):
        """The chip that collects and stores chunk i's y (half `half`: its W_O block): params y_roots, else the root."""
        if not self.y_roots:
            return self.ROOT
        r = self.y_roots[(2 * i + (half or 0)) % len(self.y_roots)]
        if r not in self.order:
            raise ValueError(f"y_roots: chip {r:02x} is not a compute chip")
        return r

    def y_order(self, r):
        """The fold cycle from collector r: r, then the cycle on from it (self.order is one cycle, so every hop of a
        chain stays between neighbours)."""
        j = self.order.index(r)
        return self.order[j:] + self.order[:j]

    def y_fold_heads(self, r):
        """The heads in the order collector r's chain adds their partials: the chain's chips, r's own partial last."""
        o = self.y_order(r)
        return [self.head[k] for k in o[1:] + o[:1]]

    def dst_shared(self, i):
        """y_dst_shared: chunk i's two blocks land in one buffer (every chunk but the last)."""
        return self.y_dst_shared and i < self.NCH - 1

    def dst(self, i, c, half=None):
        """The collector's cluster c landing buffer of chunk i's reduce: the reduced rows, then stored (y_store_halves:
        each half's own buffer; y_dst_shared: one buffer the two halves take in turn)."""
        h = "s" if half is not None and self.dst_shared(i) else half
        key = (i, c) if not self.y_store_halves or half is None else (i, c, h)
        if key not in self._dst:
            name, nb = (f"{self.TAG}_dst_i{i}", self.YB) if len(key) == 2 else (f"{self.TAG}_dst_i{i}h{h}", self.YB // 2)
            self._dst[key] = self.cx_chip(self.y_root(i, half)).at(c).l1(name, nb)
        return self._dst[key]

    def slot(self, i, c):
        """The root arm's two landing slices of the pulled partials."""
        if (i, c) not in self._slot:
            self._slot[i, c] = self.cx_chip(self.ROOT).at(c).l1(f"{self.TAG}_slots_i{i}", 2 * self.SL)
        return self._slot[i, c]

    def y_sym(self, i, c, off=0, chip_id=None):
        """Where the root stores cluster c's rows of y of chunk i (see l3_overlay)."""
        if self.l3_overlay:
            name, base = (f"{self.TAG}_xland" if c == 0 else f"{self.TAG}_xsrc"), i * self.XB
        else:
            name, base = f"{self.TAG}_y", (i * self.NC + c) * self.YB
        if chip_id is None:
            return BingoMemSymbol(name, offset=base + off)
        return BingoMemSymbol(name, offset=base + off, chip_id=chip_id)

    def y_sym_local(self, i, c, off=0):
        """y_sym as a C expression of the local address (the same on every chip), for a chain's hop."""
        s = self.y_sym(i, c, off)
        return f"((uintptr_t){s.symbol_name} + {s.offset})"

    def bufs(self, cl, i):
        """The buffers of chunk i on cluster cl. A chunk has its own, so the static L1 planner overlays
        the ones whose lifetimes cannot meet (the projection's with the previous sweep's, say)."""
        if (cl, i) in self._bufs:
            return self._bufs[cl, i]
        k, c = self.chip(cl), self.local(cl)
        rows, nq = self.ROWS, self.NQ
        l1 = lambda name, nb: self.cx_chip(k).at(c).l1(f"{self.TAG}_{name}_i{i}", nb)
        b = {}
        b["xrows"] = l1("xrows", rows * HIDDEN)
        if self.wo_cols:
            # every head's o8 of the cluster's rows, A layout [rows, 2048], and the cluster's 128 columns of y; named to
            # sort right after the x rows, whose bytes the first takes (the static packer places by name, first fit)
            b["xrowso"] = l1("xrowso", rows * HIDDEN)
            b["xrowsy"] = l1("xrowsy", rows * 128 * 2)
        # q: three D regions of 64 columns (the last is q_pe), rows x 64 FP16 each
        if self.rope_fused:
            # simd_rope's operand block [x | cos | xswap | sin], ROPE_B each: q_pe's GEMM and dequantisation write
            # slot 0, the tables slots 1 and 3, the pair swap slot 2
            b["rope4"] = l1("rope4", 4 * self.ROPE_B)
            b["q"] = [l1(f"q{r}", rows * 64 * 2) for r in range(2)] + [b["rope4"]]
        else:
            b["ropeb"] = l1("ropeb", 2 * self.ROPE_B)
            b["q"] = [l1(f"q{r}", rows * 64 * 2) for r in range(3)]
        b["a16n"] = l1("a16n", rows * Q_NOPE * 2)             # q_nope, A layout FP16
        b["qn8"] = l1("qn8", rows * Q_NOPE)
        b["qt"] = l1("qt", rows * KV_RANK * 2)                # q~, D layout FP16
        b["a16q"] = l1("a16q", rows * KV_RANK * 2)
        if not self.rope_fused:
            b["xs"] = l1("xs", rows * ROPE * 2)
            b["t1"] = l1("t1", rows * ROPE * 2)
            b["t2"] = l1("t2", rows * ROPE * 2)
        b["pe"] = l1("pe", rows * ROPE * 2)
        b["a16p"] = l1("a16p", rows * ROPE * 2)
        b["q8"] = [l1(f"q8_{q}", D_QK * BR) for q in range(nq)]
        b["a16o"] = l1("a16o", rows * D_V * 2)                # o~, A layout FP16
        b["ot8"] = l1("ot8", rows * D_V)
        b["uv"] = l1("uv", rows * Q_NOPE * 2)                 # o_h, D layout FP16
        b["a16u"] = l1("a16u", rows * Q_NOPE * 2)
        b["o8"] = l1("o8", rows * Q_NOPE)
        # the sweep's names keep their sort order inside a chunk (the softmax's strides need it)
        b["swp"] = fa_sweep_alloc(lambda name, nb: self.cx_chip(k).at(c).l1(f"{self.TAG}_{name}_i{i}", nb),
                                  self.sweep_cfg)
        self._bufs[cl, i] = b
        return b

    # ---- the ring's loads -----------------------------------------------------------------------
    def load(self, cl, n, name, deps=(), slab=None, dst=None):
        """Ring chunk n of chip(cl) into cluster cl's next slab: cluster 1 reads it, cluster 0 reads it and
        releases the slot, after cluster 1's read. Returns (load task, slab index, slab). dst: into that L1
        buffer instead, outside the cluster's load stream (slab index None, no WAR edge)."""
        k, c, dm = self.chip(cl), self.local(cl), self.roles["dm"]
        takes, pos = self.ring_take[k], self.ring_pos[k]
        if n not in pos:
            # a chunk's place in the ring is the order the program first asks for it; that is chunk n's
            # own place, except under chunk_pipeline: chunk i's front asks before chunk i-1's last W_O block
            if n != len(takes) and not self.chunk_pipeline:
                raise ValueError(f"ring chunk {n} of chip {k:#04x} asked for before {len(takes)}")
            src, nb = self._src(k, n)
            pos[n] = len(takes)
            takes.append(self.rings[k].take_ex(0, src=src, nbytes=nb) + (nb, self.rings[k].last_take))
        sl, fl, rl, seq, _, _, nb, ex = takes[pos[n]]
        ls = self.ls[cl]
        if dst is not None:
            si, slab, war = None, dst, []
        else:
            si, slab, war = ls.take(nb) if slab is None else ls.take_at(slab, nb)
        prev = [self.last_ld[cl]] if cl in self.last_ld else []
        # the slot is released by one cluster's read (cluster 0's, or cluster 1's for a chunk in ring_rel1), which
        # follows every other cluster's read of it
        rel = 1 if (k, n) in self.ring_rel1 else 0
        readers = [self.ld_node[(k, c2, n)] for c2 in range(self.NC) if c2 != rel and (k, c2, n) in self.ld_node] \
            if c == rel else []
        ld = self.node(k, c, dm, f"{name}", "__snax_bingo_kernel_idma_ring_load",
                       SnaxBingoKernelIdmaRingLoadArgs(fl, seq, sl, slab, nb, rl if c == rel else 0),
                       list(dict.fromkeys(war + prev + list(deps) + readers)))
        if c != rel and (k, rel, n) in self.ld_node:
            self.dfg.bingo_add_edge(ld, self.ld_node[(k, rel, n)])
        self.last_ld[cl] = ld
        ld.exec_chunk = ex
        self.ld_node[(k, c, n)] = ld
        return ld, si, slab

    # ---- helpers ----------------------------------------------------------------------------
    def gmm(self, cl, name, a, b, d, M, K, N, k_shift, deps):
        """One GEMM of cluster cl (gemm_full). Returns the task its output follows."""
        k, c = self.chip(cl), self.local(cl)
        args = SnaxBingoKernelGemmFullArgs(
            input_A_addr=a, input_B_addr=b, input_C_addr=0, output_D_addr=d, M=M, K=K, N=N, array_shape_idx=0,
            transpose_A=0, transpose_B=0, accumPrevC=0, int32tofp16_enable=1,
            int4_b_enable=INT4_B_LAYOUT_PAIRED if self.wbits == 4 else 0, d_shift=k_shift)
        return self.node(k, c, self.roles["gemm"], name, "__snax_bingo_kernel_gemm_full", args, list(deps))

    def simd_mul(self, cl, name, a, b, out, nelems, deps):
        k, c = self.chip(cl), self.local(cl)
        return self.node(k, c, self.roles["simd"], name, "__snax_bingo_kernel_simd_stream_elementwise",
                         SnaxBingoKernelSimdMulF16Args(a, b, out, nelems // 32, 32), list(deps))

    def simd_add(self, cl, name, a, b, out, nelems, deps):
        k, c = self.chip(cl), self.local(cl)
        return self.node(k, c, self.roles["simd"], name, "__snax_bingo_kernel_simd_stream_elementwise",
                         SnaxBingoKernelSimdAddF16Args(a, b, out, nelems // 32, 32), list(deps))

    def mul_quant(self, cl, name, a, scale, dst, nelems, inv_bits, deps):
        """dst = quantise(a (.) scale): a dequantisation and the quantise after it in one SIMD task (dq_fused)."""
        k, c = self.chip(cl), self.local(cl)
        return self.node(k, c, self.roles["simd"], name, "__snax_bingo_kernel_simd_stream_elementwise",
                         SnaxBingoKernelSimdMulI8Args(a, scale, dst, nelems // 32, 32, inv_bits), list(deps))

    def quant(self, cl, name, src, dst, nbytes, inv_bits, deps):
        k, c = self.chip(cl), self.local(cl)
        return self.node(k, c, self.roles["simd"], name, "__snax_bingo_kernel_simd_fp16_to_int8",
                         SnaxBingoKernelSimdFp16ToInt8Args(src, dst, beats=simd_beats(nbytes), rows=1,
                                                           inv_scale_f32bits=inv_bits), list(deps))

    def xd(self, cl, name, args, deps):
        k, c = self.chip(cl), self.local(cl)
        return self.node(k, c, self.roles["xdma"], name, "__snax_bingo_kernel_xdma_6d", args, list(deps))

    def dm_copy(self, cl, name, src, dst, nbytes, deps):
        k, c = self.chip(cl), self.local(cl)
        return self.node(k, c, self.roles["dm"], name, "__snax_bingo_kernel_idma_1d_copy",
                         SnaxBingoKernelIdma1dCopyArgs(src, dst, nbytes), list(deps))

    # ---- the stages of one chunk ----------------------------------------------------------------
    def stage_x_put(self, i, after):
        T, R = self.TAG, self.ROOT
        return self.node(R, self.x_put_cluster, self.roles["dm"], f"x_bput_i{i}", "__snax_bingo_kernel_idma_put_flagged",
                         SnaxBingoKernelIdmaPutFlaggedArgs(
                             BingoMemSymbol(f"{T}_xsrc", offset=i * self.XB),
                             BingoMemSymbol(f"{T}_xland", offset=i * self.XB), self.XB,
                             BingoMemSymbol(f"{T}_xflag", offset=i * 64), value=1, bcast=True, also_here=True),
                         list(after) + list(self.x_ready.get(i, [])))

    def stage_x(self, cl, i, put, deps):
        """Cluster cl's rows of chunk i into its L1: behind the landed flag, or pulled from the root."""
        T, k, c = self.TAG, self.chip(cl), self.local(cl)
        half = self.XB // self.NC
        xb = self.bufs(cl, i)["xrows"]
        if self.x_resident:
            return self.node(k, c, self.roles["dm"], f"x_copy_i{i}", "__snax_bingo_kernel_idma_1d_copy",
                             SnaxBingoKernelIdma1dCopyArgs(BingoMemSymbol(f"{T}_xsrc", offset=i * self.XB + self.xh(i, c) * half),
                                                           xb, half), list(deps))
        if self.owner_bcast:
            # the cluster's rows are tiles q0 .. q0 + n - 1 of chunk i, each multicast by its owner with its own flag
            n, q0 = self.ROWS // 16, self.r0(i, c) // 16
            puts = [t for t in self.x_ready.get(i, [])]
            nd = self.node(k, c, self.roles["dm"], f"x_wait_i{i}", "__snax_bingo_kernel_idma_fetch_flagged",
                           SnaxBingoKernelIdmaFetchFlaggedArgs(
                               BingoMemSymbol(f"{T}_xland", offset=i * self.XB + self.xh(i, c) * half), xb, half,
                               BingoMemSymbol(f"{T}_xtflag", offset=q0 * 64), n, 1),
                           list(deps) + [t for t in puts if self.flag_dep(t, k, c)])
            for t in puts:
                if not self.flag_dep(t, k, c):
                    self.dfg.bingo_add_order_edge(t, nd)
            return nd
        if self.bcast_impl == "bcast_l1" and k == self.ROOT and self.x_root == "poll2d":
            # the flagged fetch's two DMAs as two tasks: the 4-byte 2D read of the flag into the rows'
            # buffer, then the 1D copy of the rows from the landing buffer
            poll = self.node(k, c, self.roles["dm"], f"x_poll_i{i}", "__snax_bingo_kernel_idma_2d_copy",
                             SnaxBingoKernelIdma2dCopyArgs(BingoMemSymbol(f"{T}_xflag", offset=i * 64), xb, 4, 64, 4, 1),
                             list(deps) + [put])
            return self.node(k, c, self.roles["dm"], f"x_copy_i{i}", "__snax_bingo_kernel_idma_1d_copy",
                             SnaxBingoKernelIdma1dCopyArgs(BingoMemSymbol(f"{T}_xland", offset=i * self.XB + self.xh(i, c) * half),
                                                           xb, half), [poll])
        if self.bcast_impl == "bcast_l1" and k == self.ROOT and self.x_root == "copy":
            return self.node(k, c, self.roles["dm"], f"x_copy_i{i}", "__snax_bingo_kernel_idma_1d_copy",
                             SnaxBingoKernelIdma1dCopyArgs(BingoMemSymbol(f"{T}_xsrc", offset=i * self.XB + self.xh(i, c) * half),
                                                           xb, half), list(deps) + [put])
        if self.bcast_impl == "bcast_l1":
            # the root's clusters poll the same L3 flag word: cluster c starts its fetch after cluster c-1's
            first = [self._root_wait[i, c - 1]] if k == self.ROOT and (i, c - 1) in self._root_wait else []
            if k == self.ROOT and self.x_root == "flagged_zero":
                # the fetch polls its flag into the head of its destination: give that word a value first
                first.append(self.node(k, c, self.roles["dm"], f"x_zero_i{i}", "__snax_bingo_kernel_idma_1d_copy",
                                       SnaxBingoKernelIdma1dCopyArgs(BingoMemSymbol(f"{T}_zero64"), xb, 64),
                                       list(deps) + first))
            nd = self.node(k, c, self.roles["dm"], f"x_wait_i{i}", "__snax_bingo_kernel_idma_fetch_flagged",
                           SnaxBingoKernelIdmaFetchFlaggedArgs(
                               BingoMemSymbol(f"{T}_xland", offset=i * self.XB + self.xh(i, c) * half), xb, half,
                               BingoMemSymbol(f"{T}_xflag", offset=i * 64), 1, 1),
                           list(deps) + first + ([put] if k == self.ROOT else []))
            if k == self.ROOT:
                self._root_wait[i, c] = nd
            if k != self.ROOT:
                self.dfg.bingo_add_order_edge(put, nd)
            return nd
        return self.node(k, c, self.roles["dm"], f"x_pull_i{i}", "__snax_bingo_kernel_idma_1d_copy",
                         SnaxBingoKernelIdma1dCopyArgs(
                             BingoMemSymbol(f"{T}_xsrc", offset=i * self.XB + self.xh(i, c) * half, chip_id=self.ROOT), xb, half),
                         list(deps))

    def stage_proj(self, cl, i, x):
        """q, q~ and the rotated q_pe of chunk i into Q8, one query tile at a time. Returns
        {q: (Q8 handle, [tasks it follows])} and the tasks that free the chunk's x rows."""
        d, b = self.d, self.bufs(cl, i)
        k, c = self.chip(cl), self.local(cl)
        rows, mt = self.ROWS, self.ROWS // 16
        mesh = d.mesh
        # constants (once), the rotation tables of this chunk
        K0 = self.C_SQ
        sq = [self.consts[cl].view(K0 + self.SQ_B * r) for r in range(3)]
        # ---- q = x W_q, three blocks of 64 columns; the third is q_pe --------------------------
        qd, gq = [], []
        for r in range(3):
            n = self.chunk_index(i, "head") + r
            ld, si, slab = self.load(cl, n, f"Ld_wq{r}_i{i}")
            deps = [ld, x]
            gm = self.gmm(cl, f"gq{r}_i{i}", b["xrows"], slab, b["q"][r], mt, HIDDEN // 4, 4, d.k_x, deps)
            self.ls[cl].release(si, gm)
            gq.append(gm)
        # ---- dequantise the three regions --------------------------------------------------------
        dq = [None if self.dq_fused and r < 2 else
              self.simd_mul(cl, f"dq{r}_i{i}", b["q"][r], sq[r], b["q"][r], rows * 64, [gq[r]]) for r in range(3)]
        # ---- q_nope: D -> A (FP16) per row block and half, then quantise -------------------------
        d2a = []
        for r in range(2):
            for m in range(mt):
                d2a.append(self.xd(cl, f"d2an{r}{m}_i{i}", d_to_a_args(
                    mesh, 1, 4, 16, b["q"][r].view(m * 4 * TILE16), b["a16n"].view(m * 16 * Q_NOPE * 2 + r * 16 * 64 * 2)),
                    [gq[r]] if self.dq_fused else [dq[r]]))
        if self.dq_fused:
            qn = self.mul_quant(cl, f"qn_i{i}", b["a16n"], sq[0], b["qn8"], rows * Q_NOPE, self.fbits(d.s_qn.inv), d2a)
        else:
            qn = self.quant(cl, f"qn_i{i}", b["a16n"], b["qn8"], rows * Q_NOPE * 2, self.fbits(d.s_qn.inv),
                            d2a)
        # ---- absorb: q~ = q_nope W_UK ----------------------------------------------------------------
        ldu, siu, slabu = self.load(cl, self.chunk_index(i, "head") + 3, f"Ld_wuk_i{i}")
        ga = self.gmm(cl, f"gab_i{i}", b["qn8"], slabu, b["qt"], mt, Q_NOPE // 4, KV_RANK // 16, d.k_uk,
                      [ldu, qn])
        self.ls[cl].release(siu, ga)
        sqt = self.consts[cl].view(self.C_QT)
        dqt = [ga] if self.dq_fused else \
            [self.simd_mul(cl, f"dqt{m}_i{i}", b["qt"].view(m * (KV_RANK // 16) * TILE16), sqt,
                           b["qt"].view(m * (KV_RANK // 16) * TILE16), 16 * KV_RANK, [ga]) for m in range(mt)]
        d2q = self.xd(cl, f"d2aq_i{i}", d_to_a_args(mesh, mt, KV_RANK // 16, KV_RANK // 4, b["qt"], b["a16q"]),
                      dqt)
        # ---- q_pe: RoPE in the D layout, then D -> A and quantise ------------------------------------
        rope_src = BingoMemSymbol(f"{self.TAG}_rope", offset=2 * (i * self.NC + c) * self.ROPE_B)
        if self.rope_fused:
            # cos into slot 1, sin_signed into slot 3: the two tables are adjacent in L3, a slot apart in the block
            R4 = b["rope4"]
            ldr = self.node(k, c, self.roles["dm"], f"rope_ld_i{i}", "__snax_bingo_kernel_idma_2d_copy",
                            SnaxBingoKernelIdma2dCopyArgs(rope_src, R4.view(self.ROPE_B), self.ROPE_B, self.ROPE_B,
                                                          2 * self.ROPE_B, 2), [])
            if self.rope_perm:
                # each row tile's D layout is its four 16-column blocks: the swap is the first two and the last two
                # exchanged, one 2-D copy for each direction
                half, mtb = (ROPE // 32) * TILE16, (ROPE // 16) * TILE16
                sw = [self.node(k, c, self.roles["dm"], f"swap{h}_i{i}", "__snax_bingo_kernel_idma_2d_copy",
                                SnaxBingoKernelIdma2dCopyArgs(R4.view((1 - h) * half), R4.view(2 * self.ROPE_B + h * half),
                                                              half, mtb, mtb, mt), [dq[2]]) for h in range(2)]
            else:
                sw = [self.node(k, c, self.roles["dm"], f"swap_i{i}", "__snax_bingo_kernel_idma_pairwise_swap",
                                SnaxBingoKernelIdmaPairwiseSwapArgs(R4, R4.view(2 * self.ROPE_B), rows * ROPE, 2),
                                [dq[2]])]
            ad = self.node(k, c, self.roles["simd"], f"rope_i{i}", "__snax_bingo_kernel_simd_rope",
                           SnaxBingoKernelSimdRopeArgs(R4, b["pe"], ROPE, rows), [dq[2], ldr] + sw)
        else:
            ldr = self.dm_copy(cl, f"rope_ld_i{i}", rope_src, b["ropeb"], 2 * self.ROPE_B, [])
            sw = self.node(k, c, self.roles["dm"], f"swap_i{i}", "__snax_bingo_kernel_idma_pairwise_swap",
                           SnaxBingoKernelIdmaPairwiseSwapArgs(b["q"][2], b["xs"], rows * ROPE, 2), [dq[2]])
            n_pe = rows * ROPE
            m1 = self.simd_mul(cl, f"rope_cos_i{i}", b["q"][2], b["ropeb"], b["t1"], n_pe, [dq[2], ldr])
            m2 = self.simd_mul(cl, f"rope_sin_i{i}", b["xs"], b["ropeb"].view(self.ROPE_B), b["t2"], n_pe, [sw, ldr])
            ad = self.simd_add(cl, f"rope_add_i{i}", b["t1"], b["t2"], b["pe"], n_pe, [m1, m2])
        d2p = self.xd(cl, f"d2ap_i{i}", d_to_a_args(mesh, mt, 4, 16, b["pe"], b["a16p"]),
                      [ad])
        # ---- Q8 per query tile ---------------------------------------------------------------------
        q8 = {}
        for q in range(self.NQ):
            if self.dq_fused:
                qa = self.mul_quant(cl, f"q8a{q}_i{i}", b["a16q"].view(q * 16 * KV_RANK * 2), sqt, b["q8"][q],
                                    16 * KV_RANK, self.fbits(d.s_qt.inv), [d2q])
            else:
                qa = self.quant(cl, f"q8a{q}_i{i}", b["a16q"].view(q * 16 * KV_RANK * 2), b["q8"][q], 16 * KV_RANK * 2,
                                self.fbits(d.s_qt.inv), [d2q])
            qp = self.quant(cl, f"q8p{q}_i{i}", b["a16p"].view(q * 16 * ROPE * 2), b["q8"][q].view(16 * KV_RANK),
                            16 * ROPE * 2, self.fbits(d.s_qpe.inv), [d2p])
            q8[q] = (b["q8"][q], [qa, qp])
        if self.q_tiles:
            return q8
        # the sweep starts behind every projection task, so its buffers and the projection's can share L1
        done = [t for q in q8.values() for t in q[1]]
        return {q: (h, done) for q, (h, _) in q8.items()}

    @staticmethod
    def fbits(inv):
        return int(np.asarray(inv, dtype=np.float32).reshape(()).view(np.uint32))

    # ---- a fresh prompt's keys ------------------------------------------------------------------
    def fresh_tiles(self, go):
        """The block alone with a fresh prompt (no layer making x on the device): each chip's 16-token tiles of x8, A
        layout, from its own copy of the staged x source into the L1 of the cluster that makes their keys
        (kv_cluster): kv_src, as a layer's input norm leaves it."""
        T, dm, rows = self.TAG, self.roles["dm"], 16
        for k in self.chips:
            for q in self.kv_tiles[k]:
                c = self.kv_cluster(k, q)
                buf = self.cx_chip(k).at(c).l1(f"{T}_kx{q}", rows * HIDDEN)
                ld = self.node(k, c, dm, f"kx_ld_q{q}", "__snax_bingo_kernel_idma_1d_copy",
                               SnaxBingoKernelIdma1dCopyArgs(BingoMemSymbol(f"{T}_xsrc", offset=q * rows * HIDDEN),
                                                             buf, rows * HIDDEN), [go[k]])
                self.kv_src[q] = (buf, [ld])

    def stage_kv(self, k):
        """Chip k's 16-token pieces of a fresh prompt's keys (kv_cluster: cluster 0, or with params kv_clusters each
        tile's own), from the x8 tiles the layer's input norm left there (kv_src {tile: (x8 handle, [ready
        tasks])}): stage_kv_on for each cluster that makes keys."""
        if not self.kv_cols and any(q not in self.kv_src for q in self.kv_tiles[k]):
            raise ValueError("fresh: every chip's input-norm tiles come from the layer (params layer.tiles chips)")
        for c in self.kv_cls(k):
            self.stage_kv_on(k, c)

    def stage_kv_on(self, k, c):
        """Chip k cluster c's tiles of keys. Each W_KVA block from the ring serves every tile of the cluster, a GEMM
        a tile into the tile's own kv; the rest runs a tile at a time through one set of buffers, each tile's first
        writers of them behind the previous tile's put (kv_put {tile: the put that leaves its piece in the root's
        L3})."""
        d, T = self.d, self.TAG
        mesh = d.mesh
        cl, chip_tiles = self.chips.index(k) * self.NC + c, self.kv_tiles[k]
        tiles = [q for q in chip_tiles if self.kv_cluster(k, q) == c]
        simd, xdma, dm = self.roles["simd"], self.roles["xdma"], self.roles["dm"]
        l1 = lambda name, nb: self.cx_chip(k).at(c).l1(f"{T}_kv_{name}", nb)
        # the chip's first tile keeps the names of one tile a chip; the chip's tile t >= 1 adds _t{t}
        sfx = lambda t: "" if t == 0 else f"_t{t}"
        ts = [chip_tiles.index(q) for q in tiles]                               # each tile's index on the chip
        kvds = [l1("d" + sfx(t), 16 * D_QK * 2) for t in ts]                    # kv, D layout [16, 576], FP16
        crow, cu = l1("crow", 16 * KV_RANK * 2), l1("cu", 16 * KV_RANK * 2)
        scr = l1("scr", SnaxBingoKernelSimdRmsnormArgs(0, 0, 16, KV_RANK).scratch_bytes())
        ca, vt, va = l1("ca", 16 * KV_RANK * 2), l1("vt", 16 * KV_RANK * 2), l1("va", 16 * KV_RANK * 2)
        xs, t1, t2, pe, pa = (l1(nm, 16 * ROPE * 2) for nm in ("xs", "t1", "t2", "pe", "pa"))
        piece, kvc = l1("piece", KVP), self.kvc[(k, c)]
        kr4 = l1("rope4", 4 * 16 * ROPE * 2) if self.kv_rope_fused else None
        # kv = x8 W_KVA: a GEMM a block of 64 columns and a tile, each block from the ring once a cluster; one
        # dequantisation (kv_cols: the tile's rows gathered from every chip's slice in this chip's L3)
        gs = {n: [] for n in range(len(tiles))}
        for n, q in enumerate(tiles):
            if not self.kv_cols:
                break
            lt = chip_tiles.index(q)
            fl = BingoMemSymbol(f"{T}_kvrflag", offset=lt * len(self.chips) * 64)
            puts = [self.kvc_put[(q, k2)] for k2 in self.chips]
            w = self.node(k, c, dm, f"kvr_wait{sfx(ts[n])}", "__snax_bingo_kernel_wait_flags",
                          SnaxBingoKernelWaitFlagsArgs(fl, len(self.chips), 1),
                          [t for t in puts if self.flag_dep(t, k, c)])
            for t in puts:
                if not self.flag_dep(t, k, c):
                    self.dfg.bingo_add_order_edge(t, w)
            gs[n].append(self.dm_copy(cl, f"kvr_ld{sfx(ts[n])}", BingoMemSymbol(f"{T}_kvraw", offset=lt * self.KVD),
                                      kvds[n], self.KVD, [w]))
        for r in range(0 if self.kv_cols else self.kva_in_ring()):
            ld, si, slab = self.load(cl, r, f"Ld_wkv{r}")
            if c and r and (k, 0, r - 1) in self.ld_node:
                # both clusters read the W_KVA blocks: cluster c's next read follows cluster 0's last, so the reads
                # alternate and their edges each way share one dependency tag instead of taking one a block
                self.edge(self.ld_node[(k, 0, r - 1)], ld)
            for n, q in enumerate(tiles):
                x8, ready = self.kv_src[q]
                gm = self.gmm(cl, f"gkv{r}{sfx(ts[n])}", x8, slab, kvds[n].view(r * 4 * TILE16), 1, HIDDEN // 4, 4,
                              d.k_x, [ld] + list(ready))
                gs[n].append(gm)
            self.ls[cl].release(si, gm)
        R = self.ROOT
        on_root = lambda name, off: BingoMemSymbol(name, offset=off, chip_id=None if k == R else R)
        put = None
        for n, q in enumerate(tiles):
            t = ts[n]
            s, kvd = sfx(t), kvds[n]
            after = [put] if put is not None else []        # the previous tile's put: its last reader of the buffers
            dq = self.simd_mul(cl, f"kv_dq{s}", kvd, kvc, kvd, 16 * D_QK, gs[n])
            # c~: D -> row major, RMSNorm; the key rows' first 512 columns (A layout, quantised) and, transposed, the
            # value columns (the A layout of c8^T [512, 16])
            cv = self.xd(cl, f"kv_crm{s}", convert_args("D", "row_major", 16, KV_RANK, mesh, 2, kvd, crow),
                         [dq] + after)

            def c_head():
                return self.node(k, c, simd, f"kv_rms{s}", "__snax_bingo_kernel_simd_rmsnorm",
                                 SnaxBingoKernelSimdRmsnormArgs(crow, cu, 16, KV_RANK, scratch_addr=scr), [cv])

            def c_tail(rn):
                ka = self.xd(cl, f"kv_ca{s}", convert_args("row_major", "A", 16, KV_RANK, mesh, 2, cu, ca), [rn])
                kq = self.quant(cl, f"kv_c8{s}", ca, piece, 16 * KV_RANK * 2, self.fbits(d.s_c.inv), [ka])
                tr = self.node(k, c, xdma, f"kv_vt{s}", "__snax_bingo_kernel_xdma_transpose_2d",
                               SnaxBingoKernelXdmaTranspose2dArgs(cu, vt, 16, KV_RANK, 2), [rn])
                vv = self.xd(cl, f"kv_va{s}", convert_args("row_major", "A", KV_RANK, 16, mesh, 2, vt, va), [tr])
                vq = self.quant(cl, f"kv_v8{s}", va, piece.view(KB), 16 * KV_RANK * 2, self.fbits(d.s_c.inv), [vv])
                return kq, vq

            # k_pe: RoPE in the D layout (columns 512 .. 575, D blocks 32 .. 35), D -> A, quantised: the key rows' last
            # 64; the tile's own RoPE tables
            src, n_pe, rc = kvd.view((KV_RANK // 16) * TILE16), 16 * ROPE, self.KC_ROPE + t * self.KC_TILE

            def rope():
                if self.kv_rope_fused:
                    # kv_rope_fused: one simd_rope over [x | cos | xswap | sin], 2 KB each, as q_pe's (the same
                    # roundings); the DM core copies x into slot 0, the two swapped halves into slot 2 and the tile's
                    # cos and sin (adjacent in kvc) into slots 1 and 3
                    rb, half = n_pe * 2, (ROPE // 32) * TILE16
                    cx = self.node(k, c, dm, f"kv_xcp{s}", "__snax_bingo_kernel_idma_1d_copy",
                                   SnaxBingoKernelIdma1dCopyArgs(src, kr4, rb), [dq] + after)
                    sw = [self.node(k, c, dm, f"kv_swap{h}{s}", "__snax_bingo_kernel_idma_1d_copy",
                                    SnaxBingoKernelIdma1dCopyArgs(src.view((1 - h) * half), kr4.view(2 * rb + h * half),
                                                                  half), [dq] + after) for h in range(2)]
                    tb = self.node(k, c, dm, f"kv_tab{s}", "__snax_bingo_kernel_idma_2d_copy",
                                   SnaxBingoKernelIdma2dCopyArgs(kvc.view(rc), kr4.view(rb), rb, rb, 2 * rb, 2), after)
                    return self.node(k, c, simd, f"kv_rope{s}", "__snax_bingo_kernel_simd_rope",
                                     SnaxBingoKernelSimdRopeArgs(kr4, pe, ROPE, 16), [dq, cx, tb] + sw)
                if self.rope_perm:
                    # one row tile: the swap is the rotary part's two 1 KB halves exchanged
                    half = (ROPE // 32) * TILE16
                    sw = [self.node(k, c, dm, f"kv_swap{h}{s}", "__snax_bingo_kernel_idma_1d_copy",
                                    SnaxBingoKernelIdma1dCopyArgs(src.view((1 - h) * half), xs.view(h * half), half),
                                    [dq] + after) for h in range(2)]
                else:
                    sw = [self.node(k, c, dm, f"kv_swap{s}", "__snax_bingo_kernel_idma_pairwise_swap",
                                    SnaxBingoKernelIdmaPairwiseSwapArgs(src, xs, n_pe, 2), [dq] + after)]
                m1 = self.simd_mul(cl, f"kv_cos{s}", src, kvc.view(rc), t1, n_pe, [dq] + after)
                m2 = self.simd_mul(cl, f"kv_sin{s}", xs, kvc.view(rc + n_pe * 2), t2, n_pe, sw)
                return self.simd_add(cl, f"kv_rope{s}", t1, t2, pe, n_pe, [m1, m2])

            pa_ = lambda ad: self.xd(cl, f"kv_pa{s}", d_to_a_args(mesh, 1, 4, 16, pe, pa), [ad])
            p8_ = lambda d2p: self.quant(cl, f"kv_p8{s}", pa, piece.view(16 * KV_RANK), n_pe * 2,
                                         self.fbits(d.s_kpe.inv), [d2p])
            if self.kv_rope_first:
                # kv_rope_first: the in-order SIMD core takes RoPE right after the dequantisation (while the xDMA
                # reorders c) and k_pe's quantisation right after the RMSNorm (while the xDMA lays out c)
                d2p = pa_(rope())
                rn = c_head()
                pq = p8_(d2p)
                kq, vq = c_tail(rn)
            else:
                rn = c_head()
                kq, vq = c_tail(rn)
                pq = p8_(pa_(rope()))
            put = self.node(k, c, dm, f"kv_put{s}", "__snax_bingo_kernel_idma_put_flagged",
                            (SnaxBingoKernelIdmaPutFlaggedArgs(piece, BingoMemSymbol(f"{T}_kv", offset=q * KVP), KVP,
                                                               BingoMemSymbol(f"{T}_kvtflag", offset=q * 64), value=1,
                                                               bcast=True, also_here=True) if self.owner_bcast else
                             SnaxBingoKernelIdmaPutFlaggedArgs(piece, on_root(f"{T}_kv", q * KVP), KVP,
                                                               on_root(f"{T}_kvnflag", q * 64), value=1)), [kq, vq, pq])
            self.kv_put[q] = put

    def stage_kv_cols(self, go):
        """params kv_cols: every cluster's W_KVA slice for its tiles of the prompt (cluster c: tiles q = c mod NC), a
        GEMM a tile from x8 rows copied from the chip's staged x, and each tile's slice put into its owner's L3 (kvraw,
        the D layout's granules gr0 .. gr1 - 1 of the tile's [16, 576]) behind the chip's flag there (kvc_put)."""
        d, T, dm = self.d, self.TAG, self.roles["dm"]
        self.kvc_put = {}
        for j, k in enumerate(self.chips):
            gr0, gr1 = self.kvg[k]
            nb = (gr1 - gr0) * TILE16
            for c in range(self.NC):
                cl = j * self.NC + c
                tiles = [q for q in range(self.T // 16) if q % self.NC == c]
                l1 = lambda name, size: self.cx_chip(k).at(c).l1(f"{T}_kc_{name}", size)
                xb, ob = [l1(f"x{b}", 16 * HIDDEN) for b in range(2)], [l1(f"o{b}", nb) for b in range(2)]
                kx = lambda n: self.node(k, c, dm, f"kx_ld_q{tiles[n]}", "__snax_bingo_kernel_idma_1d_copy",
                                         SnaxBingoKernelIdma1dCopyArgs(
                                             BingoMemSymbol(f"{T}_xsrc", offset=tiles[n] * 16 * HIDDEN), xb[n % 2],
                                             16 * HIDDEN), [go[k]] + ([gm[n - 2]] if n >= 2 else []))
                gm, put, ld_x = {}, {}, {0: kx(0)}
                ld, si, slab = self.load(cl, 0, "Ld_wkvc")
                if len(tiles) > 1:
                    ld_x[1] = kx(1)
                for n, q in enumerate(tiles):
                    o = self.kv_owner[q]
                    lt = self.kv_tiles[o].index(q)
                    on = lambda name, off: BingoMemSymbol(name, offset=off, chip_id=None if o == k else o)
                    gm[n] = self.gmm(cl, f"gkvc_q{q}", xb[n % 2], slab, ob[n % 2], 1, HIDDEN // 4, gr1 - gr0, d.k_x,
                                     [ld, ld_x[n]] + ([put[n - 2]] if n >= 2 else []))
                    put[n] = self.node(k, c, dm, f"kvc_put_q{q}", "__snax_bingo_kernel_idma_put_flagged",
                                       SnaxBingoKernelIdmaPutFlaggedArgs(
                                           ob[n % 2], on(f"{T}_kvraw", lt * self.KVD + gr0 * TILE16), nb,
                                           on(f"{T}_kvrflag", (lt * len(self.chips) + j) * 64), value=1), [gm[n]])
                    self.kvc_put[(q, k)] = put[n]
                    if n + 2 < len(tiles):
                        ld_x[n + 2] = kx(n + 2)
                self.ls[cl].release(si, gm[len(tiles) - 1])

    def stage_kv_bcast(self):
        """The root multicasts a fresh prompt's keys a chunk's pieces at a time, from its L3 into every other chip's,
        once their pieces have landed (on the DM core that multicasts x, after x). Their puts are ordering edges, as
        every flagged collective's."""
        R, T, P = self.ROOT, self.TAG, self.C // 16
        c, dm = self.x_put_cluster, self.roles["dm"]
        self.kv_mc = []
        for h in range(self.NCH):
            puts = [self.kv_put[q] for q in range(h * P, (h + 1) * P)]
            w = self.node(R, c, dm, f"kv_nwait_h{h}", "__snax_bingo_kernel_wait_flags",
                          SnaxBingoKernelWaitFlagsArgs(BingoMemSymbol(f"{T}_kvnflag", offset=h * P * 64), P, 1),
                          [t for t in puts if self.flag_dep(t, R, c)] + self.kv_mc[-1:])
            for t in puts:
                if not self.flag_dep(t, R, c):
                    self.dfg.bingo_add_order_edge(t, w)
            src = BingoMemSymbol(f"{T}_kv", offset=h * P * KVP)
            self.kv_mc.append(self.node(R, c, dm, f"kv_bput_h{h}", "__snax_bingo_kernel_idma_put_flagged",
                                        SnaxBingoKernelIdmaPutFlaggedArgs(src, src, P * KVP,
                                                                          BingoMemSymbol(f"{T}_kvflag", offset=h * 64),
                                                                          value=1, bcast=True), [w]))

    def kv_wait(self, cl, i):
        """Cluster cl's wait, before chunk i's sweep, until the keys of chunks 0 .. i have landed in its chip's L3: the
        root's multicasts' flags, or on the root its pieces' own. The puts are ordering edges; the flags prove the
        landing. The pieces gather on getattr(self, "kv_root") where a builder sets one (s3_root), else the root."""
        k, c, R, T = self.chip(cl), self.local(cl), getattr(self, "kv_root", self.ROOT), self.TAG
        if self.owner_bcast:
            n = (i + 1) * (self.C // 16)
            puts = [self.kv_put[q] for q in range(n)]
            args = SnaxBingoKernelWaitFlagsArgs(BingoMemSymbol(f"{T}_kvtflag"), n, 1)
        elif k == R:
            n = (i + 1) * (self.C // 16)
            puts = [self.kv_put[q] for q in range(n)]
            args = SnaxBingoKernelWaitFlagsArgs(BingoMemSymbol(f"{T}_kvnflag"), n, 1)
        else:
            puts = self.kv_mc[:i + 1]
            args = SnaxBingoKernelWaitFlagsArgs(BingoMemSymbol(f"{T}_kvflag"), i + 1, 1)
        w = self.node(k, c, self.roles["dm"], f"kv_wait_i{i}", "__snax_bingo_kernel_wait_flags", args,
                      [t for t in puts if self.flag_dep(t, k, c)])
        for t in puts:
            if not self.flag_dep(t, k, c):
                self.dfg.bingo_add_order_edge(t, w)
        return w

    def ot8_bits(self):
        """fa_sweep's ot8: the quantiser's scale of o~ under ot8_fused, else None."""
        return self.fbits(self.d.s_ot.inv) if self.ot8_fused else None

    def xt_ref(self, h, r0):
        """The sweep's xt for the query tile at row r0 of head h: (bytes, length): o~ FP16 rows, or under ot8_fused
        o~ INT8 in the A layout of one 16-row tile."""
        d = self.d
        if self.ot8_fused:
            return d.layout.to_a(d.head(h)["ot8"][r0:r0 + BR], d.mesh).astype(np.int8).view(np.uint8), D_V * BR
        return d.head(h)["ot16"][r0:r0 + BR].astype(np.float16).view(np.uint8), D_V * BR * 2

    def xh(self, i, c):
        """The half of chunk i's rows that cluster c takes."""
        return c ^ (i & 1) if self.row_swap else c

    def r0(self, i, c):
        """The first row of chunk i that cluster c takes."""
        return i * self.C + self.xh(i, c) * self.ROWS

    def mask_of(self, cl, plan):
        """fa_sweep's mask hook for cluster cl: a masked pair's pattern in the cluster's constants, else None."""
        return lambda j, q: (None if plan.get((j, q)) is None
                             else self.consts[cl].view(self.C_MASK + plan[(j, q)] * self.MASK_B))

    def stage_post_a(self, cl, i, sweep):
        """o~ -> W_UV -> o of chunk i (INT8, A layout): everything before the W_O matmuls.
        Returns (the tasks that read the sweep's xt, the task that made o)."""
        d, b = self.d, self.bufs(cl, i)
        rows, mt = self.ROWS, self.ROWS // 16
        mesh = d.mesh
        # o~ rows, FP16 row-major [16, 512] a query tile -> A layout FP16 -> INT8
        cv = []
        if self.ot8_fused:
            # the sweep's xt[q] is already W_UV's A operand (INT8, one 16-row tile)
            ld, si, slab = self.load(cl, self.chunk_index(i, "tail"), f"Ld_wuv_i{i}")
            gu = [self.gmm(cl, f"guv{q}_i{i}", sweep["xt"][q], slab, b["uv"].view(q * (Q_NOPE // 16) * TILE16), 1,
                           D_V // 4, Q_NOPE // 16, d.k_uv, [ld, sweep["tr"][q]]) for q in range(self.NQ)]
            self.ls[cl].release(si, gu)                 # every query tile's GEMM reads the slab
            cv = list(gu)
        for q in range(self.NQ if not self.ot8_fused else 0):
            cv.append(self.xd(cl, f"o2a{q}_i{i}", convert_args("row_major", "A", 16, D_V, mesh, 2, sweep["xt"][q],
                                                              b["a16o"].view(q * 16 * D_V * 2)),
                              [sweep["tr"][q]] if self.o_tiles else list(sweep["tr"])))
        if self.ot8_fused:
            pass
        elif self.o_tiles:
            ld, si, slab = self.load(cl, self.chunk_index(i, "tail"), f"Ld_wuv_i{i}")
            # a query tile's A rows, its INT8 rows and its D rows are each a contiguous run of the chunk's
            gu = []
            for q in range(self.NQ):
                qo = self.quant(cl, f"ot8{q}_i{i}", b["a16o"].view(q * 16 * D_V * 2), b["ot8"].view(q * 16 * D_V),
                                16 * D_V * 2, self.fbits(d.s_ot.inv), [cv[q]])
                gu.append(self.gmm(cl, f"guv{q}_i{i}", b["ot8"].view(q * 16 * D_V), slab,
                                   b["uv"].view(q * (Q_NOPE // 16) * TILE16), 1, D_V // 4, Q_NOPE // 16, d.k_uv,
                                   [ld, qo]))
            self.ls[cl].release(si, gu)
        else:
            qo = self.quant(cl, f"ot8_i{i}", b["a16o"], b["ot8"], rows * D_V * 2, self.fbits(d.s_ot.inv), cv)
            ld, si, slab = self.load(cl, self.chunk_index(i, "tail"), f"Ld_wuv_i{i}")
            gu = [self.gmm(cl, f"guv_i{i}", b["ot8"], slab, b["uv"], mt, D_V // 4, Q_NOPE // 16, d.k_uv, [ld, qo])] * mt
            self.ls[cl].release(si, gu[0])
        suv = self.consts[cl].view(self.C_UV)
        if self.post_tiles:
            # the scale tile is per column, the same for every 16 rows; o8's tiles are each a contiguous run (the A
            # layout is row-tile major); the four in turn, so the last one's end is all four's
            qq, qs = None, []
            for q in range(self.NQ):
                d2q = self.xd(cl, f"d2au{q}_i{i}", d_to_a_args(mesh, 1, Q_NOPE // 16, Q_NOPE // 4,
                                                             b["uv"].view(q * (Q_NOPE // 16) * TILE16),
                                                             b["a16u"].view(q * 16 * Q_NOPE * 2)), [gu[q]])
                qq = self.mul_quant(cl, f"o8{q}_i{i}", b["a16u"].view(q * 16 * Q_NOPE * 2), suv,
                                    b["o8"].view(q * 16 * Q_NOPE), 16 * Q_NOPE, self.fbits(d.s_o.inv), [d2q] + qs[-1:])
                qs.append(qq)
            self._o8_tiles[(cl, i)] = qs
            return cv, qq
        du = sorted(set(gu), key=gu.index) if self.dq_fused else \
            [self.simd_mul(cl, f"dqu{m}_i{i}", b["uv"].view(m * (Q_NOPE // 16) * TILE16), suv,
                           b["uv"].view(m * (Q_NOPE // 16) * TILE16), 16 * Q_NOPE, [gu[m]]) for m in range(mt)]
        d2u = self.xd(cl, f"d2au_i{i}", d_to_a_args(mesh, mt, Q_NOPE // 16, Q_NOPE // 4, b["uv"], b["a16u"]), du)
        if self.dq_fused:
            qq = self.mul_quant(cl, f"o8_i{i}", b["a16u"], suv, b["o8"], rows * Q_NOPE, self.fbits(d.s_o.inv), [d2u])
        else:
            qq = self.quant(cl, f"o8_i{i}", b["a16u"], b["o8"], rows * Q_NOPE * 2, self.fbits(d.s_o.inv), [d2u])
        return cv, qq

    def stage_post_wo(self, cl, i, qq, r, after):
        """W_O block r of chunk i into the cluster's half-size exchange buffer (y_halves)."""
        d, b = self.d, self.bufs(cl, i)
        k, c = self.chip(cl), self.local(cl)
        ld, si, slab = self.load(cl, self.chunk_index(i, "tail") + 1 + r, f"Ld_wo{r}_i{i}")
        go = self.gmm(cl, f"gwo{r}_i{i}", b["o8"], slab, self.xch[(k, c)], self.ROWS // 16, Q_NOPE // 4, 64,
                      d.k_wo, [ld, qq] + list(after))
        self.ls[cl].release(si, go)
        return go

    def stage_post_b(self, cl, i, qq, xch_after):
        """W_O of chunk i: two blocks of 1024 columns into the two halves of the cluster's partial, in
        its exchange buffer, which `xch_after` (the previous chunk's reduce) has finished reading.
        Returns the tasks after which the partial is there (every GEMM that writes it: wo_split_rows makes several)."""
        d, b = self.d, self.bufs(cl, i)
        k, c = self.chip(cl), self.local(cl)
        mt = self.ROWS // 16
        half = self.YB // 2
        out = []
        # wo_split_rows: each block as one GEMM a pair of 16-row tiles (o8 is row-tile major, the output
        # tile (m, n) sits at (m * 64 + n) tiles)
        parts = [(0, mt)] if not self.wo_split_rows else [(m0, 2) for m0 in range(0, mt, 2)]
        for r in range(2):
            ld, si, slab = self.load(cl, self.chunk_index(i, "tail") + 1 + r, f"Ld_wo{r}_i{i}")
            # xch_per_block: only the previous chunk's chains of this block read the half it writes
            free = self._sym_blk[(k, c)][r] if self.xch_per_block and (k, c) in self._sym_blk else list(xch_after)
            gos = []
            for m0, mm in parts:
                sfx = "" if len(parts) == 1 else f"m{m0}"
                rows_o8 = self._o8_tiles[(cl, i)][m0:m0 + mm] if self.wo_row_pipe else [qq]
                gos.append(self.gmm(cl, f"gwo{r}{sfx}_i{i}", b["o8"].view(m0 * 16 * Q_NOPE), slab,
                                    self.xch[(k, c)].view(r * half + m0 * 64 * TILE16), mm, Q_NOPE // 4, 64,
                                    d.k_wo, [ld] + rows_o8 + list(free)))
            self.ls[cl].release(si, gos)                # every row pair's GEMM reads the slab
            self._wo_blk[(k, c, i, r)] = gos
            out += gos
        return out

    def y_raw(self, r0, rows, fold=None):
        """Rows r0 .. r0 + rows - 1 of the raw y the collector stores: the heads' partials folded in the chain's order
        (fold, default the root's), or (wo_cols) one sum over every head."""
        return self.d.y_cols(r0, rows) if self.wo_cols else self.d.reduce(fold or self.fold_heads, r0, rows)

    def stage_wo_cols(self, i, post, xs):
        """Chunk i's W_O by columns (params wo_cols): every head's o8 of each cluster's rows into the root's L3, the
        root's multicast of them, each cluster's gather into its L1 (A layout [rows, 2048]), its two GEMMs of 64 of its
        chip's 128 columns over the full depth, and its block of y stored into the root's y (the layout of y_halves:
        two halves of 1024 columns, D layout). Returns each cluster's last W_O GEMM."""
        T, R, N_CL, dm = self.TAG, self.ROOT, self.N_CL, self.roles["dm"]
        NH, OB = len(self.chips), self.ROWS * Q_NOPE
        # o_land: the chunk's own area; else the root's landing area of x_i
        oname, base = ((f"{T}_oland", (i % 2) * self.NC * NH * OB) if self.o_land else (f"{T}_xland", i * self.XB))
        on_root = lambda name, off, k: BingoMemSymbol(name, offset=off, chip_id=None if k == R else R)
        rcl = [cl for cl in range(N_CL) if self.chip(cl) == R]
        puts = {}
        for cl in range(N_CL):
            k, c = self.chip(cl), self.local(cl)
            h = self.head[k]
            # into the root's landing area of x_i, after the root's fetches of x_i (o_land: its own area, whose
            # chunk-before-last o8 every chip has gathered before its chunk i computes)
            puts[cl] = self.node(k, c, dm, f"o_put_i{i}", "__snax_bingo_kernel_idma_put_flagged",
                                SnaxBingoKernelIdmaPutFlaggedArgs(
                                    self.bufs(cl, i)["o8"], on_root(oname, base + (c * NH + h) * OB, k), OB,
                                    on_root(f"{T}_oflag", ((i * self.NC + c) * NH + h) * 64, k), value=1),
                                [post[cl][1]] + ([] if self.o_land else [xs[r] for r in rcl]))
        # wo_prefetch: the W_O loads next on each DM core (ring chunks in their order, after W_UV's)
        wo_ld = {cl: [self.load(cl, self.chunk_index(i, "tail") + 1 + t, f"Ld_wo{t}_i{i}") for t in range(2)]
                 for cl in range(N_CL)} if self.wo_prefetch else {}
        xc = self.x_put_cluster
        w = self.node(R, xc, dm, f"o_nwait_i{i}", "__snax_bingo_kernel_wait_flags",
                      SnaxBingoKernelWaitFlagsArgs(BingoMemSymbol(f"{T}_oflag", offset=i * self.NC * NH * 64),
                                                   self.NC * NH, 1), [puts[cl] for cl in rcl])
        for cl in range(N_CL):
            if self.chip(cl) != R:
                self.dfg.bingo_add_order_edge(puts[cl], w)
        run = Q_NOPE // 4 * 64                           # a head's columns in a row tile of the A layout

        def gather(cl, deps):
            c = self.local(cl)
            return self.node(self.chip(cl), c, dm, f"o_get_i{i}", "__snax_bingo_kernel_idma_2d_copy",
                             SnaxBingoKernelIdma2dCopyArgs(BingoMemSymbol(oname, offset=base + c * NH * OB),
                                                           self.bufs(cl, i)["xrowso"], run, run, HIDDEN // 4 * 64,
                                                           self.ROWS // 16, outer=NH, src_outer=OB, dst_outer=run),
                             list(deps))
        gets = {cl: gather(cl, [w]) for cl in rcl}
        # the multicast after the root's own gathers (the stores of y that land over this area follow it)
        mc = self.node(R, xc, dm, f"o_bput_i{i}", "__snax_bingo_kernel_idma_put_flagged",
                       SnaxBingoKernelIdmaPutFlaggedArgs(BingoMemSymbol(oname, offset=base),
                                                         BingoMemSymbol(oname, offset=base), self.NC * NH * OB,
                                                         BingoMemSymbol(f"{T}_omflag", offset=i * 64), value=1, bcast=True),
                       [w] + [gets[cl] for cl in rcl])
        for cl in range(N_CL):
            if self.chip(cl) == R:
                continue
            ow = self.node(self.chip(cl), self.local(cl), dm, f"o_wait_i{i}", "__snax_bingo_kernel_wait_flags",
                           SnaxBingoKernelWaitFlagsArgs(BingoMemSymbol(f"{T}_omflag", offset=i * 64), 1, 1), [])
            self.dfg.bingo_add_order_edge(mc, ow)
            gets[cl] = gather(cl, [ow])
        last = {}
        for cl in range(N_CL):
            k, c = self.chip(cl), self.local(cl)
            q, yb = self.head[k], self.bufs(cl, i)["xrowsy"]
            r, nb0 = q // 8, (q % 8) * 8                 # the half and the first column tile of the chip's columns
            for t in range(2):
                ld, si, slab = (wo_ld[cl][t] if wo_ld else
                                self.load(cl, self.chunk_index(i, "tail") + 1 + t, f"Ld_wo{t}_i{i}"))
                gm = self.gmm(cl, f"gwo{t}_i{i}", self.bufs(cl, i)["xrowso"], slab,
                              yb.view(t * 4 * (self.ROWS // 16) * TILE16),
                              self.ROWS // 16, HIDDEN // 4, 4, self.d.k_wo, [ld, gets[cl]])
                self.ls[cl].release(si, gm)
                last[cl] = gm
                # its 4 column tiles of each row tile into the half's D layout (64 tiles a row tile)
                dst = self.y_sym(i, c, r * (self.YB // 2) + (nb0 + 4 * t) * TILE16, chip_id=None if k == R else R)
                st = self.node(k, c, dm, f"y_st{t}_i{i}", "__snax_bingo_kernel_idma_2d_copy",
                               SnaxBingoKernelIdma2dCopyArgs(yb.view(t * 4 * (self.ROWS // 16) * TILE16), dst,
                                                             4 * TILE16, 4 * TILE16, 64 * TILE16, self.ROWS // 16),
                               [gm] + ([mc] if k == R else []))
            # the cluster's flag after its stores (one in-order DM core): every block of y it made has landed
            self._y_last.append(self.node(k, c, dm, f"y_flag_i{i}", "__snax_bingo_kernel_idma_put_flagged",
                                          SnaxBingoKernelIdmaPutFlaggedArgs(
                                              yb, on_root(f"{T}_ydum", cl * 64, k), 64,
                                              on_root(f"{T}_yflag", (i * N_CL + cl) * 64, k), value=1), [st]))
        return last

    # ---- the reduce of one chunk ------------------------------------------------------------------
    def stage_reduce(self, i, parts, war, gate=None, half=None, src_off=0, clusters=None):
        """Cluster index c's partials of chunk i into the root's cluster c: ChainGather (chain) or the
        root's pulls and adds (root). parts {(k, c): [tasks after which the partial is there]}, war
        {c: [tasks the destination waits for]}, gate {c: [tasks the reduce starts behind]} (the root's
        sweep read out, whose L1 the destination takes). Returns {c: the last task of the reduce}. With y_roots the
        partials fold into that block's collector instead, along the cycle from it."""
        R, xdma, dm, simd = self.y_root(i, half), self.roles["xdma"], self.roles["dm"], self.roles["simd"]
        if R != self.ROOT and self.reduce_impl != "chain":
            raise ValueError("y_roots: reduce chain only")
        others = self.y_order(R)[1:]
        out = {}
        before = {}
        cs = list(clusters) if clusters is not None else list(range(self.NC))
        for c in cs:
            before[c] = [t for k in self.chips for t in parts[(k, c)]] + list((gate or {}).get(c, []))
        if self.sync == "barrier":
            bar = self.barrier({k: [t for c in range(self.NC) for t in parts[(k, c)]] for k in self.chips})
            for c in range(self.NC):
                before[c] = [bar[R]] + list((gate or {}).get(c, []))
            self.bar_reduce = bar
        for c in cs:
            size, doff, tag = (self.YB, 0, "") if half is None else (self.YB // 2, half * self.YB // 2, f"_h{half}")
            if self.y_chain_l3 and half is not None:
                # the block's y straight into the collector's L3, where its store would have put it
                dstb = None
                dst = self.y_sym_local(i, c, doff)
            else:
                dstb = self.dst(i, c, half)
                if self.y_store_halves and half is not None:
                    doff = 0
                dst = f"((uint64_t)({dstb.get_c_var_name()}) + {doff})"
            if self.reduce_impl == "chain":
                ga = SnaxBingoKernelXdmaChainGatherArgs.fold_into(
                    cm.remote, R, c, others, src=self.xch_addr(c, src_off), dst=dst, size=size,
                    junction=self.jct["Elementwise"], jct_csr0=EW_F16_ADD, holds=[dstb] if dstb is not None else [])
                serial = [] if self.chain_concurrent or self._last_chain is None else [self._last_chain]
                if self.last_pair_chains and i == self.NCH - 1 and half is not None and len(cs) == self.NC:
                    if self._pair_prev is None:
                        self._pair_prev = [self._last_chain] if self._last_chain is not None else []
                    serial = list(self._pair_prev) + [self.prefetch[k] for k in self.chips]
                if self.chain_gate_prefetch:
                    serial = list(serial) + [self.prefetch[k] for k in self.chips]
                out[c] = self.node(R, c, xdma, f"y_chain_c{c}_i{i}{tag}", "__snax_bingo_kernel_xdma_chain_gather", ga,
                                   before[c] + list(war.get(c, [])) + serial)
                self._last_chain = out[c]
                self._chains.append(out[c])
                continue
            ns = self.YB // self.SL
            pulls, adds = [], []
            n = 0
            for ik, k in enumerate(others):
                for s in range(ns):
                    slot = self.slot(i, c).view((n % 2) * self.SL)
                    cp = self.node(R, c, dm, f"y_pull_c{c}_i{i}_k{k:02x}_s{s}", "__snax_bingo_kernel_idma_1d_copy",
                                   SnaxBingoKernelIdma1dCopyArgs(
                                       cm.remote(k, f"({self.xch_addr(c, 0)} + {s * self.SL})"), slot, self.SL),
                                   before[c] + list(war.get(c, [])) + pulls[-1:] + ([adds[n - 2]] if n >= 2 else []))
                    pulls.append(cp)
                    a_in = self.xch[(R, c)].view(s * self.SL) if ik == 0 else dstb.view(s * self.SL)
                    adds.append(self.node(R, c, simd, f"y_add_c{c}_i{i}_k{k:02x}_s{s}",
                                          "__snax_bingo_kernel_simd_stream_elementwise",
                                          SnaxBingoKernelSimdAddF16Args(a_in, slot, dstb.view(s * self.SL),
                                                                        self.SL // 64, 32),
                                          [cp] + before[c] + adds[-1:]))
                    n += 1
            out[c] = adds[-1]
        if self.last_pair_chains and i == self.NCH - 1 and half is not None and len(cs) == self.NC:
            self._pair_prev = [out[c] for c in cs]
        return out

    def stage_store(self, i, red, half=None, clusters=None):
        """The collector's cluster c stores chunk i's rows of y (y_store_halves: one half of them) to its L3."""
        R, dm = self.y_root(i, half), self.roles["dm"]
        st = {}
        for c in (list(clusters) if clusters is not None else range(self.NC)):
            nb, off, tag = (self.YB, 0, "") if half is None else (self.YB // 2, half * self.YB // 2, f"_h{half}")
            st[c] = self.node(R, c, dm, f"y_store_c{c}_i{i}{tag}", "__snax_bingo_kernel_idma_1d_copy",
                              SnaxBingoKernelIdma1dCopyArgs(
                                  self.dst(i, c, half), self.y_sym(i, c, off),
                                  nb), list(red[c]) if isinstance(red[c], (list, tuple)) else [red[c]])
        return st

    # ---- the prefill ------------------------------------------------------------------------------
    def build(self):
        d, R = self.d, self.ROOT
        dm = self.roles["dm"]
        if self.x_hbm:
            raise ValueError("params x_hbm: the stage workloads' builder (fa_stages_mc.py) runs it")
        self.last_ld = {}
        # a slab is never loaded again before the reader of its last load is known
        for ls in self.ls:
            ls.strict = True
        # the dequantisation constants of every cluster into its L1 before the start
        # (one chain a chip, ending on cluster 0: the start stamp sits on the root's cluster 0 DM core
        # behind that core's previous task, so a chain ending there adds no second edge into the stamp)
        pre = {k: [] for k in self.chips}
        for k in self.chips:
            for c in reversed(range(self.NC)):
                cl = self.chips.index(k) * self.NC + c
                pre[k] = [self.dm_copy(cl, f"const_ld", self.h_c[k].address, self.consts[cl],
                                       self.C_BYTES, pre[k])]
                if self.fresh and (k, c) in self.kvc:
                    pre[k] = [self.dm_copy(cl, f"kvc_ld", self.h_kc[k].address, self.kvc[(k, c)], self.KC_BYTES,
                                           pre[k])]
        if self.fresh and (self.kv_slabs not in ("split", "split4", "split_kb") or self.stop != "full"):
            raise ValueError("fresh: kv_slabs split, split4 or split_kb (FreshTileSource's slabs), stop full")
        go = self.setup(extra=pre)
        if self.p.get("relay_per_gather", True):
            self.dfg.relay_one_consumer_cores = [self.roles["xdma"]]
        # chunk-major: every cluster's chunk i, then the barriers and the reduce in the order the arm needs
        puts = []
        swp_res, post_res, parts_ready = {}, {}, {}
        xs, self._y_last = {}, []
        red, stores = {}, {}
        xch_free = {}
        pend = None
        if self.fresh:
            if self.kv_cols:
                self.stage_kv_cols(go)
            elif not self.x_device:
                self.fresh_tiles(go)
            for k in self.chips:
                self.stage_kv(k)
            if not self.owner_bcast:
                self.stage_kv_bcast()
        for i in range(self.NCH):
            if i:
                self.chunk_start(i)
            if self.bcast_impl == "bcast_l1" and not self.owner_bcast and not self.x_resident:
                puts.append(self.stage_x_put(i, [go[R]] + puts[-1:]))
            sweeps = {}
            for cl in range(self.N_CL):
                k, c = self.chip(cl), self.local(cl)
                # the rows land once the previous chunk's sweep has been read out (its L1 is theirs; wo_cols: once its
                # W_O GEMMs are done with the gathered o8, whose bytes the rows then take)
                x = self.stage_x(cl, i, puts[i] if puts else None,
                                 [go[k]] + (list(post_res[cl]["xt_rd"]) if cl in post_res else []))
                xs[cl] = x
                if self.stop == "x":
                    # with x_ring, the chunk's W_q and W_UK loads as in q8, each slab freed by its own load
                    rd = [x]
                    if self.x_ring:
                        for r in range(self.PER_CHUNK_HEAD):
                            ld, si, _ = self.load(cl, self.chunk_index(i, "head") + r, f"Ld_h{r}_i{i}")
                            self.ls[cl].release(si, ld)
                            rd.append(ld)
                    parts_ready[(k, c, i)] = rd
                    post_res[cl] = dict(xt_rd=[x])
                    continue
                q8 = self.stage_proj(cl, i, x)
                if self.reduce_early and i > 0 and k == R:
                    # chunk i-1's chains of this root cluster, behind its q side of chunk i in its xDMA core's program
                    w = {c: [stores[(i - 2, c)]] if (i - 2, c) in stores else []}
                    self._block_chains(i - 1, [c], w, None, red, stores, xch_free)
                if self.stop == "q8":
                    # every query tile's Q8 (q_tiles: each tile has its own ready tasks)
                    parts_ready[(k, c, i)] = list(dict.fromkeys(t for q in sorted(q8) for t in q8[q][1]))
                    post_res[cl] = dict(xt_rd=parts_ready[(k, c, i)])
                    continue
                early = 2 if len(self.ls[cl].slabs) >= 4 else 1
                if self.fresh:
                    # the key tiles up to the chunk's end, the causal masks on the scores
                    nt, plan = d.tile_plan(self.r0(i, c))
                    sw = fa_sweep(self.at(k, c), dataclasses.replace(self.sweep_cfg, nt=nt), self.bufs(cl, i)["swp"],
                                  FreshTileSource(self, cl, i, nt), q8, self.roles, name=f"s{i}_", early=early,
                                  mask=self.mask_of(cl, plan), mask_c=self.mask_c, ot8=self.ot8_bits(),
                                  sm_batch=self.sm_batch, prologue=self.sm_prologue, geom_core=self.sm_geom)
                else:
                    sw = fa_sweep(self.at(k, c), self.sweep_cfg, self.bufs(cl, i)["swp"], TileSource(self, cl, i), q8,
                                  self.roles, name=f"s{i}_", early=early, sm_batch=self.sm_batch, prologue=self.sm_prologue, geom_core=self.sm_geom)
                swp_res[cl] = sw
                if self.stop == "sweep":
                    parts_ready[(k, c, i)] = list(sw["tr"])
                    post_res[cl] = dict(xt_rd=parts_ready[(k, c, i)])
                    continue
                sweeps[cl] = sw
            if self.stop in ("x", "q8", "sweep"):
                continue
            # everything of chunk i before its W_O, then chunk i-1's reduce under it (the exchange buffer
            # is written again by W_O, which waits for that reduce); the reduce starts once the root's
            # sweep is read out, so its landing buffer can take the sweep's L1
            post = {cl: self.stage_post_a(cl, i, sweeps[cl]) for cl in range(self.N_CL)}
            if self.wo_cols:
                last = self.stage_wo_cols(i, post, xs)
                for cl in range(self.N_CL):
                    post_res[cl] = dict(xt_rd=list(post[cl][0]) + [last[cl]])
                continue
            if self.y_halves:
                # W_O block 0 everywhere, its gather, block 1 behind that gather, its gather, the store
                gate = {c: post[self.chips.index(R) * self.NC + c][0] for c in range(self.NC)}
                if self.chunk_pipeline:
                    # chunk i-1's block 1, held back until chunk i's front is made, then chunk i's block 0
                    if pend is not None:
                        self._halves(*pend, red, stores, xch_free, blocks=(1,))
                    pend = (i, post, gate, {}, parts_ready, [])
                    self._halves(*pend, red, stores, xch_free, blocks=(0,))
                    for cl in range(self.N_CL):
                        post_res[cl] = dict(xt_rd=post[cl][0])
                    continue
                rh, gos = [], {}
                for r in range(2):
                    for cl in range(self.N_CL):
                        k, c = self.chip(cl), self.local(cl)
                        after = xch_free.get((k, c), []) if r == 0 else [rh[0][c]]
                        gos[cl] = self.stage_post_wo(cl, i, post[cl][1], r, after)
                        post_res[cl] = dict(xt_rd=post[cl][0])
                        parts_ready[(k, c, i)] = [gos[cl]]
                    if self.stop != "full":
                        rh.append({c: gos[self.chips.index(R) * self.NC + c] for c in range(self.NC)})
                        continue
                    parts = {(k, c): [gos[self.chips.index(k) * self.NC + c]] for k in self.chips for c in range(self.NC)}
                    war = ({c: [sth[c]] for c in range(self.NC)} if self.y_store_halves and r == 1 else {})
                    rh.append(self.stage_reduce(i, parts, war, gate if r == 0 else None, half=r))
                    if self.y_store_halves and self.stop == "full":
                        sth = self.stage_store(i, rh[r], half=r)
                if self.stop == "full":
                    st = (sth if self.y_store_halves else
                          self.stage_store(i, {c: [rh[0][c], rh[1][c]] for c in range(self.NC)}))
                    for c in range(self.NC):
                        red[(i, c)] = rh[1][c]
                        stores[(i, c)] = st[c]
                        for k in self.chips:
                            xch_free[(k, c)] = [rh[1][c]]
                continue
            if self.stop == "full" and self.sync != "barrier" and i > 0 and not self.reduce_early:
                gate = {c: post[self.chips.index(R) * self.NC + c][0] for c in range(self.NC)}
                self._reduce_chunk(i - 1, parts_ready, red, stores, xch_free, gate)
            for cl in range(self.N_CL):
                k, c = self.chip(cl), self.local(cl)
                post_res[cl] = dict(xt_rd=post[cl][0])
                parts_ready[(k, c, i)] = self.stage_post_b(cl, i, post[cl][1], xch_free.get((k, c), []))
            if self.stop == "full" and self.sync == "barrier":
                self._reduce_chunk(i, parts_ready, red, stores, xch_free)
        if self.stop == "full" and self.sync != "barrier" and not self.y_halves and not self.wo_cols:
            self._reduce_chunk(self.NCH - 1, parts_ready, red, stores, xch_free)
        if pend is not None:
            self._halves(*pend, red, stores, xch_free, blocks=(1,))
        if self.wo_cols:
            # every cluster's blocks of every chunk have landed: one wait on the root, on their flags
            yw = self.node(R, self.x_put_cluster, dm, "y_wait", "__snax_bingo_kernel_wait_flags",
                           SnaxBingoKernelWaitFlagsArgs(BingoMemSymbol(f"{self.TAG}_yflag"), self.NCH * self.N_CL, 1),
                           [t for t in self._y_last if t.assigned_chiplet_id == R])
            for t in self._y_last:
                if t.assigned_chiplet_id != R:
                    self.dfg.bingo_add_order_edge(t, yw)
            ends = [yw]
        elif stores:
            ends = [stores[(self.NCH - 1, c)] for c in range(self.NC)]
        else:
            # a stop before the reduce: one probe a chip behind that chip's last chunk, the root's last
            # (the end stamp sits behind the root's previous task on its core)
            last = {k: [t for c in range(self.NC) for t in parts_ready[(k, c, self.NCH - 1)]] for k in self.chips}
            ends = [self.probe(k, f"fin{k:02x}", last[k]) for k in self.chips if k != R]
            ends.append(self.probe(R, f"fin{R:02x}", last[R]))
        self.end_window(ends)
        self._checks(swp_res, post_res)

    def _halves(self, i, post, gate, gos, parts_ready, rh, red, stores, xch_free, blocks):
        """chunk_pipeline: W_O blocks `blocks` of chunk i (y_halves), each everywhere, then its reduce; block 1 behind
        block 0's reduce; the store after block 1's. rh carries block 0's reduce to block 1."""
        R = self.ROOT
        for r in blocks:
            for cl in range(self.N_CL):
                k, c = self.chip(cl), self.local(cl)
                after = xch_free.get((k, c), []) if r == 0 else [rh[0][c]]
                gos[cl] = self.stage_post_wo(cl, i, post[cl][1], r, after)
                parts_ready[(k, c, i)] = [gos[cl]]
            parts = {(k, c): [gos[self.chips.index(k) * self.NC + c]] for k in self.chips for c in range(self.NC)}
            rh.append(self.stage_reduce(i, parts, {}, gate if r == 0 else None, half=r))
            if r == 1:
                st = self.stage_store(i, {c: [rh[0][c], rh[1][c]] for c in range(self.NC)})
                for c in range(self.NC):
                    red[(i, c)] = rh[1][c]
                    stores[(i, c)] = st[c]
                    for k in self.chips:
                        xch_free[(k, c)] = [rh[1][c]]

    def _block_chains(self, i, cs, war, gate, red, stores, xch_free):
        """y_block_chains: root clusters cs's chains of chunk i, one a W_O block, and their stores (y_store_halves: each
        block's sum in its own half-size buffer, stored right after its chain; else one store of the whole). y_dst_shared:
        block 1's chain waits for block 0's store (one buffer)."""
        cs = list(cs)
        half = self.YB // 2
        rh, st, st0 = [], {}, {}
        shared = self.dst_shared(i)
        for r in range(2):
            parts = {(k, c): list(self._wo_blk[(k, c, i, r)]) for k in self.chips for c in cs}
            w_r = war if r == 0 else ({c: [st0[c]] for c in cs} if shared else {})
            rh.append(self.stage_reduce(i, parts, w_r, gate if r == 0 else None, half=r, src_off=r * half, clusters=cs))
            if self.y_chain_l3:
                # the chain wrote the block into L3 itself: it stands for the store
                st = {c: rh[r][c] for c in cs}
            elif self.y_store_halves:
                st = self.stage_store(i, rh[r], half=r, clusters=cs)
                if r == 0:
                    h0 = st
                else:
                    # the second half's store after the first's, so it stands for the chunk's stores
                    for c in cs:
                        self.edge(h0[c], st[c])
                if r == 0 and shared:
                    st0 = st
        if not self.y_store_halves:
            st = self.stage_store(i, {c: [rh[0][c], rh[1][c]] for c in cs}, clusters=cs)
        for c in cs:
            red[(i, c)] = rh[1][c]
            stores[(i, c)] = st[c]
            for k in self.chips:
                xch_free[(k, c)] = [rh[0][c], rh[1][c]]
                self._sym_blk[(k, c)] = {0: [rh[0][c]], 1: [rh[1][c]]}

    def chunk_start(self, i):
        """Where the tasks of chunk i (i >= 1) begin, in the order they are made: nothing here; a whole layer
        starts an args phase there."""

    def _reduce_chunk(self, i, parts_ready, red, stores, xch_free, gate=None):
        war = {c: [stores[(i - 1, c)]] if (i - 1, c) in stores else [] for c in range(self.NC)}
        if self.y_block_chains:
            # one gather per W_O block, each right behind that block on every chip: block 0's runs under
            # block 1's GEMM (the exchange buffer holds both blocks)
            self._block_chains(i, range(self.NC), war, gate, red, stores, xch_free)
            return
        parts = {(k, c): parts_ready[(k, c, i)] for k in self.chips for c in range(self.NC)}
        r = self.stage_reduce(i, parts, war, gate)
        st = self.stage_store(i, r)
        for c in range(self.NC):
            red[(i, c)] = r[c]
            stores[(i, c)] = st[c]
            for k in self.chips:
                xch_free[(k, c)] = [r[c]]
        if self.sync == "barrier":
            # the next chunk starts after the root's reduce and store: a second barrier, behind them
            self.barrier({k: [st[c] for c in range(self.NC)] if k == self.ROOT else [self.bar_reduce[k]]
                          for k in self.chips})

    # ---- checks -----------------------------------------------------------------------------------
    def _checks(self, swp_res, post_res):
        lay, T, R, d = self.lay, self.TAG, self.ROOT, self.d
        checks = []
        n = 2048
        if self.fresh:
            # the keys as they landed: in the root's L3 (kv_checks_at, set by a layer whose L3 reuses the root's: that
            # chip's) the first piece's key and value heads and the last's tails; on the chip in the middle of the fold
            # order (the multicast's landing) the last piece's value tail
            last, far = self.T // 16 - 1, self.order[len(self.order) // 2]
            kc = getattr(self, "kv_checks_at", R)
            # params x_checks_first (a debug list of x tiles): those tiles' x as it sits in L3 at the end (the head of
            # the first, the tail of the last), on the root, the chip in the middle of the fold order and the last
            # chip, ahead of every other check
            for t in self.p.get("x_checks_first", []):
                off = int(t) * self.XT + (self.XT - n if int(t) == self.T // 16 - 1 else 0)
                for k in dict.fromkeys((R, far, self.order[-1])):
                    checks.append((f"x_k{k:02x}_t{t}", BingoMemSymbol(f"{T}_xsrc", offset=off,
                                                                    chip_id=None if k == R else k),
                                   self.xsrc[off:off + n].view(np.uint8), n))
            for nm, q, lo, kk in (("k0a", 0, 0, kc), ("v0a", 0, KB, kc), ("kLb", last, KB - n, kc),
                                  ("vLb", last, KVP - n, kc), ("vLf", last, KVP - n, far)):
                img = d.kv_piece(16 * q).view(np.uint8)
                checks.append((f"kv_{nm}", BingoMemSymbol(f"{T}_kv", offset=q * KVP + lo, chip_id=kk), img[lo:lo + n], n))
        if self.stop == "full":
            for i in range(self.NCH):
                for c in range(self.NC):
                    half = HIDDEN // 2
                    # each W_O block folded in its collector's chain order (y_roots), read from that collector's L3
                    yh = [self.y_raw(self.r0(i, c), self.ROWS, self.y_fold_heads(self.y_root(i, h))
                                     if self.y_roots else None) for h in range(2)]
                    img = np.concatenate([lay.to_d(yh[0][:, :half], d.mesh), lay.to_d(yh[1][:, half:], d.mesh)]
                                         ).astype(np.float16).view(np.uint8)
                    # y_halves or y_windows 4: the head and the tail of each half (each half is a W_O block)
                    wins = (((f"y_i{i}c{c}_a", 0), (f"y_i{i}c{c}_m", self.YB // 2 - n // 2), (f"y_i{i}c{c}_b", self.YB - n))
                            if not (self.y_halves or int(self.p.get("y_windows", 3)) == 4) else
                            ((f"y_i{i}c{c}_h0a", 0), (f"y_i{i}c{c}_h0b", self.YB // 2 - n),
                             (f"y_i{i}c{c}_h1a", self.YB // 2), (f"y_i{i}c{c}_h1b", self.YB - n)))
                    for nm, off in wins:
                        yr = self.y_root(i, 0 if off < self.YB // 2 else 1) if self.y_roots else R
                        checks.append((nm, self.y_sym(i, c, off, chip_id=yr), img[off:off + n], n))
        if self.stop == "x":
            # the last chunk's rows in the root's cluster 0 and in cluster 1 of the chip in the middle of the fold order
            i, half = self.NCH - 1, self.XB // self.NC
            for k, c in ((R, 0), (self.order[len(self.order) // 2], 1)):
                cl = self.chips.index(k) * self.NC + c
                base = i * self.XB + self.xh(i, c) * half
                # three 2 KiB windows (head, middle, tail): the host compares from uncached L3, byte by byte
                for nm, off in (("a", 0), ("m", half // 2 - 1024), ("b", half - 2048)):
                    name = f"x_k{k:02x}c{c}_{nm}"
                    checks.append((name, self.stash(name, k, c, self.bufs(cl, i)["xrows"].view(off), 2048, []),
                                   self.xsrc[base + off:base + off + 2048].view(np.uint8), 2048))
        if self.checks_kind == "stages" or self.stop in ("q8", "sweep"):
            # the first and the last query tile of both clusters of the chip in the middle of the fold order,
            # before the y checks
            stage = []
            far = self.order[len(self.order) // 2]
            h = self.head[far]
            i = self.NCH - 1
            for c in range(self.NC):
                cl = self.chips.index(far) * self.NC + c
                for q in sorted({0, self.NQ - 1}):
                    r0 = self.r0(i, c) + q * BR
                    tag = f"c{c}" if q == 0 else f"c{c}q{q}"
                    q8 = d.q8_a(h, r0)
                    stage.append((f"q8_{tag}", self.stash(f"q8_{tag}", far, c, self.bufs(cl, i)["q8"][q], D_QK * BR,
                                                          []),
                                  q8.view(np.uint8), D_QK * BR))
                    if (self.stop in ("sweep",) or self.checks_kind == "stages") and cl in swp_res:
                        ot, nb = self.xt_ref(h, r0)
                        stage.append((f"xt_{tag}", self.stash(f"xt_{tag}", far, c, swp_res[cl]["xt"][q], nb,
                                                              []), ot, nb))
            checks = stage + checks
        self.run_checks(checks)

    # ---- emission -----------------------------------------------------------------------------------
    def finish(self, app_name, x_prefixes=("x_wait_", "x_pull_", "x_copy_")):
        if self.fresh and not self.x_device:
            # alone, a fresh prompt's W_KVA loads run behind the cluster's tile loads (fresh_tiles): its first part of x
            x_prefixes = tuple(x_prefixes) + ("kx_ld_",)
        super().finish(app_name, x_prefixes)


def _app_name(S):
    return (f"DeepSeek-V2-Lite layer 1 MLA attention prefill, {S.T} tokens, {len(S.chips)} chips x {S.NC} "
            f"clusters, bcast={S.bcast_impl} reduce={S.reduce_impl} sync={S.sync}")


def run():
    run_driver("fa prefill workload generator", FaBuild, _app_name)
