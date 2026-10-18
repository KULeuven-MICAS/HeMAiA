# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""The shared driver of the fa PREFILL workloads (workloads/fa/dsv2_l1_prefill*/<variant>/):
DeepSeek-V2-Lite's layer-1 MLA attention for a prefill of `tokens` tokens over the REAL 8K-token
latent cache, head parallel over the 16 + 4 chiplet platform. The platform, the anchors, the
barriers and the synchronisation arm are chiplet_mc.py's; the data and the golden are
fa_prefill_data.py's; the sweep over the cache is libs/blocks/fa_sweep.py's.

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
    y_h        o . W_O[head rows, :] in two GEMMs of 1024 columns, FP16, into the symmetric buffer
    reduce     the 16 heads' y_h of cluster index c added into the root's cluster c (chain: one
               ChainGather in the fold order; root: the root pulls and adds), stored in the root's L3

THE RING. A chip's two clusters read the SAME stream of chunks: one ring a chip, pushed once by the
memory chip's engine under the chip's prefetcher. Per token chunk it is W_q (3), W_UK, then the cache
tile by tile K0 V0 K1 V1 ..., then W_UV and W_O (2). Cluster 1 loads each chunk without releasing its
slot; cluster 0's load releases it and follows cluster 1's. The ring's chunks are not all one size; a
cluster takes them into `w_buffers` L1 slabs in ring order, in turn, or (kv_slabs split, three slabs) the
K tiles always into one slab and the V tiles into the other two (fa_sweep's tile source).

THE MATMULS are queued (blocks/gemm_async.py, params gemm_async): every GEMM of a cluster is a start
and a wait task, and the cores' program order lets the next start run before the previous wait.

THE MECHANISMS (params): bcast_l1 | pull (x), chain | root (the partials), edge | barrier (every
cross-chip dependency: a BINGO edge, or the chip barrier before and after each chunk's reduce, a
full fence). The ladder base -> bcast -> bcast_reduce -> all adds one each.

PARAMETERS (params.hjson): the platform (num_chiplets, num_clusters), tokens, chunk, keys, bc,
w_chunk_bytes and w_buffers (the ring chunk and the L1 slabs), weight_ring, weight_crest (false),
bcast, reduce, sync, barrier_mode, barrier_fence, gemm_async, static_l1, l1_capacity, d2d_ddr, and
n_calib (calibration tokens), kv_cache and weights (paths under the repository), root_slice (the
root's pull slice), compact_id_maps, gid_lists, relay_per_gather (each dependency-tag relay of
an xDMA task serves that one task; default true). The orders and buffers (each explained where it is read):
x_puts_first, prefetch_after_x, l3_overlay, kv_slabs, y_halves, y_block_chains, chain_concurrent,
wo_split_rows. The checks: checks (y | stages), y_windows (3 | 4). For bisecting a failure: stop
(full | q8 | sweep | post | x), x_root, x_put_cluster, x_ring.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "common"))     # workloads/common
from driver import repo_root, run_driver  # noqa: E402
REPO = repo_root(__file__)
sys.path.insert(0, f"{REPO}/target/sw/host/runtime/libbingo/mini_compiler")
sys.path.insert(0, os.path.join(HERE, "..", "..", "ffn", "common"))
sys.path.insert(0, HERE)

import _bingo_paths  # noqa: F401,E402
from bingo_kernel_args import (  # noqa: E402
    SnaxBingoKernelGemmFullArgs, SnaxBingoKernelIdma1dCopyArgs, SnaxBingoKernelIdma2dCopyArgs, SnaxBingoKernelIdmaFetchFlaggedArgs,
    SnaxBingoKernelIdmaPairwiseSwapArgs, SnaxBingoKernelIdmaPutFlaggedArgs, SnaxBingoKernelIdmaRingLoadArgs,
    SnaxBingoKernelSimdAddF16Args, SnaxBingoKernelSimdFp16ToInt8Args, SnaxBingoKernelSimdMulF16Args,
    SnaxBingoKernelXdmaChainGatherArgs, simd_beats, simd_fp16_lanes)
from bingo_mem_handle import BingoMemSymbol  # noqa: E402
from libs.blocks import FaSweepCfg, GemmQueue, WeightRings, fa_sweep, fa_sweep_alloc  # noqa: E402
from libs.comm.nest import convert_args, d_to_a_args  # noqa: E402

import chiplet_mc as cm  # noqa: E402
import fa_prefill_data  # noqa: E402

BLK = 16                      # a GEMM column block (meshCol)
BR = 16                       # query rows a tile: the SIMD's FP16 lanes
D_QK, D_V, KV_RANK, ROPE, Q_NOPE, Q_HEAD, HIDDEN = 576, 512, 512, 64, 128, 192, 2048
TILE16 = 16 * 16 * 2          # one D-layout FP16 block
EW_F16_ADD = cm.EW_F16_ADD
STRIPE = [0, 1, 3, 4]         # the compute columns of a row


def check(p):
    """The mapping's arithmetic; raises ValueError where the parameters cannot be laid out."""
    tokens, chunk, keys, bc = (int(p[k]) for k in ("tokens", "chunk", "keys", "bc"))
    rows = chunk // 2
    if tokens % chunk or chunk % 2 or rows % BR:
        raise ValueError(f"tokens {tokens} / chunk {chunk}: a cluster takes chunk / 2 rows, a whole number of "
                         f"{BR}-row query tiles")
    if keys % bc or bc % 16 or keys // bc < 4:
        raise ValueError(f"keys {keys}: not whole tiles of bc {bc} (a multiple of 16), at least four")
    if int(p["w_chunk_bytes"]) < max(bc * D_QK, D_V * bc, HIDDEN * 64):
        raise ValueError(f"w_chunk_bytes {p['w_chunk_bytes']}: a K tile is {bc * D_QK} B, a W_q block "
                         f"{HIDDEN * 64} B")
    return dict(chunks=tokens // chunk, query_tiles=rows // BR, tiles=keys // bc,
                qk_passes=(bc // 16) * (D_QK // 4), pv_passes=(D_V // 16) * (bc // 4))


class TileSource:
    """fa_sweep's tile stream of one cluster: K and V tile j of token chunk i are ring chunks
    base + 2 j and base + 2 j + 1, through the cluster's slabs in turn."""

    def __init__(self, fb, g, i):
        self.fb, self.g, self.i = fb, g, i

    def fetch(self, kind, j):
        fb = self.fb
        n = fb.chunk_index(self.i, "kv") + 2 * j + (kind == "V")
        # kv_slabs split: every K tile in slab 0, V tiles in slabs 1 and 2 in turn, so K_{j+1} loads under
        # PV(j-1) and V_{j+1} under QK(j+1) (in turn, K_{j+2} would wait for PV(j), right before QK(j+2))
        slab = None
        if fb.kv_slabs == "split":
            slab = 0 if kind == "K" else 1 + j % 2
        ld, si, slab = fb.load(self.g, n, f"Ld_{kind}{j}_i{self.i}", slab=slab)
        return ld, slab, si

    def release(self, token, task):
        self.fb.ls[self.g].release(token, task)


class FaBuild(cm.ChipletBuild):
    TAG = "fp"

    def __init__(self, args, p, plat, hw):
        super().__init__(args, p, plat, hw)
        if self.crest:
            raise ValueError("params weight_crest: the prefill streams plain INT8")
        m = check(p)
        self.NCH, self.NQ, self.NT = m["chunks"], m["query_tiles"], m["tiles"]
        self.C = int(p["chunk"])
        self.ROWS = self.C // self.NC                  # query rows a cluster
        self.keys, self.bc = int(p["keys"]), int(p["bc"])
        if simd_fp16_lanes() != BR:
            raise ValueError(f"the SIMD has {simd_fp16_lanes()} FP16 lanes; the softmax takes one query "
                             f"a lane and the tiles are {BR} rows")
        self.gemm_async = bool(p.get("gemm_async", True))
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
        self.x_ring = bool(p.get("x_ring", True))
        # l3_overlay: the root's y lives in its x buffers (y of chunk i, cluster 0 in the landing buffer of
        # chunk i, cluster 1 in the source of chunk i): every chip's image is the same, and an 8 MiB L3 cannot
        # hold y beside both x buffers, the ring and the task tables. A chunk's y is stored after its reduce,
        # which follows every use of that chunk's x.
        self.l3_overlay = bool(p.get("l3_overlay", False))
        # x_puts_first: the root multicasts every chunk's x before any chunk's loads, so no chip waits for
        # the root's DM core to work through a chunk before the next chunk's x is sent
        self.x_puts_first = bool(p.get("x_puts_first", False))
        # y_halves: a cluster's partial y goes through a symmetric buffer of half its size, one W_O block
        # at a time: W_O block 0, its ChainGather, W_O block 1 behind that gather, its ChainGather
        self.y_halves = bool(p.get("y_halves", False))
        if self.y_halves and (self.reduce_impl != "chain" or self.sync == "barrier"):
            raise ValueError("y_halves needs reduce: chain and sync: edge")
        # chain_concurrent (default true): cluster 0's and cluster 1's ChainGathers run at the same time
        # through every chip. False: every ChainGather after the previous one of either cluster, so a
        # chain's finish never meets the other cluster's chain data in a chip's D2D framer (the MoE's
        # deadlock with a memory chip's push through a hop chip).
        self.chain_concurrent = bool(p.get("chain_concurrent", True))
        # kv_slabs (rr | split): the sweep's K and V tiles through the W slabs in turn, or (split, three
        # slabs) K tiles in slab 0 and V tiles in slabs 1 and 2 alternately
        self.kv_slabs = str(p.get("kv_slabs", "rr"))
        if self.kv_slabs not in ("rr", "split") or (self.kv_slabs == "split" and int(p.get("w_buffers", 3)) != 3):
            raise ValueError(f"params kv_slabs={self.kv_slabs!r}: rr, or split with w_buffers 3")
        # wo_split_rows: W_O as GEMMs of two 16-row tiles each instead of one of all the cluster's rows
        self.wo_split_rows = bool(p.get("wo_split_rows", False))
        # prefetch_after_x: the memory chips' pushes start after the root's x multicasts (x_puts_first)
        self.prefetch_after_x = bool(p.get("prefetch_after_x", False))
        # y_block_chains: a ChainGather per W_O block (the whole partial y in the symmetric buffer)
        self.y_block_chains = bool(p.get("y_block_chains", False))
        if self.y_block_chains and (self.y_halves or self.reduce_impl != "chain" or self.sync == "barrier"):
            raise ValueError("y_block_chains: reduce chain, sync edge, no y_halves")
        self._wo_blk = {}
        self._last_chain = None
        self.checks_kind = str(p.get("checks", "y"))
        # head h(k): a memory chip's row holds four consecutive heads, nearest the memory chip last
        by_row = sorted(self.chips, key=lambda k: (k & 15, k >> 4))
        self.head = {k: i for i, k in enumerate(by_row)}
        if len(self.head) != 16:
            raise ValueError(f"{len(self.head)} compute chips: the layer has 16 heads, one a chip")
        kv = os.path.join(REPO, p.get("kv_cache", "target/sim/automation/sweep/llm_blocks/fa/cache/"
                                                  "dsv2_l1_kv_8193.npz"))
        wc = os.path.join(REPO, p.get("weights", "target/sim/automation/sweep/common/weights/cache"))
        d = self.d = fa_prefill_data.FaPrefillData(kv, wc, args.dsv2_dir, tokens=self.T, keys=self.keys,
                                                   bc=self.bc, n_calib=int(p.get("n_calib", 32)))
        self.lay = d.layout
        # the fold order of the 16 partials: the chips' order from the base class, as heads
        self.fold_heads = [self.head[k] for k in self.fold_order()]
        for line in d.report(self.fold_heads):
            print(f"[fp] {line}")
        print(f"[fp] {self.T} tokens in {self.NCH} chunks of {self.C}: {self.NQ} query tiles of {BR} rows a "
              f"cluster, {self.NT} K/V tiles of {self.bc} keys; fold order "
              f"{[f'{k:02x}' for k in self.order]}")
        self.XB = self.C * HIDDEN                       # a chunk's x, INT8
        self.YB = self.ROWS * HIDDEN * 2                # a cluster's partial y of a chunk, FP16
        self.SL = int(p.get("root_slice", 16 * 1024))
        self._stage()
        self.make_streams()
        self._make_rings()
        self._buffers()

    # ---- the ring's plan: per token chunk ------------------------------------------------------
    PER_CHUNK_HEAD, PER_CHUNK_TAIL = 4, 3          # W_q x 3 and W_UK; W_UV and W_O x 2

    def per_chunk(self):
        """Ring chunks a token chunk takes: a run that stops early takes only what it uses."""
        if self.stop == "x":
            return self.PER_CHUNK_HEAD if self.x_ring else 0
        if self.stop == "q8":
            return self.PER_CHUNK_HEAD
        kv = 2 * self.NT
        return self.PER_CHUNK_HEAD + kv + (0 if self.stop == "sweep" else self.PER_CHUNK_TAIL)

    def chunk_index(self, i, part):
        """The ring index of token chunk i's first chunk of `part`: head (W_q, W_UK), kv, tail."""
        base = i * self.per_chunk()
        return base + {"head": 0, "kv": self.PER_CHUNK_HEAD,
                       "tail": self.PER_CHUNK_HEAD + 2 * self.NT}[part]

    # ---- data -----------------------------------------------------------------------------------
    def _stage(self):
        st, d, T, lay = self.st, self.d, self.TAG, self.lay
        mesh = d.mesh
        # the prompt's rows, A layout per chunk: the root's source and every chip's landing
        self.xsrc = np.concatenate([d.x_chunk_a(i * self.C, self.C) for i in range(self.NCH)]).astype(np.int8)
        self.h_xsrc = st.put(f"{T}_xsrc", "int8_t", self.xsrc)
        st.put_zeros(f"{T}_xland", "int8_t", self.NCH * self.XB, preload=True)
        st.put_zeros(f"{T}_xflag", "uint8_t", self.NCH * 64)
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
                cs, sn = d.rope_tables(i * self.C + c * self.ROWS, self.ROWS)
                rope += [cs, sn]
        self.h_rope = st.put(f"{T}_rope", "uint16_t", np.concatenate(rope).astype(np.float16).view(np.uint16))
        self.ROPE_B = self.ROWS * ROPE * 2
        # the cache, once in each memory chip's HBM, tile by tile K0 V0 K1 V1 ...
        tiles = d.cache_tiles()
        cache = np.concatenate([np.concatenate([k, v]) for k, v in tiles]).astype(np.int8)
        self.kv_bytes = [(k.size, v.size) for k, v in tiles]
        self.h_kv, self.h_w, self.h_c = {}, {}, {}
        for m in sorted(set(self.mem.values())):
            self.h_kv[m] = st.put_hbm(f"{T}_kv_m{m:02x}", cache, mem_chip=(m >> 4, m & 15))
        # per head: the weights as the ring streams them, and the dequantisation constants
        self.w_off = {}
        for k in self.chips:
            h = self.head[k]
            loc = (self.mem[k] >> 4, self.mem[k] & 15)
            chunks = d.weight_chunks(h)
            self.w_off[k], off = [], 0
            for nm, b in chunks:
                self.w_off[k].append((nm, off, b.size))
                off += b.size
            self.h_w[k] = st.put_hbm(f"{T}_w_k{k:02x}", np.concatenate([b for _, b in chunks]), mem_chip=loc)
            # W_q's per-column scales, one copy per 16-row tile of the cluster's rows: dq{r} multiplies all
            # of them element by element
            sq = np.concatenate([np.tile(d.scale_tile(d.q_scale(h)[64 * r:64 * (r + 1)]), self.ROWS // 16)
                                 for r in range(3)])
            const = np.concatenate([sq.view(np.uint8), d.scale_tile(d.qt_scale(h)).view(np.uint8),
                                    d.scale_tile(d.uv_scale(h)).view(np.uint8)])
            self.h_c[k] = st.put_hbm(f"{T}_c_k{k:02x}", const.astype(np.uint8), mem_chip=loc)
        self.SQ_B = (self.ROWS // 16) * 2048           # one W_q block's scales: 16 rows x 64 FP16 a row tile
        self.C_SQ, self.C_QT = 0, 3 * self.SQ_B
        self.C_UV = self.C_QT + 16384
        self.C_BYTES = self.C_UV + 4096

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
        self.rings, self.prefetch, self.ring_take = {}, {}, {}
        from bingo_kernel_args import HostBingoKernelWeightPrefetchArgs
        for k in self.chips:
            self.rings[k] = WeightRings(
                slots=self.h_wslots, flags=self.h_flags, releases=self.h_rel, n_rings=1, n_slots=self.n_slots,
                slot_bytes=self.slot_bytes, batch=self.ring_batch, head=self.ring_head,
                batch_head=self.ring_batch_head)
            self.ring_take[k] = []
            self.prefetch[k] = self.cx_chip(k).host_node(
                f"weight_prefetch_k{k:02x}", "__host_bingo_kernel_weight_prefetch",
                HostBingoKernelWeightPrefetchArgs(
                    BingoMemSymbol(f"{T}_ring_sched_k{k:02x}"), self.rings[k].slots, self.h_flags, self.h_rel,
                    self.h_seq[self.mem[k]], self.h_rec, 1, 1, self.n_slots, self.slot_bytes, self.mem[k],
                    batch=self.ring_batch, policy=int(wr.get("policy", 1)), engine=self.engine[k],
                    queued=self.ring_queued, head=self.ring_head, batch_head=self.ring_batch_head))
        self.ld_node = {}

    def _src(self, k, n):
        """(HBM address, bytes) of chip k's ring chunk n."""
        r = n % self.per_chunk()
        if r < 4:                                   # W_q x 3, W_UK
            nm, off, size = self.w_off[k][r]
            return self.h_w[k].address + off, size
        if r < 4 + 2 * self.NT:                     # K, V
            j, v = divmod(r - 4, 2)
            off = sum(a + b for a, b in self.kv_bytes[:j]) + (self.kv_bytes[j][0] if v else 0)
            return self.h_kv[self.mem[k]].address + off, self.kv_bytes[j][v]
        nm, off, size = self.w_off[k][4 + r - 4 - 2 * self.NT]      # W_UV, W_O x 2
        return self.h_w[k].address + off, size

    # ---- buffers --------------------------------------------------------------------------------
    def _buffers(self):
        d, R = self.d, self.ROOT
        mesh = d.mesh
        self.sym_bytes = self.YB // 2 if self.y_halves else self.YB
        self.sym_zero = False
        self.sym = {(k, c): self.cx_chip(k).at(c).l1(f"AAAA_{self.TAG}_sym", self.sym_bytes)
                    for k in self.chips for c in range(self.NC)}
        self.sweep_cfg = FaSweepCfg(bc=self.bc, nt=self.NT, nq=self.NQ, br=BR, k_s=d.k_s, k_o=d.k_o,
                                    a_exp=d.a_exp, a_n_f32bits=d.a_n_bits, gemm_async=self.gemm_async)
        self.consts = {g: self.cx_chip(self.chip(g)).at(self.c(g)).l1(f"{self.TAG}_consts", self.C_BYTES)
                       for g in range(self.G)}
        self._bufs, self._dst, self._slot, self._root_wait = {}, {}, {}, {}

    def dst(self, i, c):
        """The root's cluster c landing buffer of chunk i's reduce: the reduced rows, then stored."""
        if (i, c) not in self._dst:
            self._dst[i, c] = self.cx_chip(self.ROOT).at(c).l1(f"{self.TAG}_dst_i{i}", self.YB)
        return self._dst[i, c]

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

    def bufs(self, g, i):
        """The buffers of chunk i on cluster g. A chunk has its own, so the static L1 planner overlays
        the ones whose lifetimes cannot meet (the projection's with the previous sweep's, say)."""
        if (g, i) in self._bufs:
            return self._bufs[g, i]
        k, c = self.chip(g), self.c(g)
        rows, nq = self.ROWS, self.NQ
        l1 = lambda name, nb: self.cx_chip(k).at(c).l1(f"{self.TAG}_{name}_i{i}", nb)
        b = {}
        b["xrows"] = l1("xrows", rows * HIDDEN)
        b["ropeb"] = l1("ropeb", 2 * self.ROPE_B)
        # q: three D regions of 64 columns (the last is q_pe), rows x 64 FP16 each
        b["q"] = [l1(f"q{r}", rows * 64 * 2) for r in range(3)]
        b["a16n"] = l1("a16n", rows * Q_NOPE * 2)             # q_nope, A layout FP16
        b["qn8"] = l1("qn8", rows * Q_NOPE)
        b["qt"] = l1("qt", rows * KV_RANK * 2)                # q~, D layout FP16
        b["a16q"] = l1("a16q", rows * KV_RANK * 2)
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
        self._bufs[g, i] = b
        return b

    # ---- the ring's loads -----------------------------------------------------------------------
    def load(self, g, n, name, deps=(), slab=None):
        """Ring chunk n of chip(g) into cluster g's next slab: cluster 1 reads it, cluster 0 reads it and
        releases the slot, after cluster 1's read. Returns (load task, slab index, slab)."""
        k, c, dm = self.chip(g), self.c(g), self.roles["dm"]
        takes = self.ring_take[k]
        if n == len(takes):
            src, nb = self._src(k, n)
            takes.append(self.rings[k].take_ex(0, src=src, nbytes=nb) + (nb, self.rings[k].last_take))
        elif n > len(takes):
            raise ValueError(f"ring chunk {n} of chip {k:#04x} asked for before {len(takes)}")
        sl, fl, rl, seq, _, _, nb, ex = takes[n]
        ls = self.ls[g]
        si, slab, war = ls.take(nb) if slab is None else ls.take_at(slab, nb)
        prev = [self.last_ld[g]] if g in self.last_ld else []
        # the slot is released by cluster 0's read, which follows every other cluster's read of it
        readers = [self.ld_node[(k, c2, n)] for c2 in range(1, self.NC) if (k, c2, n) in self.ld_node] \
            if c == 0 else []
        ld = self.node(k, c, dm, f"{name}", "__snax_bingo_kernel_idma_ring_load",
                       SnaxBingoKernelIdmaRingLoadArgs(fl, seq, sl, slab, nb, rl if c == 0 else 0),
                       list(dict.fromkeys(war + prev + list(deps) + readers)))
        if c != 0 and (k, 0, n) in self.ld_node:
            self.dfg.bingo_add_edge(ld, self.ld_node[(k, 0, n)])
        self.last_ld[g] = ld
        ld.exec_chunk = ex
        self.ld_node[(k, c, n)] = ld
        return ld, si, slab

    # ---- helpers ----------------------------------------------------------------------------
    def gmm(self, g, name, a, b, d, M, K, N, k_shift, deps, phase="fa"):
        """One GEMM of cluster g as the queue makes it. Returns the task its output follows."""
        k, c = self.chip(g), self.c(g)
        args = SnaxBingoKernelGemmFullArgs(
            input_A_addr=a, input_B_addr=b, input_C_addr=0, output_D_addr=d, M=M, K=K, N=N, array_shape_idx=0,
            transpose_A=0, transpose_B=0, accumPrevC=0, int32tofp16_enable=1, int4_b_enable=0, d_shift=k_shift)
        if not self.gemm_async:
            return self.node(k, c, self.roles["gemm"], name, "__snax_bingo_kernel_gemm_full", args, list(deps))
        return GemmQueue.of(self.dfg, True).dispatch(self.at(k, c), name, args, list(deps), phase)

    def simd_mul(self, g, name, a, b, out, nelems, deps):
        k, c = self.chip(g), self.c(g)
        return self.node(k, c, self.roles["simd"], name, "__snax_bingo_kernel_simd_stream_elementwise",
                         SnaxBingoKernelSimdMulF16Args(a, b, out, nelems // 32, 32), list(deps))

    def simd_add(self, g, name, a, b, out, nelems, deps):
        k, c = self.chip(g), self.c(g)
        return self.node(k, c, self.roles["simd"], name, "__snax_bingo_kernel_simd_stream_elementwise",
                         SnaxBingoKernelSimdAddF16Args(a, b, out, nelems // 32, 32), list(deps))

    def quant(self, g, name, src, dst, nbytes, inv_bits, deps):
        k, c = self.chip(g), self.c(g)
        return self.node(k, c, self.roles["simd"], name, "__snax_bingo_kernel_simd_fp16_to_int8",
                         SnaxBingoKernelSimdFp16ToInt8Args(src, dst, beats=simd_beats(nbytes), rows=1,
                                                           inv_scale_f32bits=inv_bits), list(deps))

    def xd(self, g, name, args, deps):
        k, c = self.chip(g), self.c(g)
        return self.node(k, c, self.roles["xdma"], name, "__snax_bingo_kernel_xdma_6d", args, list(deps))

    def dm_copy(self, g, name, src, dst, nbytes, deps):
        k, c = self.chip(g), self.c(g)
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
                         list(after))

    def stage_x(self, g, i, put, deps):
        """Cluster g's rows of chunk i into its L1: behind the landed flag, or pulled from the root."""
        T, k, c = self.TAG, self.chip(g), self.c(g)
        half = self.XB // self.NC
        xb = self.bufs(g, i)["xrows"]
        if self.bcast_impl == "bcast_l1" and k == self.ROOT and self.x_root == "poll2d":
            # the flagged fetch's two DMAs as two tasks: the 4-byte 2D read of the flag into the rows'
            # buffer, then the 1D copy of the rows from the landing buffer
            poll = self.node(k, c, self.roles["dm"], f"x_poll_i{i}", "__snax_bingo_kernel_idma_2d_copy",
                             SnaxBingoKernelIdma2dCopyArgs(BingoMemSymbol(f"{T}_xflag", offset=i * 64), xb, 4, 64, 4, 1),
                             list(deps) + [put])
            return self.node(k, c, self.roles["dm"], f"x_copy_i{i}", "__snax_bingo_kernel_idma_1d_copy",
                             SnaxBingoKernelIdma1dCopyArgs(BingoMemSymbol(f"{T}_xland", offset=i * self.XB + c * half),
                                                           xb, half), [poll])
        if self.bcast_impl == "bcast_l1" and k == self.ROOT and self.x_root == "copy":
            return self.node(k, c, self.roles["dm"], f"x_copy_i{i}", "__snax_bingo_kernel_idma_1d_copy",
                             SnaxBingoKernelIdma1dCopyArgs(BingoMemSymbol(f"{T}_xsrc", offset=i * self.XB + c * half),
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
                               BingoMemSymbol(f"{T}_xland", offset=i * self.XB + c * half), xb, half,
                               BingoMemSymbol(f"{T}_xflag", offset=i * 64), 1, 1),
                           list(deps) + first + ([put] if k == self.ROOT else []))
            if k == self.ROOT:
                self._root_wait[i, c] = nd
            if k != self.ROOT:
                self.dfg.bingo_add_order_edge(put, nd)
            return nd
        return self.node(k, c, self.roles["dm"], f"x_pull_i{i}", "__snax_bingo_kernel_idma_1d_copy",
                         SnaxBingoKernelIdma1dCopyArgs(
                             BingoMemSymbol(f"{T}_xsrc", offset=i * self.XB + c * half, chip_id=self.ROOT), xb, half),
                         list(deps))

    def stage_proj(self, g, i, x):
        """q, q~ and the rotated q_pe of chunk i into Q8, one query tile at a time. Returns
        {q: (Q8 handle, [tasks it follows])} and the tasks that free the chunk's x rows."""
        d, b = self.d, self.bufs(g, i)
        k, c = self.chip(g), self.c(g)
        rows, mt = self.ROWS, self.ROWS // 16
        mesh = d.mesh
        # constants (once), the rotation tables of this chunk
        K0 = self.C_SQ
        sq = [self.consts[g].view(K0 + self.SQ_B * r) for r in range(3)]
        # ---- q = x W_q, three blocks of 64 columns; the third is q_pe --------------------------
        qd, gq = [], []
        for r in range(3):
            n = self.chunk_index(i, "head") + r
            ld, si, slab = self.load(g, n, f"Ld_wq{r}_i{i}")
            deps = [ld, x]
            gm = self.gmm(g, f"gq{r}_i{i}", b["xrows"], slab, b["q"][r], mt, HIDDEN // 4, 4, d.k_x, deps, "proj")
            self.ls[g].release(si, gm)
            gq.append(gm)
        # ---- dequantise the three regions --------------------------------------------------------
        dq = [self.simd_mul(g, f"dq{r}_i{i}", b["q"][r], sq[r], b["q"][r], rows * 64, [gq[r]]) for r in range(3)]
        # ---- q_nope: D -> A (FP16) per row block and half, then quantise -------------------------
        d2a = []
        for r in range(2):
            for m in range(mt):
                d2a.append(self.xd(g, f"d2an{r}{m}_i{i}", d_to_a_args(
                    mesh, 1, 4, 16, b["q"][r].view(m * 4 * TILE16), b["a16n"].view(m * 16 * Q_NOPE * 2 + r * 16 * 64 * 2)),
                    [dq[r]]))
        qn = self.quant(g, f"qn_i{i}", b["a16n"], b["qn8"], rows * Q_NOPE * 2, self.fbits(d.s_qn.inv),
                        d2a)
        # ---- absorb: q~ = q_nope W_UK ----------------------------------------------------------------
        ldu, siu, slabu = self.load(g, self.chunk_index(i, "head") + 3, f"Ld_wuk_i{i}")
        ga = self.gmm(g, f"gab_i{i}", b["qn8"], slabu, b["qt"], mt, Q_NOPE // 4, KV_RANK // 16, d.k_uk,
                      [ldu, qn], "proj")
        self.ls[g].release(siu, ga)
        sqt = self.consts[g].view(self.C_QT)
        dqt = [self.simd_mul(g, f"dqt{m}_i{i}", b["qt"].view(m * (KV_RANK // 16) * TILE16), sqt,
                             b["qt"].view(m * (KV_RANK // 16) * TILE16), 16 * KV_RANK, [ga]) for m in range(mt)]
        d2q = self.xd(g, f"d2aq_i{i}", d_to_a_args(mesh, mt, KV_RANK // 16, KV_RANK // 4, b["qt"], b["a16q"]),
                      dqt)
        # ---- q_pe: RoPE in the D layout, then D -> A and quantise ------------------------------------
        ldr = self.dm_copy(g, f"rope_ld_i{i}", BingoMemSymbol(f"{self.TAG}_rope", offset=2 * (i * self.NC + c) * self.ROPE_B),
                           b["ropeb"], 2 * self.ROPE_B, [])
        sw = self.node(k, c, self.roles["dm"], f"swap_i{i}", "__snax_bingo_kernel_idma_pairwise_swap",
                       SnaxBingoKernelIdmaPairwiseSwapArgs(b["q"][2], b["xs"], rows * ROPE, 2), [dq[2]])
        n_pe = rows * ROPE
        m1 = self.simd_mul(g, f"rope_cos_i{i}", b["q"][2], b["ropeb"], b["t1"], n_pe, [dq[2], ldr])
        m2 = self.simd_mul(g, f"rope_sin_i{i}", b["xs"], b["ropeb"].view(self.ROPE_B), b["t2"], n_pe, [sw, ldr])
        ad = self.simd_add(g, f"rope_add_i{i}", b["t1"], b["t2"], b["pe"], n_pe, [m1, m2])
        d2p = self.xd(g, f"d2ap_i{i}", d_to_a_args(mesh, mt, 4, 16, b["pe"], b["a16p"]),
                      [ad])
        # ---- Q8 per query tile ---------------------------------------------------------------------
        q8 = {}
        for q in range(self.NQ):
            qa = self.quant(g, f"q8a{q}_i{i}", b["a16q"].view(q * 16 * KV_RANK * 2), b["q8"][q], 16 * KV_RANK * 2,
                            self.fbits(d.s_qt.inv), [d2q])
            qp = self.quant(g, f"q8p{q}_i{i}", b["a16p"].view(q * 16 * ROPE * 2), b["q8"][q].view(16 * KV_RANK),
                            16 * ROPE * 2, self.fbits(d.s_qpe.inv), [d2p])
            q8[q] = (b["q8"][q], [qa, qp])
        # the sweep starts behind every projection task, so its buffers and the projection's can share L1
        done = [t for q in q8.values() for t in q[1]]
        return {q: (h, done) for q, (h, _) in q8.items()}

    @staticmethod
    def fbits(inv):
        return int(np.asarray(inv, dtype=np.float32).reshape(()).view(np.uint32))

    def stage_post_a(self, g, i, sweep):
        """o~ -> W_UV -> o of chunk i (INT8, A layout): everything before the W_O matmuls.
        Returns (the tasks that read the sweep's xt, the task that made o)."""
        d, b = self.d, self.bufs(g, i)
        rows, mt = self.ROWS, self.ROWS // 16
        mesh = d.mesh
        # o~ rows, FP16 row-major [16, 512] a query tile -> A layout FP16 -> INT8
        cv = []
        for q in range(self.NQ):
            cv.append(self.xd(g, f"o2a{q}_i{i}", convert_args("row_major", "A", 16, D_V, mesh, 2, sweep["xt"][q],
                                                              b["a16o"].view(q * 16 * D_V * 2)), list(sweep["tr"])))
        qo = self.quant(g, f"ot8_i{i}", b["a16o"], b["ot8"], rows * D_V * 2, self.fbits(d.s_ot.inv), cv)
        ld, si, slab = self.load(g, self.chunk_index(i, "tail"), f"Ld_wuv_i{i}")
        gu = self.gmm(g, f"guv_i{i}", b["ot8"], slab, b["uv"], mt, D_V // 4, Q_NOPE // 16, d.k_uv, [ld, qo], "post")
        self.ls[g].release(si, gu)
        suv = self.consts[g].view(self.C_UV)
        du = [self.simd_mul(g, f"dqu{m}_i{i}", b["uv"].view(m * (Q_NOPE // 16) * TILE16), suv,
                            b["uv"].view(m * (Q_NOPE // 16) * TILE16), 16 * Q_NOPE, [gu]) for m in range(mt)]
        d2u = self.xd(g, f"d2au_i{i}", d_to_a_args(mesh, mt, Q_NOPE // 16, Q_NOPE // 4, b["uv"], b["a16u"]), du)
        qq = self.quant(g, f"o8_i{i}", b["a16u"], b["o8"], rows * Q_NOPE * 2, self.fbits(d.s_o.inv), [d2u])
        return cv, qq

    def stage_post_wo(self, g, i, qq, r, after):
        """W_O block r of chunk i into the cluster's half-size symmetric buffer (y_halves)."""
        d, b = self.d, self.bufs(g, i)
        k, c = self.chip(g), self.c(g)
        ld, si, slab = self.load(g, self.chunk_index(i, "tail") + 1 + r, f"Ld_wo{r}_i{i}")
        go = self.gmm(g, f"gwo{r}_i{i}", b["o8"], slab, self.sym[(k, c)], self.ROWS // 16, Q_NOPE // 4, 64,
                      d.k_wo, [ld, qq] + list(after), "post")
        self.ls[g].release(si, go)
        return go

    def stage_post_b(self, g, i, qq, sym_after):
        """W_O of chunk i: two blocks of 1024 columns into the two halves of the cluster's partial, in
        its symmetric buffer, which `sym_after` (the previous chunk's reduce) has finished reading.
        Returns the task after which the partial is there."""
        d, b = self.d, self.bufs(g, i)
        k, c = self.chip(g), self.c(g)
        mt = self.ROWS // 16
        half = self.YB // 2
        go = None
        # wo_split_rows: each block as one GEMM a pair of 16-row tiles (o8 is row-tile major, the output
        # tile (m, n) sits at (m * 64 + n) tiles)
        parts = [(0, mt)] if not self.wo_split_rows else [(m0, 2) for m0 in range(0, mt, 2)]
        for r in range(2):
            ld, si, slab = self.load(g, self.chunk_index(i, "tail") + 1 + r, f"Ld_wo{r}_i{i}")
            for m0, mm in parts:
                sfx = "" if len(parts) == 1 else f"m{m0}"
                go = self.gmm(g, f"gwo{r}{sfx}_i{i}", b["o8"].view(m0 * 16 * Q_NOPE), slab,
                              self.sym[(k, c)].view(r * half + m0 * 64 * TILE16), mm, Q_NOPE // 4, 64,
                              d.k_wo, [ld, qq] + list(sym_after), "post")
            self.ls[g].release(si, go)
            self._wo_blk[(k, c, i, r)] = go
        return go

    # ---- the reduce of one chunk ------------------------------------------------------------------
    def stage_reduce(self, i, parts, war, gate=None, half=None, src_off=0):
        """Cluster index c's partials of chunk i into the root's cluster c: ChainGather (chain) or the
        root's pulls and adds (root). parts {(k, c): [tasks after which the partial is there]}, war
        {c: [tasks the destination waits for]}, gate {c: [tasks the reduce starts behind]} (the root's
        sweep read out, whose L1 the destination takes). Returns {c: the last task of the reduce}."""
        R, xdma, dm, simd = self.ROOT, self.roles["xdma"], self.roles["dm"], self.roles["simd"]
        others = self.order[1:]
        out = {}
        before = {}
        for c in range(self.NC):
            before[c] = [t for k in self.chips for t in parts[(k, c)]] + list((gate or {}).get(c, []))
        if self.sync == "barrier":
            bar = self.barrier({k: [t for c in range(self.NC) for t in parts[(k, c)]] for k in self.chips})
            for c in range(self.NC):
                before[c] = [bar[R]] + list((gate or {}).get(c, []))
            self.bar_reduce = bar
        for c in range(self.NC):
            dstb = self.dst(i, c)
            size, doff, tag = (self.YB, 0, "") if half is None else (self.YB // 2, half * self.YB // 2, f"_h{half}")
            dst = f"((uint64_t)({dstb.get_c_var_name()}) + {doff})"
            if self.reduce_impl == "chain":
                ga = SnaxBingoKernelXdmaChainGatherArgs.fold_into(
                    cm.remote, R, c, others, src=self.sx(c, src_off), dst=dst, size=size,
                    junction=self.jct["Elementwise"], jct_csr0=EW_F16_ADD, holds=[dstb])
                serial = [] if self.chain_concurrent or self._last_chain is None else [self._last_chain]
                out[c] = self.node(R, c, xdma, f"y_chain_c{c}_i{i}{tag}", "__snax_bingo_kernel_xdma_chain_gather", ga,
                                   before[c] + list(war.get(c, [])) + serial)
                self._last_chain = out[c]
                continue
            ns = self.YB // self.SL
            pulls, adds = [], []
            n = 0
            for ik, k in enumerate(others):
                for s in range(ns):
                    slot = self.slot(i, c).view((n % 2) * self.SL)
                    cp = self.node(R, c, dm, f"y_pull_c{c}_i{i}_k{k:02x}_s{s}", "__snax_bingo_kernel_idma_1d_copy",
                                   SnaxBingoKernelIdma1dCopyArgs(
                                       cm.remote(k, f"({self.sx(c, 0)} + {s * self.SL})"), slot, self.SL),
                                   before[c] + list(war.get(c, [])) + pulls[-1:] + ([adds[n - 2]] if n >= 2 else []))
                    pulls.append(cp)
                    a_in = self.sym[(R, c)].view(s * self.SL) if ik == 0 else dstb.view(s * self.SL)
                    adds.append(self.node(R, c, simd, f"y_add_c{c}_i{i}_k{k:02x}_s{s}",
                                          "__snax_bingo_kernel_simd_stream_elementwise",
                                          SnaxBingoKernelSimdAddF16Args(a_in, slot, dstb.view(s * self.SL),
                                                                        self.SL // 64, 32),
                                          [cp] + before[c] + adds[-1:]))
                    n += 1
            out[c] = adds[-1]
        return out

    def stage_store(self, i, red):
        """The root's cluster c stores chunk i's rows of y to its L3."""
        R, dm = self.ROOT, self.roles["dm"]
        st = {}
        for c in range(self.NC):
            st[c] = self.node(R, c, dm, f"y_store_c{c}_i{i}", "__snax_bingo_kernel_idma_1d_copy",
                              SnaxBingoKernelIdma1dCopyArgs(
                                  self.dst(i, c), self.y_sym(i, c),
                                  self.YB), list(red[c]) if isinstance(red[c], (list, tuple)) else [red[c]])
        return st

    # ---- the prefill ------------------------------------------------------------------------------
    def build(self):
        d, R = self.d, self.ROOT
        dm = self.roles["dm"]
        self.last_ld = {}
        # a slab is never loaded again before the reader of its last load is known
        for ls in self.ls:
            ls.strict = True
        # the dequantisation constants of every cluster into its L1 before the start
        # (one chain a chip, ending on cluster 0: the start anchor sits on the root's cluster 0 DM core
        # behind that core's previous task, so a chain ending there adds no second edge into the anchor)
        pre = {k: [] for k in self.chips}
        for k in self.chips:
            for c in reversed(range(self.NC)):
                g = self.chips.index(k) * self.NC + c
                pre[k] = [self.dm_copy(g, f"const_ld_g{g}", self.h_c[k].address, self.consts[g],
                                       self.C_BYTES, pre[k])]
        go = self.setup(extra=pre)
        if self.p.get("relay_per_gather", True):
            self.dfg.relay_one_consumer_cores = [self.roles["xdma"]]
        # chunk-major: every cluster's chunk i, then the barriers and the reduce in the order the arm needs
        puts = []
        swp_res, post_res, parts_ready = {}, {}, {}
        red, stores = {}, {}
        sym_free = {}
        if self.bcast_impl == "bcast_l1" and self.x_puts_first:
            for i in range(self.NCH):
                puts.append(self.stage_x_put(i, [go[R]] + puts[-1:]))
            # prefetch_after_x: the memory chips start pushing once the root's x multicasts are done, so no
            # push meets a multicast on the D2D links
            if self.prefetch_after_x:
                for k in self.chips:
                    self.edge(puts[-1], self.prefetch[k])
        for i in range(self.NCH):
            if self.bcast_impl == "bcast_l1" and not self.x_puts_first:
                puts.append(self.stage_x_put(i, [go[R]] + puts[-1:]))
            sweeps = {}
            for g in range(self.G):
                k, c = self.chip(g), self.c(g)
                # the rows land once the previous chunk's sweep has been read out (its L1 is theirs)
                x = self.stage_x(g, i, puts[i] if puts else None,
                                 [go[k]] + (list(post_res[g]["xt_rd"]) if g in post_res else []))
                if self.stop == "x":
                    # with x_ring, the chunk's W_q and W_UK loads as in q8, each slab freed by its own load
                    rd = [x]
                    if self.x_ring:
                        for r in range(self.PER_CHUNK_HEAD):
                            ld, si, _ = self.load(g, self.chunk_index(i, "head") + r, f"Ld_h{r}_i{i}")
                            self.ls[g].release(si, ld)
                            rd.append(ld)
                    parts_ready[(k, c, i)] = rd
                    post_res[g] = dict(xt_rd=[x])
                    continue
                q8 = self.stage_proj(g, i, x)
                if self.stop == "q8":
                    parts_ready[(k, c, i)] = list(q8[0][1])
                    post_res[g] = dict(xt_rd=parts_ready[(k, c, i)])
                    continue
                sw = fa_sweep(self.at(k, c), self.sweep_cfg, self.bufs(g, i)["swp"], TileSource(self, g, i), q8,
                              self.roles, name=f"s{i}_",
                              early=2 if len(self.ls[g].slabs) >= 4 else 1)
                swp_res[g] = sw
                if self.stop == "sweep":
                    parts_ready[(k, c, i)] = list(sw["tr"])
                    post_res[g] = dict(xt_rd=parts_ready[(k, c, i)])
                    continue
                sweeps[g] = sw
            if self.stop in ("x", "q8", "sweep"):
                continue
            # everything of chunk i before its W_O, then chunk i-1's reduce under it (the symmetric buffer
            # is written again by W_O, which waits for that reduce); the reduce starts once the root's
            # sweep is read out, so its landing buffer can take the sweep's L1
            post = {g: self.stage_post_a(g, i, sweeps[g]) for g in range(self.G)}
            if self.y_halves:
                # W_O block 0 everywhere, its gather, block 1 behind that gather, its gather, the store
                gate = {c: post[self.chips.index(R) * self.NC + c][0] for c in range(self.NC)}
                rh, gos = [], {}
                for r in range(2):
                    for g in range(self.G):
                        k, c = self.chip(g), self.c(g)
                        after = sym_free.get((k, c), []) if r == 0 else [rh[0][c]]
                        gos[g] = self.stage_post_wo(g, i, post[g][1], r, after)
                        post_res[g] = dict(xt_rd=post[g][0])
                        parts_ready[(k, c, i)] = [gos[g]]
                    if self.stop != "full":
                        rh.append({c: gos[self.chips.index(R) * self.NC + c] for c in range(self.NC)})
                        continue
                    parts = {(k, c): [gos[self.chips.index(k) * self.NC + c]] for k in self.chips for c in range(self.NC)}
                    rh.append(self.stage_reduce(i, parts, {}, gate if r == 0 else None, half=r))
                if self.stop == "full":
                    st = self.stage_store(i, {c: [rh[0][c], rh[1][c]] for c in range(self.NC)})
                    for c in range(self.NC):
                        red[(i, c)] = rh[1][c]
                        stores[(i, c)] = st[c]
                        for k in self.chips:
                            sym_free[(k, c)] = [rh[1][c]]
                continue
            if self.stop == "full" and self.sync != "barrier" and i > 0:
                gate = {c: post[self.chips.index(R) * self.NC + c][0] for c in range(self.NC)}
                self._reduce_chunk(i - 1, parts_ready, red, stores, sym_free, gate)
            for g in range(self.G):
                k, c = self.chip(g), self.c(g)
                post_res[g] = dict(xt_rd=post[g][0])
                parts_ready[(k, c, i)] = [self.stage_post_b(g, i, post[g][1], sym_free.get((k, c), []))]
            if self.stop == "full" and self.sync == "barrier":
                self._reduce_chunk(i, parts_ready, red, stores, sym_free)
        if self.stop == "full" and self.sync != "barrier" and not self.y_halves:
            self._reduce_chunk(self.NCH - 1, parts_ready, red, stores, sym_free)
        if stores:
            ends = [stores[(self.NCH - 1, c)] for c in range(self.NC)]
        else:
            # a stop before the reduce: one probe a chip behind that chip's last chunk, the root's last
            # (the end anchor sits behind the root's previous task on its core)
            last = {k: [t for c in range(self.NC) for t in parts_ready[(k, c, self.NCH - 1)]] for k in self.chips}
            ends = [self.probe(k, f"fin{k:02x}", last[k]) for k in self.chips if k != R]
            ends.append(self.probe(R, f"fin{R:02x}", last[R]))
        self.end_window(ends)
        self._checks(swp_res, post_res)

    def _reduce_chunk(self, i, parts_ready, red, stores, sym_free, gate=None):
        war = {c: [stores[(i - 1, c)]] if (i - 1, c) in stores else [] for c in range(self.NC)}
        if self.y_block_chains:
            # one gather per W_O block, each right behind that block on every chip: block 0's runs under
            # block 1's GEMM (the symmetric buffer holds both blocks)
            half = self.YB // 2
            rh = [self.stage_reduce(i, {(k, c): [self._wo_blk[(k, c, i, r)]] for k in self.chips for c in range(self.NC)},
                                    war if r == 0 else {}, gate if r == 0 else None, half=r, src_off=r * half)
                  for r in range(2)]
            st = self.stage_store(i, {c: [rh[0][c], rh[1][c]] for c in range(self.NC)})
            for c in range(self.NC):
                red[(i, c)] = rh[1][c]
                stores[(i, c)] = st[c]
                for k in self.chips:
                    sym_free[(k, c)] = [rh[0][c], rh[1][c]]
            return
        parts = {(k, c): parts_ready[(k, c, i)] for k in self.chips for c in range(self.NC)}
        r = self.stage_reduce(i, parts, war, gate)
        st = self.stage_store(i, r)
        for c in range(self.NC):
            red[(i, c)] = r[c]
            stores[(i, c)] = st[c]
            for k in self.chips:
                sym_free[(k, c)] = [r[c]]
        if self.sync == "barrier":
            # the next chunk starts after the root's reduce and store: a second barrier, behind them
            self.barrier({k: [st[c] for c in range(self.NC)] if k == self.ROOT else [self.bar_reduce[k]]
                          for k in self.chips})

    # ---- checks -----------------------------------------------------------------------------------
    def _checks(self, swp_res, post_res):
        lay, T, R, d = self.lay, self.TAG, self.ROOT, self.d
        checks = []
        n = 2048
        if self.stop == "full":
            for i in range(self.NCH):
                for c in range(self.NC):
                    y = d.reduce(self.fold_heads, i * self.C + c * self.ROWS, self.ROWS)
                    half = HIDDEN // 2
                    img = np.concatenate([lay.to_d(y[:, :half], d.mesh), lay.to_d(y[:, half:], d.mesh)]
                                         ).astype(np.float16).view(np.uint8)
                    # y_halves or y_windows 4: the head and the tail of each half (each half is a W_O block)
                    wins = (((f"y_i{i}c{c}_a", 0), (f"y_i{i}c{c}_m", self.YB // 2 - n // 2), (f"y_i{i}c{c}_b", self.YB - n))
                            if not (self.y_halves or int(self.p.get("y_windows", 3)) == 4) else
                            ((f"y_i{i}c{c}_h0a", 0), (f"y_i{i}c{c}_h0b", self.YB // 2 - n),
                             (f"y_i{i}c{c}_h1a", self.YB // 2), (f"y_i{i}c{c}_h1b", self.YB - n)))
                    for nm, off in wins:
                        checks.append((nm, self.y_sym(i, c, off, chip_id=R), img[off:off + n], n))
        if self.stop == "x":
            # the last chunk's rows in the root's cluster 0 and in cluster 1 of the chip in the middle of the fold order
            i, half = self.NCH - 1, self.XB // self.NC
            for k, c in ((R, 0), (self.order[len(self.order) // 2], 1)):
                g = self.chips.index(k) * self.NC + c
                base = i * self.XB + c * half
                # three 2 KiB windows (head, middle, tail): the host compares from uncached L3, byte by byte
                for nm, off in (("a", 0), ("m", half // 2 - 1024), ("b", half - 2048)):
                    name = f"x_k{k:02x}c{c}_{nm}"
                    checks.append((name, self.stash(name, k, c, self.bufs(g, i)["xrows"].view(off), 2048, []),
                                   self.xsrc[base + off:base + off + 2048].view(np.uint8), 2048))
        if self.checks_kind == "stages" or self.stop in ("q8", "sweep"):
            # the first and the last query tile of both clusters of the chip in the middle of the fold order,
            # before the y checks
            stage = []
            far = self.order[len(self.order) // 2]
            h = self.head[far]
            i = self.NCH - 1
            for c in range(self.NC):
                g = self.chips.index(far) * self.NC + c
                for q in sorted({0, self.NQ - 1}):
                    r0 = i * self.C + c * self.ROWS + q * BR
                    tag = f"c{c}" if q == 0 else f"c{c}q{q}"
                    q8 = d.q8_a(h, r0)
                    stage.append((f"q8_{tag}", self.stash(f"q8_{tag}", far, c, self.bufs(g, i)["q8"][q], D_QK * BR, []),
                                  q8.view(np.uint8), D_QK * BR))
                    if (self.stop in ("sweep",) or self.checks_kind == "stages") and g in swp_res:
                        ot = d.head(h)["ot16"][r0:r0 + BR].astype(np.float16).view(np.uint8)
                        stage.append((f"xt_{tag}", self.stash(f"xt_{tag}", far, c, swp_res[g]["xt"][q], D_V * BR * 2,
                                                              []), ot, D_V * BR * 2))
            checks = stage + checks
        self.run_checks(checks)

    # ---- emission -----------------------------------------------------------------------------------
    def chain_program_order(self):
        """Each device core runs its tasks in creation order, with a queued dispatch's wait behind the
        next start of its phase."""
        dev = [self.roles[r] for r in ("gemm", "simd", "xdma", "dm")]
        return GemmQueue.of(self.dfg, self.gemm_async).program_order(self.dfg, dev, tag=self.TAG)

    def finish(self, app_name, x_prefixes=("x_wait_", "x_pull_", "x_copy_")):
        self.chain_program_order()
        self.dfg.split_chip_functions = True
        super().finish(app_name, x_prefixes)


def _app_name(S):
    return (f"DeepSeek-V2-Lite layer 1 MLA attention prefill, {S.T} tokens, {len(S.chips)} chips x {S.NC} "
            f"clusters, bcast={S.bcast_impl} reduce={S.reduce_impl} sync={S.sync}")


def run():
    run_driver("fa prefill workload generator", FaBuild, _app_name)
