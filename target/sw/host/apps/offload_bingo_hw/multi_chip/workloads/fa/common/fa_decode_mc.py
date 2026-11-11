# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""DeepSeek-V2-Lite's layer-1 MLA attention as ONE DECODE STEP over a cache, mapped as the fresh prefill
(fa_stages_mc.py, stage "full"): head parallel, chip i computes head i, the 16 heads' partials folded into C00.

The token at position P (params cache, default 1,024) attends to keys 0 .. P: the P cached keys and its own. The
golden is fa_prefill_data.py's for a fresh causal prompt of P + 1 tokens, the token being its last (FaDecodeData):
the same calibration, the same keys, the same device arithmetic.

THE QUERY TILE. The GEMM array's row tile is 16 rows and the softmax takes one query a SIMD lane, so the token fills
all 16 rows of one query tile (layout.gemv_a_rep): every row of every stage is the token's, and the y row the root
stores is row 0. One cluster a chip, the query cluster (params q_cluster, default 0), runs the head's attention; the
other, the key cluster, makes the chip's share of the token's key.

THE STEPS (the prefill's stage numbers):

    stage 0  x: the token's tile (A layout, INT8, the 16 rows) sits in the HBM of the root's memory chip; that memory
             chip's push engine multicasts it into every chip's L3 behind a landed flag (__host_bingo_kernel_hbm_bcast,
             configured by the root's host), and both clusters of every chip copy it into L1 behind the flag
    cache    the cached keys, P / 16 pieces [key rows 9 KiB | value columns 8 KiB] as the prefill leaves them in every
             chip's L3 (fa_prefill_data kv_piece), sit in the same memory chip's HBM; the same engine multicasts them
             into every chip's L3, a key tile of 64 keys (four pieces) at a time, each behind its own flag. One
             memory-chip multicast reaches every chip once: every link of its tree carries the cache once, and the
             memory chips' links to rows 1-3 carry none of it (the broadcast enters those rows by the compute columns)
    stage 1  the token's key by columns: every chip's key cluster makes its W_KVA columns (kv_cols' slice, the ring's
             first chunk) for the tile, one GEMM, and puts the slice into the owner's L3 behind its flag
    stage 2  the owner (params kv_owner, default the root) finishes the key on its key cluster (FaStages.stage_kv_split:
             dequantisation, RMSNorm of c, the key rows' and value columns' INT8, RoPE of k_pe at position P), its
             piece into its own L3 at piece P / 16, the first piece of the sweep's last key tile
    stage 3  the owner multicasts the new piece into every other chip's L3, once the cache's multicast has ended (its
             last flag on the owner): one multicast source at a time
    stage 4  every query cluster: q = x W_q, q~ = q_nope W_UK, RoPE of q_pe at position P, Q8 (stage_proj_r)
    stage 5  the sweep over (P + 1) / 64 rounded up key tiles of 64: each cached tile behind its own flag, so the sweep
             runs as the cache lands, the last tile (the new piece, then zeros) behind the new piece's flag, its keys
             after position P masked (fa_prefill_data tile masks: the token is every lane)
    6 - 8    W_UV, W_O in two blocks of 1,024 columns, one reduce chain a block into the collector (the root), each
             block's row 0 stored into the root memory chip's HBM (y, FP16 [2048], row major; the output's per-column
             dequantisation stays outside the block, as the prefill's)
    append   the owner copies the new piece from its L3 into the cache's next slot in HBM (piece P / 16), after the
             chains and the y stores: no write into a memory chip while one of its engines pushes

THE ROOT'S RING. The root's memory chip pushes x, the cache and the root's own ring on one engine (each of the
memory chip's four engines serves one chip's prefetcher). The root's host runs them in turn: x, the root's first
root_ring_head ring chunks (default 5: W_KVA, W_q, W_UK), the cache, the rest of the root's ring (W_UV, W_O). The ring
holds every chunk a chip streams (weight_ring slots 8), so no prefetcher waits for a slot.

THE CHECKS. In HBM: the token's y (both W_O blocks' rows) and the appended piece (the head of its key rows, the tail of
its value columns); in the L3 of the chip in the middle of the fold order: the first and the last cached piece and the
new piece, as their multicasts landed. checks "stages" adds that chip's Q8 and o~.

PARAMETERS beyond the prefill's (params.hjson): cache (P, a whole number of 64-key tiles), q_cluster, kv_owner,
root_ring_head. The base's constructor parses every other switch; it is handed the prefill of the P cached tokens as
its shape (tokens = keys = P, chunks of prefill_chunk, default 128: only its shape checks read them) and this module's
data, and the decode's shape replaces the prefill's before anything is staged (_decode_shape). Switches of a
multi-chunk prefill are refused (UNSUPPORTED).
"""

import contextlib
import hashlib
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fa_stages_mc as fs  # noqa: E402
import fa_prefill_mc as fa  # noqa: E402
import fa_prefill_data  # noqa: E402
from fa_prefill_mc import BR, D_QK, D_V, HIDDEN, KVP, ROPE, TILE16  # noqa: E402
from bingo_kernel_args import (  # noqa: E402
    HostBingoKernelHbmBcastArgs, HostBingoKernelWeightPrefetchRangeArgs, SnaxBingoKernelIdma1dCopyArgs,
    SnaxBingoKernelIdma2dCopyArgs, SnaxBingoKernelIdmaPutFlaggedArgs, SnaxBingoKernelWaitFlagsArgs)
from bingo_mem_handle import BingoMemSymbol  # noqa: E402
from driver import run_driver  # noqa: E402

PIECE = 16                     # tokens a piece of the keys
TILE_PIECES = 4                # pieces a 64-key tile


class FaDecodeData(fa_prefill_data.FaPrefillData):
    """fa_prefill_data's data of a fresh prompt of cache + 1 tokens, the decode token at position P = cache its last:
    the calibration (the last n_calib positions), the latent's c_fold, the scales and the device's keys of all P + 1
    positions, as that prefill's golden makes them. Then the decode's views:

        rows, c8          the sweep's keys: positions 0 .. P - 1 (the cache), then the new tile: the token's key in all
                          16 rows of its piece (the device makes it from the token's 16 rows), the rest zero
        hq, cos_rep, ..   the query tile: the token in all 16 rows (head(), x_chunk_a(0, 16), rope_tables(0, 16))
        mask_patterns     the last tile's mask: every key after position P, in every lane

    head(h) is then the parent's chain for the 16 rows, attention() the sweep over the decode's tiles."""

    def __init__(self, kv_path, weights_cache, cache=1024, bc=64, n_calib=32, model="dsv2-lite-bf16",
                 rope_perm=False, mask_c=False, wbits=8):
        if cache % bc or cache // bc < 4:
            raise ValueError(f"cache {cache}: whole key tiles of {bc}, at least four (the new key starts a tile)")
        # the parent: the checkpoint's weights packed, and the prefill that left the cache (its checks take whole key
        # tiles, keys = tokens, one chunk); the T-dependent part is made again below for the P + 1 positions
        super().__init__(kv_path, weights_cache, tokens=cache, keys=cache, bc=bc, n_calib=n_calib,
                         model=model, fresh=True, chunk=cache, rope_perm=rope_perm, mask_c=mask_c, wbits=wbits)
        P, n = int(cache), int(cache) + 1
        z = np.load(kv_path)
        if z["c_unit"].shape[0] < n:
            raise ValueError(f"{kv_path}: {z['c_unit'].shape[0]} tokens; the decode needs {n}")
        self.P, self.T, self.keys = P, n, n
        self.c_unit = z["c_unit"][:n].astype(np.float32)
        self.kpe = z["kpe"][:n].astype(np.float32)
        self.pos = np.arange(n)
        x1 = z["x1"][:n].astype(np.float32)
        self.k_o = self.fp.d_shift_for_depth(n)          # the last PV's depth: the P + 1 keys it adds
        self._calibrate(x1, n_calib)
        self.hq = np.stack([self._input(x) for x in x1])
        self._cos_sin()
        self._device_kv()
        self.x1 = x1
        # the device's keys of every position (the parent's rows / c8), and the sweep's tiles
        rep = lambda a: np.repeat(np.asarray(a)[P:P + 1], BR, axis=0)    # noqa: E731
        self.rows_all, self.c8_all = self.rows, self.c8
        self.NT = -(-n // bc)
        K = np.zeros((self.NT * bc, D_QK), dtype=np.int8)
        V = np.zeros((self.NT * bc, D_V), dtype=np.int8)
        K[:P], V[:P] = self.rows_all[:P], self.c8_all[:P]
        K[P:P + PIECE], V[P:P + PIECE] = rep(self.rows_all), rep(self.c8_all)
        self.rows, self.c8 = K, V
        self.hq_all = self.hq
        self.hq, self.cos_rep, self.sin_sgn = rep(self.hq_all), rep(self.cos_rep), rep(self.sin_sgn)
        # the last tile's mask, S^T [bc keys, 16 lanes]: a key after the token's position
        keys = (self.NT - 1) * bc + np.arange(bc)[:, None]
        self.mask_patterns = [np.where(np.broadcast_to(keys > P, (bc, BR)), np.float16(self.hm.MASKED_SCORE),
                                       np.float16(0)).astype(np.float16)]
        self._plans = {0: (self.NT, {(self.NT - 1, 0): 0})}
        self._heads = {}
        with open(os.path.abspath(__file__), "rb") as f:
            src = f.read()
        self._cache_key = hashlib.sha1(repr((self._cache_key, "decode", P, bc, n_calib,
                                             hashlib.sha1(src).hexdigest())).encode()).hexdigest()[:16]

    def attention(self, q8):
        """The sweep of the query tile's lanes over the decode's key tiles, the last one masked after position P."""
        nt, plan = self._plans[0]
        return self._sweep(q8, nt, plan)

    def piece(self, q):
        """Piece q (tokens 16 q .. 16 q + 15) as the sweep reads it: [key rows | value columns] INT8."""
        return self.kv_piece(PIECE * q)

    def report(self, heads_in_order):
        """How far the device's y of the token is from the float attention of the checkpoint."""
        y = self.y16(self.reduce(heads_in_order, 0, 1))[0].astype(np.float64)
        ref = np.asarray(self.y_ref, dtype=np.float64)
        cos = float(y @ ref / max(np.linalg.norm(y) * np.linalg.norm(ref), 1e-30))
        rel = float(np.linalg.norm(y - ref) / max(np.linalg.norm(ref), 1e-30))
        a, b = np.asarray(self.c_unit16, dtype=np.float64).ravel(), self.c_unit.astype(np.float64).ravel()
        ccos = float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-30))
        sc = lambda q: f"{q.s:.4g}"     # noqa: E731
        return [f"DeepSeek-V2-Lite layer 1 MLA attention, one decode token at position {self.P} over keys 0..{self.P} "
                f"(a fresh causal prompt of {self.T} tokens, the token its last; the keys made as the device makes "
                f"them: latent vs the model's cosine {ccos:.5f}, c_fold {self.c_fold:g}), {self.NT} key tiles of "
                f"{self.bc}, {self.H} heads, INT{self.wbits} per column",
                f"D-port shifts k_x={self.k_x} k_uk={self.k_uk} k_uv={self.k_uv} k_s={self.k_s} k_o={self.k_o} "
                f"k_wo={self.k_wo}; scales s_x={sc(self.s_x)} s_qn={sc(self.s_qn)} s_qt={sc(self.s_qt)} "
                f"s_qpe={sc(self.s_qpe)} s_c={sc(self.s_c)} s_ot={sc(self.s_ot)} s_o={sc(self.s_o)}",
                f"device y of the token vs the float attention: cosine {cos:.5f}, relative L2 error {rel:.4f}"]


@contextlib.contextmanager
def _data_as(data):
    """FaBuild's constructor makes its data as fa_prefill_data.FaPrefillData(...): while it runs, that name hands back
    `data`, the decode's (built beforehand), so the base's data, report and fold heads are the decode's."""
    orig = fa_prefill_data.FaPrefillData
    fa_prefill_data.FaPrefillData = lambda *a, **kw: data
    try:
        yield
    finally:
        fa_prefill_data.FaPrefillData = orig


class DecodeTileSource(fa.FreshTileSource):
    """FreshTileSource with a wait per key tile: cached tile j behind its own multicast flag, the last tile behind the
    new piece's (FaDecode.tile_wait), each wait right before the tile's loads on the in-order DM core."""

    def __init__(self, fb, g, nt):
        super().__init__(fb, g, 0, nt)
        self.waits = {}

    def fetch(self, kind, j):
        if j not in self.waits:
            self.waits[j] = self.fb.tile_wait(self.g, j)
        self.wait = self.waits[j]
        return super().fetch(kind, j)


class FaDecode(fs.FaStages):
    TAG = "fd"
    # the switches the decode sets itself: the fresh prefill's stage "full" path, one block from HBM
    FORCED = dict(stage="full", fresh=True, kv_cols=True, x_hbm=True, x_bcast="memchip", bcast="bcast_l1",
                  reduce="chain", sync="edge", y_block_chains=True, stop="full")
    # switches of a multi-chunk prefill (or of an x / key / output path the decode does not take)
    UNSUPPORTED = ("y_halves", "chunk_pipeline", "wo_cols", "o_land", "wo_prefetch", "wo_split_rows", "wo_row_pipe",
                   "y_chain_l3", "y_dst_shared", "store_late", "sym_per_block", "keys_by_chunk", "kx_per_chunk",
                   "kx_after_chains", "kx_mid_double", "kc_xs_late", "kx_ahead", "chains_after_sweep", "x_rest_early",
                   "x_resident", "x_device", "owner_bcast", "kv_balance", "kv_clusters", "kv_wkva_first", "x_pf_split",
                   "x_bcast_after_prefetch", "prefetch_after_x", "s4_early", "s4_split_x", "split_loads", "chunks",
                   "yr_store_tile", "yr_rel1", "post_root_first", "s4_warm_ring")

    def __init__(self, args, p, plat, hw, shared=None):
        p = dict(p)
        for k, v in self.FORCED.items():
            if k in p and p[k] != v:
                raise ValueError(f"params {k}={p[k]!r}: the decode runs {v!r}")
            p[k] = v
        bad = [k for k in self.UNSUPPORTED if p.get(k)]
        if bad:
            raise ValueError(f"params {bad}: not for the decode (one query tile, one block)")
        self.P = int(p.get("cache", 1024))
        bc = int(p.get("bc", 64))
        kv = os.path.join(fa.REPO, p.get("kv_cache", "target/sim/automation/sweep/llm_blocks/fa/cache/"
                                                     "dsv2_l1_kv_8193.npz"))
        wc = os.path.join(fa.REPO, p.get("weights", "target/sim/automation/sweep/common/weights/cache"))
        # the checkpoint FaBuild's data reads (its default model)
        data = FaDecodeData(kv, wc, cache=self.P, bc=bc, n_calib=int(p.get("n_calib", 32)),
                            rope_perm=bool(p.get("rope_perm", False)), mask_c=bool(p.get("mask_c", False)),
                            wbits=int(p.get("wbits", 8)))
        # the base's constructor parses every switch; its shape checks are the prefill's, so it sees the prefill that
        # left the cache (tokens = keys = P, chunks of params prefill_chunk), and _stage applies the decode's shape
        base = dict(p, tokens=self.P, keys=self.P, chunk=int(p.get("prefill_chunk", 128)))
        with _data_as(data):
            super().__init__(args, base, plat, hw, shared=shared)
        # the pieces of the sweep's key tiles (stage_kv_split counts a piece at or past NX as made after the window)
        self.NX = self.NPC
        self.kv_root = self.owner
        self.kvx, self.kv_two, self.kv_half = {}, [], set()

    # ---- the shape and the data ---------------------------------------------------------------------------------
    def _decode_shape(self):
        """The decode's shape in place of the prefill's the base's constructor computed: one query tile of 16 rows on
        the query cluster, one chunk, the decode's key tiles, the token's key owned by kv_owner."""
        d, p = self.d, self.p
        if not isinstance(d, FaDecodeData):
            raise TypeError("FaBuild's constructor no longer makes its data as fa_prefill_data.FaPrefillData (_data_as)")
        print(f"[{self.TAG}] decode: the prefill shape above is the cache's; the step runs one query tile of {BR} rows "
              f"at position {self.P} over {d.NT} key tiles of {self.bc}")
        self.T = 1                                     # one decode token
        self.ROWS, self.NQ, self.NCH = BR, 1, 1
        self.C = self.NC * self.ROWS
        self.NT, self.keys = d.NT, d.T
        self.XB = PIECE * HIDDEN                       # the token's tile of x, INT8, A layout
        self.YB = self.ROWS * HIDDEN * 2               # a cluster's partial y, FP16
        self.NEW = self.P // PIECE                     # the new key's piece
        self.NPC = self.NT * self.bc // PIECE          # the pieces of the sweep's key tiles
        self.qc = int(p.get("q_cluster", 0))
        if self.qc not in range(self.NC):
            raise ValueError(f"params q_cluster={self.qc}: 0..{self.NC - 1}")
        self.kc = (self.qc + 1) % self.NC
        self.owner = int(str(p.get("kv_owner", f"{self.ROOT:02x}")), 16)
        if self.owner not in self.chips:
            raise ValueError(f"params kv_owner={p.get('kv_owner')}: not a compute chip")
        self.kv_tiles = {k: ([self.NEW] if k == self.owner else []) for k in self.chips}
        self.kv_owner = {self.NEW: self.owner}
        self.KVR_TILES = 1
        self.root_ring_head = int(p.get("root_ring_head", self.kva_in_ring() + self.PER_CHUNK_HEAD))

    def kv_cluster(self, k, q):
        # the token's key is the key cluster's work (stage 1 everywhere, stage 2 on the owner)
        return self.kc

    def _stage(self):
        self._decode_shape()
        st, d, T = self.st, self.d, self.TAG
        m0 = self.mem[self.ROOT]
        root_mem = (m0 >> 4, m0 & 15)
        # x: the token's tile in the root memory chip's HBM, every chip's L3 copy filled by the multicast behind a flag
        self.xsrc = d.x_chunk_a(0, PIECE)
        self.h_xsrc = st.put_zeros(f"{T}_xsrc", "int8_t", self.xsrc.size, preload=True)
        self.h_xhbm = st.put_hbm(f"{T}_xhbm", self.xsrc, mem_chip=root_mem)
        st.put_zeros(f"{T}_xtflag", "uint8_t", 64)
        # RoPE tables of the query tile (position P on every row), cos then sin_signed, D layout, a copy a cluster where
        # stage_proj_r reads chunk 0's (2 c ROPE_B)
        self.ROPE_B = self.ROWS * ROPE * 2
        cs, sn = d.rope_tables(0, self.ROWS)
        self.h_rope = st.put(f"{T}_rope", "uint16_t",
                             np.concatenate([cs, sn] * self.NC).astype(np.float16).view(np.uint16))
        # the cache: pieces 0 .. NEW - 1 in the root memory chip's HBM, then the slot the new piece is appended to
        cache = np.concatenate([np.asarray(d.piece(q)).view(np.uint8).reshape(-1) for q in range(self.NEW)]
                               + [np.zeros(KVP, dtype=np.uint8)])
        self.h_cache = st.put_hbm(f"{T}_cache", cache.view(np.int8), mem_chip=root_mem)
        # every chip's L3: the sweep's key tiles (pieces past the new one stay zero, masked), the cached tiles' flags
        # (one a 64-key tile), the new piece's multicast flag, and its flags on the owner (stage_kv_split: at piece q)
        st.put_zeros(f"{T}_kv", "int8_t", self.NPC * KVP, preload=True)
        st.put_zeros(f"{T}_cflag", "uint8_t", (self.NT - 1) * 64)
        st.put_zeros(f"{T}_kvflag", "uint8_t", 64)
        st.put_zeros(f"{T}_kvnflag", "uint8_t", self.NPC * 64)
        if self.p.get("kv_split_clusters", False):
            st.put_zeros(f"{T}_kvpflag", "uint8_t", self.NPC * 64)
        # the token's y in the root memory chip's HBM: FP16 [2048], row major
        self.h_yhbm = st.put_hbm(f"{T}_yhbm", np.zeros(HIDDEN * 2, dtype=np.uint8), mem_chip=root_mem)
        # stage 1: each chip's W_KVA slice in its memory chip's HBM; the owner's gathered latent rows and their flags
        self.h_wkvc = {k: st.put_hbm(f"{T}_wkvc_k{k:02x}", d.kva_cols(16 * self.kvg[k][0], 16 * self.kvg[k][1]),
                                     mem_chip=(self.mem[k] >> 4, self.mem[k] & 15)) for k in self.chips}
        st.put_zeros(f"{T}_kvraw", "uint8_t", self.KVR_TILES * self.KVD)
        st.put_zeros(f"{T}_kvrflag", "uint8_t", self.KVR_TILES * len(self.chips) * 64)
        # stage 2: the owner's W_KVA dequantisation (D layout [16, 576]) and the new tile's RoPE tables (position P)
        kva = d.scale_tile(d.kva_scale()).view(np.uint8)
        o = self.owner
        self.h_kc = {o: st.put_hbm(f"{T}_kc_k{o:02x}", np.concatenate(
            [kva] + [np.asarray(t, dtype=np.float16).view(np.uint8).reshape(-1) for t in (cs, sn)]),
            mem_chip=(self.mem[o] >> 4, self.mem[o] & 15))}
        self.KC_ROPE, self.KC_TILE = kva.size, 2 * PIECE * ROPE * 2
        self.KC_BYTES = kva.size + self.KC_TILE
        # per head: the weights as the ring streams them and the query cluster's dequantisation constants and mask
        # (FaBuild._stage's, for the one query tile)
        self.h_kv, self.h_w, self.h_c, self.h_wkv, self.w_off = {}, {}, {}, {}, {}
        for k in self.chips:
            h = self.head[k]
            loc = (self.mem[k] >> 4, self.mem[k] & 15)
            self.w_off[k], off = [], 0
            chunks = d.weight_chunks(h)
            for nm, b in chunks:
                self.w_off[k].append((nm, off, b.size))
                off += b.size
            self.h_w[k] = st.put_hbm(f"{T}_w_k{k:02x}", np.concatenate([b for _, b in chunks]), mem_chip=loc)
            qs = d.q_scale(h)
            if self.dq_fused:
                sq = np.concatenate([d.scale_tile_a(qs[:fa.Q_NOPE], self.ROWS).view(np.uint8),
                                     np.tile(d.scale_tile(qs[fa.Q_NOPE:]), self.ROWS // 16).view(np.uint8)])
                sqt, suv = d.scale_tile_a(d.qt_scale(h), 16), d.scale_tile_a(d.uv_scale(h), self.ROWS)
            else:
                sq = np.concatenate([np.tile(d.scale_tile(qs[64 * r:64 * (r + 1)]), self.ROWS // 16)
                                     for r in range(3)])
                sqt, suv = d.scale_tile(d.qt_scale(h)), d.scale_tile(d.uv_scale(h))
            masks = d.mask_bias_patterns() if self.mask_c else [m.astype(np.float16) for m in d.mask_patterns]
            const = np.concatenate([sq.view(np.uint8), sqt.view(np.uint8), suv.view(np.uint8),
                                    np.concatenate([m.reshape(-1) for m in masks]).view(np.uint8)])
            self.h_c[k] = st.put_hbm(f"{T}_c_k{k:02x}", const.astype(np.uint8), mem_chip=loc)
            nconst = const.size
        self.SQ_B = (self.ROWS // 16) * 2048
        self.C_SQ, self.C_QT = 0, 3 * self.SQ_B
        self.C_UV = self.C_QT + 16384
        self.C_MASK = self.C_UV + (self.ROWS * fa.Q_NOPE * 2 if self.dq_fused else 4096)
        self.MASK_B = self.bc * BR * (4 if self.mask_c else 2)
        self.C_BYTES = self.C_MASK + len(d.mask_patterns) * self.MASK_B
        if self.C_BYTES != nconst:
            raise ValueError(f"the constants' layout ({self.C_BYTES} B) is not their image ({nconst} B)")

    def _buffers(self):
        """Every cluster's symmetric buffer (the query cluster's holds its partial y), the sweep's geometry, the query
        clusters' constants, the owner's key constants."""
        d, T = self.d, self.TAG
        self.sym_bytes = self.YB
        self.sym_zero = False
        self.sym = {(k, c): self.cx_chip(k).at(c).l1(f"AAAA_{T}_sym", self.sym_bytes)
                    for k in self.chips for c in range(self.NC)}
        self.sweep_cfg = fa.FaSweepCfg(bc=self.bc, nt=self.NT, nq=self.NQ, br=BR, k_s=d.k_s, k_o=d.k_o, a_exp=d.a_exp,
                                       a_n_f32bits=d.a_n_bits)
        self.consts = {g: self.cx_chip(self.chip(g)).at(self.c(g)).l1(f"{T}_consts", self.C_BYTES)
                       for g in range(self.G) if self.c(g) == self.qc}
        self.kvc = {(k, c): self.cx_chip(k).at(c).l1(f"{T}_kvc", self.KC_BYTES)
                    for k in self.chips for c in self.kv_cls(k)}
        self._bufs, self._dst, self._slot, self._root_wait = {}, {}, {}, {}

    # ---- the steps --------------------------------------------------------------------------------------------
    def _preload(self):
        """Before the window, per chip one chain: the owner's key constants, then the query cluster's constants (the
        chain ends on the query cluster; with q_cluster 0, cluster 0, whose DM core on the root holds the start)."""
        pre = {k: [] for k in self.chips}
        self.kvc_pe = {}
        for k in self.chips:
            j = self.chips.index(k)
            if self.kv_split and k == self.owner:
                # kv_split_clusters: the owner's k_pe on the query cluster reads the key constants there too
                self.kvc_pe[(k, self.qc)] = self.cx_chip(k).at(self.qc).l1(f"{self.TAG}_kvcpe", self.KC_BYTES)
            for (kk, c), buf in list(self.kvc.items()) + list(self.kvc_pe.items()):
                if kk == k:
                    g = j * self.NC + c
                    pre[k] = [self.dm_copy(g, f"kvc_ld_g{g}", self.h_kc[k].address, buf, self.KC_BYTES, pre[k])]
            g = j * self.NC + self.qc
            pre[k] = [self.dm_copy(g, f"const_ld_g{g}", self.h_c[k].address, self.consts[g], self.C_BYTES, pre[k])]
        return pre

    def _root_ring_split(self, go):
        """The root's prefetcher in two parts: its first root_ring_head chunks (a task of their own, behind the start)
        and the rest (the base's prefetcher node, so its edges stay). Returns the first part."""
        R, n = self.ROOT, self.root_ring_head
        total = len(self.ring_take[R]) if self.ring_take[R] else self.kva_in_ring() + self.per_chunk()
        if not 0 < n < total or n > self.n_slots:
            raise ValueError(f"params root_ring_head={n}: 1..{min(total - 1, self.n_slots)} (the root's {total} ring "
                             f"chunks, {self.n_slots} slots: the first part never waits for a slot)")
        t, b = self.prefetch[R], self.prefetch[R].kernel_args
        part = lambda lo, hi: HostBingoKernelWeightPrefetchRangeArgs(    # noqa: E731
            b.sched_addr, b.ring_addr, b.flag_addr, b.release_addr, b.seq_addr, b.rec_addr, b.rec_slots, b.n_rings,
            b.n_slots, b.slot_bytes, b.memchip_id, batch=b.batch, policy=b.policy, engine=b.engine,
            batch_routed=b.batch_routed, queued=b.queued, queued_inflight_routed=b.queued_inflight_routed, head=b.head,
            batch_head=b.batch_head, chunk_lo=lo, chunk_hi=hi)
        t.kernel_args = part(n, 0)
        t.kernel_name = t.kernel_args.KERNEL_NAME
        return self.cx_chip(R).host_node(f"weight_prefetch_k{R:02x}_head", t.kernel_name, part(0, n), after=[go[R]])

    def stage_new_key(self, go):
        """Stage 1: each chip's key cluster makes its W_KVA columns of the token's tile (one GEMM over the x tile) and
        puts them into the owner's L3 (kvraw, the D layout's granules of the tile's [16, 576]) behind the chip's flag;
        the owner's key cluster waits for all of them (own_wait: where stage 2 starts)."""
        d, T, dm = self.d, self.TAG, self.roles["dm"]
        q, o, c, lt = self.NEW, self.owner, self.kc, 0
        self.kvc_put = {}
        for j, k in enumerate(self.chips):
            g = j * self.NC + c
            g0, g1 = self.kvg[k]
            nb = (g1 - g0) * TILE16
            l1 = lambda name, size, k=k: self.cx_chip(k).at(c).l1(f"{T}_kc_{name}", size)    # noqa: E731
            xb, ob, wb = l1("x", self.XB), l1("o", nb), l1("wkva", self._src(k, 0)[1])
            w = self.x_wait(k, c, 0, go)
            ldx = self.node(k, c, dm, f"kx_ld_q{q}", "__snax_bingo_kernel_idma_1d_copy",
                            SnaxBingoKernelIdma1dCopyArgs(BingoMemSymbol(f"{T}_xsrc"), xb, self.XB), [w])
            ld, _, _ = self.load(g, 0, "Ld_wkvc", dst=wb)
            gm = self.gmm(g, f"gkvc_q{q}", xb, wb, ob, 1, HIDDEN // 4, g1 - g0, d.k_x, [ld, ldx])
            on = lambda name, off, k=k: BingoMemSymbol(name, offset=off, chip_id=None if o == k else o)   # noqa: E731
            self.kvc_put[(q, k)] = self.node(k, c, dm, f"kvc_put_q{q}", "__snax_bingo_kernel_idma_put_flagged",
                                             SnaxBingoKernelIdmaPutFlaggedArgs(
                                                 ob, on(f"{T}_kvraw", lt * self.KVD + g0 * TILE16), nb,
                                                 on(f"{T}_kvrflag", (lt * len(self.chips) + j) * 64), value=1), [gm])
        puts = [self.kvc_put[(q, k)] for k in self.chips]
        w = self.node(o, c, dm, f"kvr_wait_q{q}", "__snax_bingo_kernel_wait_flags",
                      SnaxBingoKernelWaitFlagsArgs(BingoMemSymbol(f"{T}_kvrflag", offset=lt * len(self.chips) * 64),
                                                   len(self.chips), 1),
                      [t for t in puts if self.flag_dep(t, o, c)])
        for t in puts:
            if not self.flag_dep(t, o, c):
                self.dfg.bingo_add_order_edge(t, w)
        self.own_wait = {q: [w]}

    def stage_new_piece(self, go, cache_mc):
        """Stage 2 on the owner's key cluster (FaStages.stage_kv_split, chained to stage 1's wait), then stage 3: the
        owner's waits for the piece's flag(s) and for the cache multicast's last flag (one multicast source at a time),
        and its multicast of the piece from its L3 into every other chip's."""
        T, dm, o, c, q = self.TAG, self.roles["dm"], self.owner, self.kc, self.NEW
        self.kv_put, self.kvp_put, self._chained = {}, {}, True
        self.stage_kv_split(o)
        prev = []
        flags = [("kvnflag", self.kv_put[q])] + ([("kvpflag", self.kvp_put[q])] if q in self.kvp_put else [])
        for flag, put in flags:
            w = self.node(o, c, dm, f"{flag[:3]}_wait_q{q}", "__snax_bingo_kernel_wait_flags",
                          SnaxBingoKernelWaitFlagsArgs(BingoMemSymbol(f"{T}_{flag}", offset=q * 64), 1, 1),
                          ([put] if self.flag_dep(put, o, c) else []) + prev)
            if not self.flag_dep(put, o, c):
                self.dfg.bingo_add_order_edge(put, w)
            prev = [w]
        self.piece_ready = prev[-1]
        cw = self.node(o, c, dm, "cache_wait", "__snax_bingo_kernel_wait_flags",
                       SnaxBingoKernelWaitFlagsArgs(BingoMemSymbol(f"{T}_cflag", offset=(self.NT - 2) * 64), 1, 1),
                       [go[o]])
        self.dfg.bingo_add_order_edge(cache_mc, cw)
        src = BingoMemSymbol(f"{T}_kv", offset=q * KVP)
        self.kv_mc = [self.node(o, c, dm, f"kv_bput_q{q}", "__snax_bingo_kernel_idma_put_flagged",
                                SnaxBingoKernelIdmaPutFlaggedArgs(src, src, KVP, BingoMemSymbol(f"{T}_kvflag"), value=1,
                                                                  bcast=True), prev + [cw])]

    def tile_wait(self, g, j):
        """The sweep's wait before key tile j on cluster g: a cached tile's multicast flag; the last tile, on the owner
        its piece's flag(s) (the owner's own waits), elsewhere the owner's multicast flag. The multicasts are ordering
        edges; the flags prove the landing."""
        k, c, dm, T = self.chip(g), self.c(g), self.roles["dm"], self.TAG
        if j < self.NT - 1:
            w = self.node(k, c, dm, f"c_wait_j{j}", "__snax_bingo_kernel_wait_flags",
                          SnaxBingoKernelWaitFlagsArgs(BingoMemSymbol(f"{T}_cflag", offset=j * 64), 1, 1),
                          [self.go[k]])
            self.dfg.bingo_add_order_edge(self.cache_mc, w)
            return w
        if k == self.owner:
            return self.piece_ready
        w = self.node(k, c, dm, "kv_wait_new", "__snax_bingo_kernel_wait_flags",
                      SnaxBingoKernelWaitFlagsArgs(BingoMemSymbol(f"{T}_kvflag"), 1, 1), [])
        self.dfg.bingo_add_order_edge(self.kv_mc[0], w)
        return w

    def stage_store(self, i, red, half=None, clusters=None):
        """The collector's store of the token's y: row 0 of the reduced D-layout block (every row is the token's), a
        32-byte run a 16-column block, into the root memory chip's HBM (y_store_halves: one W_O block's half)."""
        R, dm = self.y_root(i, half), self.roles["dm"]
        nblk = (HIDDEN // 16) // (1 if half is None else 2)
        off = 0 if half is None else half * nblk * 32
        st = {}
        for c in (list(clusters) if clusters is not None else range(self.NC)):
            deps = list(red[c]) if isinstance(red[c], (list, tuple)) else [red[c]]
            st[c] = self.node(R, c, dm, f"y_store_c{c}_i{i}" + ("" if half is None else f"_h{half}"),
                              "__snax_bingo_kernel_idma_2d_copy",
                              SnaxBingoKernelIdma2dCopyArgs(self.dst(i, c, half), self.h_yhbm.address + off, 32, TILE16,
                                                            32, nblk), deps)
            self._y_stores.append(st[c])
        return st

    def x_multicast(self, go, bcast):
        """Stage 0's multicast of x into every chip's L3 behind its flag (fd_xsrc, fd_xtflag): the root memory chip's push
        engine from its HBM (`bcast`, build's host task maker). Returns the task the x waits are ordered after. A whole
        layer (dsv2/common/dsv2_decode_mc.py) multicasts its input norm's tile here instead."""
        T = self.TAG
        return bcast("x_hbm_bcast", self.h_xhbm.address, f"{T}_xsrc", f"{T}_xtflag", self.XB, 1, [go[self.ROOT]])

    def kv_append(self, deps):
        """The new piece from the owner's L3 into the cache's next slot in HBM (piece P / 16), on the owner's key cluster
        after `deps`. Returns the task. A whole layer (dsv2/common/dsv2_decode_mc.py) makes it at its own end instead."""
        T = self.TAG
        return self.node(self.owner, self.kc, self.roles["dm"], f"kv_append_q{self.NEW}",
                         "__snax_bingo_kernel_idma_1d_copy",
                         SnaxBingoKernelIdma1dCopyArgs(BingoMemSymbol(f"{T}_kv", offset=self.NEW * KVP),
                                                       self.h_cache.address + self.NEW * KVP, KVP), deps)

    # ---- the build ------------------------------------------------------------------------------------------------
    def build(self):
        d, R, T = self.d, self.ROOT, self.TAG
        if self.kv_slabs not in ("split", "split4", "split_kb"):
            raise ValueError("the decode's sweep: kv_slabs split, split4 or split_kb (FreshTileSource's slabs)")
        self.last_ld, self._x_extra, self._y_stores = {}, {}, []
        for ls in self.ls:
            ls.strict = True
        # the ring's chunk 0 (W_KVA) is the key cluster's alone, the rest the query cluster's: each is released by the
        # cluster that reads it
        n_ring = self.kva_in_ring() + self.per_chunk()
        for k in self.chips:
            if self.kc != 0:
                self.ring_rel1.add((k, 0))
            if self.qc != 0:
                self.ring_rel1 |= {(k, n) for n in range(1, n_ring)}
        if self.n_slots < n_ring:
            raise ValueError(f"weight_ring slots {self.n_slots}: the decode keeps a chip's {n_ring} ring chunks "
                             f"at once")
        go = self.setup(extra=self._preload())
        if self.p.get("relay_per_gather", True):
            self.dfg.relay_one_consumer_cores = [self.roles["xdma"]]
        if self.p.get("relay_prefer"):
            self.dfg.relay_prefer_cores = {self.roles[c]: [self.roles[r] for r in rs]
                                           for c, rs in self.p["relay_prefer"].items()}
        # stage 0 and the cache on the root's engine of its memory chip, the root's ring split around the cache
        m = self.mem[R]
        bcast = lambda name, src, dst, flag, nb, n, after: self.cx_chip(R).host_node(    # noqa: E731
            f"{name}_k{R:02x}", "__host_bingo_kernel_hbm_bcast",
            HostBingoKernelHbmBcastArgs(src, BingoMemSymbol(dst), BingoMemSymbol(flag), self.h_seq[m].address + 64, 1,
                                        nb, n, m, self.engine[R]), after=after)
        xm = self.x_multicast(go, bcast)
        self.x_puts = [xm]
        head = self._root_ring_split(go)
        self.edge(xm, head)
        self.cache_mc = bcast("cache_hbm_bcast", self.h_cache.address, f"{T}_kv", f"{T}_cflag", TILE_PIECES * KVP,
                              self.NT - 1, [head])
        self.edge(self.cache_mc, self.prefetch[R])
        # stage 1 (every key cluster; the ring's chunk 0 first on every chip)
        self.stage_new_key(go)
        # stage 4 (every query cluster)
        q8, dm = {}, self.roles["dm"]
        gq = [j * self.NC + self.qc for j in range(len(self.chips))]
        for g in gq:
            k, c = self.chip(g), self.c(g)
            w = self.x_wait(k, c, 0, go)
            x = self.node(k, c, dm, "x_copy_i0", "__snax_bingo_kernel_idma_1d_copy",
                          SnaxBingoKernelIdma1dCopyArgs(BingoMemSymbol(f"{T}_xsrc"), self.bufs(g, 0)["xrows"], self.XB),
                          [go[k], w])
            q8[g] = self.proj_of(0)(g, 0, x)
        # stages 2 and 3 on the owner's key cluster
        self.stage_new_piece(go, self.cache_mc)
        # stage 5 (every query cluster)
        nt, plan = d._plans[0]
        early = 2 if len(self.ls[gq[0]].slabs) >= 4 else 1
        self.sw_of = {}
        for g in gq:
            k, c = self.chip(g), self.c(g)
            self.sw_of[g] = fa.fa_sweep(self.at(k, c), fa.dataclasses.replace(self.sweep_cfg, nt=nt),
                                        self.bufs(g, 0)["swp"], DecodeTileSource(self, g, nt), q8[g], self.roles,
                                        name="s0_", early=early, mask=self.mask_of(g, plan), mask_c=self.mask_c,
                                        ot8=self.ot8_bits(), sm_batch=self.sm_batch, prologue=self.sm_prologue,
                                        geom_core=self.sm_geom)
        # stages 6 and 7
        for g in gq:
            post = self.stage_post_a(g, 0, self.sw_of[g])
            self.stage_post_b(g, 0, post[1], [])
        # stage 8: a chain a W_O block into the collector, each block's row of y into HBM
        red, stores, sym_free = {}, {}, {}
        self._block_chains(0, [self.qc], {self.qc: []}, None, red, stores, sym_free)
        # every y store behind the last chain too: no write into a memory chip beside a chain
        for st in self._y_stores:
            if not self.dfg.has_edge(self._chains[-1], st):
                self.edge(self._chains[-1], st)
        # the append: the new piece from the owner's L3 into the cache's next slot in HBM, after the y stores
        self.append = self.kv_append(self.kv_mc + list(self._y_stores))
        self.link_report()
        self.end_window(list(self._y_stores) + [self.append])
        self.decode_checks()

    # ---- checks -------------------------------------------------------------------------------------------------
    def decode_checks(self):
        """The token's y and the appended piece in HBM (copied into the root's L3 after the window), the cache's and the
        new piece's landing in the L3 of the chip in the middle of the fold order; checks "stages": that chip's Q8 and
        o~."""
        T, R, d, n = self.TAG, self.ROOT, self.d, 2048
        checks = []
        far = self.order[len(self.order) // 2]
        on = lambda k: None if k == R else k     # noqa: E731
        wins = (("c0a", 0, 0), ("cLb", self.NEW - 1, KVP - n), ("nka", self.NEW, 0), ("nvb", self.NEW, KVP - n))
        for nm, q, lo in wins:
            img = np.asarray(d.piece(q)).view(np.uint8).reshape(-1)
            checks.append((f"kv_{nm}", BingoMemSymbol(f"{T}_kv", offset=q * KVP + lo, chip_id=on(far)),
                           img[lo:lo + n], n))
        new = np.asarray(d.piece(self.NEW)).view(np.uint8).reshape(-1)
        for nm, lo in (("a", 0), ("b", KVP - n)):
            got = self.stash(f"app_{nm}", R, 0, self.h_cache.address + self.NEW * KVP + lo, n, [self.append])
            checks.append((f"app_{nm}", got, new[lo:lo + n], n))
        half = HIDDEN // 2
        for h in range(2):
            fold = self.y_fold_heads(self.y_root(0, h)) if self.y_roots else self.fold_heads
            y = np.asarray(self.y_raw(0, BR, fold)[0, h * half:(h + 1) * half], dtype=np.float16).view(np.uint8)
            got = self.stash(f"y_h{h}", R, 0, self.h_yhbm.address + h * half * 2, half * 2, list(self._y_stores))
            checks.append((f"y_h{h}", got, y, half * 2))
        if self.checks_kind == "stages":
            g = self.chips.index(far) * self.NC + self.qc
            hd = self.head[far]
            checks.insert(0, ("q8", self.stash("q8", far, self.qc, self.bufs(g, 0)["q8"][0], D_QK * BR, []),
                              d.q8_a(hd, 0).view(np.uint8), D_QK * BR))
            ot, nb = self.xt_ref(hd, 0)
            checks.insert(1, ("xt", self.stash("xt", far, self.qc, self.sw_of[g]["xt"][0], nb, []), ot, nb))
        self.run_checks(checks)

    def finish(self, app_name, **kw):
        self._check_collectives()
        super().finish(app_name, **kw)

    def _check_collectives(self):
        """Every two multicasts (x's and the cache's from the memory chip, the owner's of the new piece), every
        multicast and reduce chain, and every write into HBM after every multicast and chain, ordered by the graph
        (dependencies and flag edges): concurrent multicast sources, or a multicast beside a chain, deadlock the
        fabric. Raises otherwise."""
        import networkx as nx
        gr = nx.DiGraph(self.dfg)
        gr.add_edges_from(self.dfg.order_edges)
        nodes = self.dfg.node_list
        mc = [t for t in nodes if (t.kernel_name or "").endswith("hbm_bcast")
              or ((t.kernel_name or "").endswith("idma_put_flagged") and getattr(t.kernel_args, "bcast", 0))]
        ch = [t for t in nodes if (t.kernel_name or "").endswith("xdma_chain_gather")]
        desc = {t: nx.descendants(gr, t) for t in mc + ch}
        coll = mc + ch
        bad = [(a.node_name, b.node_name) for i, a in enumerate(coll) for b in coll[i + 1:]
               if (a in mc or b in mc) and b not in desc[a] and a not in desc[b]]
        if bad:
            raise ValueError(f"the decode: {len(bad)} multicast / chain pair(s) may run at once: {bad[:6]}")
        hbm = list(self._y_stores) + [self.append]
        late = [(a.node_name, w.node_name) for a in coll for w in hbm if w not in desc[a]]
        if late:
            raise ValueError(f"the decode: {len(late)} HBM write(s) not after a multicast or chain: {late[:6]}")
        print(f"[{self.TAG}] decode: {len(mc)} multicasts and {len(ch)} chains, each ordered with every multicast; "
              f"{len(hbm)} HBM writes after all of them")

    # ---- the links' bytes -----------------------------------------------------------------------------------------
    def link_report(self):
        """The bytes each D2D link carries in the step (both directions: a link is half duplex): every chip's ring
        pushes (from its memory chip, X then Y as the routers route), the memory chip's multicasts of x and the cache
        and the owner's of the new piece (the routers' broadcast tree), the slices to the owner, the reduce chains'
        hops, the y rows and the append. Prints the busiest links."""
        mems = sorted(set(self.mem.values()))
        grid = set(self.chips) | set(mems)
        load, kind = {}, {}

        def add(links, nb, what):
            for ln in links:
                load[ln] = load.get(ln, 0) + nb
                kind.setdefault(ln, {}).setdefault(what, 0)
                kind[ln][what] += nb

        for k in self.chips:
            add(_route(self.mem[k], k), sum(t[6] for t in self.ring_take[k]), "ring")
        add(_bcast_tree(self.mem[self.ROOT], grid), self.XB, "x")
        add(_bcast_tree(self.mem[self.ROOT], grid), (self.NT - 1) * TILE_PIECES * KVP, "cache")
        add(_bcast_tree(self.owner, grid), KVP, "piece")
        for k in self.chips:
            add(_route(k, self.owner), (self.kvg[k][1] - self.kvg[k][0]) * TILE16, "slices")
        for h in range(2):
            R = self.y_root(0, h)
            hops = self.y_order(R)[1:] + [R]
            for a, b in zip(hops, hops[1:]):
                add(_route(a, b), self.YB // 2, "chains")
            add(_route(R, self.mem[self.ROOT]), HIDDEN, "y")
        add(_route(self.owner, self.mem[self.ROOT]), KVP, "append")
        top = sorted(load, key=lambda ln: -load[ln])[:4]
        wts = {k: sum(t[6] for t in self.ring_take[k]) for k in self.chips}
        print(f"[{self.TAG}] decode streams: ring {sum(wts.values()):,} B over {len(self.chips)} chips "
              f"({min(wts.values()):,}..{max(wts.values()):,} B a chip), cache {(self.NT - 1) * TILE_PIECES * KVP:,} B "
              f"to every chip in one multicast")
        for ln in top:
            parts = ", ".join(f"{w} {b:,}" for w, b in sorted(kind[ln].items(), key=lambda x: -x[1]))
            print(f"[{self.TAG}] link {ln[0]:02x}-{ln[1]:02x}: {load[ln]:,} B ({parts})")
        self.busiest = (top[0], load[top[0]], dict(kind[top[0]]))


def _route(a, b):
    """The D2D links a unicast from chip a to chip b crosses: X first, then Y (hemaia_d2d_link_router_rc)."""
    x, y, bx, by, out = a >> 4, a & 15, b >> 4, b & 15, []
    while x != bx:
        nx_ = x + (1 if bx > x else -1)
        out.append(tuple(sorted(((x << 4) | y, (nx_ << 4) | y))))
        x = nx_
    while y != by:
        ny = y + (1 if by > y else -1)
        out.append(tuple(sorted(((x << 4) | y, (x << 4) | ny))))
        y = ny
    return out


def _bcast_tree(src, grid):
    """The D2D links a broadcast (chip id 0xFF) from `src` crosses, by hemaia_d2d_link_router_rc's rules: the source
    sends every way; a chip above it in its column north, east and west; a chip in its row away from it and south; a
    chip above its row away from its column; a chip below its row south."""
    sx, sy, out = src >> 4, src & 15, []
    N, S, E, W = (0, -1), (0, 1), (1, 0), (-1, 0)

    def dirs(x, y):
        if (x, y) == (sx, sy):
            return (N, S, E, W)
        if x == sx and y < sy:
            return (N, E, W)
        if y == sy:
            return ((W,) if x < sx else (E,)) + (S,)
        if y < sy:
            return (W,) if x < sx else (E,)
        return (S,)

    stack = [(sx, sy)]
    while stack:
        x, y = stack.pop()
        for dx, dy in dirs(x, y):
            nx_, ny = x + dx, y + dy
            if nx_ < 0 or ny < 0 or ((nx_ << 4) | ny) not in grid:
                continue
            out.append(tuple(sorted(((x << 4) | y, (nx_ << 4) | ny))))
            stack.append((nx_, ny))
    return out


def _app_name(S):
    return (f"DeepSeek-V2-Lite layer 1 MLA attention decode, one token at position {S.P} over {S.P} cached keys, "
            f"{len(S.chips)} chips x {S.NC} clusters, x and the cache from HBM, y into HBM")


def run():
    run_driver("fa decode workload generator", FaDecode, _app_name)
