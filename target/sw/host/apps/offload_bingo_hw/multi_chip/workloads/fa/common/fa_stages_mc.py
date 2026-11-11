# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""The DSv2-Lite MLA fresh prefill built stage by stage (workloads/fa/dsv2_l1_prefill/stage<N>/, params stage).

Each stage is the one before it plus one step of the mapping (the FA Prefill Tracker's "The decomposition, stage by
stage"), so every step is measured and optimised on its own before the next one is added. The data, the platform, the
rings and the helpers are fa_prefill_mc.FaBuild's; this builder replaces its build().

stage 0  X from HBM: the prompt's rows sit in the HBM of the root's memory chip (params x_hbm); the root's DM core
         (cluster x_put_cluster) multicasts them a tile of 16 tokens (32 KiB) at a time into every chip's L3 through the
         D2D routers, one multicast in flight, each tile behind a landed flag of its own. Every cluster waits for the
         tiles stage 1 reads (cluster c: the tiles t with t mod 2 = c). params x_bcast "memchip": the memory chip's
         push engine multicasts the tiles itself, from its HBM, configured by the root's host (one host task,
         __host_bingo_kernel_hbm_bcast); the root then gets its copy from the multicast like every other chip. The
         root's prefetcher shares that engine: it runs after the multicast, or before it with x_bcast_after_prefetch.
stage 1  + the keys by columns: each cluster's tiles of X from the chip's L3 into L1 behind the tile's flag, split between
         the iDMA and the xDMA (params split_loads, the xDMA's share split_frac), which reach L3 through two different
         wide ports; the chip's W_KVA columns from the ring; a GEMM a tile; each slice into the tile owner's L3 behind a
         flag; every owner waits for its tile's 16 slices. For bisecting the xDMA's loads: split_clusters (the clusters
         that split), split_frac 1 (the xDMA alone), split_after_bcast (the keys start once every tile has landed).
         params kv_balance (INT4): the two chips with two pairs of W_KVA's columns hand their second pair to helper
         chips for all but the first of kv_balance_ways runs of tiles (kv_helpers). params kv_wkva_first: each cluster
         copies W_KVA into L1 before its first tile of x.
stage 2  the owner finishes a key tile, alone (params kv_split_clusters: c on the tile's cluster, k_pe on the other; kv_norm_fused: RMSNorm writes the INT8 key rows' c and value columns itself): stage 1's result is in place before the window (each owner's gathered
         latent rows in its L3 kvraw, every slice's flag set, copied from HBM like the constants), and the window runs
         the production code (fa_prefill_mc stage_kv_on): kvr_wait, the copy into L1, dequantisation, RMSNorm of c, the
         key rows' and value columns' A layouts and INT8, RoPE of k_pe, and the put of the 17 KB piece into the root's
         L3; it ends when the root has every piece (stage 3's input).
stage 3  the keys to every chip, alone: each owner's 17 KB piece (key rows + value columns, INT8, A layout) in its L1
         before the window, as stage 2 leaves it; the window ends when every chip holds all the pieces in its L3 (stage
         4's input). params s3_route "gather" (production): each owner puts its piece into the root's L3 behind a flag,
         the root multicasts a chunk's pieces at a time once they have landed (stage_kv_bcast). "ring": the owners
         multicast their own pieces in tile order, each behind the landed flag of the one before (one multicast in
         flight: several at once deadlock the D2D routers), so no piece goes through the root.
stage "0123"  stages 0 to 3 chained, as they run in the layer: stage 1's owner waits start stage 2 (no stage 1 result
         from HBM), stage 2's pieces gather on the root, which multicasts a chunk at a time once both its flags are in
         (stage 3); the window ends when every chip has every piece. Its outputs (x and the keys in every chip's L3) are
         what the stages after it start from.
stage 4  the queries of the head, alone: x resident in every chip's L3 (stage 0's output) and each cluster's constants
         before the window; per chunk and cluster, the cluster's rows of x into L1 (stage_x), then q = x W_q, q~ =
         q_nope W_UK and RoPE on q_pe into the four query tiles Q8 (stage_proj, production code); the ring holds only
         the head's weights (W_q x 3, W_UK a chunk). Checks: the first and the last query tile of both clusters of the
         chip in the middle of the fold order, the last chunk.
stage "full"  the whole block: stages 0 to 3 as here (X from HBM), then FaBuild's chunks (stage 4 from the rows the
         multicast lands, the sweep, W_UV, W_O, the reduce chains and the stores; build_full, FaBuild.build with these
         swaps); stage 4's params (s4_reorder, s4_split_x) apply. Checks: FaBuild's (y windows, keys on the root).
stage "678"  W_UV, W_O and the reduce, alone (stages 6, 7 and 8: they hand off through the symmetric buffer, chunk i's
         W_O behind chunk i-1's reduce): every cluster's o~ tiles of both chunks (the sweep's output, the reference's) in
         its L1 before the window, then FaBuild's chunks from W_UV on (build_full without stages 0 to 5); the ring holds
         W_UV and W_O only. Checks: FaBuild's y windows at the root.
stage 5  the sweep, alone: before the window every cluster's query tiles Q8 of both chunks in its L1 (stage 4's
         output, the reference's) and every piece of the keys with its landed flags in every chip's L3 (stage 3's);
         per chunk and cluster FaBuild's sweep (fa_sweep over FreshTileSource: QK with the causal masks, the online
         softmax, PV, the normalisation and the transpose into o~ FP16). Checks: o~ of the first and the last query
         tile of both clusters of the chip in the middle of the fold order, the last chunk.

The checks: windows of X in the L3 of the root, of the chip in the middle of the fold order and of the last one; from
stage 1 also the head and the tail of three tiles' latent rows [16, 576] (FP16, D layout) in their owners' L3.
"""

import copy
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fa_prefill_mc as fa  # noqa: E402
from fa_prefill_mc import HIDDEN, KVP, TILE16  # noqa: E402
from bingo_kernel_args import (  # noqa: E402
    HostBingoKernelHbmBcastArgs, SnaxBingoKernelIdma1dCopyArgs, SnaxBingoKernelIdmaPutFlaggedArgs,
    SnaxBingoKernelWaitFlagsArgs, SnaxBingoKernelXdma1dCopyArgs)
from bingo_mem_handle import BingoMemSymbol  # noqa: E402
from driver import run_driver  # noqa: E402

STAGES = (0, 1, 2, 3, "0123", 4, 5, "678", "full", "xu")


class FaStages(fa.FaBuild):
    """FaBuild's block stage by stage (the module's docstring), in FaBuild's four steps: __init__ (the parameters),
    mem_handles() (FaBuild's, whose _stage here adds each stage's inputs), build() and finish()."""

    def __init__(self, args, p, plat, hw, shared=None):
        """The parameters; no data, memory or task yet: FaBuild's (x from HBM by default), then the stage and its
        switches with their checks."""
        p = dict(p)
        p.setdefault("x_hbm", True)
        super().__init__(args, p, plat, hw, shared=shared)
        self.stage = p.get("stage", 0)
        if self.stage not in STAGES:
            raise ValueError(f"params stage={self.stage}: one of {STAGES}")
        # params stage "xu", a unit test of the xDMA's load L3 -> L1: xu_bytes from the chip's L3 into L1 by the xDMA on
        # each (chip, cluster) of xu_chips x xu_clusters, with nothing else in flight (no multicast, no ring)
        self.xu_bytes = int(p.get("xu_bytes", 16384))
        self.xu_chips = [int(str(k), 16) for k in p.get("xu_chips", ["00"])]
        self.xu_clusters = [int(c) for c in p.get("xu_clusters", [0])]
        if not self.x_hbm and not (self.stage in (4, 5, "678") and self.x_resident):
            # stage 4 starts after stage 0: x already in every chip's L3 (x_resident)
            raise ValueError("the stage workloads take x from HBM (params x_hbm); stage 4 x_resident")
        self.XT = 16 * HIDDEN                          # a tile of x: 16 tokens, A layout, INT8
        self.NX = self.T // 16
        # params split_loads (stage 1): every tile of x into L1 by the iDMA and the xDMA at once, the xDMA's part
        # split_frac of it (whole 512-byte blocks; 1 = the xDMA alone), on the clusters in split_clusters
        self.split = bool(p.get("split_loads", False))
        xb = int(round(self.XT * float(p.get("split_frac", 0.5)) / 512)) * 512
        if self.split and not 0 < xb <= self.XT:
            raise ValueError(f"params split_frac={p.get('split_frac')}: the xDMA's part of a {self.XT}-byte tile")
        self.x_xdma = xb if self.split else 0
        self.split_cls = [int(c) for c in p.get("split_clusters", range(self.NC))]
        # params split_after_bcast (stage 1): a cluster's keys start once every tile of x has landed in its chip's L3,
        # so no load of it runs while the multicast still writes there
        self.after_bcast = bool(p.get("split_after_bcast", False))
        # params x_bcast: who multicasts x, "root" (the root cluster x_put_cluster's iDMA, also into its own L3) or
        # "memchip" (the root's memory chip's push engine, from its HBM, configured by the root's host)
        self.x_bcast = p.get("x_bcast", "root")
        if self.x_bcast not in ("root", "memchip"):
            raise ValueError(f"params x_bcast={self.x_bcast}: root or memchip")
        self.x_mem = self.x_bcast == "memchip"
        # params x_bcast_after_prefetch (memchip): the root's prefetcher, which drives the same engine, pushes the root's
        # weights first and the multicast follows, instead of the multicast first
        self.bcast_after_pf = bool(p.get("x_bcast_after_prefetch", False))
        if self.bcast_after_pf and not self.x_mem:
            raise ValueError("params x_bcast_after_prefetch: needs x_bcast memchip")
        # params x_pf_split (memchip): the root's prefetcher in two parts on its host, its W_KVA chunks before the
        # multicast and the rest after it (HostBingoKernelWeightPrefetchRangeArgs), so the root's keys do not wait
        # for the multicast and the multicast does not wait for the root's stage-4 weights
        self.pf_split = bool(p.get("x_pf_split", False))
        if self.pf_split and (not self.x_mem or self.bcast_after_pf):
            raise ValueError("params x_pf_split: needs x_bcast memchip, without x_bcast_after_prefetch")
        # params kv_wkva_first (stage 1): each cluster copies the chip's W_KVA into L1 before its first tile of x, for
        # when W_KVA is in L3 before x's first tiles land
        self.wkva_first = bool(p.get("kv_wkva_first", False))
        # params kv_split_clusters (stage 2): the owner's tile on both its clusters, c on the tile's cluster and k_pe on
        # the other (stage_kv_split)
        self.kv_split = bool(p.get("kv_split_clusters", False))
        # params kv_norm_fused (stage 2, stage_kv_split): c's RMSNorm writes the key rows' c (row_major -> A, INT8) and,
        # from c transposed, the value columns (col_major -> B, INT8: B of c [16, 512] is A of c^T) itself
        self.kv_fused = bool(p.get("kv_norm_fused", False))
        # params kv_vals_other (with both): the value columns' RMSNorm on the other cluster too, from c transposed into
        # its L1 by the tile's cluster's xDMA; that cluster puts k_pe and the value columns (adjacent in the slot)
        self.kv_vals_other = bool(p.get("kv_vals_other", False)) and self.kv_split and self.kv_fused
        # params kv_vals_own (with kv_vals_other): that cluster builds c^T itself (its own load and dequantisation of c,
        # D -> row major and the transpose on its own xDMA), so nothing crosses between the clusters
        self.kv_vals_own = bool(p.get("kv_vals_own", False)) and self.kv_vals_other
        # params kv_norm_cm (stage 2, stage_kv_split): c transposed by the xDMA first, so c's RMSNorm runs on the
        # col_major tile (one token a lane: no fold, no plane); the value columns come from its output (c^T, row major
        # [512, 16]) by the reshape and quantisation of before, the key rows' c by a transpose back first
        self.kv_norm_cm = bool(p.get("kv_norm_cm", False))
        # params s4_split_x (stage 4): the xDMA's share of each cluster's rows of x (L3 -> L1), the iDMA the rest (whole
        # 512-byte blocks; 0 = the iDMA alone); s4_overlap: a chunk's rows land without waiting for the previous chunk's
        # projection (each chunk has its own rows buffer)
        self.s4_split_x = float(p.get("s4_split_x", 0.0))
        self.s4_overlap = bool(p.get("s4_overlap", False))
        self._first_ld = set()
        # params chunks (stage "full"): only the first `chunks` token chunks' queries, sweep and output are made (a proof
        # of concept of a long prompt in a short window); stages 0-3 still make every token's keys
        self.chunks_built = p.get("chunks")
        # params chains_after_sweep (stage "full", reduce_early): the root clusters' chains and stores of chunk i - 1 are
        # made after their sweep of chunk i, so the root's in-order xDMA and DM cores take the sweep's tasks (its softmax
        # geometry, its key and value tiles) first and the chains run under the sweep's computation
        self.chains_after_sweep = bool(p.get("chains_after_sweep", False))
        # params yr_store_tile t (y_roots): a collector cluster's store of chunk i's y block goes behind its own load of
        # chunk i + 1's key tile t's values in its DM core's program. The chains and stores are made while the root's
        # tasks are, so on a collector they would sit ahead of its chunk i + 1 front, which would wait for the chain
        self.yr_store_tile = p.get("yr_store_tile")
        # params kx_mid_double (keys_by_chunk): the key tiles of the later chunks made inside the window take two x and two
        # output buffers in turn, like chunk 0's, so a tile's load and GEMM wait for the tile two before, not the one before
        self.kx_mid_double = bool(p.get("kx_mid_double", False))
        # params kc_xs_late (keys_by_chunk): a helping chip's tiles after the window take a copy of the helped pair's W_KVA
        # of their own, as its own tiles do of W_KVA, so the window's copy is free after the window's last helped tile
        self.kc_xs_late = bool(p.get("kc_xs_late", False))
        # params kx_per_chunk (keys_by_chunk): the key tiles of every chunk after the first take x and output buffers,
        # a W_KVA copy and a helped-pair copy of that chunk's own (both copies straight from HBM), and the loads that
        # start them depend on the cluster's last q projection of the chunk before, so nothing of the key loop lives
        # from chunk to chunk and the static L1 planner can lay each chunk's over the chunk before's x rows. From three
        # chunks on, one shared set beside a collector's held y block and the next chunk's x rows overflows its L1
        self.kx_per_chunk = bool(p.get("kx_per_chunk", False))
        # params kx_after_chains (kx_per_chunk, y_roots, yr_store_tile): on every cluster, chunk h's key tasks (h >= 2)
        # behind its load of chunk h - 1's key tile yr_store_tile's values (as a collector's, behind its store), and
        # their first loads behind the last reduce chain of chunk h - 2, so the key loop's transfers (its HBM copies,
        # its puts to the owners) never run beside a chain: a chain must not meet another chip's traffic on a hop
        self.kx_after_chains = bool(p.get("kx_after_chains", False))
        # params kx_l3 (kx_per_chunk, default true): the per-chunk copies of W_KVA, the helped pair and the key
        # constants from the chip's own L3 (filled from HBM before the window); false: straight from HBM
        self.kx_l3 = bool(p.get("kx_l3", True))
        # params yr_rel1 (y_roots): on each chip that collected chunk i - 1, chunk i's ring chunks are released by cluster
        # 1's read: a collector's cluster 1 runs its chain and store last, so it reads them last, and cluster 0's read,
        # the default releaser, would wait for it
        self.yr_rel1 = bool(p.get("yr_rel1", False))
        # params keys_by_chunk (stage "full", x_pf_split, full_kv_all false): each chunk's keys just ahead of its queries
        # instead of every key first. On every device core, the key tasks (stages 1-3) of chunk h follow the core's
        # cluster's stage-4 tasks of chunk h - 1; those of the chunks after the last one made, everything of the window.
        # The root's host multicasts chunk 0's x, then pushes chunk 0's stage-4 weights, then multicasts the rest of x,
        # then pushes the rest of the weights
        self.keys_by_chunk = bool(p.get("keys_by_chunk", False))
        if self.keys_by_chunk and (self.stage != "full" or not self.pf_split or p.get("full_kv_all", True)):
            raise ValueError("params keys_by_chunk: stage full, with x_pf_split and full_kv_all false")
        self._key_runs = []     # keys_by_chunk: (first node id, key chunk or None) of each run of key tasks, in order
        # params x_rest_early (keys_by_chunk): the rest of x (chunks 2 on) right after chunk 0's key multicast, ahead of
        # the root's prefetcher, not behind it. The prefetcher is one host task that streams every chunk's weights, so
        # behind it the rest lands only once the chips have consumed the last chunk's weights, which needs chunk 2's
        # keys and so its x: a full block's cycle. Only the root chip's own ring waits for the rest; chunk 1's key
        # multicast and every reduce chain follow it (one multicast source at a time)
        self.x_rest_early = bool(p.get("x_rest_early", False))
        if self.x_rest_early and not self.keys_by_chunk:
            raise ValueError("params x_rest_early: needs keys_by_chunk")
        if self.kx_per_chunk and not self.keys_by_chunk:
            raise ValueError("params kx_per_chunk: needs keys_by_chunk")
        if self.kx_after_chains and not (self.kx_per_chunk and p.get("y_roots") and p.get("yr_store_tile") is not None):
            raise ValueError("params kx_after_chains: needs kx_per_chunk, y_roots and yr_store_tile")
        # params s4_reorder (stage 4): stage_proj_r's order of the SIMD core's tasks
        self.s4_reorder = bool(p.get("s4_reorder", False))
        # params s4_tile_absorb (stage 4, with s4_reorder): the absorb GEMM a query tile at a time, each tile's q~
        # relayout and quantisation behind its own GEMM (they overlap the next tiles' GEMMs)
        self.s4_tile_absorb = bool(p.get("s4_tile_absorb", False))
        # params s4_early (stage "full"): the cluster that runs stage 3 (s3_root's x_put_cluster) releases its chip's
        # W_q / W_UK ring slots, so the other cluster there does not wait for its reads
        self.s4_early = bool(p.get("s4_early", False))
        # params s4_qpe_first (stage 4, with s4_tile_absorb and dq_fused): chunks after the first run q_pe's chain first
        self.s4_qpe_first = bool(p.get("s4_qpe_first", False))
        # params s4_q8_pairs n (stage 4, s4_reorder with s4_tile_absorb): q_pe's relayout ahead of q~ tile n's on the
        # in-order xDMA core (the RoPE ends before absorb n does), and the SIMD core quantises tile by tile, q~ then q_pe,
        # so the sweep's first QK waits for one tile's two quantisations, not every tile's q_pe then q~ tile 0
        self.s4_q8_pairs = p.get("s4_q8_pairs")
        # params whatif_q8a_free (cluster replays only, timing; the data is wrong): no q~ quantisation task, each query
        # tile's QK waits for q~'s relayout instead; the bound of moving that quantisation off the SIMD core
        self.whatif_q8a_free = bool(p.get("whatif_q8a_free", False))
        if self.whatif_q8a_free and p.get("replay_cluster") is None:
            raise ValueError("params whatif_q8a_free: timing only, for a cluster replay (replay_cluster)")
        if self.s4_q8_pairs is not None and not (self.s4_tile_absorb and 0 <= int(self.s4_q8_pairs) < self.NQ):
            raise ValueError("params s4_q8_pairs: needs s4_tile_absorb, and a q~ tile index")
        if self.s4_qpe_first and not (self.s4_tile_absorb and bool(p.get("dq_fused", False))):
            raise ValueError("params s4_qpe_first needs s4_tile_absorb and dq_fused")
        # params post_root_first (stage "678", reduce_early): the root clusters' chunk i post tasks are made before
        # chunk i-1's chains and stores, so the root's in-order DM core loads chunk i's W_UV ahead of those stores
        self.post_root_first = bool(p.get("post_root_first", False))
        if self.s4_tile_absorb and not self.s4_reorder:
            raise ValueError("params s4_tile_absorb needs s4_reorder (stage_proj_r)")
        # params s4_warm_ring (stage 4): the prefetcher pushes every chunk of the head's weights before the window (the
        # ring needs a slot a chunk: weight_ring slots >= 4 NCH), as stages 0-3 leave them in the layer
        self.s4_warm_ring = bool(p.get("s4_warm_ring", False))
        self._x_extra = {}
        self.s3_route = str(p.get("s3_route", "gather"))
        # params s3_group (s3_route gather): the root multicasts this many pieces at a time, each group once its pieces
        # have landed (default a chunk's: stage_kv_bcast)
        self.s3_group = int(p.get("s3_group", self.C // 16))
        # params s3_root (s3_route gather): the chip the pieces gather on and which multicasts them (default the root)
        self.kv_root = int(str(p.get("s3_root", f"{self.ROOT:02x}")), 16)
        if self.s3_route not in ("gather", "ring"):
            raise ValueError(f"params s3_route={self.s3_route}: gather or ring")

    def _stage(self):
        super()._stage()
        self.kv_helpers()
        if self.p.get("stage") == 3:
            self.stage2_result()
            if (self.p.get("s3_group", self.C // 16) != self.C // 16
                    or int(str(self.p.get("s3_root", f"{self.ROOT:02x}")), 16) != self.ROOT):
                # s3_group: a group's pieces landed, on every chip but the root
                self.st.put_zeros(f"{self.TAG}_kvgflag", "uint8_t", (self.T // 16) * 64)
            if self.p.get("s3_route", "gather") == "ring":
                # s3_route ring: a piece landed, on every chip
                self.st.put_zeros(f"{self.TAG}_kvtflag", "uint8_t", (self.T // 16) * 64)
        if self.p.get("stage") == 5:
            self.stage34_result()
        if self.p.get("stage") == "678":
            self.stage5_result()
        if self.p.get("stage") == 2:
            self.stage1_result()
        if self.p.get("stage") in (2, "0123", "full") and self.p.get("kv_split_clusters", False):
            # kv_split_clusters: the k_pe parts' flags on the root, one a tile
            self.st.put_zeros(f"{self.TAG}_kvpflag", "uint8_t", (self.T // 16) * 64)
        if self.kvx:
            # the helpers' slices' flags: per owned tile, one a two-pair chip
            self.st.put_zeros(f"{self.TAG}_kvxflag", "uint8_t", self.KVR_TILES * len(self.kv_two) * 64)

    def stage1_result(self):
        """Stage 2 alone: each owner's stage 1 result in its memory chip's HBM, copied into its L3 before the window:
        its tiles' latent rows as the D port leaves them (kvraw, FP16, D layout) and every slice's flag (kvrflag)."""
        d, lay, T = self.d, self.lay, self.TAG
        raw = np.asarray(d.fp.d_port(d._mm(d.hq, d.wkva.q), d.k_x), dtype=np.float16)
        nr, nf = self.KVR_TILES * self.KVD, self.KVR_TILES * len(self.chips) * 64
        self.h_kvpre = {}
        for k in self.chips:
            img = np.zeros(nr + nf, dtype=np.uint8)
            for lt, q in enumerate(self.kv_tiles[k]):
                img[lt * self.KVD:(lt + 1) * self.KVD] = lay.to_d(raw[16 * q:16 * q + 16], d.mesh).astype(
                    np.float16).view(np.uint8).reshape(-1)
            img[nr:].view(np.uint32).reshape(-1, 16)[:, 0] = 1
            m = self.mem[k]
            self.h_kvpre[k] = self.st.put_hbm(f"{T}_kvpre_k{k:02x}", img.view(np.int8), mem_chip=(m >> 4, m & 15))

    def kv_helpers(self):
        """params kv_balance (stage 1, INT4): W_KVA's 18 INT4 pairs over 16 chips leave two chips with two pairs, twice
        the keys' GEMM work. The tiles are cut into kv_balance_ways equal runs (default 2): each two-pair chip makes its
        second pair for the first run only; for every other run a helper, a one-pair chip near the same memory chip (not
        the root), makes it from the pair's copy in that chip's HBM slab (the paired layout keeps each pair whole: the
        second pair is the slab's second half). Sets kvx {helper: (two-pair chip, the pair's first granule, flag slot,
        its tiles)}, kv_two and kv_half (the tiles the two-pair chips leave to helpers)."""
        self.kvx, self.kv_two, self.kv_half = {}, [], set()
        if not self.p.get("kv_balance", False):
            return
        if self.wbits != 4 or not self.kv_cols:
            raise ValueError("params kv_balance: needs wbits 4 and kv_cols")
        pg, nx = 2, self.T // 16
        ways = int(self.p.get("kv_balance_ways", 2))
        if ways < 2 or nx % ways:
            raise ValueError(f"params kv_balance_ways={ways}: at least 2, dividing the {nx} tiles")
        run = nx // ways
        width = {k: self.kvg[k][1] - self.kvg[k][0] for k in self.chips}
        if set(width.values()) - {pg, 2 * pg}:
            raise ValueError(f"params kv_balance: chips with one or two INT4 pairs only, not {sorted(set(width.values()))}")
        dist = lambda a, b: abs((a >> 4) - (b >> 4)) + abs((a & 15) - (b & 15))
        self.kv_two = [k for k in self.chips if width[k] == 2 * pg]
        free = [k for k in self.chips if width[k] == pg and k != self.ROOT]
        if len(free) < len(self.kv_two) * (ways - 1):
            raise ValueError(f"params kv_balance_ways={ways}: {len(self.kv_two) * (ways - 1)} helpers, {len(free)} chips")
        for e, a in enumerate(self.kv_two):
            for w in range(1, ways):
                h = min(free, key=lambda k: (self.mem[k] != self.mem[a], dist(k, self.mem[a]), k))
                free.remove(h)
                self.kvx[h] = (a, self.kvg[a][0] + pg, e, set(range(w * run, (w + 1) * run)))
        self.kv_half = set(range(run, nx))
        print(f"[fp] kv_balance: " + ", ".join(f"chip {h:02x} makes chip {a:02x}'s second pair for tiles "
                                                f"{min(ts)}-{max(ts)}" for h, (a, _, _, ts) in self.kvx.items()))

    def kva_in_ring(self):
        # stages 4, 5 and 678: no W_KVA in the ring
        return 0 if self.p.get("stage") in (4, 5, "678") else super().kva_in_ring()

    def per_chunk(self):
        # stage 4: a chunk's head weights only; stage 5: nothing (its key and value tiles come from L3); 678: the tail's
        st = self.p.get("stage")
        return (self.PER_CHUNK_HEAD if st == 4 else 0 if st == 5 else self.PER_CHUNK_TAIL if st == "678"
                else super().per_chunk())

    def chunk_index(self, i, part):
        # 678: the ring is the tail's weights only (W_UV, W_O x 2 a chunk)
        if self.p.get("stage") == "678":
            return i * self.PER_CHUNK_TAIL
        return super().chunk_index(i, part)

    def _src(self, k, n):
        # 678: ring chunk n is the tail's (n % 3): W_UV, W_O block 0, W_O block 1
        if self.p.get("stage") == "678":
            nm, off, size = self.w_off[k][4 + n % self.PER_CHUNK_TAIL]
            return self.h_w[k].address + off, size
        return super()._src(k, n)

    def check_core_order(self, header, x_prefixes):
        """The base check wants an x task before every DM core's first ring load: a ring load spins on a push that only
        starts after the start stamp, which shares the root's DM core. With kv_wkva_first each Ld_wkvc depends on its
        chip's start instead (stage_keys), which the hang check verifies, so it counts as reaching the start. A graph
        without ring loads (stage 2) has nothing to check."""
        if not any((n.kernel_name or "").endswith("idma_ring_load") for n in self.dfg.node_list):
            return
        # 678 (no x): each cluster's first ring load depends on its chip's start (load), as Ld_wkvc does
        extra = (("Ld_wkvc",) if self.wkva_first else ()) + (("Ld_wuv_i0",) if self.stage == "678" else ())
        super().check_core_order(header, tuple(x_prefixes) + extra)

    def load(self, g, n, name, deps=(), slab=None, dst=None):
        # 678: a cluster's first ring load behind its chip's start, so no DM core spins on a push before the start
        if self.stage == "678" and g not in self._first_ld:
            self._first_ld.add(g)
            deps = list(deps) + [self.go[self.chip(g)]]
        return super().load(g, n, name, deps, slab, dst)

    def cl_tiles(self, c):
        """The tiles of x cluster c makes keys for (stage 1) and waits for (stage 0)."""
        return [t for t in range(self.NX) if t % self.NC == c]

    # ---- stage 0: x from HBM into every chip's L3 ----------------------------------------------------------------
    def stage_x_bcast(self, go):
        R, T, dm = self.ROOT, self.TAG, self.roles["dm"]
        if self.x_mem:
            # the root's prefetcher pushes on the same engine, after this task (build); keys_by_chunk: chunk 0's tiles,
            # chunk 1's, then the rest, each a task of its own (build_full orders them around the prefetcher)
            m = self.mem[R]
            P = self.C // 16
            cuts = ([0, P] + ([2 * P] if self.NX > 2 * P else []) + [self.NX]) if self.keys_by_chunk else [0, self.NX]
            puts = []
            for a, b in zip(cuts, cuts[1:]):
                puts.append(self.cx_chip(R).host_node(
                    f"x_hbm_bcast_k{R:02x}" + (f"_t{a}" if a else ""), "__host_bingo_kernel_hbm_bcast",
                    HostBingoKernelHbmBcastArgs(
                        self.h_xhbm.address + a * self.XT, BingoMemSymbol(f"{T}_xsrc", offset=a * self.XT),
                        BingoMemSymbol(f"{T}_xtflag", offset=a * 64), self.h_seq[m].address + 64, 1, self.XT, b - a, m,
                        self.engine[R]),
                    after=[go[R]] + puts[-1:]))
            return puts
        puts = []
        for t in range(self.NX):
            puts.append(self.node(R, self.x_put_cluster, dm, f"x_bput_t{t}", "__snax_bingo_kernel_idma_put_flagged",
                                  SnaxBingoKernelIdmaPutFlaggedArgs(
                                      self.h_xhbm.address + t * self.XT, BingoMemSymbol(f"{T}_xsrc", offset=t * self.XT),
                                      self.XT, BingoMemSymbol(f"{T}_xtflag", offset=t * 64), value=1, bcast=True,
                                      also_here=True),
                                  [go[R]] + puts[-1:]))
        return puts

    def split_root_prefetch(self, go, head=0):
        """x_pf_split: the root's W_KVA ring chunks in a prefetcher task of their own, behind the start; the root's
        prefetcher then starts at the next chunk. Returns the first task (the multicast goes between the two). head > 0
        (keys_by_chunk): the next `head` chunks (chunk 0's stage-4 weights) in a second task of their own, and the
        prefetcher starts after them; returns both."""
        from bingo_kernel_args import HostBingoKernelWeightPrefetchRangeArgs
        R, n = self.ROOT, self.kva_in_ring()
        # the first part never waits for a slot: its chunks fit the empty ring
        if not 0 < n <= self.n_slots:
            raise ValueError(f"params x_pf_split: {n} W_KVA ring chunks, 1..{self.n_slots} (the ring's slots)")
        t, b = self.prefetch[R], self.prefetch[R].kernel_args
        part = lambda lo, hi: HostBingoKernelWeightPrefetchRangeArgs(
            b.sched_addr, b.ring_addr, b.flag_addr, b.release_addr, b.seq_addr, b.rec_addr, b.rec_slots, b.n_rings,
            b.n_slots, b.slot_bytes, b.memchip_id, batch=b.batch, policy=b.policy, engine=b.engine,
            batch_routed=b.batch_routed, queued=b.queued, queued_inflight_routed=b.queued_inflight_routed, head=b.head,
            batch_head=b.batch_head, chunk_lo=lo, chunk_hi=hi)
        if head and n + head > self.n_slots:
            raise ValueError(f"params keys_by_chunk: {n} + {head} ring chunks ahead of the multicast, {self.n_slots} slots")
        t.kernel_args = part(n + head, 0)
        t.kernel_name = t.kernel_args.KERNEL_NAME
        first = self.cx_chip(R).host_node(f"weight_prefetch_k{R:02x}_kva", t.kernel_name, part(0, n), after=[go[R]])
        if not head:
            return first
        return first, self.cx_chip(R).host_node(f"weight_prefetch_k{R:02x}_head", t.kernel_name, part(n, n + head),
                                                after=[go[R]])

    def x_wait(self, k, c, t, go, name=None):
        """Chip k cluster c waits for tile t's landed flag in its chip's L3 (the flag proves the multicast's data)."""
        put = (self.x_puts[min(t // (self.C // 16), len(self.x_puts) - 1) if self.keys_by_chunk else 0] if self.x_mem
               else self.x_puts[t])
        # the root's iDMA multicast is a task of the root's: its own waits depend on it
        own = k == self.ROOT and not self.x_mem
        w = self.node(k, c, self.roles["dm"], name or f"x_wait_t{t}", "__snax_bingo_kernel_wait_flags",
                      SnaxBingoKernelWaitFlagsArgs(BingoMemSymbol(f"{self.TAG}_xtflag", offset=t * 64), 1, 1),
                      [put] if own else [go[k]])
        if not own:
            self.dfg.bingo_add_order_edge(put, w)
        return w

    # ---- stage 1: the keys by columns --------------------------------------------------------------------------
    def stage_keys(self, go, last):
        """stage_kv_cols with each tile of x behind its landed flag and, with split_loads, loaded by both DMAs; with
        kv_balance a two-pair chip's second pair for the second half of the tiles on its helper (kv_helpers)."""
        d, T, dm, xdma = self.d, self.TAG, self.roles["dm"], self.roles["xdma"]
        pg = 2
        pair_bytes = HIDDEN * 16 * pg // 2              # an INT4 pair of W_KVA's columns, paired layout
        self.kvc_put, self.kvx_put = {}, {}
        for j, k in enumerate(self.chips):
            g0, g1 = self.kvg[k]
            nb = (g1 - g0) * TILE16
            helps = self.kvx.get(k)
            for c in range(self.NC):
                g = j * self.NC + c
                tiles = self.cl_tiles(c)
                l1 = lambda name, size: self.cx_chip(k).at(c).l1(f"{T}_kc_{name}", size)
                xb, ob = [l1(f"x{b}", self.XT) for b in range(2)], [l1(f"o{b}", nb) for b in range(2)]
                # keys_by_chunk: W_KVA in a buffer of its own, outside the load stream (its last reader, the last
                # chunk's key GEMM, comes after later chunks' stream loads in the program)
                wkva = l1("wkva", self._src(k, 0)[1]) if self.keys_by_chunk else None
                # keys_by_chunk: the tiles of the chunks after the last one made (their keys come after the window) take
                # buffers of their own and a second copy of W_KVA straight from HBM, so nothing of theirs is live in the
                # window (the static L1 planner lays them over the window's buffers)
                late_from = int(self.chunks_built or self.NCH) * (self.C // 16) if self.keys_by_chunk else self.NX
                # the tiles of the later chunks made inside the window (from chunk 1) take one x and one output buffer
                # (each tile's load behind the previous such tile's GEMM, its GEMM behind that tile's put), so the
                # window carries one x buffer, not two
                mid = lambda n: self.keys_by_chunk and self.C // 16 <= tiles[n] < late_from      # noqa: E731
                if self.keys_by_chunk:
                    xl, ol = [l1(f"lx{b}", self.XT) for b in range(2)], [l1(f"lo{b}", nb) for b in range(2)]
                    wl = l1("lwkva", self._src(k, 0)[1])
                    if self.kx_mid_double:
                        xm, om = [l1(f"mx{b}", self.XT) for b in range(2)], [l1(f"mo{b}", nb) for b in range(2)]
                    else:
                        xm, om = l1("mx", self.XT), l1("mo", nb)
                dbl = self.kx_mid_double
                XB = lambda n: (xl[n % 2] if tiles[n] >= late_from else (xm[n % 2] if dbl else xm) if mid(n)  # noqa: E731
                                else xb[n % 2])
                OB = lambda n: (ol[n % 2] if tiles[n] >= late_from else (om[n % 2] if dbl else om) if mid(n)  # noqa: E731
                                else ob[n % 2])
                # the previous mid tile, which shares the single buffers (kx_mid_double: none, the tile two before covers it)
                pmid = lambda n: None if dbl else max((m for m in range(n) if mid(m)), default=None)   # noqa: E731
                ldl = {}
                if self.kx_per_chunk:
                    # kx_per_chunk: a chunk's own buffers for its tiles (chunk 0's keep the ones above)
                    PC, hbuf = self.C // 16, {}

                    def hb(n):
                        h = tiles[n] // PC
                        if h < 1:
                            return None
                        if h not in hbuf:
                            nx = 2 if dbl else 1
                            hbuf[h] = ([l1(f"hx{b}_h{h}", self.XT) for b in range(nx)],
                                       [l1(f"ho{b}_h{h}", nb) for b in range(nx)])
                        return hbuf[h]
                    XB0, OB0 = XB, OB
                    XB = lambda n: hb(n)[0][n % 2 if dbl else 0] if hb(n) else XB0(n)  # noqa: E731
                    OB = lambda n: hb(n)[1][n % 2 if dbl else 0] if hb(n) else OB0(n)  # noqa: E731

                def wsrc(n):
                    # tile n's W_KVA operand and the copy it waits for
                    if self.kx_per_chunk and tiles[n] // (self.C // 16) >= 1:
                        h = tiles[n] // (self.C // 16)
                        if ("w", h) not in ldl:
                            wb_ = l1(f"hwkva_h{h}", self._src(k, 0)[1])
                            ldl[("w", h)] = (wb_, self.node(k, c, dm, f"Ld_wkvc_q{tiles[n]}",
                                                            "__snax_bingo_kernel_idma_1d_copy",
                                                            SnaxBingoKernelIdma1dCopyArgs(
                                                                BingoMemSymbol(f"{T}_wkvc3") if self.kx_l3 else
                                                                self.h_wkvc[k].address, wb_, self._src(k, 0)[1]),
                                                            [go[k]]))
                        return ldl[("w", h)]
                    if tiles[n] < late_from:
                        return slab, ld
                    if not ldl:
                        ldl[0] = self.node(k, c, dm, f"Ld_wkvc_q{tiles[n]}", "__snax_bingo_kernel_idma_1d_copy",
                                           SnaxBingoKernelIdma1dCopyArgs(self.h_wkvc[k].address, wl, self._src(k, 0)[1]),
                                           [go[k]])
                    return wl, ldl[0]
                gm, put, ld_x, gmx, putx = {}, {}, {}, {}, []
                xx = self.x_xdma if c in self.split_cls else 0
                # split_after_bcast: one wait on the last tile's flag before anything else of the cluster
                w_all = [self.x_wait(k, c, self.NX - 1, go, name="x_wait_all")] if self.after_bcast else []

                def kx(n):
                    # the tile's wait right before its load: the in-order DM core blocks on no later tile; the x buffer
                    # is free once the tile two before it has had its GEMMs
                    q = tiles[n]
                    after = [self.x_wait(k, c, q, go)] + w_all + ([gm[n - 2]] if n >= 2 else []) + \
                        ([gmx[n - 2]] if n - 2 in gmx else [])
                    if mid(n) and pmid(n) is not None:
                        after += [gm[pmid(n)]] + ([gmx[pmid(n)]] if pmid(n) in gmx else [])
                    a = self.XT - xx
                    src = lambda off: BingoMemSymbol(f"{T}_xsrc", offset=q * self.XT + off)
                    parts = []
                    if a:
                        parts.append(self.node(k, c, dm, f"kx_ld_q{q}", "__snax_bingo_kernel_idma_1d_copy",
                                               SnaxBingoKernelIdma1dCopyArgs(src(0), XB(n), a), after))
                    if xx:
                        parts.append(self.node(k, c, xdma, f"kx_xld_q{q}", "__snax_bingo_kernel_xdma_1d_copy",
                                               SnaxBingoKernelXdma1dCopyArgs(src(a), XB(n).view(a), xx), after))
                    return parts

                if helps:
                    xs, obx = l1("xs", pair_bytes), [l1(f"ox{b}", pg * TILE16) for b in range(2)]
                ldx = None
                xsl = {}

                def kxs():
                    # the helped chip's second pair, once, from its slab in HBM
                    return self.node(k, c, dm, "Ld_wkvx", "__snax_bingo_kernel_idma_1d_copy",
                                     SnaxBingoKernelIdma1dCopyArgs(self.h_wkvc[helps[0]].address + pair_bytes, xs,
                                                                   pair_bytes), [go[k]])
                if self.wkva_first:
                    # kv_wkva_first: the in-order DM core copies the chip's W_KVA first, then the first tiles of x;
                    # the helped pair, needed only later, after them. The copy depends on the chip's start, so a ring
                    # load queued ahead of the start stamp is a dependency the hang check sees (check_core_order)
                    ld, si, slab = self.load(g, 0, "Ld_wkvc", deps=[go[k]], dst=wkva)
                    ld_x[0] = kx(0)
                    if len(tiles) > 1:
                        ld_x[1] = kx(1)
                    if helps:
                        ldx = kxs()
                else:
                    ld_x[0] = kx(0)
                    if helps:
                        ldx = kxs()
                    ld, si, slab = self.load(g, 0, "Ld_wkvc", dst=wkva)
                    if len(tiles) > 1:
                        ld_x[1] = kx(1)
                for n, q in enumerate(tiles):
                    o = self.kv_owner[q]
                    lt = self.kv_tiles[o].index(q)
                    on = lambda name, off, o=o: BingoMemSymbol(name, offset=off, chip_id=None if o == k else o)
                    # a two-pair chip's helped tiles: its first pair only, the slab's first half
                    ng = (g1 - g0) - (pg if k in self.kv_two and q in self.kv_half else 0)
                    wb, wld = wsrc(n)
                    if n not in ld_x:
                        ld_x[n] = kx(n)
                    own = [put[pmid(n)]] if mid(n) and pmid(n) is not None else []
                    gm[n] = self.gmm(g, f"gkvc_q{q}", XB(n), wb, OB(n), 1, HIDDEN // 4, ng, d.k_x,
                                     [wld] + ld_x[n] + ([put[n - 2]] if n >= 2 else []) + own)
                    # tile n + 1's x load made right behind tile n's GEMM, ahead of tile n's put: where each core's tasks
                    # keep the order they were made in (the schedule's fallback), the DM core loads it while GEMM n runs
                    # (it needs only GEMM n - 1's buffer) instead of after GEMM n's put
                    if n + 1 < len(tiles) and n + 1 not in ld_x:
                        ld_x[n + 1] = kx(n + 1)
                    put[n] = self.node(k, c, dm, f"kvc_put_q{q}", "__snax_bingo_kernel_idma_put_flagged",
                                       SnaxBingoKernelIdmaPutFlaggedArgs(
                                           OB(n), on(f"{T}_kvraw", lt * self.KVD + g0 * TILE16), ng * TILE16,
                                           on(f"{T}_kvrflag", (lt * len(self.chips) + j) * 64), value=1), [gm[n]])
                    self.kvc_put[(q, k)] = put[n]
                    if helps and q in helps[3]:
                        a, ga0, e, _ = helps
                        m = len(putx)
                        xsb, xld = xs, ldx
                        if self.kc_xs_late and q >= late_from:
                            if "b" not in xsl:
                                xsl["b"] = l1("xsl", pair_bytes)
                                xsl["ld"] = self.node(k, c, dm, f"Ld_wkvxl_q{q}", "__snax_bingo_kernel_idma_1d_copy",
                                                      SnaxBingoKernelIdma1dCopyArgs(
                                                          self.h_wkvc[helps[0]].address + pair_bytes, xsl["b"],
                                                          pair_bytes), [go[k]])
                            xsb, xld = xsl["b"], xsl["ld"]
                        if self.kx_per_chunk and q // (self.C // 16) >= 1:
                            h = q // (self.C // 16)
                            if ("xs", h) not in xsl:
                                b_ = l1(f"hxs_h{h}", pair_bytes)
                                xsl[("xs", h)] = (b_, self.node(
                                    k, c, dm, f"Ld_wkvxl_q{q}", "__snax_bingo_kernel_idma_1d_copy",
                                    SnaxBingoKernelIdma1dCopyArgs(BingoMemSymbol(f"{T}_wkvx3") if self.kx_l3 else
                                                                  self.h_wkvc[helps[0]].address + pair_bytes, b_,
                                                                  pair_bytes),
                                    [go[k]]))
                            xsb, xld = xsl[("xs", h)]
                        gmx[n] = self.gmm(g, f"gkvx_q{q}", XB(n), xsb, obx[m % 2], 1, HIDDEN // 4, pg, d.k_x,
                                          [xld] + ld_x[n] + ([putx[m - 2]] if m >= 2 else []))
                        putx.append(self.node(k, c, dm, f"kvx_put_q{q}", "__snax_bingo_kernel_idma_put_flagged",
                                              SnaxBingoKernelIdmaPutFlaggedArgs(
                                                  obx[m % 2], on(f"{T}_kvraw", lt * self.KVD + ga0 * TILE16),
                                                  pg * TILE16, on(f"{T}_kvxflag", (lt * len(self.kv_two) + e) * 64),
                                                  value=1), [gmx[n]]))
                        self.kvx_put[(q, k)] = putx[-1]
                    if n + 2 < len(tiles) and not mid(n + 2):
                        ld_x[n + 2] = kx(n + 2)
                if si is not None:
                    self.ls[g].release(si, [gm[n] for n in range(len(tiles))])     # every tile's GEMM reads it
                last[k] += list(put.values() if isinstance(put, dict) else put) + list(putx)
        # every owner waits for its tiles' 16 slices, and for a helped tile its helpers' slices: the end of the stage
        # (own_wait {tile: the waits}: where a chained stage 2 starts)
        self.own_wait = {}
        for o in self.chips:
            for lt, q in enumerate(self.kv_tiles[o]):
                c = self.kv_cluster(o, q)
                waits = [("kvr_wait", "kvrflag", len(self.chips), [self.kvc_put[(q, k2)] for k2 in self.chips])]
                if self.kvx and q in self.kv_half:
                    waits.append(("kvx_wait", "kvxflag", len(self.kv_two),
                                  [self.kvx_put[(q, h)] for h, x in self.kvx.items() if q in x[3]]))
                for name, flag, nf, puts in waits:
                    w = self.node(o, c, dm, f"{name}_q{q}", "__snax_bingo_kernel_wait_flags",
                                  SnaxBingoKernelWaitFlagsArgs(BingoMemSymbol(f"{T}_{flag}", offset=lt * nf * 64), nf, 1),
                                  [t for t in puts if self.flag_dep(t, o, c)])
                    for t in puts:
                        if not self.flag_dep(t, o, c):
                            self.dfg.bingo_add_order_edge(t, w)
                    last[o].append(w)
                    self.own_wait.setdefault(q, []).append(w)

    # ---- stage 2: the owner finishes a key tile ----------------------------------------------------------------
    def stage2_result(self):
        """Stage 3 alone: each owner's pieces in its memory chip's HBM (kv_piece: key rows then value columns, INT8),
        copied into the owner's L1 before the window."""
        d, T = self.d, self.TAG
        self.h_kvp = {}
        for k in self.chips:
            img = np.concatenate([np.asarray(d.kv_piece(16 * q)).view(np.uint8).reshape(-1) for q in self.kv_tiles[k]])
            m = self.mem[k]
            self.h_kvp[k] = self.st.put_hbm(f"{T}_kvp_k{k:02x}", img.view(np.int8), mem_chip=(m >> 4, m & 15))

    def stage3_preload(self):
        """Before the window, per chip its pieces from HBM into the L1 of their tiles' clusters (s3_piece)."""
        T = self.TAG
        pre = {k: [] for k in self.chips}
        self.s3_piece = {}
        for k in self.chips:
            for lt, q in enumerate(self.kv_tiles[k]):
                c = self.kv_cluster(k, q)
                g = self.chips.index(k) * self.NC + c
                self.s3_piece[q] = self.cx_chip(k).at(c).l1(f"{T}_s3_piece_t{lt}", KVP)
                pre[k] = [self.dm_copy(g, f"kvp_pre_t{lt}", self.h_kvp[k].address + lt * KVP, self.s3_piece[q], KVP,
                                       pre[k])]
        return pre

    def stage_keys_out(self, go, last):
        """Stage 3: every owner's pieces to every chip (s3_route), then a wait a chip until all have landed."""
        R, T, dm, n = self.ROOT, self.TAG, self.roles["dm"], self.T // 16
        tiles = sorted(self.kv_owner)

        def wait_on(k, c, name, flag, nf, puts):
            # the flags prove the landing; the puts are ordering edges where they are not this chip's
            w = self.node(k, c, dm, name, "__snax_bingo_kernel_wait_flags",
                          SnaxBingoKernelWaitFlagsArgs(BingoMemSymbol(flag[0], offset=flag[1]), nf, 1),
                          [t for t in puts if self.flag_dep(t, k, c)])
            for t in puts:
                if not self.flag_dep(t, k, c):
                    self.dfg.bingo_add_order_edge(t, w)
            return w

        self.kv_put = {}
        if self.s3_route == "gather":
            R = self.kv_root
            for q in tiles:
                k = self.kv_owner[q]
                c = self.kv_cluster(k, q)
                on_root = lambda name, off: BingoMemSymbol(name, offset=off, chip_id=None if k == R else R)
                self.kv_put[q] = self.node(k, c, dm, f"kv_put_t{self.kv_tiles[k].index(q)}",
                                           "__snax_bingo_kernel_idma_put_flagged",
                                           SnaxBingoKernelIdmaPutFlaggedArgs(self.s3_piece[q], on_root(f"{T}_kv", q * KVP),
                                                                             KVP, on_root(f"{T}_kvnflag", q * 64),
                                                                             value=1), [go[k]])
            G = self.s3_group
            if G == self.C // 16 and R == self.ROOT:
                self.stage_kv_bcast()
                flag, nf = (f"{T}_kvflag", 0), self.NCH
            else:
                # s3_group: a multicast of G pieces once they have landed, one source, in tile order
                c, self.kv_mc = self.x_put_cluster, []
                for j in range(n // G):
                    puts = [self.kv_put[q] for q in range(j * G, (j + 1) * G)]
                    w = wait_on(R, c, f"kv_gwait_j{j}", (f"{T}_kvnflag", j * G * 64), G, puts + self.kv_mc[-1:])
                    src = BingoMemSymbol(f"{T}_kv", offset=j * G * KVP)
                    self.kv_mc.append(self.node(R, c, dm, f"kv_gput_j{j}", "__snax_bingo_kernel_idma_put_flagged",
                                                SnaxBingoKernelIdmaPutFlaggedArgs(
                                                    src, src, G * KVP, BingoMemSymbol(f"{T}_kvgflag", offset=j * 64),
                                                    value=1, bcast=True), [w]))
                flag, nf = (f"{T}_kvgflag", 0), n // G
            for k in self.chips:
                if k == R:
                    last[k].append(wait_on(k, 0, "kv_all", (f"{T}_kvnflag", 0), n, list(self.kv_put.values())))
                else:
                    last[k].append(wait_on(k, 0, "kv_all", flag, nf, self.kv_mc))
            return
        # ring: tile q's owner multicasts behind tile q - 1's landed flag (in its own L3), and into its own L3 too
        prev = None
        for q in tiles:
            k = self.kv_owner[q]
            c = self.kv_cluster(k, q)
            lt = self.kv_tiles[k].index(q)
            after = [go[k]]
            if prev is not None:
                after.append(wait_on(k, c, f"kv_rwait_t{lt}", (f"{T}_kvtflag", (q - 1) * 64), 1, [prev]))
            prev = self.kv_put[q] = self.node(
                k, c, dm, f"kv_rput_t{lt}", "__snax_bingo_kernel_idma_put_flagged",
                SnaxBingoKernelIdmaPutFlaggedArgs(self.s3_piece[q], BingoMemSymbol(f"{T}_kv", offset=q * KVP), KVP,
                                                  BingoMemSymbol(f"{T}_kvtflag", offset=q * 64), value=1, bcast=True,
                                                  also_here=True), after)
        for k in self.chips:
            last[k].append(wait_on(k, 0, "kv_all", (f"{T}_kvtflag", 0), n, list(self.kv_put.values())))

    def stage1_preload(self, kvr=True):
        """Before the window, per chip one chain ending on cluster 0 (as FaBuild's constants): every key cluster's
        dequantisation scales and RoPE tables (kvc_ld), then (kvr) stage 1's result from HBM into the chip's L3."""
        T, nr, nf = self.TAG, self.KVR_TILES * self.KVD, self.KVR_TILES * len(self.chips) * 64
        pre = {k: [] for k in self.chips}
        self.kvc_pe = {}
        for k in self.chips:
            for c in reversed(range(self.NC)):
                g = self.chips.index(k) * self.NC + c
                if (k, c) in self.kvc:
                    pre[k] = [self.dm_copy(g, f"kvc_ld_g{g}", self.h_kc[k].address, self.kvc[(k, c)], self.KC_BYTES,
                                           pre[k])]
            if self.kv_split:
                # kv_split_clusters: the clusters that make no keys get the constants too, for k_pe
                for c in range(self.NC):
                    if (k, c) not in self.kvc:
                        g = self.chips.index(k) * self.NC + c
                        self.kvc_pe[(k, c)] = self.cx_chip(k).at(c).l1(f"{T}_kvcpe", self.KC_BYTES)
                        pre[k] = [self.dm_copy(g, f"kvc_ld_g{g}", self.h_kc[k].address, self.kvc_pe[(k, c)],
                                               self.KC_BYTES, pre[k])]
            if not kvr:
                continue
            g = self.chips.index(k) * self.NC
            pre[k] = [self.dm_copy(g, "kvr_pre", self.h_kvpre[k].address, BingoMemSymbol(f"{T}_kvraw"), nr, pre[k])]
            pre[k] = [self.dm_copy(g, "kvr_pref", self.h_kvpre[k].address + nr, BingoMemSymbol(f"{T}_kvrflag"), nf,
                                   pre[k])]
        return pre

    def stage34_result(self):
        """Stage 5 alone: every chip's query tiles (its head, both chunks, both clusters: the reference's Q8, A layout)
        and every piece of the keys with every chunk's and every piece's landed flag, in its memory chip's HBM."""
        d, T, n = self.d, self.TAG, self.T // 16
        kv = np.concatenate([np.asarray(d.kv_piece(16 * q)).view(np.uint8).reshape(-1) for q in range(n)])
        flags = np.zeros((self.NCH + n) * 64, dtype=np.uint8)
        flags.view(np.uint32).reshape(-1, 16)[:, 0] = 1
        self.h_kvall, self.h_q8 = {}, {}
        for m in sorted(set(self.mem.values())):
            self.h_kvall[m] = self.st.put_hbm(f"{T}_kvall_m{m:02x}", np.concatenate([kv, flags]).view(np.int8),
                                              mem_chip=(m >> 4, m & 15))
        for k in self.chips:
            h = self.head[k]
            img = np.concatenate([d.q8_a(h, self.r0(i, c) + q * fa.BR).view(np.uint8).reshape(-1)
                                  for i in range(self.NCH) for c in range(self.NC) for q in range(self.NQ)])
            m = self.mem[k]
            self.h_q8[k] = self.st.put_hbm(f"{T}_q8_k{k:02x}", img.view(np.int8), mem_chip=(m >> 4, m & 15))

    def stage5_result(self):
        """Stage "678" alone: every chip's o~ (its head, both chunks, both clusters, every query tile: the reference's
        ot16 rows, FP16 row major [16, 512]) in its memory chip's HBM."""
        d, T = self.d, self.TAG
        self.h_ot = {}
        for k in self.chips:
            h = self.head[k]
            img = np.concatenate([self.xt_ref(h, r0)[0].reshape(-1)
                                  for i in range(self.NCH) for c in range(self.NC) for q in range(self.NQ)
                                  for r0 in (self.r0(i, c) + q * fa.BR,)])
            m = self.mem[k]
            self.h_ot[k] = self.st.put_hbm(f"{T}_ot_k{k:02x}", img.view(np.int8), mem_chip=(m >> 4, m & 15))

    def stage678_preload(self):
        """Before the window, per chip: every cluster's constants, then its o~ tiles of both chunks into L1."""
        T, ob = self.TAG, fa.D_V * fa.BR * (1 if self.ot8_fused else 2)
        pre = {k: [] for k in self.chips}
        self.ot_buf, self.ot_ld = {}, {}
        for k in self.chips:
            for c in reversed(range(self.NC)):
                g = self.chips.index(k) * self.NC + c
                pre[k] = [self.dm_copy(g, f"const_ld_g{g}", self.h_c[k].address, self.consts[g], self.C_BYTES, pre[k])]
                for i in range(self.NCH):
                    for q in range(self.NQ):
                        buf = self.cx_chip(k).at(c).l1(f"{T}_ot{q}_i{i}", ob)
                        off = ((i * self.NC + c) * self.NQ + q) * ob
                        ld = self.dm_copy(g, f"ot_pre_i{i}q{q}", self.h_ot[k].address + off, buf, ob, pre[k])
                        self.ot_buf[(g, i, q)], self.ot_ld[(g, i, q)] = buf, ld
                        pre[k] = [ld]
        return pre

    def stage5_preload(self):
        """Before the window, per chip: every cluster's constants (the causal masks are in them), the keys and the flags
        into its L3, every cluster's query tiles into its L1."""
        T, n, qb = self.TAG, self.T // 16, fa.D_QK * fa.BR
        pre = {k: [] for k in self.chips}
        for k in self.chips:
            g0 = self.chips.index(k) * self.NC
            for c in reversed(range(self.NC)):
                pre[k] = [self.dm_copy(g0 + c, f"const_ld_g{g0 + c}", self.h_c[k].address, self.consts[g0 + c],
                                       self.C_BYTES, pre[k])]
            src = self.h_kvall[self.mem[k]].address
            pre[k] = [self.dm_copy(g0, "kv_pre", src, BingoMemSymbol(f"{T}_kv"), n * KVP, pre[k])]
            pre[k] = [self.dm_copy(g0, "kvf_pre", src + n * KVP, BingoMemSymbol(f"{T}_kvflag"), self.NCH * 64, pre[k])]
            pre[k] = [self.dm_copy(g0, "kvnf_pre", src + n * KVP + self.NCH * 64, BingoMemSymbol(f"{T}_kvnflag"),
                                   n * 64, pre[k])]
            for i in range(self.NCH):
                for c in range(self.NC):
                    g = g0 + c
                    for q in range(self.NQ):
                        off = ((i * self.NC + c) * self.NQ + q) * qb
                        pre[k] = [self.dm_copy(g, f"q8_pre_i{i}q{q}", self.h_q8[k].address + off,
                                               self.bufs(g, i)["q8"][q], qb, pre[k])]
        return pre

    def stage_sweep(self, go, last):
        """Stage 5: per chunk and cluster FaBuild's sweep of a fresh prompt (fa_sweep over FreshTileSource, the causal
        masks); the keys' flags are set before the window, so kv_wait's puts are the start."""
        R = self.ROOT
        self.kv_put = {q: go[R] for q in range(self.T // 16)}
        self.kv_mc = [go[R]] * self.NCH
        self.sw_of = {}
        for i in range(self.NCH):
            for g in range(self.G):
                k, c = self.chip(g), self.c(g)
                q8 = {q: (self.bufs(g, i)["q8"][q], [go[k]]) for q in range(self.NQ)}
                nt, plan = self.d.tile_plan(self.r0(i, c))
                early = 2 if len(self.ls[g].slabs) >= 4 else 1
                sw = fa.fa_sweep(self.at(k, c), fa.dataclasses.replace(self.sweep_cfg, nt=nt), self.bufs(g, i)["swp"],
                                 fa.FreshTileSource(self, g, i, nt), q8, self.roles, name=f"s{i}_", early=early,
                                 mask=self.mask_of(g, plan), mask_c=self.mask_c, ot8=self.ot8_bits(),
                                 sm_batch=self.sm_batch, prologue=self.sm_prologue, geom_core=self.sm_geom)
                self.sw_of[(g, i)] = sw
                last[k] += list(sw["tr"])

    def setup(self, extra=None):
        if not (self.stage == 4 and self.s4_warm_ring):
            return super().setup(extra)
        # s4_warm_ring: the start waits for each chip's prefetcher (FaBuild's setup without its go -> prefetch edges)
        if self.n_slots < self.PER_CHUNK_HEAD * self.NCH:
            raise ValueError(f"s4_warm_ring: weight_ring slots {self.n_slots} < {self.PER_CHUNK_HEAD * self.NCH}")
        ends = []
        for k in self.chips:
            ends += list((extra or {}).get(k, [])) + [self.prefetch[k]]
        start = self.stamp("start", ends)
        self.go = {k: start for k in self.chips}
        return self.go

    def stage_proj_r(self, g, i, x):
        """s4_reorder: FaBuild.stage_proj with the in-order SIMD core's program reordered: q_nope's quantisation before
        q_pe's dequantisation (which waits for the last q GEMM), so the absorb GEMM follows the q GEMMs without a gap;
        and in the tail every query tile's q_pe quantisation before the q~ ones, while the xDMA lays out q~."""
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
            gm = self.gmm(g, f"gq{r}_i{i}", b["xrows"], slab, b["q"][r], mt, fa.HIDDEN // 4, 4, d.k_x, deps)
            self.ls[g].release(si, gm)
            gq.append(gm)
        # ---- dequantise the three regions --------------------------------------------------------
        dq = [None if self.dq_fused and r < 2 else
              self.simd_mul(g, f"dq{r}_i{i}", b["q"][r], sq[r], b["q"][r], rows * 64, [gq[r]]) for r in range(2)]
        # ---- q_nope: D -> A (FP16) per row block and half, then quantise -------------------------
        d2a = []
        for r in range(2):
            for m in range(mt):
                d2a.append(self.xd(g, f"d2an{r}{m}_i{i}", fa.d_to_a_args(
                    mesh, 1, 4, 16, b["q"][r].view(m * 4 * fa.TILE16), b["a16n"].view(m * 16 * fa.Q_NOPE * 2 + r * 16 * 64 * 2)),
                    [gq[r]] if self.dq_fused else [dq[r]]))
        if self.dq_fused:
            qn = self.mul_quant(g, f"qn_i{i}", b["a16n"], sq[0], b["qn8"], rows * fa.Q_NOPE, self.fbits(d.s_qn.inv), d2a)
        else:
            qn = self.quant(g, f"qn_i{i}", b["a16n"], b["qn8"], rows * fa.Q_NOPE * 2, self.fbits(d.s_qn.inv),
                            d2a)
        # q_pe's dequantisation after q_nope's quantisation on the SIMD core
        dq.append(self.simd_mul(g, f"dq2_i{i}", b["q"][2], sq[2], b["q"][2], rows * 64, [gq[2]]))
        # ---- absorb: q~ = q_nope W_UK ----------------------------------------------------------------
        ldu, siu, slabu = self.load(g, self.chunk_index(i, "head") + 3, f"Ld_wuk_i{i}")
        if self.s4_tile_absorb:
            # a query tile's A rows and D rows are each a contiguous run of the chunk's
            gat = [self.gmm(g, f"gab{q}_i{i}", b["qn8"].view(q * 16 * fa.Q_NOPE), slabu,
                            b["qt"].view(q * (fa.KV_RANK // 16) * fa.TILE16), 1, fa.Q_NOPE // 4, fa.KV_RANK // 16, d.k_uk,
                            [ldu, qn]) for q in range(self.NQ)]
            ga = gat[-1]
        else:
            ga = self.gmm(g, f"gab_i{i}", b["qn8"], slabu, b["qt"], mt, fa.Q_NOPE // 4, fa.KV_RANK // 16, d.k_uk,
                          [ldu, qn])
            gat = [ga]
        self.ls[g].release(siu, gat)                    # every query tile's GEMM reads the slab
        sqt = self.consts[g].view(self.C_QT)
        if self.s4_tile_absorb and not self.dq_fused:
            raise ValueError("params s4_tile_absorb needs dq_fused")
        dqt = [ga] if self.dq_fused else \
            [self.simd_mul(g, f"dqt{m}_i{i}", b["qt"].view(m * (fa.KV_RANK // 16) * fa.TILE16), sqt,
                           b["qt"].view(m * (fa.KV_RANK // 16) * fa.TILE16), 16 * fa.KV_RANK, [ga]) for m in range(mt)]
        if self.s4_tile_absorb:
            d2qt = [self.xd(g, f"d2aq{q}_i{i}", fa.d_to_a_args(mesh, 1, fa.KV_RANK // 16, fa.KV_RANK // 4,
                                                               b["qt"].view(q * (fa.KV_RANK // 16) * fa.TILE16),
                                                               b["a16q"].view(q * 16 * fa.KV_RANK * 2)), [gat[q]])
                    for q in range(self.NQ)]
        else:
            d2q = self.xd(g, f"d2aq_i{i}", fa.d_to_a_args(mesh, mt, fa.KV_RANK // 16, fa.KV_RANK // 4, b["qt"],
                                                          b["a16q"]), dqt)
            d2qt = [d2q] * self.NQ
        # ---- q_pe: RoPE in the D layout, then D -> A and quantise ------------------------------------
        rope_src = fa.BingoMemSymbol(f"{self.TAG}_rope", offset=2 * (i * self.NC + c) * self.ROPE_B)
        if self.rope_fused:
            # cos into slot 1, sin_signed into slot 3: the two tables are adjacent in L3, a slot apart in the block
            R4 = b["rope4"]
            ldr = self.node(k, c, self.roles["dm"], f"rope_ld_i{i}", "__snax_bingo_kernel_idma_2d_copy",
                            fa.SnaxBingoKernelIdma2dCopyArgs(rope_src, R4.view(self.ROPE_B), self.ROPE_B, self.ROPE_B,
                                                          2 * self.ROPE_B, 2), [])
            if self.rope_perm:
                # each row tile's D layout is its four 16-column blocks: the swap is the first two and the last two
                # exchanged, one 2-D copy for each direction
                half, mtb = (fa.ROPE // 32) * fa.TILE16, (fa.ROPE // 16) * fa.TILE16
                sw = [self.node(k, c, self.roles["dm"], f"swap{h}_i{i}", "__snax_bingo_kernel_idma_2d_copy",
                                fa.SnaxBingoKernelIdma2dCopyArgs(R4.view((1 - h) * half), R4.view(2 * self.ROPE_B + h * half),
                                                              half, mtb, mtb, mt), [dq[2]]) for h in range(2)]
            else:
                sw = [self.node(k, c, self.roles["dm"], f"swap_i{i}", "__snax_bingo_kernel_idma_pairwise_swap",
                                fa.SnaxBingoKernelIdmaPairwiseSwapArgs(R4, R4.view(2 * self.ROPE_B), rows * fa.ROPE, 2),
                                [dq[2]])]
            ad = self.node(k, c, self.roles["simd"], f"rope_i{i}", "__snax_bingo_kernel_simd_rope",
                           fa.SnaxBingoKernelSimdRopeArgs(R4, b["pe"], fa.ROPE, rows), [dq[2], ldr] + sw)
        else:
            ldr = self.dm_copy(g, f"rope_ld_i{i}", rope_src, b["ropeb"], 2 * self.ROPE_B, [])
            sw = self.node(k, c, self.roles["dm"], f"swap_i{i}", "__snax_bingo_kernel_idma_pairwise_swap",
                           fa.SnaxBingoKernelIdmaPairwiseSwapArgs(b["q"][2], b["xs"], rows * fa.ROPE, 2), [dq[2]])
            n_pe = rows * fa.ROPE
            m1 = self.simd_mul(g, f"rope_cos_i{i}", b["q"][2], b["ropeb"], b["t1"], n_pe, [dq[2], ldr])
            m2 = self.simd_mul(g, f"rope_sin_i{i}", b["xs"], b["ropeb"].view(self.ROPE_B), b["t2"], n_pe, [sw, ldr])
            ad = self.simd_add(g, f"rope_add_i{i}", b["t1"], b["t2"], b["pe"], n_pe, [m1, m2])
        d2p = self.xd(g, f"d2ap_i{i}", fa.d_to_a_args(mesh, mt, 4, 16, b["pe"], b["a16p"]),
                      [ad])
        # ---- Q8 per query tile ---------------------------------------------------------------------
        q8, qps, qas = {}, {}, {}
        # q_pe's quantisations, then q~'s; s4_q8_pairs: tile by tile
        order = ([(q, kd) for q in range(self.NQ) for kd in ("a", "p")] if self.s4_q8_pairs is not None else
                 [(q, kd) for kd in ("p", "a") for q in range(self.NQ)])
        for q, kind in order:
            if kind == "p":
                qps[q] = self.quant(g, f"q8p{q}_i{i}", b["a16p"].view(q * 16 * fa.ROPE * 2),
                                    b["q8"][q].view(16 * fa.KV_RANK), 16 * fa.ROPE * 2, self.fbits(d.s_qpe.inv), [d2p])
            elif self.whatif_q8a_free:
                qas[q] = d2qt[q]
            elif self.dq_fused:
                qas[q] = self.mul_quant(g, f"q8a{q}_i{i}", b["a16q"].view(q * 16 * fa.KV_RANK * 2), sqt, b["q8"][q],
                                        16 * fa.KV_RANK, self.fbits(d.s_qt.inv), [d2qt[q]])
            else:
                qas[q] = self.quant(g, f"q8a{q}_i{i}", b["a16q"].view(q * 16 * fa.KV_RANK * 2), b["q8"][q],
                                    16 * fa.KV_RANK * 2, self.fbits(d.s_qt.inv), [d2q])
        for q in range(self.NQ):
            q8[q] = (b["q8"][q], [qas[q], qps[q]])
        if self.q_tiles:
            return q8
        # the sweep starts behind every projection task, so its buffers and the projection's can share L1
        done = [t for q in q8.values() for t in q[1]]
        return {q: (h, done) for q, (h, _) in q8.items()}

    def proj_of(self, i):
        """Stage 4's builder for chunk i: s4_qpe_first (chunks after the first), s4_reorder, or FaBuild's."""
        if self.s4_qpe_first and i > 0:
            return self.stage_proj_qpe
        return self.stage_proj_r if self.s4_reorder else self.stage_proj

    def stage_proj_qpe(self, g, i, x):
        """s4_qpe_first, a chunk after the first: stage_proj_r with q_pe's chain first. Its GEMM runs first (the three
        W_q loads still in the ring's order), then its dequantisation, RoPE, relayout and quantisations, all under
        q_nope's two GEMMs, so after the absorb only q~'s relayouts and quantisations remain. Needs s4_tile_absorb and
        dq_fused (the paths stage_proj_r takes with them)."""
        d, b = self.d, self.bufs(g, i)
        k, c = self.chip(g), self.c(g)
        rows, mt = self.ROWS, self.ROWS // 16
        mesh = d.mesh
        # constants (once), the rotation tables of this chunk
        K0 = self.C_SQ
        sq = [self.consts[g].view(K0 + self.SQ_B * r) for r in range(3)]
        # ---- q = x W_q: the three loads in the ring's order, q_pe's GEMM (block 2) first ---------------
        gq = [None] * 3
        lds = [self.load(g, self.chunk_index(i, "head") + r, f"Ld_wq{r}_i{i}") for r in range(3)]
        for r in (2, 0, 1):
            ld, si, slab = lds[r]
            gm = self.gmm(g, f"gq{r}_i{i}", b["xrows"], slab, b["q"][r], mt, fa.HIDDEN // 4, 4, d.k_x, [ld, x])
            self.ls[g].release(si, gm)
            gq[r] = gm
        # ---- dequantise the three regions --------------------------------------------------------
        dq = [None if self.dq_fused and r < 2 else
              self.simd_mul(g, f"dq{r}_i{i}", b["q"][r], sq[r], b["q"][r], rows * 64, [gq[r]]) for r in range(2)]
        # q_pe's dequantisation first on the SIMD core
        dq.append(self.simd_mul(g, f"dq2_i{i}", b["q"][2], sq[2], b["q"][2], rows * 64, [gq[2]]))
        # ---- q_pe: RoPE in the D layout, then D -> A and quantise ------------------------------------
        rope_src = fa.BingoMemSymbol(f"{self.TAG}_rope", offset=2 * (i * self.NC + c) * self.ROPE_B)
        if self.rope_fused:
            # cos into slot 1, sin_signed into slot 3: the two tables are adjacent in L3, a slot apart in the block
            R4 = b["rope4"]
            ldr = self.node(k, c, self.roles["dm"], f"rope_ld_i{i}", "__snax_bingo_kernel_idma_2d_copy",
                            fa.SnaxBingoKernelIdma2dCopyArgs(rope_src, R4.view(self.ROPE_B), self.ROPE_B, self.ROPE_B,
                                                          2 * self.ROPE_B, 2), [])
            if self.rope_perm:
                # each row tile's D layout is its four 16-column blocks: the swap is the first two and the last two
                # exchanged, one 2-D copy for each direction
                half, mtb = (fa.ROPE // 32) * fa.TILE16, (fa.ROPE // 16) * fa.TILE16
                sw = [self.node(k, c, self.roles["dm"], f"swap{h}_i{i}", "__snax_bingo_kernel_idma_2d_copy",
                                fa.SnaxBingoKernelIdma2dCopyArgs(R4.view((1 - h) * half), R4.view(2 * self.ROPE_B + h * half),
                                                              half, mtb, mtb, mt), [dq[2]]) for h in range(2)]
            else:
                sw = [self.node(k, c, self.roles["dm"], f"swap_i{i}", "__snax_bingo_kernel_idma_pairwise_swap",
                                fa.SnaxBingoKernelIdmaPairwiseSwapArgs(R4, R4.view(2 * self.ROPE_B), rows * fa.ROPE, 2),
                                [dq[2]])]
            ad = self.node(k, c, self.roles["simd"], f"rope_i{i}", "__snax_bingo_kernel_simd_rope",
                           fa.SnaxBingoKernelSimdRopeArgs(R4, b["pe"], fa.ROPE, rows), [dq[2], ldr] + sw)
        else:
            ldr = self.dm_copy(g, f"rope_ld_i{i}", rope_src, b["ropeb"], 2 * self.ROPE_B, [])
            sw = self.node(k, c, self.roles["dm"], f"swap_i{i}", "__snax_bingo_kernel_idma_pairwise_swap",
                           fa.SnaxBingoKernelIdmaPairwiseSwapArgs(b["q"][2], b["xs"], rows * fa.ROPE, 2), [dq[2]])
            n_pe = rows * fa.ROPE
            m1 = self.simd_mul(g, f"rope_cos_i{i}", b["q"][2], b["ropeb"], b["t1"], n_pe, [dq[2], ldr])
            m2 = self.simd_mul(g, f"rope_sin_i{i}", b["xs"], b["ropeb"].view(self.ROPE_B), b["t2"], n_pe, [sw, ldr])
            ad = self.simd_add(g, f"rope_add_i{i}", b["t1"], b["t2"], b["pe"], n_pe, [m1, m2])
        d2p = self.xd(g, f"d2ap_i{i}", fa.d_to_a_args(mesh, mt, 4, 16, b["pe"], b["a16p"]),
                      [ad])
        # ---- q_pe's quantisations, a query tile at a time ----------------------------------------------
        q8, qps, qas = {}, {}, {}
        for q in range(self.NQ):
            qps[q] = self.quant(g, f"q8p{q}_i{i}", b["a16p"].view(q * 16 * fa.ROPE * 2),
                                b["q8"][q].view(16 * fa.KV_RANK), 16 * fa.ROPE * 2, self.fbits(d.s_qpe.inv), [d2p])
        # ---- q_nope: D -> A (FP16) per row block and half, then quantise -------------------------
        d2a = []
        for r in range(2):
            for m in range(mt):
                d2a.append(self.xd(g, f"d2an{r}{m}_i{i}", fa.d_to_a_args(
                    mesh, 1, 4, 16, b["q"][r].view(m * 4 * fa.TILE16), b["a16n"].view(m * 16 * fa.Q_NOPE * 2 + r * 16 * 64 * 2)),
                    [gq[r]] if self.dq_fused else [dq[r]]))
        if self.dq_fused:
            qn = self.mul_quant(g, f"qn_i{i}", b["a16n"], sq[0], b["qn8"], rows * fa.Q_NOPE, self.fbits(d.s_qn.inv), d2a)
        else:
            qn = self.quant(g, f"qn_i{i}", b["a16n"], b["qn8"], rows * fa.Q_NOPE * 2, self.fbits(d.s_qn.inv),
                            d2a)
        # ---- absorb: q~ = q_nope W_UK ----------------------------------------------------------------
        ldu, siu, slabu = self.load(g, self.chunk_index(i, "head") + 3, f"Ld_wuk_i{i}")
        if self.s4_tile_absorb:
            # a query tile's A rows and D rows are each a contiguous run of the chunk's
            gat = [self.gmm(g, f"gab{q}_i{i}", b["qn8"].view(q * 16 * fa.Q_NOPE), slabu,
                            b["qt"].view(q * (fa.KV_RANK // 16) * fa.TILE16), 1, fa.Q_NOPE // 4, fa.KV_RANK // 16, d.k_uk,
                            [ldu, qn]) for q in range(self.NQ)]
            ga = gat[-1]
        else:
            ga = self.gmm(g, f"gab_i{i}", b["qn8"], slabu, b["qt"], mt, fa.Q_NOPE // 4, fa.KV_RANK // 16, d.k_uk,
                          [ldu, qn])
            gat = [ga]
        self.ls[g].release(siu, gat)                    # every query tile's GEMM reads the slab
        sqt = self.consts[g].view(self.C_QT)
        if self.s4_tile_absorb and not self.dq_fused:
            raise ValueError("params s4_tile_absorb needs dq_fused")
        dqt = [ga] if self.dq_fused else \
            [self.simd_mul(g, f"dqt{m}_i{i}", b["qt"].view(m * (fa.KV_RANK // 16) * fa.TILE16), sqt,
                           b["qt"].view(m * (fa.KV_RANK // 16) * fa.TILE16), 16 * fa.KV_RANK, [ga]) for m in range(mt)]
        if self.s4_tile_absorb:
            d2qt = [self.xd(g, f"d2aq{q}_i{i}", fa.d_to_a_args(mesh, 1, fa.KV_RANK // 16, fa.KV_RANK // 4,
                                                               b["qt"].view(q * (fa.KV_RANK // 16) * fa.TILE16),
                                                               b["a16q"].view(q * 16 * fa.KV_RANK * 2)), [gat[q]])
                    for q in range(self.NQ)]
        else:
            d2q = self.xd(g, f"d2aq_i{i}", fa.d_to_a_args(mesh, mt, fa.KV_RANK // 16, fa.KV_RANK // 4, b["qt"],
                                                          b["a16q"]), dqt)
            d2qt = [d2q] * self.NQ
        # ---- q~'s quantisations ----------------------------------------------------------------------
        for q in range(self.NQ):
            qas[q] = self.mul_quant(g, f"q8a{q}_i{i}", b["a16q"].view(q * 16 * fa.KV_RANK * 2), sqt, b["q8"][q],
                                    16 * fa.KV_RANK, self.fbits(d.s_qt.inv), [d2qt[q]])
        for q in range(self.NQ):
            q8[q] = (b["q8"][q], [qas[q], qps[q]])
        if self.q_tiles:
            return q8
        done = [t for q in q8.values() for t in q[1]]
        return {q: (h, done) for q, (h, _) in q8.items()}

    def x_rows(self, g, i, after):
        """The full chain: cluster g's rows of chunk i from the chip's L3, where stage 0's multicast lands them, once
        their four tiles' flags are in (x_wait); s4_split_x: the xDMA copies its share at the same time."""
        T, k, c = self.TAG, self.chip(g), self.c(g)
        half = self.XB // self.NC
        t0 = (i * self.XB + self.xh(i, c) * half) // self.XT
        waits = [self.x_wait(k, c, t, self.go, name=f"x_wait_i{i}t{t - t0}") for t in range(t0, t0 + half // self.XT)]
        xb = self.bufs(g, i)["xrows"]
        xx = int(round(half * self.s4_split_x / 512)) * 512
        self._x_extra[g] = ([self.node(k, c, self.roles["xdma"], f"x_xcopy_i{i}", "__snax_bingo_kernel_xdma_1d_copy",
                                       SnaxBingoKernelXdma1dCopyArgs(
                                           BingoMemSymbol(f"{T}_xsrc", offset=i * self.XB + self.xh(i, c) * half + (half - xx)),
                                           xb.view(half - xx), xx), list(after) + waits)] if xx else [])
        return self.node(k, c, self.roles["dm"], f"x_copy_i{i}", "__snax_bingo_kernel_idma_1d_copy",
                         SnaxBingoKernelIdma1dCopyArgs(BingoMemSymbol(f"{T}_xsrc", offset=i * self.XB + self.xh(i, c) * half), xb,
                                                       half - xx), list(after) + waits)

    def kv_wait(self, g, i):
        # the full chain: the root's pieces carry two flags (kv_split_clusters); its sweep follows the root's own
        # waits for chunks 0 .. i
        if getattr(self, "_chained", False) and self.chip(g) == self.kv_root and self.kv_split:
            return self.root_ready[i]
        return super().kv_wait(g, i)

    def _next_id(self):
        """The node id the next task made gets."""
        return max((t.node_id for t in self.dfg.node_list), default=-1) + 1

    def build_full(self):
        d, R = self.d, self.ROOT
        dm = self.roles["dm"]
        self.last_ld = {}
        # a slab is never loaded again before the reader of its last load is known
        for ls in self.ls:
            ls.strict = True
        # the dequantisation constants of every cluster into its L1 before the start
        # (one chain a chip, ending on cluster 0: the start stamp sits on the root's cluster 0 DM core
        # behind that core's previous task, so a chain ending there adds no second edge into the stamp)
        post_only = self.stage == "678"
        if post_only:
            pre = self.stage678_preload()
        else:
            pre = self.stage1_preload(kvr=False)
            for k in self.chips:
                for c in reversed(range(self.NC)):
                    g = self.chips.index(k) * self.NC + c
                    pre[k] = [self.dm_copy(g, f"const_ld_g{g}", self.h_c[k].address, self.consts[g], self.C_BYTES,
                                           pre[k])]
            if self.kx_per_chunk:
                if self.kx_l3:
                    self._kx_l3_preload(pre)
        if self.fresh and (self.kv_slabs not in ("split", "split4", "split_kb") or self.stop not in ("full", "q8", "sweep")):
            raise ValueError("fresh: kv_slabs split or split4 (fa.FreshTileSource's slabs), stop full, q8 or sweep")
        go = self.setup(extra=pre)
        if self.p.get("relay_per_gather", True):
            self.dfg.relay_one_consumer_cores = [self.roles["xdma"]]
        # params relay_prefer {consumer role: [roles]}: dep-tag relays of that role's tasks go to those cores first; a
        # relay joins its core's queue just before the first task there that its consumer reaches, so on a core busy to
        # the end of the sweep (the SIMD core) it holds the chain back until the sweep ends
        if self.p.get("relay_prefer"):
            self.dfg.relay_prefer_cores = {self.roles[c]: [self.roles[r] for r in rs]
                                           for c, rs in self.p["relay_prefer"].items()}
        # chunk-major: every cluster's chunk i, then the barriers and the reduce in the order the arm needs
        puts = []
        swp_res, post_res, parts_ready = {}, {}, {}
        xs, self._y_last = {}, []
        red, stores = {}, {}
        sym_free = {}
        pend = None
        defer = []
        # stages 0 to 3 (the stage builder's): X multicast from HBM, the keys by columns, the owners, the root's
        # multicasts of the keys
        self.x_puts = [] if post_only else self.stage_x_bcast(go)
        if self.prefetch_after_x and self.x_puts:
            for k in self.chips:
                self.edge(self.x_puts[-1], self.prefetch[k])
        elif self.x_mem and self.x_puts:
            if self.bcast_after_pf:
                self.edge(self.prefetch[R], self.x_puts[-1])
            else:
                if self.pf_split and self.keys_by_chunk:
                    # W_KVA, chunk 0's x, chunk 0's stage-4 weights, chunk 1's x, the rest of the weights; the rest of x
                    # (chunks 2 on, whose keys come after the window) once the prefetcher is done. No multicast of x
                    # then overlaps the root's key multicasts of chunks 0 and 1: two multicast sources at once
                    # deadlock the fabric (stg_dsv2poc_t1024_ab, HANG-AXI 29 Oct)
                    kva, head = self.split_root_prefetch(go, head=self.PER_CHUNK_HEAD)
                    self.edge(kva, self.x_puts[0])
                    self.edge(self.x_puts[0], head)
                    self.edge(head, self.x_puts[1])
                    if len(self.x_puts) > 2 and self.x_rest_early:
                        # x_rest_early: the rest of x (behind chunk 0's key multicast, below), then the prefetcher
                        self.edge(self.x_puts[2], self.prefetch[R])
                    else:
                        self.edge(self.x_puts[1], self.prefetch[R])
                        if len(self.x_puts) > 2:
                            self.edge(self.prefetch[R], self.x_puts[2])
                else:
                    if self.pf_split:
                        self.edge(self.split_root_prefetch(go), self.x_puts[0])
                    self.edge(self.x_puts[-1], self.prefetch[R])
        front = {k: [] for k in self.chips}
        if not post_only:
            k0 = self._next_id()
            self.stage_keys(go, front)
            if self.keys_by_chunk:
                # stage 1's tasks carry their tile (_q<n>, the x waits _t<n>); the start's loads of W_KVA are chunk 0's
                P = self.C // 16
                for t in self.dfg.node_list:
                    if t.node_id >= k0:
                        m = re.search(r"_[qt](\d+)", t.node_name)
                        t._key_chunk = int(m.group(1)) // P if m else 0
            # full_kv_all false: no chip waits for both chunks' keys at the end of stage 3 (stage "0123"'s window end);
            # the sweep waits for its own tiles, and the wait would hold cluster 0's in-order DM core ahead of stage 4
            self.stage_owner(go, front, chained=True, end_wait=bool(self.p.get("full_kv_all", True)))
            if self.keys_by_chunk and len(self.x_puts) > 2 and len(self.kv_mc) > 1:
                # the rest of x after chunk 1's key multicast too (x_rest_early: after chunk 0's, and chunk 1's waits
                # for its last tile), as a dependency (the check below proves the order)
                self.edge(self.kv_mc[0 if self.x_rest_early else 1], self.x_puts[2])
            if self.s4_early:
                self.ring_rel1 |= {(self.kv_root, self.chunk_index(i, "head") + r)
                                   for i in range(self.NCH) for r in range(self.PER_CHUNK_HEAD)}
        if self.chunks_built is not None:
            # the key stages above made every chunk's keys; the chunks below are the first chunks_built only
            self.NCH = min(self.NCH, int(self.chunks_built))
        if self.yr_rel1:
            if not self.y_roots:
                raise ValueError("params yr_rel1: needs y_roots")
            for i in range(1, self.NCH):
                for h in range(2):
                    k = self.y_root(i - 1, h)
                    self.ring_rel1 |= {(k, self.chunk_index(i, "head") + r) for r in range(self.per_chunk())}
            print(f"[{self.TAG}] yr_rel1: {sum(1 for k, n in self.ring_rel1 if k in self.y_roots)} ring chunks on the "
                  f"collectors released by cluster 1")
        for i in range(self.NCH):
            if i:
                self.chunk_start(i)
            sweeps = {}
            for g in range(self.G):
                k, c = self.chip(g), self.c(g)
                # the rows land once the previous chunk's sweep has been read out (its L1 is theirs; wo_cols: once its
                # W_O GEMMs are done with the gathered o8, whose bytes the rows then take)
                if post_only:
                    # 678: the sweep's o~ tiles were loaded before the window
                    sweeps[g] = swp_res[g] = dict(xt=[self.ot_buf[(g, i, q)] for q in range(self.NQ)],
                                                  tr=[self.ot_ld[(g, i, q)] for q in range(self.NQ)])
                    if self.reduce_early and i > 0 and k == R:
                        w = {c: [stores[(i - 2, c)]] if (i - 2, c) in stores else []}
                        if self.post_root_first:
                            defer.append((i - 1, [c], w))
                        else:
                            self._block_chains(i - 1, [c], w, None, red, stores, sym_free)
                    continue
                # s4_overlap: the rows land without waiting for the previous chunk's readers of its sweep (the L1 planner
                # then keeps the two apart)
                s4_from = self._next_id()
                x = self.x_rows(g, i, [go[k]] + (list(post_res[g]["xt_rd"]) if g in post_res and not self.s4_overlap
                                                 else []))
                xs[g] = x
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
                q8 = self.proj_of(i)(g, i, x)
                if self.reduce_early and i > 0 and k == R and not self.chains_after_sweep:
                    # chunk i-1's chains of this root cluster, behind its q side of chunk i in its xDMA core's program
                    w = {c: [stores[(i - 2, c)]] if (i - 2, c) in stores else []}
                    self._block_chains(i - 1, [c], w, None, red, stores, sym_free)
                if self.stop == "q8":
                    # every query tile's Q8 (q_tiles: each tile has its own ready tasks)
                    parts_ready[(k, c, i)] = list(dict.fromkeys(t for q in sorted(q8) for t in q8[q][1]))
                    post_res[g] = dict(xt_rd=parts_ready[(k, c, i)])
                    continue
                early = 2 if len(self.ls[g].slabs) >= 4 else 1
                if self.fresh:
                    # the key tiles up to the chunk's end, the causal masks on the scores
                    nt, plan = d.tile_plan(self.r0(i, c))
                    sw = fa.fa_sweep(self.at(k, c), fa.dataclasses.replace(self.sweep_cfg, nt=nt), self.bufs(g, i)["swp"],
                                  fa.FreshTileSource(self, g, i, nt), q8, self.roles, name=f"s{i}_", early=early,
                                  mask=self.mask_of(g, plan), mask_c=self.mask_c, ot8=self.ot8_bits(),
                                 sm_batch=self.sm_batch, prologue=self.sm_prologue, geom_core=self.sm_geom)
                else:
                    sw = fa.fa_sweep(self.at(k, c), self.sweep_cfg, self.bufs(g, i)["swp"], fa.TileSource(self, g, i), q8,
                                  self.roles, name=f"s{i}_", early=early, sm_batch=self.sm_batch, prologue=self.sm_prologue, geom_core=self.sm_geom)
                swp_res[g] = sw
                if self.chains_after_sweep and self.reduce_early and i > 0 and k == R and self.dst_shared(i - 1):
                    # y_dst_shared: every root cluster's chains of chunk i-1 together once the last one's sweep is made,
                    # block 0 of every cluster first; each block-0 store behind its cluster's load of key tile
                    # dst_store_tile's values
                    if c == self.NC - 1:
                        w = {cc: [stores[(i - 2, cc)]] if (i - 2, cc) in stores else [] for cc in range(self.NC)}
                        at = {cc: self._tile_load(R, cc, i) for cc in range(self.NC)}
                        self._block_chains(i - 1, range(self.NC), w, None, red, stores, sym_free, store_at=at)
                elif self.chains_after_sweep and self.reduce_early and i > 0 and k == R:
                    # chunk i-1's chains of this root cluster, behind its sweep of chunk i in its cores' programs
                    w = {c: [stores[(i - 2, c)]] if (i - 2, c) in stores else []}
                    self._block_chains(i - 1, [c], w, None, red, stores, sym_free)
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
            for i1, cs1, w1 in defer:
                self._block_chains(i1, cs1, w1, None, red, stores, sym_free)
            defer = []
            if self.wo_cols:
                last = self.stage_wo_cols(i, post, xs)
                for g in range(self.G):
                    post_res[g] = dict(xt_rd=list(post[g][0]) + [last[g]])
                continue
            if self.y_halves:
                # W_O block 0 everywhere, its gather, block 1 behind that gather, its gather, the store
                gate = {c: post[self.chips.index(R) * self.NC + c][0] for c in range(self.NC)}
                if self.chunk_pipeline:
                    # chunk i-1's block 1, held back until chunk i's front is made, then chunk i's block 0
                    if pend is not None:
                        self._halves(*pend, red, stores, sym_free, blocks=(1,))
                    pend = (i, post, gate, {}, parts_ready, [])
                    self._halves(*pend, red, stores, sym_free, blocks=(0,))
                    for g in range(self.G):
                        post_res[g] = dict(xt_rd=post[g][0])
                    continue
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
                            sym_free[(k, c)] = [rh[1][c]]
                continue
            if self.stop == "full" and self.sync != "barrier" and i > 0 and not self.reduce_early:
                gate = {c: post[self.chips.index(R) * self.NC + c][0] for c in range(self.NC)}
                self._reduce_chunk(i - 1, parts_ready, red, stores, sym_free, gate)
            for g in range(self.G):
                k, c = self.chip(g), self.c(g)
                post_res[g] = dict(xt_rd=post[g][0])
                parts_ready[(k, c, i)] = self.stage_post_b(g, i, post[g][1], sym_free.get((k, c), []))
            if self.stop == "full" and self.sync == "barrier":
                self._reduce_chunk(i, parts_ready, red, stores, sym_free)
        if self.stop == "q8" and not post_only and not self.p.get("full_kv_all", True):
            # the window still ends with the keys on every chip: each one's wait for them, behind its stage 4
            for k in self.chips:
                if k != self.kv_root:
                    key = (k, 0, self.NCH - 1)
                    parts_ready[key] = list(parts_ready[key]) + [self.kv_all(k)]
        if self.stop == "full" and self.sync != "barrier" and not self.y_halves and not self.wo_cols:
            self._reduce_chunk(self.NCH - 1, parts_ready, red, stores, sym_free)
        if pend is not None:
            self._halves(*pend, red, stores, sym_free, blocks=(1,))
        if self.wo_cols:
            # every cluster's blocks of every chunk have landed: one wait on the root, on their flags
            yw = self.node(R, self.x_put_cluster, dm, "y_wait", "__snax_bingo_kernel_wait_flags",
                           SnaxBingoKernelWaitFlagsArgs(BingoMemSymbol(f"{self.TAG}_yflag"), self.NCH * self.G, 1),
                           [t for t in self._y_last if t.assigned_chiplet_id == R])
            for t in self._y_last:
                if t.assigned_chiplet_id != R:
                    self.dfg.bingo_add_order_edge(t, yw)
            ends = [yw]
        elif stores:
            # every chunk's stores (the checks read every one)
            ends = [t for v in stores.values() for t in (v if isinstance(v, (list, tuple)) else [v])]
        else:
            # a stop before the reduce: one probe a chip behind that chip's last chunk, the root's last
            # (the end stamp sits behind the root's previous task on its core)
            last = {k: [t for c in range(self.NC) for t in parts_ready[(k, c, self.NCH - 1)]] for k in self.chips}
            ends = [self.probe(k, f"fin{k:02x}", last[k]) for k in self.chips if k != R]
            ends.append(self.probe(R, f"fin{R:02x}", last[R]))
        if self.keys_by_chunk and len(self.x_puts) > 2 and self._chains:
            # the rest of x is a multicast, and a reduce chain beside a multicast deadlocks the fabric
            # (stg_dsv2poc_t1024_ab_yr, HANG-AXI 30 Oct): with chunks_built (the chunks it serves come after the
            # window) it waits for every chain, else every chain waits for it; _check_multicasts proves the order
            for t in self._chains:
                if self.chunks_built is not None and not self.x_rest_early:
                    self.edge(t, self.x_puts[2])
                else:
                    self.edge(self.x_puts[2], t)
        self.end_window(ends)
        if self.keys_by_chunk:
            self._keys_by_chunk_order()
            if self.kx_per_chunk:
                self._kx_per_chunk_edges()
            if self.kx_after_chains:
                self._kx_before_chains()
        if post_only:
            # 678: y only (no keys were made)
            self.fresh = False
            self._checks(swp_res, post_res)
            self.fresh = True
        else:
            self._checks(swp_res, post_res)

    def stage4_preload(self):
        """Before the window, every cluster's dequantisation constants into its L1 (FaBuild's const_ld chain)."""
        pre = {k: [] for k in self.chips}
        for k in self.chips:
            for c in reversed(range(self.NC)):
                g = self.chips.index(k) * self.NC + c
                pre[k] = [self.dm_copy(g, f"const_ld_g{g}", self.h_c[k].address, self.consts[g], self.C_BYTES, pre[k])]
        return pre

    def stage_queries(self, go, last):
        """Stage 4: per chunk, every cluster's rows of x into L1 and its query projection (FaBuild's stage_x and
        stage_proj); a chunk's rows land once the previous chunk's projection has read the buffer."""
        self.q8_of, rd = {}, {}
        T = self.TAG
        for i in range(self.NCH):
            for g in range(self.G):
                k, c = self.chip(g), self.c(g)
                after = [go[k]] + ([] if self.s4_overlap else rd.get(g, []))
                xx = int(round(self.XB // self.NC * self.s4_split_x / 512)) * 512
                if xx:
                    # the xDMA copies the tail of the rows, the iDMA the head, at the same time
                    half = self.XB // self.NC
                    xb = self.bufs(g, i)["xrows"]
                    src = BingoMemSymbol(f"{T}_xsrc", offset=i * self.XB + self.xh(i, c) * half + (half - xx))
                    self._x_extra[g] = [self.node(k, c, self.roles["xdma"], f"x_xcopy_i{i}",
                                                  "__snax_bingo_kernel_xdma_1d_copy",
                                                  SnaxBingoKernelXdma1dCopyArgs(src, xb.view(half - xx), xx), after)]
                    x = self.node(k, c, self.roles["dm"], f"x_copy_i{i}", "__snax_bingo_kernel_idma_1d_copy",
                                  SnaxBingoKernelIdma1dCopyArgs(BingoMemSymbol(f"{T}_xsrc", offset=i * self.XB + self.xh(i, c) * half),
                                                                xb, half - xx), after)
                else:
                    self._x_extra[g] = []
                    x = self.stage_x(g, i, None, after)
                q8 = self.proj_of(i)(g, i, x)
                self.q8_of[(g, i)] = q8
                rd[g] = sorted({t for _, ts in q8.values() for t in ts}, key=lambda t: t.node_id)
                last[k] += rd[g]
        self._x_extra = {}

    def run_checks(self, checks):
        """keys_by_chunk with chunks_built: the keys of the chunks after the last one built are made after the window, so
        the checks also wait for the root's last key multicast (the last piece's landing in every chip's L3); without
        it a key check could read the last pieces before they land."""
        if self.keys_by_chunk and self.chunks_built is not None and getattr(self, "kv_mc", None):
            self.check_copies = list(self.check_copies) + [self.kv_mc[-1]]
        # the key checks read what every owner's puts wrote into its L3
        puts = list(getattr(self, "kv_put", {}).values()) + list(getattr(self, "kvp_put", {}).values())
        self.check_copies = list(self.check_copies) + [t for t in puts if t not in self.check_copies]
        return super().run_checks(checks)

    def finish(self, app_name, **kw):
        if self.keys_by_chunk:
            self._order_collectives()
            self._check_multicasts()
        super().finish(app_name, **kw)

    def _order_collectives(self):
        """keys_by_chunk: every multicast and every reduce chain of the graph one after the other, in the order they
        were made (the graph's topological order nearest to it): an ordering edge from each to the next where no path
        orders them yet. Concurrent multicast sources, or a multicast beside a chain, deadlock the fabric."""
        import networkx as nx
        g = nx.DiGraph(self.dfg.bingo_order_view())
        coll = [t for t in self.dfg.node_list if (t.kernel_name or "").endswith(("hbm_bcast", "xdma_chain_gather"))
                or ((t.kernel_name or "").endswith("idma_put_flagged") and getattr(t.kernel_args, "bcast", False))]
        pos = {t: n for n, t in enumerate(nx.lexicographical_topological_sort(g, key=lambda t: t.node_id))}
        coll.sort(key=lambda t: pos[t])
        added = 0
        for a, b in zip(coll, coll[1:]):
            if not nx.has_path(g, a, b):
                self.dfg.bingo_add_order_edge(a, b)
                g.add_edge(a, b)
                added += 1
        print(f"[{self.TAG}] collectives: {len(coll)} multicasts and chains in turn, {added} ordering edges added")

    def _check_multicasts(self):
        """keys_by_chunk: every two multicasts (the memory chip's x multicasts, the root's key multicasts), and every
        multicast and reduce chain, are ordered by the graph (dependencies and flag edges), so no two run at once: concurrent multicast sources, or a multicast beside a chain, deadlock the fabric. Raises
        otherwise."""
        import networkx as nx
        g = nx.DiGraph(self.dfg)
        g.add_edges_from(self.dfg.order_edges)
        mc = [t for t in self.dfg.node_list if (t.kernel_name or "").endswith("hbm_bcast")
              or ((t.kernel_name or "").endswith("idma_put_flagged") and getattr(t.kernel_args, "bcast", False))]
        ch = [t for t in self.dfg.node_list if (t.kernel_name or "").endswith("xdma_chain_gather")]
        desc = {t: nx.descendants(g, t) for t in mc}
        bad = [(a.node_name, b.node_name) for i, a in enumerate(mc) for b in mc[i + 1:]
               if b not in desc[a] and a not in desc[b]]
        if bad:
            raise ValueError(f"params keys_by_chunk: {len(bad)} pair(s) of multicasts may run at once: {bad[:6]}")
        # a reduce chain beside a multicast deadlocks the fabric too
        dch = {t: nx.descendants(g, t) for t in ch}
        bad = [(a.node_name, b.node_name) for a in mc for b in ch if b not in desc[a] and a not in dch[b]]
        if bad:
            raise ValueError(f"params keys_by_chunk: {len(bad)} multicast / chain pair(s) may run at once: {bad[:6]}")
        print(f"[{self.TAG}] keys_by_chunk: {len(mc)} multicasts, every two ordered; {len(ch)} chains, each ordered "
              f"with every multicast")
        if self.kx_after_chains:
            # kx_after_chains: every key transfer (puts to the owners, copies from HBM) ordered with every chain
            kt = [t for t in self.dfg.node_list
                  if re.match(r"(kvc_put|kvx_put|Ld_wkvc_q|Ld_wkvxl_q|kvc_lld_g\d+_q|kv_put_t)", t.node_name)]
            anc = {t: nx.ancestors(g, t) for t in ch}
            bad = [(a.node_name, b.node_name) for b in ch for a in kt if a not in dch[b] and a not in anc[b]]
            if bad:
                raise ValueError(f"params kx_after_chains: {len(bad)} key transfer / chain pair(s) may run at once: "
                                 f"{bad[:6]}")
            print(f"[{self.TAG}] kx_after_chains: {len(kt)} key transfers, each ordered with every chain")

    def _tile_load(self, k, c, i):
        """y_dst_shared: cluster (k, c)'s load of chunk i's key tile dst_store_tile's values (None: a shorter sweep)."""
        rx = re.compile(rf"Ld_V{self.dst_store_tile}_i{i}(_|$)")
        dm = self.roles["dm"]
        hits = [t for t in self.dfg.node_list if t.assigned_chiplet_id == k and t.assigned_cluster_id == c
                and t.assigned_core_id == dm and rx.match(t.node_name)]
        return max(hits, key=lambda t: t.node_id) if hits else None

    def _kx_l3_preload(self, pre):
        """kx_per_chunk: before the window, each chip copies its W_KVA slice, its key constants and (a helper) the
        helped chip's second pair from HBM into its own L3, so the per-chunk copies in the window read the chip's L3,
        never HBM: reads across the D2D links beside the memory chips' ring pushes stalled the fabric
        (stg_dsv2poc_t1024_c3, HANG-AXI 30 Oct)."""
        T = self.TAG
        pair = HIDDEN * 16 * 2 // 2
        st = self.st
        st.put_zeros(f"{T}_wkvc3", "uint8_t", max(self._src(k, 0)[1] for k in self.chips))
        st.put_zeros(f"{T}_kc3", "uint8_t", self.KC_BYTES)
        if self.kvx:
            st.put_zeros(f"{T}_wkvx3", "uint8_t", pair)
        for k in self.chips:
            g = self.chips.index(k) * self.NC
            pre[k] = [self.dm_copy(g, "wkvc3_ld", self.h_wkvc[k].address, BingoMemSymbol(f"{T}_wkvc3"),
                                   self._src(k, 0)[1], pre[k])]
            pre[k] = [self.dm_copy(g, "kc3_ld", self.h_kc[k].address, BingoMemSymbol(f"{T}_kc3"), self.KC_BYTES, pre[k])]
            if k in self.kvx:
                pre[k] = [self.dm_copy(g, "wkvx3_ld", self.h_wkvc[self.kvx[k][0]].address + pair,
                                       BingoMemSymbol(f"{T}_wkvx3"), pair, pre[k])]

    def _kx_store_of(self, k, c, h):
        """kx_per_chunk with y_roots and yr_store_tile: cluster (k, c)'s store of chunk h - 2's y block when chip k
        collected it (chunk h's keys, made in chunk h - 1, follow it), else None."""
        if not (self.kx_per_chunk and self.y_roots and self.yr_store_tile is not None) or h < 2:
            return None
        if not hasattr(self, "_by_name"):
            self._by_name = {t.node_name: t for t in self.dfg.node_list}
        for hh in range(2):
            if self.y_root(h - 2, hh) == k:
                return self._by_name.get(f"y_store_c{c}_i{h - 2}_h{hh}_k{k:02x}_cl{c}")
        return None

    # kx_after_chains: stage 1 of a chunk's keys (each tile's x, its GEMMs, the puts to the owners; no SIMD task), which
    # runs inside the sweep before, behind the value load
    KX_STAGE1 = r"(kx_|gkv[cx]_|kv[cx]_put_q|Ld_wkv|x_wait_t)"

    def _kx_sweep_end_of(self, k, c, h):
        """kx_after_chains: cluster (k, c)'s last SIMD task of chunk h - 1's sweep (a softmax or a normalisation)."""
        if not hasattr(self, "_sweep_end"):
            self._sweep_end = {}
            simd = self.roles["simd"]
            for x in self.dfg.node_list:
                m = re.match(r"s(\d+)_(sm|norm)", x.node_name)
                if m and x.assigned_core_id == simd:
                    key = (x.assigned_chiplet_id, x.assigned_cluster_id, int(m.group(1)))
                    if key not in self._sweep_end or x.node_id > self._sweep_end[key].node_id:
                        self._sweep_end[key] = x
        if (k, c, h - 1) not in self._sweep_end:
            raise ValueError(f"params kx_after_chains: no softmax of chunk {h - 1} on chip {k:02x} cluster {c}")
        return self._sweep_end[(k, c, h - 1)]

    def _kx_vload_of(self, k, c, h):
        """kx_after_chains: cluster (k, c)'s load of chunk h - 1's key tile yr_store_tile's values (its DM core)."""
        rx = re.compile(rf"Ld_V{int(self.yr_store_tile)}_i{h - 1}_k{k:02x}_cl{c}$")
        at = [x for x in self.dfg.node_list if x.assigned_chiplet_id == k and x.assigned_cluster_id == c
              and rx.match(x.node_name)]
        if not at:
            raise ValueError(f"params kx_after_chains: no value load of tile {self.yr_store_tile} of chunk {h - 1} on "
                             f"chip {k:02x} cluster {c}")
        return max(at, key=lambda x: x.node_id)

    def _kx_per_chunk_edges(self):
        """params kx_per_chunk: on every cluster, the loads that start chunk h's key-tile buffers (h >= 1, inside the
        window: its first two tiles' x loads, its W_KVA, helped-pair and key-constant copies) depend on the cluster's last q
        projection start of chunk h - 1 (its x rows' last reader; the projections run in order on the GEMM core) and,
        on a collector of chunk h - 2, on its store of that chunk's y block (with yr_store_tile)."""
        PC = self.C // 16
        first = re.compile(r"^(kx_ld|kx_xld|Ld_wkvc|Ld_wkvxl|kvc_lld_g\d+)_q(\d+)_k")
        by_cl = {}
        for t in self.dfg.node_list:
            by_cl.setdefault((t.assigned_chiplet_id, t.assigned_cluster_id), []).append(t)
        n = 0
        for (k, c), ts in by_cl.items():
            loads = {}
            for t in ts:
                m = first.match(t.node_name)
                if m and 1 <= int(m.group(2)) // PC < self.NCH:
                    loads.setdefault(int(m.group(2)) // PC, []).append((int(m.group(2)), t))
            for h, lst in loads.items():
                q2 = sorted({q for q, t in lst if t.node_name.startswith("kx_")})[:2]
                gq = [t for t in ts if re.match(rf"^gq\d+_i{h - 1}_k", t.node_name)]
                if not gq:
                    raise ValueError(f"params kx_per_chunk: no q projection of chunk {h - 1} on chip {k:02x} cluster {c}")
                srcs = [max(gq, key=lambda t: t.node_id)]
                st = self._kx_store_of(k, c, h)
                if st is not None:
                    # a collector of chunk h - 2: also behind its store of that chunk's y block (program order above)
                    srcs.append(st)
                if self.kx_after_chains and h >= 2:
                    # behind the last reduce chain of chunk h - 2 (the chains run one at a time unless chain_concurrent)
                    ch = [x for x in self._chains if re.match(rf"y_chain_c\d+_i{h - 2}_h\d+_", x.node_name)]
                    if not ch:
                        raise ValueError(f"params kx_after_chains: no chain of chunk {h - 2}")
                    srcs += ch if self.chain_concurrent else ch[-1:]
                for q, t in lst:
                    if q in q2 or not t.node_name.startswith("kx_"):
                        for src in srcs:
                            self.dfg.bingo_add_edge(src, t)
                            n += 1
        print(f"[{self.TAG}] kx_per_chunk: {n} key-buffer loads behind their cluster's last q projection of the chunk "
              f"before")

    KX_TRANSFER = r"(kvc_put|kvx_put|Ld_wkvc_q|Ld_wkvxl_q|kvc_lld_g\d+_q|kv_put_t)"

    def _kx_before_chains(self):
        """kx_after_chains: chunk h's key transfers (h >= 1: its HBM copies, its puts to the owners) ahead of chunk h - 1's
        reduce chains: each one an edge to every task of its chip, of that chunk, that those chains wait for (a cluster's
        last W_O GEMM of the chunk), so no key transfer of the chip runs beside a chain of the chunk before -- unless that task
        already precedes the transfer (a ring load behind the GEMM that frees its slot, which the chain's own release
        of the ring orders after the chain). With the edges of _kx_per_chunk_edges (behind chunk h - 2's chains) every
        key transfer is ordered with every chain (_check_multicasts proves it)."""
        import networkx as nx
        PC = self.C // 16
        q_of = {t: q for q, t in getattr(self, "kv_put", {}).items()}
        feeds = {}                                  # (chunk, chip) -> the chains' inputs on that chip
        for ch in self._chains:
            m = re.match(r"y_chain_c\d+_i(\d+)_", ch.node_name)
            if m:
                i = int(m.group(1))
                for pr in self.dfg.predecessors(ch):
                    if f"_i{i}_" in pr.node_name:   # the chunk's own work (not a collector's store of the one before)
                        feeds.setdefault((i, pr.assigned_chiplet_id), set()).add(pr)
        bit = {pr: 1 << i for i, pr in enumerate(sorted({p for ps in feeds.values() for p in ps},
                                                         key=lambda x: x.node_id))}
        g = self.dfg.bingo_order_view()
        reach = {}                                  # task -> the bits of the chains' inputs that precede it
        for v in nx.topological_sort(g):
            r = bit.get(v, 0)
            for u in g.predecessors(v):
                r |= reach[u]
            reach[v] = r
        n = 0
        for t in list(self.dfg.node_list):
            if not re.match(self.KX_TRANSFER, t.node_name):
                continue
            m = re.search(r"_q(\d+)_", t.node_name)
            q = int(m.group(1)) if m else q_of.get(t)
            if q is None or not 1 <= q // PC < self.NCH:
                continue
            for pr in sorted(feeds.get((q // PC - 1, t.assigned_chiplet_id), ()), key=lambda x: x.node_id):
                if pr is not t and not reach[t] & bit[pr] and not self.dfg.has_edge(t, pr):
                    self.dfg.bingo_add_edge(t, pr)
                    n += 1
        print(f"[{self.TAG}] kx_after_chains: {n} edges, each key transfer ahead of the chunk before's chains")

    def _keys_by_chunk_order(self):
        """params keys_by_chunk: the key tasks of the chunks after the last one made (stage 1 by its tiles' names, stages 2
        and 3 by the runs stage_owner recorded) run after the window: each of them that no other one of them precedes
        waits for the end stamp. On the root, whose L1 is the fullest, the static L1 planner can then lay their buffers
        over the window's; everywhere, their transfers stay out of the window's last chains, which run two at a time
        (last_pair_chains) and must not meet another chip's traffic on a hop."""
        runs = sorted(self._key_runs, key=lambda r: r[0])      # stable: a run's end before the next run's start
        for (a, h), (b, _) in zip(runs, runs[1:]):
            if h is None:
                continue
            for t in self.dfg.node_list:
                if a <= t.node_id < b:
                    t._key_chunk = h
        end_stamp = next(t for t in self.dfg.node_list if t.node_name.startswith("S1_end"))
        after = {t for t in self.dfg.node_list
                 if getattr(t, "_key_chunk", 0) >= self.NCH and t.node_id < end_stamp.node_id}
        preds = {t: set(self.dfg.predecessors(t)) for t in after}
        for x, y in self.dfg.order_edges:
            if y in preds:
                preds[y].add(x)
        heads = [t for t in after if not preds[t] & after]
        for t in sorted(heads, key=lambda t: t.node_id):
            self.dfg.bingo_add_edge(end_stamp, t)
        if heads:
            print(f"[{self.TAG}] keys_by_chunk: {len(after)} key tasks after the window, {len(heads)} behind the end "
                  f"stamp")

    def stage_owner(self, go, last, chained=False, end_wait=True):
        """Stage 2: FaBuild's stage_kv on every chip, its kvr_waits behind the chip's start (the flags are already
        set), then the root's waits for every piece, a chunk at a time, on the cluster that multicasts them in stage 3.
        chained (stage "0123"): each tile starts behind stage 1's owner waits (own_wait), and the root multicasts each
        chunk once its waits are done (stage 3); every other chip then waits for both chunks. The gather chip is
        params s3_root (default the root)."""
        R, T, dm, P = self.kv_root, self.TAG, self.roles["dm"], self.C // 16
        self._chained = chained
        if not chained:
            self.kvc_put = {(q, k2): go[k2] for q in range(self.NX) for k2 in self.chips}
        self.kvp_put = {}
        for k in self.chips:
            if self.kv_split or self.kv_fused or self.kv_norm_cm:
                self.stage_kv_split(k)
            elif self.keys_by_chunk:
                raise ValueError("params keys_by_chunk: needs kv_split_clusters (stage_kv_split's runs a tile)")
            else:
                self.stage_kv(k)
            self._key_runs.append((self._next_id(), None))
            last[k] += [self.kv_put[q] for q in self.kv_tiles[k]] + [self.kvp_put[q] for q in self.kv_tiles[k]
                                                                    if q in self.kvp_put]
        c, prev = self.x_put_cluster, []
        self.kv_mc = []
        waits = [("kv_nwait", "kvnflag", self.kv_put)] + ([("kv_pwait", "kvpflag", self.kvp_put)] if self.kv_split else [])
        for h in range(self.NCH):
            self._key_runs.append((self._next_id(), h))
            for name, flag, src in waits:
                puts = [src[q] for q in range(h * P, (h + 1) * P)]
                w = self.node(R, c, dm, f"{name}_h{h}", "__snax_bingo_kernel_wait_flags",
                              SnaxBingoKernelWaitFlagsArgs(BingoMemSymbol(f"{T}_{flag}", offset=h * P * 64), P, 1),
                              [t for t in puts if self.flag_dep(t, R, c)] + prev)
                for t in puts:
                    if not self.flag_dep(t, R, c):
                        self.dfg.bingo_add_order_edge(t, w)
                prev = [w]
                last[R].append(w)
            self.root_ready = getattr(self, "root_ready", {})
            self.root_ready[h] = prev[0]
            if chained:
                if self.keys_by_chunk and self.x_mem:
                    # one multicast source at a time: chunk 0's keys after chunk 1's x has landed, the keys of the
                    # chunks after chunk 1 after the rest of x (x_wait: the flag proves the multicast's data)
                    t = 2 * P - 1 if h == 0 else self.NX - 1 if h >= (1 if self.x_rest_early else 2) else None
                    if t is not None and t < self.NX:
                        prev = prev + [self.x_wait(R, c, t, go, name=f"x_wait_h{h}")]
                # stage 3: the chunk's pieces from the root's L3 into every other chip's, one multicast at a time
                src = BingoMemSymbol(f"{T}_kv", offset=h * P * KVP)
                self.kv_mc.append(self.node(R, c, dm, f"kv_bput_h{h}", "__snax_bingo_kernel_idma_put_flagged",
                                            SnaxBingoKernelIdmaPutFlaggedArgs(src, src, P * KVP,
                                                                              BingoMemSymbol(f"{T}_kvflag", offset=h * 64),
                                                                              value=1, bcast=True), prev))
                prev = self.kv_mc[-1:]
                last[R].append(self.kv_mc[-1])
        self._key_runs.append((self._next_id(), None))
        if chained and end_wait:
            for k in self.chips:
                if k != R:
                    last[k].append(self.kv_all(k))

    def kv_all(self, k):
        """Chip k's wait (cluster 0's DM core) for both chunks' keys from the stage-3 multicasts."""
        w = self.node(k, 0, self.roles["dm"], "kv_all", "__snax_bingo_kernel_wait_flags",
                      SnaxBingoKernelWaitFlagsArgs(BingoMemSymbol(f"{self.TAG}_kvflag"), self.NCH, 1), [])
        for t in self.kv_mc:
            self.dfg.bingo_add_order_edge(t, w)
        return w

    def gmm(self, g, name, a, b, d, M, K, N, k_shift, deps):
        # s4_split_x: the q GEMMs read the rows both DMAs copy
        if name.startswith("gq") and self._x_extra.get(g):
            deps = list(deps) + self._x_extra[g]
        return super().gmm(g, name, a, b, d, M, K, N, k_shift, deps)


    def stage_kv_split(self, k):
        """kv_split_clusters / kv_norm_fused: chip k's tiles as two parts. The tile's cluster takes c: its 512 columns
        (the first 16 KB of the tile's D layout) into L1, their dequantisation, RMSNorm, the key rows' c and the value
        columns. k_pe (the last 2 KB) is dequantised and rotated (RoPE, one simd_rope with kv_rope_fused) into the key
        rows' k_pe: with kv_split_clusters on the chip's other cluster, put into the slot behind a flag of its own
        (kvpflag) after the tile's cluster has copied c and put the value columns behind the tile's flag (kvnflag);
        without, on the tile's cluster before c's RMSNorm, and the piece goes in one put. kv_norm_fused: RMSNorm
        row_major -> A (INT8) writes the key rows' c, and RMSNorm col_major -> B (INT8) the value columns from c
        transposed (B of c [16, 512] is A of c^T), in place of the reshapes and quantisations. kv_vals_other: that
        second RMSNorm on the other cluster, which then puts k_pe and the value columns in one put. kv_norm_cm: c
        transposed first and its RMSNorm col_major -> col_major (FP16), the reshapes and quantisations after."""
        d, T, mesh = self.d, self.TAG, self.d.mesh
        simd, xdma, dm = self.roles["simd"], self.roles["xdma"], self.roles["dm"]
        R, nch, KR, RP = self.kv_root, len(self.chips), fa.KV_RANK, fa.ROPE  # the gather chip (s3_root)
        on_root = lambda name, off: BingoMemSymbol(name, offset=off, chip_id=None if k == R else R)
        cb, n_pe, half = (KR // 16) * TILE16, 16 * RP, (RP // 32) * TILE16
        sc = self.fbits(d.s_c.inv)
        if self.kv_fused:
            c8 = np.arange(16 * KR).reshape(16, KR)
            assert np.array_equal(d.layout.to_b(c8, mesh), d.layout.to_a(np.ascontiguousarray(c8.T), mesh))
        # keys_by_chunk: the tiles after the window read their own copies of the clusters' constants, made after the
        # window (the first DM task of the first such tile), so the window's copies are free once chunk 1's keys are made
        late_from = int(self.chunks_built or self.NCH) * (self.C // 16) if self.keys_by_chunk else self.NX
        self._kvc_late = getattr(self, "_kvc_late", {})
        self._kvc_late_ld = getattr(self, "_kvc_late_ld", {})      # (k, cluster, chunk) -> the copy that fills it
        for lt, q in enumerate(self.kv_tiles[k]):
            self._key_runs.append((self._next_id(), q // (self.C // 16)))
            s = "" if lt == 0 else f"_t{lt}"
            c0 = self.kv_cluster(k, q)
            c1 = (c0 + 1) % self.NC if self.kv_split else c0
            # kx_per_chunk: every chunk after the first reads copies of its own too (made with its key tasks, behind
            # the chunk before's q projections), so the window's copies are free once chunk 0's keys are made
            hk = q // (self.C // 16) if self.kx_per_chunk and q // (self.C // 16) >= 1 else None
            late = q >= late_from or hk is not None
            if late:
                for cc, src in ((c0, self.kvc), (c1, self.kvc_pe if self.kv_split else self.kvc)):
                    if (k, cc, hk) not in self._kvc_late and (k, cc) in src:
                        g = self.chips.index(k) * self.NC + cc
                        buf = self.cx_chip(k).at(cc).l1(f"{T}_lkvc{'pe' if src is self.kvc_pe else ''}"
                                                        + ("" if hk is None else f"_h{hk}"), self.KC_BYTES)
                        # kx_per_chunk: from the chip's L3 copy (_kx_l3_preload), never HBM inside the window
                        self._kvc_late_ld[(k, cc, hk)] = self.dm_copy(
                            g, f"kvc_lld_g{g}" + ("" if hk is None else f"_q{q}"),
                            self.h_kc[k].address if hk is None or not self.kx_l3 else BingoMemSymbol(f"{T}_kc3"),
                            buf, self.KC_BYTES, [])
                        self._kvc_late[(k, cc, hk)] = buf
            kvcp = (lambda cc: self._kvc_late[(k, cc, hk)]) if late else (lambda cc: self.kvc_pe[(k, cc)])
            # the tile's loads (and the RoPE table copy) behind the copy of the constants their readers read
            kdep = (lambda cc: [self._kvc_late_ld[(k, cc, hk)]] if late and (k, cc, hk) in self._kvc_late_ld else [])
            g0, g1 = self.chips.index(k) * self.NC + c0, self.chips.index(k) * self.NC + c1
            puts = [self.kvc_put[(q, k2)] for k2 in self.chips]

            def wait(c, name):
                # stage 1's slices of the tile have landed: both clusters read them (chained: stage 1's own waits)
                if self._chained:
                    return list(self.own_wait[q])
                w = self.node(k, c, dm, f"{name}{s}", "__snax_bingo_kernel_wait_flags",
                              SnaxBingoKernelWaitFlagsArgs(BingoMemSymbol(f"{T}_kvrflag", offset=lt * nch * 64), nch, 1),
                              [t for t in puts if self.flag_dep(t, k, c)])
                for t in puts:
                    if not self.flag_dep(t, k, c):
                        self.dfg.bingo_add_order_edge(t, w)
                return [w]
            src = BingoMemSymbol(f"{T}_kvraw", offset=lt * self.KVD)
            # a chip's later tiles (more tiles than chips) take buffers of their own names: one handle a buffer
            l0 = lambda name, nb, s=s: self.cx_chip(k).at(c0).l1(f"{T}_kvs_{name}{s}", nb)
            l1 = lambda name, nb, s=s: self.cx_chip(k).at(c1).l1(f"{T}_kvp_{name}{s}", nb)
            kvd, crow, piece = l0("d", cb), l0("crow", 16 * KR * 2), l0("piece", KVP)
            # kv_vals_other: [k_pe | value columns] on the other cluster, the slot's bytes 16 KR .. KVP
            pv = l1("pv", KVP - 16 * KR) if self.kv_vals_other else None
            kvc = self._kvc_late[(k, c0, hk)] if late else self.kvc[(k, c0)]
            w0 = wait(c0, "kvr_wait")
            w1 = wait(c1, "kvp_wait") if self.kv_split else w0
            pre = {}
            ld = self.dm_copy(g0, f"kvr_ld{s}", src, kvd, cb, list(w0) + kdep(c0))
            dq = self.simd_mul(g0, f"kv_dq{s}", kvd, kvc, kvd, 16 * KR, [ld])
            cv = self.xd(g0, f"kv_crm{s}", fa.convert_args("D", "row_major", 16, KR, mesh, 2, kvd, crow), [dq])

            def c_path():
                # c: the key rows' c into the piece, the value columns at KB
                if self.kv_fused:
                    beat = fa.simd_fp16_lanes() * 2
                    cv_, lv = (c1, l1) if self.kv_vals_other else (c0, l0)     # the values' cluster and its L1
                    # [seed beat | c^T [512, 16] FP16]; kv_vals_own built it already
                    xt = pre["xtbuf"] if self.kv_vals_own else lv("xt", beat + 16 * KR * 2)
                    if self.kv_vals_own:
                        tr = pre["xt"]
                    elif self.kv_vals_other:
                        # the xDMA transposes into its own cluster's L1 (a transpose written into another cluster
                        # never completes), the iDMA copies c^T over
                        xt0 = l0("xt0", 16 * KR * 2)
                        t0_ = self.node(k, c0, xdma, f"kv_vt{s}", "__snax_bingo_kernel_xdma_transpose_2d",
                                        fa.SnaxBingoKernelXdmaTranspose2dArgs(crow, xt0, 16, KR, 2), [cv])
                        tr = self.node(k, c0, dm, f"kv_vcp{s}", "__snax_bingo_kernel_idma_1d_copy",
                                       SnaxBingoKernelIdma1dCopyArgs(xt0, xt.view(beat), 16 * KR * 2), [t0_])
                    else:
                        tr = self.node(k, c0, xdma, f"kv_vt{s}", "__snax_bingo_kernel_xdma_transpose_2d",
                                       fa.SnaxBingoKernelXdmaTranspose2dArgs(crow, xt.view(beat), 16, KR, 2), [cv])
                    ra = lambda i, o, la, sd=0: fa.SnaxBingoKernelSimdRmsnormArgs(
                        i, o, 16, KR, "row_major" if not sd else "col_major", "A" if not sd else "B", seed_addr=sd,
                        out_i8=True, inv_scale_f32bits=sc, mesh=mesh, scratch_addr=la(
                            "scrb" if sd else "scra", fa.SnaxBingoKernelSimdRmsnormArgs(
                                0, 0, 16, KR, "col_major" if sd else "row_major", "B" if sd else "A", seed_addr=1,
                                out_i8=True, mesh=mesh).scratch_bytes()))
                    kq = self.node(k, c0, simd, f"kv_rmsa{s}", "__snax_bingo_kernel_simd_rmsnorm", ra(crow, piece, l0),
                                   [cv])
                    vo = pv.view(n_pe) if self.kv_vals_other else piece.view(fa.KB)
                    vq = self.node(k, cv_, simd, f"kv_rmsb{s}", "__snax_bingo_kernel_simd_rmsnorm",
                                   ra(xt.view(beat), vo, lv, xt), [tr])
                    return kq, vq
                cu, ca, vt, va = (l0(nm, 16 * KR * 2) for nm in ("cu", "ca", "vt", "va"))
                if self.kv_norm_cm:
                    beat = fa.simd_fp16_lanes() * 2
                    xt = l0("xt", beat + 16 * KR * 2)          # [seed beat | c^T [512, 16] FP16]
                    tr = self.node(k, c0, xdma, f"kv_ct{s}", "__snax_bingo_kernel_xdma_transpose_2d",
                                   fa.SnaxBingoKernelXdmaTranspose2dArgs(crow, xt.view(beat), 16, KR, 2), [cv])
                    scr = l0("scrc", fa.SnaxBingoKernelSimdRmsnormArgs(
                        0, 0, 16, KR, "col_major", "col_major", seed_addr=1).scratch_bytes())
                    rn = self.node(k, c0, simd, f"kv_rms{s}", "__snax_bingo_kernel_simd_rmsnorm",
                                   fa.SnaxBingoKernelSimdRmsnormArgs(xt.view(beat), vt, 16, KR, "col_major", "col_major",
                                                                     seed_addr=xt, scratch_addr=scr), [tr])
                    # vt = y^T, row major [512, 16]: the value columns as before; the key rows' c from y
                    vv = self.xd(g0, f"kv_va{s}", fa.convert_args("row_major", "A", KR, 16, mesh, 2, vt, va), [rn])
                    tb = self.node(k, c0, xdma, f"kv_yt{s}", "__snax_bingo_kernel_xdma_transpose_2d",
                                   fa.SnaxBingoKernelXdmaTranspose2dArgs(vt, cu, KR, 16, 2), [rn])
                    ka = self.xd(g0, f"kv_ca{s}", fa.convert_args("row_major", "A", 16, KR, mesh, 2, cu, ca), [tb])
                    vq = self.quant(g0, f"kv_v8{s}", va, piece.view(fa.KB), 16 * KR * 2, sc, [vv])
                    kq = self.quant(g0, f"kv_c8{s}", ca, piece, 16 * KR * 2, sc, [ka])
                    return kq, vq
                scr = l0("scr", fa.SnaxBingoKernelSimdRmsnormArgs(0, 0, 16, KR).scratch_bytes())
                rn = self.node(k, c0, simd, f"kv_rms{s}", "__snax_bingo_kernel_simd_rmsnorm",
                               fa.SnaxBingoKernelSimdRmsnormArgs(crow, cu, 16, KR, scratch_addr=scr), [cv])
                ka = self.xd(g0, f"kv_ca{s}", fa.convert_args("row_major", "A", 16, KR, mesh, 2, cu, ca), [rn])
                kq = self.quant(g0, f"kv_c8{s}", ca, piece, 16 * KR * 2, sc, [ka])
                tr = self.node(k, c0, xdma, f"kv_vt{s}", "__snax_bingo_kernel_xdma_transpose_2d",
                               fa.SnaxBingoKernelXdmaTranspose2dArgs(cu, vt, 16, KR, 2), [rn])
                vv = self.xd(g0, f"kv_va{s}", fa.convert_args("row_major", "A", KR, 16, mesh, 2, vt, va), [tr])
                vq = self.quant(g0, f"kv_v8{s}", va, piece.view(fa.KB), 16 * KR * 2, sc, [vv])
                return kq, vq

            def pe_path():
                # k_pe: its own load and dequantisation, RoPE, D -> A, quantised; on c1 (c0 without kv_split_clusters)
                kvc1 = kvcp(c1) if self.kv_split else kvc
                pd, pe, pa = l1("d", n_pe * 2), l1("pe", n_pe * 2), l1("pa", n_pe * 2)
                p8 = pv if self.kv_vals_other else l1("p8", n_pe) if self.kv_split else piece.view(16 * KR)
                rc = self.KC_ROPE + lt * self.KC_TILE
                kd1 = kdep(c1) if self.kv_split else kdep(c0)
                ld1 = self.dm_copy(g1, f"kvp_ld{s}", BingoMemSymbol(f"{T}_kvraw", offset=lt * self.KVD + cb), pd,
                                   n_pe * 2, list(w1) + kd1)
                if self.kv_rope_fused:
                    # the dequantisation writes x into slot 0 of [x | cos | xswap | sin]; the tables come from kvc
                    rb, r4 = n_pe * 2, l1("rope4", 4 * n_pe * 2)
                    tb = self.node(k, c1, dm, f"kvp_tab{s}", "__snax_bingo_kernel_idma_2d_copy",
                                   fa.SnaxBingoKernelIdma2dCopyArgs(kvc1.view(rc), r4.view(rb), rb, rb, 2 * rb, 2), kd1)
                    dq1 = self.simd_mul(g1, f"kvp_dq{s}", pd, kvc1.view(cb), r4, n_pe, [ld1])
                    sw = [self.node(k, c1, dm, f"kvp_swap{h}{s}", "__snax_bingo_kernel_idma_1d_copy",
                                    SnaxBingoKernelIdma1dCopyArgs(r4.view((1 - h) * half), r4.view(2 * rb + h * half),
                                                                  half), [dq1]) for h in range(2)]
                    ad = self.node(k, c1, simd, f"kvp_rope{s}", "__snax_bingo_kernel_simd_rope",
                                   fa.SnaxBingoKernelSimdRopeArgs(r4, pe, RP, 16), [dq1, tb] + sw)
                else:
                    xs, t1, t2 = l1("xs", n_pe * 2), l1("t1", n_pe * 2), l1("t2", n_pe * 2)
                    dq1 = self.simd_mul(g1, f"kvp_dq{s}", pd, kvc1.view(cb), pd, n_pe, [ld1])
                    sw = [self.node(k, c1, dm, f"kvp_swap{h}{s}", "__snax_bingo_kernel_idma_1d_copy",
                                    SnaxBingoKernelIdma1dCopyArgs(pd.view((1 - h) * half), xs.view(h * half), half),
                                    [dq1]) for h in range(2)]
                    m1 = self.simd_mul(g1, f"kvp_cos{s}", pd, kvc1.view(rc), t1, n_pe, [dq1])
                    m2 = self.simd_mul(g1, f"kvp_sin{s}", xs, kvc1.view(rc + n_pe * 2), t2, n_pe, sw)
                    ad = self.simd_add(g1, f"kvp_rope{s}", t1, t2, pe, n_pe, [m1, m2])
                d2p = self.xd(g1, f"kvp_pa{s}", fa.d_to_a_args(mesh, 1, 4, 16, pe, pa), [ad])
                return p8, d2p, lambda: self.quant(g1, f"kvp_p8{s}", pa, p8, n_pe * 2, self.fbits(d.s_kpe.inv), [d2p])

            if self.kv_vals_own:
                # kv_vals_own: c^T built on the values' cluster first (its SIMD's long dequantisation ahead of k_pe)
                beat = fa.simd_fp16_lanes() * 2
                kvd1, crow1, xt1 = l1("cd", cb), l1("crow", 16 * KR * 2), l1("xt", beat + 16 * KR * 2)
                ldv = self.dm_copy(g1, f"kvv_ld{s}", src, kvd1, cb, list(w1) + kdep(c1))
                dqv = self.simd_mul(g1, f"kvv_dq{s}", kvd1, kvcp(c1), kvd1, 16 * KR, [ldv])
                cvv = self.xd(g1, f"kvv_crm{s}", fa.convert_args("D", "row_major", 16, KR, mesh, 2, kvd1, crow1), [dqv])
                pre["xtbuf"] = xt1
                pre["xt"] = self.node(k, c1, xdma, f"kvv_vt{s}", "__snax_bingo_kernel_xdma_transpose_2d",
                                      fa.SnaxBingoKernelXdmaTranspose2dArgs(crow1, xt1.view(beat), 16, KR, 2), [cvv])
            if self.kv_vals_other:
                # the other cluster's in-order SIMD core: k_pe first (its inputs are ready first), then the values
                _, _, p8_ = pe_path()
                pq = p8_()
                kq, vq = c_path()
                self.kv_put[q] = self.node(k, c0, dm, f"kv_put{s}", "__snax_bingo_kernel_idma_put_flagged",
                                           SnaxBingoKernelIdmaPutFlaggedArgs(piece, on_root(f"{T}_kv", q * KVP), 16 * KR,
                                                                             on_root(f"{T}_kvnflag", q * 64), value=1),
                                           [kq])
                self.kvp_put[q] = self.node(k, c1, dm, f"kvp_put{s}", "__snax_bingo_kernel_idma_put_flagged",
                                            SnaxBingoKernelIdmaPutFlaggedArgs(
                                                pv, on_root(f"{T}_kv", q * KVP + 16 * KR), KVP - 16 * KR,
                                                on_root(f"{T}_kvpflag", q * 64), value=1), [pq, vq])
            elif self.kv_split:
                kq, vq = c_path()
                # the key rows' c, then the value columns behind the tile's flag: one iDMA, in that order
                pc = self.node(k, c0, dm, f"kv_cput{s}", "__snax_bingo_kernel_idma_1d_copy",
                               SnaxBingoKernelIdma1dCopyArgs(piece, on_root(f"{T}_kv", q * KVP), 16 * KR), [kq])
                self.kv_put[q] = self.node(k, c0, dm, f"kv_put{s}", "__snax_bingo_kernel_idma_put_flagged",
                                           SnaxBingoKernelIdmaPutFlaggedArgs(
                                               piece.view(fa.KB), on_root(f"{T}_kv", q * KVP + fa.KB), KR * 16,
                                               on_root(f"{T}_kvnflag", q * 64), value=1), [vq, pc])
                p8, _, p8_ = pe_path()
                pq = p8_()
                self.kvp_put[q] = self.node(k, c1, dm, f"kvp_put{s}", "__snax_bingo_kernel_idma_put_flagged",
                                            SnaxBingoKernelIdmaPutFlaggedArgs(
                                                p8, on_root(f"{T}_kv", q * KVP + 16 * KR), n_pe,
                                                on_root(f"{T}_kvpflag", q * 64), value=1), [pq])
            else:
                # one cluster: RoPE first (the in-order SIMD core takes it while the xDMA reorders c), k_pe's
                # quantisation after c's first RMSNorm, the whole piece in one put
                _, _, p8_ = pe_path()
                kq, vq = c_path()
                pq = p8_()
                self.kv_put[q] = self.node(k, c0, dm, f"kv_put{s}", "__snax_bingo_kernel_idma_put_flagged",
                                           SnaxBingoKernelIdmaPutFlaggedArgs(piece, on_root(f"{T}_kv", q * KVP), KVP,
                                                                             on_root(f"{T}_kvnflag", q * 64), value=1),
                                           [kq, vq, pq])

    # ---- stage "xu": the xDMA's load L3 -> L1 alone --------------------------------------------------------------
    def stage_xdma_unit(self, go, last):
        """Every DM core copies 64 bytes of the chip's x buffer into L1 (the core-order check wants an x task on each);
        then the xDMA of each listed (chip, cluster) loads xu_bytes of it into the same L1 buffer."""
        T, dm, xdma = self.TAG, self.roles["dm"], self.roles["xdma"]
        for k in self.chips:
            for c in range(self.NC):
                buf = self.cx_chip(k).at(c).l1(f"{T}_xu", max(self.xu_bytes, 64))
                cp = self.node(k, c, dm, "x_copy_unit", "__snax_bingo_kernel_idma_1d_copy",
                               SnaxBingoKernelIdma1dCopyArgs(BingoMemSymbol(f"{T}_xsrc"), buf, 64), [go[k]])
                last[k].append(cp)
                if k in self.xu_chips and c in self.xu_clusters:
                    last[k].append(self.node(k, c, xdma, "kx_xld_unit", "__snax_bingo_kernel_xdma_1d_copy",
                                             SnaxBingoKernelXdma1dCopyArgs(BingoMemSymbol(f"{T}_xsrc", offset=self.XT),
                                                                           buf, self.xu_bytes), [cp]))

    # ---- the build -----------------------------------------------------------------------------------------------
    def build(self):
        if self.stage in ("full", "678"):
            # the whole block: stages 0 to 3 as the stage builder makes them, then FaBuild's chunks (build_full);
            # 678: FaBuild's chunks from W_UV on
            self.last_ld = {}
            self._x_extra = {}
            return self.build_full()
        R = self.ROOT
        self.last_ld = {}
        for ls in self.ls:
            ls.strict = True
        go = self.setup(extra=self.stage1_preload() if self.stage == 2 else
                        self.stage3_preload() if self.stage == 3 else
                        self.stage1_preload(kvr=False) if self.stage == "0123" else
                        self.stage4_preload() if self.stage == 4 else
                        self.stage5_preload() if self.stage == 5 else None)
        if self.p.get("relay_per_gather", True):
            self.dfg.relay_one_consumer_cores = [self.roles["xdma"]]
        # params relay_prefer {consumer role: [roles]}: dep-tag relays of that role's tasks go to those cores first; a
        # relay joins its core's queue just before the first task there that its consumer reaches, so on a core busy to
        # the end of the sweep (the SIMD core) it holds the chain back until the sweep ends
        if self.p.get("relay_prefer"):
            self.dfg.relay_prefer_cores = {self.roles[c]: [self.roles[r] for r in rs]
                                           for c, rs in self.p["relay_prefer"].items()}
        self.x_puts = self.stage_x_bcast(go) if self.stage not in ("xu", 2, 3, 4, 5) else []
        if self.prefetch_after_x and self.x_puts:
            # the memory chips push only once the multicast of x is done: the root's own copy of a tile, read from HBM
            # over the link the root's ring pushes also arrive on, otherwise holds the root's L3 write port while its
            # data waits behind a push that needs that port
            for k in self.chips:
                self.edge(self.x_puts[-1], self.prefetch[k])
        elif self.x_mem and self.x_puts:
            # the root's prefetcher drives the same engine of the same memory chip: one runs after the other
            if self.bcast_after_pf:
                self.edge(self.prefetch[R], self.x_puts[-1])
            else:
                if self.pf_split:
                    self.edge(self.split_root_prefetch(go), self.x_puts[0])
                self.edge(self.x_puts[-1], self.prefetch[R])
        last = {k: [] for k in self.chips}
        if self.stage == "xu":
            self.stage_xdma_unit(go, last)
        elif self.stage == 0:
            for k in self.chips:
                for c in range(self.NC):
                    last[k] += [self.x_wait(k, c, t, go) for t in self.cl_tiles(c)]
        elif self.stage == 2:
            self.stage_owner(go, last)
        elif self.stage == 3:
            self.stage_keys_out(go, last)
        elif self.stage == "0123":
            self.stage_keys(go, last)
            self.stage_owner(go, last, chained=True)
        elif self.stage == 4:
            self.stage_queries(go, last)
        elif self.stage == 5:
            self.stage_sweep(go, last)
        else:
            self.stage_keys(go, last)
        # one probe a chip behind its last tasks, the root's behind its multicasts too
        ends = [self.probe(k, f"fin{k:02x}", last[k]) for k in self.chips if k != R]
        ends.append(self.probe(R, f"fin{R:02x}", last[R] + self.x_puts[-1:]))
        self.end_window(ends)
        self.stage_checks()

    def stage_checks(self):
        T, R, d = self.TAG, self.ROOT, self.d
        n, checks = 2048, []
        on = lambda k: None if k == R else k
        if self.stage == "xu":
            # the unit test checks nothing: it runs to its end or hangs
            self.run_checks([])
            return
        for k in dict.fromkeys((R, self.order[len(self.order) // 2], self.order[-1])):
            for t in (0, self.NX // 2, self.NX - 1):
                off = t * self.XT + (self.XT - n if t == self.NX - 1 else 0)
                checks.append((f"x_k{k:02x}_t{t}", BingoMemSymbol(f"{T}_xsrc", offset=off, chip_id=on(k)),
                               self.xsrc[off:off + n].view(np.uint8), n))
        if self.stage == 5:
            # o~ of the first and the last query tile of both clusters of the chip in the middle of the fold order, the
            # last chunk (the checks copy after the window, when an earlier chunk's buffers may hold the next chunk's)
            checks, far = [], self.order[len(self.order) // 2]
            h = self.head[far]
            for i in (self.NCH - 1,):
                for c in range(self.NC):
                    g = self.chips.index(far) * self.NC + c
                    sw = self.sw_of[(g, i)]
                    for q in sorted({0, self.NQ - 1}):
                        r0 = self.r0(i, c) + q * fa.BR
                        tag = f"i{i}c{c}q{q}"
                        ot, nb = self.xt_ref(h, r0)
                        checks.append((f"xt_{tag}", self.stash(f"xt_{tag}", far, c, sw["xt"][q], nb, [sw["tr"][q]]),
                                       ot, nb))
            self.run_checks(checks)
            return
        if self.stage == 4:
            # the first and the last query tile of both clusters of the chip in the middle of the fold order, the last
            # chunk (the checks copy after the window, when an earlier chunk's buffers may hold the next chunk's)
            checks, far = [], self.order[len(self.order) // 2]
            h = self.head[far]
            for i in (self.NCH - 1,):
                for c in range(self.NC):
                    g = self.chips.index(far) * self.NC + c
                    for q in sorted({0, self.NQ - 1}):
                        r0 = self.r0(i, c) + q * fa.BR
                        tag = f"i{i}c{c}q{q}"
                        handle, ts = self.q8_of[(g, i)][q]
                        checks.append((f"q8_{tag}", self.stash(f"q8_{tag}", far, c, handle, fa.D_QK * fa.BR, list(ts)),
                                       d.q8_a(h, r0).view(np.uint8), fa.D_QK * fa.BR))
            self.run_checks(checks)
            return
        if self.stage in (3, "0123"):
            # stage "0123": x as stages 0 and 1, then these
            if self.stage == 3:
                checks = []
            # three tiles' pieces in the L3 of the root, of the chip in the middle of the fold order and of the last
            checks = []
            for k in dict.fromkeys((R, self.order[len(self.order) // 2], self.order[-1])):
                for q in (0, self.NX // 2, self.NX - 1):
                    img = np.asarray(d.kv_piece(16 * q)).view(np.uint8).reshape(-1)
                    for nm, off in (("a", 0), ("p", 16 * fa.KV_RANK), ("b", KVP - n)):
                        checks.append((f"kv_k{k:02x}_q{q}{nm}", BingoMemSymbol(f"{T}_kv", offset=q * KVP + off,
                                                                               chip_id=on(k)), img[off:off + n], n))
            self.run_checks(checks)
            return
        if self.stage == 2:
            # the root's copy of three tiles' pieces: key rows [16, 576] then value columns [512, 16], INT8, A layout
            checks = []
            for q in (0, self.NX // 2, self.NX - 1):
                img = np.asarray(d.kv_piece(16 * q)).view(np.uint8).reshape(-1)
                # kv_split_clusters: also the key rows' k_pe, from the other cluster
                pe = (("p", 16 * fa.KV_RANK),) if self.kv_split else ()
                for nm, off in (("a", 0), ("k", KVP // 2 - n), ("b", KVP - n)) + pe:
                    checks.append((f"kv_q{q}{nm}", BingoMemSymbol(f"{T}_kv", offset=q * KVP + off), img[off:off + n], n))
            self.run_checks(checks)
            return
        if self.stage == 1:
            # the latent rows as the D port leaves them: x8 . W_KVA, FP16, the D layout of each tile's [16, 576]
            raw = np.asarray(d.fp.d_port(d._mm(d.hq, d.wkva.q), d.k_x), dtype=np.float16)
            for q in (0, self.NX // 2, self.NX - 1):
                o = self.kv_owner[q]
                lt = self.kv_tiles[o].index(q)
                img = self.lay.to_d(raw[16 * q:16 * q + 16], d.mesh).astype(np.float16).view(np.uint8).reshape(-1)
                for nm, off in (("a", 0), ("b", self.KVD - n)):
                    checks.append((f"kvraw_q{q}{nm}",
                                   BingoMemSymbol(f"{T}_kvraw", offset=lt * self.KVD + off, chip_id=on(o)),
                                   img[off:off + n], n))
                # kv_balance: each two-pair chip's second pair, made by that chip (first half) or its helper
                for a, ga0 in sorted({(x[0], x[1]) for x in self.kvx.values()}):
                    off, nx = ga0 * TILE16, 2 * TILE16
                    checks.append((f"kvraw_q{q}p{a:02x}",
                                   BingoMemSymbol(f"{T}_kvraw", offset=lt * self.KVD + off, chip_id=on(o)),
                                   img[off:off + nx], nx))
        self.run_checks(checks)


def _app_name(S):
    return (f"DeepSeek-V2-Lite layer 1 MLA attention prefill, stage {S.stage}, {S.T} tokens, {len(S.chips)} chips x "
            f"{S.NC} clusters, x from HBM" + (", split loads" if S.x_xdma else "") + (", xDMA unit test" if S.stage == "xu" else ""))


def run():
    run_driver("fa prefill stage workload generator", FaStages, _app_name)
