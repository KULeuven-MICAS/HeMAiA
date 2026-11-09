# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""The shared driver of the ffn prefill workloads (workloads/ffn/<model>_prefill/<variant>/): one LLM's dense SwiGLU
FFN over a prompt of T tokens, on the (16, 4, 16) GEMM shape with INT4 weights, tensor-parallel over every cluster of
the 16 + 4 chiplet platform. The platform, the weight rings, the stamps and the setup are chiplet_mc.py's; the data and
the golden are ffn_data.py's. ffn_decode_mc.py runs it for one token; the whole-layer drivers
(dsv2/common/dsv2_layer_prefill_mc.py, llama3/full_layer_prefill/common/llama3_layer_mc.py) run it inside a layer. The
stage apps (<model>_prefill/stage*/main_bingo.py) are standalone programs of its stages.

THE MAPPING. Cluster g takes the intermediate columns J_g (whole column blocks), the same columns of gate and of up. The
tokens run in passes of `rows` (T / rows passes). In a pass, every cluster holds the pass's X in L1, and each gate|up
weight chunk meets all of the pass's rows in one GEMM as it streams in (the single pass: each weight is loaded once a
pass). A chip's two clusters then split the down projection by output columns: each takes the chip's columns (K) for its
half of the outputs, so a slice of the down output holds the chip's whole partial and the chips' partials are reduced
by a ChainGather into the root. Per pass:

    X           the pass's rows (INT8, A layout) into every cluster's L1
    gate|up     per weight chunk: GEMM X . [gate | up] chunk -> G (D layout, FP16), dequantised (x the chunk's column
                scales), SwiGLU into A16 (D layout)
    A8          A16 relaid out D -> A on the xDMA, quantised to INT8 into the chip's A8, and the cluster's columns of it
                copied into the other cluster's A8
    down        per weight chunk (a slice of the cluster's outputs): GEMM A8 . down chunk -> the chip's partial of the
                slice, into an output slot of the cluster's symmetric buffer
    reduce      per slice (or group of slices): one ChainGather through every chip, each hop's xDMA adding its partial
                in transit, into the root's fold buffer; the root stores it to L3

X COMES FROM HBM: every pass's X sits in the HBM of the root's memory chip; the root's host
queues that memory chip's push engine (the root's own prefetch engine), which multicasts X a tile at a time into every
chip's L3, each tile behind a flag (__host_bingo_kernel_hbm_bcast). Every pass's X is multicast at the start, ahead of
the root's weight prefetcher, which drives the same engine and holds the host for the whole window. Every cluster
copies a pass's X from its own L3 into L1 once the pass's flags have landed.

THE WEIGHTS. Each cluster's gate|up and down chunks sit in its memory chip's HBM; the memory chip pushes them into the
chip's L3 ring and the DM core loads each into an L1 slab behind its landed flag. The first pass loads them from the
ring; a later pass copies each from the slot it landed in (the ring holds every chunk of a cluster), or with params
restream loads them from the ring again (weights larger than the ring).

THE PROGRAM ORDER. Every device core runs its tasks in the order this driver creates them (chain_program_order).

TIMING. The start stamp after every chip's setup (the scales loaded); the end stamp after the root's last stores. X's
multicast and the weight stream are inside the window.
"""

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
from bingo_cfg_reuse import reuse_gemm_start_config  # noqa: E402
from bingo_gemm_reuse import reuse_gemm_config  # noqa: E402
from bingo_kernel_args import (  # noqa: E402
    INT4_B_LAYOUT_PAIRED, HostBingoKernelHbmBcastArgs, HostBingoKernelHbmUcastArgs,
    SnaxBingoKernelGemmFullArgs, SnaxBingoKernelGemmShapeInitArgs,
    SnaxBingoKernelIdma1dCopyArgs,
    SnaxBingoKernelIdmaFetchFlaggedArgs,
    SnaxBingoKernelIdmaPutFlaggedArgs, SnaxBingoKernelSimdAddF16Args, SnaxBingoKernelSimdFp16ToInt8Args,
    SnaxBingoKernelSimdMulF16Args, SnaxBingoKernelSimdSwigluF16F16Args, SnaxBingoKernelWaitFlagsArgs,
    SnaxBingoKernelXdmaChainGatherArgs, simd_beats)
from bingo_mem_handle import BingoMemSymbol  # noqa: E402
from libs.blocks import GemmQueue, chain_gather  # noqa: E402
from libs.comm.nest import d_to_a_args  # noqa: E402

import chiplet_mc as cm  # noqa: E402
import ffn_data  # noqa: E402

TILE = 16 * 16 * 2                         # one D-layout FP16 tile
BLK = 16                                   # a GEMM column block (meshCol)
EW_F16_ADD = cm.EW_F16_ADD


class PBuild(cm.ChipletBuild):
    TAG = "pf"
    # the fewest post blocks a pass may have (params post_blocks); the decode (ffn_decode_mc.py), one row tile, has one
    MIN_POST_BLOCKS = 2

    def __init__(self, args, p, plat, hw, shared=None):
        super().__init__(args, p, plat, hw, shared=shared)
        if self.bcast_impl != "bcast_l1":
            raise ValueError("params bcast: bcast_l1 (X reaches the chips through the memory chip, params x_cast)")
        # THE ABLATION (workloads/ffn/dsv2_l0_prefill_ablation): params x_cast, reduce and sync take the paper's three
        # mechanisms off one at a time, on the same mapping.
        #   x_cast     multicast: X from HBM by one multicast; unicast: the memory chip's push engine writes a copy into
        #              each chip's L3 in turn, the same tiles and flags (__host_bingo_kernel_hbm_ucast)
        #   reduce     chain: ChainGathers; root: the root's clusters pull every chip's partial of a slice into two L1
        #              slots and fold it into their own on the SIMD, in the fold order (sync barrier)
        #   sync       edge: every cross-chip dependency a BINGO hardware edge; barrier: a chip barrier after every
        #              kernel, fencing every core of the chip, kernels never overlapping (_passes_kernel)
        self.x_cast = str(p.get("x_cast", "multicast"))
        if self.x_cast not in ("multicast", "unicast"):
            raise ValueError(f"params x_cast={self.x_cast!r}: multicast or unicast")
        self.per_kernel = self.sync == "barrier"
        if self.reduce_impl == "root" and (not self.per_kernel or p.get("chain_alt", False)):
            raise ValueError("params reduce root: with sync barrier and no chain_alt (one fold order)")
        self.barrier_fence = self.per_kernel
        # params gemm_async: every GEMM a gemm_start / gemm_wait pair (libs/blocks/gemm_async.py), the GEMM core's program
        # issuing a start before the previous GEMM's wait where the graph allows, so it programs the next dispatch while
        # the array runs; a GEMM's consumers and the buffers it reads wait for its wait. Default: one blocking gemm_full
        # each. gemm_shape_l1: each cluster's GEMM core copies the array shape's parameters into its L1 once
        # (gemm_shape_init), and every start reads them there, not in L3. gemm_cfg_ahead: each GEMM a configure task that
        # runs as soon as the previous dispatch has started and a start task that waits for the operands; gemm_cfg_hoist
        # (with it): the configure ahead of the waits before its start. A wait trails only the next start of its own
        # phase: the GEMM's kind (gu, dn) in its pass, so no wait trails a start across a kernel boundary
        self.gemm_async = bool(p.get("gemm_async", False))
        self.gemm_shape_l1 = bool(p.get("gemm_shape_l1", False))
        self.gemm_cfg_ahead = bool(p.get("gemm_cfg_ahead", False))
        self.gemm_cfg_hoist = bool(p.get("gemm_cfg_hoist", False))
        if not self.gemm_async and (self.gemm_shape_l1 or self.gemm_cfg_ahead or self.gemm_cfg_hoist):
            raise ValueError("params gemm_shape_l1 / gemm_cfg_ahead / gemm_cfg_hoist: with gemm_async")
        if self.gemm_cfg_hoist and (not self.gemm_cfg_ahead or self.per_kernel):
            # kernel by kernel, a fence orders every task made before a barrier ahead of it: a configure hoisted ahead of
            # the waits before its start would close a cycle with them
            raise ValueError("params gemm_cfg_hoist: with gemm_cfg_ahead, and not with sync barrier")
        self.gsh = {}                           # (chip, cluster) -> its L1 copy of the shape parameters
        R = self.ROWS = int(p.get("rows", 32))
        if R % 16 or self.T % R:
            raise ValueError(f"params tokens={self.T} rows={R}: rows a multiple of 16 dividing T")
        self.NI, self.MT = self.T // R, R // 16
        self.deq_per_row = bool(p.get("deq_per_row", False))
        # params x_blocks: X of the first pass fetched in this many row blocks, and the first gate|up chunk run as one
        # GEMM per block, so the arrays start when the first block has landed
        self.x_blocks = int(p.get("x_blocks", 1))
        # params post_blocks: the reshape and quantise of A16 in this many row blocks, so the SIMD quantises a block
        # while the xDMA reshapes the next (the gap between gate|up and down shrinks)
        self.post_blocks = int(p.get("post_blocks", 1))
        # params split_last (with post_blocks): the last gate|up chunk as one GEMM per post block, each block's
        # dequantise and SwiGLU, its reshape and the previous block's quantise right behind it, so only the last
        # block's work is left between the last gate|up GEMM and the first down GEMM
        self.split_last = bool(p.get("split_last", False))
        if self.split_last and (self.post_blocks < 2 or not self.deq_per_row):
            raise ValueError("params split_last needs post_blocks > 1 and deq_per_row")
        if self.post_blocks < self.MIN_POST_BLOCKS or self.MT % self.post_blocks:
            raise ValueError(f"params post_blocks={self.post_blocks}: at least {self.MIN_POST_BLOCKS}, dividing {self.MT} "
                             f"row tiles")
        if self.MT % self.x_blocks:
            raise ValueError(f"params x_blocks={self.x_blocks}: dividing {self.MT} row tiles")
        # params restream: every pass streams its weight chunks from the memory chip again (a ring load each), instead
        # of copying them from the L3 slots the first pass filled; for weights larger than the ring
        self.restream = bool(p.get("restream", False))
        self.BPC = int(p.get("blocks", 3))          # 16-column gate blocks per gate|up chunk
        # params first_blocks: the first gate|up chunk of every cluster this many gate blocks wide (0 = blocks): a
        # narrow first chunk is pushed sooner, so the first GEMM starts sooner
        self.first_blocks = int(p.get("first_blocks", 0))
        self.DNC = int(p.get("dn_cols", 512))       # output columns per down chunk
        # every INT4 weight chunk in the paired layout (blocks of 32 columns, layout.py to_b_pairs), which the B reader
        # walks 64 bytes a pass, as A: every GEMM's column tiles, and every down slice, come in whole pairs
        self.int4_b = INT4_B_LAYOUT_PAIRED
        self.dn_unit = 2 * BLK
        if self.DNC % self.dn_unit:
            raise ValueError(f"params dn_cols={self.DNC}: every down GEMM whole {self.dn_unit}-column units (pairs of "
                             f"column tiles)")
        d = self.d = self._data(args, p)
        self.lay = d.layout
        self.D, self.I = d.H, d.I
        if self.D % self.DNC:
            raise ValueError(f"{self.D} outputs are not whole {self.DNC}-column down chunks")
        self.NDN = self.D // self.DNC
        # the intermediate columns split over the clusters in whole GEMM column blocks, the root chip's clusters taking
        # the smaller share, since its cluster 0 also runs every gather and stores Y. params root_cols: the root chip's
        # clusters get exactly this many columns (one number for both, or one per cluster; the rest is split over the
        # others), so its extra duties (every gather and store) cost no window time
        rc = p.get("root_cols", 0)
        rc = [int(v) for v in rc] if isinstance(rc, (list, tuple)) else [int(rc)] * self.NC
        self.cols = self.split_columns(self.I, last=self.gs(self.ROOT), unit=BLK,
                                       fixed=dict(zip(self.gs(self.ROOT), rc)) if any(rc) else None)
        self.chunks = {g: self._gu_chunks(n) for g, (_, n) in self.cols.items()}
        # each cluster takes the chip's columns (K) for its half of the outputs, so it holds the chip's sum there and
        # nothing is folded inside a chip (the two clusters exchange their A8 once, before the first gather)
        self.kcols = {}                             # {chip: (first column, columns)}
        if self.NC != 2 or self.NDN % 2:
            raise ValueError("the down projection split by output columns needs 2 clusters and an even number of slices")
        for k in self.chips:
            (a0, an), (b0, bn) = (self.cols[g] for g in self.gs(k))
            if b0 != a0 + an:
                raise ValueError(f"chip {k:02x}'s columns are not contiguous")
            self.kcols[k] = (a0, an + bn)
        self.gold = d.golden([self.kcols[k] for k in self.chips],
                             {k: [i] for i, k in enumerate(self.chips)}, self.fold_order())
        self.dn_sl = self._dn_out_split(bool(p.get("dn_balance", False)))
        # params chain_alt: cluster 1's ChainGathers run the chips in the reverse order, so a gather's data follows the
        # previous gather's finish back along the ring instead of meeting it mid-chain (a collector retires before its
        # finish has walked back to the first hop)
        self.chain_alt = bool(p.get("chain_alt", False))
        if self.chain_alt:
            loc, add16 = self.gold["loc"], d.fp.add16
            rev = self.order[1:][::-1] + self.order[:1]
            y = loc[rev[0]]
            for k in rev[1:]:
                y = add16(y, loc[k])
            yr = self.gold["y_raw"].copy()
            for c0, w in self.dn_sl[1]:
                yr[:, c0:c0 + w] = y[:, c0:c0 + w]
            self.gold["y_raw"], self.gold["y16"] = yr, d.fp.mul16(yr, d.dn_f[None, :])
        for line in d.report(self.gold["y16"]):
            print(f"[pf] {line}")
        print(f"[pf] {self.T} tokens in {self.NI} passes of {R}; fold order "
              f"{[f'{k:02x}' for k in self.order]}; columns per cluster "
              f"{sorted({n for _, n in self.cols.values()})}; gate|up chunks per cluster "
              f"{sorted({len(c) for c in self.chunks.values()})}")
        self.XB = R * self.D                    # X of a pass, INT8
        self.YB = R * self.D * 2                # Y_raw of a pass, FP16
        self.XT = self.XB // self.x_blocks      # the multicast's tile, each behind a flag: a first-pass block of X
        self.TPP = self.XB // self.XT           # tiles a pass
        # params l3_overlay: Y_raw of a pass is stored over that pass's X buffers in L3, one region a pass ([source |
        # landing], as long as Y_raw), so the image holds Y beside neither; pass i's stores follow every reader of
        # X_i (the root's put and its clusters' fetches) through the pass's GEMMs
        self.l3_overlay = bool(p.get("l3_overlay", False))
        # params y_scatter: the root stores each slice of Y_raw row tile by row tile (16 rows) straight into the L3 of
        # the chip that owns the tile (tile q on the q-th chip, as a layer's tiles), behind a flag a slice there, instead
        # of into its own L3: a tile's whole row of Y_raw is then in its owner's L3 (D layout) once the pass's slices
        # are stored, and the block ends with one wait a chip on its tiles' flags
        self.y_scatter = bool(p.get("y_scatter", False))
        self.YT_B = 16 * self.D * 2             # y_scatter: a 16-row tile of Y_raw, D layout, FP16
        # params out_slots (default 2): the slots of each cluster's symmetric buffer the down slices go through (slice j
        # into slot j mod out_slots), so the down GEMMs run that many slices ahead of the serial gathers; the root's
        # fold buffer keeps two
        self.out_slots = int(p.get("out_slots", 2))
        if self.out_slots < 2:
            raise ValueError(f"params out_slots={self.out_slots}: at least 2")
        # params chain_group (default 1): one ChainGather carries this many consecutive down slices of a cluster, the
        # output slots they sit in back to back, into a fold slot of that many slices at the root, stored to L3 by one
        # copy (or row tile by row tile with y_scatter)
        self.chain_group = int(p.get("chain_group", 1))
        h = self.NDN // self.NC
        if h % self.chain_group or (self.out_slots % self.chain_group and not self.per_kernel):
            raise ValueError(f"params chain_group={self.chain_group}: must divide the {h} slices a cluster and "
                             f"out_slots={self.out_slots}")
        if self.per_kernel:
            # kernel by kernel, no task of one kernel runs inside another: the down slices all wait in their output slots
            # until the reduce kernel, slot j for slice j, the symmetric buffer's two slots then the X buffer's (dead
            # until the next pass's X), so the X buffer sits right behind the symmetric buffer on every cluster
            bad = [k for k in ("split_last", "split_first_dn", "dn_load_early", "x_early", "y_scatter") if p.get(k, False)]
            if bad or "out_slots" in p:
                raise ValueError(f"params sync barrier: no {bad or 'out_slots'} (a kernel's tasks never run inside "
                                 f"another kernel)")
            if self.chain_group != 1 and self.reduce_impl != "chain":
                raise ValueError("params chain_group: with reduce chain")
            self.out_slots = h                  # slot j for slice j of the pass
        self.NTL = -(-self.T // 16 // len(self.chips))
        self._stage()
        self.make_streams()
        self.make_rings()
        self._buffers()

    def _data(self, args, p):
        """The FFN's data and golden (a layer's block gives it the layer's rows)."""
        cache = os.path.join(REPO, p.get("weights", "target/sim/automation/sweep/llm_blocks/ffn/cache"))
        return ffn_data.FFNData(cache, p.get("model", "dsv2-lite-l0"), seed=int(p.get("seed", 1)),
                                wbits=self.wbits, tokens=self.T, inter=p.get("inter"))

    def pass_start(self, i):
        """Where the tasks of pass i (i >= 1) begin, in the order they are made: nothing here; a layer starts an args
        phase (llama3/full_layer_prefill/common/llama3_layer_mc.py)."""

    # ---- the plan -------------------------------------------------------------------------
    def _gu_chunks(self, n):
        """[(first gate block, blocks)] of a cluster's n columns: params first_blocks, then chunks of blocks."""
        nb = n // BLK
        f = self.first_blocks
        if not f:
            return [(b0, min(self.BPC, nb - b0)) for b0 in range(0, nb, self.BPC)]
        if f > self.BPC or f > nb:
            raise ValueError(f"params first_blocks={f}: at most blocks={self.BPC} and the {nb} blocks of a cluster")
        return [(0, f)] + [(b0, min(self.BPC, nb - b0)) for b0 in range(f, nb, self.BPC)]

    def _gu_cols(self, g, b0, nb_):
        """The gate|up chunk's columns of [gate | up]: its gate blocks, then the same up blocks."""
        c0, _ = self.cols[g]
        gate = np.arange(c0 + BLK * b0, c0 + BLK * (b0 + nb_))
        return np.concatenate([gate, self.I + gate])

    # ---- data -----------------------------------------------------------------------------
    def _pack_b(self, q):
        """An INT4 weight chunk [K, N] as the GEMMs read it: the paired layout."""
        lay, mesh = self.lay, self.d.mesh
        if q.shape[1] % (2 * mesh[2]):
            raise ValueError(f"a weight chunk of {q.shape[1]} columns is not whole pairs of {mesh[2]}-column tiles")
        return lay.pack_int4(lay.to_b_pairs(q, mesh))

    def x_a(self):
        """Every pass's X, A layout, pass after pass (INT8)."""
        R = self.ROWS
        return np.concatenate([self.lay.to_a(self.d.hq[i * R:(i + 1) * R]) for i in range(self.NI)]).astype(np.int8)

    def _stage(self):
        st, d, T, lay = self.st, self.d, self.TAG, self.lay
        # every chip's X landing area; params l3_overlay: one region a pass as long as Y_raw of the pass, X landing in its
        # second half
        if self.l3_overlay:
            st.put_zeros(f"{T}_xy", "int8_t", 2 * self.NI * self.XB, preload=True)
        else:
            st.put_zeros(f"{T}_xland", "int8_t", self.NI * self.XB, preload=True)
        # every pass's X in the HBM of the root's memory chip, a flag a tile
        m = self.mem[self.ROOT]
        self.h_xhbm = st.put_hbm(f"{T}_xhbm", self.x_a().reshape(-1), mem_chip=(m >> 4, m & 15))
        st.put_zeros(f"{T}_xtflag", "uint8_t", 64 * self.NI * self.TPP)
        if self.x_cast == "unicast":            # the root's progress flags, one a (chip, tile) of every pass
            st.put_zeros(f"{T}_xuprog", "uint8_t", 64 * self.NI * len(self.chips) * self.TPP)
        if not self.l3_overlay:
            st.put_zeros(f"{T}_y", "uint8_t", self.NI * self.YB, preload=True)
        if self.y_scatter:
            st.put_zeros(f"{T}_yt", "uint8_t", self.NTL * self.YT_B, preload=True)    # the chip's tiles of Y_raw
            st.put_zeros(f"{T}_ytflag", "uint8_t", self.NTL * self.NDN * 64)          # a slice of one landed
        self.img, self.chunk_off, self.sdup = {}, {}, {}
        for g, (c0, n) in self.cols.items():
            k = self.chip(g)
            loc = (self.mem[k] >> 4, self.mem[k] & 15)
            parts, offs, sc = [], [], []
            for b0, nb_ in self.chunks[g]:
                cols = self._gu_cols(g, b0, nb_)
                parts.append(self._pack_b(d.gu_q[:, cols]))
                s = np.tile(d.gu_f[cols].astype(np.float16), (16, 1))
                tile = lay.to_d(s, d.mesh)
                # params deq_per_row: the scales of one row tile, applied to every row tile in turn
                sc.append(np.concatenate([tile] * (1 if self.deq_per_row else self.MT)))
            kc0, kn = self.kcols[k]             # the chip's columns, this cluster's share of the outputs
            for o0, w in self.dn_sl[self.c(g)]:
                parts.append(self._pack_b(d.dn_q[kc0:kc0 + kn, o0:o0 + w]))
            off = 0
            for b in parts:
                offs.append((off, b.size))
                off += b.size
            self.img[g] = st.put_hbm(f"{T}_w_g{g}", np.concatenate(parts).astype(np.int8),
                                     mem_chip=loc)
            self.chunk_off[g] = offs
            s = np.concatenate(sc).astype(np.float16)
            self.sdup[g] = (st.put_hbm(f"{T}_sgu_g{g}", s.view(np.int8), mem_chip=loc), s.nbytes)

    def _buffers(self):
        R, MT = self.ROWS, self.MT
        # the down output is reduced slice by slice (a slice's columns of all rows), through out_slots slots of each
        # cluster's symmetric buffer and two fold slots (of chain_group slices) of the root clusters' fold buffers
        self.SLICE = R * max(w for v in self.dn_sl.values() for _, w in v) * 2
        # sync barrier: two output slots in the symmetric buffer, the rest in the X buffer behind it (named to sort right
        # after it, so the static packer puts it there on every cluster: check_symmetric)
        self.sym_slots = 2 if self.per_kernel else self.out_slots
        if self.per_kernel and (self.out_slots - self.sym_slots) * self.SLICE > self.XB:
            raise ValueError(f"sync barrier: {self.out_slots - self.sym_slots} output slots of {self.SLICE} B do "
                             f"not fit the {self.XB}-byte X buffer")
        self.sym_bytes = self.sym_slots * self.SLICE
        self.sym = {(k, c): self.cx_chip(k).at(c).l1(f"AAAA_{self.TAG}_sym", self.sym_bytes)
                    for k in self.chips for c in range(self.NC)}
        self.p1c = {c: self.cx_chip(self.ROOT).at(c).l1(f"{self.TAG}_p1c", 2 * self.chain_group * self.SLICE)
                    for c in range(self.NC)}
        self.xbuf, self.gbuf, self.sbuf, self.a16d, self.a16a, self.a8 = {}, {}, {}, {}, {}, {}
        for g, (_, n) in self.cols.items():
            l1 = lambda name, nb: self.cx_chip(self.chip(g)).at(self.c(g)).l1(f"{self.TAG}_{name}", nb)
            self.xbuf[g] = [l1("AAAB_xo" if self.per_kernel else "x0", self.XB)]
            gb = MT * 2 * self.BPC * TILE
            self.gbuf[g] = [l1(f"g{b}", gb) for b in range(2)]
            self.sbuf[g] = l1("sgu", self.sdup[g][1])
            self.a16d[g] = l1("a16d", R * n * 2)
            self.a16a[g] = l1("a16a", R * n * 2)
            a8 = l1("a8_0", R * self.kcols[self.chip(g)][1])    # the chip's A8: both clusters' columns, one buffer
            self.a8[g] = [a8, a8]

    # ---- helpers --------------------------------------------------------------------------
    def chain_program_order(self):
        """Each device core runs its tasks in creation order: an edge between consecutive tasks
        of every (chip, cluster, core). Returns the number of edges added."""
        dev = [self.roles[r] for r in ("gemm", "simd", "xdma", "dm")]
        return GemmQueue.of(self.dfg, self.gemm_async).program_order(self.dfg, dev, tag=self.TAG)

    def gemm_node(self, k, c, name, args, deps, i):
        """One GEMM of pass i on chip k cluster c; returns the task its output is ready after and its operands are free
        after: the blocking gemm_full, or (params gemm_async) the wait of its gemm_start / gemm_wait pair."""
        gemm = self.roles["gemm"]
        if not self.gemm_async:
            return self.node(k, c, gemm, name, "__snax_bingo_kernel_gemm_full", args, deps)
        shape_addr = 0
        if self.gemm_shape_l1:
            if (k, c) not in self.gsh:
                buf = self.cx_chip(k).at(c).l1(f"{self.TAG}_gsh", SnaxBingoKernelGemmShapeInitArgs.BYTES)
                self.node(k, c, gemm, "gsh_init", "__snax_bingo_kernel_gemm_shape_init",
                          SnaxBingoKernelGemmShapeInitArgs(buf, args.array_shape_idx), [])
                self.gsh[(k, c)] = buf
            shape_addr = self.gsh[(k, c)]
        q = GemmQueue.of(self.dfg, True)
        q.cfg_ahead, q.cfg_hoist = self.gemm_cfg_ahead, self.gemm_cfg_hoist
        return q.dispatch(self.at(k, c), name, args, list(deps), phase=f"{name.split('_')[0]}_i{i}",
                          shape_addr=shape_addr)

    def gather_node(self, c, name, args, deps):
        """One ChainGather into the root's cluster c, on its xDMA core; returns the task."""
        return chain_gather(lambda n, kernel, a, d: self.node(self.ROOT, c, self.roles["xdma"], n, kernel, a, d),
                            name, args, deps)

    def _load(self, g, i, j, deps=()):
        """Weight chunk j of cluster g into the next slab: a ring load in pass 0 (every pass with params restream), a
        copy from its slot later. Returns (load task, slab index, slab)."""
        k, c, dm = self.chip(g), self.c(g), self.roles["dm"]
        ls = self.ls[g]
        off, nb = self.chunk_off[g][j]
        si, slab, war = ls.take(nb)
        if i == 0 or self.restream:
            ld, sl = ls.rings.load_chunk(
                lambda n, core, kern, args, after: self.node(k, c, core, n, kern, args, after), c,
                f"Ld_c{j}_i{i}_g{g}", src=self.img[g].address + off, nbytes=nb, slab=slab, war=war,
                deps=deps, dm=dm)
            self.slot[(g, j)] = (sl, ld)
        else:
            sl, first = self.slot[(g, j)]
            ld = self.node(k, c, dm, f"Ld_c{j}_i{i}_g{g}", "__snax_bingo_kernel_idma_1d_copy",
                           SnaxBingoKernelIdma1dCopyArgs(sl, slab, nb),
                           list(dict.fromkeys(war + [first] + list(deps))))
        return ld, si, slab

    # ---- X ----------------------------------------------------------------------------------
    def _xland(self, off, chip_id=None):
        """Where X lands in every chip's L3, at byte `off`; params l3_overlay: in the second half of its pass's region."""
        if self.l3_overlay:
            return BingoMemSymbol(f"{self.TAG}_xy", offset=off // self.XB * self.YB + self.XB + off % self.XB,
                                  chip_id=chip_id)
        return BingoMemSymbol(f"{self.TAG}_xland", offset=off, chip_id=chip_id)

    def _ysym(self, off, **kw):
        """Y_raw in the root's L3 at byte `off`; params l3_overlay: over the X buffers, at the same offset."""
        return BingoMemSymbol(f"{self.TAG}_{'xy' if self.l3_overlay else 'y'}", offset=off, **kw)

    def y_owner(self, q):
        """y_scatter: (chip, its tile index) of 16-row tile q of Y_raw."""
        return self.chips[q % len(self.chips)], q // len(self.chips)

    def yt_sym(self, q, off=0, chip_id=None):
        """y_scatter: tile q of Y_raw in its owner's L3 at byte `off` (chip_id: as another chip names it)."""
        return BingoMemSymbol(f"{self.TAG}_yt", offset=self.y_owner(q)[1] * self.YT_B + off, chip_id=chip_id)

    def out_view(self, g, so):
        """Cluster g's output slot so: the symmetric buffer's slots, then (sync barrier) the X buffer's, which
        follows it at the same offset on every cluster, so self.sx(c, so * SLICE) names the slot on every chip."""
        k, c = self.chip(g), self.c(g)
        if so < self.sym_slots:
            return self.sym[(k, c)].view(so * self.SLICE)
        return self.xbuf[g][0].view((so - self.sym_slots) * self.SLICE)

    def x_hbm_bcasts(self, go):
        """The root's host queues its memory chip's push engine, one task a pass: the pass's X a tile at a time
        into every chip's L3, each tile behind its flag. The root's prefetcher drives the same engine and holds the host
        for the whole window, so it starts after the last of them. Returns the tasks, one a pass."""
        R, T = self.ROOT, self.TAG
        m = self.mem[R]
        one = self.h_seq[m].address + 64                    # a word holding 1 in the memory chip's HBM: the flags' value
        out = []
        for i in range(self.NI):
            src, dst = self.h_xhbm.address + i * self.XB, self._xland(i * self.XB)
            flag = BingoMemSymbol(f"{T}_xtflag", offset=64 * i * self.TPP)
            if self.x_cast == "unicast":                # params x_cast unicast: a copy a chip, in the fold order
                prog = BingoMemSymbol(f"{T}_xuprog", offset=64 * i * len(self.chips) * self.TPP)
                kern, kargs = "__host_bingo_kernel_hbm_ucast", HostBingoKernelHbmUcastArgs(
                    src, dst, flag, prog, one, 1, self.XT, self.TPP, m, self.engine[R], self.order)
            else:
                kern, kargs = "__host_bingo_kernel_hbm_bcast", HostBingoKernelHbmBcastArgs(
                    src, dst, flag, one, 1, self.XT, self.TPP, m, self.engine[R])
            out.append(self.cx_chip(R).host_node(f"x_hbm_bcast_i{i}_k{R:02x}", kern, kargs, after=[go[R]] + out[-1:]))
        self.edge(out[-1], self.prefetch[R])
        return out

    def _x_fetch(self, g, i, off, nb, src, after, name, block=False):
        """Bytes off .. off + nb of X_i into cluster g's L1, behind their tiles' flags. `src` is the multicast task (the
        hbm_bcast of pass i)."""
        T, dm = self.TAG, self.roles["dm"]
        k, c = self.chip(g), self.c(g)
        xb = self.xbuf[g][0]
        t0 = (i * self.XB + off) // self.XT
        flag, nf = BingoMemSymbol(f"{T}_xtflag", offset=64 * t0), nb // self.XT
        nd = self.node(k, c, dm, name, "__snax_bingo_kernel_idma_fetch_flagged",
                       SnaxBingoKernelIdmaFetchFlaggedArgs(self._xland(i * self.XB + off), xb.view(off) if block else xb,
                                                           nb, flag, nf, 1),
                       list(after))
        self.dfg.bingo_add_order_edge(src, nd)
        return nd

    def x_wait_block(self, g, b, src, after=()):
        """params x_blocks: row block b of the first pass's X into cluster g's L1, behind its flags."""
        nb = self.XB // self.x_blocks
        return self._x_fetch(g, 0, b * nb, nb, src, after, f"x_wait_i0b{b}_g{g}", block=True)

    def stage_x(self, i, src, after):
        """X_i into every cluster's L1 buffer, behind its flags. {g: the task after which it is there}."""
        return {g: self._x_fetch(g, i, 0, self.XB, src, after.get(g, []), f"x_wait_i{i}_g{g}") for g in range(self.G)}

    # ---- gate|up and the A8 ---------------------------------------------------------------------
    def stage_gu(self, i, x):
        """Every cluster's gate|up chunks, each dequantised and through SwiGLU. Returns {g: the
        last SwiGLU} and {g: the last GEMM}."""
        gemm, simd = self.roles["gemm"], self.roles["simd"]
        last_sw, last_gemm = {}, {}
        for g, (_, n) in self.cols.items():
            k, c = self.chip(g), self.c(g)
            nb = n // BLK
            xb = self.xbuf[g][0]
            soff = 0
            sw = None
            xblk = self.x_blocks > 1 and i == 0
            xws = [x[g]]
            for j, (b0, nb_) in enumerate(self.chunks[g]):
                ld, si, slab = self._load(g, i, j, [x[g]] if j == 0 else ())
                if self.dn_load_early and j == len(self.chunks[g]) - 1:
                    # params dn_load_early: the first down chunk's load right behind the last gate|up chunk's, ahead
                    # of the A8 copies in the DM core's program (those wait for the quantise)
                    self.dn_first_load[g] = self._load(g, i, j + 1)
                if xblk and j == 0:
                    # the other X blocks behind the first weight load in the DM core's program
                    xws += [self.x_wait_block(g, bb, self.x_bc[0]) for bb in range(1, self.x_blocks)]
                b = self.gu_count[g] % 2
                self.gu_count[g] += 1
                gb = self.gbuf[g][b]
                war = self.gb_reader[g][b]
                nblk = 2 * nb_                      # its gate blocks and its up blocks
                half = nb_ * TILE                   # one row block's gate (or up) tiles
                kargs = dict(input_B_addr=slab, input_C_addr=0, K=self.D // 4, N=nblk, array_shape_idx=0,
                             transpose_A=0, transpose_B=0, accumPrevC=0, int32tofp16_enable=1, int4_b_enable=self.int4_b,
                             d_shift=self.d.k_x)
                last = self.split_last and j == len(self.chunks[g]) - 1 and not (xblk and j == 0)
                if (xblk and j == 0) or last:       # one GEMM per X block (per post block): its row tiles only
                    nsp = self.post_blocks if last else self.x_blocks
                    mb = self.MT // nsp
                    gms = [self.gemm_node(k, c, f"gu_c{j}_i{i}b{bb}_g{g}",
                                          SnaxBingoKernelGemmFullArgs(
                                              input_A_addr=xb.view(bb * mb * (self.XB // self.MT)),
                                              output_D_addr=gb.view(bb * mb * 2 * half), M=mb, **kargs),
                                          [ld, xws[-1] if last else xws[bb]] + ([war] if war is not None else []), i)
                           for bb in range(nsp)]
                    gm_of = lambda m, gms=gms, mb=mb: gms[m // mb]
                    gm = gms[-1]
                else:
                    gm = self.gemm_node(k, c, f"gu_c{j}_i{i}_g{g}",
                                        SnaxBingoKernelGemmFullArgs(input_A_addr=xb, output_D_addr=gb, M=self.MT, **kargs),
                                        [ld, xws[-1]] + ([war] if war is not None else []), i)
                    gm_of = lambda m: gm
                self.ls[g].release(si, gm)
                last_gemm[g] = gm
                if last:
                    sw, swi_m = self._gu_tail(g, i, j, b0, nb_, gb, soff, gm_of)
                    soff += nblk * 256 * 2
                    self.gb_reader[g][b] = sw
                    continue
                if self.deq_per_row:                # one task per row tile, all with the chunk's one scale tile
                    el = nblk * 256
                    dqs = [self.node(k, c, simd, f"deq_c{j}m{m}_i{i}_g{g}", "__snax_bingo_kernel_simd_stream_elementwise",
                                     SnaxBingoKernelSimdMulF16Args(gb.view(m * 2 * half), self.sbuf[g].view(soff),
                                                                   gb.view(m * 2 * half), el // 32, 32), [gm_of(m)])
                           for m in range(self.MT)]
                else:
                    el = self.MT * nblk * 256
                    dq = self.node(k, c, simd, f"deq_c{j}_i{i}_g{g}", "__snax_bingo_kernel_simd_stream_elementwise",
                                   SnaxBingoKernelSimdMulF16Args(gb, self.sbuf[g].view(soff), gb,
                                                                 el // 32, 32), [gm])
                    dqs = [dq] * self.MT
                soff += el * 2
                swi_m = []
                for m in range(self.MT):
                    dq = dqs[m]
                    gate = gb.view(m * 2 * half)
                    up = gb.view(m * 2 * half + half)
                    out = self.a16d[g].view((m * nb + b0) * TILE)
                    war_a = [self.d2a[(g, i - 1)]] if (g, i - 1) in self.d2a else []
                    sw = self.node(k, c, simd, f"swi_c{j}m{m}_i{i}_g{g}",
                                   "__snax_bingo_kernel_simd_swiglu_f16_f16",
                                   SnaxBingoKernelSimdSwigluF16F16Args(gate, up, out, 1,
                                                                       nb_ * 256),
                                   [dq] + (war_a if j == 0 and m == 0 else []))
                    swi_m.append(sw)
                self.gb_reader[g][b] = sw
            last_sw[g] = sw
            self.last_swi[(g, i)] = swi_m           # the last chunk's SwiGLU of each row tile
        return last_sw, last_gemm

    def _gu_tail(self, g, i, j, b0, nb_, gb, soff, gm_of):
        """params split_last: the last chunk's dequantise and SwiGLU row tile by row tile; behind each post
        block's last SwiGLU its reshape (xDMA) and the previous block's quantise (SIMD, whose reshape is done
        by then). Returns (the last SwiGLU, the SwiGLU of each row tile)."""
        k, c, simd = self.chip(g), self.c(g), self.roles["simd"]
        n = self.cols[g][1]
        nb, half, el = n // BLK, nb_ * TILE, 2 * nb_ * 256
        pmb = self.MT // self.post_blocks
        swi_m, rs, sw = [], [], None
        for m in range(self.MT):
            dq = self.node(k, c, simd, f"deq_c{j}m{m}_i{i}_g{g}", "__snax_bingo_kernel_simd_stream_elementwise",
                           SnaxBingoKernelSimdMulF16Args(gb.view(m * 2 * half), self.sbuf[g].view(soff),
                                                         gb.view(m * 2 * half), el // 32, 32), [gm_of(m)])
            sw = self.node(k, c, simd, f"swi_c{j}m{m}_i{i}_g{g}", "__snax_bingo_kernel_simd_swiglu_f16_f16",
                           SnaxBingoKernelSimdSwigluF16F16Args(gb.view(m * 2 * half), gb.view(m * 2 * half + half),
                                                               self.a16d[g].view((m * nb + b0) * TILE), 1, nb_ * 256),
                           [dq])
            swi_m.append(sw)
            if (m + 1) % pmb == 0:
                bb = m // pmb
                rs.append(self._post_d2a(g, i, bb, [sw]))
                if bb:
                    self.quant[(g, i)] = self._post_quant(g, i, bb - 1, rs[bb - 1])
        self.quant[(g, i)] = self._post_quant(g, i, self.post_blocks - 1, rs[-1])
        self.d2a[(g, i)] = rs[-1]
        self.post_made.add((g, i))
        return sw, swi_m

    def _post_d2a(self, g, i, b, after):
        """Post block b of A16 (its row tiles, every column) reshaped to the A layout."""
        k, c, n = self.chip(g), self.c(g), self.cols[g][1]
        mb, nb, m0 = self.MT // self.post_blocks, n // BLK, b * (self.MT // self.post_blocks)
        war_q = [self.quant[(g, i - 1)]] if (g, i - 1) in self.quant else []
        return self.node(k, c, self.roles["xdma"], f"d2a_i{i}b{b}_g{g}", "__snax_bingo_kernel_xdma_6d",
                         d_to_a_args(self.d.mesh, mb, nb, n // 4, self.a16d[g].view(m0 * nb * TILE),
                                     self.a16a[g].view(m0 * 16 * n * 2)), list(after) + war_q)

    def _post_quant(self, g, i, b, rs):
        """Post block b of A16a quantised, one row tile at a time, into this cluster's run of the chip's A8, each run
        then copied into the other cluster's A8 (self.ex_push[(g, i)]); a later pass's first copy also follows
        self.push_war[g]."""
        k, c, n = self.chip(g), self.c(g), self.cols[g][1]
        mb, m0 = self.MT // self.post_blocks, b * (self.MT // self.post_blocks)
        war = self.a8_reader[g][i % 2]
        kn = self.kcols[k][1]
        other = self.gs(k)[1 - c]
        coff = 16 * (kn - n) if c else 0    # this cluster's columns sit after the first cluster's
        q = None
        for m in range(m0, m0 + mb):
            off = m * 16 * kn + coff
            q = self.node(k, c, self.roles["simd"], f"quant_i{i}b{b}m{m}_g{g}",
                          "__snax_bingo_kernel_simd_fp16_to_int8",
                          SnaxBingoKernelSimdFp16ToInt8Args(self.a16a[g].view(m * 16 * n * 2), self.a8[g][0].view(off),
                                                            beats=simd_beats(16 * n * 2), rows=1,
                                                            inv_scale_f32bits=self.d.inv_a),
                          [rs] + ([war] if war is not None else []))
            self.q_tile.setdefault(g, {})[m] = q
            first = (g, i) not in self.ex_push
            self.ex_push.setdefault((g, i), []).append(
                self.node(k, c, self.roles["dm"], f"a8_push_m{m}_g{g}" if i == 0 else f"a8_push_i{i}m{m}_g{g}",
                          "__snax_bingo_kernel_idma_1d_copy",
                          SnaxBingoKernelIdma1dCopyArgs(self.a8[g][0].view(off), self.a8[other][0].view(off), 16 * n),
                          [q] + (self.push_war.get(g, []) if first else [])))
        return q

    def stage_post(self, i, sw):
        """A16 reshaped to the A layout and quantised into the chip's A8, post block by post block (split_last made the
        last chunk's already). {g: the last quantise}."""
        out = {}
        for g in self.cols:
            if (g, i) in self.post_made:
                out[g] = self.quant[(g, i)]
                continue
            mb = self.MT // self.post_blocks
            q = None
            for b in range(self.post_blocks):
                rs = self._post_d2a(g, i, b, [self.last_swi[(g, i)][b * mb + mb - 1]])
                q = self._post_quant(g, i, b, rs)
            self.d2a[(g, i)] = rs
            out[g] = q
            self.quant[(g, i)] = q
        return out

    # ---- the prefill ----------------------------------------------------------------------
    def pre_window(self):
        """{chip: [task]}: what the start waits for: the dequantisation factors into L1. params start_per_chip: a chip's
        loads one after the other, cluster 0's last, so the start waits for one task a chip (on the root, the load ahead
        of it on its own core) instead of one a cluster."""
        dm = self.roles["dm"]
        per_chip = bool(self.p.get("start_per_chip", False))
        pre = {k: [] for k in self.chips}
        for g in sorted(range(self.G), key=lambda g: (self.chip(g), -self.c(g))) if per_chip else range(self.G):
            k, c = self.chip(g), self.c(g)
            h, nb = self.sdup[g]
            t = self.node(k, c, dm, f"sgu_load_g{g}", "__snax_bingo_kernel_idma_1d_copy",
                          SnaxBingoKernelIdma1dCopyArgs(h.address, self.sbuf[g], nb), pre[k][-1:] if per_chip else [])
            pre[k] = [t] if per_chip else pre[k] + [t]
        return pre

    def build(self):
        self.slot = {}
        self.gu_count = {g: 0 for g in range(self.G)}
        self.gb_reader = {g: [None, None] for g in range(self.G)}
        self.a8_reader = {g: [None, None] for g in range(self.G)}
        # A16 (D layout) and A16a (A layout) are one buffer each: a pass's first SwiGLU waits for the previous pass's
        # last reshape, its first reshape for the previous pass's last quantise. d2a[(g, i)], quant[(g, i)]: the tasks
        self.d2a, self.quant, self.last_swi = {}, {}, {}
        self.post_made = set()                  # (g, i) whose post blocks the split last chunk made
        self.ex_push = {}                       # {(g, i): its A8 runs copied to the other cluster}
        self.push_war = {}                      # {g: what its first A8 copy of the pass follows}
        self.q_tile = {}                        # {g: {row tile: its quantise}}
        self.dn_load_early = bool(self.p.get("dn_load_early", False))
        # dep-tag relays of xDMA tasks (the gathers) go to the SIMD cores first, idle in the down phase, not behind the DM
        # cores' blocking ring loads or the GEMM cores' long GEMMs; each serves that one task, so a gather waits for its
        # own slice's GEMMs only, not for the next gather's too (one relay would hold both)
        self.dfg.relay_prefer_cores = {self.roles["xdma"]: [self.roles["simd"]]}
        self.dfg.relay_one_consumer_cores = [self.roles["xdma"]]
        # params gather_relay_only: a dep-tag cell of the xDMA tasks (the gathers) that is over budget relays its
        # leftover edges instead of ordering them, so no down GEMM waits for an earlier gather
        if self.p.get("gather_relay_only", False):
            self.dfg.relay_not_order_cores = [self.roles["xdma"]]
        self.dn_first_load = {}                 # dn_load_early: {g: the first down chunk's load}
        # the symmetric buffers are not zeroed before the start: every slot is written whole by its down GEMMs before it
        # is read, and an unzeroed buffer is live only from then on, so the static L1 plan can lay it over X
        self.sym_zero = False
        go = self.setup(extra=self.pre_window())
        self._passes(go, {g: [go[self.chip(g)]] for g in range(self.G)})
        if self.NI > 1 and not self.restream:
            # a pass after the first copies every weight chunk from the ring slot it landed in, so no slot may be pushed
            # twice: each ring holds all of its cluster's chunks
            for k in self.chips:
                for r, e in enumerate(self.rings[k].entries[:self.NC]):
                    if len(e) > self.n_slots:
                        raise ValueError(f"params weight_ring slots={self.n_slots}: chip {k:02x} ring {r} takes {len(e)} "
                                         f"chunks, and the passes after the first read each from its slot")

    def dn_slices(self, c):
        """The down slices (DNC output columns each) cluster c computes: its half of the outputs."""
        h = self.NDN // self.NC
        return range(c * h, (c + 1) * h)

    def _dn_out_split(self, balance):
        """{cluster: [(first output column, columns)]}, NDN / 2 slices each. Default: the two halves of the outputs in
        DNC-column slices. params dn_balance: cluster 0's share of the outputs moves (a multiple of 16, the same on every
        chip, since a gather takes the same slice from every chip) to where the busiest cluster's array steps (gate|up
        of its own columns + down of its chip's) are fewest, and each share is cut into slices of whole 16-column blocks
        as even as possible."""
        h = self.NDN // self.NC
        if not balance:
            return {c: [(j * self.DNC, self.DNC) for j in self.dn_slices(c)] for c in range(self.NC)}
        mt, kd = self.MT, self.D // 4

        def worst(S):
            w = 0
            for k in self.chips:
                kn = self.kcols[k][1]
                for c, g in enumerate(self.gs(k)):
                    out = S if c == 0 else self.D - S
                    w = max(w, mt * kd * 2 * self.cols[g][1] // BLK + mt * (kn // 4) * out // BLK)
            return w
        u = self.dn_unit                            # whole pairs of column tiles
        S = min(range(u, self.D, u), key=lambda v: (worst(v), abs(v - self.D // 2)))

        def cut(o0, width):
            nb = width // u
            sizes = [nb // h + (1 if i < nb % h else 0) for i in range(h)]
            out, c0 = [], o0
            for b in sizes:
                out.append((c0, b * u))
                c0 += b * u
            return out
        print(f"[pf] dn_balance: cluster 0 takes outputs 0..{S - 1}, cluster 1 {S}..{self.D - 1}; busiest cluster "
              f"{worst(S):,} array steps (even halves: {worst(self.D // 2):,})")
        return {0: cut(0, S), 1: cut(S, self.D - S)}

    def _passes(self, go, after_x):
        """The passes, one after the other, each the single pass over its rows with the down projection split by output
        columns. Each cluster's down GEMMs read the chip's whole A8 (K = the chip's columns) for its half of the
        outputs, one slice at a time into an output slot of its symmetric buffer, so the slot holds the chip's sum and
        the root's cluster c gathers that slot of every chip's cluster c straight into its fold buffer and stores it. The
        gathers run one at a time, alternating between the two clusters' xDMAs.

        Pass 0 loads the weights from the ring; a later pass copies every chunk from the L3 slot it landed in (or with
        params restream loads it from the ring again). X_(i+1) comes into L1 after the last gate|up GEMM that reads the
        buffer. A pass's first A8 copy follows the previous pass's last gather (no copy into a cluster while a gather
        passes through it) and the receiving cluster's last down GEMM (its A8); its first down GEMMs follow the previous
        pass's gathers of their slots; the stores keep their lag across passes. Every pass's X is multicast at the
        start, and the first gather follows it, so no multicast meets a gather."""
        if self.per_kernel:
            return self._passes_kernel(go, after_x)
        R, dm, xdma, gemm = self.ROOT, self.roles["dm"], self.roles["xdma"], self.roles["gemm"]
        self.x_bc = self.x_hbm_bcasts(go)
        if self.x_blocks > 1:
            x = {g: self.x_wait_block(g, 0, self.x_bc[0], after_x[g]) for g in range(self.G)}
        else:
            x = self.stage_x(0, self.x_bc[0], after_x)
        SL, others, h = self.SLICE, self.order[1:], self.NDN // self.NC
        p1x = {c: f"((uint64_t)({self.p1c[c].get_c_var_name()}))" for c in range(self.NC)}
        sym_reader, p1_reader, stores, pending, last_st = {}, {}, [], [], {}
        lag = 1                                 # a gather's stores follow the next gather (two fold slots)
        # params split_first_dn: each cluster's first down slice as one GEMM per post block, each behind its block's
        # quantise and A8 copy only, so the end of the post-processing runs under down GEMMs
        split0 = bool(self.p.get("split_first_dn", False))

        def load(i, jl):
            if jl == 0 and self.dn_first_load:
                return dict(self.dn_first_load)
            return {g: self._load(g, i, len(self.chunks[g]) + jl) for g in range(self.G)}

        yt_puts = {}

        GS = self.chain_group

        def store(c, i, j, red, s):
            # slices j .. j + GS - 1 (one gather), slice n of them at n x SL in the root's fold slot s
            sl = [self.dn_sl[c][j - c * h + n] for n in range(GS)]
            base = s * GS * SL
            if self.y_scatter:
                # the slice's row tiles (D layout: each 16 rows x w contiguous) into their owners' tiles, at the
                # slice's columns, each behind slice j's flag of that tile
                puts = []
                for n, (o0, w) in enumerate(sl):
                    jn = j + n
                    for m in range(self.ROWS // 16):
                        q = i * (self.ROWS // 16) + m
                        own, lt = self.y_owner(q)
                        far = None if own == R else own
                        sfx = f"j{jn}m{m}" if i == 0 else f"i{i}j{jn}m{m}"
                        puts.append(self.node(R, c, dm, f"y_sc_{sfx}", "__snax_bingo_kernel_idma_put_flagged",
                                              SnaxBingoKernelIdmaPutFlaggedArgs(
                                                  self.p1c[c].view(base + n * SL + m * 16 * w * 2),
                                                  self.yt_sym(q, o0 * 16 * 2, chip_id=far), 16 * w * 2,
                                                  BingoMemSymbol(f"{self.TAG}_ytflag", offset=(lt * self.NDN + jn) * 64,
                                                                 chip_id=far), value=1), [red]))
                        yt_puts.setdefault(own, []).append(puts[-1])
                stores.extend(puts)
                p1_reader[(c, s)] = puts
                return
            sts = self._y_stores(c, i, j, red, s)
            stores.extend(sts)
            last_st[c] = sts
            p1_reader[(c, s)] = sts

        # params x_early: X_{i+1} fetched behind pass i's second down chunk load in the DM core's program (the down
        # GEMMs leave the DM core idle), not at the top of pass i + 1 behind every down load and store of pass i
        x_early = bool(self.p.get("x_early", False))
        xpre = {}
        last_red, last_gu, last_dn = None, {}, {}

        def fetch_x(i):
            # X_i (i >= 1) into the buffer once the last gate|up GEMM that read it (pass i - 1) is done
            return self.stage_x(i, self.x_bc[i], {g: list(after_x[g]) + [last_gu[i - 1][g]] for g in range(self.G)})
        for i in range(self.NI):
            if i:
                self.pass_start(i)
                x = xpre.pop(i) if i in xpre else fetch_x(i)
                self.push_war = {g: [last_dn[self.gs(self.chip(g))[1 - self.c(g)]], last_red] for g in range(self.G)}
            next_put = None
            sw, last_gu[i] = self.stage_gu(i, x)
            post = self.stage_post(i, sw)
            ex_last = {g: self.ex_push[(g, i)][-1] for g in range(self.G)}
            a8_tile = lambda g, m, i=i: [self.q_tile[g][m], self.ex_push[(self.gs(self.chip(g))[1 - self.c(g)], i)][m]]
            a8_all = lambda g: [post[g], ex_last[self.gs(self.chip(g))[1 - self.c(g)]]]
            if i == 0:
                next_put = self.x_bc[-1]            # every multicast before the first gather
            lds = load(i, 0)
            prev_dn = {}
            for jl in range(h):
                so = jl % self.out_slots                # the clusters' output slot
                dn = {}
                for g in range(self.G):
                    k, c = self.chip(g), self.c(g)
                    ld, si, slab = lds[g]
                    kn = self.kcols[k][1]
                    nt = self.dn_sl[c][jl][1] // BLK        # this slice's column tiles
                    kargs = dict(input_B_addr=slab, input_C_addr=0, K=kn // 4, N=nt, array_shape_idx=0,
                                 transpose_A=0, transpose_B=0, accumPrevC=0, int32tofp16_enable=1, int4_b_enable=self.int4_b,
                                 d_shift=self.d.k_dn)
                    if split0 and jl == 0:          # one GEMM per post block, behind that block's A8 rows only
                        mb = self.MT // self.post_blocks
                        for b in range(self.post_blocks):
                            m1 = (b + 1) * mb - 1   # the block's last row tile: its quantise and copy come last
                            dn[g] = self.gemm_node(k, c, f"dn_c{c * h}_i{i}b{b}_g{g}",
                                                   SnaxBingoKernelGemmFullArgs(
                                                       input_A_addr=self.a8[g][0].view(b * mb * 16 * kn),
                                                       output_D_addr=self.sym[(k, c)].view(so * SL + b * mb * nt * TILE),
                                                       M=mb, **kargs),
                                                   [ld] + a8_tile(g, m1) + sym_reader.get((k, c, so), []), i)
                    else:
                        dn[g] = self.gemm_node(k, c, f"dn_c{c * h + jl}_i{i}_g{g}",
                                               SnaxBingoKernelGemmFullArgs(input_A_addr=self.a8[g][0],
                                                                           output_D_addr=self.sym[(k, c)].view(so * SL),
                                                                           M=self.MT, **kargs),
                                               [ld] + a8_all(g) + sym_reader.get((k, c, so), []), i)
                    self.ls[g].release(si, dn[g])
                    if jl % GS:
                        # params chain_group: a gather waits for its group's last down GEMM on each chip only; the
                        # GEMM core runs its queue in order, so the group's GEMMs go into it in slice order
                        self.dfg.bingo_add_order_edge(prev_dn[g], dn[g])
                    prev_dn[g] = dn[g]
                if jl + 1 < h:
                    lds = load(i, jl + 1)
                if x_early and jl == 0 and i + 1 < self.NI:
                    xpre[i + 1] = fetch_x(i + 1)
                while len(pending) > lag * self.NC:
                    store(*pending.pop(0))
                if (jl + 1) % GS:
                    continue
                # the group's gathers: slices j0 .. jl, in output slots so0 .. so (back to back), into the root's fold
                # slot s (GS slices)
                j0 = jl + 1 - GS
                s, so0 = (j0 // GS) % 2, j0 % self.out_slots
                for c in range(self.NC):
                    j = c * h + j0
                    # params chain_alt: cluster 1's gathers run the ring the other way, so each gather's data follows the
                    # previous gather's finish back along the links instead of meeting it
                    ordr = others[::-1] if self.chain_alt and c == 1 else others
                    ga = SnaxBingoKernelXdmaChainGatherArgs.fold_into(
                        cm.remote, R, c, ordr, src=self.sx(c, so0 * SL), dst=f"({p1x[c]} + {s * GS * SL})",
                        size=(GS - 1) * SL + self.ROWS * self.dn_sl[c][jl][1] * 2,
                        junction=self.jct["Elementwise"], jct_csr0=EW_F16_ADD, holds=[self.p1c[c]])
                    # the A8 copies into cluster c precede its gathers through its own down GEMMs
                    first = [last_red] if last_red is not None else []
                    if next_put is not None and j0 == 0 and c == 0:
                        first.append(next_put)
                    ins = [dn[g] for g in range(self.G) if self.c(g) == c]
                    red = self.gather_node(c, f"y_chain_j{j}" if i == 0 else f"y_chain_i{i}j{j}", ga,
                                           ins + p1_reader.get((c, s), []) + first)
                    last_red = red
                    for k in self.chips:
                        for o in range(so0, so0 + GS):
                            sym_reader[(k, c, o)] = [red]
                    pending.append((c, i, j, red, s))
            last_dn = dict(dn)
            for g in range(self.G):
                self.a8_reader[g] = [dn[g], dn[g]]
        while pending:
            store(*pending.pop(0))
        if self.y_scatter:
            # each chip's tiles of Y_raw have landed: one wait a chip, on its cluster 0
            waits = []
            for k in self.chips:
                puts = yt_puts.get(k, [])
                w = self.node(k, 0, dm, "yt_wait", "__snax_bingo_kernel_wait_flags",
                              SnaxBingoKernelWaitFlagsArgs(BingoMemSymbol(f"{self.TAG}_ytflag"), len(puts), 1),
                              [t for t in puts if t.assigned_chiplet_id == k])
                for t in puts:
                    if t.assigned_chiplet_id != k:
                        self.dfg.bingo_add_order_edge(t, w)
                waits.append(w)
            self.end_window(waits)
        else:
            self.end_window([t for c in sorted(last_st) for t in last_st[c]])   # each root cluster's last stores
        self._checks()

    def _passes_kernel(self, go, after_x):
        """params sync barrier: the passes kernel by kernel, as a conventional system runs them, a
        chip barrier after every kernel fencing every core of the chip (fence_barriers), so no two kernels overlap and
        nothing crosses chips but at a barrier. Per pass:
            K1  X into L1, then every gate|up chunk, its dequantise and SwiGLU the GEMM's epilogue
            K2  A16 reshaped and quantised into the chip's A8, each cluster's columns copied into the other's
            K3  every down slice, slice j into output slot j (out_view: the symmetric buffer's two slots, then the X
                buffer's, which nothing reads until the next pass's X)
            K4  every slice reduced into the root (_reduce_kernel) and stored
        The chunks, slices, buffers and kernels of the tuned schedule (_passes); only the order and the
        synchronisation differ."""
        gemm, h = self.roles["gemm"], self.NDN // self.NC
        allk = lambda: {k: [] for k in self.chips}
        self.x_bc = self.x_hbm_bcasts(go)
        if self.x_blocks > 1:
            x = {g: self.x_wait_block(g, 0, self.x_bc[0], after_x[g]) for g in range(self.G)}
        else:
            x = self.stage_x(0, self.x_bc[0], after_x)
        last_dn, last_st = {}, []
        for i in range(self.NI):
            if i:
                self.pass_start(i)
                x = self.stage_x(i, self.x_bc[i], after_x)
                self.push_war = {g: [last_dn[self.gs(self.chip(g))[1 - self.c(g)]]] for g in range(self.G)}
            sw, _ = self.stage_gu(i, x)                                 # K1
            self.barrier(allk())
            post = self.stage_post(i, sw)                               # K2
            ex_last = {g: self.ex_push[(g, i)][-1] for g in range(self.G)}
            self.barrier(allk())
            dn = {}                                                     # K3
            lds = {g: self._load(g, i, len(self.chunks[g])) for g in range(self.G)}
            for jl in range(h):
                for g in range(self.G):
                    k, c = self.chip(g), self.c(g)
                    ld, si, slab = lds[g]
                    dn[g] = self.gemm_node(k, c, f"dn_c{c * h + jl}_i{i}_g{g}",
                                           SnaxBingoKernelGemmFullArgs(
                                               input_A_addr=self.a8[g][0], output_D_addr=self.out_view(g, jl), M=self.MT,
                                               input_B_addr=slab, input_C_addr=0, K=self.kcols[k][1] // 4,
                                               N=self.dn_sl[c][jl][1] // BLK, array_shape_idx=0, transpose_A=0,
                                               transpose_B=0, accumPrevC=0, int32tofp16_enable=1, int4_b_enable=self.int4_b,
                                               d_shift=self.d.k_dn),
                                           [ld, post[g], ex_last[self.gs(k)[1 - c]]], i)
                    self.ls[g].release(si, dn[g])
                if jl + 1 < h:
                    lds = {g: self._load(g, i, len(self.chunks[g]) + jl + 1) for g in range(self.G)}
            last_dn = dict(dn)
            for g in range(self.G):
                self.a8_reader[g] = [dn[g], dn[g]]
            self.barrier(allk())
            last_st = self._reduce_kernel(i)                            # K4
            if i + 1 < self.NI:
                self.barrier(allk())
        self.end_window(last_st)
        self._checks()

    def _y_store(self, c, i, j, src, after):
        """Slice j of Y_raw (in root cluster c's half of the outputs) from the root's L1 `src` into its L3."""
        o0, w = self.dn_sl[c][j - c * (self.NDN // self.NC)]
        return self.node(self.ROOT, c, self.roles["dm"], f"y_store_j{j}" if i == 0 else f"y_store_i{i}j{j}",
                         "__snax_bingo_kernel_idma_1d_copy",
                         SnaxBingoKernelIdma1dCopyArgs(src, self._ysym(i * self.YB + o0 * self.ROWS * 2),
                                                       self.ROWS * w * 2), list(after))

    def _y_stores(self, c, i, j, red, s):
        """The gather `red` of slices j .. j + chain_group - 1 of root cluster c, in its fold slot s, into Y_raw in L3: one
        copy when the slices fill their fold slots and their columns follow each other (Y_raw_i in L3 is slice by slice, a
        slice's ROWS x w at its first column x ROWS); else one copy a slice. Returns the copies."""
        GS, SL, h = self.chain_group, self.SLICE, self.NDN // self.NC
        sl = [self.dn_sl[c][j - c * h + n] for n in range(GS)]
        whole = all(self.ROWS * w * 2 == SL for _, w in sl[:-1]) and \
            all(sl[n][0] == sl[n - 1][0] + sl[n - 1][1] for n in range(1, GS))
        runs = [(0, sl[0][0], (GS - 1) * SL + self.ROWS * sl[-1][1] * 2)] if whole else \
            [(n * SL, o0, self.ROWS * w * 2) for n, (o0, w) in enumerate(sl)]
        sts = []
        for n, (off, o0, nb) in enumerate(runs):
            sfx = f"j{j}" if n == 0 else f"j{j}n{n}"
            sts.append(self.node(self.ROOT, c, self.roles["dm"], f"y_store_{sfx}" if i == 0 else f"y_store_i{i}{sfx}",
                                 "__snax_bingo_kernel_idma_1d_copy",
                                 SnaxBingoKernelIdma1dCopyArgs(self.p1c[c].view(s * GS * SL + off),
                                                               self._ysym(i * self.YB + o0 * self.ROWS * 2), nb),
                                 [red]))
        return sts

    def _reduce_kernel(self, i):
        """K4 of pass i (sync barrier): every slice of both halves of the outputs reduced into the root and stored
        to Y_raw in its L3, slice by slice. reduce chain: the ChainGathers of _passes, one at a time, into the root's
        two fold slots, each stored a gather later. reduce root: each root cluster pulls every other
        chip's partial of its half (one pull in flight, two L1 slots) and its SIMD adds it into the cluster's own, in
        the fold order, then stores the slice. Returns each root cluster's last store."""
        R, dm, xdma, simd = self.ROOT, self.roles["dm"], self.roles["xdma"], self.roles["simd"]
        SL, others, h = self.SLICE, self.order[1:], self.NDN // self.NC
        last = {}
        if self.reduce_impl == "chain":
            p1x = {c: f"((uint64_t)({self.p1c[c].get_c_var_name()}))" for c in range(self.NC)}
            lag = 1
            pending, p1_reader, last_red = [], {}, None
            GS = self.chain_group               # params chain_group: a gather carries this many slices, as in _passes
            for jl in range(0, h, GS):
                s = jl // GS % 2
                while len(pending) > lag * self.NC:
                    c, j, red, s0 = pending.pop(0)
                    p1_reader[(c, s0)] = last[c] = self._y_stores(c, i, j, red, s0)
                for c in range(self.NC):
                    j = c * h + jl
                    ordr = others[::-1] if self.chain_alt and c == 1 else others
                    ga = SnaxBingoKernelXdmaChainGatherArgs.fold_into(
                        cm.remote, R, c, ordr, src=self.sx(c, jl * SL), dst=f"({p1x[c]} + {s * GS * SL})",
                        size=(GS - 1) * SL + self.ROWS * self.dn_sl[c][jl + GS - 1][1] * 2,
                        junction=self.jct["Elementwise"], jct_csr0=EW_F16_ADD,
                        # the root's own output slots past the symmetric buffer sit in its X buffer, which the gather
                        # reads through an address expression the static planner does not see
                        holds=[self.p1c[c], self.xbuf[self.gs(R)[c]][0]])
                    last_red = self.gather_node(c, f"y_chain_j{j}" if i == 0 else f"y_chain_i{i}j{j}", ga,
                                                p1_reader.get((c, s), []) + ([last_red] if last_red is not None else []))
                    pending.append((c, j, last_red, s))
            for c, j, red, s0 in pending:
                last[c] = self._y_stores(c, i, j, red, s0)
        else:
            for c in range(self.NC):
                g = self.gs(R)[c]
                pulls, adds = [], []
                for jl in range(h):
                    j = c * h + jl
                    nb = self.ROWS * self.dn_sl[c][jl][1] * 2
                    acc = self.out_view(g, jl)          # the root's own partial, the others folded into it
                    for n_, k in enumerate(others):
                        q = len(pulls)
                        slot = self.p1c[c].view((q % 2) * SL)
                        sfx = f"j{j}n{n_}" if i == 0 else f"i{i}j{j}n{n_}"
                        pulls.append(self.node(R, c, dm, f"y_pull_{sfx}", "__snax_bingo_kernel_idma_1d_copy",
                                               SnaxBingoKernelIdma1dCopyArgs(cm.remote(k, self.sx(c, jl * SL)), slot, nb),
                                               pulls[-1:] + adds[q - 2:q - 1]))
                        adds.append(self.node(R, c, simd, f"y_add_{sfx}", "__snax_bingo_kernel_simd_stream_elementwise",
                                              SnaxBingoKernelSimdAddF16Args(acc, slot, acc, nb // 64, 32),
                                              [pulls[-1]] + adds[-1:]))
                    last[c] = [self._y_store(c, i, j, acc, adds[-1:])]
        return [t for c in sorted(last) for t in last[c]]

    def _sync_graph(self):
        """The graph the hazard check walks: the DFG, plus (sync barrier) what a chip barrier orders across chips: for
        each barrier round a node behind every chip's barrier task and ahead of every task that follows any of them."""
        if self.sync != "barrier":
            return self.dfg
        import networkx as nx
        H = nx.DiGraph(self.dfg)
        for n, rnd in enumerate(self.barrier_rounds):
            v = ("barrier_round", n)
            for t in rnd.values():
                H.add_edge(t, v)
                H.add_edges_from((v, s) for s in self.dfg.successors(t))
        return H

    def check_window_local(self):
        """sync barrier: no dependency edge crosses chips between the start barrier and the end stamp; only the
        barriers synchronise the chips there. Returns the offending edges."""
        import networkx as nx
        pre = set().union(*(nx.ancestors(self.dfg, t) for t in self.go.values()))
        post = nx.descendants(self.dfg, self.end) | {self.end}
        return [(u, v) for u, v in self.dfg.edges()
                if u.assigned_chiplet_id != v.assigned_chiplet_id and u not in pre and u not in post and v not in post]

    def check_symmetric(self):
        """Every symmetric buffer at offset 0 of its arena; with sync barrier also every X buffer right behind it,
        where the output slots after the symmetric buffer's sit."""
        super().check_symmetric()
        if self.per_kernel:
            pl = self.dfg.static_l1_placement
            bad = [(g, pl[id(self.xbuf[g][0])][1] if id(self.xbuf[g][0]) in pl else None) for g in range(self.G)
                   if id(self.xbuf[g][0]) not in pl or pl[id(self.xbuf[g][0])][1] != self.sym_bytes]
            if bad:
                raise ValueError(f"sync barrier: X buffers not at offset {self.sym_bytes}: {bad[:4]}")

    def _checks(self):
        """Y_raw sampled on the root's L3 (the first and the last 2 KiB of every pass), or with y_scatter the first, a
        middle and the last tile in their owners' L3; and the root's own A8 of the last pass (2 KiB)."""
        lay, R = self.lay, self.ROOT
        y = self.gold["y_raw"]
        checks = []
        n = 2048
        if self.y_scatter:
            nt = self.T // 16
            for q in (0, nt // 2, nt - 1):
                img = lay.to_d(y[q * 16:(q + 1) * 16], self.d.mesh).astype(np.float16).reshape(-1).view(np.uint8)
                for name, off in ((f"yt_q{q}_a", 0), (f"yt_q{q}_m", self.YT_B // 2 - n // 2), (f"yt_q{q}_b", self.YT_B - n)):
                    checks.append((name, self.yt_sym(q, off, chip_id=self.y_owner(q)[0]), img[off:off + n], n))
        for i in range(self.NI if not self.y_scatter else 0):
            rows = y[i * self.ROWS:(i + 1) * self.ROWS]
            img = np.concatenate([lay.to_d(rows[:, c0:c0 + w], self.d.mesh)
                                  for c0, w in self.y_slices()]).astype(np.float16).view(np.uint8)
            for name, off in ((f"y_i{i}_a", 0), (f"y_i{i}_b", self.YB - n)):
                checks.append((name, self._ysym(i * self.YB + off, chip_id=R),
                               img[off:off + n], n))
        g = self.gs(R)[0]
        c0, ncol = self.kcols[R]                # the chip's A8, both clusters
        i = self.NI - 1
        a8 = self.d.a8[i * self.ROWS:(i + 1) * self.ROWS, c0:c0 + ncol]
        img = lay.to_a(a8, self.d.mesh).reshape(-1).view(np.uint8)
        got = self.stash("a8_root", R, 0, self.a8[g][i % 2], n, [])
        checks.append(("a8_root", got, img[:n], n))
        self.run_checks(checks)

    def y_slices(self):
        """[(first column, columns)] of Y in the order its D-layout slices sit in L3."""
        return [sl for c in range(self.NC) for sl in self.dn_sl[c]]

    def check_hazards(self):
        """The passes reuse L1 buffers while an earlier slice's or pass's tail is still running on other cores;
        nothing orders those but the edges the generator adds. For every cluster and every pair of consecutive uses
        of a buffer slot, the last reader of the earlier use must reach the first writer of the later one in the DFG
        (the cores run in the order of their edges): a missing edge is a data race that no golden of one pass shows.
        Returns the violations."""
        import re
        import networkx as nx
        by, byk = {}, {}
        for t in self.dfg.node_list:
            m = re.match(r"^(.*)_k([0-9a-f]{2})_cl(\d)$", t.node_name)
            if m:
                by[m.group(1)] = t                                   # per-cluster tasks: names carry g
                byk[(m.group(1), int(m.group(2), 16), int(m.group(3)))] = t   # per-chip tasks
        bad = []
        H = self._sync_graph()
        # a GEMM's last read of its operands: its task, or (params gemm_async) its wait
        rd = lambda name: by.get(f"{name}_w") or by.get(name)

        def ordered(a, b, what):
            if a is None or b is None:
                bad.append(f"{what}: a task is missing")
            elif not nx.has_path(H, a, b):
                bad.append(f"{what}: {a.node_name} is not before {b.node_name}")

        # cluster c's output slot: its gather of a slice before the next down GEMM into that slot (out_slots slices
        # later, or in the next pass); the root's fold slot: the store of a slice before the next gather into it; every
        # A8 copy into a cluster before that cluster's first gather of the pass. Between passes: the A8, A16, A16a and X
        # buffers' last readers of a pass before their first writers of the next
        R, h, pb, GS = self.ROOT, self.NDN // self.NC, self.post_blocks, self.chain_group
        grp = lambda j, c: c * h + (j - c * h) // GS * GS       # params chain_group: the gather of slice j
        store = lambda i, j, c: byk.get((f"y_store_j{grp(j, c)}" if i == 0 else f"y_store_i{i}j{grp(j, c)}", R, c))
        # the task after which every chip's output slot of slice j has been read: its gather, or (params reduce root)
        # the slice's store, behind every pull and add of it
        chain = (lambda i, j, c: byk.get((f"y_chain_j{grp(j, c)}" if i == 0 else f"y_chain_i{i}j{grp(j, c)}", R, c))) \
            if self.reduce_impl == "chain" else store
        push = lambda i, m, g: by.get(f"a8_push_m{m}_g{g}" if i == 0 else f"a8_push_i{i}m{m}_g{g}")

        def dn_first(i, c, jl, g):     # the first down GEMM of a slice (a split first slice: its block 0)
            return by.get(f"dn_c{c * h + jl}_i{i}_g{g}") or by.get(f"dn_c{c * h}_i{i}b0_g{g}")
        for c in range(self.NC):
            uses = [(i, jl) for i in range(self.NI) for jl in range(h)]
            S = self.out_slots
            for n_, (i, jl) in enumerate(uses):
                j = c * h + jl
                nxt = next(((i2, jl2) for i2, jl2 in uses[n_ + 1:] if jl2 % S == jl % S), None)
                if nxt is not None:
                    for g in range(self.G):
                        if self.c(g) == c:
                            ordered(chain(i, j, c), dn_first(nxt[0], c, nxt[1], g),
                                    f"output slot g{g} i{i}j{j}->i{nxt[0]}j{c * h + nxt[1]}")
                # the next gather into the same fold slot (another group); a root fold's slots are its own pipeline's
                nxt = next(((i2, jl2) for i2, jl2 in uses[n_ + 1:] if jl2 // GS % 2 == jl // GS % 2
                            and (i2, jl2 // GS) != (i, jl // GS)), None)
                if nxt is not None and self.reduce_impl == "chain":
                    ordered(store(i, j, c), chain(nxt[0], c * h + nxt[1], c),
                            f"root fold slot c{c} i{i}j{j}->i{nxt[0]}j{c * h + nxt[1]}")
        for i in range(self.NI):
            for g in range(self.G):          # every copy into cluster c before cluster c's first gather of the pass
                c = self.c(g)
                first = chain(i, c * h, c)
                other = self.gs(self.chip(g))[1 - c]
                for m in range(self.MT):
                    ordered(push(i, m, other), first, f"A8 copy i{i}m{m} into g{g} before its first gather")
                if i == 0:
                    continue
                nG = len(self.chunks[g])
                last_dn = rd(f"dn_c{c * h + h - 1}_i{i - 1}_g{g}")
                ordered(last_dn, by.get(f"quant_i{i}b0m0_g{g}"), f"A8 g{g} i{i - 1}->i{i} (quantise)")
                ordered(last_dn, push(i, 0, other), f"A8 g{g} i{i - 1}->i{i} (copy from g{other})")
                ordered(chain(i - 1, self.NDN - 1, self.NC - 1), push(i, 0, g),
                        f"g{g}'s first A8 copy of pass {i} after the last gather of pass {i - 1}")
                ordered(by.get(f"d2a_i{i - 1}b{pb - 1}_g{g}"), by.get(f"swi_c0m0_i{i}_g{g}"), f"a16d g{g} i{i - 1}->i{i}")
                ordered(by.get(f"quant_i{i - 1}b{pb - 1}m{self.MT - 1}_g{g}"), by.get(f"d2a_i{i}b0_g{g}"),
                        f"a16a g{g} i{i - 1}->i{i}")
                lg = rd(f"gu_c{nG - 1}_i{i - 1}b{pb - 1}_g{g}") or rd(f"gu_c{nG - 1}_i{i - 1}_g{g}")
                ordered(lg, by.get(f"x_wait_i{i}_g{g}"), f"x g{g} i{i - 1}->i{i}")
        if self.per_kernel:
            # the output slots in the X buffer: read by pass i - 1's reduce before X_i lands there, and X_i read by the
            # pass's last gate|up GEMM before a down GEMM writes there
            for i in range(self.NI):
                for g in range(self.G):
                    c = self.c(g)
                    lg = rd(f"gu_c{len(self.chunks[g]) - 1}_i{i}_g{g}")
                    for jl in range(self.sym_slots, h):
                        if i:
                            ordered(chain(i - 1, c * h + jl, c), by.get(f"x_wait_i{i}_g{g}"),
                                    f"X buffer g{g}: slot {jl} of pass {i - 1} reduced before X_{i} lands")
                        ordered(lg, dn_first(i, c, jl, g), f"X buffer g{g}: X_{i} read before slot {jl} is written")
        if self.l3_overlay:
            # Y_raw_i over X_i's L3 buffers: the root clusters' fetches of X_i before every store of pass i
            for i in range(self.NI):
                blk = [f"i0b{b}" for b in range(self.x_blocks)] if i == 0 and self.x_blocks > 1 else [f"i{i}"]
                readers = [by.get(f"x_wait_{t}_g{g}") for g in self.gs(R) for t in blk]
                for c in range(self.NC):
                    for jl in range(h):
                        for r in readers:
                            ordered(r, store(i, c * h + jl, c), f"X_{i} read before the store over it, y i{i}j{c * h + jl}")
        return bad

    def finish(self, app_name, x_prefixes=("x_wait_",)):
        self.chain_program_order()
        if not self.gemm_async:
            # a GEMM whose configuration equals the previous GEMM's on its core (every
            # field but the four operand addresses) runs as gemm_minimal; each core runs its tasks in
            # creation order (chain_program_order), so the previous GEMM there is the one created before it
            print(f"[{self.TAG}] gemm_reuse: {reuse_gemm_config(self.dfg)} GEMMs run as gemm_minimal")
        if self.sync == "barrier":
            if self.barrier_fence:              # the fences first, so the hazard check sees them
                self.fence_barriers()
                self.barrier_fence = False
            cross = self.check_window_local()
            if cross:
                raise ValueError(f"sync barrier: {len(cross)} dependency edges cross chips inside the window, e.g. "
                                 + "; ".join(f"{u.node_name} -> {v.node_name}" for u, v in cross[:4]))
            print(f"[{self.TAG}] sync barrier (kernel): {self.n_barrier} barriers a chip, no edge crosses "
                  f"chips inside the window")
        if self.p.get("args_phase_bytes"):
            # params args_phase_bytes: each cluster's device args in phases of at most this many bytes, two alternating
            # L1 regions (mini_compiler passes/bingo_args_phases.py), for a cluster whose args do not fit beside its static
            # buffers (the root's, with params reduce root); after the barrier fences, so the phase loads are not fenced
            from bingo_args_phases import split_args_phases
            split_args_phases(self, int(self.p["args_phase_bytes"]), [self.roles[r] for r in ("gemm", "simd", "xdma", "dm")])
        if self.gemm_async:
            # with gemm_async: a gemm_start that repeats its cluster's previous dispatch (every field but
            # the four operand addresses) writes only those addresses and starts (bingo_cfg_reuse), along the cores'
            # programs as they stand after the args phases
            n = reuse_gemm_start_config(self.dfg, [self.roles["gemm"]])
            n_cfg = sum(t.kernel_name.endswith("gemm_start") and t.kernel_args.stage != 2 for t in self.dfg.node_list)
            print(f"[{self.TAG}] gemm_reuse: {n} of {n_cfg} gemm_start configurations reuse")
        bad = self.check_hazards()
        if bad:
            raise ValueError(f"{len(bad)} buffer hazards, e.g. " + "; ".join(bad[:4]))
        print(f"[{self.TAG}] buffer hazards: none (A16, A16a, A8, X, output and fold slots; every cluster)")
        # one host function per chip: a single one of every chip's block compiles too slowly
        self.dfg.split_chip_functions = True
        super().finish(app_name, x_prefixes)


def _app_name(S):
    arms = "" if (S.x_cast, S.reduce_impl, S.sync) == ("multicast", "chain", "edge") else \
        f"; X {S.x_cast}, reduce {S.reduce_impl}, sync {S.sync}"
    return (f"{S.p.get('model', 'dsv2-lite-l0')} FFN prefill, {S.T} tokens in {S.NI} x {S.ROWS}, "
            f"{len(S.chips)} chips x {S.NC} clusters, X from HBM{arms}")


def run():
    run_driver("ffn prefill workload generator", PBuild, _app_name)
