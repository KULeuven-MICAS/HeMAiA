# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""The shared driver of the ffn prefill workloads (workloads/ffn/<model>_prefill/<variant>/): one
LLM's dense SwiGLU FFN over a prompt of T tokens, `rows` tokens an iteration, on the (16, 4, 16)
GEMM shape with INT4 weights, tensor-parallel over every cluster of the 16 + 4 chiplet platform.
The platform, the weight rings, the anchors and the synchronisation arm are chiplet_mc.py's; the
data and the golden are ffn_data.py's.

THE MAPPING (Megatron, as ffn_mc.py): cluster g takes the intermediate columns J_g (whole
32-column pairs), the same columns of gate and of up, and the matching rows of down, so its
partial Y_g covers every output and Y = sum_g Y_g. Per iteration i (rows 32 i .. 32 i + 31):

    X_i         the root's prompt rows (INT8, A layout) into every cluster's L1
    gate|up     per weight chunk: GEMM X_i . [gate | up] chunk -> G (D layout, FP16, one tile
                per chunk), dequantised in place (x s_gu), SwiGLU into A16 (D layout)
    A8          A16 reshaped D -> A on the xDMA (FP16), quantised to INT8
    down        per weight chunk (512 output columns): GEMM A8 . down chunk -> its slice of Y_g
    chip fold   cluster 1 hands Y_g to cluster 0, which adds it into its own (loc)
    reduce      the chips' loc folded into the root, which stores Y_raw_i in its L3

THE WEIGHTS stream once. Each cluster's gate|up and down chunks sit in its memory chip's HBM;
in iteration 0 the memory chip pushes them into the chip's L3 ring and the DM core loads each
behind its landed flag; the ring holds every chunk of a cluster (no slot is pushed twice), so
the later iterations copy each chunk from the slot it landed in. A gate|up chunk is `blocks`
16-column blocks of gate, then the same blocks of up, so its SwiGLU needs nothing outside the
chunk.

THE MECHANISMS (params), the ladder base -> bcast -> bcast_reduce -> all adding one each:
    bcast   bcast_l1: the root multicasts X_i into every chip's L3 (one D2D multicast and a
            landed flag) and each cluster fetches it into L1 behind the flag. pull: every
            cluster's iDMA reads X_i from the root's L3.
    reduce  chain: one ChainGather per iteration through the fold order, every chip's writer
            junction adding its loc in transit, into the root. root: the root's iDMA loads
            every chip's loc, 32 KiB at a time, into two L1 slots and its SIMD folds each into
            the root's (iDMA load-all + local fold).
    sync    edge: every cross-chip dependency a BINGO hardware edge, and the program runs
            software-pipelined: the arrays take GU0 GU1 DN0 GU2 DN1 GU3 DN2 DN3, so an
            iteration's SwiGLU, reduce and the next X overlap other GEMMs. barrier: bulk
            synchronous; each iteration's partials meet at a chip barrier before the reduce and
            the next iteration starts after another, and every barrier fences every core of its
            chip (chiplet_mc.fence_barriers).

THE PROGRAM ORDER. Every device core runs its tasks in the order this driver creates them
(chain_program_order), the same rule in every variant; a variant differs only in that order's
schedule and in its mechanisms.

TIMING. The start anchor after every chip's setup (the scales loaded, the buffers zeroed); the
end anchor after the root stores the last iteration's Y_raw. The weight stream is inside the
window.
"""

import argparse
import os
import sys

import hjson
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = HERE.split("/target/sw/host/")[0]
sys.path.insert(0, f"{REPO}/target/sw/host/runtime/libbingo/mini_compiler")
sys.path.insert(0, HERE)

import _bingo_paths  # noqa: F401,E402
from bingo_kernel_args import (  # noqa: E402
    INT4_B_LAYOUT_B, INT4_B_LAYOUT_PAIRED, SnaxBingoKernelGemmFullArgs, SnaxBingoKernelGemmMinimalArgs, SnaxBingoKernelIdma1dCopyArgs, SnaxBingoKernelIdmaFetchFlaggedArgs,
    SnaxBingoKernelIdmaPutFlaggedArgs, SnaxBingoKernelIdmaRingLoadArgs,
    SnaxBingoKernelSimdAddF16Args, SnaxBingoKernelSimdFp16ToInt8Args, SnaxBingoKernelSimdMulF16Args,
    SnaxBingoKernelSimdSwigluF16F16Args, SnaxBingoKernelXdmaChainGatherArgs, simd_beats)
from bingo_mem_handle import BingoMemSymbol  # noqa: E402
from bingo_platform import load_cluster_cfg, parse_platform_cfg  # noqa: E402
from libs.comm.nest import d_to_a_args  # noqa: E402

import chiplet_mc as cm  # noqa: E402
import ffn_data  # noqa: E402

TILE = 16 * 16 * 2                         # one D-layout FP16 tile
BLK = 16                                   # a GEMM column block (meshCol)
EW_F16_ADD = cm.EW_F16_ADD


class PBuild(cm.ChipletBuild):
    TAG = "pf"

    def __init__(self, args, p, plat, hw):
        super().__init__(args, p, plat, hw)
        if self.crest:
            raise ValueError("params weight_crest: the prefill streams plain INT4")
        R = self.ROWS = int(p.get("rows", 32))
        if R % 16 or self.T % R:
            raise ValueError(f"params tokens={self.T} rows={R}: rows a multiple of 16 dividing T")
        self.NI, self.MT = self.T // R, R // 16
        self.single = str(p.get("mapping", "iterations")) == "single_pass"
        self.deq_per_row = bool(p.get("deq_per_row", False))
        # params x_blocks (single pass): X multicast and fetched in this many row blocks, and the first
        # gate|up chunk run as one GEMM per block, so the arrays start when the first block has landed
        self.x_blocks = int(p.get("x_blocks", 1))
        # params post_blocks (single pass): the reshape and quantise of A16 in this many row blocks, so the SIMD
        # quantises a block while the xDMA reshapes the next (the gap between gate|up and down shrinks)
        self.post_blocks = int(p.get("post_blocks", 1))
        # params tail_split (single pass): the last down slice as this many narrower GEMMs, each with its own fold,
        # gather and store, so the reduce of the last part overlaps the GEMM of the next (the tail shrinks)
        self.tail_split = int(p.get("tail_split", 1))
        # params split_last (single pass, with post_blocks): the last gate|up chunk as one GEMM per post block,
        # each block's dequantise and SwiGLU, its reshape and the previous block's quantise right behind it,
        # so only the last block's work is left between the last gate|up GEMM and the first down GEMM
        self.split_last = bool(p.get("split_last", False))
        if self.split_last and (self.post_blocks < 2 or not self.deq_per_row):
            raise ValueError("params split_last needs post_blocks > 1 and deq_per_row")
        if self.post_blocks > 1 and (not self.single or self.MT % self.post_blocks):
            raise ValueError(f"params post_blocks={self.post_blocks}: single_pass only, dividing {self.MT} row tiles")
        if self.x_blocks > 1 and (not self.single or self.MT % self.x_blocks):
            raise ValueError(f"params x_blocks={self.x_blocks}: single_pass only, dividing {self.MT} row tiles")
        self.BPC = int(p.get("blocks", 3))          # 16-column gate blocks per gate|up chunk
        # params first_blocks / last_blocks: the first and the last gate|up chunk of every cluster this
        # many gate blocks wide (0 = blocks): a narrow first chunk is pushed sooner, so the first GEMM
        # starts sooner; a narrow last chunk leaves less dequantise and SwiGLU between gate|up and down
        self.first_blocks = int(p.get("first_blocks", 0))
        self.last_blocks = int(p.get("last_blocks", 0))
        self.DNC = int(p.get("dn_cols", 512))       # output columns per down chunk
        # params w4_paired: every INT4 weight chunk in the paired layout (blocks of 32 columns, util/layout.py
        # to_b_pairs), which the B reader walks 64 bytes a pass, as A; default the B-layout (32 bytes a pass). Every
        # GEMM's column tiles, and every down slice, then come in whole pairs
        self.w4_paired = bool(p.get("w4_paired", False))
        self.int4_b = INT4_B_LAYOUT_PAIRED if self.w4_paired else INT4_B_LAYOUT_B
        self.dn_unit = 2 * BLK if self.w4_paired else BLK
        if self.DNC % (self.dn_unit * self.tail_split):
            raise ValueError(f"params dn_cols={self.DNC} tail_split={self.tail_split}: every down GEMM whole "
                             f"{self.dn_unit}-column units (w4_paired: pairs of column tiles)")
        cache = os.path.join(REPO, p.get("weights", "target/sim/automation/sweep/llm_blocks/ffn/cache"))
        d = self.d = ffn_data.FfnData(cache, p.get("model", "dsv2-lite-l0"), args.dsv2_dir,
                                      seed=int(p.get("seed", 1)), wbits=self.wbits,
                                      tokens=self.T, inter=p.get("inter"))
        self.lay = d.layout
        self.D, self.I = d.H, d.I
        if self.D % self.DNC:
            raise ValueError(f"{self.D} outputs are not whole {self.DNC}-column down chunks")
        self.NDN = self.D // self.DNC
        # params root_light: the root chip's clusters take the smaller column share, since its
        # cluster 0 also multicasts X, runs every gather and stores Y
        self.root_light = bool(p.get("root_light", False))
        # params col_unit: the intermediate columns are split over the clusters in units of this many (default 32;
        # 16 = one GEMM column block, which evens the chips' totals out to 43 or 42 blocks)
        # params root_cols: the root chip's clusters get exactly this many columns (one number for both, or one per
        # cluster; the rest is split over the others), so its extra duties (X multicast, every gather and store)
        # cost no window time
        rc = p.get("root_cols", 0)
        rc = [int(v) for v in rc] if isinstance(rc, (list, tuple)) else [int(rc)] * self.NC
        self.cols = self.split_columns(self.I, last=self.gs(self.ROOT) if self.root_light else (),
                                       unit=int(p.get("col_unit", 32)),
                                       fixed=dict(zip(self.gs(self.ROOT), rc)) if any(rc) else None)
        self.chunks = {g: self._gu_chunks(n) for g, (_, n) in self.cols.items()}
        # params dn_split (single pass): rows = each cluster's down GEMMs take its own columns (K) for every output,
        # the chip's two partials folded by a push and a SIMD add; cols = each cluster takes the chip's columns (K)
        # for its half of the outputs, so it holds the chip's sum there and nothing is folded inside a chip (the two
        # clusters exchange their A8 once, before the first gather)
        self.dn_cols = str(p.get("dn_split", "rows")) == "cols"
        self.kcols = {}                             # {chip: (first column, columns)} with dn_split cols
        if self.dn_cols:
            if not self.single or self.NC != 2 or self.NDN % 2 or self.tail_split != 1 or self.post_blocks < 2:
                raise ValueError("params dn_split cols: single_pass, 2 clusters, an even number of down slices, "
                                 "no tail_split, post_blocks > 1")
            for k in self.chips:
                (a0, an), (b0, bn) = (self.cols[g] for g in self.gs(k))
                if b0 != a0 + an:
                    raise ValueError(f"params dn_split cols: chip {k:02x}'s columns are not contiguous")
                self.kcols[k] = (a0, an + bn)
            self.gold = d.golden([self.kcols[k] for k in self.chips],
                                 {k: [i] for i, k in enumerate(self.chips)}, self.fold_order())
            self.dn_sl = self._dn_out_split(bool(p.get("dn_balance", False)))
        else:
            self.gold = d.golden([self.cols[g] for g in range(self.G)],
                                 {k: self.gs(k) for k in self.chips}, self.fold_order())
        # params chain_alt (single pass): every other ChainGather runs the chips in the reverse order, so its
        # data follows the previous gather's finish back along the ring instead of meeting it mid-chain (a
        # collector retires before its finish has walked back to the first hop)
        self.chain_alt = bool(p.get("chain_alt", False))
        if self.chain_alt:
            if not self.single:
                raise ValueError("params chain_alt: single_pass only")
            loc, add16 = self.gold["loc"], d.fp.add16
            rev = self.order[1:][::-1] + self.order[:1]
            y = loc[rev[0]]
            for k in rev[1:]:
                y = add16(y, loc[k])
            yr = self.gold["y_raw"].copy()
            # the reversed gathers: every odd one, or with dn_split cols (which alternates the clusters' gathers)
            # every gather of cluster 1
            rsl = self.dn_sl[1] if self.dn_cols else [sl for n_, sl in enumerate(self.y_slices()) if n_ % 2]
            for c0, w in rsl:
                yr[:, c0:c0 + w] = y[:, c0:c0 + w]
            self.gold["y_raw"], self.gold["y16"] = yr, d.fp.mul16(yr, d.dn_f[None, :])
        for line in d.report(self.gold["y16"]):
            print(f"[pf] {line}")
        print(f"[pf] {self.T} tokens in {self.NI} iterations of {R}; fold order "
              f"{[f'{k:02x}' for k in self.order]}; columns per cluster "
              f"{sorted({n for _, n in self.cols.values()})}; gate|up chunks per cluster "
              f"{sorted({len(c) for c in self.chunks.values()})}")
        self.XB = R * self.D                    # X_i, INT8
        self.YB = R * self.D * 2                # a partial / Y_raw_i, FP16
        self.SL = int(p.get("root_slice", 32 * 1024))
        self._stage()
        self.make_streams()
        self.make_rings({})
        self._buffers()

    # ---- the plan -------------------------------------------------------------------------
    def _gu_chunks(self, n):
        """[(first gate block, blocks)] of a cluster's n columns: params first_blocks, then chunks of
        blocks, then params last_blocks."""
        nb = n // BLK
        f, l = self.first_blocks, self.last_blocks
        if not f and not l:
            return [(b0, min(self.BPC, nb - b0)) for b0 in range(0, nb, self.BPC)]
        if f > self.BPC or l > self.BPC or f + l > nb:
            raise ValueError(f"params first_blocks={f} / last_blocks={l}: at most blocks={self.BPC} each, "
                             f"together at most the {nb} blocks of a cluster")
        out = [(0, f)] if f else []
        end = nb - l
        out += [(b0, min(self.BPC, end - b0)) for b0 in range(f, end, self.BPC)]
        return out + ([(end, l)] if l else [])

    def _gu_cols(self, g, b0, nb_):
        """The gate|up chunk's columns of [gate | up]: its gate blocks, then the same up blocks."""
        c0, _ = self.cols[g]
        gate = np.arange(c0 + BLK * b0, c0 + BLK * (b0 + nb_))
        return np.concatenate([gate, self.I + gate])

    # ---- data -----------------------------------------------------------------------------
    def _pack_b(self, q):
        """An INT4 weight chunk [K, N] as the GEMMs read it: the B-layout, or (params w4_paired) the paired layout."""
        lay, mesh = self.lay, self.d.mesh
        if self.w4_paired:
            if q.shape[1] % (2 * mesh[2]):
                raise ValueError(f"params w4_paired: a weight chunk of {q.shape[1]} columns is not whole pairs of "
                                 f"{mesh[2]}-column tiles")
            return lay.pack_int4(lay.to_b_pairs(q, mesh))
        return lay.pack_int4(lay.to_b(q, mesh))

    def _stage(self):
        st, d, T, lay = self.st, self.d, self.TAG, self.lay
        R = self.ROWS
        # the prompt's rows, A layout per iteration: the root's source and every chip's landing
        xs = [lay.to_a(d.hq[i * R:(i + 1) * R]) for i in range(self.NI)]
        self.h_xsrc = st.put(f"{T}_xsrc", "int8_t", np.concatenate(xs).astype(np.int8))
        st.put_zeros(f"{T}_xland", "int8_t", self.NI * self.XB, preload=True)
        st.put_zeros(f"{T}_xflag", "uint8_t", self.NI * 64 * max(1, self.x_blocks))
        st.put_zeros(f"{T}_y", "uint8_t", self.NI * self.YB, preload=True)
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
            if self.dn_cols:                    # the chip's columns, this cluster's share of the outputs
                kc0, kn = self.kcols[k]
                for o0, w in self.dn_sl[self.c(g)]:
                    q = d.dn_q[kc0:kc0 + kn, o0:o0 + w]
                    parts.append(self._pack_b(q))
            else:
                for j in range(self.NDN):
                    q = d.dn_q[c0:c0 + n, j * self.DNC:(j + 1) * self.DNC]
                    parts.append(self._pack_b(q))
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
        self.OFF_LOC = 0
        # single pass: the down output is reduced slice by slice (DNC columns of all rows), through two
        # slots of each cluster's symmetric buffer and of cluster 0's fold buffer
        self.SLICE = R * (max(w for v in self.dn_sl.values() for _, w in v) if self.dn_cols else self.DNC) * 2
        self.sym_bytes = 2 * self.SLICE if self.single else self.YB
        # every cluster's partial is its symmetric buffer (cluster 0's is the chip's loc too)
        self.sym = {(k, c): self.cx_chip(k).at(c).l1(f"AAAA_{self.TAG}_sym", self.sym_bytes)
                    for k in self.chips for c in range(self.NC)}
        if self.dn_cols:                        # the gathers' destinations: the root's two clusters only
            self.p1 = {}
            self.p1c = {c: self.cx_chip(self.ROOT).at(c).l1(f"{self.TAG}_p1c", self.sym_bytes) for c in range(self.NC)}
        else:
            self.p1 = {k: self.cx_chip(k).at(0).l1(f"{self.TAG}_p1", self.sym_bytes) for k in self.chips}
        if not self.single:
            self.slots = self.cx_chip(self.ROOT).at(0).l1(f"{self.TAG}_slots", 2 * self.SL)
        nxb = int(self.p.get("x_buffers", 1 if self.single else 2))
        self.xbuf, self.gbuf, self.sbuf, self.a16d, self.a16a, self.a8 = {}, {}, {}, {}, {}, {}
        for g, (_, n) in self.cols.items():
            l1 = lambda name, nb: self.cx_chip(self.chip(g)).at(self.c(g)).l1(f"{self.TAG}_{name}", nb)
            self.xbuf[g] = [l1(f"x{b}", self.XB) for b in range(nxb)]
            gb = MT * 2 * self.BPC * TILE
            self.gbuf[g] = [l1(f"g{b}", gb) for b in range(2)]
            self.sbuf[g] = l1("sgu", self.sdup[g][1])
            self.a16d[g] = l1("a16d", R * n * 2)
            self.a16a[g] = l1("a16a", R * n * 2)
            if self.dn_cols:                    # the chip's A8: both clusters' columns, one buffer
                a8 = l1("a8_0", R * self.kcols[self.chip(g)][1])
                self.a8[g] = [a8, a8]
            else:
                self.a8[g] = [l1(f"a8_{b}", R * n) for b in range(2)]

    # ---- helpers --------------------------------------------------------------------------
    def chain_program_order(self):
        """Each device core runs its tasks in creation order: an edge between consecutive tasks
        of every (chip, cluster, core). Returns the number of edges added."""
        import networkx as nx
        dev = {self.roles[r] for r in ("gemm", "simd", "xdma", "dm")}
        groups = {}
        for t in self.dfg.node_list:
            if t.assigned_core_id in dev:
                groups.setdefault((t.assigned_chiplet_id, t.assigned_cluster_id,
                                   t.assigned_core_id), []).append(t)
        added = 0
        for ts in groups.values():
            ts.sort(key=lambda t: t.node_id)
            for a, b in zip(ts, ts[1:]):
                if not self.dfg.has_edge(a, b):
                    self.dfg.bingo_add_edge(a, b)
                    added += 1
        if not nx.is_directed_acyclic_graph(self.dfg):
            cyc = nx.find_cycle(self.dfg)
            raise ValueError(f"the program order closes a cycle: "
                             f"{[(a.node_name, b.node_name) for a, b in cyc[:6]]}")
        print(f"[{self.TAG}] program order: {added} core-order edges")
        return added

    def _load(self, g, i, j, deps=()):
        """Weight chunk j of cluster g into the next slab: a ring load in iteration 0, a copy from
        its slot later. Returns (load task, slab index, slab)."""
        k, c, dm = self.chip(g), self.c(g), self.roles["dm"]
        ls = self.ls[g]
        off, nb = self.chunk_off[g][j]
        si, slab, war = ls.take(nb)
        if i == 0:
            rings = ls.rings
            sl, fl, rl, seq, _, _ = rings.take_ex(c, src=self.img[g].address + off, nbytes=nb)
            prev = [rings.last[c]] if rings.last.get(c) is not None else []
            ld = self.node(k, c, dm, f"Ld_c{j}_i0_g{g}", "__snax_bingo_kernel_idma_ring_load",
                           SnaxBingoKernelIdmaRingLoadArgs(fl, seq, sl, slab, nb, rl),
                           list(dict.fromkeys(war + prev + list(deps))))
            rings.last[c] = ld
            ld.exec_chunk = rings.last_take
            self.slot[(g, j)] = (sl, ld)
        else:
            sl, first = self.slot[(g, j)]
            ld = self.node(k, c, dm, f"Ld_c{j}_i{i}_g{g}", "__snax_bingo_kernel_idma_1d_copy",
                           SnaxBingoKernelIdma1dCopyArgs(sl, slab, nb),
                           list(dict.fromkeys(war + [first] + list(deps))))
        return ld, si, slab

    # ---- the stages of one iteration --------------------------------------------------------
    def stage_x_put(self, i, after):
        """The root multicasts X_i into every chip's L3 (bcast_l1). Returns the put."""
        T, R = self.TAG, self.ROOT
        return self.node(R, 0, self.roles["dm"], f"x_bput_i{i}", "__snax_bingo_kernel_idma_put_flagged",
                         SnaxBingoKernelIdmaPutFlaggedArgs(
                             BingoMemSymbol(f"{T}_xsrc", offset=i * self.XB),
                             BingoMemSymbol(f"{T}_xland", offset=i * self.XB), self.XB,
                             BingoMemSymbol(f"{T}_xflag", offset=i * 64), value=1, bcast=True,
                             also_here=True), list(after))

    def stage_x_put_block(self, b, after):
        """params x_blocks: the root multicasts row block b of X (single pass, iteration 0)."""
        T, R, nb = self.TAG, self.ROOT, self.XB // self.x_blocks
        return self.node(R, 0, self.roles["dm"], f"x_bput_i0b{b}", "__snax_bingo_kernel_idma_put_flagged",
                         SnaxBingoKernelIdmaPutFlaggedArgs(
                             BingoMemSymbol(f"{T}_xsrc", offset=b * nb),
                             BingoMemSymbol(f"{T}_xland", offset=b * nb), nb,
                             BingoMemSymbol(f"{T}_xflag", offset=b * 64), value=1, bcast=True,
                             also_here=True), list(after))

    def x_wait_block(self, g, b, put, after=()):
        """params x_blocks: row block b of X into cluster g's L1, behind its landed flag."""
        T, dm, nb = self.TAG, self.roles["dm"], self.XB // self.x_blocks
        k, c = self.chip(g), self.c(g)
        nd = self.node(k, c, dm, f"x_wait_i0b{b}_g{g}", "__snax_bingo_kernel_idma_fetch_flagged",
                       SnaxBingoKernelIdmaFetchFlaggedArgs(
                           BingoMemSymbol(f"{T}_xland", offset=b * nb), self.xbuf[g][0].view(b * nb), nb,
                           BingoMemSymbol(f"{T}_xflag", offset=b * 64), 1, 1),
                       list(after) + ([put] if k == self.ROOT else []))
        if k != self.ROOT:
            self.dfg.bingo_add_order_edge(put, nd)
        return nd

    def stage_x(self, i, put, after):
        """X_i into every cluster's L1 buffer (i mod buffers): behind its landed flag, or pulled
        from the root. {g: the task after which it is there}."""
        T, dm = self.TAG, self.roles["dm"]
        out = {}
        for g in range(self.G):
            k, c = self.chip(g), self.c(g)
            xb = self.xbuf[g][i % len(self.xbuf[g])]
            deps = list(after.get(g, []))
            if self.bcast_impl == "bcast_l1":
                nd = self.node(k, c, dm, f"x_wait_i{i}_g{g}", "__snax_bingo_kernel_idma_fetch_flagged",
                               SnaxBingoKernelIdmaFetchFlaggedArgs(
                                   BingoMemSymbol(f"{T}_xland", offset=i * self.XB), xb, self.XB,
                                   BingoMemSymbol(f"{T}_xflag", offset=i * 64), 1, 1),
                               deps + ([put] if k == self.ROOT else []))
                if k != self.ROOT:
                    self.dfg.bingo_add_order_edge(put, nd)
            else:
                nd = self.node(k, c, dm, f"x_pull_i{i}_g{g}", "__snax_bingo_kernel_idma_1d_copy",
                               SnaxBingoKernelIdma1dCopyArgs(
                                   BingoMemSymbol(f"{T}_xsrc", offset=i * self.XB, chip_id=self.ROOT),
                                   xb, self.XB), deps)
            out[g] = nd
        return out

    def stage_gu(self, i, x):
        """Every cluster's gate|up chunks, each dequantised and through SwiGLU. Returns {g: the
        last SwiGLU} and {g: the last GEMM}."""
        gemm, simd = self.roles["gemm"], self.roles["simd"]
        last_sw, last_gemm = {}, {}
        for g, (_, n) in self.cols.items():
            k, c = self.chip(g), self.c(g)
            nb = n // BLK
            xb = self.xbuf[g][i % len(self.xbuf[g])]
            soff = 0
            sw = None
            xblk = self.x_blocks > 1 and i == 0
            xws = [x[g]]
            for j, (b0, nb_) in enumerate(self.chunks[g]):
                ld, si, slab = self._load(g, i, j, [x[g]] if j == 0 else ())
                if self.dn_load_early and j == len(self.chunks[g]) - 1:
                    # params dn_load_early (dn_split cols): the first down chunk's load right behind the last gate|up
                    # chunk's, ahead of the A8 copies in the DM core's program (those wait for the quantise)
                    self.dn_first_load[g] = self._load(g, i, j + 1)
                if xblk and j == 0:
                    # the other X blocks behind the first weight load in the DM core's program
                    xws += [self.x_wait_block(g, bb, self._xputs[bb]) for bb in range(1, self.x_blocks)]
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
                    gms = [self.node(k, c, gemm, f"gu_c{j}_i{i}b{bb}_g{g}", "__snax_bingo_kernel_gemm_full",
                                     SnaxBingoKernelGemmFullArgs(
                                         input_A_addr=xb.view(bb * mb * (self.XB // self.MT)),
                                         output_D_addr=gb.view(bb * mb * 2 * half), M=mb, **kargs),
                                     [ld, xws[-1] if last else xws[bb]] + ([war] if war is not None else []))
                           for bb in range(nsp)]
                    gm_of = lambda m, gms=gms, mb=mb: gms[m // mb]
                    gm = gms[-1]
                else:
                    gm = self.node(k, c, gemm, f"gu_c{j}_i{i}_g{g}", "__snax_bingo_kernel_gemm_full",
                                   SnaxBingoKernelGemmFullArgs(input_A_addr=xb, output_D_addr=gb, M=self.MT, **kargs),
                                   [ld, xws[-1]] + ([war] if war is not None else []))
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
                    war_a = [self.d2a[(g, i - 1)]] if self.war_a16 and (g, i - 1) in self.d2a else []
                    sw = self.node(k, c, simd, f"swi_c{j}m{m}_i{i}_g{g}",
                                   "__snax_bingo_kernel_simd_swiglu_f16_f16",
                                   SnaxBingoKernelSimdSwigluF16F16Args(gate, up, out, 1,
                                                                       nb_ * 256),
                                   [dq] + (war_a if j == 0 and m == 0 else []))
                    if j == 0 and m == 0:
                        self.swi0[(g, i)] = sw
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
        war_q = [self.quant[(g, i - 1)]] if self.war_a16 and (g, i - 1) in self.quant else []
        return self.node(k, c, self.roles["xdma"], f"d2a_i{i}b{b}_g{g}", "__snax_bingo_kernel_xdma_6d",
                         d_to_a_args(self.d.mesh, mb, nb, n // 4, self.a16d[g].view(m0 * nb * TILE),
                                     self.a16a[g].view(m0 * 16 * n * 2)), list(after) + war_q)

    def _post_quant(self, g, i, b, rs):
        """Post block b of A16a quantised into A8 (buffer i mod 2). dn_split cols: one quantise per row tile into
        this cluster's run of the chip's A8, each run then copied into the other cluster's A8 (self.ex_push)."""
        k, c, n = self.chip(g), self.c(g), self.cols[g][1]
        mb, m0 = self.MT // self.post_blocks, b * (self.MT // self.post_blocks)
        war = self.a8_reader[g][i % 2]
        if self.dn_cols:
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
                self.ex_push.setdefault(g, []).append(
                    self.node(k, c, self.roles["dm"], f"a8_push_m{m}_g{g}", "__snax_bingo_kernel_idma_1d_copy",
                              SnaxBingoKernelIdma1dCopyArgs(self.a8[g][0].view(off), self.a8[other][0].view(off), 16 * n),
                              [q]))
            return q
        return self.node(k, c, self.roles["simd"], f"quant_i{i}b{b}_g{g}", "__snax_bingo_kernel_simd_fp16_to_int8",
                         SnaxBingoKernelSimdFp16ToInt8Args(self.a16a[g].view(m0 * 16 * n * 2), self.a8[g][i % 2].view(m0 * 16 * n),
                                                           beats=simd_beats(mb * 16 * n * 2), rows=1,
                                                           inv_scale_f32bits=self.d.inv_a),
                         [rs] + ([war] if war is not None else []))

    def stage_post(self, i, sw):
        """A16 reshaped to the A layout and quantised into A8 (buffer i mod 2). {g: quant}."""
        xdma, simd = self.roles["xdma"], self.roles["simd"]
        out = {}
        for g, (_, n) in self.cols.items():
            k, c = self.chip(g), self.c(g)
            war_q = [self.quant[(g, i - 1)]] if self.war_a16 and (g, i - 1) in self.quant else []
            if (g, i) in self.post_made:
                out[g] = self.quant[(g, i)]
                continue
            if self.post_blocks > 1:
                mb = self.MT // self.post_blocks
                q = None
                for b in range(self.post_blocks):
                    rs = self._post_d2a(g, i, b, [self.last_swi[(g, i)][b * mb + mb - 1]])
                    q = self._post_quant(g, i, b, rs)
                self.d2a[(g, i)] = rs
                out[g] = q
                self.quant[(g, i)] = q
                continue
            rs = self.node(k, c, xdma, f"d2a_i{i}_g{g}", "__snax_bingo_kernel_xdma_6d",
                           d_to_a_args(self.d.mesh, self.MT, n // BLK, n // 4, self.a16d[g],
                                       self.a16a[g]), [sw[g]] + war_q)
            self.d2a[(g, i)] = rs
            if self.war_a16 and (g, i + 1) in self.swi0:
                self.dfg.bingo_add_edge(rs, self.swi0[(g, i + 1)])
            a8 = self.a8[g][i % 2]
            war = self.a8_reader[g][i % 2]
            out[g] = self.node(k, c, simd, f"quant_i{i}_g{g}", "__snax_bingo_kernel_simd_fp16_to_int8",
                               SnaxBingoKernelSimdFp16ToInt8Args(self.a16a[g], a8,
                                                                 beats=simd_beats(self.ROWS * n * 2),
                                                                 rows=1,
                                                                 inv_scale_f32bits=self.d.inv_a),
                               [rs] + ([war] if war is not None else []))
            self.quant[(g, i)] = out[g]
        return out

    def stage_dn(self, i, qa, war_loc):
        """Every cluster's down chunks into its partial (its symmetric buffer). war_loc {g: what
        must finish before the partial is overwritten}. Returns {g: the last down GEMM}."""
        gemm = self.roles["gemm"]
        out = {}
        for g, (_, n) in self.cols.items():
            k, c = self.chip(g), self.c(g)
            nG = len(self.chunks[g])
            a8 = self.a8[g][i % 2]
            gm = None
            for j in range(self.NDN):
                ld, si, slab = self._load(g, i, nG + j)
                y = self.sym[(k, c)].view(j * self.MT * (self.DNC // BLK) * TILE)
                gm = self.node(k, c, gemm, f"dn_c{j}_i{i}_g{g}", "__snax_bingo_kernel_gemm_full",
                               SnaxBingoKernelGemmFullArgs(
                                   input_A_addr=a8, input_B_addr=slab, input_C_addr=0,
                                   output_D_addr=y, M=self.MT, K=n // 4, N=self.DNC // BLK,
                                   array_shape_idx=0, transpose_A=0, transpose_B=0, accumPrevC=0,
                                   int32tofp16_enable=1, int4_b_enable=self.int4_b, d_shift=self.d.k_dn),
                               [ld, qa[g]] + list(war_loc.get(g, [])))
                self.ls[g].release(si, gm)
            self.a8_reader[g][i % 2] = gm
            out[g] = gm
        return out

    def stage_fold(self, i, dn, war_p1):
        """Each chip's clusters added on its cluster 0, into its loc. {chip: loc_add},
        {g of cluster 1: its push}."""
        dm, simd = self.roles["dm"], self.roles["simd"]
        loc, push = {}, {}
        for k in self.chips:
            g0, g1 = self.gs(k)
            push[g1] = self.node(k, 1, dm, f"part_push_i{i}", "__snax_bingo_kernel_idma_1d_copy",
                                 SnaxBingoKernelIdma1dCopyArgs(self.sym[(k, 1)], self.p1[k], self.YB),
                                 [dn[g1]] + list(war_p1.get(k, [])))
            loc[k] = self.node(k, 0, simd, f"loc_add_i{i}", "__snax_bingo_kernel_simd_stream_elementwise",
                               SnaxBingoKernelSimdAddF16Args(self.sym[(k, 0)], self.p1[k],
                                                             self.sym[(k, 0)], self.YB // 64, 32),
                               [dn[g0], push[g1]])
        return loc, push

    def stage_reduce(self, i, loc, store=True):
        """The chips' loc folded into the root's p1 (Y_raw_i), stored to the root's L3. Returns
        (the store, {chip: the task after which its loc is free}); with store=False the store is
        left to the caller (stage_store) and the first item is the reduce's last task."""
        R, dm, simd, xdma = self.ROOT, self.roles["dm"], self.roles["simd"], self.roles["xdma"]
        others = self.order[1:]
        p1r = self.p1[R]
        p1x = f"((uint64_t)({p1r.get_c_var_name()}))"
        if self.reduce_impl == "chain":
            chain = [cm.remote(k, self.sx(0, self.OFF_LOC)) for k in others] + [cm.remote(R, p1x)]
            ga = SnaxBingoKernelXdmaChainGatherArgs(
                local_src=cm.remote(R, self.sx(0, self.OFF_LOC)), chain=chain, size=self.YB,
                junction=str(self.jct["Elementwise"]), jct_csr0=EW_F16_ADD,
                hops=[(k, 0) for k in others] + [(R, 0)])
            ga.holds = [p1r]
            red = self.node(R, 0, xdma, f"y_chain_i{i}", "__snax_bingo_kernel_xdma_chain_gather", ga,
                            list(dict.fromkeys(loc[k] for k in self.chips)))
            free = {k: red for k in self.chips}
        else:
            ns = self.YB // self.SL
            pulls, adds, free = [], [], {}
            n = 0
            for ik, k in enumerate(others):
                for s in range(ns):
                    slot = self.slots.view((n % 2) * self.SL)
                    cp = self.node(R, 0, dm, f"y_pull_i{i}_k{k:02x}_s{s}", "__snax_bingo_kernel_idma_1d_copy",
                                   SnaxBingoKernelIdma1dCopyArgs(
                                       cm.remote(k, f"({self.sx(0, self.OFF_LOC)} + {s * self.SL})"),
                                       slot, self.SL),
                                   [loc[k]] + pulls[-1:] + ([adds[n - 2]] if n >= 2 else []))
                    pulls.append(cp)
                    a_in = self.sym[(R, 0)].view(s * self.SL) if ik == 0 else p1r.view(s * self.SL)
                    add = self.node(R, 0, simd, f"y_add_i{i}_k{k:02x}_s{s}",
                                    "__snax_bingo_kernel_simd_stream_elementwise",
                                    SnaxBingoKernelSimdAddF16Args(a_in, slot, p1r.view(s * self.SL),
                                                                  self.SL // 64, 32),
                                    [cp, loc[R]] + adds[-1:])
                    adds.append(add)
                    n += 1
                free[k] = pulls[-1]
            red = adds[-1]
            free[R] = red
        if not store:
            return red, free
        return self.stage_store(i, red), free

    def stage_store(self, i, red):
        """Y_raw_i from the root's p1 into its L3, after the reduce red. Returns the store."""
        R, p1r = self.ROOT, self.p1[self.ROOT]
        return self.node(R, 0, self.roles["dm"], f"y_store_i{i}", "__snax_bingo_kernel_idma_1d_copy",
                         SnaxBingoKernelIdma1dCopyArgs(p1r, BingoMemSymbol(f"{self.TAG}_y", offset=i * self.YB),
                                                       self.YB), [red])

    # ---- the prefill ----------------------------------------------------------------------
    def build(self):
        dm = self.roles["dm"]
        R = self.ROOT
        self.slot = {}
        self.gu_count = {g: 0 for g in range(self.G)}
        self.gb_reader = {g: [None, None] for g in range(self.G)}
        self.a8_reader = {g: [None, None] for g in range(self.G)}
        # A16 (D layout) and A16a (A layout) are one buffer each: the pipelined schedule runs an
        # iteration's SwiGLU, reshape and quantise after the next iteration's gate|up has begun,
        # so each buffer's next writer waits for its last reader (the BSP schedule has every
        # earlier stage done at the barrier). d2a[(g, i)], quant[(g, i)]: the tasks; swi0[(g, i)]:
        # the first SwiGLU of iteration i
        self.war_a16 = self.sync == "edge" and not self.p.get("no_war_a16", False)
        self.d2a, self.quant, self.swi0, self.last_swi = {}, {}, {}, {}
        self.post_made = set()                  # (g, i) whose post blocks the split last chunk made
        self.ex_push = {}                       # dn_split cols: {g: its A8 runs copied to the other cluster}
        self.q_tile = {}                        # dn_split cols: {g: {row tile: its quantise}}
        self.dn_load_early = bool(self.p.get("dn_load_early", False)) and self.dn_cols
        # params relay_on_simd (single pass): dep-tag relays of xDMA tasks (the gathers) go to the SIMD cores first,
        # idle in the down phase, not behind the DM cores' blocking ring loads or the GEMM cores' long GEMMs
        if self.single and self.p.get("relay_on_simd", False):
            self.dfg.relay_prefer_cores = {self.roles["xdma"]: [self.roles["simd"]]}
        # params relay_per_gather (single pass): each dep-tag relay of an xDMA task serves that one task, so a
        # gather waits for its own slice's GEMMs only, not for the next gather's too (one relay held both)
        if self.single and self.p.get("relay_per_gather", False):
            self.dfg.relay_one_consumer_cores = [self.roles["xdma"]]
        # params gather_relay_only (single pass): a dep-tag cell of the xDMA tasks (the gathers) that is over budget
        # relays its leftover edges instead of ordering them, so no down GEMM waits for an earlier gather
        if self.single and self.p.get("gather_relay_only", False):
            self.dfg.relay_not_order_cores = [self.roles["xdma"]]
        self.dn_first_load = {}                 # dn_load_early: {g: the first down chunk's load}
        # params sym_zero (single pass): false skips zeroing the symmetric buffers before the start; every
        # slot is written whole by its down GEMMs (or push) before it is read, and an unzeroed buffer is live
        # only from then on, so the static L1 plan can lay it over X
        self.sym_zero = bool(self.p.get("sym_zero", True))
        if not self.sym_zero and not self.single:
            raise ValueError("params sym_zero false: single_pass only")
        # the dequantisation factors into L1, before the start
        pre = {k: [] for k in self.chips}
        for g in range(self.G):
            k, c = self.chip(g), self.c(g)
            h, nb = self.sdup[g]
            pre[k].append(self.node(k, c, dm, f"sgu_load_g{g}", "__snax_bingo_kernel_idma_1d_copy",
                                    SnaxBingoKernelIdma1dCopyArgs(h.address, self.sbuf[g], nb)))
        go = self.setup(extra=pre)
        after_x = {g: [go[self.chip(g)]] for g in range(self.G)}
        if self.single:
            if self.NI != 1 or self.sync != "edge" or self.bcast_impl != "bcast_l1" or self.reduce_impl != "chain":
                raise ValueError("params mapping=single_pass needs rows = tokens, sync edge, bcast bcast_l1, reduce chain")
            (self._build_single_cols if self.dn_cols else self._build_single)(go, after_x)
        elif self.sync == "edge":
            self._build_pipelined(go, after_x)
        else:
            self._build_bsp(go, after_x)

    def _build_pipelined(self, go, after_x):
        R = self.ROOT
        puts = []
        x_war = {}
        sw, post, stores = {}, {}, []
        free = {}
        p1_war = {}
        # params x_ahead: each core runs its tasks in creation order, and the root's DM core both
        # puts X and stores Y. Without it, the put of X_(i+2) and the root cluster's loads for
        # iteration i+2 are created after the reduce and store of iteration i, so iteration i+2
        # cannot start before iteration i is stored. With it, X_(i+1) is put right after
        # iteration i's gate|up loads (every iteration has its own landing area and flag in L3, so
        # the put waits for nothing else), and the store of iteration i is created after
        # iteration i+2's gate|up, before iteration i+1's fold (whose root push overwrites p1).
        x_ahead = bool(self.p.get("x_ahead", False)) and self.bcast_impl == "bcast_l1"
        # params store_slot (with x_ahead): gu = the store of iteration i right after gate|up of
        # i+2; dn = later, inside iteration i+1's down stage, after its weight loads and before
        # its fold, so the root's down loads do not queue behind the store either
        store_slot = str(self.p.get("store_slot", "gu"))
        if store_slot not in ("gu", "dn"):
            raise ValueError(f"params store_slot={store_slot!r}: gu or dn")
        # params post_ahead: the reshape and quantise of iteration i are created at the end of its
        # gate|up, not at the start of its down stage, so on the root's xDMA core the reshape of i
        # runs before the chain gather of i-1 (the arrays run gate|up of i before down of i-1)
        post_ahead = bool(self.p.get("post_ahead", False))
        pending = []

        def store(i, red):
            st = self.stage_store(i, red)
            stores.append(st)
            p1_war[R] = [st]

        def gu(i):
            # X_i is put with its own iteration, so the root's DM core loads iteration i's
            # weights before it puts X_(i+1)
            if self.bcast_impl == "bcast_l1" and len(puts) == i:
                puts.append(self.stage_x_put(i, [go[R]] + puts[-1:]))
            after = {g: list(after_x[g]) + ([x_war[(g, i)]] if (g, i) in x_war else [])
                     for g in range(self.G)}
            x = self.stage_x(i, puts[i] if puts else None, after)
            s, lg = self.stage_gu(i, x)
            sw[i] = s
            nxb = len(self.xbuf[0])
            for g in range(self.G):
                x_war[(g, i + nxb)] = lg[g]
            if x_ahead and i + 1 < self.NI:
                puts.append(self.stage_x_put(i + 1, [go[R]] + puts[-1:]))
            if post_ahead:
                post[i] = self.stage_post(i, sw[i])

        def dn(i, upto="full"):
            if i not in post:
                post[i] = self.stage_post(i, sw[i])
            war = {}
            for g in range(self.G):
                k = self.chip(g)
                if self.c(g) == 0 and k in free:
                    war[g] = [free[k]]
                elif self.c(g) == 1 and (k, "push") in free:
                    war[g] = [free[(k, "push")]]
            d = self.stage_dn(i, post[i], war)
            if x_ahead and store_slot == "dn":
                while pending:              # the store of iteration i-1, behind iteration i's down loads
                    store(*pending.pop(0))
            loc, push = self.stage_fold(i, d, p1_war)
            if upto == "fold":
                return loc
            st, fr = self.stage_reduce(i, loc, store=not x_ahead)
            free.clear()
            free.update(fr)
            for k in self.chips:
                g0, g1 = self.gs(k)
                free[(k, "push")] = push[g1]
                if k != R:
                    p1_war[k] = [loc[k]]
            if x_ahead:
                pending.append((i, st))
            else:
                stores.append(st)
                p1_war[R] = [st]

        # params stop: how far the program runs, for bisecting a failure (one iteration only):
        # a8 = X_0 landed, gate|up, SwiGLU, reshape and quantise of iteration 0, checked in every
        # cluster; fold = + the down GEMMs and the chip fold, every cluster's partial and every
        # chip's sum checked; full = the whole prefill, Y checked per iteration
        stop = str(self.p.get("stop", "full"))
        if stop not in ("a8", "fold", "full"):
            raise ValueError(f"params stop={stop!r}: a8, fold or full")
        if stop != "full" and self.NI != 1:
            raise ValueError(f"params stop={stop!r} needs one iteration (tokens = rows), not {self.NI}")

        # the arrays' order: GU0 GU1 DN0 GU2 DN1 ... GU(n-1) DN(n-2) DN(n-1)
        gu(0)
        if stop == "a8":
            post[0] = self.stage_post(0, sw[0])
            self.end_window(list(post[0].values()))
            self._stage_checks(stop)
            return
        if stop == "fold":
            loc = dn(0, "fold")
            self.end_window(list(loc.values()))
            self._stage_checks(stop)
            return
        for i in range(1, self.NI):
            gu(i)
            while pending and store_slot == "gu":   # x_ahead: the store of iteration i-2, behind gate|up of i
                store(*pending.pop(0))
            dn(i - 1)
        while pending and store_slot == "gu":
            store(*pending.pop(0))
        dn(self.NI - 1)
        while pending:
            store(*pending.pop(0))
        self.end_window([stores[-1]])
        self._checks()

    def _build_single(self, go, after_x):
        """params mapping single_pass: every token in one pass. Each weight chunk is used for all rows as
        it streams in (M = rows / 16 row tiles a GEMM; the weights are loaded into L1 once), and the down
        output is reduced in slices of DNC columns: slice j's GEMMs write slot j mod 2 of each cluster's
        symmetric buffer, cluster 1 pushes its slot to cluster 0's fold buffer, cluster 0 adds it, the
        root's ChainGather sums slot j mod 2 of every chip into its own fold buffer slot and the root
        stores it to L3. Slice j + 2 reuses the slots once every reader of slice j is done. The weight
        load of slice j + 1 is created before slice j's push, so a DM core prefetches across the fold."""
        R, dm, simd, xdma, gemm = self.ROOT, self.roles["dm"], self.roles["simd"], self.roles["xdma"], self.roles["gemm"]
        if self.x_blocks > 1:
            self._xputs = []
            for b in range(self.x_blocks):
                self._xputs.append(self.stage_x_put_block(b, [go[R]] + self._xputs[-1:]))
            x = {g: self.x_wait_block(g, 0, self._xputs[0], after_x[g]) for g in range(self.G)}
        else:
            put = self.stage_x_put(0, [go[R]])
            x = self.stage_x(0, put, after_x)
        sw, _ = self.stage_gu(0, x)
        post = self.stage_post(0, sw)
        SL, others = self.SLICE, self.order[1:]
        p1r = self.p1[R]
        p1x = f"((uint64_t)({p1r.get_c_var_name()}))"
        sym_reader, p1_reader, stores = {}, {}, []

        def load(j):
            return {g: self._load(g, 0, len(self.chunks[g]) + j) for g in range(self.G)}

        # params store_lag (single pass): create slice j's store this many slices later, behind the next
        # weight loads in the root DM core's program, so those loads do not wait for the gather of slice j
        lag = int(self.p.get("store_lag", 0))
        pending, stored_slice = [], {}

        def store(j, red, off, size, yo, sfx):
            st = self.node(R, 0, dm, f"y_store_j{j}{sfx}", "__snax_bingo_kernel_idma_1d_copy",
                           SnaxBingoKernelIdma1dCopyArgs(p1r.view(off), BingoMemSymbol(f"{self.TAG}_y", offset=yo), size),
                           [red])
            stores.append(st)
            key = (R, j % 2)
            p1_reader[key] = (p1_reader[key] if stored_slice.get(key) == j else []) + [st]
            stored_slice[key] = j

        lds = load(0)
        yoff = 0                                     # where the next output part sits in Y (L3)
        ngather = 0                                  # gathers so far (params chain_alt: odd ones reversed)
        # params push_after_gather (single pass): each cluster-1 push waits for the previous gather, so no push
        # into a cluster 0 runs while a gather passes through that cluster's xDMA (the two deadlock)
        pag, last_red = bool(self.p.get("push_after_gather", False)), None
        for j in range(self.NDN):
            s = j % 2
            # the parts of this slice: the whole slice, or (the last one) tail_split narrower GEMMs on its weight chunk
            nparts = self.tail_split if j == self.NDN - 1 else 1
            nt = self.DNC // BLK // nparts           # column tiles a part
            psz = SL // nparts
            parts = []
            for t in range(nparts):
                off = s * SL + t * psz
                dn = {}
                for g, (_, n) in self.cols.items():
                    k, c = self.chip(g), self.c(g)
                    ld, si, slab = lds[g]
                    dn[g] = self.node(k, c, gemm, f"dn_c{j}_i0{'' if nparts == 1 else f'p{t}'}_g{g}",
                                      "__snax_bingo_kernel_gemm_full",
                                      SnaxBingoKernelGemmFullArgs(
                                          input_A_addr=self.a8[g][0],
                                          input_B_addr=slab.view(t * nt * 8 * n) if t else slab, input_C_addr=0,
                                          output_D_addr=self.sym[(k, c)].view(off), M=self.MT, K=n // 4,
                                          N=nt, array_shape_idx=0, transpose_A=0, transpose_B=0,
                                          accumPrevC=0, int32tofp16_enable=1, int4_b_enable=self.int4_b, d_shift=self.d.k_dn),
                                      [ld, post[g]] + sym_reader.get((k, c, s), []))
                    if t == nparts - 1:
                        self.ls[g].release(si, dn[g])
                parts.append((t, off, dn))
            if j + 1 < self.NDN:
                lds = load(j + 1)
            while len(pending) > lag or (pending and j == self.NDN - 1 and lag == 0):
                store(*pending.pop(0))
            for t, off, dn in parts:
                sfx = "" if nparts == 1 else f"p{t}"
                loc = {}
                for k in self.chips:
                    g0, g1 = self.gs(k)
                    push = self.node(k, 1, dm, f"part_push_j{j}{sfx}", "__snax_bingo_kernel_idma_1d_copy",
                                     SnaxBingoKernelIdma1dCopyArgs(self.sym[(k, 1)].view(off), self.p1[k].view(off), psz),
                                     [dn[g1]] + p1_reader.get((k, s), []) + ([last_red] if pag and last_red else []))
                    loc[k] = self.node(k, 0, simd, f"loc_add_j{j}{sfx}", "__snax_bingo_kernel_simd_stream_elementwise",
                                       SnaxBingoKernelSimdAddF16Args(self.sym[(k, 0)].view(off), self.p1[k].view(off),
                                                                     self.sym[(k, 0)].view(off), psz // 64, 32),
                                       [dn[g0], push])
                    # the readers of a slot: every part's (a split slice has several)
                    sym_reader[(k, 1, s)] = (sym_reader[(k, 1, s)] if t else []) + [push]
                ordr = others[::-1] if self.chain_alt and ngather % 2 else others
                ngather += 1
                chain = [cm.remote(k, self.sx(0, self.OFF_LOC + off)) for k in ordr] + [cm.remote(R, f"({p1x} + {off})")]
                ga = SnaxBingoKernelXdmaChainGatherArgs(
                    local_src=cm.remote(R, self.sx(0, self.OFF_LOC + off)), chain=chain, size=psz,
                    junction=str(self.jct["Elementwise"]), jct_csr0=EW_F16_ADD,
                    hops=[(k, 0) for k in ordr] + [(R, 0)])
                ga.holds = [p1r]
                red = self.node(R, 0, xdma, f"y_chain_j{j}{sfx}", "__snax_bingo_kernel_xdma_chain_gather", ga,
                                list(loc.values()))
                last_red = red
                for k in self.chips:
                    sym_reader[(k, 0, s)] = (sym_reader[(k, 0, s)] if t else []) + [red]
                    if k != R:
                        p1_reader[(k, s)] = (p1_reader[(k, s)] if t else []) + [loc[k]]
                pending.append((j, red, off, psz, yoff, sfx))
                yoff += psz
                if lag == 0:
                    store(*pending.pop(0))
        while pending:
            store(*pending.pop(0))
        self.end_window([stores[-1]])
        self._checks()

    def dn_slices(self, c):
        """dn_split cols: the down slices (DNC output columns each) cluster c computes: its half of the outputs."""
        h = self.NDN // self.NC
        return range(c * h, (c + 1) * h)

    def _dn_out_split(self, balance):
        """dn_split cols: {cluster: [(first output column, columns)]}, NDN / 2 slices each. Default: the two
        halves of the outputs in DNC-column slices. params dn_balance: cluster 0's share of the outputs moves (a
        multiple of 16, the same on every chip, since a gather takes the same slice from every chip) to where the
        busiest cluster's array steps (gate|up of its own columns + down of its chip's) are fewest, and each share
        is cut into slices of whole 16-column blocks as even as possible. params dn_head [us per cluster]: each
        cluster's array starts this much later than the window does (cluster 1's first weight chunk lands after
        cluster 0's, both pushed by one memory-chip engine), counted as array steps at the (16, 4, 16) array's
        one step per 2.002 ns when the busiest cluster is found."""
        h = self.NDN // self.NC
        if not balance:
            return {c: [(j * self.DNC, self.DNC) for j in self.dn_slices(c)] for c in range(self.NC)}
        mt, kd = self.MT, self.D // 4
        dh = self.p.get("dn_head", [0] * self.NC)
        if len(dh) != self.NC:
            raise ValueError(f"params dn_head={dh}: one start offset (us) per cluster")
        head = [int(round(float(x) / 0.002002)) for x in dh]

        def worst(S):
            w = 0
            for k in self.chips:
                kn = self.kcols[k][1]
                for c, g in enumerate(self.gs(k)):
                    out = S if c == 0 else self.D - S
                    w = max(w, head[c] + mt * kd * 2 * self.cols[g][1] // BLK + mt * (kn // 4) * out // BLK)
            return w
        u = self.dn_unit                            # whole column tiles, or (w4_paired) whole pairs of them
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
              f"{worst(S):,} array steps with its start offset (even halves: {worst(self.D // 2):,})")
        return {0: cut(0, S), 1: cut(S, self.D - S)}

    def _build_single_cols(self, go, after_x):
        """params dn_split cols: the single pass with each chip's down projection split by output columns. Each
        cluster's down GEMMs read the chip's whole A8 (K = the chip's columns) for its half of the outputs, one slice
        at a time into slot (slice mod 2) of its symmetric buffer, so the slot holds the chip's sum and the root's
        cluster c gathers slot s of every chip's cluster c straight into its fold buffer and stores it. The gathers
        run one at a time, alternating between the two clusters' xDMAs; the first waits for every A8 copy, so no copy
        into a cluster runs while a gather passes through it."""
        R, dm, xdma, gemm = self.ROOT, self.roles["dm"], self.roles["xdma"], self.roles["gemm"]
        if self.x_blocks > 1:
            self._xputs = []
            for b in range(self.x_blocks):
                self._xputs.append(self.stage_x_put_block(b, [go[R]] + self._xputs[-1:]))
            x = {g: self.x_wait_block(g, 0, self._xputs[0], after_x[g]) for g in range(self.G)}
        else:
            put = self.stage_x_put(0, [go[R]])
            x = self.stage_x(0, put, after_x)
        sw, _ = self.stage_gu(0, x)
        post = self.stage_post(0, sw)
        SL, others, h = self.SLICE, self.order[1:], self.NDN // self.NC
        p1x = {c: f"((uint64_t)({self.p1c[c].get_c_var_name()}))" for c in range(self.NC)}
        ex_last = {g: self.ex_push[g][-1] for g in range(self.G)}
        sym_reader, p1_reader, stores, pending = {}, {}, [], []
        lag = int(self.p.get("store_lag", 0))
        # params split_first_dn (dn_split cols): each cluster's first down slice as one GEMM per post block, each behind
        # its block's quantise and A8 copy only, so the end of the post-processing runs under down GEMMs
        split0 = bool(self.p.get("split_first_dn", False))

        def load(jl):
            if jl == 0 and self.dn_first_load:
                return dict(self.dn_first_load)
            return {g: self._load(g, 0, len(self.chunks[g]) + jl) for g in range(self.G)}

        def store(c, j, red, s):
            o0, w = self.dn_sl[c][j - c * h]
            st = self.node(R, c, dm, f"y_store_j{j}", "__snax_bingo_kernel_idma_1d_copy",
                           SnaxBingoKernelIdma1dCopyArgs(self.p1c[c].view(s * SL), BingoMemSymbol(f"{self.TAG}_y", offset=o0 * self.ROWS * 2),
                                                         self.ROWS * w * 2),
                           [red])
            stores.append(st)
            p1_reader[(c, s)] = [st]

        # params gather_pairs (dn_split cols): the two clusters' gathers of a slice pair run at the same time (on
        # disjoint xDMAs); pairs still run one after the other
        pairs = bool(self.p.get("gather_pairs", False))
        # params first_gather_waits_copies (default true): the first gather also waits for every A8 copy. False: only
        # the copies into cluster c precede cluster c's gathers, through cluster c's own down GEMMs
        copies = list(ex_last.values()) if bool(self.p.get("first_gather_waits_copies", True)) else []
        lds = load(0)
        last_red, prev_pair = None, []
        for jl in range(h):
            s = jl % 2
            dn = {}
            for g in range(self.G):
                k, c = self.chip(g), self.c(g)
                ld, si, slab = lds[g]
                other, kn = self.gs(k)[1 - c], self.kcols[k][1]
                nt = self.dn_sl[c][jl][1] // BLK        # this slice's column tiles
                kargs = dict(input_B_addr=slab, input_C_addr=0, K=kn // 4, N=nt, array_shape_idx=0,
                             transpose_A=0, transpose_B=0, accumPrevC=0, int32tofp16_enable=1, int4_b_enable=self.int4_b,
                             d_shift=self.d.k_dn)
                if split0 and jl == 0:          # one GEMM per post block, behind that block's A8 rows only
                    mb = self.MT // self.post_blocks
                    for b in range(self.post_blocks):
                        m1 = (b + 1) * mb - 1   # the block's last row tile: its quantise and copy come last
                        dn[g] = self.node(k, c, gemm, f"dn_c{c * h}_i0b{b}_g{g}", "__snax_bingo_kernel_gemm_full",
                                          SnaxBingoKernelGemmFullArgs(
                                              input_A_addr=self.a8[g][0].view(b * mb * 16 * kn),
                                              output_D_addr=self.sym[(k, c)].view(s * SL + b * mb * nt * TILE),
                                              M=mb, **kargs),
                                          [ld, self.q_tile[g][m1], self.ex_push[other][m1]])
                else:
                    dn[g] = self.node(k, c, gemm, f"dn_c{c * h + jl}_i0_g{g}", "__snax_bingo_kernel_gemm_full",
                                      SnaxBingoKernelGemmFullArgs(input_A_addr=self.a8[g][0],
                                                                  output_D_addr=self.sym[(k, c)].view(s * SL),
                                                                  M=self.MT, **kargs),
                                      [ld, post[g], ex_last[other]] + sym_reader.get((k, c, s), []))
                self.ls[g].release(si, dn[g])
            if jl + 1 < h:
                lds = load(jl + 1)
            while len(pending) > lag * self.NC:
                store(*pending.pop(0))
            for c in range(self.NC):
                j = c * h + jl
                # params chain_alt: cluster 1's gathers run the ring the other way, so each gather's data follows the
                # previous gather's finish back along the links instead of meeting it
                ordr = others[::-1] if self.chain_alt and c == 1 else others
                chain = [cm.remote(k, self.sx(c, s * SL)) for k in ordr] + [cm.remote(R, f"({p1x[c]} + {s * SL})")]
                ga = SnaxBingoKernelXdmaChainGatherArgs(
                    local_src=cm.remote(R, self.sx(c, s * SL)), chain=chain, size=self.ROWS * self.dn_sl[c][jl][1] * 2,
                    junction=str(self.jct["Elementwise"]), jct_csr0=EW_F16_ADD,
                    hops=[(k, c) for k in ordr] + [(R, c)])
                ga.holds = [self.p1c[c]]
                if pairs:
                    first = list(copies) if jl == 0 else list(prev_pair)
                else:
                    first = list(copies) if last_red is None else [last_red]
                red = self.node(R, c, xdma, f"y_chain_j{j}", "__snax_bingo_kernel_xdma_chain_gather", ga,
                                [dn[g] for g in range(self.G) if self.c(g) == c] + p1_reader.get((c, s), []) + first)
                last_red = red
                pair_now = (pair_now if c else []) + [red]
                for k in self.chips:
                    sym_reader[(k, c, s)] = [red]
                pending.append((c, j, red, s))
            prev_pair = pair_now
        while pending:
            store(*pending.pop(0))
        self.end_window(stores[-self.NC:])          # the last store of each of the root's clusters
        self._checks()

    def _build_bsp(self, go, after_x):
        R = self.ROOT
        stores = []
        bar = go
        for i in range(self.NI):
            put = None
            if self.bcast_impl == "bcast_l1":
                put = self.stage_x_put(i, [bar[R]])
            x = self.stage_x(i, put, {g: [bar[self.chip(g)]] for g in range(self.G)})
            sw, _ = self.stage_gu(i, x)
            post = self.stage_post(i, sw)
            d = self.stage_dn(i, post, {})
            loc, _ = self.stage_fold(i, d, {})
            b2 = self.barrier({k: [loc[k]] for k in self.chips})
            st, _ = self.stage_reduce(i, {k: b2[R] for k in self.chips})
            stores.append(st)
            bar = self.barrier({k: [st] if k == R else [b2[k]] for k in self.chips})
        self.end_window([stores[-1]])
        self._checks()

    def _checks(self):
        """Y_raw sampled on the root's L3 (the first and the last 2 KiB of every iteration), and
        the root's own A8 of the last iteration (2 KiB)."""
        lay, T, R = self.lay, self.TAG, self.ROOT
        y = self.gold["y_raw"]
        checks = []
        n = 2048
        for i in range(self.NI):
            rows = y[i * self.ROWS:(i + 1) * self.ROWS]
            img = np.concatenate([lay.to_d(rows[:, c0:c0 + w], self.d.mesh)
                                  for c0, w in self.y_slices()]).astype(np.float16).view(np.uint8)
            for name, off in ((f"y_i{i}_a", 0), (f"y_i{i}_b", self.YB - n)):
                checks.append((name, BingoMemSymbol(f"{T}_y", offset=i * self.YB + off, chip_id=R),
                               img[off:off + n], n))
        g = self.gs(R)[0]
        c0, ncol = self.kcols[R] if self.dn_cols else self.cols[g]   # dn_split cols: the chip's A8, both clusters
        i = self.NI - 1
        a8 = self.d.a8[i * self.ROWS:(i + 1) * self.ROWS, c0:c0 + ncol]
        got = self.stash("a8_root", R, 0, self.a8[g][i % 2], n, [])
        checks.append(("a8_root", got, lay.to_a(a8, self.d.mesh)[:n].view(np.uint8), n))
        if self.p.get("check_x"):
            checks.insert(0, self._check_x(n))
        self.run_checks(checks)

    def y_slices(self):
        """[(first column, columns)] of Y in the order its D-layout slices sit in L3: DNC columns each, the last
        one in tail_split parts in the single pass."""
        if self.dn_cols:
            return [sl for c in range(self.NC) for sl in self.dn_sl[c]]
        sl = [(j * self.DNC, self.DNC) for j in range(self.NDN)]
        if self.single and self.tail_split > 1:
            c0, w = sl.pop()
            h = w // self.tail_split
            sl += [(c0 + t * h, h) for t in range(self.tail_split)]
        return sl

    def _y_img(self, y, i):
        """Iteration i's rows of a (T, D) FP16 array, in the D layout of the Y buffers, as bytes."""
        rows = y[i * self.ROWS:(i + 1) * self.ROWS]
        return np.concatenate([self.lay.to_d(rows[:, j * self.DNC:(j + 1) * self.DNC], self.d.mesh)
                               for j in range(self.NDN)]).astype(np.float16).view(np.uint8)

    def _gather_check(self, name, items, n):
        """items [(chip, cluster, L1 source, golden)]: after the end anchor each cluster's DM core
        copies the first n bytes of its source into its own slice of one L3 symbol on the root, and
        the root's host compares the whole symbol in one check. The first bad offset // n names the
        item. Returns the check for run_checks."""
        T, R = self.TAG, self.ROOT
        sym = f"{T}_chk_{name}"
        self.st.put_zeros(sym, "uint8_t", len(items) * n)
        for j, (k, c, src, _) in enumerate(items):
            self.check_copies.append(self.node(
                k, c, self.roles["dm"], f"chk_{name}_{j}", "__snax_bingo_kernel_idma_1d_copy",
                SnaxBingoKernelIdma1dCopyArgs(src, BingoMemSymbol(sym, offset=j * n, chip_id=R), n),
                [self.end]))
        gold = np.concatenate([np.ascontiguousarray(g).view(np.uint8).reshape(-1)[:n] for *_, g in items])
        return (name, BingoMemSymbol(sym, chip_id=R), gold, len(items) * n)

    def _check_x(self, n):
        """params check_x: X_0's first n bytes in every cluster's L1 after the window (needs one
        iteration, or x_buffers above 1)."""
        x0 = self.lay.to_a(self.d.hq[:self.ROWS])
        return self._gather_check("x_all", [(self.chip(g), self.c(g), self.xbuf[g][0], x0)
                                            for g in range(self.G)], n)

    def _stage_checks(self, stop):
        """The checks of a stopped run (params stop). check_bytes bytes a slice: a host compare
        costs ~30 ns a byte."""
        n = int(self.p.get("check_bytes", 64))
        lay, mesh = self.lay, self.d.mesh
        checks = [self._check_x(n)]
        if stop == "a8":
            items = []
            for g in range(self.G):
                c0, ncol = self.cols[g]
                a8 = self.d.a8[:self.ROWS, c0:c0 + ncol]
                items.append((self.chip(g), self.c(g), self.a8[g][0], lay.to_a(a8, mesh)))
            checks.append(self._gather_check("a8_all", items, n))
        else:
            # every chip's cluster-1 partial (as the chip's cluster 0 received it) and every chip's
            # sum, cluster 0's partial plus cluster 1's
            items = []
            for k in self.chips:
                g0, g1 = self.gs(k)
                items.append((k, 1, self.sym[(k, 1)], self._y_img(self.gold["part"][g1], 0)))
                items.append((k, 0, self.sym[(k, 0)], self._y_img(self.gold["loc"][k], 0)))
            checks.append(self._gather_check("part_loc", items, n))
        self.run_checks(checks)

    def check_hazards(self):
        """The pipelined schedule reuses L1 buffers across iterations while an earlier iteration's
        tail is still running on other cores; nothing orders those but the edges the generator
        adds. For every cluster and every pair of consecutive uses of a buffer slot, the last
        reader of the earlier use must reach the first writer of the later one in the DFG
        (the cores run in the order of their edges): a missing edge is a data race that no
        golden of one iteration shows. Returns the violations."""
        import re
        import networkx as nx
        by, byk = {}, {}
        for t in self.dfg.node_list:
            m = re.match(r"^(.*)_k([0-9a-f]{2})_cl(\d)$", t.node_name)
            if m:
                by[m.group(1)] = t                                   # per-cluster tasks: names carry g
                byk[(m.group(1), int(m.group(2), 16), int(m.group(3)))] = t   # per-chip tasks
        bad = []

        def ordered(a, b, what):
            if a is None or b is None:
                bad.append(f"{what}: a task is missing")
            elif not nx.has_path(self.dfg, a, b):
                bad.append(f"{what}: {a.node_name} is not before {b.node_name}")

        nxb = len(self.xbuf[0])
        for g in range(self.G):
            nG = len(self.chunks[g])
            for i in range(self.NI):
                get = lambda base: by.get(base)
                # A16 (D layout): read by the reshape of i, rewritten by the first SwiGLU of i + 1
                if i + 1 < self.NI:
                    ordered(get(f"d2a_i{i}_g{g}"), get(f"swi_c0m0_i{i + 1}_g{g}"), f"a16d g{g} i{i}->i{i + 1}")
                    # A16a: read by the quantise of i, rewritten by the reshape of i + 1
                    ordered(get(f"quant_i{i}_g{g}"), get(f"d2a_i{i + 1}_g{g}"), f"a16a g{g} i{i}->i{i + 1}")
                # A8[i % 2]: read by the last down GEMM of i, rewritten by the quantise of i + 2
                if i + 2 < self.NI:
                    ordered(get(f"dn_c{self.NDN - 1}_i{i}_g{g}"), get(f"quant_i{i + 2}_g{g}"), f"a8 g{g} i{i}->i{i + 2}")
                # X buffer i % nxb: read by the last gate|up GEMM of i, rewritten by the fetch of i + nxb
                if i + nxb < self.NI:
                    ordered(get(f"gu_c{nG - 1}_i{i}_g{g}"), get(f"x_wait_i{i + nxb}_g{g}") or get(f"x_pull_i{i + nxb}_g{g}"), f"x g{g} i{i}->i{i + nxb}")
                # the partial (the cluster's symmetric buffer): read by the chip fold of i,
                # rewritten by the first down GEMM of i + 1
                if i + 1 < self.NI:
                    k = self.chip(g)
                    rd = byk.get((f"part_push_i{i}", k, 1)) if self.c(g) == 1 else byk.get((f"loc_add_i{i}", k, 0))
                    ordered(rd, get(f"dn_c0_i{i + 1}_g{g}"), f"partial g{g} i{i}->i{i + 1}")
        if self.single and self.dn_cols:
            # cluster c's slot s: its gather of a slice before the down GEMM two slices later; the root's fold slot:
            # the store of a slice before the gather two slices later; every A8 copy before the first gather
            R, h = self.ROOT, self.NDN // self.NC
            for c in range(self.NC):
                for jl in range(h - 2):
                    j = c * h + jl
                    for g in range(self.G):
                        if self.c(g) == c:
                            ordered(byk.get((f"y_chain_j{j}", R, c)), by.get(f"dn_c{j + 2}_i0_g{g}"),
                                    f"output slot g{g} j{j}->j{j + 2}")
                    ordered(byk.get((f"y_store_j{j}", R, c)), byk.get((f"y_chain_j{j + 2}", R, c)),
                            f"root fold slot c{c} j{j}->j{j + 2}")
            for g in range(self.G):          # every copy into cluster c before cluster c's first gather
                c = self.c(g)
                first = byk.get((f"y_chain_j{c * h}", R, c))
                other = self.gs(self.chip(g))[1 - c]
                for m in range(self.MT):
                    ordered(by.get(f"a8_push_m{m}_g{other}"), first, f"A8 copy m{m} into g{g} before its first gather")
        elif self.single:
            # slot j mod 2: every reader of slice j before the first writer of slice j + 2
            R = self.ROOT
            for j in range(self.NDN - 2):
                for g in range(self.G):
                    k, c = self.chip(g), self.c(g)
                    rd = byk.get((f"part_push_j{j}", k, 1)) if c == 1 else byk.get((f"y_chain_j{j}", R, 0))
                    # a split slice (tail_split) names its parts p0, p1, ...: the first writer is part 0
                    w = by.get(f"dn_c{j + 2}_i0_g{g}") or by.get(f"dn_c{j + 2}_i0p0_g{g}")
                    ordered(rd, w, f"output slot g{g} j{j}->j{j + 2}")
                for k in self.chips:
                    w = byk.get((f"part_push_j{j + 2}", k, 1)) or byk.get((f"part_push_j{j + 2}p0", k, 1))
                    ordered(byk.get((f"loc_add_j{j}", k, 0)), w, f"fold slot chip {k:02x} j{j}->j{j + 2}")
                    if k == R:
                        ordered(byk.get((f"y_store_j{j}", R, 0)), w, f"root fold slot j{j}->j{j + 2} (store)")
        return bad

    def reuse_gemm_config(self):
        """params gemm_reuse: a GEMM whose configuration equals the previous GEMM's on the same core (every
        field but the four operand addresses) becomes gemm_minimal, which rewrites the addresses and starts.
        Each core runs its tasks in creation order (chain_program_order), so the previous GEMM there is the
        one created before it. Returns the number converted."""
        full = "__snax_bingo_kernel_gemm_full"
        addr = ("input_A_addr", "input_B_addr", "input_C_addr", "output_D_addr")
        prev, n = {}, 0
        for t in sorted(self.dfg.node_list, key=lambda t: t.node_id):
            if t.kernel_name != full:
                continue
            key = (t.assigned_chiplet_id, t.assigned_cluster_id, t.assigned_core_id)
            sig = tuple(sorted((k, v) for k, v in vars(t.kernel_args).items() if k not in addr and not k.startswith("_")))
            if prev.get(key) == sig:
                a = t.kernel_args
                t.kernel_name = "__snax_bingo_kernel_gemm_minimal"
                t.kernel_args = SnaxBingoKernelGemmMinimalArgs(a.input_A_addr, a.input_B_addr, a.input_C_addr, a.output_D_addr)
                n += 1
            prev[key] = sig
        print(f"[{self.TAG}] gemm_reuse: {n} GEMMs run as gemm_minimal")
        return n

    def finish(self, app_name, x_prefixes=("x_wait_", "x_pull_")):
        self.chain_program_order()
        if self.p.get("gemm_reuse"):
            self.reuse_gemm_config()
        if self.sync == "edge":
            bad = self.check_hazards()
            if bad:
                raise ValueError(f"{len(bad)} buffer hazards in the pipelined schedule, e.g. " + "; ".join(bad[:4]))
            print(f"[{self.TAG}] buffer hazards: none (A16, A16a, A8, X, partial; every cluster)")
        # one host function per chip: a single one of every chip's block compiles too slowly
        self.dfg.split_chip_functions = True
        super().finish(app_name, x_prefixes)


def run():
    ap = argparse.ArgumentParser(description="ffn prefill workload generator")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--data_h", default=None)
    ap.add_argument("--output_offload_file_name", default="offload_bingo_hw.h")
    ap.add_argument("-c", "--cfg", required=True)
    ap.add_argument("--hwcfg", required=True)
    ap.add_argument("--platformcfg", required=True)
    ap.add_argument("--dsv2_dir", required=True,
                    help="snax_cluster's target/snitch_cluster/sw/apps/dsv2 (the numerics models)")
    args = ap.parse_args()
    with open(args.cfg) as f:
        p = dict(hjson.load(f))
    plat = parse_platform_cfg(args.platformcfg)
    hw = load_cluster_cfg(args.hwcfg)
    S = PBuild(args, p, plat, hw)
    S.build()
    S.finish(f"{p.get('model', 'dsv2-lite-l0')} FFN prefill, {S.T} tokens in {S.NI} x {S.ROWS}, "
             f"{len(S.chips)} chips x {S.NC} clusters, bcast={S.bcast_impl} "
             f"reduce={S.reduce_impl} sync={S.sync}")
