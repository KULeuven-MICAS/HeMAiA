#!/usr/bin/env python3
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""DeepSeek-V2-Lite's layer-1 MLA attention, stages 6 to 8 of the 256-token prefill on sixteen chiplets alone: W_UV,
W_O and the reduce of the 16 heads' partials into y.

Stage 5's result (every chip's o~: its head, both chunks, both clusters, every query tile, INT8 in the A layout of a
16-row tile) is in each chip's memory chip's HBM and is copied into L1 before the window, with every cluster's
constants. The ring streams per chunk W_UV and W_O's two blocks of 1,024 columns. Per chunk and cluster: W_UV a query
tile at a time on o~, each tile's o_h relaid D -> A and quantised behind its own GEMM (its dequantisation folded in);
W_O's two blocks into the two halves of the cluster's partial y in its exchange buffer. Per W_O block and root
cluster one ChainGather along the fold cycle adds the 16 heads' partials into the root (both clusters' chains at once,
every chain behind the landing of every chip's ring's last chunk: no memory-chip push transits a hop chip while two
chains run), each block's sum stored into the root's L3 right after its chain; the root's chains and stores of chunk
i - 1 are made at the head of chunk i, ahead of its W_UV, so its in-order cores run them under chunk i.

          before the window     const_ld, ot_pre_i<i>q<q>                  DM: constants and o~ into L1
          S0_start
          every cluster, chunk i:
            Ld_wuv_i<i>(_push), guv<q>_i<i>, d2au<q>_i<i>, o8<q>_i<i>      W_UV, o_h D -> A and quantised, by tile
            Ld_wo<r>_i<i>(_push), gwo<r>_i<i>                              W_O block r into the exchange buffer's half r
          the root, chunk i, block h: y_chain_c<c>_i<i>_h<h>, y_store_c<c>_i<i>_h<h>   the ChainGather, the store
          S1_end (behind every store) -> fp_report, the checks

Checks: the head and the tail of each W_O block's rows of y, of every chunk and cluster, in the root's L3 (the
partials folded in the fold order).

params.hjson: the platform, the tokens, the chunk and key-tile sizes, the weight chunk and ring sizes, the weights'
width."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from stage_common import (BR, D_V, EW_F16_ADD, HIDDEN, Q_NOPE, TILE16, BingoMemSymbol, StageBase,  # noqa: E402
                          app_name)
from bingo_kernel_args import SnaxBingoKernelIdma1dCopyArgs, SnaxBingoKernelXdmaChainGatherArgs  # noqa: E402
from driver import run_driver  # noqa: E402
from libs.comm.nest import d_to_a_args  # noqa: E402

import chiplet_mc as cm  # noqa: E402

CHK = 2048                                  # bytes a checked window


class Stage678(StageBase):
    """Stages 6 to 8 alone (the module docstring)."""
    STAGE = "678"
    X_HBM = False
    KVA_IN_RING = 0
    PER_CHUNK = StageBase.PER_CHUNK_TAIL           # a chunk's W_UV and W_O x 2

    def __init__(self, args, p, plat, hw, shared=None):
        super().__init__(args, p, plat, hw, shared=shared)
        self._wo_blk, self._o8_tiles = {}, {}
        self._last_chain, self._pair_prev = None, None
        self._chains = []                           # every reduce chain, in creation order

    def stage_inputs(self):
        """Every chip's o~ (its head, both chunks, both clusters, every query tile: the reference's o~, INT8 in the A
        layout of a 16-row tile) in its memory chip's HBM."""
        T = self.TAG
        self.h_ot = {}
        for k in self.chips:
            h = self.head[k]
            img = np.concatenate([self.xt_ref(h, r0)[0].reshape(-1)
                                  for i in range(self.NCH) for c in range(self.NC) for q in range(self.NQ)
                                  for r0 in (self.r0(i, c) + q * BR,)])
            m = self.mem[k]
            self.h_ot[k] = self.st.put_hbm(f"{T}_ot_k{k:02x}", img.view(np.int8), mem_chip=(m >> 4, m & 15))

    def xt_ref(self, h, r0):
        """The sweep's xt for the query tile at row r0 of head h: (bytes, length): o~ INT8 in the A layout of one
        16-row tile."""
        d = self.d
        return d.layout.to_a(d.head(h)["ot8"][r0:r0 + BR], d.mesh).astype(np.int8).view(np.uint8), D_V * BR

    # ---- the ring: W_UV and W_O only --------------------------------------------------------------------------------
    def chunk_index(self, i, part):
        """The ring index of token chunk i's first chunk (its W_UV)."""
        return i * self.PER_CHUNK_TAIL

    def _src(self, k, n):
        """(HBM address, bytes) of chip k's ring chunk n: W_UV, W_O block 0, W_O block 1 in turn."""
        nm, off, size = self.w_off[k][4 + n % self.PER_CHUNK_TAIL]
        return self.h_w[k].address + off, size

    def load(self, cl, n, name, deps=(), slab=None, dst=None):
        """A cluster's first ring load behind its chip's start, so no DM core spins on a push before the start."""
        if cl not in self._first_ld:
            self._first_ld.add(cl)
            deps = list(deps) + [self.go[self.chip(cl)]]
        return super().load(cl, n, name, deps, slab, dst)

    # ---- the window ---------------------------------------------------------------------------------------------
    def build(self):
        self.last_ld = {}
        for ls in self.ls:
            ls.strict = True
        self.setup(extra=self.stage678_preload())
        self.relays()
        self.x_puts = []
        res = dict(post={}, parts={}, red={}, stores={}, xch_free={})
        for i in range(self.NCH):
            self.chunk_tasks(i, res)
        self._reduce_chunk(self.NCH - 1, res["red"], res["stores"], res["xch_free"])
        self.end_window([t for v in res["stores"].values() for t in (v if isinstance(v, (list, tuple)) else [v])])
        self.run_checks(self.y_checks())

    def stage678_preload(self):
        """Before the window, per chip: every cluster's constants, then its o~ tiles of both chunks into L1."""
        T, ob = self.TAG, D_V * BR
        pre = {k: [] for k in self.chips}
        self.ot_buf, self.ot_ld = {}, {}
        for k in self.chips:
            for c in reversed(range(self.NC)):
                cl = self.chips.index(k) * self.NC + c
                pre[k] = [self.dm_copy(cl, "const_ld", self.h_c[k].address, self.consts[cl], self.C_BYTES, pre[k])]
                for i in range(self.NCH):
                    for q in range(self.NQ):
                        buf = self.cx_chip(k).at(c).l1(f"{T}_ot{q}_i{i}", ob)
                        off = ((i * self.NC + c) * self.NQ + q) * ob
                        ld = self.dm_copy(cl, f"ot_pre_i{i}q{q}", self.h_ot[k].address + off, buf, ob, pre[k])
                        self.ot_buf[(cl, i, q)], self.ot_ld[(cl, i, q)] = buf, ld
                        pre[k] = [ld]
        return pre

    def chunk_tasks(self, i, res):
        """Chunk i: per cluster its o~ (loaded before the window); on the root, chunk i - 1's chains and stores; then
        W_UV (stage_post_a) and W_O (stage_post_b) of every cluster."""
        R = self.ROOT
        red, stores, xch_free = res["red"], res["stores"], res["xch_free"]
        sweeps = {}
        for cl in range(self.N_CL):
            k, c = self.chip(cl), self.local(cl)
            sweeps[cl] = dict(xt=[self.ot_buf[(cl, i, q)] for q in range(self.NQ)],
                              tr=[self.ot_ld[(cl, i, q)] for q in range(self.NQ)])
            if i > 0 and k == R:
                w = {c: [stores[(i - 2, c)]] if (i - 2, c) in stores else []}
                self._block_chains(i - 1, [c], w, None, red, stores, xch_free)
        # everything of chunk i before its W_O, then W_O, which waits for the chunk before's chains (the exchange
        # buffer)
        post = {cl: self.stage_post_a(cl, i, sweeps[cl]) for cl in range(self.N_CL)}
        for cl in range(self.N_CL):
            k, c = self.chip(cl), self.local(cl)
            res["post"][cl] = dict(xt_rd=post[cl][0])
            res["parts"][(k, c, i)] = self.stage_post_b(cl, i, post[cl][1], xch_free.get((k, c), []))

    # ---- W_UV and W_O -------------------------------------------------------------------------------------------
    def stage_post_a(self, cl, i, sweep):
        """o~ -> W_UV -> o of chunk i (INT8, A layout), a query tile at a time: each tile's W_UV GEMM, then its o_h
        relaid D -> A and quantised (the dequantisation folded in), the tiles in turn. Returns (the tasks that read the
        sweep's xt, the task that made o)."""
        d, b = self.d, self.bufs(cl, i)
        mesh = d.mesh
        ld, si, slab = self.load(cl, self.chunk_index(i, "tail"), f"Ld_wuv_i{i}")
        gu = [self.gmm(cl, f"guv{q}_i{i}", sweep["xt"][q], slab, b["uv"].view(q * (Q_NOPE // 16) * TILE16), 1,
                       D_V // 4, Q_NOPE // 16, d.k_uv, [ld, sweep["tr"][q]]) for q in range(self.NQ)]
        self.ls[cl].release(si, gu)                 # every query tile's GEMM reads the slab
        suv = self.consts[cl].view(self.C_UV)
        qq, qs = None, []
        for q in range(self.NQ):
            d2q = self.xd(cl, f"d2au{q}_i{i}", d_to_a_args(mesh, 1, Q_NOPE // 16, Q_NOPE // 4,
                                                         b["uv"].view(q * (Q_NOPE // 16) * TILE16),
                                                         b["a16u"].view(q * 16 * Q_NOPE * 2)), [gu[q]])
            qq = self.mul_quant(cl, f"o8{q}_i{i}", b["a16u"].view(q * 16 * Q_NOPE * 2), suv,
                                b["o8"].view(q * 16 * Q_NOPE), 16 * Q_NOPE, self.fbits(d.s_o.inv), [d2q] + qs[-1:])
            qs.append(qq)
        self._o8_tiles[(cl, i)] = qs
        return list(gu), qq

    def stage_post_b(self, cl, i, qq, xch_after):
        """W_O of chunk i: two blocks of 1024 columns into the two halves of the cluster's partial, in its exchange
        buffer, which `xch_after` (the previous chunk's reduce) has finished reading. Returns the GEMMs."""
        d, b = self.d, self.bufs(cl, i)
        k, c = self.chip(cl), self.local(cl)
        mt = self.ROWS // 16
        half = self.YB // 2
        out = []
        for r in range(2):
            ld, si, slab = self.load(cl, self.chunk_index(i, "tail") + 1 + r, f"Ld_wo{r}_i{i}")
            gos = [self.gmm(cl, f"gwo{r}_i{i}", b["o8"].view(0), slab, self.xch[(k, c)].view(r * half), mt,
                            Q_NOPE // 4, 64, d.k_wo, [ld, qq] + list(xch_after))]
            self.ls[cl].release(si, gos)                # every row pair's GEMM reads the slab
            self._wo_blk[(k, c, i, r)] = gos
            out += gos
        return out

    # ---- the reduce ---------------------------------------------------------------------------------------------
    def stage_reduce(self, i, parts, war, gate, half, src_off, clusters):
        """Cluster index c's partials of W_O block `half` of chunk i into the root's cluster c, one ChainGather along
        the fold cycle. parts {(k, c): [tasks after which the partial is there]}, war {c: [tasks the destination waits
        for]}, gate {c: [tasks the reduce starts behind]}. Returns {c: the chain}. Every chain waits for the landing of
        every chip's ring's last chunk (after_pushes_landed); the last chunk's two clusters' chains run at the same
        time, after the chains before them."""
        R = self.ROOT
        others = self.y_order(R)[1:]
        out, before = {}, {}
        cs = list(clusters)
        for c in cs:
            before[c] = [t for k in self.chips for t in parts[(k, c)]] + list((gate or {}).get(c, []))
        last_pair = i == self.NCH - 1 and len(cs) == self.NC
        for c in cs:
            dstb = self.dst(i, c, half)
            ga = SnaxBingoKernelXdmaChainGatherArgs.fold_into(
                cm.remote, R, c, others, src=self.xch_addr(c, src_off),
                dst=f"((uint64_t)({dstb.get_c_var_name()}) + 0)", size=self.YB // 2,
                junction=self.jct["Elementwise"], jct_csr0=EW_F16_ADD, holds=[dstb])
            serial = []
            if last_pair:
                if self._pair_prev is None:
                    self._pair_prev = [self._last_chain] if self._last_chain is not None else []
                serial = list(self._pair_prev)
            out[c] = self.chain_gather(R, c, f"y_chain_c{c}_i{i}_h{half}", ga,
                                       before[c] + list(war.get(c, [])) + serial)
            self.after_pushes_landed(self, out[c])
            self._last_chain = out[c]
            self._chains.append(out[c])
        if last_pair:
            self._pair_prev = [out[c] for c in cs]
        return out

    def stage_store(self, i, red, half, clusters):
        """The root's cluster c stores W_O block `half` of chunk i's rows of y to its L3."""
        R, dm = self.ROOT, self.roles["dm"]
        st = {}
        for c in clusters:
            nb, off = self.YB // 2, half * self.YB // 2
            st[c] = self.node(R, c, dm, f"y_store_c{c}_i{i}_h{half}", "__snax_bingo_kernel_idma_1d_copy",
                              SnaxBingoKernelIdma1dCopyArgs(self.dst(i, c, half), self.y_sym(i, c, off), nb),
                              list(red[c]) if isinstance(red[c], (list, tuple)) else [red[c]])
        return st

    def _block_chains(self, i, cs, war, gate, red, stores, xch_free):
        """Root clusters cs's chains of chunk i, one a W_O block, each block's sum in its own half-size buffer, stored
        right after its chain (the second block's store after the first's)."""
        cs = list(cs)
        half = self.YB // 2
        rh = []
        for r in range(2):
            parts = {(k, c): list(self._wo_blk[(k, c, i, r)]) for k in self.chips for c in cs}
            rh.append(self.stage_reduce(i, parts, war if r == 0 else {}, gate if r == 0 else None, half=r,
                                        src_off=r * half, clusters=cs))
            st = self.stage_store(i, rh[r], half=r, clusters=cs)
            if r == 0:
                h0 = st
            else:
                # the second half's store after the first's, so it stands for the chunk's stores
                for c in cs:
                    self.edge(h0[c], st[c])
        for c in cs:
            red[(i, c)] = rh[1][c]
            stores[(i, c)] = st[c]
            for k in self.chips:
                xch_free[(k, c)] = [rh[0][c], rh[1][c]]

    def _reduce_chunk(self, i, red, stores, xch_free):
        """Chunk i's chains and stores, the destinations behind the chunk before's stores."""
        war = {c: [stores[(i - 1, c)]] if (i - 1, c) in stores else [] for c in range(self.NC)}
        self._block_chains(i, range(self.NC), war, None, red, stores, xch_free)

    # ---- where y lands ------------------------------------------------------------------------------------------
    def y_order(self, r):
        """The fold cycle from collector r: r, then the cycle on from it."""
        j = self.order.index(r)
        return self.order[j:] + self.order[:j]

    def dst(self, i, c, half):
        """The root's cluster c landing buffer of W_O block `half` of chunk i's reduce (a half-size buffer each)."""
        key = (i, c, half)
        if key not in self._dst:
            self._dst[key] = self.cx_chip(self.ROOT).at(c).l1(f"{self.TAG}_dst_i{i}h{half}", self.YB // 2)
        return self._dst[key]

    def y_sym(self, i, c, off=0, chip_id=None):
        """Where the root stores cluster c's rows of y of chunk i: over x (cluster 0 in chunk i's landing area, cluster
        1 in its source)."""
        name, base = (f"{self.TAG}_xland" if c == 0 else f"{self.TAG}_xsrc"), i * self.XB
        if chip_id is None:
            return BingoMemSymbol(name, offset=base + off)
        return BingoMemSymbol(name, offset=base + off, chip_id=chip_id)

    # ---- checks -------------------------------------------------------------------------------------------------
    def y_checks(self):
        """The head and the tail of each W_O block's rows of y, of every chunk and cluster, in the root's L3: the heads'
        partials folded in the root's fold order."""
        lay, d, R, n, half = self.lay, self.d, self.ROOT, CHK, HIDDEN // 2
        checks = []
        for i in range(self.NCH):
            for c in range(self.NC):
                y = d.reduce(self.fold_heads, self.r0(i, c), self.ROWS)
                img = np.concatenate([lay.to_d(y[:, :half], d.mesh), lay.to_d(y[:, half:], d.mesh)]
                                     ).astype(np.float16).view(np.uint8)
                wins = ((f"y_i{i}c{c}_h0a", 0), (f"y_i{i}c{c}_h0b", self.YB // 2 - n),
                        (f"y_i{i}c{c}_h1a", self.YB // 2), (f"y_i{i}c{c}_h1b", self.YB - n))
                for nm, off in wins:
                    checks.append((nm, self.y_sym(i, c, off, chip_id=R), img[off:off + n], n))
        return checks


if __name__ == "__main__":
    run_driver("fa prefill stages 6 to 8 workload generator", Stage678, app_name)
