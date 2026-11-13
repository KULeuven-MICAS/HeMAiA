#!/usr/bin/env python3
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""DeepSeek-V2-Lite's layer-1 MLA attention, stage 4 of the 256-token prefill on sixteen chiplets alone: the queries
of every head.

x is already in every chip's L3 (stage 0's result) and every cluster's dequantisation constants are loaded before the
window. Per chunk and cluster (its half of the chunk's rows): its rows of x into L1, q = x W_q (three GEMMs of 64
columns over the ring's W_q chunks; the third is q_pe), q_nope relaid D -> A and quantised (its dequantisation folded
in), the absorb q~ = q_nope W_UK a query tile at a time (each tile's q~ relaid and quantised behind its own GEMM),
q_pe dequantised and rotated (RoPE in the D layout: one simd_rope over [x | cos | xswap | sin]), relaid and quantised:
the query tiles Q8 [16, 576] (q~ then q_pe). The ring streams per chunk W_q x 3 and W_UK only.

          before the window     const_ld                       DM: every cluster's constants into its L1
          S0_start
          every cluster, chunk i:
            x_copy_i<i>                                        DM: its rows of x, L3 -> L1
            Ld_wq<r>_i<i>(_push), gq<r>_i<i>                   the W_q chunk, q's 64-column block r
            d2an<r><m>_i<i>, qn_i<i>, dq2_i<i>                 q_nope D -> A and quantised; q_pe dequantised
            Ld_wuk_i<i>, gab<q>_i<i>, d2aq<q>_i<i>             W_UK, the absorb a query tile at a time, q~ D -> A
            rope_ld_i<i>, swap<h>_i<i>, rope_i<i>, d2ap_i<i>   q_pe's RoPE tables, the pair swap, RoPE, D -> A
            q8p<q>_i<i>, q8a<q>_i<i>                           SIMD: Q8, q_pe's quantisations then q~'s
          P_fin<chip> -> S1_end -> fp_report, the checks

Checks: the first and the last query tile of both clusters of the chip in the middle of the fold order, the last chunk.

params.hjson: the platform, the tokens, the chunk and key-tile sizes, the weight chunk and ring sizes, the weights'
width."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from stage_common import (BR, D_QK, HIDDEN, KV_RANK, Q_NOPE, ROPE, TILE16, BingoMemSymbol,  # noqa: E402
                          StageBase, app_name)
from bingo_kernel_args import (SnaxBingoKernelIdma1dCopyArgs, SnaxBingoKernelIdma2dCopyArgs,  # noqa: E402
                               SnaxBingoKernelSimdRopeArgs)
from driver import run_driver  # noqa: E402
from libs.comm.nest import d_to_a_args  # noqa: E402


class Stage4(StageBase):
    """Stage 4 alone (the module docstring)."""
    STAGE = 4
    X_HBM = False
    KVA_IN_RING = 0
    PER_CHUNK = StageBase.PER_CHUNK_HEAD           # a chunk's W_q x 3 and W_UK

    def build(self):
        self.last_ld = {}
        for ls in self.ls:
            ls.strict = True
        go = self.setup(extra=self.const_preload({k: [] for k in self.chips}))
        self.relays()
        self.x_puts = []
        last = {k: [] for k in self.chips}
        self.stage_queries(go, last)
        self.window_end(last)
        self.run_checks(self.q8_checks())

    def const_preload(self, pre):
        """Every cluster's dequantisation constants into its L1 (const_ld), one chain a chip ending on cluster 0 behind
        `pre` {chip: [tasks]}."""
        for k in self.chips:
            for c in reversed(range(self.NC)):
                cl = self.chips.index(k) * self.NC + c
                pre[k] = [self.dm_copy(cl, "const_ld", self.h_c[k].address, self.consts[cl], self.C_BYTES, pre[k])]
        return pre

    # ---- the queries --------------------------------------------------------------------------------------------
    def stage_queries(self, go, last):
        """Per chunk, every cluster's rows of x into L1 (stage_x) and its query projection (stage_proj); a chunk's rows
        land without waiting for the previous chunk's projection (each chunk has its own rows buffer)."""
        self.q8_of = {}
        for i in range(self.NCH):
            for cl in range(self.N_CL):
                k = self.chip(cl)
                x = self.stage_x(cl, i, [go[k]])
                q8 = self.stage_proj(cl, i, x)
                self.q8_of[(cl, i)] = q8
                last[k] += sorted({t for _, ts in q8.values() for t in ts}, key=lambda t: t.node_id)

    def stage_x(self, cl, i, deps):
        """Cluster cl's rows of chunk i into its L1, copied from its chip's own copy of x."""
        T, k, c = self.TAG, self.chip(cl), self.local(cl)
        half = self.XB // self.NC
        xb = self.bufs(cl, i)["xrows"]
        return self.node(k, c, self.roles["dm"], f"x_copy_i{i}", "__snax_bingo_kernel_idma_1d_copy",
                         SnaxBingoKernelIdma1dCopyArgs(BingoMemSymbol(f"{T}_xsrc", offset=i * self.XB + self.xh(i, c) * half),
                                                       xb, half), list(deps))

    def stage_proj(self, cl, i, x):
        """q, q~ and the rotated q_pe of chunk i into Q8, one query tile at a time (the module docstring). The in-order
        SIMD core's program: q_nope's quantisation before q_pe's dequantisation, so the absorb GEMM follows the q GEMMs
        without a gap; in the tail every query tile's q_pe quantisation before the q~ ones, while the xDMA lays out q~.
        Returns {q: (Q8 handle, [the tasks it follows])}."""
        d, b = self.d, self.bufs(cl, i)
        k, c = self.chip(cl), self.local(cl)
        rows, mt = self.ROWS, self.ROWS // 16
        mesh = d.mesh
        sq = [self.consts[cl].view(self.C_SQ + self.SQ_B * r) for r in range(3)]
        # ---- q = x W_q, three blocks of 64 columns; the third is q_pe --------------------------
        gq = []
        for r in range(3):
            ld, si, slab = self.load(cl, self.chunk_index(i, "head") + r, f"Ld_wq{r}_i{i}")
            gm = self.gmm(cl, f"gq{r}_i{i}", b["xrows"], slab, b["q"][r], mt, HIDDEN // 4, 4, d.k_x, [ld, x])
            self.ls[cl].release(si, gm)
            gq.append(gm)
        # ---- q_nope: D -> A (FP16) per row block and half, then its dequantisation and quantisation in one task ----
        d2a = []
        for r in range(2):
            for m in range(mt):
                d2a.append(self.xd(cl, f"d2an{r}{m}_i{i}", d_to_a_args(
                    mesh, 1, 4, 16, b["q"][r].view(m * 4 * TILE16), b["a16n"].view(m * 16 * Q_NOPE * 2 + r * 16 * 64 * 2)),
                    [gq[r]]))
        qn = self.mul_quant(cl, f"qn_i{i}", b["a16n"], sq[0], b["qn8"], rows * Q_NOPE, self.fbits(d.s_qn.inv), d2a)
        # q_pe's dequantisation after q_nope's quantisation on the SIMD core
        dq2 = self.simd_mul(cl, f"dq2_i{i}", b["q"][2], sq[2], b["q"][2], rows * 64, [gq[2]])
        # ---- absorb: q~ = q_nope W_UK, a query tile at a time (its A rows and D rows each a contiguous run) --------
        ldu, siu, slabu = self.load(cl, self.chunk_index(i, "head") + 3, f"Ld_wuk_i{i}")
        gat = [self.gmm(cl, f"gab{q}_i{i}", b["qn8"].view(q * 16 * Q_NOPE), slabu,
                        b["qt"].view(q * (KV_RANK // 16) * TILE16), 1, Q_NOPE // 4, KV_RANK // 16, d.k_uk,
                        [ldu, qn]) for q in range(self.NQ)]
        self.ls[cl].release(siu, gat)                   # every query tile's GEMM reads the slab
        sqt = self.consts[cl].view(self.C_QT)
        d2qt = [self.xd(cl, f"d2aq{q}_i{i}", d_to_a_args(mesh, 1, KV_RANK // 16, KV_RANK // 4,
                                                         b["qt"].view(q * (KV_RANK // 16) * TILE16),
                                                         b["a16q"].view(q * 16 * KV_RANK * 2)), [gat[q]])
                for q in range(self.NQ)]
        # ---- q_pe: RoPE in the D layout (one simd_rope over [x | cos | xswap | sin]), then D -> A and quantise -------
        rope_src = BingoMemSymbol(f"{self.TAG}_rope", offset=2 * (i * self.NC + c) * self.ROPE_B)
        R4 = b["rope4"]
        # cos into slot 1, sin_signed into slot 3: the two tables are adjacent in L3, a slot apart in the block
        ldr = self.node(k, c, self.roles["dm"], f"rope_ld_i{i}", "__snax_bingo_kernel_idma_2d_copy",
                        SnaxBingoKernelIdma2dCopyArgs(rope_src, R4.view(self.ROPE_B), self.ROPE_B, self.ROPE_B,
                                                      2 * self.ROPE_B, 2), [])
        # the rotary columns are in half order: each row tile's D layout is its four 16-column blocks, the swap is the
        # first two and the last two exchanged, one 2-D copy for each direction
        half, mtb = (ROPE // 32) * TILE16, (ROPE // 16) * TILE16
        sw = [self.node(k, c, self.roles["dm"], f"swap{h}_i{i}", "__snax_bingo_kernel_idma_2d_copy",
                        SnaxBingoKernelIdma2dCopyArgs(R4.view((1 - h) * half), R4.view(2 * self.ROPE_B + h * half),
                                                      half, mtb, mtb, mt), [dq2]) for h in range(2)]
        ad = self.node(k, c, self.roles["simd"], f"rope_i{i}", "__snax_bingo_kernel_simd_rope",
                       SnaxBingoKernelSimdRopeArgs(R4, b["pe"], ROPE, rows), [dq2, ldr] + sw)
        d2p = self.xd(cl, f"d2ap_i{i}", d_to_a_args(mesh, mt, 4, 16, b["pe"], b["a16p"]), [ad])
        # ---- Q8 per query tile: q_pe's quantisations, then q~'s ------------------------------------------------
        qps, qas = {}, {}
        for q in range(self.NQ):
            qps[q] = self.quant(cl, f"q8p{q}_i{i}", b["a16p"].view(q * 16 * ROPE * 2),
                                b["q8"][q].view(16 * KV_RANK), 16 * ROPE * 2, self.fbits(d.s_qpe.inv), [d2p])
        for q in range(self.NQ):
            qas[q] = self.mul_quant(cl, f"q8a{q}_i{i}", b["a16q"].view(q * 16 * KV_RANK * 2), sqt, b["q8"][q],
                                    16 * KV_RANK, self.fbits(d.s_qt.inv), [d2qt[q]])
        return {q: (b["q8"][q], [qas[q], qps[q]]) for q in range(self.NQ)}

    # ---- checks -------------------------------------------------------------------------------------------------
    def q8_checks(self):
        """The first and the last query tile of both clusters of the chip in the middle of the fold order, the last
        chunk (the checks copy after the window, when an earlier chunk's buffers may hold the next chunk's)."""
        d = self.d
        far = self.order[len(self.order) // 2]
        h, i = self.head[far], self.NCH - 1
        checks = []
        for c in range(self.NC):
            cl = self.chips.index(far) * self.NC + c
            for q in sorted({0, self.NQ - 1}):
                r0 = self.r0(i, c) + q * BR
                tag = f"i{i}c{c}q{q}"
                handle, ts = self.q8_of[(cl, i)][q]
                checks.append((f"q8_{tag}", self.stash(f"q8_{tag}", far, c, handle, D_QK * BR, list(ts)),
                               d.q8_a(h, r0).view("uint8"), D_QK * BR))
        return checks


if __name__ == "__main__":
    run_driver("fa prefill stage 4 workload generator", Stage4, app_name)
