# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""Stage 4, a chunk's queries of the head: q = x W_q, q~ = q_nope W_UK and RoPE on q_pe into the query tiles Q8; stage
5, its sweep over the keys (fa_sweep: QK with the causal masks, the online softmax, PV, the normalisation that quantises
o~)."""

import dataclasses

from bingo_kernel_args import SnaxBingoKernelIdma2dCopyArgs, SnaxBingoKernelSimdRopeArgs
from bingo_mem_handle import BingoMemSymbol
from fa_prefill_mc import HIDDEN, KV_RANK, Q_NOPE, ROPE, TILE16, FreshTileSource
from libs.blocks import fa_sweep
from libs.comm.nest import d_to_a_args


class FaStagesQueries:

    def stage_proj(self, cl, i, x):
        """q, q~ and the rotated q_pe of chunk i into Q8, one query tile at a time. The in-order SIMD core's program:
        q_nope's quantisation (its dequantisation folded in) before q_pe's dequantisation (which waits for the last q
        GEMM), so the absorb GEMM follows the q GEMMs without a gap; the absorb a query tile at a time, each tile's q~
        relayout and quantisation behind its own GEMM (they overlap the next tiles' GEMMs); in the tail every query
        tile's q_pe quantisation before the q~ ones, while the xDMA lays out q~ (s4_q8_pairs: tile by tile). Returns
        {q: (Q8 handle, [the tasks it follows])}."""
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
        # ---- Q8 per query tile: q_pe's quantisations, then q~'s; s4_q8_pairs: tile by tile ------------------------
        qps, qas = {}, {}
        order = ([(q, kd) for q in range(self.NQ) for kd in ("a", "p")] if self.s4_q8_pairs is not None else
                 [(q, kd) for kd in ("p", "a") for q in range(self.NQ)])
        for q, kind in order:
            if kind == "p":
                qps[q] = self.quant(cl, f"q8p{q}_i{i}", b["a16p"].view(q * 16 * ROPE * 2),
                                    b["q8"][q].view(16 * KV_RANK), 16 * ROPE * 2, self.fbits(d.s_qpe.inv), [d2p])
            else:
                qas[q] = self.mul_quant(cl, f"q8a{q}_i{i}", b["a16q"].view(q * 16 * KV_RANK * 2), sqt, b["q8"][q],
                                        16 * KV_RANK, self.fbits(d.s_qt.inv), [d2qt[q]])
        return {q: (b["q8"][q], [qas[q], qps[q]]) for q in range(self.NQ)}

    def sweep(self, cl, i, q8):
        """Cluster cl's sweep of chunk i over the key tiles up to the chunk's end (FreshTileSource), the causal masks on
        the scores, o~ quantised into W_UV's A operand; each softmax arena's geometry built on the cluster's xDMA core
        at the sweep's head."""
        k, c = self.chip(cl), self.local(cl)
        nt, plan = self.d.tile_plan(self.r0(i, c))
        early = 2 if len(self.ls[cl].slabs) >= 4 else 1
        return fa_sweep(self.at(k, c), dataclasses.replace(self.sweep_cfg, nt=nt), self.bufs(cl, i)["swp"],
                        FreshTileSource(self, cl, i, nt), q8, self.roles, name=f"s{i}_", early=early,
                        mask=self.mask_of(cl, plan), mask_c=True, ot8=self.ot8_bits(), geom_core="xdma")
