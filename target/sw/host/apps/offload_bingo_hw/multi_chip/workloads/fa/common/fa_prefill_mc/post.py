# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""FaBuild's chunk after its sweep: W_UV on the sweep's o~ (INT8, the sweep's normalisation quantises it into W_UV's A
operand), o's relayout and quantisation, and W_O's two blocks into the cluster's exchange buffer."""

import numpy as np

from libs.comm.nest import d_to_a_args

from .consts import BR, D_V, Q_NOPE, TILE16


class FaPost:

    def ot8_bits(self):
        """fa_sweep's ot8: the quantiser's scale of o~, which the sweep's normalisation applies."""
        return self.fbits(self.d.s_ot.inv)

    def xt_ref(self, h, r0):
        """The sweep's xt for the query tile at row r0 of head h: (bytes, length): o~ INT8 in the A layout of one
        16-row tile."""
        d = self.d
        return d.layout.to_a(d.head(h)["ot8"][r0:r0 + BR], d.mesh).astype(np.int8).view(np.uint8), D_V * BR

    def mask_of(self, cl, plan):
        """fa_sweep's mask hook for cluster cl: a masked pair's pattern in the cluster's constants, else None."""
        return lambda j, q: (None if plan.get((j, q)) is None
                             else self.consts[cl].view(self.C_MASK + plan[(j, q)] * self.MASK_B))

    def stage_post_a(self, cl, i, sweep):
        """o~ -> W_UV -> o of chunk i (INT8, A layout): everything before the W_O matmuls, W_UV a query tile at a
        time. Returns (the tasks that read the sweep's xt, the task that made o)."""
        d, b = self.d, self.bufs(cl, i)
        rows, mt = self.ROWS, self.ROWS // 16
        mesh = d.mesh
        ld, si, slab = self.load(cl, self.chunk_index(i, "tail"), f"Ld_wuv_i{i}")
        gu = [self.gmm(cl, f"guv{q}_i{i}", sweep["xt"][q], slab, b["uv"].view(q * (Q_NOPE // 16) * TILE16), 1,
                       D_V // 4, Q_NOPE // 16, d.k_uv, [ld, sweep["tr"][q]]) for q in range(self.NQ)]
        self.ls[cl].release(si, gu)                 # every query tile's GEMM reads the slab
        cv = list(gu)
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
        # o's dequantisation is folded into its quantisation (one SIMD task after the D -> A relayout)
        du = sorted(set(gu), key=gu.index)
        d2u = self.xd(cl, f"d2au_i{i}", d_to_a_args(mesh, mt, Q_NOPE // 16, Q_NOPE // 4, b["uv"], b["a16u"]), du)
        qq = self.mul_quant(cl, f"o8_i{i}", b["a16u"], suv, b["o8"], rows * Q_NOPE, self.fbits(d.s_o.inv), [d2u])
        return cv, qq

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
            free = list(xch_after)
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
        (fold, default the root's)."""
        return self.d.reduce(fold or self.fold_heads, r0, rows)
