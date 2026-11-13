# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""The steps up to the query (README.md, THE STEPS: x, new key's columns, q, query): x into every cluster's L1, W_KVA's
columns by GEMV into the root's L3, every head's query row into the root's L3, and the root's query tile multicast."""

from bingo_kernel_args import SnaxBingoKernelIdmaFetchFlaggedArgs
from libs.comm.nest import convert_args

import chiplet_mc as cm

from .consts import (BR, C0_COS, C0_FPE, C0_FQT, C0_NSP, C0_SIN, D_QK, HIDDEN, KV_RANK, Q_HEAD, Q_NOPE, QB, QC, QROW,
                     ROPE, WCH, WQ_ORDER, XB)


class KsQuery:
    """FaDecodeKs's steps from x to the query tile; build() runs them in this order."""

    def _x_fetch(self, go, xm):
        """Every cluster fetches x's a_row from its chip's L3 into L1 behind fk_xtflag, ordered after xm (the task that
        queues x's multicast). Returns ({(chip, cluster): x in L1}, {(chip, cluster): the fetch})."""
        xa, xw = {}, {}
        for k in self.chips:
            for c in range(self.NC):
                xa[k, c] = self.l1(k, c, "x", XB)
                w = self.nd(k, c, "dm", "x_wait", "__snax_bingo_kernel_idma_fetch_flagged",
                            SnaxBingoKernelIdmaFetchFlaggedArgs(self.l3("xland"), xa[k, c], XB, self.l3("xtflag"), 1,
                                                                1), [go[k]])
                self.dfg.bingo_add_order_edge(self.queued_of(xm), w)
                xw[k, c] = w
        return xa, xw

    def _kva_columns(self, xa, xw):
        """The new key's columns: a GEMV a W_KVA chunk on the clusters 1, each FP16 row put into the root's L3 behind
        its flag. Returns the puts."""
        d, KC_, kv_puts = self.d, self.KC, []
        for k in self.chips:
            for n in self.kva[k]:
                wb = self.l1(k, KC_, f"wkva{n}", WCH)
                ld = self.load(k, f"kva{n}", wb, [xw[k, KC_]])
                ob = self.l1(k, KC_, f"kvo{n}", 2 * QC)
                gm = self.gemv(k, KC_, f"gkva{n}", xa[k, KC_], wb, ob, HIDDEN, QC, d.k_x, [ld, xw[k, KC_]])
                kv_puts.append(self.put(k, KC_, f"kva_put{n}", ob, self.l3("kvraw", 2 * QC * n, self.far(k)), 2 * QC,
                                        self.l3("kvrflag", 64 * n, self.far(k)), [gm]))
        return kv_puts

    def _q_rows(self, xa, xw):
        """Every cluster 0, its head: W_q's GEMVs, RoPE of q_pe at P, q_nope into a_row at s_qn, the absorb q~ =
        q_nope W_UK dequantised, and the head's FP16 row [q~ | q_pe] put into the root's L3 behind its flag. Returns
        {chip: the put}."""
        d, q_puts = self.d, {}
        for k in self.chips:
            h, c, b = self.head[k], self.QC, self.b0[k]
            cst = b["c"]
            qraw, qrow = self.l1(k, c, "qraw", 2 * Q_HEAD), self.l1(k, c, "qrow", QROW)
            gq = {}
            for cc in WQ_ORDER:
                wb = self.l1(k, c, f"wq{cc}", WCH)
                ld = self.load(k, f"wq{cc}", wb, [xw[k, c]])
                gq[cc] = self.gemv(k, c, f"gq{cc}", xa[k, c], wb, qraw.view(2 * QC * cc), HIDDEN, QC, d.k_x,
                                   [ld, xw[k, c]])
                if cc == WQ_ORDER[1]:
                    # q_pe: dequantised, then x cos + (its other half) x sin, at position P
                    pe, rt, rs = self.l1(k, c, "qpe", 2 * ROPE), self.l1(k, c, "qrt", 2 * ROPE), self.l1(k, c, "qrs",
                                                                                                         2 * ROPE)
                    hf = ROPE // 2
                    dq = self.mul(k, c, "dq_pe", qraw.view(2 * Q_NOPE), cst.view(C0_FPE), pe, ROPE, [gq[4], gq[5]])
                    tc = self.mul(k, c, "rp_cos", pe, cst.view(C0_COS), rt, ROPE, [dq])
                    sa = self.mul(k, c, "rp_sa", pe.view(2 * hf), cst.view(C0_SIN), rs, hf, [dq])
                    sb = self.mul(k, c, "rp_sb", pe, cst.view(C0_SIN + 2 * hf), rs.view(2 * hf), hf, [dq])
                    rope = self.add(k, c, "rp_add", rt, rs, qrow.view(2 * KV_RANK), ROPE, [tc, sa, sb])
            # q_nope into a_row, dequantised and quantised at s_qn; the absorb q~ = q_nope W_UK, dequantised
            sp = self.spread(k, c, "qn_spread", qraw, b["nsp"], Q_NOPE, [gq[cc] for cc in range(4)])
            qn8 = self.l1(k, c, "qn8", 2 * Q_NOPE)
            qn = self.muli8(k, c, "qn8", b["nsp"], cst.view(C0_NSP), qn8, 2 * Q_NOPE, d.s_qn.inv, [sp])
            wuk = self.l1(k, c, "wuk", WCH)
            ld = self.load(k, "wuk", wuk, [xw[k, c]])
            qtraw = self.l1(k, c, "qtraw", 2 * KV_RANK)
            gu = self.gemv(k, c, "guk", qn8, wuk, qtraw, Q_NOPE, KV_RANK, d.k_uk, [ld, qn])
            dqt = self.mul(k, c, "dq_qt", qtraw, cst.view(C0_FQT), qrow, KV_RANK, [gu])
            q_puts[k] = self.put(k, c, "q_put", qrow, self.l3("qrows", QROW * h, self.far(k)), QROW,
                                 self.l3("qflag", 64 * h, self.far(k)), [dqt, rope])
        return q_puts

    def _query_tile(self, q_puts):
        """The root waits for the 16 rows, converts them to the A layout, quantises (q~ at s_qt, q_pe at s_qpe) and
        multicasts Q8 into every cluster 0's exchange buffer; the root's cluster 1 copies it. Returns {(chip, cluster):
        the task after which its Q8 is there}."""
        d, R = self.d, self.ROOT
        qrm, qa16, qa8 = self.l1(R, 0, "qrm", BR * QROW), self.l1(R, 0, "qa16", BR * QROW), self.l1(R, 0, "qa8", QB)
        qg = self.fetch(R, 0, "qg_wait", self.l3("qrows"), qrm, BR * QROW, self.l3("qflag"), len(self.chips),
                        list(q_puts.values()))
        qcv = self.xd(R, 0, "q_toa", convert_args("row_major", "A", BR, D_QK, cm.MESH, 2, qrm, qa16), [qg])
        qq1 = self.quant(R, 0, "q8_lat", qa16, qa8, 2 * BR * KV_RANK, d.s_qt.inv, [qcv])
        qq2 = self.quant(R, 0, "q8_pe", qa16.view(2 * BR * KV_RANK), qa8.view(BR * KV_RANK), 2 * BR * ROPE,
                         d.s_qpe.inv, [qcv])
        qin = self.broadcast("q", qa8, QB, self.OFF_X, self.OFF_FLAG,
                             {k: ([qq1, qq2][-1] if k == R else q_puts[k]) for k in self.chips}, clusters=[0])
        self.edge(qq1, self.bcast_puts[0])
        qin[(R, 1)] = self.copy(R, 1, "q_cp", self.xch_view(R, 0, self.OFF_X), self.xch_view(R, 1, self.OFF_X), QB,
                                [qin[(R, 0)]])
        self.q_mc = list(self.bcast_puts)
        return qin
