# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""From the partials to y (README.md, THE STEPS: reduce, W_UV, W_O): the partials folded into the root by one
ChainGather, o~ normalised and multicast, every head's W_UV, and W_O by columns into the root's y row."""

from bingo_kernel_args import (SnaxBingoKernelSimdAddF16Args, SnaxBingoKernelSimdMlaNormaliseArgs,
                               SnaxBingoKernelXdmaChainGatherArgs, SnaxBingoKernelXdmaTranspose2dArgs)

import chiplet_mc as cm

from .consts import BR, C0_UVSP, D_V, HIDDEN, KV_RANK, LAT, OAR, OB, PB, QC, V_HEAD, XB


class KsOutput:
    """FaDecodeKs's steps after the attention: build() runs _reduce, _o_tilde, _w_uv, _w_o in this order."""

    def _reduce(self, part, lcp):
        """The root's two clusters' partials added, then one ChainGather through every chip (the fold order) with the
        Elementwise junction's FP16 ADD into the root. Returns the chain."""
        R, QC_ = self.ROOT, self.QC
        loc = {k: lcp[k, QC_] for k in self.chips}
        pp = self.copy(R, 1, "part_push", part[R, 1], self.p1[R], PB, [lcp[R, 1]])
        loc[R] = self.nd(R, 0, "simd", "loc_add", "__snax_bingo_kernel_simd_stream_elementwise",
                         SnaxBingoKernelSimdAddF16Args(part[R, 0], self.p1[R], self.xch_view(R, 0, self.OFF_LOC),
                                                       PB // 64, 32), [lcp[R, 0], pp])
        ga = SnaxBingoKernelXdmaChainGatherArgs.fold_into(
            cm.remote, R, 0, self.order[1:], src=self.xch_addr(0, self.OFF_LOC), dst=self.dst_expr(), size=PB,
            junction=self.jct["Elementwise"], jct_csr0=cm.EW_F16_ADD, holds=[self.dst_buf])
        chain = self.gather(R, 0, "o_chain", ga, [loc[k] for k in self.chips])
        self._chains.append(chain)
        return chain

    def _o_tilde(self, chain):
        """o~ on the root: normalised (O c / l), a head a row, spread into a_row, quantised at s_ot and multicast into
        every chip's L3. Returns the quantise."""
        d, R = self.d, self.ROOT
        self.ot, self.xt = self.l1(R, 0, "ot", OB), self.l1(R, 0, "xt", OB)
        nrm = self.nd(R, 0, "simd", "o_norm", "__snax_bingo_kernel_simd_mla_normalise",
                      SnaxBingoKernelSimdMlaNormaliseArgs(self.dst.view(OB), self.dst_buf, self.dst, self.ot, D_V,
                                                          d.a_n_bits), [chain])
        tr = self.nd(R, 0, "xdma", "o_tr", "__snax_bingo_kernel_xdma_transpose_2d",
                     SnaxBingoKernelXdmaTranspose2dArgs(self.ot, self.xt, D_V, BR, 2), [nrm])
        osp = self.br["osp"]
        ospr = self.spread(R, 0, "o_spread", self.xt, osp, BR * KV_RANK, [tr])
        self.oar = self.l1(R, 0, "oar", BR * LAT)
        oq = self.quant(R, 0, "o_q8", osp, self.oar, 2 * BR * LAT, d.s_ot.inv, [ospr])
        self.ol_mc = self.put(R, 0, "ol_bput", self.oar, self.l3("olat"), BR * LAT, self.l3("olflag"), [oq], bcast=True)
        return oq

    def _w_uv(self, oq):
        """Every cluster 0 its head's o~ row: o_h = o~_h W_UV,h, spread into a_row, dequantised and quantised at s_o,
        o8_h put into the root's L3; the root waits for the 16 and multicasts the o8 row. Returns the root's wait."""
        d, R, QC_ = self.d, self.ROOT, self.QC
        o_puts = {}
        for k in self.chips:
            h, c, bk = self.head[k], QC_, self.b0[k]
            if k == R:
                ola, got = self.oar.view(LAT * h), oq
            else:
                ola = self.l1(k, c, "ola", LAT)
                got = self.fetch(k, c, "ol_get", self.l3("olat", LAT * h), ola, LAT, self.l3("olflag"), 1, [self.ol_mc])
            ohr, o8 = self.l1(k, c, "ohraw", 2 * V_HEAD), self.l1(k, c, "o8", OAR)
            gv = self.gemv(k, c, "guv", ola, self.wuv[k], ohr, KV_RANK, V_HEAD, d.k_uv, [got, self.wuv_ld[k]])
            osr = self.spread(k, c, "oh_spread", ohr, bk["osp"], V_HEAD, [gv])
            oqk = self.muli8(k, c, "o8", bk["osp"], bk["c"].view(C0_UVSP), o8, 2 * V_HEAD, d.s_o.inv, [osr])
            o_puts[k] = self.put(k, c, "o8_put", o8, self.l3("oland", OAR * h, self.far(k)), OAR,
                                 self.l3("oflag", 64 * h, self.far(k)), [oqk])
        ow = self.wait(R, 0, "o8_wait", self.l3("oflag"), len(self.chips), list(o_puts.values()))
        self.o8_mc = self.put(R, 0, "o8_bput", self.l3("oland"), self.l3("oland"), len(self.chips) * OAR,
                              self.l3("omflag"), [ow], bcast=True)
        return ow

    def _w_o(self, ow):
        """W_O by columns: every cluster its 64 columns over the full depth (two GEMVs of 32), put into the root's y row
        behind its flag. Returns the root's wait for the 32."""
        d, R = self.d, self.ROOT
        y_puts = []
        for i, k in enumerate(self.chips):
            for c in range(self.NC):
                cl = i * self.NC + c
                ofl, yw_ = self.l1(k, c, "ofull", XB), self.l1(k, c, "ywo", 4 * QC)
                if k == R:
                    get = self.copy(R, c, "o8_get", self.l3("oland"), ofl, len(self.chips) * OAR, [ow])
                else:
                    get = self.fetch(k, c, "o8_get", self.l3("oland"), ofl, len(self.chips) * OAR, self.l3("omflag"), 1,
                                     [self.o8_mc])
                gms = [self.gemv(k, c, f"gwo{s}", ofl, self.wo[k, c, s], yw_.view(2 * QC * s), HIDDEN, QC, d.k_wo,
                                 [get, self.wo_ld[k, c, s]]) for s in range(2)]
                y_puts.append(self.put(k, c, "y_put", yw_, self.l3("yrow", 4 * QC * cl, self.far(k)), 4 * QC,
                                       self.l3("yflag", 64 * cl, self.far(k)), gms))
        return self.wait(R, 0, "y_wait", self.l3("yflag"), self.N_CL, y_puts)
