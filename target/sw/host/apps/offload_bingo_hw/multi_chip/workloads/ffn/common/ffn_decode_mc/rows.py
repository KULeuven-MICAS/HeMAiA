# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""X, gate|up and the A8 as rows: the zeroed row buffers, X's fetch, the gate|up GEMVs with their dequantise and
SwiGLU, and the A16 row spread into a_row, quantised into the chip's A8 row and exchanged between the clusters."""

from bingo_kernel_args import (SnaxBingoKernelGemvArgs, SnaxBingoKernelIdma1dCopyArgs,
                               SnaxBingoKernelIdmaFetchFlaggedArgs, SnaxBingoKernelSimdFp16ToInt8Args,
                               SnaxBingoKernelSimdMulF16Args, SnaxBingoKernelSimdSwigluF16F16Args,
                               SnaxBingoKernelXdma6dArgs, SnaxBingoKernelXdmaMemsetArgs, simd_beats)
from bingo_mem_handle import BingoMemSymbol
from ffn_prefill_mc import BLK

from .params import ATOM, LANES


class DRows:
    """The decode's X, gate|up and post-processing (PBuild's stage_x, stage_gu and stage_post as rows)."""

    def pre_window(self):
        """The prefill's (the scales into L1), with y_dequant s_dn into the root's cluster 0, and every cluster's A16
        row and its a_row form zeroed (the a_row form's free bytes stay zero, so the A8 row's do)."""
        pre = super().pre_window()
        if self.y_dequant:
            R = self.ROOT
            pre[R].append(self.node(R, 0, self.roles["dm"], "sdn_load", "__snax_bingo_kernel_idma_1d_copy",
                                    SnaxBingoKernelIdma1dCopyArgs(self.h_sdn, self.sdn, 2 * self.D), pre[R][-1:]))
        for cl in range(self.N_CL):
            k, c = self.chip(cl), self.local(cl)
            pre[k].append(self.node(k, c, self.roles["xdma"], "row_zero", "__snax_bingo_kernel_xdma_memset",
                                    SnaxBingoKernelXdmaMemsetArgs(self.rowp[cl], self.rowp[cl].size,
                                                                  SnaxBingoKernelXdmaMemsetArgs.PATTERN_ZERO), []))
        return pre

    def stage_x(self, i, src, after):
        """The a_row row into every cluster's L1 in one flagged fetch, behind the multicast's flag. {cl: the task after
        which it is there}."""
        T, dm = self.TAG, self.roles["dm"]
        out = {}
        for cl in range(self.N_CL):
            k, c = self.chip(cl), self.local(cl)
            out[cl] = self.node(k, c, dm, f"x_wait_i{i}", "__snax_bingo_kernel_idma_fetch_flagged",
                                SnaxBingoKernelIdmaFetchFlaggedArgs(self._xland(0), self.xbuf[cl][0], self.XR,
                                                                    BingoMemSymbol(f"{T}_xtflag"), 1, 1),
                                list(after.get(cl, [])))
            # the multicast's flag says the row has landed: an order edge, not a dependency
            self.dfg.bingo_add_order_edge(self.x_post(src, 0, k), out[cl])
        return out

    def stage_gu(self, i, x):
        """Every cluster's gate|up chunks, each a GEMV of the X row into the chunk's FP16 row as it lands, the row
        dequantised and its gate half through SwiGLU with its up half into the chunk's columns of the A16 row. Returns
        {cl: the last SwiGLU}, {cl: the last GEMV}."""
        gemm, simd = self.roles["gemm"], self.roles["simd"]
        last_sw, last_gemm = {}, {}
        for cl in self.cols:
            k, c = self.chip(cl), self.local(cl)
            sw = None
            for j, (b0, nb_) in enumerate(self.chunks[cl]):
                ld, si, slab = self._load(cl, i, j, [x[cl]] if j == 0 else ())
                N, off = 2 * nb_ * BLK, 4 * BLK * b0    # its gate columns then its up columns; 64 B a gate block
                row = self.grow[cl].view(off)
                gm = self.node(k, c, gemm, f"gu_c{j}_i{i}", "__snax_bingo_kernel_gemv",
                               SnaxBingoKernelGemvArgs(self.xbuf[cl][0], slab, row, K=self.D, N=N, d_shift=self.d.k_x,
                                                       a_blk=8, w4=True), [ld, x[cl]])
                self.ls[cl].release(si, gm)
                last_gemm[cl] = gm
                dq = self.node(k, c, simd, f"deq_c{j}_i{i}", "__snax_bingo_kernel_simd_stream_elementwise",
                               SnaxBingoKernelSimdMulF16Args(row, self.sbuf[cl].view(off), row, N // 32, 32), [gm])
                sw = self.node(k, c, simd, f"swi_c{j}_i{i}", "__snax_bingo_kernel_simd_swiglu_f16_f16",
                               SnaxBingoKernelSimdSwigluF16F16Args(row, self.grow[cl].view(off + 2 * nb_ * BLK),
                                                                   self.a16r[cl].view(2 * BLK * b0), 1, nb_ * BLK),
                               [dq] + ([sw] if sw is not None else []))     # in turn: the last stands for every one
            last_sw[cl] = sw
        return last_sw, last_gemm

    def stage_post(self, i, sw):
        """Behind each cluster's last SwiGLU its A16 row spread into a_row's FP16 form (xDMA: 8-byte atoms 16 bytes
        apart, whole beats), quantised into the cluster's run of the chip's A8 row (SIMD), and that run copied into the
        other cluster's A8 row (DM; self.ex_push, named as the prefill's row-tile copy). {cl: the quantise}."""
        out = {}
        for cl, (_, n) in self.cols.items():
            k, c = self.chip(cl), self.local(cl)
            kn = self.kcols[k][1]
            other = self.clusters(k)[1 - c]
            off = 2 * (kn - n) if c else 0     # this cluster's columns sit after the first cluster's, 2 B a column
            beats = -(-n // (4 * LANES))
            sp = self.node(k, c, self.roles["xdma"], f"a2r_i{i}", "__snax_bingo_kernel_xdma_6d",
                           SnaxBingoKernelXdma6dArgs(self.a16r[cl], self.ar16[cl], ATOM, 2 * ATOM, [LANES * ATOM],
                                                     [beats], [LANES * 2 * ATOM], [beats]), [sw[cl]])
            run = self.a8[cl][0].view(off)
            q = self.node(k, c, self.roles["simd"], f"quant_i{i}", "__snax_bingo_kernel_simd_fp16_to_int8",
                          SnaxBingoKernelSimdFp16ToInt8Args(self.ar16[cl], run, beats=simd_beats(4 * n), rows=1,
                                                            inv_scale_f32bits=self.d.inv_a), [sp])
            self.ex_push.setdefault((cl, i), []).append(
                self.node(k, c, self.roles["dm"], "a8_push_m0" if i == 0 else f"a8_push_i{i}m0",
                          "__snax_bingo_kernel_idma_1d_copy",
                          SnaxBingoKernelIdma1dCopyArgs(run, self.a8[other][0].view(off), 2 * n), [q]))
            self.quant[(cl, i)] = out[cl] = q
        return out
