# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""The key-split attention (README.md, THE STEPS: pass 1, m*, pass 2): every attention cluster's row maximum over its
tiles, m* met by one ChainGather and multicast, and the softmax seeded at m* with PV, each cluster's partial."""

from bingo_kernel_args import SnaxBingoKernelGemmFaQkArgs, SnaxBingoKernelXdmaChainGatherArgs
from libs import at_offset

import chiplet_mc as cm

from .consts import (ARENA, BC, BEAT, BR, C1_MASK, D_QK, D_V, EW_F16_MAX, FA, KB, LRUN, MRUN, OB, PB, PVK, SB, VB,
                     WCH, XFER)


class KsSweep:
    """FaDecodeKs's attention: build() runs _pass1, _m_star, _pass2 in this order."""

    def _pass1(self, qin, new_ready):
        """Every attention cluster (units) over its tiles: K from the ring (the new tile's from new_ready), S^T = K Q^T
        and the FA softmax with its own seed, the lanewise maximum. Returns (score tiles, last pass-1 softmax, pass-1
        arena), each by (chip, cluster)."""
        d, R, b1 = self.d, self.ROOT, self.b1
        s16, ar1, sm1 = {}, {}, {}
        for k, c, ts in self.units:
            q8 = self.xch_view(k, c, self.OFF_X)
            ar1[k, c] = self.l1(R, 0, "ar1", ARENA) if (k, c) == (R, 0) else self.xch_view(k, c, self.OFF_MX)
            s16[k, c] = self.l1(k, c, "s16", len(ts) * SB)
            p8 = self.l1(k, c, "p8a", FA.p8_bytes(BC, BR, FA.EXACT))
            sm = []
            for i, t in enumerate(ts):
                if t == self.NTC:
                    kb, ready, bias = b1["kt"], new_ready["K"], b1["c"].view(C1_MASK)
                else:
                    kb = self.l1(k, c, f"kt{i}", KB)
                    ready, bias = [self.load(k, f"K{t}", kb, [qin[(k, c)]])], 0
                qk = self.nd(k, c, "gemm", f"qk{i}", "__snax_bingo_kernel_gemm_fa_qk",
                             SnaxBingoKernelGemmFaQkArgs(kb, q8, bias, at_offset(s16[k, c], i * SB + BEAT), M=BC // 16,
                                                         K=D_QK // 4, N=BR // 16, d_shift=d.k_s),
                             ready + [qin[(k, c)]])
                sm.append(self.nd(k, c, "simd", f"s1_{i}", "__snax_bingo_kernel_simd_fa_softmax",
                                  FA(at_offset(s16[k, c], i * SB + BEAT), p8, ar1[k, c], BC, 1, tile_idx=i,
                                     seed_state=1, geom_mode=FA.GEOM_SELF, score_scale=d.a_exp, p_mode=FA.EXACT,
                                     p8_pitch=0),
                                  [qk] + sm[-1:]))
            sm1[k, c] = sm[-1]
        return s16, sm1, ar1

    def _m_star(self, sm1, ar1, qin):
        """Each chip's maximum (the root: its two clusters'), then m*: one ChainGather of the maxima through every chip
        with the Elementwise junction's FP16 MAX into the root, after every chip's last ring load (the V tiles and the
        tail loaded first), multicast into every chip's pass-2 arena (cluster 0; the root's cluster 1 copies it).
        Returns (m* landed, by (chip, cluster); the V tiles' loads, by (chip, tile))."""
        R, QC_ = self.ROOT, self.QC
        mloc = {k: sm1[k, QC_] for k in self.chips if k != R}
        mt, p8m = self.l1(R, 0, "mt", 3 * BEAT), self.l1(R, 0, "p8m", FA.p8_bytes(2, BR, 0))
        cp0 = self.copy(R, 0, "m_cp0", at_offset(ar1[R, 0], MRUN), at_offset(mt, BEAT), BEAT, [sm1[R, 0]])
        cp1 = self.copy(R, 1, "m_cp1", self.xch_view(R, 1, self.OFF_MLOC), at_offset(mt, 2 * BEAT), BEAT, [sm1[R, 1]])
        mloc[R] = self.nd(R, 0, "simd", "m_max", "__snax_bingo_kernel_simd_fa_softmax",
                          FA(at_offset(mt, BEAT), p8m, self.xch_view(R, 0, self.OFF_MX), 2, 1, tile_idx=0, seed_state=1,
                             geom_mode=FA.GEOM_SELF, score_scale=1.0, p_mode=0, p8_pitch=0), [cp0, cp1])
        vld = {}
        for k in self.chips:
            vld.update(self._v_loads(k, qin))
        for k in self.chips:
            self._stream_tail(k)
        ga = SnaxBingoKernelXdmaChainGatherArgs.fold_into(
            cm.remote, R, 0, self.order[1:], src=self.xch_addr(0, self.OFF_MLOC),
            dst=self.xch_addr(0, self.OFF_MR + MRUN),
            size=XFER, junction=self.jct["Elementwise"], jct_csr0=EW_F16_MAX)
        made = self.gather(R, 0, "mg_chain", ga, [mloc[k] for k in self.chips] + [self.last_chunk[k] for k in self.chips])
        self._chains.append(made)
        self.m_made, self.msrc = made, self.xch_view(R, 0, self.OFF_MR + MRUN)
        marr = self.broadcast("m", self.msrc, XFER, self.OFF_M, self.OFF_MFLAG,
                              {k: (made if k == R else vld[k, self.tiles[k][-1]]) for k in self.chips}, clusters=[0])
        self.m_mc = list(self.bcast_puts)
        marr[(R, 1)] = self.copy(R, 1, "m_cp", self.msrc, self.xch_view(R, 1, self.OFF_M), XFER, [made])
        return marr, vld

    def _pass2(self, s16, sm1, marr, vld, new_ready):
        """Every attention cluster: the softmax seeded at m* and PV over its tiles (V from the ring, the new tile's
        from new_ready), O^T leaving the D port as FP16, then its row sums: the partial. Returns (partial, the copy of
        its row sums), each by (chip, cluster)."""
        d, b1 = self.d, self.b1
        part, lcp = {}, {}
        for k, c, ts in self.units:
            ar2 = self.xch_view(k, c, self.OFF_A2)
            p8 = [self.l1(k, c, f"p8b{j}", FA.p8_bytes(BC, BR, FA.EXACT)) for j in range(2)]
            oacc = self.l1(k, c, "oacc", D_V * BR * 4) if len(ts) > 1 else None
            single = sum(1 for kk, _, _ in self.units if kk == k) == 1
            part[k, c] = self.xch_view(k, 0, self.OFF_LOC) if single else self.l1(k, c, "part", PB)
            corr = FA.corr_offset(BC, BR)
            sm, pv = [], []
            for i, t in enumerate(ts):
                first, last = i == 0, i == len(ts) - 1
                vb, ready = (b1["vt"], new_ready["V"]) if t == self.NTC else (self.vbuf[k, t], [vld[k, t]])
                sm.append(self.nd(k, c, "simd", f"s2_{i}", "__snax_bingo_kernel_simd_fa_softmax",
                                  FA(at_offset(s16[k, c], i * SB + BEAT), p8[i % 2], ar2, BC, 1, tile_idx=i,
                                     seed_state=0, geom_mode=FA.GEOM_SELF, score_scale=d.a_exp, p_mode=FA.EXACT,
                                     p8_pitch=0),
                                  [marr[(k, c)], sm1[k, c]] + sm[-1:] + (pv[i - 2:i - 1] if i >= 2 else [])))
                flags = PVK.B_KMAJOR | (0 if first else PVK.C_COLSCALE) | (PVK.D_FP16 if last else 0)
                pv.append(self.nd(k, c, "gemm", f"pv{i}", "__snax_bingo_kernel_gemm_fa_pv",
                                  PVK(vb, p8[i % 2], 0 if first else oacc, part[k, c] if last else oacc, M=D_V // 16,
                                      K=BC // 4, N=BR // 16, flags=flags,
                                      corr_addr=0 if first else at_offset(p8[i % 2], corr),
                                      d_shift=d.k_o if last else 0),
                                  [sm[i]] + ready + pv[-1:]))
            lcp[k, c] = self.copy(k, c, "l_cp", self.xch_view(k, c, self.OFF_A2 + LRUN), at_offset(part[k, c], OB),
                                  XFER, [sm[-1], pv[-1]])
        return part, lcp

    def _v_loads(self, k, qin):
        """Chip k's V tiles into L1 on its cluster 0 (pass 2's tiles, loaded while m* is made)."""
        out = {}
        for i, t in enumerate(self.tiles[k]):
            vb = self.vbuf[k, t] = self.l1(k, self.QC, f"vt{i}", VB)
            out[k, t] = self.load(k, f"V{t}", vb, [qin[(k, self.QC)]])
        return out

    def _stream_tail(self, k):
        """Chip k's last ring chunks into L1: W_UV and W_O on cluster 0, W_O on cluster 1 (cluster 0's last W_O chunk is
        the chip's last push)."""
        self.wuv[k] = self.l1(k, self.QC, "wuv", WCH)
        self.wuv_ld[k] = self.load(k, "wuv", self.wuv[k])
        for c in (self.KC, self.QC):
            for s in range(2):
                self.wo[k, c, s] = self.l1(k, c, f"wo{s}", WCH)
                self.wo_ld[k, c, s] = self.load(k, f"wo{c}{s}", self.wo[k, c, s])
