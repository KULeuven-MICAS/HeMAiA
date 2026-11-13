# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""The weight loads, gate|up (GEMM, dequantise, SwiGLU into A16) and the post-processing into the chip's A8 (A16
reshaped D -> A, quantised, each cluster's columns copied into the other cluster's A8)."""

from bingo_kernel_args import (SnaxBingoKernelGemmFullArgs, SnaxBingoKernelIdma1dCopyArgs,
                               SnaxBingoKernelSimdFp16ToInt8Args, SnaxBingoKernelSimdMulF16Args,
                               SnaxBingoKernelSimdSwigluF16F16Args, simd_beats)
from libs.comm.nest import d_to_a_args

from .consts import BLK, TILE


class PGateUp:
    """A pass's gate|up and its post-processing, cluster by cluster."""

    # ---- the weight stream ----------------------------------------------------------------------
    def gemm_node(self, k, c, name, args, deps):
        """One GEMM on chip k cluster c (gemm_full); returns the task its output is ready after and its operands are
        free after."""
        return self.node(k, c, self.roles["gemm"], name, "__snax_bingo_kernel_gemm_full", args, deps)

    def _load(self, cl, i, j, deps=()):
        """Weight chunk j of cluster cl into the next slab: a ring load in pass 0 (every pass with params restream), a
        copy from its slot later. Returns (load task, slab index, slab)."""
        k, c, dm = self.chip(cl), self.local(cl), self.roles["dm"]
        ls = self.ls[cl]
        off, nb = self.chunk_off[cl][j]
        si, slab, war = ls.take(nb)
        if i == 0 or self.restream:
            ld, sl = ls.rings.load_chunk(
                lambda n, core, kern, args, after: self.node(k, c, core, n, kern, args, after), c,
                f"Ld_c{j}_i{i}", src=self.img[cl].address + off, nbytes=nb, slab=slab, war=war,
                deps=deps, dm=dm)
            self.slot[(cl, j)] = (sl, ld)
        else:
            sl, first = self.slot[(cl, j)]
            ld = self.node(k, c, dm, f"Ld_c{j}_i{i}", "__snax_bingo_kernel_idma_1d_copy",
                           SnaxBingoKernelIdma1dCopyArgs(sl, slab, nb),
                           list(dict.fromkeys(war + [first] + list(deps))))
        return ld, si, slab

    # ---- gate|up ----------------------------------------------------------------------------------
    def stage_gu(self, i, x):
        """Every cluster's gate|up chunks, each dequantised and through SwiGLU. Returns {cl: the last SwiGLU} and {cl:
        every gate|up GEMM of the pass} (the readers of X)."""
        simd = self.roles["simd"]
        last_sw, last_gemm = {}, {}
        for cl, (_, n) in self.cols.items():
            k, c = self.chip(cl), self.local(cl)
            nb = n // BLK
            xb = self.xbuf[cl][0]
            soff = 0
            sw = None
            xblk = self.x_blocks > 1 and i == 0
            xws = [x[cl]]
            for j, (b0, nb_) in enumerate(self.chunks[cl]):
                ld, si, slab = self._load(cl, i, j, [x[cl]] if j == 0 else ())
                if not self.per_kernel and j == len(self.chunks[cl]) - 1:
                    # the first down chunk's load right behind the last gate|up chunk's, ahead of the A8 copies in the
                    # DM core's program (those wait for the quantise): where each core's tasks keep the order they were
                    # made in (the schedule's fallback), it is not held behind them
                    self.dn_first_load[cl] = self._load(cl, i, j + 1)
                if xblk and j == 0:
                    # the other X blocks behind the first weight load in the DM core's program
                    xws += [self.x_wait_block(cl, bb, self.x_bc[0]) for bb in range(1, self.x_blocks)]
                b = self.gu_count[cl] % 2
                self.gu_count[cl] += 1
                gb = self.gbuf[cl][b]
                war = list(self.gb_reader[cl][b])   # the buffer's readers of two chunks ago
                nblk = 2 * nb_                      # its gate blocks and its up blocks
                half = nb_ * TILE                   # one row block's gate (or up) tiles
                kargs = dict(input_B_addr=slab, input_C_addr=0, K=self.D // 4, N=nblk, array_shape_idx=0,
                             transpose_A=0, transpose_B=0, accumPrevC=0, int32tofp16_enable=1, int4_b_enable=self.int4_b,
                             d_shift=self.d.k_x)
                last = self.split_last and j == len(self.chunks[cl]) - 1 and not (xblk and j == 0)
                if (xblk and j == 0) or last:       # one GEMM per X block (per post block): its row tiles only
                    nsp = self.post_blocks if last else self.x_blocks
                    mb = self.MT // nsp
                    gms = [self.gemm_node(k, c, f"gu_c{j}_i{i}b{bb}",
                                          SnaxBingoKernelGemmFullArgs(
                                              input_A_addr=xb.view(bb * mb * (self.XB // self.MT)),
                                              output_D_addr=gb.view(bb * mb * 2 * half), M=mb, **kargs),
                                          [ld] + (xws if last else [xws[bb]]) + war)
                           for bb in range(nsp)]
                    gm_of = lambda m, gms=gms, mb=mb: gms[m // mb]     # noqa: E731
                    gm = gms[-1]
                else:
                    gm = self.gemm_node(k, c, f"gu_c{j}_i{i}",
                                        SnaxBingoKernelGemmFullArgs(input_A_addr=xb, output_D_addr=gb, M=self.MT, **kargs),
                                        [ld] + xws + war)
                    gms = [gm]
                    gm_of = lambda m, gm=gm: gm                       # noqa: E731
                self.ls[cl].release(si, gms)            # every GEMM of the chunk reads the slab
                last_gemm.setdefault(cl, []).extend(gms)
                if last:
                    sw, swi_m, readers = self._gu_tail(cl, i, j, b0, nb_, gb, soff, gm_of)
                    soff += nblk * 256 * 2
                    self.gb_reader[cl][b] = readers
                    continue
                if self.deq_per_row:                # one task per row tile, all with the chunk's one scale tile
                    el = nblk * 256
                    dqs = [self.node(k, c, simd, f"deq_c{j}m{m}_i{i}", "__snax_bingo_kernel_simd_stream_elementwise",
                                     SnaxBingoKernelSimdMulF16Args(gb.view(m * 2 * half), self.sbuf[cl].view(soff),
                                                                   gb.view(m * 2 * half), el // 32, 32), [gm_of(m)])
                           for m in range(self.MT)]
                else:
                    el = self.MT * nblk * 256
                    dq = self.node(k, c, simd, f"deq_c{j}_i{i}", "__snax_bingo_kernel_simd_stream_elementwise",
                                   SnaxBingoKernelSimdMulF16Args(gb, self.sbuf[cl].view(soff), gb,
                                                                 el // 32, 32), [gm])
                    dqs = [dq] * self.MT
                soff += el * 2
                swi_m = []
                for m in range(self.MT):
                    dq = dqs[m]
                    gate = gb.view(m * 2 * half)
                    up = gb.view(m * 2 * half + half)
                    out = self.a16d[cl].view((m * nb + b0) * TILE)
                    sw = self.node(k, c, simd, f"swi_c{j}m{m}_i{i}",
                                   "__snax_bingo_kernel_simd_swiglu_f16_f16",
                                   SnaxBingoKernelSimdSwigluF16F16Args(gate, up, out, 1,
                                                                       nb_ * 256),
                                   [dq] + self._a16_war(cl, i, m))
                    swi_m.append(sw)
                    self.a16_writers[(cl, i)].setdefault(m, []).append(sw)
                self.gb_reader[cl][b] = list(dict.fromkeys(dqs)) + swi_m
            last_sw[cl] = sw
            self.last_swi[(cl, i)] = swi_m          # the last chunk's SwiGLU of each row tile
        return last_sw, last_gemm

    def _gu_tail(self, cl, i, j, b0, nb_, gb, soff, gm_of):
        """params split_last: the last chunk's dequantise and SwiGLU row tile by row tile; behind each post block's last
        SwiGLU its reshape (xDMA) and the previous block's quantise (SIMD, whose reshape is done by then). Returns (the
        last SwiGLU, the SwiGLU of each row tile, every reader of the gate|up buffer)."""
        k, c, simd = self.chip(cl), self.local(cl), self.roles["simd"]
        n = self.cols[cl][1]
        nb, half, el = n // BLK, nb_ * TILE, 2 * nb_ * 256
        pmb = self.MT // self.post_blocks
        swi_m, rs, sw, dqs = [], [], None, []
        for m in range(self.MT):
            dq = self.node(k, c, simd, f"deq_c{j}m{m}_i{i}", "__snax_bingo_kernel_simd_stream_elementwise",
                           SnaxBingoKernelSimdMulF16Args(gb.view(m * 2 * half), self.sbuf[cl].view(soff),
                                                         gb.view(m * 2 * half), el // 32, 32), [gm_of(m)])
            sw = self.node(k, c, simd, f"swi_c{j}m{m}_i{i}", "__snax_bingo_kernel_simd_swiglu_f16_f16",
                           SnaxBingoKernelSimdSwigluF16F16Args(gb.view(m * 2 * half), gb.view(m * 2 * half + half),
                                                               self.a16d[cl].view((m * nb + b0) * TILE), 1, nb_ * 256),
                           [dq] + self._a16_war(cl, i, m))
            swi_m.append(sw)
            dqs.append(dq)
            self.a16_writers[(cl, i)].setdefault(m, []).append(sw)
            if (m + 1) % pmb == 0:
                bb = m // pmb
                rs.append(self._post_d2a(cl, i, bb))
                if bb:
                    self.quant[(cl, i)] = self._post_quant(cl, i, bb - 1, rs[bb - 1])
        self.quant[(cl, i)] = self._post_quant(cl, i, self.post_blocks - 1, rs[-1])
        self.d2a[(cl, i)] = rs
        self.post_made.add((cl, i))
        return sw, swi_m, dqs + swi_m

    def _a16_war(self, cl, i, m):
        """A SwiGLU into row tile m of A16 (pass i): after the previous pass's reshape of that row's post block."""
        prev = self.d2a.get((cl, i - 1))
        return [prev[m // (self.MT // self.post_blocks)]] if prev else []

    # ---- A16 -> the chip's A8 ---------------------------------------------------------------------
    def _post_d2a(self, cl, i, b):
        """Post block b of A16 (its row tiles, every column) reshaped to the A layout: after every SwiGLU into its rows
        (each chunk's) and the previous pass's quantise of the A16a rows it overwrites."""
        k, c, n = self.chip(cl), self.local(cl), self.cols[cl][1]
        mb, nb, m0 = self.MT // self.post_blocks, n // BLK, b * (self.MT // self.post_blocks)
        rows = range(m0, m0 + mb)
        war_q = [self.q_all[(cl, i - 1)][m] for m in rows] if (cl, i - 1) in self.q_all else []
        return self.node(k, c, self.roles["xdma"], f"d2a_i{i}b{b}", "__snax_bingo_kernel_xdma_6d",
                         d_to_a_args(self.d.mesh, mb, nb, n // 4, self.a16d[cl].view(m0 * nb * TILE),
                                     self.a16a[cl].view(m0 * 16 * n * 2)),
                         [w for m in rows for w in self.a16_writers[(cl, i)][m]] + war_q)

    def _post_quant(self, cl, i, b, rs):
        """Post block b of A16a quantised, one row tile at a time, into this cluster's run of the chip's A8, each run
        then copied into the other cluster's A8 (self.ex_push[(cl, i)]); a later pass's copies also follow
        self.push_war[cl] (the other cluster's readers of its A8 in the pass before)."""
        k, c, n = self.chip(cl), self.local(cl), self.cols[cl][1]
        mb, m0 = self.MT // self.post_blocks, b * (self.MT // self.post_blocks)
        war = self.a8_reader[cl][i % 2]
        kn = self.kcols[k][1]
        other = self.clusters(k)[1 - c]
        coff = 16 * (kn - n) if c else 0    # this cluster's columns sit after the first cluster's
        q = None
        for m in range(m0, m0 + mb):
            off = m * 16 * kn + coff
            q = self.node(k, c, self.roles["simd"], f"quant_i{i}b{b}m{m}",
                          "__snax_bingo_kernel_simd_fp16_to_int8",
                          SnaxBingoKernelSimdFp16ToInt8Args(self.a16a[cl].view(m * 16 * n * 2),
                                                            self.a8[cl][0].view(off),
                                                            beats=simd_beats(16 * n * 2), rows=1,
                                                            inv_scale_f32bits=self.d.inv_a),
                          [rs] + war)
            self.q_tile.setdefault(cl, {})[m] = q
            self.q_all.setdefault((cl, i), {})[m] = q
            self.ex_push.setdefault((cl, i), []).append(
                self.node(k, c, self.roles["dm"], f"a8_push_m{m}" if i == 0 else f"a8_push_i{i}m{m}",
                          "__snax_bingo_kernel_idma_1d_copy",
                          SnaxBingoKernelIdma1dCopyArgs(self.a8[cl][0].view(off), self.a8[other][0].view(off), 16 * n),
                          [q] + self.push_war.get(cl, [])))    # each copy after the other cluster's readers
        return q

    def stage_post(self, i, sw):
        """A16 reshaped to the A layout and quantised into the chip's A8, post block by post block (split_last made the
        last chunk's already). {cl: the last quantise}."""
        out = {}
        for cl in self.cols:
            if (cl, i) in self.post_made:
                out[cl] = self.quant[(cl, i)]
                continue
            q, rs = None, []
            for b in range(self.post_blocks):
                rs.append(self._post_d2a(cl, i, b))
                q = self._post_quant(cl, i, b, rs[-1])
            self.d2a[(cl, i)] = rs
            out[cl] = q
            self.quant[(cl, i)] = q
        return out
