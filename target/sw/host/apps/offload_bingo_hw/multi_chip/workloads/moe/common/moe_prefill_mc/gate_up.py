# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""MBuild's gate|up phase: the shared MLP's SwiGLU output quantised block by block, its later weight chunks' passes,
the experts' gate|up slot by slot, their quantise into the A8 rows and each MLP's A8 exchange between a chip's two
clusters (README.md, PER CLUSTER)."""

import moe_prefill_data as md
from bingo_kernel_args import (SnaxBingoKernelIdma2dCopyArgs, SnaxBingoKernelIdmaGatherRowsArgs,
                               SnaxBingoKernelMoeDispatchArgs, SnaxBingoKernelSimdDeqSwigluArgs,
                               SnaxBingoKernelSimdQuantRowsArgs)

from .common import BLK, D_BLOCK, SIMD_BEAT


class GateUp:

    def _next_g(self, cl):
        """The next of cluster cl's two GEMM output buffers, in turn."""
        i = self.gcount[cl] % 2
        self.gcount[cl] += 1
        return i, self.B[cl]["gbuf"][i]

    # ---- the shared MLP's quantise, block by block ------------------------------------------------
    def _sh_a16(self, cl, b, b0):
        """Where a shared chunk's SwiGLU output of a row block goes: one block's buffer, at the chunk's column b0."""
        return self.B[cl]["a16sh"].view((b0 // BLK) * D_BLOCK)

    def _sh_quant_block(self, cl, j, b, b0, n, sw):
        """Block b's columns b0 .. b0 + n of the cluster's shared MLP quantised into its A8 right after the block's
        SwiGLU (sw); the next SwiGLU into the buffer follows it. A chunk whose first column sits mid-beat is quantised
        into the staging rows instead (sh_stage_pitch). Returns the quantise."""
        d, k, c = self.d, self.chip(cl), self.local(cl)
        XR = self.XR
        sn = self.ch(k, c)[0][1]
        coff = 0 if c == 0 else BLK * self.shb[k][0]
        after = [self._sh_cp[cl]] if cl in self._sh_cp else []    # the staging rows read, every copy before this done
        pq = self.sh_stage_pitch(k, c)
        if (coff + b0) % SIMD_BEAT and pq:
            dst, pitch = self.B[cl]["shq_st"].view(b * XR * pq), pq
        else:
            dst, pitch = self.B[cl]["a8sh"].view(self.beat_aligned(b * XR * self.PS + coff + b0, "q_sh")), self.PS
        q = self.node(k, c, self.roles["simd"], f"q_sh_c{j}_b{b}", "__snax_bingo_kernel_simd_quant_rows",
                      SnaxBingoKernelSimdQuantRowsArgs(self._sh_a16(cl, b, b0), XR, n // BLK, (sn // BLK) * D_BLOCK,
                                                       d.sh["inv_a"], dst, pitch), [sw] + after)
        self._sh_qlast[cl] = q
        return q

    def _sh_quant_chunk_end(self, cl, j, b0, n):
        """A chunk quantised into the staging rows, every row block of it, copied into its columns of the A8 by the DM
        core; the copy then stands for every quantise of the cluster so far."""
        k, c = self.chip(cl), self.local(cl)
        coff = 0 if c == 0 else BLK * self.shb[k][0]
        pq = self.sh_stage_pitch(k, c)
        if not pq or not (coff + b0) % SIMD_BEAT:
            return
        self._sh_cp[cl] = self._sh_qlast[cl] = self.node(
            k, c, self.roles["dm"], f"q_sh_cp_c{j}", "__snax_bingo_kernel_idma_2d_copy",
            SnaxBingoKernelIdma2dCopyArgs(self.B[cl]["shq_st"], self.B[cl]["a8sh"].view(coff + b0), n, pq, self.PS,
                                          self.T), [self._sh_qlast[cl]])

    def _quant_sh(self, cl):
        """The shared MLP's A8 of cluster cl, quantised block by block already: (its last quantise, the cluster's
        first column, its columns), or None when the cluster holds no shared columns."""
        k, c = self.chip(cl), self.local(cl)
        (sc0, sn), _ = self.ch(k, c)
        if not sn:
            return None
        coff = 0 if c == 0 else BLK * self.shb[k][0]
        self._qa[(cl, "sh")] = self._sh_qlast[cl]
        return self._sh_qlast[cl], coff, sn

    def _exchange(self, cl, key, src, coff, ncols, rows, pitch, q):
        """Cluster cl's columns of an MLP's A8 (key: an expert slot or "sh") copied into the other cluster's A8."""
        k, c = self.chip(cl), self.local(cl)
        other = self.clusters(k)[1 - c]
        dst = (self.B[other]["a8sh"] if key == "sh" else self.B[other]["a8e"][key]).view(coff)
        # the shared MLP: after the other cluster's last quantise, whose odd chunk writes past its columns
        after = [self._sh_qlast[other]] if key == "sh" and other in self._sh_qlast else []
        self._ex_copy[(cl, key)] = self.node(k, c, self.roles["dm"], f"a8_copy_{key}", "__snax_bingo_kernel_idma_2d_copy",
                                             SnaxBingoKernelIdma2dCopyArgs(src.view(coff), dst, ncols, pitch, pitch,
                                                                           rows),
                                             [q] + after)

    def _finish_sh(self, cl):
        """The shared MLP's A8 of cluster cl done: its exchange made one slot later."""
        r = self._quant_sh(cl)
        if r is not None:
            q, coff, sn = r
            self._pend[cl].append(("sh", self.B[cl]["a8sh"], coff, sn, self.T, self.PS, q))

    # ---- the shared MLP's later chunks ------------------------------------------------------------
    def _shared_pass(self, cl, j):
        """params shared_after: shared gate|up chunk j of cluster cl over every X row block, the blocks copied again
        from the chip's X landing into the expert row buffers, which the slots before have finished with and the slots
        after gather into behind it."""
        d, R, XR = self.d, self.ROOT, self.XR
        dm, simd = self.roles["dm"], self.roles["simd"]
        k, c = self.chip(cl), self.local(cl)
        bg = self.B[cl]
        sn = self.ch(k, c)[0][1]
        b0, n = self.chunks(sn)[j]
        nsh = sn // BLK
        ld, si, slab, sc = self._load(cl, j)
        sw = gm = None
        gms = []                                            # every X block's GEMM reads the chunk's slab
        for b in range(self.XB):
            xb = bg["xe"][b % 2]
            xc = self.node(k, c, dm, f"x_copy_p{j}_b{b}", "__snax_bingo_kernel_idma_2d_copy",
                           SnaxBingoKernelIdma2dCopyArgs(self.xland(b * XR * self.H), xb, self.H, self.H, self.PX, XR),
                           list(self._xe_reader[cl][b % 2]))
            if k == R:
                self._root_gat[c] = xc                      # the root cluster's last read of X (Y lies over it)
            gi, gb = self._next_g(cl)
            gm = self.gemm(k, c, f"sg_c{j}_b{b}", xb, self.PX, slab, gb, XR // 8, self.H // 8, 2 * n // BLK, d.k_x,
                           [xc, ld] + ([self._gb_reader[cl][gi]] if self._gb_reader[cl][gi] is not None else []))
            self._xe_reader[cl][b % 2] = [gm]
            gms.append(gm)
            sw = self.node(k, c, simd, f"ssw_c{j}_b{b}", "__snax_bingo_kernel_simd_deq_swiglu",
                           SnaxBingoKernelSimdDeqSwigluArgs(gb, sc.addr, bg["sg"], self._sh_a16(cl, b, b0),
                                                            XR // 8, n // BLK, nsh * D_BLOCK),
                           [gm] + sc.deps + ([self._last_sw[cl]] if self._last_sw[cl] else []))
            self._gb_reader[cl][gi] = sw
            self._last_sw[cl] = self._sh_quant_block(cl, j, b, b0, n, sw)
        self._sh_quant_chunk_end(cl, j, b0, n)
        self._scale_read(cl, sc, sw)
        self._release(cl, si, gms, sw, sc)

    def _sh_passes(self, cl):
        """{slot: [shared chunks whose pass follows that slot's gate|up]} of cluster cl (params shared_after)."""
        nj = len(self.chunks(self.ch(self.chip(cl), self.local(cl))[0][1]))
        if self.shared_after is None or nj <= 1:
            return {}
        if len(self.shared_after) < nj - 1 or any(not 0 <= x < self.EPC for x in self.shared_after):
            raise ValueError(f"params shared_after={self.shared_after}: {nj - 1} expert slots in 0..{self.EPC - 1}")
        out = {}
        for j in range(1, nj):
            out.setdefault(self.shared_after[j - 1], []).append(j)
        return out

    # ---- the experts' gate|up, slot by slot ----------------------------------------------------------
    def _experts(self):
        """Slot by slot on every cluster: the slot's rows gathered from the chip's X (by the row lists), one GEMM a
        weight chunk, its dequantisation and SwiGLU into the A16, the quantise into the slot's A8 times the route
        weight; the previous MLP's exchange behind this slot's loads in the DM core's program; the shared passes
        shared_after names. barrier_at kernel: every exchange after the gate|up, between two barriers."""
        d, R = self.d, self.ROOT
        dm, simd = self.roles["dm"], self.roles["simd"]
        passes = {cl: self._sh_passes(cl) for cl in range(self.N_CL)}
        for cl in range(self.N_CL):
            if not passes[cl]:
                self._finish_sh(cl)
        npass = {cl: 0 for cl in range(self.N_CL)}          # each cluster's expert gate|up passes: pass p in xe[p % 2]
        for s in range(self.EPC):
            for cl in range(self.N_CL):
                k, c = self.chip(cl), self.local(cl)
                bg = self.B[cl]
                e = self.held[k][s]
                rows = md.pad8(d.cnt[e])
                _, ex = self.ch(k, c)
                e0, en = ex[e]
                lists = bg["lists"]
                caps = [md.pad8(d.cnt[x]) for x in self.held[k]]
                a16 = bg["a16e"]
                sw, gm, q = None, None, None
                chs = self.chunks(en)
                coff = 0 if c == 0 else e0
                ids0 = SnaxBingoKernelMoeDispatchArgs.rows_offset(caps, s)
                w0 = SnaxBingoKernelMoeDispatchArgs.w_offset(caps, s)
                blocks = self.row_blocks(rows)
                for r, (r0, mr) in enumerate(blocks):
                    # params expert_rows: rows r0 .. r0 + mr - 1 of the slot, its names and gate|up pass its own (r)
                    rs = "" if len(blocks) == 1 else f"r{r}"
                    xi = npass[cl] % 2
                    npass[cl] += 1
                    xe = bg["xe"][xi]
                    gat = self.node(k, c, dm, f"gather_s{s}{rs}", "__snax_bingo_kernel_idma_gather_rows",
                                    SnaxBingoKernelIdmaGatherRowsArgs(self.xland(), self.H, lists.view(ids0 + 2 * r0),
                                                                      mr, xe, self.PX),
                                    [self._disp[cl]] + list(self._xe_reader[cl][xi]))
                    if k == R:
                        self._root_gat[c] = gat             # the root cluster's last row gather (Y lies over X)
                    w_rows = lists.view(w0 + 2 * r0)
                    pass_gms = []
                    for j, (b0, n) in enumerate(chs):
                        jj = len(self.chunks(self.ch(k, c)[0][1])) + sum(len(self.chunks(ex[x][1]))
                                                                         for x in self.held[k][:s]) + j
                        ld, si, slab, sc = self._load(cl, jj, tag=f"r{r}" if r else "")
                        gi, gb = self._next_g(cl)
                        gm = self.gemm(k, c, f"gu_s{s}{rs}c{j}", xe, self.PX, slab, gb, mr // 8, self.H // 8,
                                       2 * n // BLK, d.k_x,
                                       [gat, ld] + ([self._gb_reader[cl][gi]] if self._gb_reader[cl][gi] is not None
                                                    else []))
                        pass_gms.append(gm)
                        ne = en // BLK
                        dst, stride = a16.view((b0 // BLK) * D_BLOCK), ne * D_BLOCK
                        war = [self._a16_reader[cl]] if (j == 0 and self._a16_reader[cl] is not None) else []
                        sw = self.node(k, c, simd, f"esw_s{s}{rs}c{j}", "__snax_bingo_kernel_simd_deq_swiglu",
                                       SnaxBingoKernelSimdDeqSwigluArgs(gb, sc.addr, bg["sg"], dst, mr // 8, n // BLK,
                                                                        stride),
                                       [gm] + sc.deps + war + ([self._last_sw[cl]] if self._last_sw[cl] else []))
                        self._gb_reader[cl][gi] = sw
                        self._last_sw[cl] = sw
                        self._scale_read(cl, sc, sw)
                        self._release(cl, si, gm, sw, sc)
                    self._xe_reader[cl][xi] = pass_gms
                    q = self.node(k, c, simd, f"q_s{s}{rs}", "__snax_bingo_kernel_simd_quant_rows",
                                  SnaxBingoKernelSimdQuantRowsArgs(a16, mr, en // BLK, (en // BLK) * D_BLOCK,
                                                                   d.expert(e)["inv_a"],
                                                                   bg["a8e"][s].view(self.beat_aligned(
                                                                       r0 * self.PE + coff, f"q_s{s}")), self.PE,
                                                                   w_addr=w_rows),
                                  [sw] + ([q] if q is not None else []))
                    self._a16_reader[cl] = q
                self._qa[(cl, s)] = q
                # the exchange of the previous MLP, behind this slot's loads in the DM core's program
                if self.per_kernel:
                    self._held_ex[cl] += self._pend[cl]
                else:
                    for item in self._pend[cl]:
                        self._exchange(cl, *item)
                self._pend[cl] = [(s, bg["a8e"][s], coff, en, rows, self.PE, q)]
                for j in passes[cl].get(s, []):
                    self._shared_pass(cl, j)
                if passes[cl] and s == max(passes[cl]):
                    self._finish_sh(cl)
        if self.per_kernel:
            # params barrier_at kernel: K2 ends with the gate|up; K3, the A8 exchange
            self.barrier({k: [] for k in self.chips})
            for cl in range(self.N_CL):
                self._pend[cl] = self._held_ex[cl] + self._pend[cl]
        for cl in range(self.N_CL):
            for item in self._pend[cl]:
                self._exchange(cl, *item)
        if self.per_kernel:
            self.barrier({k: [] for k in self.chips})
