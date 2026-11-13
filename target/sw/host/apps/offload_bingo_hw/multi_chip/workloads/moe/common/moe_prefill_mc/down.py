# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""MBuild's down phase: per output slice the down projection on every cluster, the reduce into the root (one
ChainGather a slice, or with reduce root the root's pulls and adds) and the root's stores of Y (README.md, PER CLUSTER
and THE REDUCE)."""

import chiplet_mc as cm
import moe_prefill_data as md
from bingo_kernel_args import (SnaxBingoKernelIdma1dCopyArgs, SnaxBingoKernelMoeDispatchArgs,
                               SnaxBingoKernelSimdAddF16Args, SnaxBingoKernelSimdMulTilesArgs,
                               SnaxBingoKernelSimdScatterAddRowsArgs, SnaxBingoKernelXdmaChainGatherArgs)

from .common import BLK, DTMP_BUFS, EW_F16_ADD, NOUT


class Down:

    def _store(self, c, jl, red):
        """The root's store of cluster c's reduced slice jl from p1c to L3, after its chain (Y lies over X's source
        and the root's X landing, so the first store follows the other root cluster's last read of X)."""
        R = self.ROOT
        s2 = jl % 2
        over = [self._root_gat[1 - c]] if jl == 0 else []
        st = self.node(R, c, self.roles["dm"], f"y_store_c{c}j{jl}", "__snax_bingo_kernel_idma_1d_copy",
                       SnaxBingoKernelIdma1dCopyArgs(self.p1c[c].view(s2 * self.SL), self.ysym(self.y_off(c, jl)),
                                                     self.SL),
                       [red] + over)
        self._p1_reader[c][s2] = st
        self._stores.append(st)

    def _down_cl(self, cl, jl, so, last_add):
        """Cluster cl's down projection of output slice jl into its output slot so: the shared MLP's down GEMM and its
        dequantisation (and on the residual's chip the residual), then each expert's down GEMM, dequantisation and
        scatter-add into the slice (last_add[cl])."""
        d, T = self.d, self.T
        simd = self.roles["simd"]
        k, c = self.chip(cl), self.local(cl)
        bg = self.B[cl]
        other = self.clusters(k)[1 - c]
        nt = self.SW // BLK
        y = self.xch[(k, c)].view(so * self.SL)
        jj = self._jbase[cl] + jl * (1 + self.EPC)
        xr = self._xch_reader[cl][so]
        ld, si, slab, sc = self._load(cl, jj)
        dep_sh = [self._qa[(cl, "sh")]] if (cl, "sh") in self._qa else []
        dep_sh += [self._ex_copy[(other, "sh")]] if (other, "sh") in self._ex_copy else []
        gm = self.gemm(k, c, f"sd_j{jl}", bg["a8sh"], self.PS, slab, y, T // 8, BLK * self.shk[k] // 8,
                       nt, d.k_dn_sh, [ld] + dep_sh + ([xr] if xr else []))
        prev = self.node(k, c, simd, f"sd_deq_j{jl}", "__snax_bingo_kernel_simd_mul_tiles",
                         SnaxBingoKernelSimdMulTilesArgs(y, sc.addr, y, T // 8, nt), [gm] + sc.deps)
        self._scale_read(cl, sc, prev)
        self._release(cl, si, gm, prev, sc)
        if self.resid is not None and k == self.resid["chip"]:
            prev = self._resid_add(cl, jl, y, prev)
        caps = [md.pad8(d.cnt[x]) for x in self.held[k]]
        for s, e in enumerate(self.held[k]):
            rows = md.pad8(d.cnt[e])
            ld, si, slab, sc = self._load(cl, jj + 1 + s)
            self._last_ld[cl] = ld
            di = self._ndt_used[cl] % DTMP_BUFS
            self._ndt_used[cl] += 1
            dt = bg["dtmp"][di]
            dr = self._dt_reader[cl][di]
            gm = self.gemm(k, c, f"dn_j{jl}s{s}", bg["a8e"][s], self.PE, slab, dt, rows // 8, self.IE // 8,
                           nt, d.k_dn_e, [ld, self._qa[(cl, s)], self._ex_copy[(other, s)]] + ([dr] if dr else []))
            ids = bg["lists"].view(SnaxBingoKernelMoeDispatchArgs.rows_offset(caps, s))
            mt = self.node(k, c, simd, f"dn_deq_j{jl}s{s}", "__snax_bingo_kernel_simd_mul_tiles",
                           SnaxBingoKernelSimdMulTilesArgs(dt, sc.addr, dt, rows // 8, nt), [gm] + sc.deps)
            self._scale_read(cl, sc, mt)
            self._release(cl, si, gm, mt, sc)
            prev = self.node(k, c, simd, f"scat_j{jl}s{s}", "__snax_bingo_kernel_simd_scatter_add_rows",
                             SnaxBingoKernelSimdScatterAddRowsArgs(y, dt, nt, ids, int(d.cnt[e]), y_bytes=self.SL),
                             [mt, prev])
            self._dt_reader[cl][di] = prev
        last_add[cl] = prev

    def _fold_store(self, c, jl, acc, last):
        """reduce root: root cluster c's folded slice jl from its own output slot into L3 (Y lies over X's landing, so
        the first store follows the other root cluster's last read of X)."""
        over = [self._root_gat[1 - c]] if jl == 0 else []
        self._stores.append(self.node(self.ROOT, c, self.roles["dm"], f"y_store_c{c}j{jl}",
                                      "__snax_bingo_kernel_idma_1d_copy",
                                      SnaxBingoKernelIdma1dCopyArgs(acc, self.ysym(self.y_off(c, jl)), self.SL),
                                      [last] + over))

    def _down_per_kernel(self):
        """params barrier_at kernel: K4 and K5 in turns -- the down projection of as many slices as the output slots
        hold, then their reduce into the root and its stores -- each ending at a barrier. reduce chain: the ChainGathers
        one at a time into the root's two p1c slots, each slice stored after its chain. reduce root: each root cluster
        pulls every other chip's partial of its slice (one pull in flight, its two p1c slots) and its SIMD adds it into
        the cluster's own, in fold_order_c, then stores the slice."""
        R = self.ROOT
        dm, simd, xdma = self.roles["dm"], self.roles["simd"], self.roles["xdma"]
        pull_last, add_hist = {c: None for c in range(self.NC)}, {c: [] for c in range(self.NC)}
        for j0 in range(0, self.NSLC, NOUT):
            grp = list(range(j0, min(j0 + NOUT, self.NSLC)))
            for jl in grp:
                last_add = {}
                for cl in range(self.N_CL):
                    self._down_cl(cl, jl, jl % NOUT, last_add)
            self.barrier({k: [] for k in self.chips})
            for jl in grp:
                so, s2 = jl % NOUT, jl % 2
                for c in (list(range(self.NC))[::-1] if jl % 2 else list(range(self.NC))):
                    if self.reduce_impl == "chain":
                        p1x = f"((uint64_t)({self.p1c[c].get_c_var_name()}))"
                        ga = SnaxBingoKernelXdmaChainGatherArgs.fold_into(
                            cm.remote, R, c, self.chain_order(c)[:-1], src=self.xch_addr(c, so * self.SL),
                            dst=f"({p1x} + {s2 * self.SL})", size=self.SL, junction=self.jct["Elementwise"],
                            jct_csr0=EW_F16_ADD, holds=[self.p1c[c]])
                        p1 = [self._p1_reader[c][s2]] if self._p1_reader[c][s2] is not None else []
                        red = self.chain_gather(R, c, f"y_chain_c{c}j{jl}", ga,
                                        ([self._last_any] if self._last_any is not None else []) + p1)
                        self._last_red[c] = self._last_any = red
                        self._store(c, jl, red)
                        continue
                    acc = self.xch[(R, c)].view(so * self.SL)
                    for n_, k in enumerate(self.fold_order_c(c)[1:]):
                        q = len(add_hist[c])
                        slot = self.p1c[c].view((q % 2) * self.SL)
                        # one pull in flight; the slot free once the add two pulls back has read it
                        pl = self.node(R, c, dm, f"y_pull_c{c}j{jl}n{n_}", "__snax_bingo_kernel_idma_1d_copy",
                                       SnaxBingoKernelIdma1dCopyArgs(cm.remote(k, self.xch_addr(c, so * self.SL)),
                                                                     slot, self.SL),
                                       ([pull_last[c]] if pull_last[c] is not None else []) + add_hist[c][q - 2:q - 1])
                        pull_last[c] = pl
                        add_hist[c].append(self.node(R, c, simd, f"y_add_c{c}j{jl}n{n_}",
                                                     "__snax_bingo_kernel_simd_stream_elementwise",
                                                     SnaxBingoKernelSimdAddF16Args(acc, slot, acc, self.SL // 64, 32),
                                                     [pl] + add_hist[c][-1:]))
                    self._fold_store(c, jl, acc, add_hist[c][-1])
            if grp[-1] + 1 < self.NSLC:
                self.barrier({k: [] for k in self.chips})

    def _down_tuned(self):
        """The tuned schedule: slice by slice, every cluster's down projection into its output slot, then per root
        cluster the slice's ChainGather (the clusters' order alternating slice by slice), each after the previous
        gather of either cluster, the last slice's two after their own cluster's previous one, each stored by the root.
        barrier_at cross: a barrier before each slice's reduce, which the chains wait for instead of every chip's last
        adds. Every cluster's last scatter-add also waits for the other cluster's last weight load, so a last-slice
        chain starts only once every push has landed."""
        R, xdma = self.ROOT, self.roles["xdma"]
        freed = []                                          # barrier_at cross: (cluster, slot) the next barrier frees
        last_add = {}
        for jl in range(self.NSLC):
            so = jl % NOUT
            last_add = {}
            for cl in range(self.N_CL):
                self._down_cl(cl, jl, so, last_add)
            s2 = jl % 2
            tail = jl >= self.NSLC - 1
            corder = list(range(self.NC))[::-1] if jl % 2 else list(range(self.NC))
            bar = None
            if self.cross:
                # params barrier_at cross: the slice's barrier takes every chip's last adds of the slice (before slice 0
                # also its A8 copies, which no chain may meet) and the root's reduces of the slice before; the output
                # slots those read are free once it is passed
                bar = self.barrier({k: [last_add[cl] for cl in self.clusters(k)]
                                    + ([t for (cl, _), t in self._ex_copy.items() if self.chip(cl) == k] if jl == 0
                                       else [])
                                    + ([self._last_red[c] for c in range(self.NC) if self._last_red[c] is not None]
                                       if k == R else [])
                                    for k in self.chips})
                for cl, o in freed:
                    self._xch_reader[cl][o] = bar[self.chip(cl)]
                freed = []
            for c in corder:
                p1x = f"((uint64_t)({self.p1c[c].get_c_var_name()}))"
                ga = SnaxBingoKernelXdmaChainGatherArgs.fold_into(
                    cm.remote, R, c, self.chain_order(c)[:-1], src=self.xch_addr(c, so * self.SL),
                    dst=f"({p1x} + {s2 * self.SL})", size=self.SL, junction=self.jct["Elementwise"],
                    jct_csr0=EW_F16_ADD, holds=[self.p1c[c]])
                first = list(self._copies_in[c]) if jl == 0 and not self.cross else []
                before = self._last_red[c] if tail else self._last_any
                after = first + ([before] if before is not None else [])
                p1 = [self._p1_reader[c][s2]] if self._p1_reader[c][s2] is not None else []
                ins = [bar[R]] if self.cross else [last_add[cl] for cl in range(self.N_CL) if self.local(cl) == c]
                red = self.chain_gather(R, c, f"y_chain_c{c}j{jl}", ga,
                                ins + after + p1)
                for cl in range(self.N_CL):
                    if self.local(cl) == c:
                        if self.cross and self.chip(cl) != R:
                            freed.append((cl, so))
                        else:
                            self._xch_reader[cl][so] = red
                self._last_red[c] = self._last_any = red
                self._store(c, jl, red)
        # every push has landed before a last-slice chain
        for cl in range(self.N_CL):
            for o in self.clusters(self.chip(cl)):
                if o != cl:
                    self.edge(self._last_ld[o], last_add[cl])

    def _down(self):
        """The down phase (the args phase "MoE down" with params args_phases): the per-kernel or the tuned schedule."""
        for cl in range(self.N_CL):
            k, c = self.chip(cl), self.local(cl)
            (_, sn), ex = self.ch(k, c)
            self._jbase[cl] = len(self.chunks(sn)) + sum(len(self.chunks(ex[x][1])) for x in self.held[k])
        self._copies_in = {c: [self._ex_copy[(cl, key)] for (cl, key) in self._ex_copy if self.local(cl) != c]
                           for c in range(self.NC)}
        if self.args_phases:
            self.args_phase("MoE down")
        if self.per_kernel:
            self._down_per_kernel()
        else:
            self._down_tuned()
