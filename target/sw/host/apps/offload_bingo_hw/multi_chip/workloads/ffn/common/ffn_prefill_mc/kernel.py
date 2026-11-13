# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""params sync barrier (the ablation's base, bcast and bcast_reduce): the passes kernel by kernel, a chip barrier after
every kernel, and the reduce of a pass as one kernel (ChainGathers, or with reduce root the root's pulls and SIMD adds)."""

from bingo_kernel_args import (SnaxBingoKernelGemmFullArgs, SnaxBingoKernelIdma1dCopyArgs,
                               SnaxBingoKernelSimdAddF16Args, SnaxBingoKernelXdmaChainGatherArgs)

import chiplet_mc as cm

from .consts import BLK, EW_F16_ADD


class PKernel:
    """The passes as a conventional system runs them (sync barrier)."""

    def out_view(self, cl, so):
        """Cluster cl's output slot so: the exchange buffer's slots, then (sync barrier) the X buffer's, which follows it
        at the same offset on every cluster, so self.xch_addr(c, so * SLICE) names the slot on every chip."""
        k, c = self.chip(cl), self.local(cl)
        if so < self.xch_slots:
            return self.xch[(k, c)].view(so * self.SLICE)
        return self.xbuf[cl][0].view((so - self.xch_slots) * self.SLICE)

    def _passes_kernel(self, go, after_x):
        """params sync barrier: the passes kernel by kernel, as a conventional system runs them, a chip barrier after
        every kernel fencing every core of the chip (fence_barriers), so no two kernels overlap and nothing crosses chips
        but at a barrier. Per pass:
            K1  X into L1, then every gate|up chunk, its dequantise and SwiGLU the GEMM's epilogue
            K2  A16 reshaped and quantised into the chip's A8, each cluster's columns copied into the other's
            K3  every down slice, slice j into output slot j (out_view: the exchange buffer's two slots, then the X
                buffer's, which nothing reads until the next pass's X)
            K4  every slice reduced into the root (_reduce_kernel) and stored
        The chunks, slices, buffers and kernels of the edge-synchronised passes (passes.py); only the order and the
        synchronisation differ."""
        h = self.NDN // self.NC
        allk = lambda: {k: [] for k in self.chips}     # noqa: E731
        self.x_bc = self.x_hbm_bcasts(go)
        x = self.first_x(after_x)
        last_dn, last_st = {}, []
        for i in range(self.NI):
            if i:
                self.pass_start(i)
                x = self.stage_x(i, self.x_bc[i], after_x)
                self.push_war = {cl: last_dn[self.clusters(self.chip(cl))[1 - self.local(cl)]]
                                 for cl in range(self.N_CL)}
            sw, _ = self.stage_gu(i, x)                                 # K1
            self.barrier(allk())
            post = self.stage_post(i, sw)                               # K2
            ex_last = {cl: self.ex_push[(cl, i)][-1] for cl in range(self.N_CL)}
            self.barrier(allk())
            dn, dn_pass = {}, {cl: [] for cl in range(self.N_CL)}       # K3
            lds = {cl: self._load(cl, i, len(self.chunks[cl])) for cl in range(self.N_CL)}
            for jl in range(h):
                for cl in range(self.N_CL):
                    k, c = self.chip(cl), self.local(cl)
                    ld, si, slab = lds[cl]
                    dn[cl] = self.gemm_node(k, c, f"dn_c{c * h + jl}_i{i}",
                                           SnaxBingoKernelGemmFullArgs(
                                               input_A_addr=self.a8[cl][0], output_D_addr=self.out_view(cl, jl),
                                               M=self.MT, input_B_addr=slab, input_C_addr=0, K=self.kcols[k][1] // 4,
                                               N=self.dn_sl[c][jl][1] // BLK, array_shape_idx=0, transpose_A=0,
                                               transpose_B=0, accumPrevC=0, int32tofp16_enable=1, int4_b_enable=self.int4_b,
                                               d_shift=self.d.k_dn),
                                           [ld, post[cl], ex_last[self.clusters(k)[1 - c]]])
                    dn_pass[cl].append(dn[cl])
                    self.ls[cl].release(si, dn[cl])
                if jl + 1 < h:
                    lds = {cl: self._load(cl, i, len(self.chunks[cl]) + jl + 1) for cl in range(self.N_CL)}
            last_dn = dn_pass
            for cl in range(self.N_CL):
                self.a8_reader[cl] = [dn_pass[cl] + self.ex_push[(cl, i)]] * 2
            self.barrier(allk())
            last_st = self._reduce_kernel(i)                            # K4
            if i + 1 < self.NI:
                self.barrier(allk())
        self.end_window(last_st)
        self._checks()

    def _reduce_kernel(self, i):
        """K4 of pass i (sync barrier): every slice of both halves of the outputs reduced into the root and stored to
        Y_raw in its L3, slice by slice. reduce chain: the ChainGathers of the edge-synchronised passes, one at a time,
        into the root's two fold slots, each stored a gather later. reduce root: each root cluster pulls every other
        chip's partial of its half (one pull in flight, two L1 slots) and its SIMD adds it into the cluster's own, in the
        fold order, then stores the slice. Returns every store (the checks read every slice)."""
        R, dm, simd = self.ROOT, self.roles["dm"], self.roles["simd"]
        SL, others, h = self.SLICE, self.order[1:], self.NDN // self.NC
        last, stores = {}, []
        if self.reduce_impl == "chain":
            p1x = {c: f"((uint64_t)({self.p1c[c].get_c_var_name()}))" for c in range(self.NC)}
            lag = 1
            pending, p1_reader, last_red = [], {}, None
            GS = self.chain_group               # params chain_group: a gather carries this many slices
            for jl in range(0, h, GS):
                s = jl // GS % 2
                while len(pending) > lag * self.NC:
                    c, j, red, s0 = pending.pop(0)
                    p1_reader[(c, s0)] = last[c] = self._y_stores(c, i, j, red, s0)
                    stores += last[c]
                for c in range(self.NC):
                    j = c * h + jl
                    ordr = others[::-1] if self.chain_alt and c == 1 else others
                    ga = SnaxBingoKernelXdmaChainGatherArgs.fold_into(
                        cm.remote, R, c, ordr, src=self.xch_addr(c, jl * SL), dst=f"({p1x[c]} + {s * GS * SL})",
                        size=(GS - 1) * SL + self.ROWS * self.dn_sl[c][jl + GS - 1][1] * 2,
                        junction=self.jct["Elementwise"], jct_csr0=EW_F16_ADD,
                        # the root's own output slots past the exchange buffer sit in its X buffer, which the gather
                        # reads through an address expression the static planner does not see
                        holds=[self.p1c[c], self.xbuf[self.clusters(R)[c]][0]])
                    last_red = self.gather_node(c, f"y_chain_j{j}" if i == 0 else f"y_chain_i{i}j{j}", ga,
                                                p1_reader.get((c, s), []) + ([last_red] if last_red is not None else []))
                    pending.append((c, j, last_red, s))
            for c, j, red, s0 in pending:
                stores += self._y_stores(c, i, j, red, s0)
            return stores
        for c in range(self.NC):
            cl = self.clusters(R)[c]
            pulls, adds = [], []
            for jl in range(h):
                j = c * h + jl
                nb = self.ROWS * self.dn_sl[c][jl][1] * 2
                acc = self.out_view(cl, jl)         # the root's own partial, the others folded into it
                for n_, k in enumerate(others):
                    q = len(pulls)
                    slot = self.p1c[c].view((q % 2) * SL)
                    sfx = f"j{j}n{n_}" if i == 0 else f"i{i}j{j}n{n_}"
                    pulls.append(self.node(R, c, dm, f"y_pull_{sfx}", "__snax_bingo_kernel_idma_1d_copy",
                                           SnaxBingoKernelIdma1dCopyArgs(cm.remote(k, self.xch_addr(c, jl * SL)), slot, nb),
                                           pulls[-1:] + adds[q - 2:q - 1]))
                    adds.append(self.node(R, c, simd, f"y_add_{sfx}", "__snax_bingo_kernel_simd_stream_elementwise",
                                          SnaxBingoKernelSimdAddF16Args(acc, slot, acc, nb // 64, 32),
                                          [pulls[-1]] + adds[-1:]))
                stores.append(self._y_store(c, i, j, acc, adds[-1:]))
        return stores
