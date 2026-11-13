# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""The decode's one pass: X, gate|up and the A8 as rows (rows.py), the down GEMVs into the output slots, one
ChainGather a group of slices into the root's fold slots, the token's row of every folded slice into Y_raw's row, and
Y into HBM."""

from types import SimpleNamespace

from bingo_kernel_args import (SnaxBingoKernelGemvArgs, SnaxBingoKernelIdma1dCopyArgs, SnaxBingoKernelSimdMulF16Args,
                               SnaxBingoKernelXdmaChainGatherArgs)
from ffn_prefill_mc import EW_F16_ADD

import chiplet_mc as cm


class DPasses:
    """DBuild._passes (PBuild.build makes the window and calls it) and y_out."""

    def _passes(self, go, after_x):
        """The one pass: X from HBM, the gate|up GEMVs and the post-processing (rows.py), then the down slices, each
        cluster's in output slots j mod out_slots, one ChainGather a group of chain_group slices, the token's row of each
        folded group into Y_raw's row (a gather late), and Y_raw into HBM."""
        R, h = self.ROOT, self.NDN // self.NC
        block0 = max((t.node_id for t in self.dfg.node_list), default=-1) + 1     # this block's first node id
        self.x_bc = self.x_hbm_bcasts(go)
        x = self.stage_x(0, self.x_bc[0], after_x)
        sw, _ = self.stage_gu(0, x)
        post = self.stage_post(0, sw)
        if self.chains_after_stream and self.out_slots < h:
            raise ValueError(f"params chains_after_stream: out_slots={self.out_slots} must hold the {h} slices a "
                             f"cluster (no output slot is read by a gather before the last down GEMV)")
        # what the down slices, the gathers and the stores share: the readers of each output slot (k, c, slot) and of
        # each root fold slot (c, s), the gathers waiting for their stores (a gather late), the last stores of each root
        # cluster, every store, the last gather
        rs = self._rs = SimpleNamespace(xch_reader={}, p1_reader={}, pending=[], last_st={}, all_st=[], last_red=None)
        self._row_downs(post)
        while rs.pending:
            self._row_store(*rs.pending.pop(0))
        # Y's row into HBM once every copy into it is done (every gather has ended by then)
        ready = list(rs.all_st)
        if self.y_dequant:                      # Y16 = Y_raw (.) s_dn, in place, the prefill's dequantise kernel
            ready = [self.node(R, 0, self.roles["simd"], "y_deq", "__snax_bingo_kernel_simd_stream_elementwise",
                               SnaxBingoKernelSimdMulF16Args(self.yrow, self.sdn, self.yrow, self.D // 32, 32), ready)]
        self.y_hbm = self.y_out(ready)
        self._check_gemv_only(block0)
        self.end_window([self.y_hbm])
        self._checks()

    def _row_loads(self, jl):
        """{cl: (load, slab index, slab)} of down slice jl."""
        return {cl: self._load(cl, 0, len(self.chunks[cl]) + jl) for cl in range(self.N_CL)}

    def _row_downs(self, post):
        """Per down slice every cluster's GEMV into its output slot, the next slice's loads, the stores a gather late,
        and per group of chain_group slices the group's gathers (params chains_after_stream: every group's gathers after
        the last slice, the first behind every cluster's last GEMV)."""
        rs, h, GS, SL = self._rs, self.NDN // self.NC, self.chain_group, self.SLICE
        lag = 1                                 # a gather's stores follow the next gather (two fold slots)
        other = lambda cl: self.clusters(self.chip(cl))[1 - self.local(cl)]     # noqa: E731
        ex_last = {cl: self.ex_push[(cl, 0)][-1] for cl in range(self.N_CL)}
        lds = self._row_loads(0)
        prev_dn, groups = {}, []
        for jl in range(h):
            so = jl % self.out_slots                # the clusters' output slot
            dn = {}
            for cl in range(self.N_CL):
                k, c = self.chip(cl), self.local(cl)
                ld, si, slab = lds[cl]
                deps = [ld, post[cl], ex_last[other(cl)]] + rs.xch_reader.get((k, c, so), [])
                dn[cl] = self.node(k, c, self.roles["gemm"], f"dn_c{c * h + jl}_i0", "__snax_bingo_kernel_gemv",
                                   SnaxBingoKernelGemvArgs(self.a8[cl][0], slab, self.xch[(k, c)].view(so * SL),
                                                           K=self.kcols[k][1], N=self.dn_sl[c][jl][1],
                                                           d_shift=self.d.k_dn, a_blk=8, w4=True), deps)
                self.ls[cl].release(si, dn[cl])
                if jl % GS:
                    # params chain_group: a gather waits for its group's last down GEMV on each chip only; the GEMM core
                    # runs its queue in order, so the group's GEMVs go into it in slice order
                    self.dfg.bingo_add_order_edge(prev_dn[cl], dn[cl])
                prev_dn[cl] = dn[cl]
            if jl + 1 < h:
                lds = self._row_loads(jl + 1)
            while len(rs.pending) > lag * self.NC:
                self._row_store(*rs.pending.pop(0))
            if (jl + 1) % GS:
                continue
            if self.chains_after_stream:
                groups.append((jl, dn))
                continue
            self._row_gathers(jl, dn, [])
        # chains_after_stream: the gathers in the loop's order, the first behind every cluster's last down GEMV
        for jl, dn in groups:
            while len(rs.pending) > lag * self.NC:
                self._row_store(*rs.pending.pop(0))
            self._row_gathers(jl, dn, [prev_dn[cl] for cl in range(self.N_CL)])

    def _row_gathers(self, jl, dn, last_dn):
        """The gathers of the group ending at slice jl: slices j0 .. jl, in output slots so0 .. so (back to back), into
        the root's fold slot s, one a root cluster; the first behind X's multicast and `last_dn`."""
        rs, h, GS, SL, R = self._rs, self.NDN // self.NC, self.chain_group, self.SLICE, self.ROOT
        others = self.order[1:]
        j0 = jl + 1 - GS
        s, so0 = (j0 // GS) % 2, j0 % self.out_slots
        for c in range(self.NC):
            j = c * h + j0
            # params chain_alt: cluster 1's gathers run the ring the other way
            ordr = others[::-1] if self.chain_alt and c == 1 else others
            p1x = f"((uint64_t)({self.p1c[c].get_c_var_name()}))"
            ga = SnaxBingoKernelXdmaChainGatherArgs.fold_into(
                cm.remote, R, c, ordr, src=self.xch_addr(c, so0 * SL), dst=f"({p1x} + {s * GS * SL})",
                size=(GS - 1) * SL + 2 * self.dn_sl[c][jl][1],
                junction=self.jct["Elementwise"], jct_csr0=EW_F16_ADD, holds=[self.p1c[c]])
            # the A8 copies into cluster c precede its gathers through its own down GEMVs
            first = [rs.last_red] if rs.last_red is not None else []
            if j0 == 0 and c == 0:
                first.append(self.done_of(self.x_bc[-1]))    # X's multicast before the first gather
                first += list(last_dn)
            # chains_after_stream: a gather after the first follows it (last_red), and so every down GEMV already
            ins = [] if self.chains_after_stream and rs.last_red is not None else \
                [dn[cl] for cl in range(self.N_CL) if self.local(cl) == c]
            red = self.chain_gather(R, c, f"y_chain_j{j}", ga,
                            ins + rs.p1_reader.get((c, s), []) + first)
            rs.last_red = red
            for k in self.chips:
                for o in range(so0, so0 + GS):
                    rs.xch_reader[(k, c, o)] = [red]
            rs.pending.append((c, j, red, s))

    def _row_store(self, c, j, red, s):
        """The token's row of slices j .. j + chain_group - 1 (one gather, slice n at n x SLICE of the root's fold slot
        s) into Y_raw's row at their columns: one copy when the slices fill their fold slots and their columns follow
        each other, else one a slice."""
        rs, h, GS, SL = self._rs, self.NDN // self.NC, self.chain_group, self.SLICE
        sl = [self.dn_sl[c][j - c * h + n] for n in range(GS)]
        base = s * GS * SL
        whole = all(2 * w == SL for _, w in sl[:-1]) and \
            all(sl[n][0] == sl[n - 1][0] + sl[n - 1][1] for n in range(1, GS))
        runs = [(0, sl[0][0], sum(w for _, w in sl))] if whole else \
            [(n * SL, o0, w) for n, (o0, w) in enumerate(sl)]
        sts = []
        for n, (off, o0, w) in enumerate(runs):
            sts.append(self.node(self.ROOT, c, self.roles["dm"], f"y_store_j{j}" if n == 0 else f"y_store_j{j}n{n}",
                                 "__snax_bingo_kernel_idma_1d_copy",
                                 SnaxBingoKernelIdma1dCopyArgs(self.p1c[c].view(base + off), self.yrow.view(2 * o0), 2 * w),
                                 [red]))
        rs.last_st[c] = sts
        rs.all_st.extend(sts)
        rs.p1_reader[(c, s)] = sts

    def y_out(self, ready):
        """The root's DM core writes the row (yrow, on the root's cluster 0) into the HBM of the root's memory chip behind
        `ready`. Returns the task the end follows. A whole layer (llama3/common/llama3_decode_layer_mc.py) puts its final
        residual here instead."""
        return self.node(self.ROOT, 0, self.roles["dm"], "y_hbm", "__snax_bingo_kernel_idma_1d_copy",
                         SnaxBingoKernelIdma1dCopyArgs(self.yrow, self.h_yhbm, 2 * self.D), ready)
