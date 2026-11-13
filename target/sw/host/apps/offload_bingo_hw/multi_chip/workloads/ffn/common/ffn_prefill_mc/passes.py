# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""The window and the passes (sync edge): per pass X, gate|up, the A8, then the down slices into the output slots, one
ChainGather a group of slices into the root's fold slots, and the stores of Y_raw."""

from types import SimpleNamespace

from bingo_kernel_args import (SnaxBingoKernelGemmFullArgs, SnaxBingoKernelIdma1dCopyArgs,
                               SnaxBingoKernelIdmaPutFlaggedArgs, SnaxBingoKernelWaitFlagsArgs,
                               SnaxBingoKernelXdmaChainGatherArgs)
from bingo_mem_handle import BingoMemSymbol
from libs.blocks import chain_gather

import chiplet_mc as cm

from .consts import BLK, EW_F16_ADD, TILE


class PPasses:
    """PBuild.build: the window's opening and the passes, one after the other."""

    # ---- the window -----------------------------------------------------------------------------
    def pre_window(self):
        """{chip: [task]}: what the start waits for: the dequantisation factors into L1. params start_per_chip: a chip's
        loads one after the other, cluster 0's last, so the start waits for one task a chip (on the root, the load ahead
        of it on its own core) instead of one a cluster."""
        dm = self.roles["dm"]
        per_chip = self.start_per_chip
        pre = {k: [] for k in self.chips}
        by_chip = lambda cl: (self.chip(cl), -self.local(cl))     # noqa: E731
        for cl in sorted(range(self.N_CL), key=by_chip) if per_chip else range(self.N_CL):
            k, c = self.chip(cl), self.local(cl)
            h, nb = self.sdup[cl]
            t = self.node(k, c, dm, "sgu_load", "__snax_bingo_kernel_idma_1d_copy",
                          SnaxBingoKernelIdma1dCopyArgs(h.address, self.sbuf[cl], nb), pre[k][-1:] if per_chip else [])
            pre[k] = [t] if per_chip else pre[k] + [t]
        return pre

    def pass_start(self, i):
        """Where the tasks of pass i (i >= 1) begin, in the order they are made: nothing here; a layer starts an args
        phase (llama3/full_layer_prefill/common/llama3_layer_mc.py)."""

    def build(self):
        self.slot = {}
        self.gu_count = {cl: 0 for cl in range(self.N_CL)}
        self.gb_reader = {cl: [[], []] for cl in range(self.N_CL)}
        self.a8_reader = {cl: [[], []] for cl in range(self.N_CL)}
        # A16 (D layout) and A16a (A layout) are one buffer each: a pass's first SwiGLU waits for the previous pass's
        # last reshape, its first reshape for the previous pass's last quantise. d2a[(cl, i)], quant[(cl, i)]: the tasks
        self.d2a, self.quant, self.last_swi = {}, {}, {}
        self.a16_writers = {(cl, i): {} for cl in range(self.N_CL) for i in range(self.NI)}  # row tile -> its SwiGLUs
        self.q_all = {}                         # (cl, i) -> row tile -> its quantise
        self.post_made = set()                  # (cl, i) whose post blocks the split last chunk made
        self.ex_push = {}                       # {(cl, i): its A8 runs copied to the other cluster}
        self.push_war = {}                      # {cl: what its first A8 copy of the pass follows}
        self.q_tile = {}                        # {cl: {row tile: its quantise}}
        # dep-tag relays of xDMA tasks (the gathers) go to the SIMD cores first, idle in the down phase, not behind the DM
        # cores' blocking ring loads or the GEMM cores' long GEMMs; each serves that one task, so a gather waits for its
        # own slice's GEMMs only, not for the next gather's too (one relay would hold both)
        self.dfg.relay_prefer_cores = {self.roles["xdma"]: [self.roles["simd"]]}
        self.dfg.relay_one_consumer_cores = [self.roles["xdma"]]
        if self.gather_relay_only:
            self.dfg.relay_not_order_cores = [self.roles["xdma"]]
        self.dn_first_load = {}                 # {cl: the first down chunk's load, made with the gate|up loads}
        # the exchange buffers are not zeroed before the start: every slot is written whole by its down GEMMs before it
        # is read, and an unzeroed buffer is live only from then on, so the static L1 plan can lay it over X
        self.xch_zero = False
        go = self.setup(extra=self.pre_window())
        self._passes(go, {cl: [go[self.chip(cl)]] for cl in range(self.N_CL)})
        if self.NI > 1 and not self.restream:
            # a pass after the first copies every weight chunk from the ring slot it landed in, so no slot may be pushed
            # twice: each ring holds all of its cluster's chunks
            for k in self.chips:
                for r, e in enumerate(self.rings[k].entries[:self.NC]):
                    if len(e) > self.n_slots:
                        raise ValueError(f"params weight_ring slots={self.n_slots}: chip {k:02x} ring {r} takes {len(e)} "
                                         f"chunks, and the passes after the first read each from its slot")

    # ---- the passes -----------------------------------------------------------------------------
    def _passes(self, go, after_x):
        """The passes, one after the other, each the single pass over its rows with the down projection split by output
        columns. Each cluster's down GEMMs read the chip's whole A8 (K = the chip's columns) for its half of the
        outputs, one slice at a time into an output slot of its exchange buffer, so the slot holds the chip's sum and
        the root's cluster c gathers that slot of every chip's cluster c straight into its fold buffer and stores it. The
        gathers run one at a time, alternating between the two clusters' xDMAs.

        Pass 0 loads the weights from the ring; a later pass copies every chunk from the L3 slot it landed in (or with
        params restream loads it from the ring again). X_(i+1) comes into L1 after the last gate|up GEMM that reads the
        buffer. A pass's first A8 copy follows the previous pass's last gather (no copy into a cluster while a gather
        passes through it) and the receiving cluster's last down GEMM (its A8); its first down GEMMs follow the previous
        pass's gathers of their slots; the stores keep their lag across passes. Every pass's X is multicast at the
        start, and the first gather follows it, so no multicast meets a gather. sync barrier: kernel.py."""
        if self.per_kernel:
            return self._passes_kernel(go, after_x)
        self.x_bc = self.x_hbm_bcasts(go)
        x = self.first_x(after_x)
        # what the down slices, the gathers and the stores share across slices and passes: the readers of each output
        # slot (k, c, slot) and of each root fold slot (c, s), the gathers waiting for their stores (a gather late), the
        # last stores of each root cluster, y_scatter's puts by owner chip, the last gather
        ps = self._ps = SimpleNamespace(xch_reader={}, p1_reader={}, stores=[], pending=[], last_st={}, yt_puts={},
                                        last_red=None)
        last_gu, last_dn = {}, {}
        for i in range(self.NI):
            if i:
                self.pass_start(i)
                # X_i into the buffer once the last gate|up GEMM that read it (pass i - 1) is done
                x = self.stage_x(i, self.x_bc[i], {cl: list(after_x[cl]) + last_gu[i - 1][cl] for cl in range(self.N_CL)})
                self.push_war = {cl: last_dn[self.clusters(self.chip(cl))[1 - self.local(cl)]] + [ps.last_red]
                                 for cl in range(self.N_CL)}
            sw, last_gu[i] = self.stage_gu(i, x)
            self.stage_post(i, sw)
            last_dn = self._down_pass(i)
            # the A8's readers of this pass: every down GEMM, and the copies of this cluster's runs out of it
            for cl in range(self.N_CL):
                self.a8_reader[cl] = [last_dn[cl] + self.ex_push[(cl, i)]] * 2
        while ps.pending:
            self._store_group(*ps.pending.pop(0))
        self._end_passes()
        self._checks()

    def _dn_loads(self, i, jl):
        """{cl: (load, slab index, slab)} of down slice jl of pass i (the first slice's was made with the gate|up loads)."""
        if jl == 0 and self.dn_first_load:
            return dict(self.dn_first_load)
        return {cl: self._load(cl, i, len(self.chunks[cl]) + jl) for cl in range(self.N_CL)}

    def _a8_tile(self, i, cl, m):
        """The tasks after which row tile m of cluster cl's A8 (pass i) is whole: its quantise, the other cluster's copy."""
        return [self.q_tile[cl][m], self.ex_push[(self.clusters(self.chip(cl))[1 - self.local(cl)], i)][m]]

    def _a8_all(self, i, cl):
        """The chip's whole A8 (pass i): every quantise of this cluster's columns, every copy of the other cluster's."""
        return list(self.q_all[(cl, i)].values()) + self.ex_push[(self.clusters(self.chip(cl))[1 - self.local(cl)], i)]

    def _down_pass(self, i):
        """Pass i's down slices: per slice every cluster's GEMM(s) into its output slot, the next slice's loads, the
        stores a gather late, and per group of chain_group slices the group's gathers. Returns {cl: its down GEMMs}."""
        ps, h, GS = self._ps, self.NDN // self.NC, self.chain_group
        lag = 1                                 # a gather's stores follow the next gather (two fold slots)
        dn_pass = {cl: [] for cl in range(self.N_CL)}
        next_put = self.done_of(self.x_bc[-1]) if i == 0 else None      # every multicast before the first gather
        lds = self._dn_loads(i, 0)
        prev_dn = {}
        for jl in range(h):
            dn = {}
            for cl in range(self.N_CL):
                dn[cl] = self._down_gemm(i, jl, cl, lds[cl], dn_pass)
                if jl % GS:
                    # params chain_group: a gather waits for its group's last down GEMM on each chip only, so the
                    # group's GEMMs run in slice order
                    self.dfg.bingo_add_order_edge(prev_dn[cl], dn[cl])
                prev_dn[cl] = dn[cl]
            if jl + 1 < h:
                lds = self._dn_loads(i, jl + 1)
            while len(ps.pending) > lag * self.NC:
                self._store_group(*ps.pending.pop(0))
            if (jl + 1) % GS:
                continue
            self._gathers(i, jl, dn, next_put)
        return dn_pass

    def _down_gemm(self, i, jl, cl, load, dn_pass):
        """Cluster cl's down slice jl of pass i into its output slot (slice jl mod out_slots), after the slot's last
        gather; params split_first_dn: the first slice as one GEMM per post block, each behind that block's A8 rows.
        Returns the slice's last GEMM (the slab is released behind it)."""
        k, c = self.chip(cl), self.local(cl)
        ld, si, slab = load
        so, SL = jl % self.out_slots, self.SLICE
        kn = self.kcols[k][1]
        nt = self.dn_sl[c][jl][1] // BLK        # this slice's column tiles
        reader = self._ps.xch_reader.get((k, c, so), [])
        kargs = dict(input_B_addr=slab, input_C_addr=0, K=kn // 4, N=nt, array_shape_idx=0, transpose_A=0,
                     transpose_B=0, accumPrevC=0, int32tofp16_enable=1, int4_b_enable=self.int4_b, d_shift=self.d.k_dn)
        h = self.NDN // self.NC
        if self.split_first_dn and jl == 0:     # one GEMM per post block, behind that block's A8 rows only
            mb, dn = self.MT // self.post_blocks, None
            for b in range(self.post_blocks):
                m1 = (b + 1) * mb - 1           # the block's last row tile
                blk = self.gemm_node(k, c, f"dn_c{c * h}_i{i}b{b}",
                                     SnaxBingoKernelGemmFullArgs(
                                         input_A_addr=self.a8[cl][0].view(b * mb * 16 * kn),
                                         output_D_addr=self.xch[(k, c)].view(so * SL + b * mb * nt * TILE), M=mb, **kargs),
                                     [ld] + [t for m in range(b * mb, m1 + 1) for t in self._a8_tile(i, cl, m)] + reader)
                if b:
                    # the blocks in row order, so the slice's last block is after all of them (its slab release, the
                    # gather of the slice)
                    self.dfg.bingo_add_order_edge(dn, blk)
                dn = blk
                dn_pass[cl].append(blk)
        else:
            dn = self.gemm_node(k, c, f"dn_c{c * h + jl}_i{i}",
                                SnaxBingoKernelGemmFullArgs(input_A_addr=self.a8[cl][0],
                                                            output_D_addr=self.xch[(k, c)].view(so * SL),
                                                            M=self.MT, **kargs),
                                [ld] + self._a8_all(i, cl) + reader)
            dn_pass[cl].append(dn)
        self.ls[cl].release(si, dn)
        return dn

    # ---- the reduce -------------------------------------------------------------------------------
    def gather_node(self, c, name, args, deps):
        """One ChainGather into the root's cluster c, on its xDMA core; returns the task."""
        return chain_gather(lambda n, kernel, a, d: self.node(self.ROOT, c, self.roles["xdma"], n, kernel, a, d),
                            name, args, deps)

    def _gathers(self, i, jl, dn, next_put):
        """The gathers of the group ending at slice jl of pass i: slices j0 .. jl, in output slots so0 .. so (back to
        back), into the root's fold slot s (chain_group slices), one a root cluster; the first of the pass's first group
        after `next_put` (the last X multicast)."""
        ps, GS, SL, R = self._ps, self.chain_group, self.SLICE, self.ROOT
        h, others = self.NDN // self.NC, self.order[1:]
        j0 = jl + 1 - GS
        s, so0 = (j0 // GS) % 2, j0 % self.out_slots
        for c in range(self.NC):
            j = c * h + j0
            # params chain_alt: cluster 1's gathers run the ring the other way, so each gather's data follows the
            # previous gather's finish back along the links instead of meeting it
            ordr = others[::-1] if self.chain_alt and c == 1 else others
            p1x = f"((uint64_t)({self.p1c[c].get_c_var_name()}))"
            ga = SnaxBingoKernelXdmaChainGatherArgs.fold_into(
                cm.remote, R, c, ordr, src=self.xch_addr(c, so0 * SL), dst=f"({p1x} + {s * GS * SL})",
                size=(GS - 1) * SL + self.ROWS * self.dn_sl[c][jl][1] * 2,
                junction=self.jct["Elementwise"], jct_csr0=EW_F16_ADD, holds=[self.p1c[c]])
            # the A8 copies into cluster c precede its gathers through its own down GEMMs
            first = [ps.last_red] if ps.last_red is not None else []
            if next_put is not None and j0 == 0 and c == 0:
                first.append(next_put)
            ins = [dn[cl] for cl in range(self.N_CL) if self.local(cl) == c]
            red = self.gather_node(c, f"y_chain_j{j}" if i == 0 else f"y_chain_i{i}j{j}", ga,
                                   ins + ps.p1_reader.get((c, s), []) + first)
            ps.last_red = red
            for k in self.chips:
                for o in range(so0, so0 + GS):
                    ps.xch_reader[(k, c, o)] = [red]
            ps.pending.append((c, i, j, red, s))

    # ---- Y_raw ----------------------------------------------------------------------------------
    def _store_group(self, c, i, j, red, s):
        """The stores of gather `red` (slices j .. j + chain_group - 1 of root cluster c, slice n at n x SLICE of the
        root's fold slot s): into Y_raw in the root's L3 (_y_stores), or with y_scatter row tile by row tile into the
        tiles' owners."""
        ps, GS, SL, R, dm = self._ps, self.chain_group, self.SLICE, self.ROOT, self.roles["dm"]
        h = self.NDN // self.NC
        sl = [self.dn_sl[c][j - c * h + n] for n in range(GS)]
        base = s * GS * SL
        if self.y_scatter:
            # the slice's row tiles (D layout: each 16 rows x w contiguous) into their owners' tiles, at the slice's
            # columns, each behind slice j's flag of that tile
            puts = []
            for n, (o0, w) in enumerate(sl):
                jn = j + n
                for m in range(self.ROWS // 16):
                    q = i * (self.ROWS // 16) + m
                    own, lt = self.y_owner(q)
                    far = None if own == R else own
                    sfx = f"j{jn}m{m}" if i == 0 else f"i{i}j{jn}m{m}"
                    puts.append(self.node(R, c, dm, f"y_sc_{sfx}", "__snax_bingo_kernel_idma_put_flagged",
                                          SnaxBingoKernelIdmaPutFlaggedArgs(
                                              self.p1c[c].view(base + n * SL + m * 16 * w * 2),
                                              self.yt_sym(q, o0 * 16 * 2, chip_id=far), 16 * w * 2,
                                              BingoMemSymbol(f"{self.TAG}_ytflag", offset=(lt * self.NDN + jn) * 64,
                                                             chip_id=far), value=1), [red]))
                    ps.yt_puts.setdefault(own, []).append(puts[-1])
            ps.stores.extend(puts)
            ps.p1_reader[(c, s)] = puts
            return
        sts = self._y_stores(c, i, j, red, s)
        ps.stores.extend(sts)
        ps.last_st[c] = sts
        ps.p1_reader[(c, s)] = sts

    def _end_passes(self):
        """The window's end: behind each root cluster's last stores, or with y_scatter behind one wait a chip on its
        tiles' flags (each chip's tiles of Y_raw have landed)."""
        ps, dm = self._ps, self.roles["dm"]
        if not self.y_scatter:
            self.end_window([t for c in sorted(ps.last_st) for t in ps.last_st[c]])   # each root cluster's last stores
            return
        waits = []
        for k in self.chips:
            puts = ps.yt_puts.get(k, [])
            w = self.node(k, 0, dm, "yt_wait", "__snax_bingo_kernel_wait_flags",
                          SnaxBingoKernelWaitFlagsArgs(BingoMemSymbol(f"{self.TAG}_ytflag"), len(puts), 1),
                          [t for t in puts if t.assigned_chiplet_id == k])
            for t in puts:
                if t.assigned_chiplet_id != k:
                    self.dfg.bingo_add_order_edge(t, w)
            waits.append(w)
        self.end_window(waits)

    def _y_store(self, c, i, j, src, after):
        """Slice j of Y_raw (in root cluster c's half of the outputs) from the root's L1 `src` into its L3."""
        o0, w = self.dn_sl[c][j - c * (self.NDN // self.NC)]
        return self.node(self.ROOT, c, self.roles["dm"], f"y_store_j{j}" if i == 0 else f"y_store_i{i}j{j}",
                         "__snax_bingo_kernel_idma_1d_copy",
                         SnaxBingoKernelIdma1dCopyArgs(src, self._ysym(i * self.YB + o0 * self.ROWS * 2),
                                                       self.ROWS * w * 2), list(after))

    def _y_stores(self, c, i, j, red, s):
        """The gather `red` of slices j .. j + chain_group - 1 of root cluster c, in its fold slot s, into Y_raw in L3: one
        copy when the slices fill their fold slots and their columns follow each other (Y_raw_i in L3 is slice by slice, a
        slice's ROWS x w at its first column x ROWS); else one copy a slice. Returns the copies."""
        GS, SL, h = self.chain_group, self.SLICE, self.NDN // self.NC
        sl = [self.dn_sl[c][j - c * h + n] for n in range(GS)]
        whole = all(self.ROWS * w * 2 == SL for _, w in sl[:-1]) and \
            all(sl[n][0] == sl[n - 1][0] + sl[n - 1][1] for n in range(1, GS))
        runs = [(0, sl[0][0], (GS - 1) * SL + self.ROWS * sl[-1][1] * 2)] if whole else \
            [(n * SL, o0, self.ROWS * w * 2) for n, (o0, w) in enumerate(sl)]
        sts = []
        for n, (off, o0, nb) in enumerate(runs):
            sfx = f"j{j}" if n == 0 else f"j{j}n{n}"
            sts.append(self.node(self.ROOT, c, self.roles["dm"], f"y_store_{sfx}" if i == 0 else f"y_store_i{i}{sfx}",
                                 "__snax_bingo_kernel_idma_1d_copy",
                                 SnaxBingoKernelIdma1dCopyArgs(self.p1c[c].view(s * GS * SL + off),
                                                               self._ysym(i * self.YB + o0 * self.ROWS * 2), nb),
                                 [red]))
        return sts
