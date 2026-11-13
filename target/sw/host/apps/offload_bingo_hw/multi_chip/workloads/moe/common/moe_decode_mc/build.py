# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""DBuild's task graph, phase by phase: X, the route and the rings' split at it, every piece's gate|up and quantise
with the A8 exchange, the shared and the routed pieces' down projection, the reduce and y (README.md, THE STEPS)."""

import chiplet_mc as cm
import moe_prefill_data as md
from bingo_kernel_args import (HostBingoKernelHbmBcastArgs, SnaxBingoKernelGemvArgs, SnaxBingoKernelIdma1dCopyArgs,
                               SnaxBingoKernelIdmaFetchFlaggedArgs, SnaxBingoKernelIdmaPutFlaggedArgs,
                               SnaxBingoKernelMoeDispatchArgs, SnaxBingoKernelMoeTopkRowsArgs,
                               SnaxBingoKernelSimdAddF16Args, SnaxBingoKernelSimdMulF16Args,
                               SnaxBingoKernelSimdQuantRowsArgs, SnaxBingoKernelSimdSiluF16F16Args,
                               SnaxBingoKernelSimdSoftmaxF16F16Args, SnaxBingoKernelWaitFlagsArgs,
                               SnaxBingoKernelXdmaChainGatherArgs, SnaxBingoKernelXdmaMemsetArgs)
from bingo_mem_handle import BingoMemSymbol
from libs import at_offset

from moe_prefill_mc.common import BLK, D_BLOCK, DTMP_BUFS, EW_F16_ADD, ROUTE_B

from .common import ROWS, SimdMulRowsF16Args, spread_args


class Build:

    # ---- helpers ------------------------------------------------------------------------------------------------------
    def _load(self, cl, j):
        """Weight chunk j of cluster cl from its ring into the next slab. Returns (load, slab index, slab, the chunk's
        factor row in the slab)."""
        k, c = self.chip(cl), self.local(cl)
        ls = self.ls[cl]
        off, nb, foff = self.chunk_off[cl][j]
        si, slab, war = ls.take(nb)
        ld, _ = ls.rings.load_chunk(
            lambda n, core, kern, args, after: self.node(k, c, core, n, kern, args, after), c,
            f"Ld_w{j}", src=self.img[cl].address + off, nbytes=nb, slab=slab, war=war, deps=(), dm=self.roles["dm"])
        return ld, si, slab, slab.view(foff)

    def _ld(self, cl, key):
        """The ring load of cluster cl's chunk `key`: loads are created in the image's order, so a ring's entries are
        its image's chunks and the route-free runs are the image's first chunks."""
        j = self.cidx[cl][key]
        k, c = self.chip(cl), self.local(cl)
        if len(self.rings[k].entries[c]) != j:
            raise RuntimeError(f"cluster {cl}: chunk {key} (image index {j}) loaded as ring entry "
                               f"{len(self.rings[k].entries[c])}")
        return self._load(cl, j)

    def gemv_task(self, k, c, name, a, b, dst, K, N, k_shift, deps, w4=True):
        """One GEMV on chip k's cluster c, a blocking task on its GEMM core: the a_row a [K] times b [K, N] (w4: the
        paired INT4 layout; else INT8, the B layout) into the FP16 row dst [N], at D-port shift k_shift."""
        return self.node(k, c, self.roles["gemm"], name, "__snax_bingo_kernel_gemv",
                         SnaxBingoKernelGemvArgs(a, b, dst, K=K, N=N, d_shift=k_shift, a_blk=8, w4=w4), list(deps))

    # ---- the window ---------------------------------------------------------------------------------------------------
    def _warm(self, pre):
        """A small GEMV and a stream multiply on every cluster (and the softmax on the root's cluster 0) before the
        start, so the first tasks find their code cached; the root's hold the start stamp."""
        R, dm, simd = self.ROOT, self.roles["dm"], self.roles["simd"]
        zero = BingoMemSymbol(f"{self.TAG}_zero1k")
        for cl in range(self.N_CL):
            k, c = self.chip(cl), self.local(cl)
            wz = self.B[cl]["wz"]
            wz_init = self.node(k, c, dm, "wz_init", "__snax_bingo_kernel_idma_1d_copy",
                                SnaxBingoKernelIdma1dCopyArgs(zero, wz, 1024))
            warm = pre[k] if k == R else []
            warm.append(self.gemv_task(k, c, "warm_gemv", wz, wz.view(256), wz.view(512), 4, 2 * BLK, 0, [wz_init]))
            warm.append(self.node(k, c, simd, "warm_scale", "__snax_bingo_kernel_simd_stream_elementwise",
                                  # reads the zeroed head (the GEMV reads it too), not the GEMV's output
                                  SnaxBingoKernelSimdMulF16Args(wz, wz.view(128), wz.view(768), 1, 64),
                                  [wz_init]))
            if cl == 0:
                warm.append(self.node(k, c, simd, "warm_softmax", "__snax_bingo_kernel_simd_softmax_f16_f16",
                                      SnaxBingoKernelSimdSoftmaxF16F16Args(wz, wz.view(768), 2, 64), [warm[-1]]))

    def _open_window(self):
        """The warm-up and the zeroed rows (the quantisers' inputs, the A16 rows' padding, the softmax's rows 1 to 7)
        before the window, a chip's two clusters one after the other, the start behind the second (one dependency a
        chip); the start; X's multicast and the rings' first posts after it; every cluster's copy of X into L1. Returns
        ({chip: go}, {cluster: X's copy})."""
        Tg, dm, xdma = self.TAG, self.roles["dm"], self.roles["xdma"]
        pre = {k: [] for k in self.chips}
        self._warm(pre)
        for k in self.chips:
            prev = []
            for cl in self.clusters(k):
                z, zb = self.B[cl]["rowz"]
                zero = SnaxBingoKernelXdmaMemsetArgs(z, zb, SnaxBingoKernelXdmaMemsetArgs.PATTERN_ZERO)
                prev = [self.node(k, self.local(cl), xdma, "row_zero", "__snax_bingo_kernel_xdma_memset", zero, prev)]
            pre[k] += prev
        go = self.setup(extra=pre)
        self.x_bcast = self.x_multicast(go)
        self._split_rings(go, [self.done_of(self.x_bcast)])
        xc = {}
        for cl in range(self.N_CL):
            k, c = self.chip(cl), self.local(cl)
            w = self.node(k, c, dm, "x_wait", "__snax_bingo_kernel_wait_flags",
                          SnaxBingoKernelWaitFlagsArgs(BingoMemSymbol(f"{Tg}_xtflag"), 1, 1), [go[k]])
            self.dfg.bingo_add_order_edge(self.queued_of(self.x_bcast), w)
            xc[cl] = self.node(k, c, dm, "x_copy", "__snax_bingo_kernel_idma_1d_copy",
                              SnaxBingoKernelIdma1dCopyArgs(self.xland(), self.B[cl]["x"], self.XT), [w])
        return go, xc

    def _split_rings(self, go, after):
        """Each ring's first post after the start and `after`, and its chunk n_free starts a run of its own, which
        waits for the route (_after_route)."""
        for k in self.chips:
            for t in [go[k]] + list(after):
                self.rings[k].start_after(t)
            for r in range(self.rings[k].n_rings):
                self.rings[k].boundary(r, self.n_free[k])

    def _after_route(self, disp):
        """The routed chunks are pushed only once the chip has the device's route and its check passed: the post of
        each ring's run from chunk n_free on waits for its clusters' moe_dispatch tasks (disp {cluster: task})."""
        for k in self.chips:
            ds = [disp[cl] for cl in self.clusters(k)]
            for r in range(self.rings[k].n_rings):
                self.rings[k].post_after(r, self.n_free[k], ds)

    def x_multicast(self, go):
        """X into every chip's L3 behind one flag (md_xland, md_xtflag): the root's host queues its memory chip's push
        engine, which multicasts the token's row from HBM (ChipletBuild.hbm_cast). Returns the HbmCast the X waits and
        the rings' first posts follow. A whole layer (dsv2/common/dsv2_decode_mc.py) multicasts its glue's row here
        instead."""
        R, m = self.ROOT, self.mem[self.ROOT]
        one = self.h_seq[m].address + 64                  # a word holding 1 in the memory chip's HBM: the flag's value
        return self.hbm_cast(
            R, f"x_hbm_bcast_k{R:02x}",
            HostBingoKernelHbmBcastArgs(self.h_xhbm.address, self.xland(), BingoMemSymbol(f"{self.TAG}_xtflag"), one, 1,
                                        self.XT, self.NXT, m, self.engine[R]), after=[go[R]])

    # ---- the route ----------------------------------------------------------------------------------------------------
    def _route(self, go, xc):
        """The route on the root's cluster 0 (an INT8 GEMV, its factors, the softmax, the top 6), the table to every
        chip, each cluster's row lists; the routed chunks' posts after them. Returns {cluster: its moe_dispatch}."""
        d, R, Tg = self.d, self.ROOT, self.TAG
        dm, simd, xdma = self.roles["dm"], self.roles["simd"], self.roles["xdma"]
        b0 = self.B[0]
        ld, si, slab, fac = self._ld(0, ("rw",))
        rg = self.gemv_task(R, 0, "rt_t0", b0["x"], slab, b0["lg"], self.H, md.N_EXP, d.k_x, [xc[0], ld], w4=False)
        mt = self.node(R, 0, simd, "rt_deq_t0", "__snax_bingo_kernel_simd_stream_elementwise",
                       SnaxBingoKernelSimdMulF16Args(b0["lg"], fac, b0["pin"], 1, md.N_EXP), [rg])
        self.ls[0].release(si, mt)
        sm = self.node(R, 0, simd, "rt_softmax_t0", "__snax_bingo_kernel_simd_softmax_f16_f16",
                       SnaxBingoKernelSimdSoftmaxF16F16Args(b0["pin"], b0["pp"], ROWS, 64), [mt])
        tk = self.node(R, 0, xdma, "rt_topk_t0", "__snax_bingo_kernel_moe_topk_rows",
                       SnaxBingoKernelMoeTopkRowsArgs(b0["pp"], 128, 64, md.TOP_K, 0, self.T, b0["rt"]), [sm])
        # one multicast source at a time: X's multicast has landed on the root before the router ran
        rput = self.broadcast_put(R, 0, "rt_bput",
                         SnaxBingoKernelIdmaPutFlaggedArgs(b0["rt"], BingoMemSymbol(f"{Tg}_rland"), ROUTE_B,
                                                           BingoMemSymbol(f"{Tg}_rflag"), value=1, bcast=True,
                                                           also_here=True), [tk])
        disp = {}
        for cl in range(self.N_CL):
            k, c = self.chip(cl), self.local(cl)
            bg = self.B[cl]
            f = self.node(k, c, dm, "rt_fetch", "__snax_bingo_kernel_idma_fetch_flagged",
                          SnaxBingoKernelIdmaFetchFlaggedArgs(BingoMemSymbol(f"{Tg}_rland"), bg["rt"], ROUTE_B,
                                                              BingoMemSymbol(f"{Tg}_rflag"), 1, 1),
                          [rput] if k == R else [go[k]])
            if k != R:
                self.dfg.bingo_add_order_edge(rput, f)
            n = len(self.held[k])
            disp[cl] = self.node(k, c, xdma, "rt_dispatch", "__snax_bingo_kernel_moe_dispatch",
                                SnaxBingoKernelMoeDispatchArgs(bg["rt"], self.T, md.TOP_K, self.held[k], bg["lists"],
                                                               [ROWS] * n, [1] * n, pad_row=self.T), [f])
        # the routed chunks are pushed only once the chip has the device's route and its check passed
        self._after_route(disp)
        return disp

    # ---- gate|up, quantise, exchange --------------------------------------------------------------------------------
    def _gate_up(self, cl, pi, xc, disp):
        """Cluster cl's columns of its chip's piece pi: per chunk the GEMV into the piece's G row and its factors; then
        SiLU and the gate-times-up into the A16 row, the spread and the quantise into the piece's A8 row."""
        d, simd, xdma = self.d, self.roles["simd"], self.roles["xdma"]
        k, c = self.chip(cl), self.local(cl)
        bg, pc = self.B[cl], self.pieces[k][pi]
        e0, en = pc.cols(c)
        nb = en // BLK
        gr = bg["gr"][pi]
        dq = None
        for j, (cb, n) in enumerate(self.chunks(en)):
            ld, si, slab, fac = self._ld(cl, ("sg", j) if pc.e is None else ("gu", pi, j))
            row = gr.view(4 * cb)                         # 64 B a block: its gate values, then its up values
            name = f"sg_c{j}" if pc.e is None else f"gu_p{pi}c{j}"
            gm = self.gemv_task(k, c, name, bg["x"], slab, row, self.H, 2 * n, d.k_x, [xc[cl], ld])
            dq = self.node(k, c, simd, (f"sdq_c{j}" if pc.e is None else f"edq_p{pi}c{j}"),
                           "__snax_bingo_kernel_simd_stream_elementwise",
                           SnaxBingoKernelSimdMulF16Args(row, fac, row, 1, 2 * n),
                           [gm] + ([dq] if dq is not None else []))   # in turn: the last stands for all
            self.ls[cl].release(si, dq)
        sl = self.node(k, c, simd, f"silu_p{pi}", "__snax_bingo_kernel_simd_silu_f16_f16",
                       SnaxBingoKernelSimdSiluF16F16Args(gr, bg["sgr"][pi], 1, 2 * en), [dq])
        sw = self.node(k, c, simd, f"swi_p{pi}", "__snax_bingo_kernel_simd_stream_elementwise",
                       SimdMulRowsF16Args(bg["sgr"][pi], gr.view(2 * BLK), bg["a16"][pi], nb, BLK, 4 * BLK), [sl])
        sp = self.node(k, c, xdma, f"a2q_p{pi}", "__snax_bingo_kernel_xdma_6d",
                       spread_args(bg["a16"][pi], bg["sq"][pi], en), [sw])
        if pc.e is None:
            inv_a, w_addr, deps = d.sh["inv_a"], 0, [sp]
        else:
            s = self.held[k].index(pc.e)
            inv_a = d.expert(pc.e)["inv_a"]
            w_addr = bg["lists"].view(SnaxBingoKernelMoeDispatchArgs.w_offset([ROWS] * len(self.held[k]), s))
            deps = [sp, disp[cl]]
        # one row: its en / 8 beats 256 B apart, two atoms each, into the cluster's run of the A8 row (2 B a column)
        self._qa[(cl, pi)] = self.node(k, c, simd, f"q_p{pi}", "__snax_bingo_kernel_simd_quant_rows",
                                       SnaxBingoKernelSimdQuantRowsArgs(
                                           bg["sq"][pi], 1, en // 8, D_BLOCK, inv_a,
                                           bg["a8"][pi].view(self.beat_aligned(2 * e0, f"q_p{pi}")), 2 * en,
                                           w_addr=w_addr), deps)

    def _exchange(self, cl, pi):
        """Cluster cl's run of piece pi's A8 row copied into the other cluster's."""
        k, c = self.chip(cl), self.local(cl)
        e0, en = self.pieces[k][pi].cols(c)
        other = self.clusters(k)[1 - c]
        self._ex[(cl, pi)] = self.node(k, c, self.roles["dm"], f"a8_copy_p{pi}", "__snax_bingo_kernel_idma_1d_copy",
                                       SnaxBingoKernelIdma1dCopyArgs(self.B[cl]["a8"][pi].view(2 * e0),
                                                                     self.B[other]["a8"][pi].view(2 * e0), 2 * en),
                                       [self._qa[(cl, pi)]])

    # ---- the down projection ----------------------------------------------------------------------------------------
    def _shared_down(self):
        """The shared piece's down projection of every output slice into the cluster's exchange buffer, times its
        factor row. Returns {(cl, slice): the task that last wrote the slice}."""
        d, simd = self.d, self.roles["simd"]
        xch_add = {}
        for cl in range(self.N_CL):
            k, c = self.chip(cl), self.local(cl)
            other = self.clusters(k)[1 - c]
            sh = self.pieces[k][0]
            for jl in range(self.NSLC):
                ld, si, slab, fac = self._ld(cl, ("sd", jl))
                y = self.xch[(k, c)].view(jl * self.SL)
                gm = self.gemv_task(k, c, f"sd_j{jl}", self.B[cl]["a8"][0], slab, y, sh.n, self.SW, d.k_dn_sh,
                                    [ld, self._qa[(cl, 0)], self._ex[(other, 0)]])
                dq = self.node(k, c, simd, f"sd_deq_j{jl}", "__snax_bingo_kernel_simd_stream_elementwise",
                               SnaxBingoKernelSimdMulF16Args(y, fac, y, 1, self.SW), [gm])
                self.ls[cl].release(si, dq)
                xch_add[(cl, jl)] = dq
        return xch_add

    def _routed_down(self, xch_add):
        """The routed pieces' down projection by output slice: per piece the GEMV into a row buffer, times its factor
        row, added into the slice; every cluster's last add also waits for the other cluster's last weight load, so a
        last-slice chain starts only once every push has landed."""
        d, simd = self.d, self.roles["simd"]
        npc = len(self.pieces[self.chips[0]])
        dt_reader = {cl: [None] * DTMP_BUFS for cl in range(self.N_CL)}
        ndt_used = {cl: 0 for cl in range(self.N_CL)}
        last_ld = {}
        for jl in range(self.NSLC):
            for cl in range(self.N_CL):
                k, c = self.chip(cl), self.local(cl)
                bg = self.B[cl]
                other = self.clusters(k)[1 - c]
                y = self.xch[(k, c)].view(jl * self.SL)
                prev = xch_add[(cl, jl)]
                for pi in range(1, npc):
                    pc = self.pieces[k][pi]
                    ld, si, slab, fac = self._ld(cl, ("dn", jl, pi))
                    last_ld[cl] = ld
                    di = ndt_used[cl] % DTMP_BUFS
                    ndt_used[cl] += 1
                    dt = bg["dtmp"][di]
                    gm = self.gemv_task(k, c, f"dn_j{jl}s{pi}", bg["a8"][pi], slab, dt, pc.n, self.SW, d.k_dn_e,
                                        [ld, self._qa[(cl, pi)], self._ex[(other, pi)]]
                                        + ([dt_reader[cl][di]] if dt_reader[cl][di] else []))
                    mt = self.node(k, c, simd, f"dn_deq_j{jl}s{pi}", "__snax_bingo_kernel_simd_stream_elementwise",
                                   SnaxBingoKernelSimdMulF16Args(dt, fac, dt, 1, self.SW), [gm])
                    self.ls[cl].release(si, mt)
                    prev = self.node(k, c, simd, f"dn_add_j{jl}s{pi}",
                                     "__snax_bingo_kernel_simd_stream_elementwise",
                                     SnaxBingoKernelSimdAddF16Args(y, dt, y, 1, self.SW), [mt, prev])
                    dt_reader[cl][di] = prev
                xch_add[(cl, jl)] = prev
        for cl in range(self.N_CL):
            for o in self.clusters(self.chip(cl)):
                if o != cl:
                    self.edge(last_ld[o], xch_add[(cl, self.NSLC - 1)])

    def _reduce(self, xch_add):
        """One ChainGather per output slice of each cluster into the root's fold row, one at a time but the last
        slice's two, which run at once. Returns {root cluster: its last chain}."""
        R, xdma = self.ROOT, self.roles["xdma"]
        copies_in = {c: [t for (cl, _), t in self._ex.items() if self.local(cl) != c] for c in range(self.NC)}
        last_red, last_any = {c: None for c in range(self.NC)}, None
        for jl in range(self.NSLC):
            tail = jl >= self.NSLC - 1
            for c in range(self.NC):
                p1x = f"((uint64_t)({self.p1c[c].get_c_var_name()}))"
                ga = SnaxBingoKernelXdmaChainGatherArgs.fold_into(
                    cm.remote, R, c, self.chain_order(c)[:-1], src=self.xch_addr(c, jl * self.SL),
                    dst=f"({p1x} + {jl * self.SL})", size=self.SL, junction=self.jct["Elementwise"],
                    jct_csr0=EW_F16_ADD, holds=[self.p1c[c]])
                before = last_red[c] if tail else last_any
                red = self.chain_gather(R, c, f"y_chain_c{c}j{jl}", ga,
                                [xch_add[(cl, jl)] for cl in range(self.N_CL) if self.local(cl) == c]
                                + (copies_in[c] if jl == 0 else []) + ([before] if before is not None else []))
                last_red[c] = last_any = red
        return last_red

    def y_out(self, rows, last_red):
        """Each root cluster writes its half of y's row (yrow, made by rows[c]) into the HBM of the root's memory chip
        once both root clusters' last chains (last_red) are done. Returns the tasks the end follows. A whole layer
        (dsv2/common/dsv2_decode_mc.py) adds its residual here instead."""
        R, dm, hc = self.ROOT, self.roles["dm"], self.H // self.NC
        return [self.node(R, c, dm, f"y_hbm_c{c}", "__snax_bingo_kernel_idma_1d_copy",
                          SnaxBingoKernelIdma1dCopyArgs(self.yrow[c], at_offset(self.h_yhbm, 2 * hc * c), 2 * hc),
                          [rows[c]] + list(last_red.values()))
                for c in range(self.NC)]

    # ---- the decode ---------------------------------------------------------------------------------------------------
    def build(self):
        """The window, the route, the shared piece's gate|up, quantise and exchange, then its down projection of every
        output slice; the routed pieces' gate|up (each piece's exchange behind the next piece's loads in the DM core's
        program) and down; the reduce; y's row to HBM, each root cluster's fold row once no chain is in flight."""
        first = max((t.node_id for t in self.dfg.node_list), default=-1) + 1
        self.dfg.relay_one_consumer_cores = [self.roles["xdma"]]
        go, xc = self._open_window()
        disp = self._route(go, xc)
        self._qa, self._ex = {}, {}
        for cl in range(self.N_CL):
            self._gate_up(cl, 0, xc, disp)
        for cl in range(self.N_CL):
            self._exchange(cl, 0)
        xch_add = self._shared_down()
        npc = len(self.pieces[self.chips[0]])
        for pi in range(1, npc):
            for cl in range(self.N_CL):
                self._gate_up(cl, pi, xc, disp)
                if pi > 1:
                    self._exchange(cl, pi - 1)
        for cl in range(self.N_CL):
            self._exchange(cl, npc - 1)
        self._routed_down(xch_add)
        last_red = self._reduce(xch_add)
        self.end_window(self.y_out(dict(last_red), last_red))
        self._check_gemv_only(first)
        self._checks()

    def _check_gemv_only(self, first):
        """Every array task this block made (node ids from `first`) is a GEMV, so no gemm_full follows a GEMV on a GEMM
        core."""
        gemm = self.roles["gemm"]
        arr = [t for t in self.dfg.node_list if t.node_id >= first and t.assigned_core_id == gemm
               and (t.kernel_name or "").startswith("__snax")]
        bad = [t.node_name for t in arr if t.kernel_name != "__snax_bingo_kernel_gemv"]
        if bad:
            raise ValueError(f"array tasks that are not GEMVs: {bad[:4]}")
        print(f"[{self.TAG}] gemv: {len(arr)} array tasks, every one a GEMV (the router INT8, gate|up and down INT4); "
              f"X a_row ({self.XT} B), factor rows, one row a slice ({self.SL} B a chain)")
