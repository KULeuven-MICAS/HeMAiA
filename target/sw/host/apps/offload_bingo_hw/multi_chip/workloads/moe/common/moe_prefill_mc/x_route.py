# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""MBuild's first phase: the window's start, X and the router's weights from HBM, every cluster's route of its own
tiles, the route table to every chip and each cluster's row lists, and the shared MLP's gate|up of X's row blocks
(README.md, X AND THE ROUTE)."""

import moe_prefill_data as md
from bingo_kernel_args import (SnaxBingoKernelIdma1dCopyArgs, SnaxBingoKernelIdma2dCopyArgs,
                               SnaxBingoKernelIdmaFetchFlaggedArgs, SnaxBingoKernelIdmaPutFlaggedArgs,
                               SnaxBingoKernelMoeDispatchArgs, SnaxBingoKernelMoeTopkRowsArgs,
                               SnaxBingoKernelSimdDeqSwigluArgs, SnaxBingoKernelSimdMulTilesArgs,
                               SnaxBingoKernelSimdSoftmaxF16F16Args, SnaxBingoKernelWaitFlagsArgs)
from bingo_mem_handle import BingoMemSymbol

from .common import BLK, D_BLOCK, ROUTE_B


class XRoute:

    # ---- the window's start ---------------------------------------------------------------------
    def _open_window(self):
        """The warm-up before the start, the start, the HBM casts of the router's weights and X, every routing
        cluster's load of the router's weights once they land, and (sync barrier) each chip's wait for X's last tile.
        Returns {chip: go task}."""
        R, Tg = self.ROOT, self.TAG
        dm, simd, xdma = self.roles["dm"], self.roles["simd"], self.roles["xdma"]
        pre = {k: [] for k in self.chips}
        zero = BingoMemSymbol(f"{Tg}_zero1k")
        self.rw_ld = {}
        # the code of the first tasks into the instruction cache, on zeros, before the start (the multicasts would
        # otherwise starve its fetches): A 8 rows of pitch 32 at 0, B at 256, D at 512, the SIMD's output at 768. Every
        # chip's clusters, which all route
        for k in self.chips:
            for c in range(self.NC):
                wz = self.B[self.clusters(k)[c]]["wz"]
                wz_init = self.node(k, c, dm, "wz_init", "__snax_bingo_kernel_idma_1d_copy",
                                    SnaxBingoKernelIdma1dCopyArgs(zero, wz, 1024))
                # the root's warm-up holds the start stamp; another chip's only runs first on its cores
                warm = pre[k] if k == R else []
                warm.append(self.gemm(k, c, "warm_gemm", wz, 32, wz.view(256), wz.view(512), 1, 2, 1, 0, [wz_init]))
                warm.append(self.node(k, c, simd, "warm_scale", "__snax_bingo_kernel_simd_mul_tiles",
                                      SnaxBingoKernelSimdMulTilesArgs(wz.view(256), wz.view(256), wz.view(768), 1, 1),
                                      [wz_init]))
                warm.append(self.node(k, c, simd, "warm_softmax", "__snax_bingo_kernel_simd_softmax_f16_f16",
                                      SnaxBingoKernelSimdSoftmaxF16F16Args(wz, wz.view(768), 2, 64), [warm[-1]]))
        go = self.setup(extra=pre)
        self.rw_hbm_cast, self.x_hbm_cast = self.hbm_bcasts(go)
        # what the waits on the multicasts' flags are ordered after: the task that queues each one's last tile
        self.rw_bcast, self.x_bcast = self.rw_hbm_cast.posts[-1], self.x_hbm_cast.posts[-1]
        # every routing cluster's router weights and factors into L1 once the multicast has landed them
        for cl in range(self.N_CL):
            if self.rt_tiles[cl]:
                k, c, bg = self.chip(cl), self.local(cl), self.B[cl]
                w = self.node(k, c, dm, "rw_wait", "__snax_bingo_kernel_wait_flags",
                              SnaxBingoKernelWaitFlagsArgs(BingoMemSymbol(f"{Tg}_rwflag"), 1, 1), [go[k]])
                self.dfg.bingo_add_order_edge(self.rw_bcast, w)
                self.rw_ld[cl] = (self.node(k, c, dm, "rw_load", "__snax_bingo_kernel_idma_1d_copy",
                                           SnaxBingoKernelIdma1dCopyArgs(self.h_rw, bg["rw"], self.H * 64), [w]),
                                 self.node(k, c, dm, "rws_load", "__snax_bingo_kernel_idma_1d_copy",
                                           SnaxBingoKernelIdma1dCopyArgs(self.h_rws, bg["rws"], 4 * D_BLOCK), [w]))
        if self.sync == "barrier":
            # params sync barrier: no ring waits for the root's multicast across chips; each chip's waits for X's last
            # tile in its own L3 (a flag), on its cluster 0 xDMA core, which is idle until its top 6
            self.x_landed = {}
            for k in self.chips:
                w = self.node(k, 0, xdma, "x_landed", "__snax_bingo_kernel_wait_flags",
                              SnaxBingoKernelWaitFlagsArgs(BingoMemSymbol(f"{Tg}_xtflag", offset=64 * (self.NXT - 1)),
                                                           1, 1), [go[k]])
                self.dfg.bingo_add_order_edge(self.x_bcast, w)
                self.x_landed[k] = w
        return go

    # ---- the route ------------------------------------------------------------------------------
    def _route_tile(self, cl, ts, a, deps):
        """Cluster cl's tiles ts (consecutive, 8 tokens each, named by the first) from their rows at `a`: the router
        GEMM, its factors, the softmax and the top 6 into the cluster's route table."""
        d, k, c = self.d, self.chip(cl), self.local(cl)
        simd, xdma = self.roles["simd"], self.roles["xdma"]
        bg = self.B[cl]
        i, n = ts[0], len(ts)
        rg = self.gemm(k, c, f"rt_t{i}", a, self.PX, bg["rw"], bg["lg"], n, self.H // 8, 4, d.k_x,
                       list(deps) + [self.rw_ld[cl][0]], int4=0)
        mt = self.node(k, c, simd, f"rt_deq_t{i}", "__snax_bingo_kernel_simd_mul_tiles",
                       SnaxBingoKernelSimdMulTilesArgs(bg["lg"], bg["rws"], bg["pin"], n, 4, dst_pitch=128),
                       [rg, self.rw_ld[cl][1]])
        sm = self.node(k, c, simd, f"rt_softmax_t{i}", "__snax_bingo_kernel_simd_softmax_f16_f16",
                       SnaxBingoKernelSimdSoftmaxF16F16Args(bg["pin"], bg["pp"], 8 * n, 64), [mt])
        tk = self.node(k, c, xdma, f"rt_topk_t{i}", "__snax_bingo_kernel_moe_topk_rows",
                       SnaxBingoKernelMoeTopkRowsArgs(bg["pp"], 128, 64, md.TOP_K, 0, 8 * n,
                                                      bg["rt"].view(8 * i * ROUTE_B)), [sm])
        for t in ts:
            self._rt_tile[t] = (cl, tk)
        return rg

    def _route_tiles_first(self, go):
        """Every cluster's tiles first: their rows straight from the chip's X landing once their flags are in, copied
        into the cluster's own buffer, then routed."""
        Tg, dm = self.TAG, self.roles["dm"]
        for cl in range(self.N_CL):
            k, c = self.chip(cl), self.local(cl)
            ts = self.rt_tiles[cl]
            for i in ts[:1]:
                w = self.node(k, c, dm, f"rt_xwait_t{i}", "__snax_bingo_kernel_wait_flags",
                              SnaxBingoKernelWaitFlagsArgs(BingoMemSymbol(f"{Tg}_xtflag", offset=64 * i), len(ts), 1),
                              [go[k]])
                self.dfg.bingo_add_order_edge(self.x_bcast, w)
                xc = self.node(k, c, dm, f"rt_xcopy_t{i}", "__snax_bingo_kernel_idma_2d_copy",
                               SnaxBingoKernelIdma2dCopyArgs(self.xland(8 * i * self.H), self.B[cl]["rx"], self.H,
                                                             self.H, self.PX, 8 * len(ts)),
                               [w])
                self._route_tile(cl, ts, self.B[cl]["rx"], [xc])

    def _route_up(self, i):
        """Tile i's entries and flag into the root's landing area."""
        R, Tg = self.ROOT, self.TAG
        cl, tk = self._rt_tile[i]
        k, c, o = self.chip(cl), self.local(cl), 8 * i * ROUTE_B
        self._rt_ups[i] = self.node(k, c, self.roles["dm"], f"rt_up_t{i}", "__snax_bingo_kernel_idma_put_flagged",
                                    SnaxBingoKernelIdmaPutFlaggedArgs(
                                        self.B[cl]["rt"].view(o), BingoMemSymbol(f"{Tg}_rland", offset=o, chip_id=R),
                                        8 * ROUTE_B, BingoMemSymbol(f"{Tg}_rup", offset=64 * i, chip_id=R), value=1),
                                    [tk])
        w = self._rt_wait.get(8 * i // self.XR)
        if w is not None:                                   # the root's wait already exists
            if k == R:
                self.edge(self._rt_ups[i], w)
            else:
                self.dfg.bingo_add_order_edge(self._rt_ups[i], w)

    def _root_collect(self, go):
        """Per row block, the root waits for the block's tile flags and sends the block's table on to every chip (and
        its own flag); the first once X's last tile has landed here (one multicast source at a time)."""
        R, T, Tg, XR, dm = self.ROOT, self.T, self.TAG, self.XR, self.roles["dm"]
        xdone = self.node(R, 0, dm, "rt_xdone", "__snax_bingo_kernel_wait_flags",
                          SnaxBingoKernelWaitFlagsArgs(BingoMemSymbol(f"{Tg}_xtflag", offset=64 * (self.NXT - 1)),
                                                       1, 1), [go[R]])
        self.dfg.bingo_add_order_edge(self.x_bcast, xdone)
        rputs, rt_ups = self._rputs, self._rt_ups
        for b in range(self.XB):
            tl = [i for i in range(T // 8) if 8 * i // XR == b]
            local = [rt_ups[i] for i in tl if i in rt_ups and self.chip(self._rt_tile[i][0]) == R]
            w = self.node(R, 0, dm, f"rt_wait_b{b}", "__snax_bingo_kernel_wait_flags",
                          SnaxBingoKernelWaitFlagsArgs(BingoMemSymbol(f"{Tg}_rup", offset=64 * tl[0]), len(tl), 1),
                          [go[R]] + local + rputs[-1:] + ([xdone] if b == 0 else []))
            for i in tl:
                if i in rt_ups and self.chip(self._rt_tile[i][0]) != R:
                    self.dfg.bingo_add_order_edge(rt_ups[i], w)
            self._rt_wait[b] = w
            o = b * XR * ROUTE_B
            if self.x_cast == "unicast":
                # params x_cast unicast: the block's table into each chip's L3 in turn (self.order: the root's own first,
                # a copy onto itself that sets its flag), each with its flag there
                prev = [w]
                for kk in self.order:
                    there = None if kk == R else kk
                    self._rput_to[(b, kk)] = self.node(
                        R, 0, dm, f"rt_uput_b{b}_k{kk:02x}", "__snax_bingo_kernel_idma_put_flagged",
                        SnaxBingoKernelIdmaPutFlaggedArgs(
                            BingoMemSymbol(f"{Tg}_rland", offset=o), BingoMemSymbol(f"{Tg}_rland", offset=o, chip_id=there),
                            XR * ROUTE_B, BingoMemSymbol(f"{Tg}_rflag", offset=64 * b, chip_id=there), value=1),
                        prev)
                    prev = [self._rput_to[(b, kk)]]
                rputs.append(self._rput_to[(b, self.order[-1])])
                continue
            rputs.append(self.broadcast_put(R, 0, f"rt_bput_b{b}",
                                   SnaxBingoKernelIdmaPutFlaggedArgs(
                                       BingoMemSymbol(f"{Tg}_rland", offset=o), BingoMemSymbol(f"{Tg}_rland", offset=o),
                                       XR * ROUTE_B, BingoMemSymbol(f"{Tg}_rflag", offset=64 * b), value=1,
                                       bcast=True, also_here=True), [w]))

    def _fetch_dispatch(self, cl, go):
        """Cluster cl fetches each block's table as the root sends it and turns it into its chip's row lists."""
        d, R, T, Tg, XR = self.d, self.ROOT, self.T, self.TAG, self.XR
        k, c = self.chip(cl), self.local(cl)
        bg = self.B[cl]
        caps = [md.pad8(d.cnt[e]) for e in self.held[k]]
        cnts = [int(d.cnt[e]) for e in self.held[k]]
        prev = []
        for b in range(self.XB):
            o = b * XR * ROUTE_B
            put = self._rput_to.get((b, k), self._rputs[b])
            f = self.node(k, c, self.roles["dm"], f"rt_fetch_b{b}", "__snax_bingo_kernel_idma_fetch_flagged",
                          SnaxBingoKernelIdmaFetchFlaggedArgs(
                              BingoMemSymbol(f"{Tg}_rland", offset=o), bg["rt"].view(o), XR * ROUTE_B,
                              BingoMemSymbol(f"{Tg}_rflag", offset=64 * b), 1, 1),
                          [put] if k == R else [go[k]])
            if k != R:
                self.dfg.bingo_add_order_edge(put, f)
            prev = [self.node(k, c, self.roles["xdma"], f"rt_dispatch_b{b}", "__snax_bingo_kernel_moe_dispatch",
                              SnaxBingoKernelMoeDispatchArgs(bg["rt"], T, md.TOP_K, self.held[k], bg["lists"],
                                                             caps, cnts, pad_row=T, t0=b * XR, t1=(b + 1) * XR,
                                                             state_addr=bg["dst"]), [f] + prev)]
        self._disp[cl] = prev[0]

    def _route_whole(self, go):
        """params barrier_at kernel, K1: the route whole -- every cluster's upload, the root's table out to every chip,
        every cluster's row lists -- before any row block of X."""
        for cl in range(self.N_CL):
            for i in self.rt_tiles[cl]:
                self._route_up(i)
        self._root_collect(go)
        for cl in range(self.N_CL):
            self._fetch_dispatch(cl, go)
        self.barrier({k: [] for k in self.chips})

    def _route_rest(self, go):
        """Every tile's upload not made yet, and every cluster's row lists not made yet."""
        for i in sorted(self._rt_tile):
            if i not in self._rt_ups:
                self._route_up(i)
        for cl in range(self.N_CL):
            if cl not in self._disp:
                self._fetch_dispatch(cl, go)

    # ---- X's row blocks: the shared MLP's gate|up ---------------------------------------------------
    def _x_blocks(self, go):
        """Per row block of X and cluster: the wait for the block's tiles, its copy into the row buffer, and the shared
        MLP's gate|up of the first weight chunk (held in its slab over every block): the GEMM, the dequantisation and
        SwiGLU, and the quantise into the shared A8. The tuned schedule makes each cluster's upload and row lists behind
        its first row blocks' copies (b1)."""
        d, Tg, XR = self.d, self.TAG, self.XR
        dm, simd = self.roles["dm"], self.roles["simd"]
        b1 = min(1, self.XB - 1)
        for b in range(self.XB):
            for cl in range(self.N_CL):
                k, c = self.chip(cl), self.local(cl)
                xb = self.B[cl]["xblk"][b % 2]
                w = self.node(k, c, dm, f"x_wait_b{b}", "__snax_bingo_kernel_wait_flags",
                              SnaxBingoKernelWaitFlagsArgs(
                                  BingoMemSymbol(f"{Tg}_xtflag", offset=64 * (b * XR // 8)), XR // 8, 1), [go[k]])
                self.dfg.bingo_add_order_edge(self.x_bcast, w)
                xc = self.node(k, c, dm, f"x_copy_b{b}", "__snax_bingo_kernel_idma_2d_copy",
                               SnaxBingoKernelIdma2dCopyArgs(self.xland(b * XR * self.H), xb, self.H, self.H, self.PX,
                                                             XR),
                               [w] + self._xblk_reader[cl][b % 2])
                self._xblk_reader[cl][b % 2] = []
                (sc0, sn), _ = self.ch(k, c)
                if b == 0:                                  # the shared gate|up's first chunk, held for every block
                    self._sh_ld[cl] = [self._load(cl, j) for j in range(min(1, len(self.chunks(sn))))]
                if not self.per_kernel and b == b1:         # the cluster's upload, behind its first row blocks' copies
                    for i in self.rt_tiles[cl]:
                        self._route_up(i)
                    if cl == 0:
                        self._root_collect(go)
                    self._fetch_dispatch(cl, go)
                for j, (b0, n) in enumerate(self.chunks(sn)[:len(self._sh_ld[cl])]):
                    ld, si, slab, sc = self._sh_ld[cl][j]
                    gi, gb = self._next_g(cl)
                    gm = self.gemm(k, c, f"sg_c{j}_b{b}", xb, self.PX, slab, gb, XR // 8, self.H // 8,
                                   2 * n // BLK, d.k_x,
                                   [xc, ld] + ([self._gb_reader[cl][gi]] if self._gb_reader[cl][gi] is not None else []))
                    self._xblk_reader[cl][b % 2].append(gm)
                    self._sg_gms.setdefault((cl, j), []).append(gm)
                    nsh = sn // BLK
                    sw = self.node(k, c, simd, f"ssw_c{j}_b{b}", "__snax_bingo_kernel_simd_deq_swiglu",
                                   SnaxBingoKernelSimdDeqSwigluArgs(
                                       gb, sc.addr, self.B[cl]["sg"], self._sh_a16(cl, b, b0),
                                       XR // 8, n // BLK, nsh * D_BLOCK),
                                   [gm] + sc.deps + ([self._last_sw[cl]] if self._last_sw[cl] else []))
                    self._gb_reader[cl][gi] = sw
                    self._last_sw[cl] = self._sh_quant_block(cl, j, b, b0, n, sw)
                    self._scale_read(cl, sc, sw)
                    if b == self.XB - 1:
                        self._release(cl, si, self._sg_gms[(cl, j)], sw, sc)

    def _rings_after_x(self):
        """The root host's wait for X's landing (the cast's end), then every chip's weight rings start (their first
        posts) once X's multicast is done (that wait; sync barrier: X's last tile in the chip's own L3), so the
        multicast has the half-duplex D2D links to itself instead of crossing the memory chips' pushes."""
        landed = self.x_hbm_cast.end
        for k in self.chips:
            self.rings[k].start_after(self.x_landed[k] if self.sync == "barrier" else landed)
