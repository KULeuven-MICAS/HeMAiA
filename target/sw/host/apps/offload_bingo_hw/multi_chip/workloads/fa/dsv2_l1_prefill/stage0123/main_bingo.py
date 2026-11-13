#!/usr/bin/env python3
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""DeepSeek-V2-Lite's layer-1 MLA attention, stages 0 to 3 of the 256-token prefill on sixteen chiplets chained: x
from HBM into every chip's L3, the keys by columns, the owners' pieces and their multicasts into every chip.

Stage 0 and stage 1 as their apps (../stage0, ../stage1: x multicast by the root's memory chip, each cluster's tiles of
x into L1 split between the iDMA and the xDMA, a GEMM a tile over the chip's W_KVA columns with the INT4 balance, each
slice into the tile owner's L3, the owners' waits); then each owner's stage 2 starts behind its stage-1 waits (as
../stage2: c on cluster 0, k_pe and the value columns on cluster 1, both parts put into the root's L3); the root waits
for a chunk's pieces and multicasts them into every other chip's L3 (stage 3, one multicast at a time), and every
other chip waits for both chunks. The key constants are loaded before the window.

          before the window     kvc_ld                        DM: the key constants into every cluster's L1
          S0_start
          stage 0               x_hbm_bcast_post<n> / _land<n>, x_wait_t<t>
          stage 1               Ld_wkvc_push, Ld_wkvc, kx_ld_q<q>, kx_xld_q<q>, gkvc_q<q>, kvc_put_q<q>
                                (helpers: Ld_wkvx, gkvx_q<q>, kvx_put_q<q>); kvr_wait_q<q>, kvx_wait_q<q>
          stage 2               per tile: kvr_ld .. kv_rmsa, kv_put (cluster 0); kvv_ld .. kv_rmsb, kvp_put (cluster 1)
          stage 3               the root: kv_nwait_h<h>, kv_pwait_h<h>, kv_bput_h<h>; every other chip: kv_all
          P_fin<chip> -> S1_end -> fp_report, the checks

Checks: three tiles' pieces in the L3 of the root, of the chip in the middle of the fold order and of the last (the
key rows' head, their k_pe, the value columns' tail).

params.hjson: the platform, the tokens, the chunk and key-tile sizes, the weight chunk and ring sizes, the weights'
width."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from stage_common import (HIDDEN, KV_RANK, KVP, PG, ROPE, TILE16, BingoMemSymbol, StageBase,  # noqa: E402
                          app_name)
from bingo_kernel_args import (SnaxBingoKernelIdma1dCopyArgs, SnaxBingoKernelIdmaPutFlaggedArgs,  # noqa: E402
                               SnaxBingoKernelSimdRmsnormArgs, SnaxBingoKernelWaitFlagsArgs,
                               SnaxBingoKernelXdma1dCopyArgs, SnaxBingoKernelXdmaTranspose2dArgs, simd_fp16_lanes)
from driver import run_driver  # noqa: E402
from libs.comm.nest import convert_args, d_to_a_args  # noqa: E402

CHK = 2048                                  # bytes a checked window
PAIR_BYTES = HIDDEN * 16 * PG // 2          # an INT4 pair of W_KVA's columns, paired layout


class Stage0123(StageBase):
    """Stages 0 to 3 chained (the module docstring)."""
    STAGE = "0123"
    X_XDMA = True
    KV_BALANCE_WAYS = 4

    def stage_inputs(self):
        """The k_pe parts' flags on the root, one a tile."""
        self.st.put_zeros(f"{self.TAG}_kvpflag", "uint8_t", (self.T // 16) * 64)

    def build(self):
        self.last_ld = {}
        for ls in self.ls:
            ls.strict = True
        go = self.setup(extra=self.key_consts_preload())
        self.relays()
        self.x_puts = self.stage_x_bcast(go)
        last = {k: [] for k in self.chips}
        self.stage_keys(go, last)
        self.stage_owner(go, last)
        self.window_end(last)
        self.run_checks(self.piece_checks())

    def key_consts_preload(self):
        """Before the window, per chip one chain ending on cluster 0: every cluster's key constants (W_KVA's
        dequantisation scales and RoPE tables: kvc on the tiles' cluster, kvc_pe on the other, for k_pe)."""
        T = self.TAG
        pre = {k: [] for k in self.chips}
        self.kvc_pe = {}
        for k in self.chips:
            for c in reversed(range(self.NC)):
                cl = self.chips.index(k) * self.NC + c
                if (k, c) in self.kvc:
                    pre[k] = [self.dm_copy(cl, "kvc_ld", self.h_kc[k].address, self.kvc[(k, c)], self.KC_BYTES,
                                           pre[k])]
            for c in range(self.NC):
                if (k, c) not in self.kvc:
                    cl = self.chips.index(k) * self.NC + c
                    self.kvc_pe[(k, c)] = self.cx_chip(k).at(c).l1(f"{T}_kvcpe", self.KC_BYTES)
                    pre[k] = [self.dm_copy(cl, "kvc_ld", self.h_kc[k].address, self.kvc_pe[(k, c)],
                                           self.KC_BYTES, pre[k])]
        return pre

    # ---- the keys by columns ------------------------------------------------------------------------------------
    def stage_keys(self, go, last):
        """Every cluster's key tiles (keys_cluster), then every owner's waits for its tiles' slices (owner_waits)."""
        self.kvc_put, self.kvx_put = {}, {}
        for j, k in enumerate(self.chips):
            for c in range(self.NC):
                self.keys_cluster(j, k, c, go, last)
        self.owner_waits(last)

    def keys_cluster(self, j, k, c, go, last):
        """Cluster c of chip k (chips[j]): W_KVA first on the in-order DM core (the copy depends on the chip's start),
        then the first tiles of x; a GEMM a tile into its owner's L3 (two x and two output buffers, tile n takes
        n % 2); a helper's tiles of a two-pair chip's second pair."""
        d, T, dm, xdma = self.d, self.TAG, self.roles["dm"], self.roles["xdma"]
        gr0, gr1 = self.kvg[k]
        nb = (gr1 - gr0) * TILE16
        helps = self.kvx.get(k)
        cl = j * self.NC + c
        tiles = self.cl_tiles(c)
        l1 = lambda name, size: self.cx_chip(k).at(c).l1(f"{T}_kc_{name}", size)    # noqa: E731
        xb, ob = [l1(f"x{b}", self.XT) for b in range(2)], [l1(f"o{b}", nb) for b in range(2)]
        gm, put, ld_x, gmx, putx = {}, {}, {}, {}, []
        xx = self.x_xdma

        def kx(n):
            # the tile's wait right before its load: the in-order DM core blocks on no later tile; the x buffer is free
            # once the tile two before it has had its GEMMs
            q = tiles[n]
            after = [self.x_wait(k, c, q, go)] + ([gm[n - 2]] if n >= 2 else []) + \
                ([gmx[n - 2]] if n - 2 in gmx else [])
            a = self.XT - xx
            src = lambda off: BingoMemSymbol(f"{T}_xsrc", offset=q * self.XT + off)     # noqa: E731
            parts = []
            if a:
                parts.append(self.node(k, c, dm, f"kx_ld_q{q}", "__snax_bingo_kernel_idma_1d_copy",
                                       SnaxBingoKernelIdma1dCopyArgs(src(0), xb[n % 2], a), after))
            if xx:
                parts.append(self.node(k, c, xdma, f"kx_xld_q{q}", "__snax_bingo_kernel_xdma_1d_copy",
                                       SnaxBingoKernelXdma1dCopyArgs(src(a), xb[n % 2].view(a), xx), after))
            return parts

        hx = None
        if helps:
            hx = dict(xs=l1("xs", PAIR_BYTES), obx=[l1(f"ox{b}", PG * TILE16) for b in range(2)], ld=None)
        ld, si, slab = self.load(cl, 0, "Ld_wkvc", deps=[go[k]])
        ld_x[0] = kx(0)
        if len(tiles) > 1:
            ld_x[1] = kx(1)
        if helps:
            # the helped chip's second pair, once, from its slab in HBM
            hx["ld"] = self.node(k, c, dm, "Ld_wkvx", "__snax_bingo_kernel_idma_1d_copy",
                                 SnaxBingoKernelIdma1dCopyArgs(self.h_wkvc[helps[0]].address + PAIR_BYTES, hx["xs"],
                                                               PAIR_BYTES), [go[k]])
        for n, q in enumerate(tiles):
            o = self.kv_owner[q]
            lt = self.kv_tiles[o].index(q)
            on = lambda name, off, o=o: BingoMemSymbol(name, offset=off, chip_id=None if o == k else o)  # noqa: E731
            # a two-pair chip's helped tiles: its first pair only, the slab's first half
            ng = (gr1 - gr0) - (PG if k in self.kv_two and q in self.kv_half else 0)
            if n not in ld_x:
                ld_x[n] = kx(n)
            gm[n] = self.gmm(cl, f"gkvc_q{q}", xb[n % 2], slab, ob[n % 2], 1, HIDDEN // 4, ng, d.k_x,
                             [ld] + ld_x[n] + ([put[n - 2]] if n >= 2 else []))
            # tile n + 1's x load made right behind tile n's GEMM, ahead of tile n's put: the DM core loads it while
            # GEMM n runs (it needs only GEMM n - 1's buffer) instead of after GEMM n's put
            if n + 1 < len(tiles) and n + 1 not in ld_x:
                ld_x[n + 1] = kx(n + 1)
            put[n] = self.node(k, c, dm, f"kvc_put_q{q}", "__snax_bingo_kernel_idma_put_flagged",
                               SnaxBingoKernelIdmaPutFlaggedArgs(
                                   ob[n % 2], on(f"{T}_kvraw", lt * self.KVD + gr0 * TILE16), ng * TILE16,
                                   on(f"{T}_kvrflag", (lt * len(self.chips) + j) * 64), value=1), [gm[n]])
            self.kvc_put[(q, k)] = put[n]
            if helps and q in helps[3]:
                gmx[n] = self.helped_tile(k, c, cl, q, lt, on, helps, hx, xb[n % 2], ld_x[n], putx)
            if n + 2 < len(tiles):
                ld_x[n + 2] = kx(n + 2)
        if si is not None:
            self.ls[cl].release(si, [gm[n] for n in range(len(tiles))])    # every tile's GEMM reads it
        last[k] += list(put.values()) + list(putx)

    def helped_tile(self, k, c, cl, q, lt, on, helps, hx, x_buf, x_lds, putx):
        """Helper chip k's GEMM of tile q over the helped chip's second pair, and its put into the tile owner's L3
        behind the pair's flag. Returns the GEMM."""
        d, T, dm = self.d, self.TAG, self.roles["dm"]
        a, ga0, e, _ = helps
        m = len(putx)
        gmx = self.gmm(cl, f"gkvx_q{q}", x_buf, hx["xs"], hx["obx"][m % 2], 1, HIDDEN // 4, PG, d.k_x,
                       [hx["ld"]] + x_lds + ([putx[m - 2]] if m >= 2 else []))
        putx.append(self.node(k, c, dm, f"kvx_put_q{q}", "__snax_bingo_kernel_idma_put_flagged",
                              SnaxBingoKernelIdmaPutFlaggedArgs(
                                  hx["obx"][m % 2], on(f"{T}_kvraw", lt * self.KVD + ga0 * TILE16),
                                  PG * TILE16, on(f"{T}_kvxflag", (lt * len(self.kv_two) + e) * 64),
                                  value=1), [gmx]))
        self.kvx_put[(q, k)] = putx[-1]
        return gmx

    def owner_waits(self, last):
        """Every owner waits for its tiles' 16 slices, and for a helped tile its helpers' slices: the end of stage 1
        (own_wait {tile: the waits}: where each tile's stage 2 starts)."""
        T, dm = self.TAG, self.roles["dm"]
        self.own_wait = {}
        for o in self.chips:
            for lt, q in enumerate(self.kv_tiles[o]):
                waits = [("kvr_wait", "kvrflag", len(self.chips), [self.kvc_put[(q, k2)] for k2 in self.chips])]
                if self.kvx and q in self.kv_half:
                    waits.append(("kvx_wait", "kvxflag", len(self.kv_two),
                                  [self.kvx_put[(q, h)] for h, x in self.kvx.items() if q in x[3]]))
                for name, flag, nf, puts in waits:
                    w = self.node(o, 0, dm, f"{name}_q{q}", "__snax_bingo_kernel_wait_flags",
                                  SnaxBingoKernelWaitFlagsArgs(BingoMemSymbol(f"{T}_{flag}", offset=lt * nf * 64), nf, 1),
                                  [t for t in puts if self.flag_dep(t, o, 0)])
                    for t in puts:
                        if not self.flag_dep(t, o, 0):
                            self.dfg.bingo_add_order_edge(t, w)
                    last[o].append(w)
                    self.own_wait.setdefault(q, []).append(w)

    # ---- the owners -----------------------------------------------------------------------------------------------
    def stage_owner(self, go, last):
        """Stage 2 chained behind stage 1: kv_tile on every chip's tiles, each behind stage 1's owner waits (own_wait);
        the root's waits for every piece, a chunk at a time, on the cluster that multicasts them; stage 3: the root
        multicasts each chunk once its waits are done, one multicast at a time; then every other chip waits for both
        chunks (kv_all)."""
        R, T, dm, P = self.ROOT, self.TAG, self.roles["dm"], self.C // 16
        self.kvp_put = {}
        d, mesh = self.d, self.d.mesh
        c8 = np.arange(16 * KV_RANK).reshape(16, KV_RANK)
        assert np.array_equal(d.layout.to_b(c8, mesh), d.layout.to_a(np.ascontiguousarray(c8.T), mesh))
        for k in self.chips:
            for lt, q in enumerate(self.kv_tiles[k]):
                self.kv_tile(k, lt, q)
            last[k] += [self.kv_put[q] for q in self.kv_tiles[k]] + [self.kvp_put[q] for q in self.kv_tiles[k]
                                                                    if q in self.kvp_put]
        c, prev = self.x_put_cluster, []
        self.kv_mc = []
        waits = [("kv_nwait", "kvnflag", self.kv_put), ("kv_pwait", "kvpflag", self.kvp_put)]
        for h in range(self.NCH):
            for name, flag, src in waits:
                puts = [src[q] for q in range(h * P, (h + 1) * P)]
                w = self.node(R, c, dm, f"{name}_h{h}", "__snax_bingo_kernel_wait_flags",
                              SnaxBingoKernelWaitFlagsArgs(BingoMemSymbol(f"{T}_{flag}", offset=h * P * 64), P, 1),
                              [t for t in puts if self.flag_dep(t, R, c)] + prev)
                for t in puts:
                    if not self.flag_dep(t, R, c):
                        self.dfg.bingo_add_order_edge(t, w)
                prev = [w]
                last[R].append(w)
            # stage 3: the chunk's pieces from the root's L3 into every other chip's, one multicast at a time
            src = BingoMemSymbol(f"{T}_kv", offset=h * P * KVP)
            self.kv_mc.append(self.broadcast_put(R, c, f"kv_bput_h{h}", SnaxBingoKernelIdmaPutFlaggedArgs(
                src, src, P * KVP, BingoMemSymbol(f"{T}_kvflag", offset=h * 64), value=1, bcast=True), prev))
            prev = self.kv_mc[-1:]
            last[R].append(self.kv_mc[-1])
        for k in self.chips:
            if k != R:
                last[k].append(self.kv_all(k))

    def kv_all(self, k):
        """Chip k's wait (cluster 0's DM core) for both chunks' keys from the stage-3 multicasts."""
        w = self.node(k, 0, self.roles["dm"], "kv_all", "__snax_bingo_kernel_wait_flags",
                      SnaxBingoKernelWaitFlagsArgs(BingoMemSymbol(f"{self.TAG}_kvflag"), self.NCH, 1), [])
        for t in self.kv_mc:
            self.dfg.bingo_add_order_edge(t, w)
        return w

    def kv_tile(self, k, lt, q):
        """Chip k's tile q (its lt-th) as two parts. Cluster 0 takes c: its 512 columns (the first 16 KB of the tile's
        D layout) into L1, their dequantisation, and RMSNorm row_major -> A (INT8) writes the key rows' c. Cluster 1
        builds c transposed itself (its own load and dequantisation of c, D -> row major and the transpose on its own
        xDMA) and RMSNorm col_major -> B (INT8) writes the value columns from it (B of c [16, 512] is A of c^T); k_pe
        (the last 2 KB) is dequantised and rotated (RoPE) into the key rows' k_pe there too, first on its in-order SIMD
        core. Cluster 0 puts the key rows' c behind the tile's flag (kvnflag); cluster 1 puts k_pe and the value
        columns (adjacent in the slot) behind a flag of its own (kvpflag)."""
        d, T, mesh = self.d, self.TAG, self.d.mesh
        simd, xdma, dm = self.roles["simd"], self.roles["xdma"], self.roles["dm"]
        R, KR, RP = self.ROOT, KV_RANK, ROPE
        on_root = lambda name, off: BingoMemSymbol(name, offset=off, chip_id=None if k == R else R)    # noqa: E731
        cb, n_pe, half = (KR // 16) * TILE16, 16 * RP, (RP // 32) * TILE16
        sc = self.fbits(d.s_c.inv)
        beat = simd_fp16_lanes() * 2
        s = "" if lt == 0 else f"_t{lt}"
        c0, c1 = 0, 1
        cl0, cl1 = self.chips.index(k) * self.NC + c0, self.chips.index(k) * self.NC + c1
        # stage 1's slices of the tile have landed: both clusters read them (stage 1's own waits)
        wait = lambda c, name: list(self.own_wait[q])    # noqa: E731
        src = BingoMemSymbol(f"{T}_kvraw", offset=lt * self.KVD)
        # a chip's later tiles (more tiles than chips) take buffers of their own names: one handle a buffer
        l0 = lambda name, nb, s=s: self.cx_chip(k).at(c0).l1(f"{T}_kvs_{name}{s}", nb)    # noqa: E731
        l1 = lambda name, nb, s=s: self.cx_chip(k).at(c1).l1(f"{T}_kvp_{name}{s}", nb)    # noqa: E731
        kvd, crow, piece = l0("d", cb), l0("crow", 16 * KR * 2), l0("piece", KVP)
        pv = l1("pv", KVP - 16 * KR)              # [k_pe | value columns], the slot's bytes 16 KR .. KVP
        kvc = self.kvc[(k, c0)]
        w0 = wait(c0, "kvr_wait")
        w1 = wait(c1, "kvp_wait")
        ld = self.dm_copy(cl0, f"kvr_ld{s}", src, kvd, cb, list(w0))
        dq = self.simd_mul(cl0, f"kv_dq{s}", kvd, kvc, kvd, 16 * KR, [ld])
        cv = self.xd(cl0, f"kv_crm{s}", convert_args("D", "row_major", 16, KR, mesh, 2, kvd, crow), [dq])
        # c^T on cluster 1, first (its SIMD's long dequantisation ahead of k_pe): [seed beat | c^T [512, 16] FP16]
        kvd1, crow1, xt1 = l1("cd", cb), l1("crow", 16 * KR * 2), l1("xt", beat + 16 * KR * 2)
        ldv = self.dm_copy(cl1, f"kvv_ld{s}", src, kvd1, cb, list(w1))
        dqv = self.simd_mul(cl1, f"kvv_dq{s}", kvd1, self.kvc_pe[(k, c1)], kvd1, 16 * KR, [ldv])
        cvv = self.xd(cl1, f"kvv_crm{s}", convert_args("D", "row_major", 16, KR, mesh, 2, kvd1, crow1), [dqv])
        trv = self.node(k, c1, xdma, f"kvv_vt{s}", "__snax_bingo_kernel_xdma_transpose_2d",
                        SnaxBingoKernelXdmaTranspose2dArgs(crow1, xt1.view(beat), 16, KR, 2), [cvv])
        # k_pe on cluster 1: its own load and dequantisation, RoPE, D -> A, quantised into the slot
        kvc1 = self.kvc_pe[(k, c1)]
        pd, pe, pa = l1("d", n_pe * 2), l1("pe", n_pe * 2), l1("pa", n_pe * 2)
        rc = self.KC_ROPE + lt * self.KC_TILE
        ld1 = self.dm_copy(cl1, f"kvp_ld{s}", BingoMemSymbol(f"{T}_kvraw", offset=lt * self.KVD + cb), pd,
                           n_pe * 2, list(w1))
        xs, t1, t2 = l1("xs", n_pe * 2), l1("t1", n_pe * 2), l1("t2", n_pe * 2)
        dq1 = self.simd_mul(cl1, f"kvp_dq{s}", pd, kvc1.view(cb), pd, n_pe, [ld1])
        sw = [self.node(k, c1, dm, f"kvp_swap{h}{s}", "__snax_bingo_kernel_idma_1d_copy",
                        SnaxBingoKernelIdma1dCopyArgs(pd.view((1 - h) * half), xs.view(h * half), half),
                        [dq1]) for h in range(2)]
        m1 = self.simd_mul(cl1, f"kvp_cos{s}", pd, kvc1.view(rc), t1, n_pe, [dq1])
        m2 = self.simd_mul(cl1, f"kvp_sin{s}", xs, kvc1.view(rc + n_pe * 2), t2, n_pe, sw)
        ad = self.simd_add(cl1, f"kvp_rope{s}", t1, t2, pe, n_pe, [m1, m2])
        d2p = self.xd(cl1, f"kvp_pa{s}", d_to_a_args(mesh, 1, 4, 16, pe, pa), [ad])
        pq = self.quant(cl1, f"kvp_p8{s}", pa, pv, n_pe * 2, self.fbits(d.s_kpe.inv), [d2p])
        # c: RMSNorm row_major -> A (INT8) on cluster 0 into the key rows, col_major -> B (INT8) on cluster 1 into the
        # value columns, each with the INT8 scale applied
        ra = lambda i, o, la, sd=0: SnaxBingoKernelSimdRmsnormArgs(   # noqa: E731
            i, o, 16, KR, "row_major" if not sd else "col_major", "A" if not sd else "B", seed_addr=sd,
            out_i8=True, inv_scale_f32bits=sc, mesh=mesh, scratch_addr=la(
                "scrb" if sd else "scra", SnaxBingoKernelSimdRmsnormArgs(
                    0, 0, 16, KR, "col_major" if sd else "row_major", "B" if sd else "A", seed_addr=1,
                    out_i8=True, mesh=mesh).scratch_bytes()))
        kq = self.node(k, c0, simd, f"kv_rmsa{s}", "__snax_bingo_kernel_simd_rmsnorm", ra(crow, piece, l0), [cv])
        vq = self.node(k, c1, simd, f"kv_rmsb{s}", "__snax_bingo_kernel_simd_rmsnorm",
                       ra(xt1.view(beat), pv.view(n_pe), l1, xt1), [trv])
        self.kv_put[q] = self.node(k, c0, dm, f"kv_put{s}", "__snax_bingo_kernel_idma_put_flagged",
                                   SnaxBingoKernelIdmaPutFlaggedArgs(piece, on_root(f"{T}_kv", q * KVP), 16 * KR,
                                                                     on_root(f"{T}_kvnflag", q * 64), value=1),
                                   [kq])
        self.kvp_put[q] = self.node(k, c1, dm, f"kvp_put{s}", "__snax_bingo_kernel_idma_put_flagged",
                                    SnaxBingoKernelIdmaPutFlaggedArgs(
                                        pv, on_root(f"{T}_kv", q * KVP + 16 * KR), KVP - 16 * KR,
                                        on_root(f"{T}_kvpflag", q * 64), value=1), [pq, vq])

    # ---- checks -------------------------------------------------------------------------------------------------
    def piece_checks(self):
        """Three tiles' pieces in the L3 of the root, of the chip in the middle of the fold order and of the last."""
        d, T, R, n = self.d, self.TAG, self.ROOT, CHK
        far = self.order[len(self.order) // 2]
        checks = []
        for k in dict.fromkeys((R, far, self.order[-1])):
            for q in (0, self.NX // 2, self.NX - 1):
                img = np.asarray(d.kv_piece(16 * q)).view(np.uint8).reshape(-1)
                for nm, off in (("a", 0), ("p", 16 * KV_RANK), ("b", KVP - n)):
                    checks.append((f"kv_k{k:02x}_q{q}{nm}", BingoMemSymbol(f"{T}_kv", offset=q * KVP + off,
                                                                           chip_id=None if k == R else k),
                                   img[off:off + n], n))
        return checks


if __name__ == "__main__":
    run_driver("fa prefill stages 0 to 3 workload generator", Stage0123, app_name)
