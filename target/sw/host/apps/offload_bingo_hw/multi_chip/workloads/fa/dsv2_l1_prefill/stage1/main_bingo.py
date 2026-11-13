#!/usr/bin/env python3
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""DeepSeek-V2-Lite's layer-1 MLA attention, stage 1 of the 256-token prefill on sixteen chiplets alone: stage 0 (x
from HBM into every chip's L3) and the keys by columns.

Every chip streams its own columns of W_KVA (two or three 16-column granules; INT4, whole pairs) through its weight
ring and makes them for every token. Each cluster takes the tiles t with t mod 2 = c: per tile it waits for the
tile's landed flag, loads it into L1 (70 % by the iDMA, 30 % by the xDMA: the two reach L3 through different wide
ports), runs one GEMM over the chip's columns, and puts the [16 x columns] slice into the tile owner's L3 behind a
flag of its own (kvrflag). Every owner waits for its tile's 16 slices. The INT4 balance: W_KVA's 18 pairs over 16
chips leave two chips with two pairs; the tiles are cut into four runs, each two-pair chip makes its second pair for
the first run only, and for each other run a helper chip (a one-pair chip near the same memory chip) makes it from
the pair's copy in its HBM slab and puts it behind a flag of its own (kvxflag), which the owner waits for too.

          stage 0                       x_hbm_bcast_post<n> / _land<n>, x_wait_t<t>      (stage0/main_bingo.py)
          every cluster (cluster c, its tiles q):
            Ld_wkvc_push, Ld_wkvc       DM: the chip's W_KVA run posted, the ring chunk into an L1 slab
            kx_ld_q<q>, kx_xld_q<q>     DM / xDMA: tile q of x into L1 behind its flag (x_wait)
            gkvc_q<q>                   GEMM: x tile . the chip's W_KVA columns (two x and two output buffers in turn)
            kvc_put_q<q>                DM: the slice into the owner's L3, behind its flag
            (a helper) Ld_wkvx, gkvx_q<q>, kvx_put_q<q>: the helped chip's second pair for its run of tiles
          every owner: kvr_wait_q<q> (and kvx_wait_q<q>): the tile's 16 slices (and the helper's) have landed
          P_fin<chip> -> S1_end -> fp_report, the checks

Checks: three windows of x in the L3 of the root, of the middle chip and of the last (as stage 0); the latent rows as
the D port leaves them (x8 . W_KVA, FP16, the D layout of each tile's [16, 576]) of three tiles in their owners' L3,
with each two-pair chip's second pair.

params.hjson: the platform, the tokens, the chunk and key-tile sizes, the weight chunk and ring sizes, the weights'
width."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from stage_common import HIDDEN, PG, TILE16, BingoMemSymbol, StageBase, app_name  # noqa: E402
from bingo_kernel_args import (SnaxBingoKernelIdma1dCopyArgs, SnaxBingoKernelIdmaPutFlaggedArgs,  # noqa: E402
                               SnaxBingoKernelWaitFlagsArgs, SnaxBingoKernelXdma1dCopyArgs)
from driver import run_driver  # noqa: E402

CHK = 2048                                  # bytes a checked window
PAIR_BYTES = HIDDEN * 16 * PG // 2          # an INT4 pair of W_KVA's columns, paired layout


class Stage1(StageBase):
    """Stage 1 alone (the module docstring)."""
    STAGE = 1
    X_XDMA = True
    KV_BALANCE_WAYS = 4

    def build(self):
        self.last_ld = {}
        for ls in self.ls:
            ls.strict = True
        go = self.setup()
        self.relays()
        self.x_puts = self.stage_x_bcast(go)
        last = {k: [] for k in self.chips}
        self.stage_keys(go, last)
        self.window_end(last)
        self.run_checks(self.x_checks() + self.kvraw_checks())

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
        """Every owner waits for its tiles' 16 slices, and for a helped tile its helpers' slices."""
        T, dm = self.TAG, self.roles["dm"]
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

    # ---- checks -------------------------------------------------------------------------------------------------
    def x_checks(self):
        """Three windows of x in the L3 of the root, of the chip in the middle of the fold order and of the last."""
        T, R, n = self.TAG, self.ROOT, CHK
        far = self.order[len(self.order) // 2]
        checks = []
        for k in dict.fromkeys((R, far, self.order[-1])):
            for t in (0, self.NX // 2, self.NX - 1):
                off = t * self.XT + (self.XT - n if t == self.NX - 1 else 0)
                checks.append((f"x_k{k:02x}_t{t}", BingoMemSymbol(f"{T}_xsrc", offset=off,
                                                                    chip_id=None if k == R else k),
                               self.xsrc[off:off + n].view(np.uint8), n))
        return checks

    def kvraw_checks(self):
        """The latent rows as the D port leaves them (x8 . W_KVA, FP16, the D layout of each tile's [16, 576]) of three
        tiles in their owners' L3, with each two-pair chip's second pair (made by that chip or its helper)."""
        d, T, R, n = self.d, self.TAG, self.ROOT, CHK
        raw = np.asarray(d.fp.d_port(d._mm(d.hq, d.wkva.q), d.k_x), dtype=np.float16)
        checks = []
        for q in (0, self.NX // 2, self.NX - 1):
            o = self.kv_owner[q]
            lt = self.kv_tiles[o].index(q)
            on = None if o == R else o
            img = self.lay.to_d(raw[16 * q:16 * q + 16], d.mesh).astype(np.float16).view(np.uint8).reshape(-1)
            for nm, off in (("a", 0), ("b", self.KVD - n)):
                checks.append((f"kvraw_q{q}{nm}", BingoMemSymbol(f"{T}_kvraw", offset=lt * self.KVD + off, chip_id=on),
                               img[off:off + n], n))
            for a, ga0 in sorted({(x[0], x[1]) for x in self.kvx.values()}):
                off, nx = ga0 * TILE16, 2 * TILE16
                checks.append((f"kvraw_q{q}p{a:02x}",
                               BingoMemSymbol(f"{T}_kvraw", offset=lt * self.KVD + off, chip_id=on),
                               img[off:off + nx], nx))
        return checks


if __name__ == "__main__":
    run_driver("fa prefill stage 1 workload generator", Stage1, app_name)
