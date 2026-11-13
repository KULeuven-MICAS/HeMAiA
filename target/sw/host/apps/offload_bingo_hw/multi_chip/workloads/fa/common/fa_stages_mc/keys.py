# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""Stage 1, the keys by columns: each cluster's tiles of x from the chip's L3 into L1 behind the tile's flag, loaded by
the iDMA and the xDMA at once; the chip's W_KVA columns from the ring; a GEMM a tile; each slice into the tile owner's
L3 behind a flag; every owner waits for its tile's 16 slices (and a helped tile's helper's slice)."""

from bingo_kernel_args import (SnaxBingoKernelIdma1dCopyArgs, SnaxBingoKernelIdmaPutFlaggedArgs,
                               SnaxBingoKernelWaitFlagsArgs, SnaxBingoKernelXdma1dCopyArgs)
from bingo_mem_handle import BingoMemSymbol
from fa_prefill_mc import HIDDEN, TILE16

PG = 2                                      # granules an INT4 pair of W_KVA's columns
PAIR_BYTES = HIDDEN * 16 * PG // 2          # an INT4 pair of W_KVA's columns, paired layout


class FaStagesKeys:

    def kv_helpers(self):
        """params kv_balance (INT4): W_KVA's 18 INT4 pairs over 16 chips leave two chips with two pairs, twice the keys'
        GEMM work. The tiles are cut into kv_balance_ways equal runs (default 2): each two-pair chip makes its second
        pair for the first run only; for every other run a helper, a one-pair chip near the same memory chip (not the
        root), makes it from the pair's copy in that chip's HBM slab (the paired layout keeps each pair whole: the
        second pair is the slab's second half). Sets kvx {helper: (two-pair chip, the pair's first granule, flag slot,
        its tiles)}, kv_two and kv_half (the tiles the two-pair chips leave to helpers)."""
        self.kvx, self.kv_two, self.kv_half = {}, [], set()
        if not self.p.get("kv_balance", False):
            return
        if self.wbits != 4:
            raise ValueError("params kv_balance: needs wbits 4")
        nx = self.T // 16
        ways = int(self.p.get("kv_balance_ways", 2))
        if ways < 2 or nx % ways:
            raise ValueError(f"params kv_balance_ways={ways}: at least 2, dividing the {nx} tiles")
        run = nx // ways
        width = {k: self.kvg[k][1] - self.kvg[k][0] for k in self.chips}
        if set(width.values()) - {PG, 2 * PG}:
            raise ValueError(f"params kv_balance: chips with one or two INT4 pairs only, not {sorted(set(width.values()))}")
        dist = lambda a, b: abs((a >> 4) - (b >> 4)) + abs((a & 15) - (b & 15))     # noqa: E731
        self.kv_two = [k for k in self.chips if width[k] == 2 * PG]
        free = [k for k in self.chips if width[k] == PG and k != self.ROOT]
        if len(free) < len(self.kv_two) * (ways - 1):
            raise ValueError(f"params kv_balance_ways={ways}: {len(self.kv_two) * (ways - 1)} helpers, {len(free)} chips")
        for e, a in enumerate(self.kv_two):
            for w in range(1, ways):
                h = min(free, key=lambda k: (self.mem[k] != self.mem[a], dist(k, self.mem[a]), k))
                free.remove(h)
                self.kvx[h] = (a, self.kvg[a][0] + PG, e, set(range(w * run, (w + 1) * run)))
        self.kv_half = set(range(run, nx))
        print("[fp] kv_balance: " + ", ".join(f"chip {h:02x} makes chip {a:02x}'s second pair for tiles "
                                                f"{min(ts)}-{max(ts)}" for h, (a, _, _, ts) in self.kvx.items()))

    def stage_keys(self, go, last):
        """Every cluster's key tiles (keys_cluster), then every owner's waits for its tiles' slices (owner_waits)."""
        self.kvc_put, self.kvx_put = {}, {}
        for j, k in enumerate(self.chips):
            for c in range(self.NC):
                self.keys_cluster(j, k, c, go, last)
        self.owner_waits(last)

    def key_buffers(self, k, c, tiles, nb):
        """Cluster c of chip k's L1 for its key tiles: two x and two output buffers (tile n takes n % 2); keys_by_chunk:
        W_KVA's own buffer, and the tiles of the chunks after the last one made (late, after the window) and those of
        the later chunks made inside the window (mid: one x and one output buffer, or kx_mid_double two of each) take
        buffers of their own; kx_per_chunk: every chunk after the first its own (made as they are first asked for).
        Returns (l1, wkva, late_from, mid, pmid, XB, OB, wl): XB(n) and OB(n) tile n's buffers, pmid(n) the mid tile
        before n that shares its buffers, wl the late tiles' W_KVA copy."""
        T, dbl = self.TAG, self.kx_mid_double
        l1 = lambda name, size: self.cx_chip(k).at(c).l1(f"{T}_kc_{name}", size)    # noqa: E731
        xb, ob = [l1(f"x{b}", self.XT) for b in range(2)], [l1(f"o{b}", nb) for b in range(2)]
        wkva = l1("wkva", self._src(k, 0)[1]) if self.keys_by_chunk else None
        late_from = self.NCH * (self.C // 16) if self.keys_by_chunk else self.NX
        mid = lambda n: self.keys_by_chunk and self.C // 16 <= tiles[n] < late_from      # noqa: E731
        wl = None
        if self.keys_by_chunk:
            xl, ol = [l1(f"lx{b}", self.XT) for b in range(2)], [l1(f"lo{b}", nb) for b in range(2)]
            wl = l1("lwkva", self._src(k, 0)[1])
            if dbl:
                xm, om = [l1(f"mx{b}", self.XT) for b in range(2)], [l1(f"mo{b}", nb) for b in range(2)]
            else:
                xm, om = l1("mx", self.XT), l1("mo", nb)
        XB = lambda n: (xl[n % 2] if tiles[n] >= late_from else (xm[n % 2] if dbl else xm) if mid(n)  # noqa: E731
                        else xb[n % 2])
        OB = lambda n: (ol[n % 2] if tiles[n] >= late_from else (om[n % 2] if dbl else om) if mid(n)  # noqa: E731
                        else ob[n % 2])
        # the previous mid tile, which shares the single buffers (kx_mid_double: none, the tile two before covers it)
        pmid = lambda n: None if dbl else max((m for m in range(n) if mid(m)), default=None)   # noqa: E731
        if self.kx_per_chunk:
            PC, hbuf = self.C // 16, {}

            def hb(n):
                h = tiles[n] // PC
                if h < 1:
                    return None
                if h not in hbuf:
                    nx = 2 if dbl else 1
                    hbuf[h] = ([l1(f"hx{b}_h{h}", self.XT) for b in range(nx)],
                               [l1(f"ho{b}_h{h}", nb) for b in range(nx)])
                return hbuf[h]
            XB0, OB0 = XB, OB
            XB = lambda n: hb(n)[0][n % 2 if dbl else 0] if hb(n) else XB0(n)  # noqa: E731
            OB = lambda n: hb(n)[1][n % 2 if dbl else 0] if hb(n) else OB0(n)  # noqa: E731
        return l1, wkva, late_from, mid, pmid, XB, OB, wl

    def keys_cluster(self, j, k, c, go, last):
        """Cluster c of chip k (chips[j]): W_KVA first on the in-order DM core (the copy depends on the chip's start, so a
        ring load queued ahead of the start stamp is a dependency the hang check sees), then the first tiles of x;
        a GEMM a tile into its owner's L3; a helper's tiles of a two-pair chip's second pair (kv_balance)."""
        d, T, dm, xdma = self.d, self.TAG, self.roles["dm"], self.roles["xdma"]
        gr0, gr1 = self.kvg[k]
        nb = (gr1 - gr0) * TILE16
        helps = self.kvx.get(k)
        cl = j * self.NC + c
        tiles = self.cl_tiles(c)
        l1, wkva, late_from, mid, pmid, XB, OB, wl = self.key_buffers(k, c, tiles, nb)
        ldl = {}

        def wsrc(n):
            # tile n's W_KVA operand and the copy it waits for
            if self.kx_per_chunk and tiles[n] // (self.C // 16) >= 1:
                h = tiles[n] // (self.C // 16)
                if ("w", h) not in ldl:
                    wb_ = l1(f"hwkva_h{h}", self._src(k, 0)[1])
                    ldl[("w", h)] = (wb_, self.node(k, c, dm, f"Ld_wkvc_q{tiles[n]}", "__snax_bingo_kernel_idma_1d_copy",
                                                    SnaxBingoKernelIdma1dCopyArgs(BingoMemSymbol(f"{T}_wkvc3"), wb_,
                                                                                  self._src(k, 0)[1]),
                                                    [go[k]]))
                return ldl[("w", h)]
            if tiles[n] < late_from:
                return slab, ld
            if not ldl:
                ldl[0] = self.node(k, c, dm, f"Ld_wkvc_q{tiles[n]}", "__snax_bingo_kernel_idma_1d_copy",
                                   SnaxBingoKernelIdma1dCopyArgs(self.h_wkvc[k].address, wl, self._src(k, 0)[1]),
                                   [go[k]])
            return wl, ldl[0]
        gm, put, ld_x, gmx, putx = {}, {}, {}, {}, []
        xx = self.x_xdma

        def kx(n):
            # the tile's wait right before its load: the in-order DM core blocks on no later tile; the x buffer is free
            # once the tile two before it has had its GEMMs
            q = tiles[n]
            after = [self.x_wait(k, c, q, go)] + ([gm[n - 2]] if n >= 2 else []) + \
                ([gmx[n - 2]] if n - 2 in gmx else [])
            if mid(n) and pmid(n) is not None:
                after += [gm[pmid(n)]] + ([gmx[pmid(n)]] if pmid(n) in gmx else [])
            a = self.XT - xx
            src = lambda off: BingoMemSymbol(f"{T}_xsrc", offset=q * self.XT + off)     # noqa: E731
            parts = []
            if a:
                parts.append(self.node(k, c, dm, f"kx_ld_q{q}", "__snax_bingo_kernel_idma_1d_copy",
                                       SnaxBingoKernelIdma1dCopyArgs(src(0), XB(n), a), after))
            if xx:
                parts.append(self.node(k, c, xdma, f"kx_xld_q{q}", "__snax_bingo_kernel_xdma_1d_copy",
                                       SnaxBingoKernelXdma1dCopyArgs(src(a), XB(n).view(a), xx), after))
            return parts

        hx = None
        if helps:
            hx = dict(xs=l1("xs", PAIR_BYTES), obx=[l1(f"ox{b}", PG * TILE16) for b in range(2)], xsl={},
                      ld=None)
        ld, si, slab = self.load(cl, 0, "Ld_wkvc", deps=[go[k]], dst=wkva)
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
            wb, wld = wsrc(n)
            if n not in ld_x:
                ld_x[n] = kx(n)
            own = [put[pmid(n)]] if mid(n) and pmid(n) is not None else []
            gm[n] = self.gmm(cl, f"gkvc_q{q}", XB(n), wb, OB(n), 1, HIDDEN // 4, ng, d.k_x,
                             [wld] + ld_x[n] + ([put[n - 2]] if n >= 2 else []) + own)
            # tile n + 1's x load made right behind tile n's GEMM, ahead of tile n's put: where each core's tasks keep
            # the order they were made in (the schedule's fallback), the DM core loads it while GEMM n runs (it needs
            # only GEMM n - 1's buffer) instead of after GEMM n's put
            if n + 1 < len(tiles) and n + 1 not in ld_x:
                ld_x[n + 1] = kx(n + 1)
            put[n] = self.node(k, c, dm, f"kvc_put_q{q}", "__snax_bingo_kernel_idma_put_flagged",
                               SnaxBingoKernelIdmaPutFlaggedArgs(
                                   OB(n), on(f"{T}_kvraw", lt * self.KVD + gr0 * TILE16), ng * TILE16,
                                   on(f"{T}_kvrflag", (lt * len(self.chips) + j) * 64), value=1), [gm[n]])
            self.kvc_put[(q, k)] = put[n]
            if helps and q in helps[3]:
                gmx[n] = self.helped_tile(k, c, cl, q, lt, on, helps, hx, late_from, XB(n), ld_x[n], putx, go)
            if n + 2 < len(tiles) and not mid(n + 2):
                ld_x[n + 2] = kx(n + 2)
        if si is not None:
            self.ls[cl].release(si, [gm[n] for n in range(len(tiles))])    # every tile's GEMM reads it
        last[k] += list(put.values()) + list(putx)

    def helped_tile(self, k, c, cl, q, lt, on, helps, hx, late_from, x_buf, x_lds, putx, go):
        """kv_balance: helper chip k's GEMM of tile q (its n-th) over the helped chip's second pair, and its put into
        the tile owner's L3 behind the pair's flag. kc_xs_late: the tiles after the window take a copy of the pair of
        their own; kx_per_chunk: every chunk after the first its own (from the chip's L3). Returns the GEMM."""
        d, T, dm = self.d, self.TAG, self.roles["dm"]
        a, ga0, e, _ = helps
        m = len(putx)
        xsb, xld, xsl = hx["xs"], hx["ld"], hx["xsl"]
        if self.kc_xs_late and q >= late_from:
            if "b" not in xsl:
                xsl["b"] = self.cx_chip(k).at(c).l1(f"{T}_kc_xsl", PAIR_BYTES)
                xsl["ld"] = self.node(k, c, dm, f"Ld_wkvxl_q{q}", "__snax_bingo_kernel_idma_1d_copy",
                                      SnaxBingoKernelIdma1dCopyArgs(
                                          self.h_wkvc[helps[0]].address + PAIR_BYTES, xsl["b"], PAIR_BYTES), [go[k]])
            xsb, xld = xsl["b"], xsl["ld"]
        if self.kx_per_chunk and q // (self.C // 16) >= 1:
            h = q // (self.C // 16)
            if ("xs", h) not in xsl:
                b_ = self.cx_chip(k).at(c).l1(f"{T}_kc_hxs_h{h}", PAIR_BYTES)
                xsl[("xs", h)] = (b_, self.node(
                    k, c, dm, f"Ld_wkvxl_q{q}", "__snax_bingo_kernel_idma_1d_copy",
                    SnaxBingoKernelIdma1dCopyArgs(BingoMemSymbol(f"{T}_wkvx3"), b_, PAIR_BYTES), [go[k]]))
            xsb, xld = xsl[("xs", h)]
        gmx = self.gmm(cl, f"gkvx_q{q}", x_buf, xsb, hx["obx"][m % 2], 1, HIDDEN // 4, PG, d.k_x,
                       [xld] + x_lds + ([putx[m - 2]] if m >= 2 else []))
        putx.append(self.node(k, c, dm, f"kvx_put_q{q}", "__snax_bingo_kernel_idma_put_flagged",
                              SnaxBingoKernelIdmaPutFlaggedArgs(
                                  hx["obx"][m % 2], on(f"{T}_kvraw", lt * self.KVD + ga0 * TILE16),
                                  PG * TILE16, on(f"{T}_kvxflag", (lt * len(self.kv_two) + e) * 64),
                                  value=1), [gmx]))
        self.kvx_put[(q, k)] = putx[-1]
        return gmx

    def owner_waits(self, last):
        """Every owner waits for its tiles' 16 slices, and for a helped tile its helpers' slices: the end of stage 1
        (own_wait {tile: the waits}: where a chained stage 2 starts)."""
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
