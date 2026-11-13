# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""Where X and Y live in L3, X's multicast from HBM and its fetches into every cluster's L1."""

from bingo_kernel_args import (HostBingoKernelHbmBcastArgs, HostBingoKernelHbmUcastArgs,
                               SnaxBingoKernelIdmaFetchFlaggedArgs)
from bingo_mem_handle import BingoMemSymbol

import chiplet_mc as cm


class PX:
    """X: its landing area, its casts from the root memory chip's HBM, its fetches; and Y_raw's places in L3."""

    # ---- places in L3 -----------------------------------------------------------------------
    def _xland(self, off, chip_id=None):
        """Where X lands in every chip's L3, at byte `off`; params l3_overlay: in the second half of its pass's region."""
        if self.l3_overlay:
            return BingoMemSymbol(f"{self.TAG}_xy", offset=off // self.XB * self.YB + self.XB + off % self.XB,
                                  chip_id=chip_id)
        return BingoMemSymbol(f"{self.TAG}_xland", offset=off, chip_id=chip_id)

    def _ysym(self, off, **kw):
        """Y_raw in the root's L3 at byte `off`; params l3_overlay: over the X buffers, at the same offset."""
        return BingoMemSymbol(f"{self.TAG}_{'xy' if self.l3_overlay else 'y'}", offset=off, **kw)

    def y_owner(self, q):
        """y_scatter: (chip, its tile index) of 16-row tile q of Y_raw."""
        return self.chips[q % len(self.chips)], q // len(self.chips)

    def yt_sym(self, q, off=0, chip_id=None):
        """y_scatter: tile q of Y_raw in its owner's L3 at byte `off` (chip_id: as another chip names it)."""
        return BingoMemSymbol(f"{self.TAG}_yt", offset=self.y_owner(q)[1] * self.YT_B + off, chip_id=chip_id)

    # ---- X from HBM ---------------------------------------------------------------------------
    def x_hbm_bcasts(self, go):
        """The root's host posts into its memory chip's push engine, one cast a pass (ChipletBuild.hbm_cast: a post a
        group of tiles): the pass's X a tile at a time into every chip's L3, each tile behind its flag, ahead of the
        root's weight rings in that engine's queue. Returns the casts (HbmCast), one a pass."""
        R, T = self.ROOT, self.TAG
        m = self.mem[R]
        one = self.h_seq[m].address + 64                    # a word holding 1 in the memory chip's HBM: the flags' value
        out = []
        for i in range(self.NI):
            src, dst = self.h_xhbm.address + i * self.XB, self._xland(i * self.XB)
            flag = BingoMemSymbol(f"{T}_xtflag", offset=64 * i * self.TPP)
            if self.x_cast == "unicast":                # params x_cast unicast: a copy a chip, in the fold order
                prog = BingoMemSymbol(f"{T}_xuprog", offset=64 * i * len(self.chips) * self.TPP)
                kargs = HostBingoKernelHbmUcastArgs(src, dst, flag, prog, one, 1, self.XT, self.TPP, m, self.engine[R],
                                                    self.order)
            else:
                kargs = HostBingoKernelHbmBcastArgs(src, dst, flag, one, 1, self.XT, self.TPP, m, self.engine[R])
            out.append(self.hbm_cast(R, f"x_hbm_bcast_i{i}_k{R:02x}", kargs, after=[go[R]] + out[-1:]))
        return out

    def x_post(self, cast, t, k):
        """The task of `cast` (x_hbm_bcasts: an HbmCast, or the task a layer puts X with) that queues pass-local tile t
        for chip k: what a fetch of that tile spins behind (its order edge's producer)."""
        if not isinstance(cast, cm.HbmCast):
            return cast
        return cast.tile(t, chip=k if self.x_cast == "unicast" else None)

    # ---- X into L1 ------------------------------------------------------------------------------
    def _x_fetch(self, cl, i, off, nb, src, after, name, block=False):
        """Bytes off .. off + nb of X_i into cluster cl's L1, behind their tiles' flags. `src` is pass i's cast
        (x_hbm_bcasts); the order edge comes from the task that queues the last of those tiles."""
        T, dm = self.TAG, self.roles["dm"]
        k, c = self.chip(cl), self.local(cl)
        xb = self.xbuf[cl][0]
        t0 = (i * self.XB + off) // self.XT
        flag, nf = BingoMemSymbol(f"{T}_xtflag", offset=64 * t0), nb // self.XT
        nd = self.node(k, c, dm, name, "__snax_bingo_kernel_idma_fetch_flagged",
                       SnaxBingoKernelIdmaFetchFlaggedArgs(self._xland(i * self.XB + off), xb.view(off) if block else xb,
                                                           nb, flag, nf, 1),
                       list(after))
        self.dfg.bingo_add_order_edge(self.x_post(src, off // self.XT + nf - 1, k), nd)
        return nd

    def x_wait_block(self, cl, b, src, after=()):
        """params x_blocks: row block b of the first pass's X into cluster cl's L1, behind its flags."""
        nb = self.XB // self.x_blocks
        return self._x_fetch(cl, 0, b * nb, nb, src, after, f"x_wait_i0b{b}", block=True)

    def stage_x(self, i, src, after):
        """X_i into every cluster's L1 buffer, behind its flags. {cl: the task after which it is there}."""
        return {cl: self._x_fetch(cl, i, 0, self.XB, src, after.get(cl, []), f"x_wait_i{i}") for cl in range(self.N_CL)}

    def first_x(self, after_x):
        """The first pass's X into every cluster's L1: whole, or (params x_blocks) its first row block, the others
        fetched behind the first weight load (stage_gu). {cl: the task after which it is there}."""
        if self.x_blocks > 1:
            return {cl: self.x_wait_block(cl, 0, self.x_bc[0], after_x[cl]) for cl in range(self.N_CL)}
        return self.stage_x(0, self.x_bc[0], after_x)
