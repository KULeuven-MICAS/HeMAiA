# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""Stage 0: x from the HBM of the root's memory chip into every chip's L3 (that memory chip's push engine multicasts
it, queued by the root's host), every cluster's waits for its tiles, and a cluster's rows of a chunk from its chip's
L3."""

from bingo_kernel_args import (HostBingoKernelHbmBcastArgs, SnaxBingoKernelIdma1dCopyArgs,
                               SnaxBingoKernelWaitFlagsArgs)
from bingo_mem_handle import BingoMemSymbol


class FaStagesX:

    def cl_tiles(self, c):
        """The tiles of x cluster c makes keys for (stage 1)."""
        return [t for t in range(self.NX) if t % self.NC == c]

    def stage_x_bcast(self, go):
        """x's multicast from HBM: ChipletBuild.hbm_cast, a part's posts and waits (HbmCast; first_of, done_of,
        queued_of name its tasks), each part after the one before; keys_by_chunk: chunk 0's tiles, chunk 1's, then the
        rest, each a cast of its own (_cluster_x_order places the root's ring runs among them). The root's ring pushes
        on the same engine: its posts queue behind the multicast's, or with x_pf_split by the root ring's boundaries."""
        R, T = self.ROOT, self.TAG
        m = self.mem[R]
        P = self.C // 16
        cuts = ([0, P] + ([2 * P] if self.NX > 2 * P else []) + [self.NX]) if self.keys_by_chunk else [0, self.NX]
        puts = []
        for a, b in zip(cuts, cuts[1:]):
            puts.append(self.hbm_cast(
                R, f"x_hbm_bcast_k{R:02x}" + (f"_t{a}" if a else ""),
                HostBingoKernelHbmBcastArgs(
                    self.h_xhbm.address + a * self.XT, BingoMemSymbol(f"{T}_xsrc", offset=a * self.XT),
                    BingoMemSymbol(f"{T}_xtflag", offset=a * 64), self.h_seq[m].address + 64, 1, self.XT, b - a, m,
                    self.engine[R]),
                after=[go[R]] + puts[-1:],
                ahead_of_rings=not self.pf_split))
        return puts

    def x_wait(self, k, c, t, go, name=None):
        """Chip k cluster c waits for tile t's landed flag in its chip's L3 (the flag proves the multicast's data; the
        task that queues the tile is an ordering edge)."""
        put = self.x_puts[min(t // (self.C // 16), len(self.x_puts) - 1) if self.keys_by_chunk else 0]
        w = self.node(k, c, self.roles["dm"], name or f"x_wait_t{t}", "__snax_bingo_kernel_wait_flags",
                      SnaxBingoKernelWaitFlagsArgs(BingoMemSymbol(f"{self.TAG}_xtflag", offset=t * 64), 1, 1), [go[k]])
        self.dfg.bingo_add_order_edge(self.queued_of(put), w)
        return w

    def x_rows(self, cl, i, after):
        """Cluster cl's rows of chunk i from the chip's L3, where stage 0's multicast lands them, once their four tiles'
        flags are in (x_wait)."""
        T, k, c = self.TAG, self.chip(cl), self.local(cl)
        half = self.XB // self.NC
        t0 = (i * self.XB + self.xh(i, c) * half) // self.XT
        waits = [self.x_wait(k, c, t, self.go, name=f"x_wait_i{i}t{t - t0}") for t in range(t0, t0 + half // self.XT)]
        xb = self.bufs(cl, i)["xrows"]
        return self.node(k, c, self.roles["dm"], f"x_copy_i{i}", "__snax_bingo_kernel_idma_1d_copy",
                         SnaxBingoKernelIdma1dCopyArgs(BingoMemSymbol(f"{T}_xsrc", offset=i * self.XB + self.xh(i, c) * half),
                                                       xb, half), list(after) + waits)
