# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""FaBuild's inputs of a chunk: a cluster's wait for the keys' pieces before its sweep."""

from bingo_kernel_args import SnaxBingoKernelWaitFlagsArgs
from bingo_mem_handle import BingoMemSymbol


class FaInputs:

    def kv_wait(self, cl, i):
        """Cluster cl's wait, before chunk i's sweep, until the keys of chunks 0 .. i have landed in its chip's L3: the
        root's multicasts' flags, or on the root its pieces' own. The puts are ordering edges; the flags prove the
        landing."""
        k, c, R, T = self.chip(cl), self.local(cl), self.ROOT, self.TAG
        if k == R:
            n = (i + 1) * (self.C // 16)
            puts = [self.kv_put[q] for q in range(n)]
            args = SnaxBingoKernelWaitFlagsArgs(BingoMemSymbol(f"{T}_kvnflag"), n, 1)
        else:
            puts = self.kv_mc[:i + 1]
            args = SnaxBingoKernelWaitFlagsArgs(BingoMemSymbol(f"{T}_kvflag"), i + 1, 1)
        w = self.node(k, c, self.roles["dm"], f"kv_wait_i{i}", "__snax_bingo_kernel_wait_flags", args,
                      [t for t in puts if self.flag_dep(t, k, c)])
        for t in puts:
            if not self.flag_dep(t, k, c):
                self.dfg.bingo_add_order_edge(t, w)
        return w
