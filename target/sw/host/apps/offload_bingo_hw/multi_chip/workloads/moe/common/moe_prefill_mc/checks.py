# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""MBuild's checks (the root's host compares them byte for byte after the end stamp) and its finish: the barrier
schedule's check and the args phases before the compile (README.md, CHECKS)."""

import numpy as np
from bingo_mem_handle import BingoMemSymbol

from .common import BLK, MESH3, ROUTE_B, lists_bytes, route_bytes


class Checks:

    def _route_bytes(self):
        return route_bytes(self.d, self.T)

    def _lists_bytes(self, k):
        return lists_bytes(self.d, self.held[k])

    def _checks(self):
        """The route table (the root's landing area, the whole table), the root cluster 1's row lists, the root
        cluster 0's A8 of its first expert slot and of the shared MLP (row 0), and three windows of Y."""
        R, T, Tg = self.ROOT, self.T, self.TAG
        checks = []
        n = 2048
        checks.append(("route", BingoMemSymbol(f"{Tg}_rland", chip_id=R), self._route_bytes(), T * ROUTE_B))
        lb = self._lists_bytes(R)
        got = self.stash("lists_root_c1", R, 1, self.B[1]["lists"], lb.size, [])
        checks.append(("lists", got, lb, lb.size))
        e = self.held[R][0]
        a8 = self.d.a8_expert(e)
        img = np.zeros((a8.shape[0], self.PE), dtype=np.int8)
        img[:, :self.IE] = a8
        # row 0's columns only: the rest of a row's pitch is never written, and a DMA write of
        # unwritten (X) TCDM fails mem_wide_narrow_mux's DMAWriteDataCorrect in simulation
        nb = min(n, self.IE)
        got = self.stash("a8_root_s0", R, 0, self.B[0]["a8e"][0], nb, [])
        checks.append(("a8_s0", got, img.reshape(-1)[:nb].copy(), nb))
        # the shared MLP's A8 on the root's cluster 0, row 0: its own columns and, after the exchange, cluster 1's
        ns = BLK * self.shk[R]
        sh = np.ascontiguousarray(self.d.a8_shared()[0, BLK * self.sho[R]:BLK * self.sho[R] + ns]).astype(np.int8)
        got = self.stash("a8sh_root", R, 0, self.B[0]["a8sh"], ns, [])
        checks.append(("a8sh_r0", got, sh.view(np.uint8), ns))
        img = np.concatenate([self.lay.to_d(self.y_raw[:, o0:o0 + w], MESH3).astype(np.float16)
                              for c in range(self.NC) for o0, w in self.out_slices(c)]).view(np.uint8)
        for name, off in (("y_a", 0), ("y_mid", img.size // 2), ("y_b", img.size - n)):
            checks.append((name, self.ysym(off, chip_id=R), img[off:off + n], n))
        self.run_checks(checks)

    def check_window_local(self):
        """sync barrier: no dependency edge crosses chips between the start barrier and the end stamp; only the
        barriers synchronise the chips there (data a chip waits for by its flag is an order edge, not a dependency).
        Returns the offending edges."""
        import networkx as nx
        pre = set().union(*(nx.ancestors(self.dfg, t) for t in self.go.values()))
        post = nx.descendants(self.dfg, self.end) | {self.end}
        return [(u, v) for u, v in self.dfg.edges()
                if u.assigned_chiplet_id != v.assigned_chiplet_id and u not in pre and u not in post and v not in post]

    def finish(self, app_name):
        if self.sync == "barrier":
            if self.barrier_fence:                          # the fences before the args phases' loads exist
                self.fence_barriers()
                self.barrier_fence = False
            cross = self.check_window_local()
            if cross:
                raise ValueError(f"sync barrier: {len(cross)} dependency edges cross chips inside the window, e.g. "
                                 + "; ".join(f"{u.node_name} -> {v.node_name}" for u, v in cross[:4]))
            print(f"[{self.TAG}] sync barrier ({self.barrier_at}): {self.n_barrier} barriers a chip, no edge crosses "
                  f"chips inside the window")
        if self.args_phases:
            self._assign_phases()
        super().finish(app_name)
