# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""DBuild's checks: the root's host compares them byte for byte after the end stamp (README.md, CHECKS)."""

import numpy as np
from bingo_mem_handle import BingoMemSymbol

from moe_prefill_mc.common import ROUTE_B, lists_bytes, route_bytes

from .common import a_row_bytes


class Checks:

    def _route_bytes(self):
        return route_bytes(self.d, self.T)

    def _lists_bytes(self, k):
        return lists_bytes(self.d, self.held[k])

    def _checks(self):
        """The route table's entry on the root, the root cluster 1's row lists, the root cluster 0's A8 row of its
        shared piece and of its first routed piece (both clusters' columns, cluster 1's after the exchange), and y in
        HBM."""
        R, Tg = self.ROOT, self.TAG
        checks = [("route", BingoMemSymbol(f"{Tg}_rland", chip_id=R), self._route_bytes(), ROUTE_B)]
        lb = self._lists_bytes(R)
        checks.append(("lists", self.stash("lists_root_c1", R, 1, self.B[1]["lists"], lb.size, []), lb, lb.size))
        for pi, name in ((0, "a8sh"), (1, "a8_p1")):
            gold = self.a8_gold(R, pi)
            checks.append((name, self.stash(f"{name}_root", R, 0, self.B[0]["a8"][pi], gold.size, []), gold, gold.size))
        checks.append(("y_hbm", self.h_yhbm, self.y_row.view(np.uint8), 2 * self.H))
        self.run_checks(checks)

    def a8_gold(self, k, pi):
        """The bytes of the token's A8 row of chip k's piece pi as its clusters hold it (both clusters' columns): the
        A8 row in a_row (2 B a column, the free bytes zero)."""
        pc = self.pieces[k][pi]
        a8 = self.d.a8_shared() if pc.e is None else self.d.a8_expert(pc.e)
        row = np.ascontiguousarray(a8[0, pc.k0:pc.k0 + pc.n]).astype(np.int8)
        return a_row_bytes(row).view(np.uint8)
