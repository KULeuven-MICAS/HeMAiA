# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""The step's D2D traffic (link_report): the bytes each link carries and the stream floor."""

import fa_decode_data as fdd

from .consts import BR, HIDDEN, LAT, LINK_BW, OAR, PB, PK, PVB, QB, QC, QROW, XB, XFER


class KsReport:
    """FaDecodeKs.link_report, which build() prints."""

    def link_report(self):
        """The bytes each D2D link carries in the step (both directions): the rings (memory chip to chip), x's
        multicast, the root's multicasts (query, m*, o~, o8), the gathers into the root, the chains' hops (the maxima's
        beat and the partial) and y and the append into the root memory chip. Prints the busiest links and the stream
        floor (the busiest link at LINK_BW)."""
        R, nchip = self.ROOT, len(self.chips)
        grid = set(self.chips) | set(self.mem.values())
        load, kind = {}, {}

        def add(links, nb, what):
            for ln in links:
                load[ln] = load.get(ln, 0) + nb
                kind.setdefault(ln, {}).setdefault(what, 0)
                kind[ln][what] += nb

        for k in self.chips:
            add(fdd._route(self.mem[k], k), self.ring_bytes[k], "ring")
        add(fdd._bcast_tree(self.mem[R], grid), XB, "x")
        for nb, what in ((QB, "query"), (XFER, "m*"), (BR * LAT, "o~"), (nchip * OAR, "o8")):
            add(fdd._bcast_tree(R, grid), nb, what)
        for k in self.chips:
            add(fdd._route(k, R), QROW + OAR + self.NC * 4 * QC + 2 * QC * len(self.kva[k]), "gathers")
        hops = self.order[1:] + [R]
        for a, b in zip(hops, hops[1:]):
            add(fdd._route(a, b), PB + XFER, "chain")
        add(fdd._route(R, self.mem[R]), 2 * HIDDEN + PK + PVB, "y+append")
        top = sorted(load, key=lambda ln: -load[ln])[:4]
        self.floor = load[top[0]] / LINK_BW
        print(f"[{self.TAG}] streams: ring {sum(self.ring_bytes.values()):,} B over {nchip} chips "
              f"({min(self.ring_bytes.values()):,}..{max(self.ring_bytes.values()):,} B a chip); stream floor "
              f"{self.floor * 1e6:.1f} us (the busiest link at {LINK_BW / 1e9:.1f} GB/s)")
        for ln in top:
            parts = ", ".join(f"{w} {b:,}" for w, b in sorted(kind[ln].items(), key=lambda x: -x[1]))
            print(f"[{self.TAG}] link {ln[0]:02x}-{ln[1]:02x}: {load[ln]:,} B ({parts})")
