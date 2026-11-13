# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""DBuild's placement: the pieces of the token's experts and of the shared MLP on the chips, each piece's columns on
a chip's two clusters, the gate|up chunks and the output slices (README.md, THE PLACEMENT and THE CLUSTERS)."""

from bingo_kernel_args import SnaxBingoKernelMoeDispatchArgs

import moe_prefill_data as md
from moe_prefill_mc.common import BLK, beat_aligned

from .common import Piece


class Placement:

    def _place(self, p):
        """self.pieces {chip: [shared piece, routed pieces]}, self.held {chip: the routed pieces' experts}, and each
        piece's split over the chip's clusters."""
        d, nch = self.d, len(self.chips)
        self.route_ids = [int(e) for e in d.ids[0]]       # the generator's route of the token, slot order
        if len(set(self.route_ids)) != md.TOP_K or any(int(d.cnt[e]) != 1 for e in self.route_ids):
            raise ValueError(f"the token's route {self.route_ids}: {md.TOP_K} distinct experts")
        if self.SHB % nch:
            raise ValueError(f"{self.SHB} shared blocks over {nch} chips")
        self.shk = {k: self.SHB // nch for k in self.chips}
        self.sho = {k: i * self.SHB // nch for i, k in enumerate(self.chips)}
        R = self.routed_split = int(p.get("routed_split", 8))
        if R < 1 or self.EB % R or (md.TOP_K * R) % nch:
            raise ValueError(f"params routed_split={R}: dividing an expert's {self.EB} column blocks, {md.TOP_K} x it "
                             f"a multiple of the {nch} chips")
        self.PB = self.EB // R                            # column blocks a routed piece
        self.pieces = {k: [Piece(None, BLK * self.sho[k], BLK * self.shk[k])] for k in self.chips}
        for s, e in enumerate(self.route_ids):
            for i in range(R):
                k = self.chips[(s * R + i) % nch]
                self.pieces[k].append(Piece(e, BLK * i * self.PB, BLK * self.PB))
        self.held = {}
        for k in self.chips:
            es = [pc.e for pc in self.pieces[k][1:]]
            if len(set(es)) != len(es) or not 0 < len(es) <= SnaxBingoKernelMoeDispatchArgs.MAX_SLOTS:
                raise ValueError(f"chip {k:02x} holds pieces of experts {es}: distinct, "
                                 f"{SnaxBingoKernelMoeDispatchArgs.MAX_SLOTS} at most (moe_dispatch's slots)")
            self.held[k] = es
            self._split(self.pieces[k], 1 if k == self.ROOT else 0)
        for k in self.chips:
            print(f"[md] chip {k:02x}: " + ", ".join(f"{pc} ({pc.b0}+{pc.nb - pc.b0} blocks)" for pc in self.pieces[k]))

    def _split(self, pieces, extra0):
        """Each piece's blocks on cluster 0 (b0): an even number, at least 2, leaving cluster 1 at least one. For each
        such split of the shared piece, the routed pieces start at half (rounded down to even) and take moves of two
        blocks from the busier cluster, each to the piece it leaves the most even, while that evens the chip's two
        clusters. The split kept: the two clusters' blocks the most even, then the most route-free chunks in the
        fewer of the two rings (cluster 0's has `extra0` more: the router's), then the shared piece the most even."""
        for pc in pieces:
            if pc.nb < 3:
                raise ValueError(f"a piece of {pc.nb} column blocks: at least 3 (cluster 0 an even number, cluster 1 "
                                 f"at least one)")
        sh, routed = pieces[0], pieces[1:]
        nchunks = lambda nb: len(self.chunks(BLK * nb))
        best = None
        for s0 in range(2, sh.nb, 2):
            b0 = [max(2, pc.nb // 2 - (pc.nb // 2) % 2) for pc in routed]
            for _ in range(sum(pc.nb for pc in routed)):
                n0 = s0 + sum(b0)
                gap = n0 - (sh.nb + sum(pc.nb for pc in routed) - n0)
                step = -2 if gap > 0 else 2
                cand = [i for i, pc in enumerate(routed) if 2 <= b0[i] + step <= pc.nb - 1]
                if gap == 0 or not cand or abs(gap + 2 * step) >= abs(gap):
                    break
                i = min(cand, key=lambda i: (abs(2 * (b0[i] + step) - routed[i].nb), i))
                b0[i] += step
            n0 = s0 + sum(b0)
            imbalance = abs(2 * n0 - sum(pc.nb for pc in pieces))
            free = min(extra0 + nchunks(s0), nchunks(sh.nb - s0)) + self.NSLC
            key = (imbalance, -free, abs(2 * s0 - sh.nb))
            if best is None or key < best[0]:
                best = (key, s0, b0)
        sh.b0 = best[1]
        for pc, b in zip(routed, best[2]):
            pc.b0 = b

    def chunks(self, n_cols):
        """[(first column, columns)] of a run of n_cols intermediate columns in gate|up chunks: as few chunks of at most
        `blocks` column blocks as cover the run, their sizes as even as whole blocks allow."""
        nb = n_cols // BLK
        n = -(-nb // self.BPC)
        out, b = [], 0
        for i in range(n):
            m = nb // n + (1 if i < nb % n else 0)
            out.append((BLK * b, BLK * m))
            b += m
        return out

    beat_aligned = staticmethod(beat_aligned)

    def out_slices(self, c):
        """Cluster c's output slices: [(first output column, columns)]."""
        h = self.H // self.NC
        return [(c * h + j * self.SW, self.SW) for j in range(self.NSLC)]

    def chain_order(self, c):
        """The chips a ChainGather of cluster c passes, the root last; both clusters' the same way."""
        return list(self.chain_path) if self.chain_path else self.order[1:] + [self.ROOT]
