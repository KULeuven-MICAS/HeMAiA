# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""MBuild's placement: the experts on the chips, their columns on a chip's clusters, the chunks and slices
(README.md, THE PLACEMENT)."""

import moe_prefill_data as md

from .common import BLK, SIMD_BEAT, beat_aligned


class Placement:

    def _place(self):
        """4 experts a chip, most padded rows first onto the chip with the least work so far among those holding fewer
        than 4 (the shared MLP's work counted too: the same on every chip with the even split)."""
        cnt = self.d.cnt
        load = {k: self.T * self.shk[k] for k in self.chips}
        held = {k: [] for k in self.chips}
        for e in sorted(range(md.N_EXP), key=lambda e: (-md.pad8(cnt[e]), e)):
            k = min((k for k in self.chips if len(held[k]) < self.EPC), key=lambda k: (load[k], self.chips.index(k)))
            held[k].append(e)
            load[k] += md.pad8(cnt[e]) * self.EB
        self.held = held

    def _split_all(self):
        """Per chip: the shared blocks (s0, s1) and per expert (b0, b1) of its two clusters, cluster 0 an even number,
        blocks moved two at a time from the busier cluster while that evens them."""
        cnt, T = self.d.cnt, self.T
        self.shb, self.eb = {}, {}
        for k in self.chips:
            S = self.shk[k]
            s0 = S // 2 + (S // 2) % 2
            sh = [s0, S - s0]
            eb = {}
            for e in self.held[k]:
                b0 = self.EB // 2 - (self.EB // 2) % 2
                eb[e] = [b0, self.EB - b0]

            def loads():
                return [sh[c] * T + sum(eb[e][c] * md.pad8(cnt[e]) for e in eb) for c in (0, 1)]
            for _ in range(4 * self.EB):
                L = loads()
                hi = 0 if L[0] > L[1] else 1
                gap = abs(L[0] - L[1])
                best = None
                for e in eb:
                    mv = 4 * md.pad8(cnt[e])
                    if eb[e][hi] >= 4 and abs(gap - mv) < gap and (best is None or abs(gap - mv) < best[0]):
                        best = (abs(gap - mv), e)
                if best is None:
                    break
                eb[best[1]][hi] -= 2
                eb[best[1]][1 - hi] += 2
            self.shb[k] = sh
            self.eb[k] = eb

    def ch(self, k, c):
        """Cluster c of chip k's column ranges: shared (first global shared column, columns) and per expert (first
        column within the expert, columns)."""
        s0, s1 = self.shb[k]
        sh = (BLK * (self.sho[k] + (0 if c == 0 else s0)), BLK * (s0 if c == 0 else s1))
        ex = {e: ((0, BLK * b0) if c == 0 else (BLK * b0, BLK * b1)) for e, (b0, b1) in self.eb[k].items()}
        return sh, ex

    def sh_stage_pitch(self, k, c):
        """The row pitch of cluster c's staging rows on chip k, or 0 when every one of its shared chunks starts on a
        SIMD beat of its A8 rows. The quantise writes whole 32-byte beats, so a chunk whose first column sits mid-beat
        is quantised into staging rows and copied into place by the DM core."""
        sn = self.ch(k, c)[0][1]
        coff = 0 if c == 0 else BLK * self.shb[k][0]
        odd = [n for b0, n in self.chunks(sn) if (coff + b0) % SIMD_BEAT]
        return -(-(max(odd) + BLK) // SIMD_BEAT) * SIMD_BEAT if odd else 0

    beat_aligned = staticmethod(beat_aligned)

    def chunks(self, n_cols):
        """[(first column, columns)] of a run of n_cols intermediate columns in gate|up chunks."""
        nb = n_cols // BLK
        return [(BLK * b, BLK * min(self.BPC, nb - b)) for b in range(0, nb, self.BPC)]

    def row_blocks(self, rows):
        """An expert slot's padded rows as its gate|up passes take them (params expert_rows): [(first row, rows)]."""
        mb = self.expert_rows
        if not mb or rows <= mb:
            return [(0, rows)]
        nb, tiles = -(-rows // mb), rows // 8
        out, r0 = [], 0
        for i in range(nb):
            n = 8 * (tiles // nb + (1 if i < tiles % nb else 0))
            out.append((r0, n))
            r0 += n
        return out

    def out_slices(self, c):
        """Cluster c's output slices: [(first output column, columns)]."""
        h = self.H // self.NC
        return [(c * h + j * self.SW, self.SW) for j in range(self.NSLC)]

    def chain_order(self, c):
        """The chips a ChainGather of cluster c passes, the root last; both clusters' the same way."""
        return list(self.chain_path) if self.chain_path else self.order[1:] + [self.ROOT]

    def fold_order_c(self, c):
        """The chips in the order cluster c's reduce adds their partials (FP16 addition does not associate, so the
        golden adds in it): a ChainGather's (chain_order); the root's pulls (reduce root): its own partial first, then
        every other chip's in the chain's order."""
        ch = self.chain_order(c)
        return ch if self.reduce_impl == "chain" else [self.ROOT] + [k for k in ch if k != self.ROOT]
