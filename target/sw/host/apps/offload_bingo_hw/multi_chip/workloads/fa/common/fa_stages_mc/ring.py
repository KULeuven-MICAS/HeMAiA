# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""FaStages' ring: the order of the root's ring runs and x's multicast in the root memory chip's engine queue (FaBuild's
ring streams every weight)."""


class FaStagesRing:

    def _cluster_x_order(self):
        """x_pf_split: the root memory chip's engine queue, by boundaries of the root's ring (WeightRings.boundary): its
        W_KVA runs, x's first part, then the rest of the ring once that part has landed. keys_by_chunk: W_KVA, chunk
        0's x, chunk 0's stage-4 runs (PER_CHUNK_HEAD chunks, behind that x's landing), chunk 1's x, the rest of the
        ring behind chunk 1's x landing (x_rest_early: behind the third part's, which goes after chunk 0's key
        multicast), else the third part of x after the ring's last run."""
        ring, xp, n = self.rings[self.ROOT], self.x_puts, self.kva_in_ring()
        if not self.keys_by_chunk:
            ring.boundary(0, n, before=[self.first_of(xp[0])], after=[self.done_of(xp[-1])])
            return
        rest_early = len(xp) > 2 and self.x_rest_early
        ring.boundary(0, n, before=[self.first_of(xp[0])], after=[self.done_of(xp[0])])
        ring.boundary(0, n + self.PER_CHUNK_HEAD, before=[self.first_of(xp[1])],
                      after=[self.done_of(xp[2] if rest_early else xp[1])])
        if len(xp) > 2 and not rest_early:
            ring.tail_before(0, [self.first_of(xp[2])])
