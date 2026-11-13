# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""FaBuild's checks: the keys' pieces where they landed, y's windows in each collector's L3, and (checks stages, or a
run that stops at Q8) the query tiles and o~ of the chip in the middle of the fold order."""

import numpy as np

from bingo_mem_handle import BingoMemSymbol

from .consts import BR, D_QK, HIDDEN, KB, KVP


class FaChecks:

    keys_made = True                # the window makes the keys (a run of W_UV, W_O and the reduce alone does not)

    def _checks(self, swp_res, post_res):
        lay, T, R, d = self.lay, self.TAG, self.ROOT, self.d
        checks = []
        n = 2048
        if self.keys_made:
            # the keys as they landed: in the root's L3 the first piece's key and value heads and the last's tails; on
            # the chip in the middle of the fold order (the multicast's landing) the last piece's value tail
            last, far = self.T // 16 - 1, self.order[len(self.order) // 2]
            for nm, q, lo, kk in (("k0a", 0, 0, R), ("v0a", 0, KB, R), ("kLb", last, KB - n, R),
                                  ("vLb", last, KVP - n, R), ("vLf", last, KVP - n, far)):
                img = d.kv_piece(16 * q).view(np.uint8)
                checks.append((f"kv_{nm}", BingoMemSymbol(f"{T}_kv", offset=q * KVP + lo, chip_id=kk), img[lo:lo + n], n))
        if self.stop == "full":
            for i in range(self.NCH):
                for c in range(self.NC):
                    half = HIDDEN // 2
                    # each W_O block folded in its collector's chain order (y_roots), read from that collector's L3
                    yh = [self.y_raw(self.r0(i, c), self.ROWS, self.y_fold_heads(self.y_root(i, h))
                                     if self.y_roots else None) for h in range(2)]
                    img = np.concatenate([lay.to_d(yh[0][:, :half], d.mesh), lay.to_d(yh[1][:, half:], d.mesh)]
                                         ).astype(np.float16).view(np.uint8)
                    # the head and the tail of each half (each half is a W_O block)
                    wins = ((f"y_i{i}c{c}_h0a", 0), (f"y_i{i}c{c}_h0b", self.YB // 2 - n),
                            (f"y_i{i}c{c}_h1a", self.YB // 2), (f"y_i{i}c{c}_h1b", self.YB - n))
                    for nm, off in wins:
                        yr = self.y_root(i, 0 if off < self.YB // 2 else 1) if self.y_roots else R
                        checks.append((nm, self.y_sym(i, c, off, chip_id=yr), img[off:off + n], n))
        if self.checks_kind == "stages" or self.stop == "q8":
            # the first and the last query tile of both clusters of the chip in the middle of the fold order,
            # before the y checks
            stage = []
            far = self.order[len(self.order) // 2]
            h = self.head[far]
            i = self.NCH - 1
            for c in range(self.NC):
                cl = self.chips.index(far) * self.NC + c
                for q in sorted({0, self.NQ - 1}):
                    r0 = self.r0(i, c) + q * BR
                    tag = f"c{c}" if q == 0 else f"c{c}q{q}"
                    q8 = d.q8_a(h, r0)
                    stage.append((f"q8_{tag}", self.stash(f"q8_{tag}", far, c, self.bufs(cl, i)["q8"][q], D_QK * BR,
                                                          []),
                                  q8.view(np.uint8), D_QK * BR))
                    if self.checks_kind == "stages" and cl in swp_res:
                        ot, nb = self.xt_ref(h, r0)
                        stage.append((f"xt_{tag}", self.stash(f"xt_{tag}", far, c, swp_res[cl]["xt"][q], nb,
                                                              []), ot, nb))
            checks = stage + checks
        self.run_checks(checks)
