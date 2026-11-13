#!/usr/bin/env python3
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""DeepSeek-V2-Lite's layer-1 MLA attention, stage 0 of the 256-token prefill on sixteen chiplets alone: x from the
HBM of the root's memory chip into every chip's L3.

The root's host queues its memory chip's push engine, which multicasts the prompt's x (INT8, A layout per chunk) a
16-token tile (32 KiB) at a time into every compute chip's L3, each tile followed by its landed flag; the root gets
its copy from the multicast like every other chip. Every cluster then waits for the flags of the tiles stage 1 reads
(cluster c: the tiles t with t mod 2 = c). No weight is streamed. The task graph (task names without their
_k<chip>_cl<c> suffix):

                                 S0_start                          the start stamp: root chip, cluster 0, DM core
                     ________________|_________________
                    |                                  |
         x_hbm_bcast_post<n>  - - - - - - - - - - > x_wait_t<t>    left: the root's host queues the tiles; right: every
         x_hbm_bcast_land<n>     (ordering edge)        |          cluster's DM core waits for its tiles' flags
                    |                                  |
                    |__________ P_fin<chip> ___________|           a probe a chip behind its waits (the root's also
                                      |                            behind the multicast's landing)
                                   S1_end  ->  fp_report, Check_fp_x_*    the end stamp, the report, the checks

Checks: three windows of x (the first, the middle and the last tile) in the L3 of the root, of the chip in the
middle of the fold order and of the last chip of it.

params.hjson: the platform, the tokens, the chunk and key-tile sizes, the weight chunk and ring sizes (the block's
image), the weights' width."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from stage_common import BingoMemSymbol, StageBase, app_name  # noqa: E402
from driver import run_driver  # noqa: E402

CHK = 2048                                  # bytes a checked window


class Stage0(StageBase):
    """Stage 0 alone (the module docstring)."""
    STAGE = 0

    def build(self):
        self.last_ld = {}
        for ls in self.ls:
            ls.strict = True
        go = self.setup()
        self.relays()
        self.x_puts = self.stage_x_bcast(go)
        last = {k: [] for k in self.chips}
        for k in self.chips:
            for c in range(self.NC):
                last[k] += [self.x_wait(k, c, t, go) for t in self.cl_tiles(c)]
        self.window_end(last)
        self.run_checks(self.x_checks())

    def x_checks(self):
        """Three windows of x in the L3 of the root, of the chip in the middle of the fold order and of the last."""
        T, R, n = self.TAG, self.ROOT, CHK
        far = self.order[len(self.order) // 2]
        checks = []
        for k in dict.fromkeys((R, far, self.order[-1])):
            for t in (0, self.NX // 2, self.NX - 1):
                off = t * self.XT + (self.XT - n if t == self.NX - 1 else 0)
                checks.append((f"x_k{k:02x}_t{t}", BingoMemSymbol(f"{T}_xsrc", offset=off,
                                                                    chip_id=None if k == R else k),
                               self.xsrc[off:off + n].view(np.uint8), n))
        return checks


if __name__ == "__main__":
    run_driver("fa prefill stage 0 workload generator", Stage0, app_name)
