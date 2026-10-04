#!/usr/bin/env python3
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""DeepSeek-V2-Lite layer 1 on six chiplets, stage 2: W_Q by head on all eight clusters -> q16.

Checked: q16, each cluster's heads. Builds on stage 1 (x8 in every chip's L3). Global
cluster g streams its heads' W_Q columns (2 heads, 384 columns) from its own memory chiplet
through its weight ring and dequantises them; stage 3 puts W_DKV ahead of them in the latent
cluster's stream.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "common"))
from dsv2_mc import A_ROW, D, L3, QH, f16, load_stage, run  # noqa: E402

PREV = load_stage(__file__, "stage1_rmsnorm")
STAGE = 2


def build(S):
    """Per cluster, the GEMVs its stream runs over x8: W_Q's columns of its heads, with any job
    a later stage puts before or after them (`S.before_wq` / `S.after_wq`: stage 3's W_DKV),
    dequantised."""
    PREV.build(S)
    S.q16, S.kv16 = [None] * S.G, None
    before_wq = getattr(S, "before_wq", lambda g: [])
    after_wq = getattr(S, "after_wq", lambda g: [])
    QC = QH * S.HPG
    for g in range(S.G):
        x8 = S.x8[S.chip(g)].out()
        # (name, columns, factor key, first column of the factors, what to do with it)
        jobs = before_wq(g) + [("q", QC, "s_q", QC * g, None)] + after_wq(g)
        for i, (nm, cols, s_key, c0, done) in enumerate(jobs):
            y = S.gemv(f"{nm}_g{g}", g, D, cols, S.ks["x"], x8,
                       S.wh[(g, "wdkv" if nm == "kv" else "wq")], x_layout=A_ROW, x_level=L3,
                       first=i == 0)
            y16 = S.deq(f"{nm}16_g{g}", g, cols, y.out(), S.h[s_key], 2 * c0)
            if done:
                done(g, y16, len(jobs))
            else:
                S.q16[g] = y16


def checks(S):
    PREV.checks(S)
    QC = QH * S.HPG
    for g in range(S.G):
        S.check(f"q16_g{g}", S.q16[g], None, S.gold(f"q16_g{g}", f16(S.H["q16"][QC * g:
                                                                              QC * (g + 1)])),
                2 * QC, g)


if __name__ == "__main__":
    run(sys.modules[__name__])
