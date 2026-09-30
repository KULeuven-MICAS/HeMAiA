#!/usr/bin/env python3
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""DeepSeek-V2-Lite layer 1, stage 2: W_Q by head on the four clusters, dequantised -> q16.

Checked: q16. Builds on stage 1 (x8). Each cluster streams its four heads' W_Q columns (768)
through its weight ring and dequantises them; stage 3 puts W_DKV ahead of them in the same
stream.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "common"))
from dsv2_staged import A_ROW, D, HBM, L3, MESH, NCL, QC, load_stage, run  # noqa: E402

from libs import at_offset  # noqa: E402
from libs.blocks import Linear  # noqa: E402

PREV = load_stage(__file__, "stage1_rmsnorm")
STAGE = 2


def build(S):
    """Per cluster, the GEMVs its stream runs over x8: W_Q's 768 columns, preceded by any job
    a later stage puts before them (`S.before_wq`: stage 3's W_DKV), each dequantised."""
    PREV.build(S)
    S.q16, S.kv16 = [None] * NCL, [None] * NCL
    before_wq = getattr(S, "before_wq", lambda c: [])
    for c in range(NCL if S.FROM == 1 else 0):
        # (name, columns, factor key, first column, what to do with the result)
        jobs = before_wq(c) + [("q", QC, "s_q", QC * c, None)]
        for i, (nm, cols, s_key, c0, done) in enumerate(jobs):
            y = S.pipe.add(Linear(tokens=S.T, d_in=D, d_out=cols, mesh=MESH, cluster=c,
                                  gemv=True, d_shift=S.ks["x"], x_level=L3, x_layout=A_ROW,
                                  w_level=HBM, stream=S.ls[c],
                                  stream_after_x=S.hold and i == 0, w_bits=S.wbits),
                           f"{nm}_c{c}", x=S.x8p.out(),
                           w=S.staged(S.wspec(D, cols, S.wbits, HBM),
                                      at_offset(S.h_img, S.w_at_off[(nm, c)])))
            y16 = S.deq(f"{nm}16_c{c}", c, cols, y.out(), S.h[s_key], 2 * c0, rows=S.T)
            if done:
                done(c, y16, len(jobs))
            else:
                S.q16[c] = y16


def checks(S):
    """q16 of every cluster, every token's rows cluster-major for a pass."""
    PREV.checks(S)
    if S.FROM != 1:
        return
    if S.T == 1:
        for c in range(NCL):
            S.check(f"q16_c{c}", S.q16[c], None, S.h[f"gold_q16_c{c}"], 2 * QC)
        return
    # every token's rows, cluster-major, against goldens staged in the same order
    T = S.T
    gq = S.st.put("dsv2_gold_q16_t", "uint16_t", np.concatenate(
        [tk["q16"][QC * c: QC * (c + 1)] for c in range(NCL) for tk in S.toks]
    ).view(np.uint16))
    for c in range(NCL):
        S.check(f"q16_c{c}", S.q16[c], None, at_offset(gq, 2 * QC * T * c), 2 * QC * T, st=2)


if __name__ == "__main__":
    run(sys.modules[__name__])
