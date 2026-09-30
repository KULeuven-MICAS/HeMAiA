#!/usr/bin/env python3
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""DeepSeek-V2-Lite layer 1, stage 3: W_DKV (its columns split by dkv_cols), dequantised -> kv16.

Checked: kv16. Builds on stage 2. W_DKV runs on every cluster that holds some of its columns
(params dkv_cols; stages 4+ want all 576 on cluster 0), AHEAD of W_Q in that cluster's
stream: its chunks come first in the weight image, and with fence_kv W_Q's loads wait for
kv16's dequantisation, so the in-order DM core does not park kv16's factor load behind W_Q's
first ring chunk.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "common"))
from dsv2_staged import NCL, load_stage, run  # noqa: E402

from libs import at_offset  # noqa: E402

PREV = load_stage(__file__, "stage2_wq")
STAGE = 3


def _wdkv_job(S, c):
    """W_DKV's GEMV ahead of W_Q in cluster c's stream, if c holds any of its columns."""
    if not S.dkv[c]:
        return []

    def done(c, y16, njobs):
        S.kv16[c] = y16
        if S.fence_kv and njobs > 1:
            # the in-order DM core must not park kv16's factor load behind W_Q's first ring
            # chunk: W_Q's loads wait for the dequantisation
            S.pipe.raw(lambda c=c, y16=y16: S.ls[c].fence(y16.out().port.ends[-1]),
                       f"fence_c{c}")
    return [("kv", S.dkv[c], "s_kv", S.dkv0[c], done)]


def build(S):
    """Stage 2's per-cluster loop, with W_DKV first in each stream that holds some of it."""
    S.before_wq = lambda c: _wdkv_job(S, c)
    PREV.build(S)


def checks(S):
    """kv16, each cluster's slice of it."""
    PREV.checks(S)
    if S.FROM != 1:
        return
    if S.T == 1:
        for c in range(NCL):
            if S.kv16[c] is not None:
                S.check(f"kv16_c{c}", S.kv16[c], None,
                        at_offset(S.h["gold_kv16"], 2 * S.dkv0[c]), 2 * S.dkv[c])
        return
    T = S.T
    gkv = S.st.put("dsv2_gold_kv16_t", "uint16_t", np.concatenate(
        [tk["kv16"][S.dkv0[c]: S.dkv0[c] + S.dkv[c]] for c in range(NCL) for tk in S.toks]
    ).view(np.uint16))
    for c in range(NCL):
        if S.kv16[c] is not None:
            S.check(f"kv16_c{c}", S.kv16[c], None, at_offset(gkv, 2 * T * S.dkv0[c]),
                    2 * S.dkv[c] * T, st=3)


if __name__ == "__main__":
    run(sys.modules[__name__])
