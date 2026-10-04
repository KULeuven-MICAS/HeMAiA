#!/usr/bin/env python3
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""DeepSeek-V2-Lite layer 1 on six chiplets, stage 3: W_DKV on the latent's cluster -> kv16.

Checked: kv16. Builds on stage 2. The whole W_DKV (576 columns) streams on the latent's
cluster (LAT), ahead of its W_Q heads -- its output feeds the latent's norm, k_pe's RoPE and
the cache append, all on LAT -- with W_Q's loads fenced behind the dequantisation so the
in-order DM core does not park kv16's factor load behind a W_Q chunk not yet pushed.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "common"))
from dsv2_mc import F16, KV, L1, RM, f16, load_stage, run  # noqa: E402

from libs import PortSpec  # noqa: E402
from libs.blocks import Pull  # noqa: E402

PREV = load_stage(__file__, "stage2_wq")
STAGE = 3


def _wdkv_job(S, g, fence=True):
    split = bool(S.p.get("wdkv_split", False))
    if g != S.LAT and not (split and g == S.ATT):
        return []
    h = KV // 2

    def fenced(g, y16, njobs):
        if fence and S.p.get("fence_kv", True) and njobs > 1:
            S.pipe.raw(lambda: S.ls[g].fence(y16.out().port.ends[-1]), "fence_kv")

    def done(g, y16, njobs):
        S.kv16 = y16
        fenced(g, y16, njobs)

    if not split:
        return [("kv", KV, "s_kv", 0, done)]

    # params wdkv_split: LAT and ATT (one chip) each run half of W_DKV's columns -- LAT's
    # GEMM otherwise runs its W_Q, all 576 columns and its W_UK before q~ is ready, while
    # ATT's idles until the attention -- and LAT stacks the halves (a local copy and one from
    # ATT's L1) into the latent row stage 4 reads
    def half(g, y16, njobs):
        S.kv_half = getattr(S, "kv_half", {})
        S.kv_half[g] = y16
        fenced(g, y16, njobs)
        if len(S.kv_half) == 2:
            c = S.c
            both = S.add(Pull(rows=1, cols=h, layout=RM, dtype=F16, src=[c(S.LAT), c(S.ATT)],
                              dst=c(S.LAT)), "kv16_stack", S.LAT,
                         x0=S.kv_half[S.LAT].out(), x1=S.kv_half[S.ATT].out())
            S.kv16 = S.view("kv16_g", S.LAT, both.out(),
                            PortSpec(RM, F16, (2, h), mem_level=L1, cluster=c(S.LAT)),
                            PortSpec(RM, F16, (1, KV), mem_level=L1, cluster=c(S.LAT)))
    return [("kv", h, "s_kv", 0 if g == S.LAT else h, half)]


def build(S):
    # W_DKV before or after LAT's own W_Q heads. Before (the default) the latent is ready
    # first; after, LAT's q -- which every attention tile needs, while the latent is needed
    # only by the cache append and the last tile -- is not held behind 576 columns of W_DKV
    # (`wdkv_after_wq`). After, no fence: nothing of this stream's follows it in the jobs.
    # `latent_late`: W_DKV is built by stage 4, after LAT's W_UK -- the latent only feeds
    # the cache append and the attention's last tile, so it need not hold up q~ at all.
    S.latent_late = bool(S.p.get("latent_late", False))
    if S.latent_late:
        pass
    elif S.p.get("wdkv_after_wq", False):
        S.after_wq = lambda g: _wdkv_job(S, g, fence=False)
    else:
        S.before_wq = lambda g: _wdkv_job(S, g)
    PREV.build(S)


def checks(S):
    PREV.checks(S)
    S.check("kv16", S.kv16, None, S.gold("kv16", f16(S.H["ckv16"])), 2 * KV, S.LAT)


if __name__ == "__main__":
    run(sys.modules[__name__])
