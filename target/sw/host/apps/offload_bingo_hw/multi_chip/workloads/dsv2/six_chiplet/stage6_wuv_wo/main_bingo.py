#!/usr/bin/env python3
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""DeepSeek-V2-Lite layer 1 on six chiplets, stage 6: W_UV per head, W_O by output columns, the residual -> h.

Checked: oh, attn, h, per cluster. Builds on stage 5. The attention's output o~ is SCATTERED
by head to the clusters (each its heads' rows), each absorbs them through W_UV; o (every
head's) is ALL-GATHERED onto every cluster; each computes its 256 output columns of W_O and
adds the token's matching slice -> its slice of h.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "common"))
from dsv2_mc import D, F16, HBM, KVR, L1, RM, dg, f16, load_stage, run  # noqa: E402

from libs import PortSpec, at_offset  # noqa: E402
from libs.blocks import AddRow, Fetch, QuantizeARow  # noqa: E402

PREV = load_stage(__file__, "stage5_attention")
STAGE = 6


def build(S):
    PREV.build(S)
    row, view, inv, ks, HPG, G = S.spec_row, S.view, S.inv, S.ks, S.HPG, S.G
    ATT = S.ATT
    # o~ by head: ATT's xt [32, 512] (rows = heads), cluster g's rows 2g .. 2g + 1
    xt = S.scatter("xt", ATT, S.att.out("xt"), row(KVR, ATT, rows=32),
                   [2 * HPG * KVR] * G, list(range(G)),
                   lambda g: row(KVR, g, rows=HPG))
    S.oh16, S.attn16, S.hsl = [None] * G, [None] * G, [None] * G
    for g in range(G):
        c = S.c(g)
        xt_r = view(f"xt_row_g{g}", g, xt[g].out(), row(KVR, g, rows=HPG), row(HPG * KVR, g))
        ot8 = S.add(QuantizeARow(cols=HPG * KVR, cluster=c, inv_scale_f32bits=inv["ot"]),
                    f"ot8_g{g}", g, x=xt_r.out())
        oh = S.gemv(f"oh_g{g}", g, KVR, dg.V_HEAD, ks["uv"], ot8.out(), S.wh[(g, "wuv")],
                    groups=HPG)
        S.oh16[g] = S.deq(f"oh16_g{g}", g, HPG * dg.V_HEAD, oh.out(), S.h["s_uv"],
                          2 * HPG * dg.V_HEAD * g)
    o_all = S.allgather("o", {g: S.oh16[g].out() for g in range(G)},
                        lambda g: row(HPG * dg.V_HEAD, g), 2 * HPG * dg.V_HEAD,
                        list(range(G)), lambda g: row(D, g))
    for g in range(G):
        c = S.c(g)
        oa = S.add(QuantizeARow(cols=D, cluster=c, inv_scale_f32bits=inv["o"]), f"oa_g{g}", g,
                   x=o_all[g].out())
        n, o0 = S.o_cols[g], S.o0[g]
        at = S.gemv(f"attn_g{g}", g, D, n, ks["x"], oa.out(), S.wh[(g, "wo")])
        S.attn16[g] = S.deq(f"attn16_g{g}", g, n, at.out(), S.h["s_o"], 2 * o0)
        xs = S.add(Fetch(src=row(n, level=HBM), cluster=c, nbytes=2 * n), f"x_g{g}", g,
                   x=S.staged(row(n, level=HBM), at_offset(S.h_xh[S.mem[S.chip(g)]], 2 * o0)))
        S.hsl[g] = S.add(AddRow(cols=n, cluster=c), f"h_g{g}", g, a=xs.out(),
                         b=S.attn16[g].out())


def checks(S):
    PREV.checks(S)
    H, HPG = S.H, S.HPG
    for g in range(S.G):
        S.check(f"oh_g{g}", S.oh16[g], None,
                S.gold(f"oh_g{g}", f16(H["o16"][HPG * g: HPG * (g + 1)]).reshape(-1)),
                2 * HPG * dg.V_HEAD, g)
    for g in range(S.G):
        o0, n = S.o0[g], S.o_cols[g]
        S.check(f"attn_g{g}", S.attn16[g], None,
                S.gold(f"attn_g{g}", f16(H["attn16"][o0: o0 + n])), 2 * n, g)
        S.check(f"h_g{g}", S.hsl[g], None, S.gold(f"h_g{g}", f16(H["h16"][o0: o0 + n])),
                2 * n, g)


if __name__ == "__main__":
    run(sys.modules[__name__])
