#!/usr/bin/env python3
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""DeepSeek-V2-Lite layer 1, stage 6: W_UV per head, W_O by output columns, the residual -> h.

Checked: oh, attn, h. Builds on stage 5. Each cluster pulls its four heads' attention
output, quantises it and absorbs it through W_UV (-> oh); every cluster then gathers all 16
heads' oh, runs W_O for its 512 output columns and adds the residual x -> its slice of h.
A pass streams W_UV and W_O once for all tokens (the GEMV's token loop).
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "common"))
from dsv2_staged import (A_ROW, ATT, D, F16, HBM, HPC, I8, KVR, L1, NCL, RM, SL,  # noqa: E402
                         dg, load_stage, row, run)

from libs import PortSpec, at_offset  # noqa: E402
from libs.blocks import AddRow, Fetch, Pull, QuantizeARow  # noqa: E402

PREV = load_stage(__file__, "stage5_attention")
STAGE = 6


def build(S):
    PREV.build(S)
    if S.T == 1:
        _one_token(S)
    elif S.FROM == 1:
        _pass(S)


def _one_token(S):
    pipe, h, inv, ks, view, gemv, deq = S.pipe, S.h, S.inv, S.ks, S.view, S.gemv, S.deq
    oh16, hsl, xs, attn16 = [], [], [], []
    for c in range(NCL):
        xt_c = pipe.add(Pull(rows=32, cols=KVR, layout=RM, dtype=F16, src=ATT, dst=c,
                             row0=HPC * c, nrows=HPC), f"xt_c{c}", x0=S.att.out("xt"))
        xt_r = view(f"xt_row_c{c}", xt_c.out(), row(KVR, c, rows=HPC), row(HPC * KVR, c))
        ot8 = pipe.add(QuantizeARow(cols=HPC * KVR, cluster=c,
                                    inv_scale_f32bits=inv["ot"]), f"ot8_c{c}",
                       x=xt_r.out())
        oh = gemv(f"oh_c{c}", c, KVR, dg.V_HEAD, ks["uv"], ot8.out(), h["wuv"],
                  HPC * c * KVR * dg.V_HEAD, groups=HPC)
        oh16.append(deq(f"oh16_c{c}", c, HPC * dg.V_HEAD, oh.out(), h["s_uv"],
                        2 * HPC * dg.V_HEAD * c))
    for c in range(NCL):
        o_all = pipe.add(Pull(rows=1, cols=SL, layout=RM, dtype=F16,
                              src=tuple(range(NCL)), dst=c), f"o_all_c{c}",
                         **{f"x{i}": oh16[i].out() for i in range(NCL)})
        o_r = view(f"o_row_c{c}", o_all.out(), row(SL, c, rows=NCL), row(D, c))
        oa = pipe.add(QuantizeARow(cols=D, cluster=c, inv_scale_f32bits=inv["o"]),
                      f"oa_c{c}", x=o_r.out())
        at = gemv(f"attn_c{c}", c, D, SL, ks["x"], oa.out(), h["wo"], SL * c * D)
        attn16.append(deq(f"attn16_c{c}", c, SL, at.out(), h["s_o"], 2 * SL * c))
        xs.append(pipe.add(Fetch(src=row(SL, None, HBM), cluster=c, nbytes=2 * SL),
                           f"x_c{c}", x=S.staged(row(SL, None, HBM),
                                                 at_offset(S.h_xh, 2 * SL * c))))
        hsl.append(pipe.add(AddRow(cols=SL, cluster=c), f"h_c{c}", a=xs[c].out(),
                            b=attn16[c].out()))
    S.oh16, S.hsl, S.xs, S.attn16 = oh16, hsl, xs, attn16


def _pass(S):
    pipe, h, inv, ks, view, gemv, deq = S.pipe, S.h, S.inv, S.ks, S.view, S.gemv, S.deq
    T, PS, ATTS, atts, gput, check = S.T, S.PS, S.ATTS, S.atts, S.gput, S.check
    f16 = dg.f16
    oh16, attn16, hsl = [], [], [[None] * T for _ in range(NCL)]
    for c in range(NCL):
        # this cluster's heads of every token, token-major, from the groups' xt
        xt_c = pipe.add(Pull(rows=32, cols=KVR, layout=RM, dtype=F16,
                             src=tuple(ATTS[t // 2] for t in range(T)), dst=c,
                             row0=tuple((t % 2) * 16 + HPC * c for t in range(T)),
                             nrows=HPC), f"xt_c{c}",
                        **{f"x{t}": atts[t // 2].out("xt") for t in range(T)})
        xt_r = view(f"xt_row_c{c}", xt_c.out(), row(KVR, c, rows=T * HPC),
                    row(T * HPC * KVR, c))
        ot8 = pipe.add(QuantizeARow(cols=KVR, cluster=c, inv_scale_f32bits=inv["ot"],
                                    segs=T * HPC, seg_pitch=2 * KVR, a_row=True),
                       f"ot8_c{c}", x=xt_r.out())
        otx = view(f"ot8_rows_c{c}", ot8.out(),
                   PortSpec(A_ROW, I8, (1, T * HPC * KVR), mem_level=L1, cluster=c),
                   PortSpec(A_ROW, I8, (T, HPC * KVR), mem_level=L1, cluster=c))
        oh = gemv(f"oh_c{c}", c, KVR, dg.V_HEAD, ks["uv"], otx.out(), h["wuv"],
                  HPC * c * KVR * dg.V_HEAD, groups=HPC, tokens=T, x_layout=A_ROW)
        oh16.append(deq(f"oh16_c{c}", c, HPC * dg.V_HEAD, oh.out(), h["s_uv"],
                        2 * HPC * dg.V_HEAD * c, rows=T))
    tm = tuple(i for t in range(T) for i in range(NCL))        # token-major sources
    r0 = tuple(t for t in range(T) for i in range(NCL))
    for c in range(NCL):
        o_all = pipe.add(Pull(rows=T, cols=SL, layout=RM, dtype=F16, src=tm, dst=c,
                              row0=r0, nrows=1), f"o_all_c{c}",
                         **{f"x{t * NCL + i}": oh16[i].out() for t in range(T)
                            for i in range(NCL)})
        o_r = view(f"o_row_c{c}", o_all.out(), row(SL, c, rows=T * NCL), row(T * D, c))
        oa = pipe.add(QuantizeARow(cols=D, cluster=c, inv_scale_f32bits=inv["o"], segs=T,
                                   seg_pitch=2 * D, a_row=True), f"oa_c{c}", x=o_r.out())
        oax = view(f"oa_rows_c{c}", oa.out(),
                   PortSpec(A_ROW, I8, (1, T * D), mem_level=L1, cluster=c),
                   PortSpec(A_ROW, I8, (T, D), mem_level=L1, cluster=c))
        at = gemv(f"attn_c{c}", c, D, SL, ks["x"], oax.out(), h["wo"], SL * c * D,
                  tokens=T, x_layout=A_ROW)
        attn16.append(deq(f"attn16_c{c}", c, SL, at.out(), h["s_o"], 2 * SL * c, rows=T))
        for t in range(T):
            xs_t = pipe.add(Fetch(src=row(SL, None, HBM), cluster=c, nbytes=2 * SL),
                            f"x_c{c}_t{t}", x=S.staged(row(SL, None, HBM),
                                                       at_offset(S.h_xh, 2 * D * t + 2 * SL * c)))
            # The fetch has no producer, so the DM core would run it first and its edge to
            # the residual add would hold a dependency tag for the whole layer: behind
            # W_O's last weight load (same core, no tag) it is live only for the add.
            pipe.raw(lambda at=at, xs_t=xs_t: S.dfg.bingo_add_edge(
                at.result.extra["loads"][-1], xs_t.result.nodes[0]), f"late_x_c{c}_t{t}")
            a_t = view(f"attn16_c{c}_t{t}", attn16[c].out(), row(SL, c, rows=T),
                       row(SL, c), 2 * SL * t)
            hsl[c][t] = pipe.add(AddRow(cols=SL, cluster=c), f"h_c{c}_t{t}",
                                 a=xs_t.out(), b=a_t.out())
    for c in range(NCL):
        check(f"oh_c{c}", oh16[c], None,
              gput(f"oh_c{c}", [f16(tk["o16"][HPC * c: HPC * (c + 1)]) for tk in PS["toks"]]),
              2 * HPC * dg.V_HEAD * T, st=6)
        check(f"attn_c{c}", attn16[c], None,
              gput(f"attn_c{c}", [f16(tk["attn16"][SL * c: SL * (c + 1)])
                                  for tk in PS["toks"]]), 2 * SL * T, st=6)
    for t in range(T):
        check(f"h_c0_t{t}", hsl[0][t], None,
              gput(f"h_c0_t{t}", [f16(PS["toks"][t]["h16"][:SL])]), 2 * SL, st=6)
    S.oh16, S.attn16, S.hsl = oh16, attn16, hsl


def checks(S):
    PREV.checks(S)
    if S.T == 1:
        h = S.h
        for c in range(NCL):
            S.check(f"oh_c{c}", S.oh16[c], None, h[f"gold_oh_c{c}"], 2 * HPC * dg.V_HEAD)
        for c in range(NCL):
            S.check(f"attn_c{c}", S.attn16[c], None, h[f"gold_attn_c{c}"], 2 * SL)
            S.check(f"h_c{c}", S.hsl[c], None, h[f"gold_h_c{c}"], 2 * SL)


if __name__ == "__main__":
    run(sys.modules[__name__])
