#!/usr/bin/env python3
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""DeepSeek-V2-Lite layer 1, stage 7: the MoE's RMSNorm, the router, softmax and the top 6.

Checked: hn, lg16, probs, the route record. Builds on stage 6. One token: h all-gathered to
every cluster and normalised there (every MoE GEMV reads its own copy), the router GEMV
(INT8) on cluster 0, its softmax, and the top 6 written as an expert-slot record whose
weight addresses the routed experts read at run time. A pass: token t's h normalised on
cluster t and packed into one a_row [T, D] in L3, and the route builds the UNION of the
tokens' experts, one record per token (moe_route tokens / union_n).

params from_stage 8 (a pass only) starts HERE, from the pass's golden h and ha, without
stages 1-6: stage 8 alone, for when the whole pass's graph does not fit.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "common"))
from dsv2_staged import (A_ROW, D, F16, HBM, I8, L1, L3, MESH, NCL, RM, SL,  # noqa: E402
                         dg, dg_pass_records, load_stage, row, run)

from libs import PortSpec, at_offset  # noqa: E402
from libs.blocks import (ARowPack, Fetch, Join, Linear, MoeRoute, Pull,  # noqa: E402
                        QuantizeARow, RMSNormRow, SoftmaxRow)

PREV = load_stage(__file__, "stage6_wuv_wo")
STAGE = 7


class _Staged:
    """A staged port where a stage is expected."""

    def __init__(self, port):
        self.port = port

    def out(self, name=None):
        return self.port


def build(S):
    if S.FROM == 1:
        PREV.build(S)
    if S.T == 1:
        _one_token(S)
        return
    if S.FROM == 8:
        _pass_inputs_from_golden(S)
    else:
        _pass_inputs(S)
    _pass_route(S)


def _one_token(S):
    pipe, h, inv, ks, view = S.pipe, S.h, S.inv, S.ks, S.view
    h_all, hn, ha = [], [], []
    for c in range(NCL):
        hg = pipe.add(Pull(rows=1, cols=SL, layout=RM, dtype=F16, src=tuple(range(NCL)),
                           dst=c), f"h_all_c{c}",
                      **{f"x{i}": S.hsl[i].out() for i in range(NCL)})
        h_all.append(view(f"h_row_c{c}", hg.out(), row(SL, c, rows=NCL), row(D, c)))
        hn.append(pipe.add(RMSNormRow(cols=D, cluster=c, in_level=L1), f"hn_c{c}",
                           x=h_all[c].out()))
        ha.append(pipe.add(QuantizeARow(cols=D, cluster=c, inv_scale_f32bits=inv["h"]),
                           f"ha_c{c}", x=hn[c].out()))
    lg = S.gemv("router", 0, D, dg.N_EXP, ks["x"], ha[0].out(), h["wr"], bits=8)
    lg16 = S.deq("lg16", 0, dg.N_EXP, lg.out(), h["s_r"])
    pr = pipe.add(SoftmaxRow(cols=dg.N_EXP, cluster=0), "probs", x=lg16.out())
    rec0 = pipe.add(MoeRoute(n=dg.N_EXP, k=dg.TOP_K, cluster=0), "route", p=pr.out(),
                    table=S.staged(PortSpec(RM, I8, (dg.N_EXP, 64), mem_level=L3),
                                   h["table"]))
    S.h_all, S.hn, S.ha, S.lg16, S.pr, S.rec0 = h_all, hn, ha, lg16, pr, rec0


def _pass_inputs_from_golden(S):
    """Stage 8 alone: token t's h row (the golden's) on cluster t, ha from the golden."""
    T, PS, pipe, gput = S.T, S.PS, S.pipe, S.gput
    h_ha = gput("ha_in", [S.a_row_bytes(tk["hq"]) for tk in PS["toks"]])
    S.h_ha = S.g_ha = h_ha
    S.ha_l3 = _Staged(S.staged(PortSpec(A_ROW, I8, (T, D), mem_level=L3), h_ha))
    S.hrow = []
    for t in range(T):
        h_in = gput(f"h_in_t{t}", [dg.f16(PS["toks"][t]["h16"])])
        S.hrow.append(pipe.add(Fetch(src=row(D, None, L3), cluster=t % NCL, nbytes=2 * D),
                               f"h_row_t{t}", x=S.staged(row(D, None, L3), h_in)))


def _pass_inputs(S):
    """Token t's h on cluster t: norm + quantiser, packed into one a_row [T, D] in L3 that
    every MoE GEMV reads (the stage-1 path)."""
    T, PS, pipe, inv, view = S.T, S.PS, S.pipe, S.inv, S.view
    S.h_ha = S.st.put_zeros("dsv2_ha", "int8_t", T * 2 * D)
    packs7, S.hrow = [], []
    for t in range(T):
        ct = t % NCL
        hg = pipe.add(Pull(rows=1, cols=SL, layout=RM, dtype=F16, src=tuple(range(NCL)),
                           dst=ct), f"h_all_t{t}",
                      **{f"x{i}": S.hsl[i][t].out() for i in range(NCL)})
        S.hrow.append(view(f"h_row_t{t}", hg.out(), row(SL, ct, rows=NCL), row(D, ct)))
        nrm = pipe.add(RMSNormRow(cols=D, cluster=ct, in_level=L1, quant_inv=inv["h"]),
                       f"hn_t{t}", x=S.hrow[t].out())
        packs7.append(pipe.add(ARowPack(cols=D, cluster=ct, dst=at_offset(S.h_ha, 2 * D * t)),
                               f"pack_ha_t{t}", x=nrm.out()))
    S.ha_l3 = pipe.add(Join(parts=[PortSpec(A_ROW, I8, (1, D), mem_level=L3)] * T,
                            dst=PortSpec(A_ROW, I8, (T, D), mem_level=L3)), "ha",
                       **{f"x{t}": p_.out() for t, p_ in enumerate(packs7)})
    S.g_ha = S.gput("ha", [S.a_row_bytes(tk["hq"]) for tk in PS["toks"]])


def _pass_route(S):
    """The router once for the pass, its softmax, and the union route."""
    T, PS, U, pipe, h, ks, gput, check = S.T, S.PS, S.U, S.pipe, S.h, S.ks, S.gput, S.check
    f16 = dg.f16
    lg = pipe.add(Linear(tokens=T, d_in=D, d_out=dg.N_EXP, mesh=MESH, cluster=0, gemv=True,
                         d_shift=ks["x"], x_level=L3, x_layout=A_ROW, w_level=HBM,
                         stream=S.ls[0], w_bits=8),
                  "router", x=S.ha_l3.out(),
                  w=S.staged(S.wspec(D, dg.N_EXP, 8, HBM), h["wr"]))
    lg16 = S.deq("lg16", 0, dg.N_EXP, lg.out(), h["s_r"], rows=T)
    pr = pipe.add(SoftmaxRow(cols=dg.N_EXP, cluster=0, rows=T), "probs", x=lg16.out())
    rec0 = pipe.add(MoeRoute(n=dg.N_EXP, k=dg.TOP_K, cluster=0, tokens=T, union=U), "route",
                    p=pr.out(), table=S.staged(PortSpec(RM, I8, (dg.N_EXP, 64), mem_level=L3),
                                               h["table"]))
    check("lg16", lg16, None, gput("lg16", [f16(tk["logits16"]) for tk in PS["toks"]]),
          2 * dg.N_EXP * T, st=7)
    check("probs", pr, None, gput("probs", [f16(tk["p16"]) for tk in PS["toks"]]),
          2 * dg.N_EXP * T, st=7)
    g_rec, _ = dg_pass_records(PS, h)
    check("route", rec0, "rec", gput("rec", [g_rec]), T * U * 128, st=7)
    S.lg16, S.pr, S.rec0 = lg16, pr, rec0


def checks(S):
    PREV.checks(S)
    if S.T == 1:
        h = S.h
        S.check("hn_c0", S.hn[0], None, h["gold_hn"], 2 * D)
        S.check("lg16", S.lg16, None, h["gold_lg"], 2 * dg.N_EXP)
        S.check("probs", S.pr, None, h["gold_p"], 2 * dg.N_EXP)
        S.check("route", S.rec0, "rec", h["gold_rec"], dg.TOP_K * 128)


if __name__ == "__main__":
    run(sys.modules[__name__])
