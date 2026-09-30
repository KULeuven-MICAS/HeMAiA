#!/usr/bin/env python3
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""DeepSeek-V2-Lite layer 1, stage 5: the cache append, Q8, and the MLA attention over L+1 keys.

Checked: q8, the key tile, ot, xt. Builds on stage 4. One token: its row appended to both
cache copies (cluster 0), every cluster's q~ and q_pe gathered onto the attention's cluster
(ATT) and assembled into Q8, and the attention over the L + 1 keys in 64-key tiles. A pass of
several tokens: each token at L + t appends its row, and the attention runs as T / 2 GROUPS
of two tokens -- the 32 query lanes one token half-fills -- each on a cluster of its own
(att_clusters), token t masked from the keys of the tokens after it.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "common"))
from dsv2_staged import (A, ATT, F16, HPC, I8, KV, KVR, L3, NCL, RM, RP, dg,  # noqa: E402
                         load_stage, row, run)

from libs import PortSpec  # noqa: E402
from libs.blocks import CacheAppend, LoadStream, MlaAttention, Pull, QAssemble  # noqa: E402

PREV = load_stage(__file__, "stage4_latent_rope_wuk")
STAGE = 5


def build(S):
    PREV.build(S)
    if S.T == 1:
        _one_token(S)
    elif S.FROM == 1:
        _pass(S)


def _one_token(S):
    pipe, p, h, inv, ks, data, ls, view = S.pipe, S.p, S.h, S.inv, S.ks, S.data, S.ls, S.view
    cap = data["cap"]
    app = pipe.add(CacheAppend(pos=data["L"], cap=cap, cluster=0), "append",
                   c8=S.c8.out(), kpe8=S.kpe8.out(),
                   key=S.staged(PortSpec(A, I8, (cap, KV), mem_level=L3), h["key"]),
                   val=S.staged(PortSpec(A, I8, (KVR, cap), mem_level=L3), h["val"]))
    qt_rows = [view(f"qt_rows_c{c}", S.qt16[c].out(), row(HPC * KVR, c),
                    row(KVR, c, rows=HPC)) for c in range(NCL)]
    qt_all = pipe.add(Pull(rows=HPC, cols=KVR, layout=RM, dtype=F16,
                           src=tuple(range(NCL)), dst=ATT), "qt_all",
                      **{f"x{c}": qt_rows[c].out() for c in range(NCL)})
    rot4 = [view(f"rot4_c{c}", S.rot[c].out(), row(RP, c, rows=HPC + (c == 0)),
                 row(RP, c, rows=HPC)) for c in range(NCL)]
    qpe_all = pipe.add(Pull(rows=HPC, cols=RP, layout=RM, dtype=F16,
                            src=tuple(range(NCL)), dst=ATT), "qpe_all",
                       **{f"x{c}": rot4[c].out() for c in range(NCL)})
    q8 = pipe.add(QAssemble(inv_qt_f32bits=inv["qt"], inv_qpe_f32bits=inv["qpe"],
                            cluster=ATT), "q8", qt=qt_all.out(), qpe=qpe_all.out())
    # params fence_s4: every cluster's later weight loads (W_UV, W_O, ...) wait for its
    # part of the attention's inputs -- cluster 0's cache append, the attention
    # cluster's Q8, the others' q~ -- so the in-order DM core does not park that work
    # behind a ring chunk the prefetcher has not pushed yet (measured: cluster 0's RoPE,
    # factor load and append 40 us late, the attention with them)
    if p.get("fence_s4", False):
        fence_at = {c: (S.qt16[c], None) for c in range(NCL)}
        fence_at[0], fence_at[ATT] = (app, "key"), (q8, None)
        for c, (stg, prt) in fence_at.items():
            pipe.raw(lambda c=c, stg=stg, prt=prt: ls[c].fence(
                stg.out(prt).port.ends[-1]), f"fence_s4_c{c}")
    # params att_stream: the attention's K and V tiles through one slab each of their own
    # (36 + 32 KiB), so the attention cluster's weight stream -- its W_UV and W_O
    # chunks -- keeps loading while the attention runs instead of queueing behind it.
    # Two 68 KiB K|V slabs overflowed cluster 1's L1 by stage 6.
    if p.get("att_stream", False):
        att_ls = LoadStream(S.ctx, ATT, nbytes=dg.BC * KV, nbuf=1, name="kstream")
        att_vs = LoadStream(S.ctx, ATT, nbytes=KVR * dg.BC, nbuf=1, name="vstream")
    else:
        att_ls, att_vs = ls[ATT], None
    S.att = pipe.add(MlaAttention(keys=data["keys"], cap=cap, k_s=ks["s"], k_o=data["k_o"],
                                  a_exp=data["a_exp"], a_n_f32bits=data["a_n"], cluster=ATT,
                                  bc=dg.BC, stream=att_ls, vstream=att_vs),
                     "attn", q8=q8.out(), key=app.out("key"), val=app.out("val"))
    S.app, S.q8 = app, q8


def _pass(S):
    pipe, p, h, inv, ks, data, ls, view = S.pipe, S.p, S.h, S.inv, S.ks, S.data, S.ls, S.view
    T, PS, ATTS, gput, check = S.T, S.PS, S.ATTS, S.gput, S.check
    f16 = dg.f16
    cap = PS["cap"]
    app = None
    for t in range(T):
        kb = app.out("key") if app else S.staged(PortSpec(A, I8, (cap, KV), mem_level=L3),
                                                 h["key"])
        vb = app.out("val") if app else S.staged(PortSpec(A, I8, (KVR, cap), mem_level=L3),
                                                 h["val"])
        app = pipe.add(CacheAppend(pos=data["L"] + t, cap=cap, cluster=0), f"append_t{t}",
                       c8=S.c8s[t].out(), kpe8=S.kpe8s[t].out(), key=kb, val=vb)
    h_mask = S.st.put("dsv2_mask16", "uint16_t", np.full(64, 0xFBFF, dtype=np.uint16))
    if p.get("fence_s4", False):
        # cluster 0 after the last append, the non-attention clusters after their q~
        pipe.raw(lambda a=app: ls[0].fence(a.out("key").port.ends[-1]), "fence_s4_c0")
        for c in range(1, NCL):
            if c not in ATTS:
                pipe.raw(lambda c=c: ls[c].fence(S.qt16[c].out().port.ends[-1]),
                         f"fence_s4_c{c}")
    atts, q8g = [], []
    for gi, grp in enumerate(PS["groups"]):
        cl = ATTS[gi]
        qb = {}
        for u, t in enumerate(grp["tokens"]):
            qt_rows = [view(f"qt_rows_c{c}_t{t}", S.qt16[c].out(), row(HPC * KVR, c, rows=T),
                            row(KVR, c, rows=HPC), 2 * HPC * KVR * t) for c in range(NCL)]
            qa = pipe.add(Pull(rows=HPC, cols=KVR, layout=RM, dtype=F16,
                               src=tuple(range(NCL)), dst=cl), f"qt_all_t{t}",
                          **{f"x{c}": qt_rows[c].out() for c in range(NCL)})
            rot4 = [view(f"rot4_c{c}_t{t}", S.rots[c][t].out(),
                         row(RP, c, rows=HPC + (c == 0)), row(RP, c, rows=HPC))
                    for c in range(NCL)]
            qp = pipe.add(Pull(rows=HPC, cols=RP, layout=RM, dtype=F16,
                               src=tuple(range(NCL)), dst=cl), f"qpe_all_t{t}",
                          **{f"x{c}": rot4[c].out() for c in range(NCL)})
            sfx = "" if u == 0 else "1"
            qb["qt" + sfx], qb["qpe" + sfx] = qa.out(), qp.out()
        q8g.append(pipe.add(QAssemble(inv_qt_f32bits=inv["qt"], inv_qpe_f32bits=inv["qpe"],
                                      cluster=cl, tokens=2), f"q8_g{gi}", **qb))
        if p.get("att_stream", False):
            a_ls = LoadStream(S.ctx, cl, nbytes=dg.BC * KV, nbuf=1, name=f"kstream{gi}")
            a_vs = LoadStream(S.ctx, cl, nbytes=KVR * dg.BC, nbuf=1, name=f"vstream{gi}")
        else:
            a_ls, a_vs = ls[cl], None
        atts.append(pipe.add(
            MlaAttention(keys=PS["keys"], cap=cap, k_s=ks["s"], k_o=PS["k_o"],
                         a_exp=PS["a_exp"], a_n_f32bits=PS["a_n"], cluster=cl, bc=dg.BC,
                         stream=a_ls, vstream=a_vs, masks=tuple(grp["masks"]),
                         mask_src=h_mask),
            f"attn_g{gi}", q8=q8g[gi].out(), key=app.out("key"), val=app.out("val")))
        if p.get("fence_s4", False):
            # as for one token: the group's cluster loads no later weight before its Q8
            pipe.raw(lambda cl=cl, q=q8g[gi]: ls[cl].fence(q.out().port.ends[-1]),
                     f"fence_s4_g{gi}")
        check(f"q8_g{gi}", q8g[gi], "q8", gput(f"q8_g{gi}", [grp["q8"]]), 32 * KV, st=5)
        check(f"ot_g{gi}", atts[gi], "ot", gput(f"ot_g{gi}", [f16(grp["ot"])]),
              2 * KVR * 32, st=5)
    S.app, S.atts, S.q8g = app, atts, q8g


def checks(S):
    PREV.checks(S)
    if S.T == 1:
        h = S.h
        S.check("q8", S.q8, "q8", h["gold_q8"], 32 * KV)
        S.check("ot", S.att, "ot", h["gold_ot"], 2 * KVR * 32)
        S.check("xt", S.att, "xt", h["gold_xt"], 2 * KVR * 32)


if __name__ == "__main__":
    run(sys.modules[__name__])
