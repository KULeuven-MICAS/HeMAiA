#!/usr/bin/env python3
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""DeepSeek-V2-Lite layer 1, stage 8: the shared and routed experts and the combine -> out.

Checked: the shared expert, the routed slots, out. Builds on stage 7: THE WHOLE LAYER. The
shared expert on one cluster, or spread (params shared_spread: gate|up and SwiGLU on every
cluster's 704 columns, the down projection by output columns after an all-gather); each
routed slot streams the expert its record names; cluster 0 adds them in the golden's order.
A pass streams the union's slots once for all tokens, and its combine is a running sum that
travels slot to slot (a fan-in of every slot into cluster 0 needs more dependency tags than
DepTagWidth 5 has).
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "common"))
from dsv2_staged import (A, A_ROW, D, F16, I8, ISH_C, L1, L3, NCL, RM, SL,  # noqa: E402
                         dg, load_stage, row, run)

from libs import PortSpec, at_offset  # noqa: E402
from libs.blocks import (AddRow, Pull, ScaleRowBySlot, SlotLoad, Stash,  # noqa: E402
                        SwigluARow, record_spec)

PREV = load_stage(__file__, "stage7_router")
STAGE = 8


def build(S):
    PREV.build(S)
    if S.T == 1:
        _one_token(S)
    else:
        _pass(S)


def _one_token(S):
    pipe, p, h, inv, ks, view, gemv, deq = S.pipe, S.p, S.h, S.inv, S.ks, S.view, S.gemv, S.deq
    ls, fr, rec0, ha, h_all = S.ls, S.fr, S.rec0, S.ha, S.h_all
    if S.rings is not None:
        pipe.add(Stash(src=record_spec(dg.TOP_K, 0), nbytes=dg.TOP_K * 128,
                       dst=S.h_rec_l3), "rec_l3", x=rec0.out("rec"))
    # params fence_route: cluster 0's later weight loads (its shared-expert chunks) wait
    # for the route, so its in-order DM core runs the route -- which every routed push
    # waits for -- before copying chunks the ring holds anyway (measured: the route
    # 30-50 us behind cluster 0's shared gate|up loads)
    if p.get("fence_route", False):
        pipe.raw(lambda: ls[0].fence(rec0.out("rec").port.ends[-1]), "fence_route")
    SH = int(p.get("shared_cluster", 0))
    slot_cl = [c for c in range(NCL) if c != SH for _ in range(2)]
    recs = {0: rec0}
    for c in sorted(set(slot_cl) - {0}):
        recs[c] = pipe.add(Pull(rows=dg.TOP_K, cols=128, layout=RM, dtype=I8, src=0,
                                dst=c), f"rec_c{c}", x0=rec0.out())
    if not S.spread:
        sgu = gemv("sh_gu", SH, D, 2 * dg.I_SH, ks["x"], ha[SH].out(), h["sh_gu"])
        S.sg16 = deq("sh_g16", SH, 2 * dg.I_SH, sgu.out(), h["s_sh_gu"])
        S.sa8 = pipe.add(SwigluARow(inter=dg.I_SH, cluster=SH,
                                    inv_scale_f32bits=inv["sh_a"]), "sh_a8", g=S.sg16.out())
        sdn = gemv("sh_dn", SH, dg.I_SH, D, ks["sd"], S.sa8.out(), h["sh_dn"])
        ys = deq("ys", SH, D, sdn.out(), h["s_sh_dn"])
    else:
        # gate|up and SwiGLU on every cluster, its 704 intermediate columns
        sg16s, sa8s = [], []
        for c in range(NCL):
            gu_c = gemv(f"sh_gu_c{c}", c, D, 2 * ISH_C, ks["x"], ha[c].out(), S.h_shgu,
                        w_off=2 * ISH_C * D * c)
            sg16s.append(deq(f"sh_g16_c{c}", c, 2 * ISH_C, gu_c.out(), S.h_s_shgu,
                             2 * 2 * ISH_C * c))
            sa8s.append(pipe.add(SwigluARow(inter=ISH_C, cluster=c,
                                            inv_scale_f32bits=inv["sh_a"]),
                                 f"sh_a8_c{c}", g=sg16s[c].out()))
        # all-gather the four A-operand slices: an A-layout row concatenates along K
        # block by block, so the slices' bytes back to back ARE the whole operand
        a8b = [view(f"sh_a8b_c{c}", sa8s[c].out(),
                    PortSpec(A, I8, (1, ISH_C), mem_level=L1, cluster=c),
                    PortSpec(RM, I8, (1, 16 * ISH_C), mem_level=L1, cluster=c))
               for c in range(NCL)]
        ysl = []
        for c in range(NCL):
            ag = pipe.add(Pull(rows=1, cols=16 * ISH_C, layout=RM, dtype=I8,
                               src=tuple(range(NCL)), dst=c), f"sh_a8_all_c{c}",
                          **{f"x{i}": a8b[i].out() for i in range(NCL)})
            a8_all = view(f"sh_a8_op_c{c}", ag.out(),
                          PortSpec(RM, I8, (NCL, 16 * ISH_C), mem_level=L1, cluster=c),
                          PortSpec(A, I8, (1, dg.I_SH), mem_level=L1, cluster=c))
            dn_c = gemv(f"sh_dn_c{c}", c, dg.I_SH, SL, ks["sd"], a8_all.out(),
                        h["sh_dn"], w_off=SL * c * dg.I_SH)
            ysl.append(deq(f"ys_c{c}", c, SL, dn_c.out(), h["s_sh_dn"], 2 * SL * c))
        ys_g = pipe.add(Pull(rows=1, cols=SL, layout=RM, dtype=F16, src=tuple(range(NCL)),
                             dst=0), "ys_all", **{f"x{i}": ysl[i].out() for i in range(NCL)})
        ys = view("ys", ys_g.out(), row(SL, 0, rows=NCL), row(D, 0))
        SH = 0                                              # ys is on cluster 0
        S.sg16s, S.sa8s = sg16s, sa8s
    e16, g16, a8 = {}, {}, {}
    # params factor_first: with factor_ring, take every slot's factor chunks before any
    # routed weight chunk, so the pushes right after the route bring them all; taken in
    # the slot loop they sat between the weight chunks, each a run of its own the DM core
    # waited for (15-35 us) with the next expert's chunks queued behind it
    pre = {}
    if p.get("factor_first", False):
        for s in range(dg.TOP_K):
            c = slot_cl[s]
            pre[s] = (pipe.add(SlotLoad(slot=s, field="gu_s", cols=2 * dg.I_EXP, cluster=c,
                                        stream=fr[c]), f"gus_s{s}", rec=recs[c].out()),
                      pipe.add(SlotLoad(slot=s, field="dn_s", cols=D, cluster=c,
                                        stream=fr[c]), f"dns_s{s}", rec=recs[c].out()))
    for s in range(dg.TOP_K):
        c = slot_cl[s]
        rc = recs[c].out()
        gu = gemv(f"gu_s{s}", c, D, 2 * dg.I_EXP, ks["x"], ha[c].out(), w_slot=(s, "gu"),
                  rec=rc)
        gus = pre[s][0] if s in pre else pipe.add(
            SlotLoad(slot=s, field="gu_s", cols=2 * dg.I_EXP, cluster=c, stream=fr[c]),
            f"gus_s{s}", rec=rc)
        g16[s] = deq(f"g16_s{s}", c, 2 * dg.I_EXP, gu.out(), gus)
        a8[s] = pipe.add(SwigluARow(inter=dg.I_EXP, cluster=c, slot=s), f"a8_s{s}",
                         g=g16[s].out(), rec=rc)
        dn = gemv(f"dn_s{s}", c, dg.I_EXP, D, ks["ed"], a8[s].out(), w_slot=(s, "dn"),
                  rec=rc)
        dns = pre[s][1] if s in pre else pipe.add(
            SlotLoad(slot=s, field="dn_s", cols=D, cluster=c, stream=fr[c]),
            f"dns_s{s}", rec=rc)
        e16[s] = deq(f"e16_s{s}", c, D, dn.out(), dns)
    ys0 = ys if SH == 0 else pipe.add(
        Pull(rows=1, cols=D, layout=RM, dtype=F16, src=SH, dst=0), "ys_on0", x0=ys.out())
    out = pipe.add(AddRow(cols=D, cluster=0), "out_sh", a=h_all[0].out(), b=ys0.out())
    for s in range(dg.TOP_K):
        e0 = e16[s] if slot_cl[s] == 0 else pipe.add(
            Pull(rows=1, cols=D, layout=RM, dtype=F16, src=slot_cl[s], dst=0),
            f"e_on0_s{s}", x0=e16[s].out())
        we = pipe.add(ScaleRowBySlot(cols=D, cluster=0, slot=s), f"we_s{s}",
                      x=e0.out(), rec=rec0.out())
        out = pipe.add(AddRow(cols=D, cluster=0), f"out_s{s}", a=out.out(), b=we.out())
    S.ys, S.g16, S.a8, S.e16, S.out = ys, g16, a8, e16, out


def _pass(S):
    """The union's slots, every token through each, and the travelling combine."""
    pipe, p, h, inv, ks, view, gemv, deq = S.pipe, S.p, S.h, S.inv, S.ks, S.view, S.gemv, S.deq
    T, U, PS, ls, fr, rec0, ha_l3 = S.T, S.U, S.PS, S.ls, S.fr, S.rec0, S.ha_l3
    gput, check, f16 = S.gput, S.check, dg.f16
    if not S.spread:
        raise ValueError("params tokens > 1 through stage 8 needs shared_spread: true")
    rec_t = [view(f"rec_t{t}", rec0.out(), record_spec(T * U, 0), record_spec(U, 0),
                  t * U * 128) for t in range(T)]
    if S.rings is not None:
        pipe.add(Stash(src=record_spec(U, 0), nbytes=U * 128, dst=S.h_rec_l3), "rec_l3",
                 x=rec_t[0].out())
    if p.get("fence_route", False):
        pipe.raw(lambda: ls[0].fence(rec0.out("rec").port.ends[-1]), "fence_route")
    slot_cl = [int(v) for v in p.get("slot_clusters", [])] or \
        [(s + 1) % NCL for s in range(U)]               # cluster 0 also runs the combine
    # every slot cluster gets all T records (the combine reads token t's weights); its
    # slot loads read token 0's, the first U slots
    recs, recv0 = {0: rec0}, {0: rec_t[0]}
    for c in sorted(set(slot_cl) - {0}):
        recs[c] = pipe.add(Pull(rows=T * U, cols=128, layout=RM, dtype=I8, src=0, dst=c),
                           f"rec_c{c}", x0=rec0.out())
        recv0[c] = view(f"rec0_c{c}", recs[c].out(), record_spec(T * U, c),
                        record_spec(U, c))
    # the shared expert, spread: gate|up and SwiGLU on every cluster, its 704 columns
    sg16s, sa8s = [], []
    for c in range(NCL):
        gu_c = gemv(f"sh_gu_c{c}", c, D, 2 * ISH_C, ks["x"], ha_l3.out(), S.h_shgu,
                    w_off=2 * ISH_C * D * c, tokens=T, x_layout=A_ROW, x_level=L3)
        sg16s.append(deq(f"sh_g16_c{c}", c, 2 * ISH_C, gu_c.out(), S.h_s_shgu,
                         2 * 2 * ISH_C * c, rows=T))
        sa8s.append(pipe.add(SwigluARow(inter=ISH_C, cluster=c, rows=T,
                                        inv_scale_f32bits=inv["sh_a"]),
                             f"sh_a8_c{c}", g=sg16s[c].out()))
    a8b = [view(f"sh_a8b_c{i}", sa8s[i].out(),
                PortSpec(A_ROW, I8, (T, ISH_C), mem_level=L1, cluster=i),
                PortSpec(RM, I8, (T, 2 * ISH_C), mem_level=L1, cluster=i))
           for i in range(NCL)]
    tm = tuple(i for t in range(T) for i in range(NCL))
    r0 = tuple(t for t in range(T) for i in range(NCL))
    ysl = []
    for c in range(NCL):
        ag = pipe.add(Pull(rows=T, cols=2 * ISH_C, layout=RM, dtype=I8, src=tm, dst=c,
                           row0=r0, nrows=1), f"sh_a8_all_c{c}",
                      **{f"x{t * NCL + i}": a8b[i].out() for t in range(T)
                         for i in range(NCL)})
        a8_all = view(f"sh_a8_op_c{c}", ag.out(),
                      PortSpec(RM, I8, (T * NCL, 2 * ISH_C), mem_level=L1, cluster=c),
                      PortSpec(A_ROW, I8, (T, dg.I_SH), mem_level=L1, cluster=c))
        dn_c = gemv(f"sh_dn_c{c}", c, dg.I_SH, SL, ks["sd"], a8_all.out(), h["sh_dn"],
                    w_off=SL * c * dg.I_SH, tokens=T, x_layout=A_ROW)
        ysl.append(deq(f"ys_c{c}", c, SL, dn_c.out(), h["s_sh_dn"], 2 * SL * c, rows=T))
    ys_g = pipe.add(Pull(rows=T, cols=SL, layout=RM, dtype=F16, src=tm, dst=0, row0=r0,
                         nrows=1), "ys_all",
                    **{f"x{t * NCL + i}": ysl[i].out() for t in range(T)
                       for i in range(NCL)})
    # the union's slots, every token through each
    e16 = {}
    for s_ in range(U):
        c = slot_cl[s_]
        rc = recv0[c].out()
        gu = gemv(f"gu_s{s_}", c, D, 2 * dg.I_EXP, ks["x"], ha_l3.out(), w_slot=(s_, "gu"),
                  rec=rc, tokens=T, x_layout=A_ROW, x_level=L3)
        gus = pipe.add(SlotLoad(slot=s_, field="gu_s", cols=2 * dg.I_EXP, cluster=c,
                                rec_slots=U, stream=fr[c]), f"gus_s{s_}", rec=rc)
        g16 = deq(f"g16_s{s_}", c, 2 * dg.I_EXP, gu.out(), gus, rows=T)
        a8 = pipe.add(SwigluARow(inter=dg.I_EXP, cluster=c, slot=s_, rec_slots=U, rows=T),
                      f"a8_s{s_}", g=g16.out(), rec=rc)
        dn = gemv(f"dn_s{s_}", c, dg.I_EXP, D, ks["ed"], a8.out(), w_slot=(s_, "dn"),
                  rec=rc, tokens=T, x_layout=A_ROW)
        dns = pipe.add(SlotLoad(slot=s_, field="dn_s", cols=D, cluster=c, rec_slots=U,
                                stream=fr[c]),
                       f"dns_s{s_}", rec=rc)
        e16[s_] = deq(f"e16_s{s_}", c, D, dn.out(), dns, rows=T)
    # THE COMBINE: a running sum R [T, D], token t in row t, starts as h + ys on cluster
    # 0 and TRAVELS slot to slot, in union order, to the cluster that ran each slot: there
    # R += w_t,s e_s,t for every token (a zero weight adds nothing). One add after the
    # other is the golden's own order, so it is byte-exact; and it is a CHAIN, so every
    # hand-over waits for the one before -- fanning 22 slots' results into cluster 0
    # instead needed 33 concurrent dependency tags in one cell (32 at DepTagWidth 5).
    TD = PortSpec(RM, F16, (1, T * D), mem_level=L1, cluster=0)
    h_st = pipe.add(Pull(rows=1, cols=D, layout=RM, dtype=F16,
                         src=tuple(t % NCL for t in range(T)), dst=0), "h_st",
                    **{f"x{t}": S.hrow[t].out() for t in range(T)})
    R = pipe.add(AddRow(cols=T * D, cluster=0), "out_sh",
                 a=view("h_st_row", h_st.out(), row(D, 0, rows=T), TD).out(),
                 b=view("ys_row", ys_g.out(), row(SL, 0, rows=T * NCL), TD).out())
    r_cl = 0

    def td(c):
        return PortSpec(RM, F16, (1, T * D), mem_level=L1, cluster=c)

    for s_ in range(U):
        c = slot_cl[s_]
        r_in = R if r_cl == c else pipe.add(
            Pull(rows=T, cols=D, layout=RM, dtype=F16, src=r_cl, dst=c), f"r_to_s{s_}",
            x0=view(f"r_rows_s{s_}", R.out(), td(r_cl), row(D, r_cl, rows=T)).out())
        r_in_row = r_in.out() if r_cl == c else view(
            f"r_in_s{s_}", r_in.out(), row(D, c, rows=T), td(c)).out()
        we = pipe.add(ScaleRowBySlot(cols=D, cluster=c, slot=s_, rec_slots=U, rows=T),
                      f"we_s{s_}", x=e16[s_].out(), rec=recs[c].out())
        R = pipe.add(AddRow(cols=T * D, cluster=c), f"out_s{s_}", a=r_in_row,
                     b=view(f"we_row_s{s_}", we.out(), row(D, c, rows=T), td(c)).out())
        r_cl = c
        # the next slot of this cluster dequantises its result only once the sum has
        # taken this one's (same core, no tag): each cluster then holds ONE finished
        # slot waiting for the sum, not every slot it ran ahead with
        nxt = [q for q in range(s_ + 1, U) if slot_cl[q] == c]
        if nxt:
            pipe.raw(lambda R=R, q=nxt[0]: S.dfg.bingo_add_edge(
                R.out().port.ends[-1], e16[q].result.nodes[0]), f"hold_e16_s{nxt[0]}")
    if r_cl != 0:
        R = pipe.add(Pull(rows=T, cols=D, layout=RM, dtype=F16, src=r_cl, dst=0), "out_on0",
                     x0=view("r_rows_end", R.out(), td(r_cl), row(D, r_cl, rows=T)).out())
    S.outs.append(R)
    tk = PS["toks"]
    check("ys", ys_g, None, gput("ys", [f16(x["shared"]["y16"]) for x in tk]), 2 * D * T, st=8)
    # the first and last slot only: a stash reads its buffer whenever the DM core gets to
    # it, so a check per slot keeps every slot's result alive in L1 (the out check below
    # covers every slot byte for byte)
    for s_ in sorted({0, U - 1}):
        check(f"e16_s{s_}", e16[s_], None,
              gput(f"e16_s{s_}", [f16(x["slots"][s_]["y16"]) for x in tk]), 2 * D * T, st=8)
    check("out", R, None, gput("out", [x["out16"] for x in tk]), 2 * D * T, st=8)
    S.out = R


def checks(S):
    PREV.checks(S)
    if S.T != 1:
        return
    h, check, st = S.h, S.check, S.st
    if not S.spread:
        check("sh_g16", S.sg16, None, h["gold_sh_g"], 4 * dg.I_SH)
        check("sh_a8", S.sa8, None, h["gold_sh_a"], 16 * dg.I_SH)
    else:
        g_sg = S.data["gold"]["sh_g"]
        g_sgs = st.put("dsv2_goldT_sh_g_split", "uint16_t", np.concatenate(
            [np.concatenate([g_sg[a: a + ISH_C] for a in (ISH_C * c, dg.I_SH + ISH_C * c)])
             for c in range(NCL)]).view(np.uint16))
        for c in range(NCL):
            check(f"sh_g16_c{c}", S.sg16s[c], None, at_offset(g_sgs, 4 * ISH_C * c), 4 * ISH_C)
            check(f"sh_a8_c{c}", S.sa8s[c], None, at_offset(h["gold_sh_a"], 16 * ISH_C * c),
                  16 * ISH_C)
    check("ys", S.ys, None, h["gold_ys"], 2 * D)
    for s in range(dg.TOP_K):
        check(f"g16_s{s}", S.g16[s], None, h[f"gold_g_s{s}"], 4 * dg.I_EXP)
        check(f"a8_s{s}", S.a8[s], None, h[f"gold_a_s{s}"], 16 * dg.I_EXP)
        check(f"e16_s{s}", S.e16[s], None, h[f"gold_e_s{s}"], 2 * D)
    check("out", S.out, None, h["gold_out"], 2 * D)


if __name__ == "__main__":
    run(sys.modules[__name__])
