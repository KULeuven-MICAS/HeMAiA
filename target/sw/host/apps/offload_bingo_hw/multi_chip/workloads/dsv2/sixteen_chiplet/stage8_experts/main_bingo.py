#!/usr/bin/env python3
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""DeepSeek-V2-Lite layer 1 on sixteen chiplets, stage 8: the shared and routed experts, the combine -> out.

Checked: ys, a slot's expert output, out -- per cluster. Builds on stage 7: THE WHOLE LAYER.
Every expert is split over all eight clusters: gate|up by intermediate columns (352 of the
shared's 2,816, 192 or 160 of a routed expert's 1,408), the SwiGLU output ALL-GATHERED onto
every cluster, down by output columns (256 a cluster). A routed slot's weights come from the
chip's own record: each cluster streams its slice of the chip's slice (record field + its
offset). Each cluster then combines ITS 256 output columns: h + shared, then each slot
weighted, in top-k order -- the golden's own sum, column by column, so nothing is gathered.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "common"))
from dsv2_mc import (A, D, F16, I8, I_EXP, I_SH, L1, RM, TOP_K, f16, load_stage, run)  # noqa: E402

from libs import PortSpec  # noqa: E402
from libs.blocks import AddRow, Join, ScaleRowBySlot, SlotLoad, SwigluARow  # noqa: E402

PREV = load_stage(__file__, "stage7_router")
STAGE = 8
LAG = 0            # experts built between one's gate|up and its all-gather + down: > 0
                   # needs more dependency tags than DepTagWidth 4 has (22 for lag 1)


def _a8_spec(g, c, n):
    return PortSpec(A, I8, (1, n), mem_level=L1, cluster=c)


def _routed_colgroup(S, hx, slot_stream, lag=0, first=None, first_down=None):
    """The routed slots, column by column (params expert_map colgroup, see dsv2_mc.py).

    Slot s runs on its column's clusters: gate|up by intermediate units (352 each of 1,408),
    the SwiGLU output all-gathered within the column -- over the column's own chip-to-chip
    link, which carries no weight pushes -- then down for the cluster's own 256 output
    columns and its partner's (512 in one GEMV). After every slot, each cluster hands its
    partner the partner's halves, so every cluster ends with every slot at its own columns
    and combines them in top-k order, as the golden adds.

    `lag` > 0 (params expert_gather_lag): a slot's all-gather and down are built `lag` slots
    of its column later, so each cluster's GEMM core runs the next slot's gate|up while the
    gather waits for the column's slowest cluster (each such wait would idle the GEMM).
    `first` = (unit, its SwiGLU outputs) heads the queue -- the shared expert, whose down
    `first_down(unit, a8)` builds."""
    G, ks = S.G, S.ks
    full = lambda st: _spec(st)
    e16w = [dict() for _ in range(TOP_K)]
    last_gu = {}            # g -> the newest gate|up GEMV built on cluster g

    def collect_late(name, gl):
        """params lag_collect_late: each cluster's collect of `name` after the last weight load
        of the gate|up built just before it, on the same in-order DM core. Sorted as early as
        its edges allow, a collect sits among those loads and, waiting for the slowest cluster
        of the column, holds every load behind it: the GEMM starves mid-expert."""
        if not S.p.get("lag_collect_late", False):
            return
        for g in gl:
            if g in last_gu and (name, g) in S.xget:
                # every read of it, each from the load (fan-out). Chained instead (load -> read 0
                # -> read 1 ...) to save dependency tags, the reads spread out between the next
                # loads, and the last one waits behind a routed push run
                S.pipe.raw(lambda a=last_gu[g], b=S.xget[(name, g)]: [S.dfg.bingo_add_edge(
                    a.result.extra["loads"][-1], n) for n in b.result.nodes],
                    f"late_{name}_g{g}")

    def gate_up(s):
        gi = S.slot_group[s]
        a8 = {}
        for g in S.gcl[gi]:
            c, n, rc = S.c(g), S.exg_cols[g], S.rec[g].out()
            gu = S.gemv(f"gu_s{s}_g{g}", g, D, 2 * n, ks["x"], hx[g], w_slot=(s, "gu"), rec=rc)
            last_gu[g] = gu
            gus = S.add(SlotLoad(slot=s, field="gu_s", cols=2 * n, cluster=c,
                                 offset=S.slot_off[g]["gu_s"], stream=slot_stream(g)),
                        f"gus_s{s}_g{g}", g, rec=rc)
            g16 = S.deq(f"g16_s{s}_g{g}", g, 2 * n, gu.out(), gus)
            a8[g] = S.add(SwigluARow(inter=n, cluster=c, slot=s), f"a8_s{s}_g{g}", g,
                          g=g16.out(), rec=rc).out()
        return a8

    def gather_down(s, a8):
        gi = S.slot_group[s]
        gl = S.gcl[gi]
        S.a8_all[s] = S.allgather(f"a8_s{s}", a8, lambda g: _a8_spec(g, S.c(g), S.exg_cols[g]),
                                  [16 * S.exg_cols[g] for g in gl], gl,
                                  lambda g: _a8_spec(g, S.c(g), I_EXP), chips=S.groups[gi])
        if lag:
            collect_late(f"a8_s{s}", gl)
        for g in gl:
            c, n, rc = S.c(g), 2 * S.o_cols[g], S.rec[g].out()
            dn = S.gemv(f"dn_s{s}_g{g}", g, I_EXP, n, ks["ed"], S.a8_all[s][g].out(),
                        w_slot=(s, "dn"), rec=rc)
            dns = S.add(SlotLoad(slot=s, field="dn_s", cols=n, cluster=c,
                                 offset=S.slot_off[g]["dn_s"], stream=slot_stream(g)),
                        f"dns_s{s}_g{g}", g, rec=rc)
            e16w[s][g] = S.deq(f"e16w_s{s}_g{g}", g, n, dn.out(), dns)

    if not lag and first is None:
        for s in range(TOP_K):
            gather_down(s, gate_up(s))
    else:
        # step j: the j-th slot of every column (the columns' clusters are disjoint, so a
        # step's slots run side by side); a step's gathers and downs wait `lag` steps
        cols = [[s for s in range(TOP_K) if S.slot_group[s] == gi] for gi in range(len(S.gcl))]
        # params sh_down_late: the shared expert's down goes just before the last step's, so
        # its GEMVs cover the last slot's all-gather -- the one nothing else would (its weights
        # then reach the ring after the routed ones before them)
        late = first is not None and bool(S.p.get("sh_down_late", False))
        pending = [[first]] if first is not None and not late else []
        for j in range(max(len(c) for c in cols)):
            pending.append([(s, gate_up(s)) for c in cols if j < len(c) for s in [c[j]]])
            while len(pending) > lag:
                for u, a8 in pending.pop(0):
                    (first_down if u == "sh" else gather_down)(u, a8)
                    if u == "sh":
                        collect_late("sh_a8", range(G))
        if late:
            first_down(*first)
            collect_late("sh_a8", range(G))
        for step in pending:
            for u, a8 in step:
                (first_down if u == "sh" else gather_down)(u, a8)
    # own halves in place; the partner's halves of the other column's slots handed over
    half = lambda g: PortSpec(RM, F16, (1, S.o_cols[g]), mem_level=L1, cluster=S.c(g))
    for s in range(TOP_K):
        for g in S.gcl[S.slot_group[s]]:
            S.e16[s][g] = S.view(f"e16_s{s}_g{g}", g, e16w[s][g].out(), full(e16w[s][g]),
                                 half(g))
    for g in range(G):
        p = S.partner[g]
        sp = [s for s in range(TOP_K) if S.slot_group[s] == S.group_of[p]]
        parts = [S.view(f"e16h_s{s}_g{p}", p, e16w[s][p].out(), full(e16w[s][p]), half(p),
                        2 * S.o_cols[p]).out() for s in sp]
        nb = 2 * S.o_cols[g]
        got = S.xfer(f"ex_g{g}", p, parts, half(p), nb, g,
                     PortSpec(RM, F16, (len(sp), S.o_cols[g]), mem_level=L1, cluster=S.c(g)))
        for i, s in enumerate(sp):
            S.e16[s][g] = S.view(f"e16_s{s}_g{g}", g, got.out(), full(got), half(g), i * nb)


def _spec(stage):
    """A one-output stage's output spec."""
    return next(iter(stage.template.outputs.values()))


def build(S):
    PREV.build(S)
    G, ks, inv = S.G, S.ks, S.inv
    lag = int(S.p.get("expert_gather_lag", LAG))
    # THE ORDER IS THE SCHEDULE. Every DM core runs its tasks in order, and so does every
    # GEMM core, so the order the blocks are built in is the order they run in. An expert
    # (the shared one, then each routed slot) is gate|up + SwiGLU, an all-gather of the
    # SwiGLU output (remote reads, over links that push weights the other way), then down.
    # Built expert by expert, each all-gather sat between its gate|up and its down on the
    # DM and GEMM cores, and the next expert's gate|up queued behind it: seven global round
    # trips in a row on the critical path. Here an expert's all-gather and down are built
    # `lag` experts later, so the GEMM core has the next gate|ups to run while a gather is
    # in flight. More than a few gathers in flight at once do not fit the dependency tags:
    # every stash -> collect edge between two DM cores is live until the collect runs.
    #
    # Every expert GEMV also reads ha only after its chip's top-6 (`hx`): the shared
    # expert does not need the route, but started before it, its weight loads took the DM
    # core the router's own loads queue on, and their instruction refills starved the
    # softmax's. Its weights still stream into the ring in the background either way.
    # (a Join: ha's own buffer, no node, no copy -- only the extra wait on the record)
    after_route = bool(S.p.get("expert_after_route", True))
    hx = [S.add(Join(parts=[_spec(S.ha[g]), _spec(S.rec[g])], dst=_spec(S.ha[g])),
                f"hx_g{g}", g, x0=S.ha[g].out(), x1=S.rec[g].out()).out() if after_route
          else S.ha[g].out() for g in range(G)]
    # `slot_stream`: a routed expert's dequant factors come PUSHED through the cluster's
    # weight ring, right after its weights, instead of PULLED across the link by the DM
    # core (a pull waits for the push stream to drain while the in-order DM core holds
    # every later task behind it).
    slot_stream = (lambda g: S.ls[g]) if S.p.get("slot_stream", False) else (lambda g: None)
    S.sg16, S.ys = [None] * G, [None] * G
    S.e16 = [[None] * G for _ in range(TOP_K)]
    # params sh_after_h: the shared expert's first loads wait for the cluster's own h gather.
    # Loaded as soon as W_O freed their slabs, one sat between h's AddRow and its stash on the
    # in-order DM core and held the layer's last all-gather back
    if S.p.get("sh_after_h", False):
        for g in range(G):
            S.pipe.raw(lambda g=g: S.ls[g].wait_for(S.xget[("h", g)].result.nodes[-1]),
                       f"sh_after_h_g{g}")
    S.a8_all = [None] * TOP_K
    # params slot_after_gather: on every cluster, routed slot s's weight loads and its slot
    # copies (gus, dns) wait for the previous expert's all-gather on that cluster (the shared
    # expert's for slot 0). The sort places a task as early as its edges allow, so a slot's
    # loads -- waiting for pushes that start only after the route -- and its slot copies sat
    # in the in-order DM queue ahead of the previous expert's gather reads, and the gather, a
    # step every cluster waits for, waited behind them.
    order_slots = bool(S.p.get("slot_after_gather", False))
    prev_gather = lambda s: "sh_a8" if s == 0 else f"a8_s{s - 1}"

    def after_gather(gname, g, stage_, label):
        """Order stage_'s first node after the all-gather `gname` on cluster g."""
        if order_slots:
            S.pipe.raw(lambda: S.dfg.bingo_add_edge(S.xget[(gname, g)].result.nodes[-1],
                                                    stage_.result.nodes[0]),
                       f"after_{gname}_{label}")

    def gate_up(u):
        """Expert u's gate|up and SwiGLU on every cluster: {g: its SwiGLU output}."""
        a8 = {}
        if order_slots and u != "sh":
            for g in range(G):
                S.pipe.raw(lambda g=g: S.ls[g].wait_for(
                    S.xget[(prev_gather(u), g)].result.nodes[-1]), f"slot{u}_loads_g{g}")
        for g in range(G):
            c = S.c(g)
            if u == "sh":
                n = S.sh_cols[g]
                # params sh_wait_route: the shared expert's loads really wait for its x (and
                # so for the route) on the in-order DM core, ahead of nothing the router needs
                gu = S.gemv(f"sh_gu_g{g}", g, D, 2 * n, ks["x"], hx[g], S.wh[(g, "sh_gu")],
                            first=after_route,
                            wait_x=after_route and bool(S.p.get("sh_wait_route", False)))
                S.sg16[g] = S.deq(f"sh_g16_g{g}", g, 2 * n, gu.out(), S.h_s_shgu,
                                  2 * 2 * S.sh0[g])
                a8[g] = S.add(SwigluARow(inter=n, cluster=c, inv_scale_f32bits=inv["sh_a"]),
                              f"sh_a8_g{g}", g, g=S.sg16[g].out()).out()
                continue
            s, n, rc = u, S.ex_cols[g], S.rec[g].out()
            gu = S.gemv(f"gu_s{s}_g{g}", g, D, 2 * n, ks["x"], hx[g], w_slot=(s, "gu"),
                        rec=rc)
            gus = S.add(SlotLoad(slot=s, field="gu_s", cols=2 * n, cluster=c,
                                 offset=S.slot_off[g]["gu_s"], stream=slot_stream(g)),
                        f"gus_s{s}_g{g}", g, rec=rc)
            after_gather(prev_gather(s), g, gus, f"gus_s{s}_g{g}")
            g16 = S.deq(f"g16_s{s}_g{g}", g, 2 * n, gu.out(), gus)
            a8[g] = S.add(SwigluARow(inter=n, cluster=c, slot=s), f"a8_s{s}_g{g}", g,
                          g=g16.out(), rec=rc).out()
        return a8

    def gather_down(u, a8):
        """Expert u's all-gather and down projection on every cluster."""
        if u == "sh":
            S.sh_a8 = S.allgather("sh_a8", a8, lambda g: _a8_spec(g, S.c(g), S.sh_cols[g]),
                                  [16 * S.sh_cols[g] for g in range(G)], list(range(G)),
                                  lambda g: _a8_spec(g, S.c(g), I_SH))
            for g in range(G):
                n = S.o_cols[g]
                dn = S.gemv(f"sh_dn_g{g}", g, I_SH, n, ks["sd"], S.sh_a8[g].out(),
                            S.wh[(g, "sh_dn")])
                S.ys[g] = S.deq(f"ys_g{g}", g, n, dn.out(), S.h["s_sh_dn"], 2 * S.o0[g])
            return
        s = u
        S.a8_all[s] = S.allgather(f"a8_s{s}", a8, lambda g: _a8_spec(g, S.c(g), S.ex_cols[g]),
                                  [16 * S.ex_cols[g] for g in range(G)], list(range(G)),
                                  lambda g: _a8_spec(g, S.c(g), I_EXP))
        for g in range(G):
            c, n, rc = S.c(g), S.o_cols[g], S.rec[g].out()
            dn = S.gemv(f"dn_s{s}_g{g}", g, I_EXP, n, ks["ed"], S.a8_all[s][g].out(),
                        w_slot=(s, "dn"), rec=rc)
            dns = S.add(SlotLoad(slot=s, field="dn_s", cols=n, cluster=c,
                                 offset=S.slot_off[g]["dn_s"], stream=slot_stream(g)),
                        f"dns_s{s}_g{g}", g, rec=rc)
            after_gather(f"a8_s{s}", g, dns, f"dns_s{s}_g{g}")
            S.e16[s][g] = S.deq(f"e16_s{s}_g{g}", g, n, dn.out(), dns)

    if S.expert_map == "colgroup":
        if lag:
            # the shared expert queues like a routed slot: the first slots' gate|ups run
            # on the GEMM cores while its all-gather crosses the pushing links
            _routed_colgroup(S, hx, slot_stream, lag, ("sh", gate_up("sh")), gather_down)
        else:
            gather_down("sh", gate_up("sh"))
            _routed_colgroup(S, hx, slot_stream)
    else:
        pending = []
        for u in ["sh"] + list(range(TOP_K)):
            pending.append((u, gate_up(u)))
            if len(pending) > lag:
                gather_down(*pending.pop(0))
        for u, a8 in pending:
            gather_down(u, a8)
    # ---- the combine, each cluster its own columns -----------------------------------------
    S.out = [None] * G
    for g in range(G):
        c, n = S.c(g), S.o_cols[g]
        R = S.add(AddRow(cols=n, cluster=c), f"out_sh_g{g}", g, a=S.hsl[g].out(),
                  b=S.ys[g].out())
        for s in range(TOP_K):
            we = S.add(ScaleRowBySlot(cols=n, cluster=c, slot=s), f"we_s{s}_g{g}", g,
                       x=S.e16[s][g].out(), rec=S.rec[g].out())
            R = S.add(AddRow(cols=n, cluster=c), f"out_s{s}_g{g}", g, a=R.out(), b=we.out())
        S.out[g] = R
        S.outs.append(R)


def checks(S):
    PREV.checks(S)
    H = S.H
    for g in range(S.G):
        o0, n = S.o0[g], S.o_cols[g]
        S.check(f"ys_g{g}", S.ys[g], None, S.gold(f"ys_g{g}", f16(H["shared"]["y16"][o0: o0 + n])),
                2 * n, g)
    for s in range(TOP_K):
        S.check(f"e16_s{s}_g0", S.e16[s][0], None,
                S.gold(f"e16_s{s}_g0", f16(H["slots"][s]["y16"][S.o0[0]: S.o0[0] + S.o_cols[0]])),
                2 * S.o_cols[0], 0)
    for g in range(S.G):
        o0, n = S.o0[g], S.o_cols[g]
        S.check(f"out_g{g}", S.out[g], None, S.gold(f"out_g{g}", f16(H["out16"][o0: o0 + n])),
                2 * n, g)


if __name__ == "__main__":
    run(sys.modules[__name__])
