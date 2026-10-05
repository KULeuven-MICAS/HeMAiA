#!/usr/bin/env python3
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""DeepSeek-V2-Lite layer 1 on sixteen chiplets, stage 7: the MoE's RMSNorm, router, softmax, top 6 -- on every chip.

Checked: hn, lg16, probs, the route (every chip's record). Builds on stage 6. h is
ALL-GATHERED onto every cluster, each normalises and quantises it (ha, the operand of its
expert GEMVs); each chip's cluster 0 runs the router, the softmax and the top-6 route ITSELF,
from its own expert table -- whose addresses are its own memory chiplet's slices -- so no
record crosses a chip. The record goes to its chip's L3 (the prefetcher pushes the routed
slices from it) and to the chip's other cluster.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "common"))
from dsv2_mc import (D, F16, I8, KVR, L1, L3, RM, TOP_K, N_EXP, dg, f16,  # noqa: E402
                     load_stage, record_bytes, run)

from libs import PortSpec  # noqa: E402
from libs.blocks import (After, Fetch, MoeRoute, Pull, QuantizeARow, RMSNormRow,  # noqa: E402
                         SoftmaxRow, Stash, record_spec)

PREV = load_stage(__file__, "stage6_wuv_wo")
STAGE = 7


def build(S):
    PREV.build(S)
    row, G, inv = S.spec_row, S.G, S.inv
    h_all = S.allgather("h", {g: S.hsl[g].out() for g in range(G)},
                        lambda g: row(S.o_cols[g], g), [2 * S.o_cols[g] for g in range(G)],
                        list(range(G)), lambda g: row(D, g))
    S.h_all, S.hn, S.ha = h_all, [None] * G, [None] * G
    for g in range(G):
        c = S.c(g)
        S.hn[g] = S.add(RMSNormRow(cols=D, cluster=c, in_level=L1), f"hn_g{g}", g,
                        x=h_all[g].out())
        S.ha[g] = S.add(QuantizeARow(cols=D, cluster=c, inv_scale_f32bits=inv["h"]),
                        f"ha_g{g}", g, x=S.hn[g].out())
    S.lg16, S.pr, S.rec = {}, {}, {}
    for k in S.chips:
        g0, g1 = S.gs(k)[0], S.gs(k)[1:]
        lg = S.gemv(f"router_k{k:02x}", g0, D, N_EXP, S.ks["x"], S.ha[g0].out(),
                    S.wh[(g0, "wr")], bits=8)
        S.lg16[k] = S.deq(f"lg16_k{k:02x}", g0, N_EXP, lg.out(), S.h["s_r"])
        S.pr[k] = S.add(SoftmaxRow(cols=N_EXP, cluster=0), f"probs_k{k:02x}", g0,
                        x=S.lg16[k].out())
        table = S.staged(PortSpec(RM, I8, (N_EXP, 64), mem_level=L3), S.tables[k])
        level = L3
        if S.p.get("route_table_l1", False):
            if g0 not in S.hd0:
                raise ValueError(f"params route_table_l1: anchored on q~, which chip {k:#04x} "
                                 f"(no heads) does not compute")
            # params route_table_l1: the table copied into cluster 0's L1 during the
            # attention (after its own q~, so it does not open the DM stream), and the route
            # copies its k entries from there instead of k latency-bound L3 reads
            tspec = PortSpec(RM, I8, (N_EXP, 64), mem_level=L3)
            anchored = S.add(After(x=tspec, after=S.spec_row(S.HPG * KVR, g0)),
                             f"rtab_after_k{k:02x}", g0, x=table, after=S.qt16[g0].out())
            table = S.add(Fetch(src=tspec, cluster=0, nbytes=N_EXP * 64),
                          f"rtab_k{k:02x}", g0, x=anchored.out()).out()
            level = L1
        rec = S.add(MoeRoute(n=N_EXP, k=TOP_K, cluster=0, table_level=level),
                    f"route_k{k:02x}", g0, p=S.pr[k].out(), table=table)
        S.rec[g0] = rec
        # the prefetcher's copy, in this chip's L3
        S.add(Stash(src=record_spec(TOP_K, 0), nbytes=TOP_K * 128, dst=S.h_rec_l3),
              f"rec_l3_k{k:02x}", g0, x=rec.out("rec"))
        for g in g1:
            S.rec[g] = S.add(Pull(rows=TOP_K, cols=128, layout=RM, dtype=I8, src=0,
                                  dst=S.c(g)), f"rec_g{g}", g, x0=rec.out())


def checks(S):
    PREV.checks(S)
    H = S.H
    k0 = S.chips[0]
    S.check("hn", S.hn[0], None, S.gold("hn", f16(H["hn"])), 2 * D, 0)
    S.check("lg16", S.lg16[k0], None, S.gold("lg", f16(H["logits16"])), 2 * N_EXP, 0)
    S.check("probs", S.pr[k0], None, S.gold("p", f16(H["p16"])), 2 * N_EXP, 0)
    for k in S.chips:
        g0 = S.gs(k)[0]
        rec = record_bytes(S.data["ids"], S.data["w16"].view(np.uint16), S.table_bytes[k])
        S.check(f"route_k{k:02x}", S.rec[g0], "rec", S.gold(f"rec_k{k:02x}", rec),
                TOP_K * 128, g0)


import numpy as np  # noqa: E402

if __name__ == "__main__":
    run(sys.modules[__name__])
