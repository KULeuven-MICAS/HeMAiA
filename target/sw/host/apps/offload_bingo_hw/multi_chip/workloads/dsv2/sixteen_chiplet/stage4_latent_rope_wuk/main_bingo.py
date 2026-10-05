#!/usr/bin/env python3
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""DeepSeek-V2-Lite layer 1 on sixteen chiplets, stage 4: the latent's RMSNorm, RoPE, W_UK absorbed.

Checked: cn, c8, kpe8, rot, qt16. Builds on stage 3. On the latent's cluster (LAT): the latent
c (the first 512 of kv16) normalised and quantised -> c8, and k_pe rotated -> kpe8. On every
cluster: RoPE on its heads' q_pe, then its heads' q_nope quantised and absorbed through W_UK
-> q~ (qt16), the latent-space query the attention reads.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "common"))
from dsv2_mc import A, A_ROW, D, I8, KV, KVR, L1, L3, QH, RM, RP, F16, dg, f16, load_stage, run  # noqa: E402

from libs import PortSpec  # noqa: E402
from libs.blocks import Pull, Quantize, QuantizeARow, RMSNormRow, RopeRows, RowCfg  # noqa: E402

PREV = load_stage(__file__, "stage3_wdkv")
STAGE = 4


def build(S):
    PREV.build(S)
    row, view, inv, H = S.spec_row, S.view, S.inv, S.H
    LAT, HPG = S.LAT, S.HPG
    cl = S.c(LAT)
    late = getattr(S, "latent_late", False)

    def latent():
        """W_DKV's latent on LAT: its norm and quantiser (c8), and k_pe rotated (kpe8)."""
        if late and S.p.get("wdkv_split", False):
            # params wdkv_split: half of W_DKV on LAT, half on ATT (one chip), each after its
            # cluster's W_UK -- both q~ first -- then LAT stacks the halves (stage 3's split)
            h = KV // 2
            halves = {}
            for g, c0 in ((LAT, 0), (S.ATT, h)):
                y = S.gemv(f"kv_g{g}", g, D, h, S.ks["x"], S.x8[S.chip(g)].out(),
                           S.wh[(g, "wdkv")], x_layout=A_ROW, x_level=L3)
                halves[g] = S.deq(f"kv16_g{g}", g, h, y.out(), S.h["s_kv"], 2 * c0)
            both = S.add(Pull(rows=1, cols=h, layout=RM, dtype=F16,
                              src=[S.c(LAT), S.c(S.ATT)], dst=cl), "kv16_stack", LAT,
                         x0=halves[LAT].out(), x1=halves[S.ATT].out())
            S.kv16 = view("kv16_g", LAT, both.out(),
                          PortSpec(RM, F16, (2, h), mem_level=L1, cluster=cl), row(KV, LAT))
        elif late:
            # W_DKV itself, here: after LAT's W_UK in its stream (stage 3 left it out)
            y = S.gemv(f"kv_g{LAT}", LAT, D, KV, S.ks["x"], S.x8[S.chip(LAT)].out(),
                       S.wh[(LAT, "wdkv")], x_layout=A_ROW, x_level=L3)
            S.kv16 = S.deq(f"kv16_g{LAT}", LAT, KV, y.out(), S.h["s_kv"], 0)
        kv = S.kv16
        cv = view("c_lat", LAT, kv.out(), row(KV, LAT), row(KVR, LAT))
        S.cn = S.add(RMSNormRow(cols=KVR, cluster=cl, in_level=L1), "cn", LAT, x=cv.out())
        S.c8 = S.add(Quantize(RowCfg(rows=1, cols=KVR, cluster=cl), inv_scale_f32bits=inv["c"],
                              layout=RM), "c8", LAT, x=S.cn.out())
        n = HPG + 1
        rk = S.rot[LAT] if not late else S.add(
            RopeRows(heads=HPG, kpe=True, cluster=cl), f"rotk_g{LAT}", LAT, q=S.q16[LAT].out(),
            kv=kv.out(), cos=S.staged(row(RP, level=L3, rows=n), S.h[f"cos{n}"]),
            sin=S.staged(row(RP, level=L3, rows=n), S.h[f"sin{n}"]))
        S.rot_kpe = rk
        kpe = view("kpe", LAT, rk.out(), row(RP, LAT, rows=HPG + 1), row(RP, LAT), 2 * RP * HPG)
        S.kpe8 = S.add(Quantize(RowCfg(rows=1, cols=RP, cluster=cl),
                                inv_scale_f32bits=inv["kpe"], layout=RM), "kpe8", LAT,
                       x=kpe.out())

    S.rot, S.qt16, S.rot_rows = [None] * S.G, [None] * S.G, {}
    if not late:
        # (the original order: the latent's norm first, LAT's RoPE carries k_pe)
        kv = S.kv16
        cv = view("c_lat", LAT, kv.out(), row(KV, LAT), row(KVR, LAT))
        S.cn = S.add(RMSNormRow(cols=KVR, cluster=cl, in_level=L1), "cn", LAT, x=cv.out())
        S.c8 = S.add(Quantize(RowCfg(rows=1, cols=KVR, cluster=cl), inv_scale_f32bits=inv["c"],
                              layout=RM), "c8", LAT, x=S.cn.out())
    for g in S.HG:
        with_kpe = g == LAT and not late
        n = HPG + (1 if with_kpe else 0)
        binds = dict(q=S.q16[g].out(),
                     cos=S.staged(row(RP, level=L3, rows=n), S.h[f"cos{n}"]),
                     sin=S.staged(row(RP, level=L3, rows=n), S.h[f"sin{n}"]))
        if with_kpe:
            binds["kv"] = S.kv16.out()
        S.rot[g] = S.add(RopeRows(heads=HPG, kpe=with_kpe, cluster=S.c(g)), f"rot_g{g}", g,
                         **binds)
        S.rot_rows[g] = n
    if not late:
        S.rot_kpe = S.rot[LAT]
        kpe = view("kpe", LAT, S.rot[LAT].out(), row(RP, LAT, rows=HPG + 1), row(RP, LAT),
                   2 * RP * HPG)
        S.kpe8 = S.add(Quantize(RowCfg(rows=1, cols=RP, cluster=cl),
                                inv_scale_f32bits=inv["kpe"], layout=RM), "kpe8", LAT,
                       x=kpe.out())
    # W_UK per head: the cluster's heads' q_nope quantised by one task (a segment per head),
    # one grouped GEMV (a group per head)
    for g in S.HG:
        c = S.c(g)
        qn = S.add(QuantizeARow(cols=dg.Q_NOPE, cluster=c, inv_scale_f32bits=inv["qn"],
                                segs=HPG, seg_pitch=2 * QH), f"qn_g{g}", g, x=S.q16[g].out())
        qt = S.gemv(f"qt_g{g}", g, dg.Q_NOPE, KVR, S.ks["uk"], qn.out(), S.wh[(g, "wuk")],
                    groups=HPG)
        S.qt16[g] = S.deq(f"qt16_g{g}", g, HPG * KVR, qt.out(), S.h["s_uk"],
                          2 * KVR * S.hd0[g])
    if late and S.p.get("latent_after_q", False):
        # params latent_after_q: stage 5 builds the latent after LAT's q~ and q_pe stashes,
        # and holds W_DKV's loads behind them on the in-order DM core
        S.latent_deferred = latent
    elif late:
        latent()


def checks(S):
    PREV.checks(S)
    H, HPG, LAT = S.H, S.HPG, S.LAT
    S.check("cn", S.cn, None, S.gold("cn", f16(H["cn16"])), 2 * KVR, LAT)
    S.check("c8", S.c8, None, S.gold("c8", H["c8_new"]), KVR, LAT)
    for g in S.HG:
        rows = [f16(H["qpe_rot"][S.hd0[g]: S.hd0[g] + HPG])] + \
            ([f16(H["kpe_rot"])[None]] if g == LAT else [])
        S.check(f"rot_g{g}", S.rot_kpe if g == LAT else S.rot[g], None,
                S.gold(f"rot_g{g}", np.concatenate(rows, 0)),
                2 * RP * (HPG + (g == LAT)), g)
    S.check("kpe8", S.kpe8, None, S.gold("kpe8", H["kpe8_new"]), RP, LAT)
    for g in S.HG:
        S.check(f"qt16_g{g}", S.qt16[g], None,
                S.gold(f"qt16_g{g}", f16(H["qt16"][S.hd0[g]: S.hd0[g] + HPG]).reshape(-1)),
                2 * HPG * KVR, g)


import numpy as np  # noqa: E402

if __name__ == "__main__":
    run(sys.modules[__name__])
