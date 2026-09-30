#!/usr/bin/env python3
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""DeepSeek-V2-Lite layer 1, stage 4: the latent's RMSNorm, RoPE on q_pe and k_pe, W_UK absorbed.

Checked: cn, c8, rot, kpe8, qt16. Builds on stage 3. Per token: the latent c (the first 512
of kv16) normalised and quantised on cluster 0 -> c8; RoPE at the token's position on every
cluster's q_pe (and k_pe on cluster 0) -> kpe8; then each cluster's heads' q_nope quantised
and absorbed through W_UK -> q~ (qt16), the latent-space query the attention reads.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "common"))
from dsv2_staged import (A, HPC, I8, KV, KVR, L1, L3, NCL, QC, QH, RM, RP, F16,  # noqa: E402
                         dg, load_stage, row, run)

from libs import PortSpec  # noqa: E402
from libs.blocks import Pull, Quantize, QuantizeARow, RMSNormRow, RopeRows, RowCfg  # noqa: E402

PREV = load_stage(__file__, "stage3_wdkv")
STAGE = 4


def build(S):
    PREV.build(S)
    if S.FROM != 1:
        return
    T, pipe, inv, dkv, view = S.T, S.pipe, S.inv, S.dkv, S.view
    q16, kv16 = S.q16, S.kv16
    # per token: the latent's norm and quantiser (cluster 0), RoPE at the token's position
    # on every cluster's q_pe (and k_pe on cluster 0)
    cns, c8s, kpe8s = [], [], []
    rots = [[None] * T for _ in range(NCL)]
    if dkv[0] != KV:
        # each cluster's kv16 slice as 64-column rows (claimed up to the widest slice's
        # count, of which the gather reads only its own rows)
        mx = max(dkv) // 64
        kv64 = {c: view(f"kv16_r64_c{c}", kv16[c].out(), row(dkv[c], c, rows=T),
                        row(64, c, rows=T * mx)) for c in range(NCL) if dkv[c]}
    for t in range(T):
        sfx = "" if T == 1 else f"_t{t}"
        if dkv[0] == KV:
            kvt = kv16[0] if T == 1 else view(f"kv16{sfx}", kv16[0].out(),
                                              row(KV, 0, rows=T), row(KV, 0), 2 * KV * t)
        else:
            # token t's row from every cluster's slice, as 64-column rows, onto cluster 0
            cs = [c for c in range(NCL) if dkv[c]]
            nr = [dkv[c] // 64 for c in cs]
            g = pipe.add(Pull(rows=T * max(nr), cols=64, layout=RM, dtype=F16,
                              src=tuple(cs), dst=0, row0=tuple(t * n for n in nr),
                              nrows=tuple(nr)), f"kv16_all{sfx}",
                         **{f"x{i}": kv64[c].out() for i, c in enumerate(cs)})
            kvt = view(f"kv16_row{sfx}", g.out(), row(64, 0, rows=sum(nr)), row(KV, 0))
        cv = view(f"c_lat{sfx}", kvt.out(), row(KV, 0), row(KVR, 0))
        cns.append(pipe.add(RMSNormRow(cols=KVR, cluster=0, in_level=L1), f"cn{sfx}",
                            x=cv.out()))
        c8s.append(pipe.add(Quantize(RowCfg(rows=1, cols=KVR, cluster=0),
                                     inv_scale_f32bits=inv["c"], layout=RM), f"c8{sfx}",
                            x=cns[-1].out()))
        for c in range(NCL):
            n = HPC + (1 if c == 0 else 0)
            qtk = q16[c] if T == 1 else view(f"q16_c{c}{sfx}", q16[c].out(),
                                             row(QC, c, rows=T), row(QC, c), 2 * QC * t)
            binds = dict(q=qtk.out(),
                         cos=S.staged(row(RP, None, L3, rows=n), S.rope_h[("cos", n, t)]),
                         sin=S.staged(row(RP, None, L3, rows=n), S.rope_h[("sin", n, t)]))
            if c == 0:
                binds["kv"] = kvt.out()
            rots[c][t] = pipe.add(RopeRows(heads=HPC, kpe=(c == 0), cluster=c),
                                  f"rot_c{c}{sfx}", **binds)
        kpe = view(f"kpe{sfx}", rots[0][t].out(), row(RP, 0, rows=HPC + 1), row(RP, 0),
                   2 * RP * HPC)
        kpe8s.append(pipe.add(Quantize(RowCfg(rows=1, cols=RP, cluster=0),
                                       inv_scale_f32bits=inv["kpe"], layout=RM),
                              f"kpe8{sfx}", x=kpe.out()))
    # W_UK per head: every token's heads' q_nope quantised by one task (T * 4 segments of
    # the T rows of q16), one grouped GEMV task per token over the shared chunk
    qt16 = []
    for c in range(NCL):
        qsrc = q16[c] if T == 1 else view(f"q16_flat_c{c}", q16[c].out(),
                                          row(QC, c, rows=T), row(T * QC, c))
        qn = pipe.add(QuantizeARow(cols=dg.Q_NOPE, cluster=c, inv_scale_f32bits=inv["qn"],
                                   segs=HPC * T, seg_pitch=2 * QH), f"qn_c{c}",
                      x=qsrc.out())
        qnx = qn if T == 1 else view(
            f"qn_rows_c{c}", qn.out(),
            PortSpec(A, I8, (1, T * HPC * dg.Q_NOPE), mem_level=L1, cluster=c),
            PortSpec(A, I8, (T, HPC * dg.Q_NOPE), mem_level=L1, cluster=c))
        qt = S.gemv(f"qt_c{c}", c, dg.Q_NOPE, KVR, S.ks["uk"], qnx.out(), S.h["wuk"],
                    HPC * c * dg.Q_NOPE * KVR, groups=HPC, tokens=T)
        qt16.append(S.deq(f"qt16_c{c}", c, HPC * KVR, qt.out(), S.h["s_uk"],
                          2 * HPC * KVR * c, rows=T))
    S.cns, S.c8s, S.kpe8s, S.rots, S.qt16 = cns, c8s, kpe8s, rots, qt16
    S.cn, S.c8, S.kpe8 = cns[0], c8s[0], kpe8s[0]              # the one-token stages 5+
    S.rot = [rots[c][0] for c in range(NCL)]


def checks(S):
    PREV.checks(S)
    h, check = S.h, S.check
    if S.T == 1:
        check("cn", S.cn, None, h["gold_cn"], 2 * KVR)
        check("c8", S.c8, None, h["gold_c8"], KVR)
        for c in range(NCL):
            check(f"rot_c{c}", S.rot[c], None, h[f"gold_rot_c{c}"], 2 * RP * (HPC + (c == 0)))
        check("kpe8", S.kpe8, None, h["gold_kpe8"], RP)
        for c in range(NCL):
            check(f"qt16_c{c}", S.qt16[c], None, h[f"gold_qt16_c{c}"], 2 * HPC * KVR)
    elif S.FROM == 1:
        T, gput = S.T, S.gput
        for t in range(T):
            tk = S.toks[t]
            check(f"cn_t{t}", S.cns[t], None, gput(f"cn_t{t}", [tk["cn"]]), 2 * KVR, st=4)
            check(f"c8_t{t}", S.c8s[t], None, gput(f"c8_t{t}", [tk["c8"]]), KVR, st=4)
            for c in range(NCL):
                rows_ = [tk["qpe_rot"][HPC * c: HPC * (c + 1)]] + \
                    ([tk["kpe_rot"]] if c == 0 else [])
                check(f"rot_c{c}_t{t}", S.rots[c][t], None, gput(f"rot_c{c}_t{t}", rows_),
                      2 * RP * (HPC + (c == 0)), st=4)
            check(f"kpe8_t{t}", S.kpe8s[t], None, gput(f"kpe8_t{t}", [tk["kpe8"]]), RP, st=4)
        for c in range(NCL):
            check(f"qt16_c{c}", S.qt16[c], None,
                  gput(f"qt16_c{c}", [tk["qt16"][HPC * c: HPC * (c + 1)] for tk in S.toks]),
                  2 * HPC * KVR * T, st=4)


if __name__ == "__main__":
    run(sys.modules[__name__])
