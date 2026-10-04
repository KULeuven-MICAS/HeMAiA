#!/usr/bin/env python3
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""DeepSeek-V2-Lite layer 1 on six chiplets, stage 5: cache append, Q8, MLA attention over L+1 keys.

Checked: q8, ot, xt. Builds on stage 4. The latent's cluster appends the token's c8 and kpe8
to both cache copies in its chip's L3; every cluster's q~ and q_pe rows are GATHERED onto the
attention's cluster (ATT, on the same chip as the cache) -- its chip's clusters read theirs
locally, the other chips' across the links -- and assembled into Q8; the attention runs there,
all 16 heads over the 512 keys, as the golden does.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "common"))
from dsv2_mc import (A, F16, HEADS, I8, KV, KVR, L1, L3, RM, RP, BingoMemSymbol,  # noqa: E402
                     _l3, dg, load_stage, run)

from libs import PortSpec  # noqa: E402
from libs.blocks import (AddRow, After, CacheAppend, Collect, Join,  # noqa: E402
                         LoadStream, MlaAttention, MlaDimOut, MlaDimPV, MlaRowMax, MlaShardOut,
                         MlaShardP, MlaShardPV, MlaShardScores, Pull, QAssemble, Stash)

PREV = load_stage(__file__, "stage4_latent_rope_wuk")
STAGE = 5


def build(S):
    PREV.build(S)
    row, view, data, h, HPG = S.spec_row, S.view, S.data, S.h, S.HPG
    LAT, ATT = S.LAT, S.ATT
    if S.chip(LAT) != S.chip(ATT):
        raise ValueError("params latent_cluster / att_cluster: the attention reads the cache "
                         "the append writes, in one chip's L3")
    cap = data["cap"]

    def append():
        S.app = S.add(CacheAppend(pos=data["L"], cap=cap, cluster=S.c(LAT)), "append", LAT,
                      c8=S.c8.out(), kpe8=S.kpe8.out(),
                      key=S.staged(PortSpec(A, I8, (cap, KV), mem_level=L3), h["key"]),
                      val=S.staged(PortSpec(A, I8, (KVR, cap), mem_level=L3), h["val"]))

    deferred = getattr(S, "latent_deferred", None)
    if deferred is not None:
        # params latent_after_q (with latent_late): the attention starts on q8 alone -- only
        # its last tile reads the appended row -- so the queries go first. Built after them,
        # W_DKV would still LOAD first: the in-order DM core takes a load as early as its slab
        # frees, and each waits for its push, while LAT's q~ / q_pe stashes queue behind it.
        # Its loads now wait for those stashes. _q8_at_att runs this once it has built them
        # (also for the column split, whose Q8 comes from ATT's).
        if S.p.get("att_split", False) and (S.p.get("att_split_mode", "key") != "dim" or
                                            S.p.get("att_split_chips") is None):
            raise ValueError("params latent_after_q: the one-cluster attention or the column "
                             "split (att_split_mode dim with att_split_chips)")

        def after_q8():
            put = [S.xput[(nm, LAT)] for nm in ("qt", "qpe")]
            S.pipe.raw(lambda: [S.ls[LAT].wait_for(s.out().port.ends[-1]) for s in put],
                       "latent_after_q")
            deferred()
            append()
        S.after_q8 = after_q8
    else:
        append()
    ca = S.c(ATT)
    if S.p.get("att_split", False):
        if S.p.get("att_split_mode", "key") == "dim":
            return _build_split_dim(S, cap)
        return _build_split(S, cap)
    _q8_at_att(S)
    # params att_stream: K and V tiles through slabs of their own, so ATT's weight stream (its
    # W_UV and W_O) keeps loading while the attention runs
    if S.p.get("att_stream", False):
        ks_ = LoadStream(S.ctx_of(ATT), ca, nbytes=dg.BC * KV, nbuf=1, name="kstream")
        vs_ = LoadStream(S.ctx_of(ATT), ca, nbytes=KVR * dg.BC, nbuf=1, name="vstream")
    else:
        ks_, vs_ = S.ls[ATT], None
    S.att = S.add(MlaAttention(keys=data["keys"], cap=cap, k_s=S.ks["s"], k_o=data["k_o"],
                               a_exp=data["a_exp"], a_n_f32bits=data["a_n"], cluster=ca,
                               bc=dg.BC, stream=ks_, vstream=vs_),
                  "attn", ATT, q8=S.q8.out(), key=S.app.out("key"), val=S.app.out("val"))
    # params att_loads_first: ATT's next weight loads (W_UV, W_O: needed after the attention)
    # wait for the attention's last K / V tile -- the in-order DM core otherwise loaded them as
    # their slabs freed, ahead of tile 7, and the last QK waited for its keys
    if S.p.get("att_loads_first", False):
        S.pipe.raw(lambda: [S.ls[ATT].wait_for(n) for n in S.att.result.extra["last_loads"]],
                   "att_loads_first")


def _q8_fused(S):
    """params q_fused: _q8_at_att with ONE gather. Each cluster stashes its heads as rows of
    [q~ | q_pe] (1,152 B a head) into its chip's run, so ATT reads one run per chip -- four
    reads instead of eight, three of them across links that are pushing weights, each
    waiting for a push to end -- and QAssemble reads both halves at that row pitch."""
    row, view, HPG, ATT, ca = S.spec_row, S.view, S.HPG, S.ATT, S.c(S.ATT)
    W = 2 * (KVR + RP)
    sym = S.xsym("dsv2_xchg_qq", S.G * HPG * W)
    runs = []
    for k in S.chips:
        parts, st = [], []
        for g in S.gs(k):
            for h in range(HPG):
                at = (g * HPG + h) * W
                for nm, src, spec, n, off in (
                        ("t", S.qt16[g], row(HPG * KVR, g), KVR, 0),
                        ("p", S.rot[g], row(RP, g, rows=S.rot_rows[g]), RP, 2 * KVR)):
                    one = row(n, g)
                    v = view(f"qq{nm}_g{g}_h{h}", g, src.out(), spec, one, 2 * n * h)
                    st.append(S.add(Stash(src=one, nbytes=2 * n,
                                          dst=BingoMemSymbol(sym, at + off, chip_id=k)),
                                    f"xput_qq{nm}_g{g}_h{h}", g, x=v.out()))
                    parts.append(_l3(one))
                    # the cluster's last q~ and last q_pe stash, by name (latent_after_q,
                    # q_stash_first wait for both). One entry for both was the LAST BUILT, a
                    # q_pe stash, which runs long before q~ exists: the waits were on q_pe only
                    S.xput[("qt" if nm == "t" else "qpe", g)] = st[-1]
        runs.append(S.add(Join(parts=parts, dst=PortSpec(RM, I8, (1, len(S.gs(k)) * HPG * W),
                                                         mem_level=L3)),
                          f"xrun_qq_k{k:02x}", at_chip=k,
                          **{f"x{i}": s.out() for i, s in enumerate(st)}))
    nb = [len(S.gs(k)) * HPG * W for k in S.chips]
    qq = S.add(Collect(parts=[PortSpec(RM, I8, (1, n), mem_level=L3) for n in nb], nbytes=nb,
                       dst=PortSpec(RM, F16, (HEADS, KVR + RP), mem_level=L1, cluster=ca),
                       cluster=ca),
               "xget_qq", ATT, **{f"x{i}": r.out() for i, r in enumerate(runs)})
    S.q_collects = [qq]
    S.q8 = S.add(QAssemble(inv_qt_f32bits=S.inv["qt"], inv_qpe_f32bits=S.inv["qpe"],
                           cluster=ca, fused=True), "q8", ATT, qq=qq.out())


def _q8_at_att(S):
    """Every cluster's q~ and q_pe rows GATHERED onto ATT -- its chip's clusters read theirs
    locally, the other chips' across the links -- and assembled into Q8 there: S.q8. Then
    S.after_q8, if a stage left one (latent_after_q)."""
    (_q8_fused if S.p.get("q_fused", False) else _q8_plain)(S)
    # params q_stash_first: on every cluster, the next weight loads wait for its own q~ / q_pe
    # stashes. The in-order DM core otherwise loads W_UV and W_O (needed after the attention)
    # as soon as their slabs free; a load waits for its push, and the RoPE, factor and stash
    # tasks of the queries -- the attention's whole input -- queue behind it.
    # On ATT the next loads are the attention's own K / V tiles: they wait for the q gather's
    # last read instead -- placed after ATT's stash, the six tile loads hold the reads of the
    # other chips' q~ back, and QK0 waits for those reads (Q8), not for the tiles.
    if S.p.get("q_stash_first", False):
        for g in range(S.G):
            put = [S.xput[(nm, g)] for nm in ("qt", "qpe")]
            if g == S.ATT and getattr(S, "q_collects", None):
                S.pipe.raw(lambda: S.ls[S.ATT].wait_for(S.q_collects[-1].result.nodes[-1]),
                           f"q_first_g{g}")
                continue
            S.pipe.raw(lambda put=put, g=g: [S.ls[g].wait_for(s.out().port.ends[-1])
                                             for s in put], f"q_first_g{g}")
    hook, S.after_q8 = getattr(S, "after_q8", None), None
    if hook is not None:
        hook()


def _q8_plain(S):
    row, view, HPG, ATT, ca = S.spec_row, S.view, S.HPG, S.ATT, S.c(S.ATT)
    qt_rows = {g: view(f"qt_rows_g{g}", g, S.qt16[g].out(), row(HPG * KVR, g),
                       row(KVR, g, rows=HPG)).out() for g in range(S.G)}
    qt_all = S.allgather("qt", qt_rows, lambda g: row(KVR, g, rows=HPG), 2 * HPG * KVR, [ATT],
                         lambda g: PortSpec(RM, F16, (HEADS, KVR), mem_level=L1,
                                            cluster=ca))[ATT]
    qpe_rows = {g: view(f"qpe_rows_g{g}", g, S.rot[g].out(),
                        row(RP, g, rows=S.rot_rows[g]), row(RP, g, rows=HPG)).out()
                for g in range(S.G)}
    qpe_all = S.allgather("qpe", qpe_rows, lambda g: row(RP, g, rows=HPG), 2 * HPG * RP, [ATT],
                          lambda g: PortSpec(RM, F16, (HEADS, RP), mem_level=L1,
                                             cluster=ca))[ATT]
    S.q8 = S.add(QAssemble(inv_qt_f32bits=S.inv["qt"], inv_qpe_f32bits=S.inv["qpe"],
                           cluster=ca), "q8", ATT, qt=qt_all.out(), qpe=qpe_all.out())


def _q8_from_att(S, parts):
    """Q8 assembled on ATT as the one-cluster attention does (_q8_at_att), stashed once into
    ATT's chip's L3 and read from there by every other cluster in `parts` -- ATT's chip
    locally, a chip in its column over the north-south link no weight push uses: {g: Q8 port}."""
    ATT, ka, nb = S.ATT, S.chip(S.ATT), 32 * KV
    _q8_at_att(S)
    qspec = lambda g: PortSpec(A, I8, (32, KV), mem_level=L1, cluster=S.c(g))
    raw = lambda g: PortSpec(RM, I8, (1, nb), mem_level=L1, cluster=S.c(g))
    sym = S.xsym("dsv2_xchg_q8", nb)
    st = S.add(Stash(src=qspec(ATT), nbytes=nb, dst=BingoMemSymbol(sym, chip_id=ka)),
               "xput_q8", ATT, x=S.q8.out())
    run_ = S.add(Join(parts=[_l3(qspec(ATT))], dst=PortSpec(RM, I8, (1, nb), mem_level=L3)),
                 "xrun_q8", at_chip=ka, x0=st.out())
    out = {ATT: S.q8.out()}
    for g in parts:
        if g == ATT:
            continue
        cp = S.add(Collect(parts=[PortSpec(RM, I8, (1, nb), mem_level=L3)], nbytes=[nb],
                           dst=raw(g), cluster=S.c(g)), f"xget_q8_g{g}", g, x0=run_.out())
        out[g] = S.view(f"q8_g{g}", g, cp.out(), raw(g), qspec(g)).out()
    return out


def _q8_everywhere(S):
    """Every cluster gathers all heads' q~ and q_pe and assembles its own Q8: {g: stage}."""
    row, view, HPG, G = S.spec_row, S.view, S.HPG, S.G
    l1 = lambda shape, g: PortSpec(RM, F16, shape, mem_level=L1, cluster=S.c(g))
    everyone = list(range(G))
    qt_rows = {g: view(f"qt_rows_g{g}", g, S.qt16[g].out(), row(HPG * KVR, g),
                       row(KVR, g, rows=HPG)).out() for g in everyone}
    qt_all = S.allgather("qt", qt_rows, lambda g: row(KVR, g, rows=HPG), 2 * HPG * KVR,
                         everyone, lambda g: l1((HEADS, KVR), g))
    qpe_rows = {g: view(f"qpe_rows_g{g}", g, S.rot[g].out(),
                        row(RP, g, rows=S.rot_rows[g]), row(RP, g, rows=HPG)).out()
                for g in everyone}
    qpe_all = S.allgather("qpe", qpe_rows, lambda g: row(RP, g, rows=HPG), 2 * HPG * RP,
                          everyone, lambda g: l1((HEADS, RP), g))
    return {g: S.add(QAssemble(inv_qt_f32bits=S.inv["qt"], inv_qpe_f32bits=S.inv["qpe"],
                               cluster=S.c(g)), f"q8_g{g}", g, qt=qt_all[g].out(),
                     qpe=qpe_all[g].out()) for g in everyone}


def _anchored(S, name, g, spec, src):
    """A static cache copy's port that orders its loads after cluster g's own q~: with no
    producer they would open the DM stream, ahead of the whole layer; after q~ they still load
    long before QK, which waits for the gathered Q8."""
    return S.add(After(x=spec, after=S.spec_row(S.HPG * KVR, g)), f"att_{name}_g{g}", g,
                 x=src, after=S.qt16[g].out()).out()


def _build_split_dim(S, cap):
    """params att_split_mode dim: QK and the softmax split by key, PV by latent
    (dg.att_split_plan mode dim; the golden is dg.split_attention_golden).

    Every cluster assembles its own Q8 and runs pass 1 over its shard of key tiles (no V);
    the row maxima are all-gathered and each cluster takes m*; pass 2 writes the shard's P
    (seeded at m*) and its row sums, and both are all-gathered in slot order -- the shards'
    P back to back are the whole P^T. Each cluster then computes its d_v / G rows of O^T over
    every key in one matmul, out of a V^T slice of its own chip's value copy (the rows this
    token appended read from the attention's chip's), adds the row sums as a tree and
    normalises its slice; the attention's cluster gathers the slices and transposes."""
    data, h, G = S.data, S.h, S.G
    ATT, ca = S.ATT, S.c(S.ATT)
    keys, L = data["keys"], data["L"]
    l1 = lambda shape, g, dt=F16: PortSpec(RM, dt, shape, mem_level=L1, cluster=S.c(g))
    # params att_split_chips: the chip indices (platform order) taking part; default all. A
    # subset keeps the exchanges off the links the weight pushes use (ATT's column: the
    # north-south link only).
    sub = S.p.get("att_split_chips")
    plan = dg.att_split_plan(len(S.chips), S.NC, ATT, keys // dg.BC, "dim", sub)
    G = plan["G"]
    if not isinstance(h["val"], BingoMemSymbol):
        raise ValueError("att_split_mode dim reads the appended V^T rows from the attention's "
                         "chip's copy by symbol; the value copy is not one here")
    S.att_plan = plan
    everyone = sorted(plan["slots"])
    corder = [S.chips[i] for i in plan["chip_order"]]
    ka, last = S.chip(ATT), plan["slots"][-1]
    if sub is None:
        st8 = _q8_everywhere(S)
        S.q8 = st8[ATT]
        q8 = {g: st.out() for g, st in st8.items()}
    else:
        q8 = _q8_from_att(S, everyone)          # S.q8: ATT's
    kspec, vspec = PortSpec(A, I8, (cap, KV), mem_level=L3), PortSpec(A, I8, (KVR, cap),
                                                                       mem_level=L3)
    key0, val0 = S.staged(kspec, h["key"]), S.staged(vspec, h["val"])
    app_v = S.app.out("val")
    # the attention's chip's value copy, by name from any chip, read after the append
    valf_h = S.staged(vspec, BingoMemSymbol(h["val"].symbol_name, h["val"].offset, chip_id=ka))
    common = dict(cap=cap, k_s=S.ks["s"], k_o=data["k_o"], a_exp=data["a_exp"],
                  a_n_f32bits=data["a_n"], bc=dg.BC)
    s1 = {}
    for g in everyone:
        t0, nt = plan["shards"][g]
        # only the last slot's tiles hold the appended row; it sits on the attention's chip
        key = S.app.out("key") if g == last else _anchored(S, "key", g, kspec, key0)
        # (params att_fresh_tile: only the appended tile of ATT's shard waits for the append)
        ft = (keys - 1) // dg.BC if S.p.get("att_fresh_tile", False) else -1
        s1[g] = S.add(MlaShardScores(tile0=t0, nt=nt, cluster=S.c(g), with_v=False,
                                     fresh_tile=ft, **common),
                      f"att1_g{g}", g, q8=q8[g], key=key)
    m_all = S.allgather("attm", {g: s1[g].out("m") for g in everyone},
                        lambda g: l1((1, 32), g), 64, everyone, lambda g: l1((G, 32), g),
                        chips=corder)
    s2 = {}
    for g in everyone:
        t0, nt = plan["shards"][g]
        mx = S.add(MlaRowMax(rows=G, cluster=S.c(g)), f"attmax_g{g}", g, x=m_all[g].out())
        s2[g] = S.add(MlaShardP(tile0=t0, nt=nt, cluster=S.c(g), **common), f"att2_g{g}", g,
                      m=mx.out("m"), s16=s1[g].out("s16"))
    pb = plan["nt"] * dg.BC * 32
    p_all = S.allgather("attp", {g: s2[g].out("p") for g in everyone},
                        lambda g: l1((1, pb), g, I8), pb, everyone,
                        lambda g: l1((1, keys * 32), g, I8), chips=corder)
    l_all = S.allgather("attl", {g: s2[g].out("l") for g in everyone},
                        lambda g: l1((1, 32), g), 64, everyone, lambda g: l1((G, 32), g),
                        chips=corder)
    d_s = KVR // G
    ot = {}
    for g in everyone:
        if S.chip(g) == ka:
            vb, k_static = dict(val=app_v), keys
        else:
            vb = dict(val=_anchored(S, "val", g, vspec, val0),
                      valf=S.add(After(x=vspec, after=vspec), f"att_valf_g{g}", g, x=valf_h,
                                 after=app_v).out())
            k_static = L // 4 * 4
        ot[g] = S.add(MlaDimPV(slot=plan["slot"][g], G=G, keys=keys, cap=cap, k_o=data["k_o"],
                               a_n_f32bits=data["a_n"], k_static=k_static, cluster=S.c(g)),
                      f"attpv_g{g}", g, p=p_all[g].out(), l=l_all[g].out(), **vb)
    o_all = S.allgather("atto", {g: ot[g].out("ot") for g in everyone},
                        lambda g: l1((d_s, 32), g), d_s * 64, [ATT],
                        lambda g: l1((KVR, 32), g), chips=corder)
    S.att = S.add(MlaDimOut(tile0=0, nt=1, cluster=ca, **common), "attn", ATT,
                  ot=o_all[ATT].out())


def _build_split(S, cap):
    """params att_split: the attention over every cluster, one shard of key tiles each
    (dg.att_split_plan; the golden is dg.split_attention_golden).

    Every cluster gathers all heads' q~ and q_pe and assembles its own Q8; runs pass 1 over its
    shard (its own chip's cache copy -- only the attention's cluster's shard holds the appended
    row, and only its chip's copy got it); the shards' row maxima are all-gathered and each
    cluster takes their max m*; pass 2 runs the shard's softmax seeded at m* and its PV. The
    FP16 partials are added on each chip's lead (L1 to L1), the other chips' sums are read by
    the attention's cluster, which adds them in the plan's order and normalises."""
    data, h, G = S.data, S.h, S.G
    ATT, ca = S.ATT, S.c(S.ATT)
    l1 = lambda shape, g, dt=F16: PortSpec(RM, dt, shape, mem_level=L1, cluster=S.c(g))
    plan = dg.att_split_plan(len(S.chips), S.NC, ATT, data["keys"] // dg.BC)
    if plan["G"] != G:
        raise ValueError(f"att_split: the plan has {plan['G']} clusters, the platform {G}")
    S.att_plan = plan
    everyone = list(range(G))
    q8 = _q8_everywhere(S)
    S.q8 = q8[ATT]
    key0 = S.staged(PortSpec(A, I8, (cap, KV), mem_level=L3), h["key"])
    val0 = S.staged(PortSpec(A, I8, (KVR, cap), mem_level=L3), h["val"])
    common = dict(cap=cap, k_s=S.ks["s"], k_o=data["k_o"], a_exp=data["a_exp"],
                  a_n_f32bits=data["a_n"], bc=dg.BC)
    s1, s2 = {}, {}
    for g in everyone:
        t0, nt = plan["shards"][g]
        if g == ATT:
            kv = dict(key=S.app.out("key"), val=S.app.out("val"))
        else:
            # the static copy has no producer: without an anchor its loads would open the
            # DM stream, ahead of the whole layer. After its own q~ they still load long
            # before QK, which waits for the gathered Q8.
            kv = {n: _anchored(S, n, g, sp, src)
                  for n, sp, src in (("key", PortSpec(A, I8, (cap, KV), mem_level=L3), key0),
                                     ("val", PortSpec(A, I8, (KVR, cap), mem_level=L3), val0))}
        s1[g] = S.add(MlaShardScores(tile0=t0, nt=nt, cluster=S.c(g), **common),
                      f"att1_g{g}", g, q8=q8[g].out(), **kv)
    m_all = S.allgather("attm", {g: s1[g].out("m") for g in everyone},
                        lambda g: l1((1, 32), g), 64, everyone, lambda g: l1((G, 32), g))
    for g in everyone:
        t0, nt = plan["shards"][g]
        mx = S.add(MlaRowMax(rows=G, cluster=S.c(g)), f"attmax_g{g}", g, x=m_all[g].out())
        s2[g] = S.add(MlaShardPV(tile0=t0, nt=nt, cluster=S.c(g), **common), f"att2_g{g}", g,
                      m=mx.out("m"), s16=s1[g].out("s16"), v=s1[g].out("v"))
    n_o = 512 * 32
    acc = {}
    for k, mem in enumerate(plan["chips"]):
        lead = mem[0]
        o, l = s2[lead].out("o"), s2[lead].out("l")
        for q in mem[1:]:
            po = S.add(Pull(rows=1, cols=n_o, layout=RM, dtype=F16, src=S.c(q), dst=S.c(lead)),
                       f"attpo_g{q}", lead, x0=s2[q].out("o"))
            pl = S.add(Pull(rows=1, cols=32, layout=RM, dtype=F16, src=S.c(q), dst=S.c(lead)),
                       f"attpl_g{q}", lead, x0=s2[q].out("l"))
            o = S.add(AddRow(cols=n_o, cluster=S.c(lead)), f"attao_g{q}", lead, a=o,
                      b=po.out()).out()
            l = S.add(AddRow(cols=32, cluster=S.c(lead)), f"attal_g{q}", lead, a=l,
                      b=pl.out()).out()
        acc[k] = (lead, o, l)
    order = plan["order"]
    _, o, l = acc[order[0]]
    for i, k in enumerate(order[1:]):
        lead, ok, lk = acc[k]
        ro = S.xfer(f"attxo_k{k}", lead, [ok], l1((1, n_o), lead), 2 * n_o, ATT,
                    l1((1, n_o), ATT)).out()
        rl = S.xfer(f"attxl_k{k}", lead, [lk], l1((1, 32), lead), 64, ATT,
                    l1((1, 32), ATT)).out()
        if i < len(order) - 2:
            o = S.add(AddRow(cols=n_o, cluster=ca), f"attxao_k{k}", ATT, a=o, b=ro).out()
            l = S.add(AddRow(cols=32, cluster=ca), f"attxal_k{k}", ATT, a=l, b=rl).out()
        else:
            S.att = S.add(MlaShardOut(tile0=0, nt=1, cluster=ca, **common), "attn", ATT,
                          o_acc=o, o_last=ro, l_acc=l, l_last=rl)


def checks(S):
    PREV.checks(S)
    g = S.data["gold"]
    S.check("q8", S.q8, "q8", S.gold("q8", g["q8"]), 32 * KV, S.ATT)
    S.check("ot", S.att, "ot", S.gold("ot", g["ot"]), 2 * KVR * 32, S.ATT)
    S.check("xt", S.att, "xt", S.gold("xt", g["xt"]), 2 * KVR * 32, S.ATT)


if __name__ == "__main__":
    run(sys.modules[__name__])
