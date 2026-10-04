# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""Data of the dsv2 workloads: the DeepSeek-V2-Lite layer-1 golden, cut to their mapping.

ONE GOLDEN FOR EVERY LEVEL. The weights, the token, the cache and every stage's expected
output come from snax_cluster's golden package (target/snitch_cluster/sw/apps/dsv2/util),
the same one the snax-dsv2-* apps check against on one cluster: seeded random BF16
weights at the real shapes, quantised INT8 per output column with the RMSNorm gains folded
in, static activation scales, and a hardware-exact model of every stage (hwmodel.py). So
every check here is BIT-EXACT; a tolerance would only hide a wrong byte.

WHERE EACH PIECE GOES.
    HBM (memory chiplet)   every weight, in the array's B layout, as the device streams
                           it, and the routed experts' dequant factors (their addresses
                           go into the expert table, so they must be numbers the
                           compiler knows -- the HBM's are, the host image's are not)
    L3 (host image)        the token x, the other dequant factors, the RoPE tables, the
                           latent cache (both copies, L tokens), the expert table, and
                           every golden
"""

import dataclasses
import glob
import os
import sys

import numpy as np

# The layer's shapes, checked against the golden's own config below.
D_MODEL = 2048
HEADS = 16
Q_HEAD = 192          # 128 without position + 64 with RoPE, per head
Q_NOPE = 128
ROPE = 64
KV_RANK = 512
KV = 576              # [c 512 | k_pe 64]
V_HEAD = 128
N_EXP = 64
TOP_K = 6
I_EXP = 1408
I_SH = 2816
NCL = 4
HPC = HEADS // NCL    # heads per cluster
BC = 64               # keys per attention tile


def import_dsv2(dsv2_dir):
    """snax_cluster's golden package, imported as `util` from `dsv2_dir`.

    The package is named `util` in its repository, which is a name other trees use too, so
    this checks it got THAT one rather than trusting the path order.
    """
    dsv2_dir = os.path.abspath(dsv2_dir)
    if dsv2_dir not in sys.path:
        sys.path.insert(0, dsv2_dir)
    import util
    from util import fp, golden, hwmodel, layout, pack
    where = os.path.dirname(os.path.abspath(util.__file__))
    if where != os.path.join(dsv2_dir, "util"):
        raise ImportError(f"imported `util` from {where}, not from {dsv2_dir}/util: another "
                          f"package of that name shadows the dsv2 golden")
    return golden, hwmodel, layout, fp, pack


def slot_cluster(s):
    """The cluster that runs routed slot s: two slots on each of clusters 1..3 (cluster 0
    runs the shared experts, the same 16.5 MiB of weights as two slots)."""
    return 1 + s // 2


def f16(a):
    return np.ascontiguousarray(np.asarray(a, dtype=np.float16))


def att_split_plan(n_chips, nc, att, n_tiles, mode="key", chips=None):
    """The split attention (params att_split), as the golden and the device graph both build it.

    Every cluster attends over its own shard of key tiles for all 16 heads.

    mode "key" (the default): PV split by key too. The attention's cluster takes the LAST
    shard: it holds the row this token appends, which only its chip's cache copy receives.
    Each chip's clusters add their partials on the chip's lead (the attention's cluster on its
    chip, else the first), then the attention's cluster adds the other chips' in chip order.
    shards[g] = (first tile, tiles); chips[k] = [lead, the others]; order = chips, in the order
    the attention's cluster adds them (its own first).

    mode "dim" (params att_split_mode: dim): QK and the softmax split by key, PV by latent:
    every cluster gets every shard's P and computes its own d_v / G rows of O^T over ALL the
    keys, so nothing is added in FP16 but the row sums. One cluster order serves every
    gathered array: slots = the clusters chip by chip, the attention's chip LAST (an
    all-gather over chips in that order lays the parts out in it). Slot i holds key tiles
    [i nt, (i+1) nt) and O^T rows [i d_v/G, (i+1) d_v/G); the last slot, on the attention's
    chip, holds the appended row. The row sums add as a halving tree over the slots.

    `chips` (mode dim only, params att_split_chips): the chip indices (the platform's chiplet
    order) that take part, the attention's among them; default every chip. g keeps its global
    numbering; G is then the clusters taking part."""
    ka = att // nc
    part = list(range(n_chips)) if chips is None else [int(k) for k in chips]
    if chips is not None and (mode != "dim" or ka not in part or len(set(part)) != len(part)
                              or not all(0 <= k < n_chips for k in part)):
        raise ValueError(f"att_split_chips={chips}: mode dim, distinct chip indices < {n_chips}, "
                         f"the attention's chip {ka} among them")
    G = len(part) * nc
    if n_tiles % G:
        raise ValueError(f"att_split: {n_tiles} key tiles do not split over {G} clusters")
    nt = n_tiles // G
    if mode == "dim":
        corder = [k for k in part if k != ka] + [ka]
        slots = [k * nc + c for k in corder for c in range(nc)]
        if G & (G - 1):
            raise ValueError(f"att_split_mode dim: the row-sum tree halves {G} slots")
        return dict(G=G, nt=nt, mode="dim", slots=slots, chip_order=corder, att=att,
                    shards={g: (i * nt, nt) for i, g in enumerate(slots)},
                    slot={g: i for i, g in enumerate(slots)})
    if mode != "key":
        raise ValueError(f"att_split_mode={mode!r}: key or dim")
    cl = [g for g in range(G) if g != att] + [att]
    shards = {g: (s * nt, nt) for s, g in enumerate(cl)}
    chips = []
    for k in range(n_chips):
        mem = [k * nc + c for c in range(nc)]
        lead = att if att in mem else mem[0]
        chips.append([lead] + [g for g in mem if g != lead])
    ka = att // nc
    return dict(G=G, nt=nt, shards=shards, chips=chips, mode="key",
                order=[ka] + [k for k in range(n_chips) if k != ka], att=att)


def split_attention_golden(g, hm, p):
    """g.hw with the attention split over the clusters by key (att_split, one token).

    Pass 1: each shard's row maximum, as the softmax kernel leaves it after the shard's tiles;
    the global maximum m* is their lanewise max (the same kernel run on the gathered rows).
    Pass 2: each shard's softmax SEEDED at m* (seed_state 0), so every shard quantises P on one
    scale and its PV needs no correction; its last tile leaves O16_c through the D port. The
    partials and their row sums are added in FP16 in the plan's order, then the normalise, W_UV,
    W_O and the MoE run exactly as hwmodel.run_tokens does them."""
    F16, F32, F64 = hm.F16, hm.F32, hm.F64
    simd = hm.simd
    H = dict(g.hw)
    Q8, Kc, Vc = H["Q8"], H["Kc"], H["Vc"]
    a32, k_s, k_o, a_n, bc = F32(H["a_exp"]), H["ks"]["s"], H["k_o"], H["a_n"], g.bc
    T, Br = Kc.shape[0], Q8.shape[0]
    if T % bc:
        raise ValueError(f"att_split: {T} keys are not whole tiles of {bc}")
    plan = att_split_plan(int(p["num_chiplets"]), int(p["num_clusters"]),
                          int(p.get("att_cluster", 1)), T // bc, p.get("att_split_mode", "key"),
                          p.get("att_split_chips"))
    lane_max = lambda a: simd.reduce_lanewise(a, simd.MAX)

    def tile(j):
        K = Kc[j * bc:(j + 1) * bc].astype(np.int64)
        V = Vc[j * bc:(j + 1) * bc].astype(np.int64)
        return hm.d_port(K @ Q8.astype(np.int64).T, k_s), V

    seed = np.full(Br, -65504.0, dtype=F16)
    m_c = {}
    for c, (t0, nt) in plan["shards"].items():
        m16 = seed
        for j in range(t0, t0 + nt):
            m16 = lane_max(np.stack([lane_max(tile(j)[0]), m16]))
        m_c[c] = m16
    mstar = lane_max(np.stack([lane_max(np.stack(list(m_c.values()))), seed]))

    dim = plan["mode"] == "dim"
    O16, lc, oc, o_all = {}, {}, {}, 0
    for c, (t0, nt) in plan["shards"].items():
        m16, l16, o = mstar, np.zeros(Br, dtype=F16), None
        for j in range(t0, t0 + nt):
            S16, V = tile(j)
            mnew = lane_max(np.stack([lane_max(S16), m16]))
            corr16 = hm.exp16(hm.add16(m16, -mnew), a32)
            p16 = hm.exp16(hm.add16(S16, -mnew[None, :]), a32)
            rsum16 = simd.reduce_lanewise(p16, simd.ADD)
            l16 = simd.reduce_lanewise(np.stack([rsum16, hm.mul16(corr16, l16)]), simd.ADD)
            pv = V.T @ hm.quant_i8(p16, hm.P8_SCALE).astype(np.int64)
            if dim:
                # one INT32 matmul over every key: no column scale (corr is 1 at m*)
                o_all = o_all + pv
            else:
                o = pv if o is None else np.clip(
                    np.rint(o.astype(F64) * corr16.astype(F64)[None, :]),
                    -2**31, 2**31 - 1).astype(np.int64) + pv
            m16 = mnew
        lc[c] = l16
        if not dim:
            O16[c], oc[c] = hm.d_port(o, k_o), o

    if dim:
        # O^T leaves the D port once, at k_o; the row sums add as the device's halving tree
        Ot, oc = hm.d_port(o_all, k_o), {0: o_all}
        ls = [lc[g] for g in plan["slots"]]
        while len(ls) > 1:
            h_ = len(ls) // 2
            ls = [hm.add16(ls[i], ls[i + h_]) for i in range(h_)]
        lt = ls[0]
    else:
        def fold(members):
            acc_o, acc_l = O16[members[0]], lc[members[0]]
            for c in members[1:]:
                acc_o, acc_l = hm.add16(acc_o, O16[c]), hm.add16(acc_l, lc[c])
            return acc_o, acc_l
        parts = [fold(plan["chips"][k]) for k in plan["order"]]
        Ot, lt = parts[0]
        for o_, l_ in parts[1:]:
            Ot, lt = hm.add16(Ot, o_), hm.add16(lt, l_)
    t16 = simd.stream_map(lt, a_n, 0.0, simd.RSQRT)
    rsc16 = hm.mul16(t16, t16)
    ot = hm.mul16(Ot, rsc16[None, :]).T
    heads = g.pack.W.d.heads
    # a gross error would not survive this: the split is the same attention, re-rounded
    ref = np.asarray(H["ot16"], dtype=np.float64)
    err = np.max(np.abs(ot[:heads].astype(np.float64) - ref)) / max(np.max(np.abs(ref)), 1e-6)
    if err > 0.05:
        raise AssertionError(f"att_split golden drifts {err:.3f} from the sequential attention")
    att = dict(o=sum(oc.values()), m=mstar, l=lt, tiles=[],
               split=dict(plan=plan, m_c=m_c, O16=O16, l=lc, mstar=mstar))
    tk = dict(H)
    tk.update(hm.mla_output(g.pack, g.scales, tk["x16"], ot[:heads]))
    tk.update(hm.moe_route(g.pack, g.scales, tk["h16"]))
    order = hm.union_order([tk["ids"]])
    tk.update(hm.moe_experts(g.pack, g.scales, tk["h16"], tk,
                             [e for e in order if e in list(tk["ids"])]))
    pair = dict(Q8=Q8, att=att, O16=Ot, rsc16=rsc16, masked=[])
    tk.update(pair, pairs=[pair], order=order)
    return tk


def _golden_cached(golden, dsv2_dir, seed, L, wbits):
    """export_only: the golden of (seed, L, wbits) from a pickle cache (BINGO_GOLDEN_CACHE, default
    ~/.cache/hemaia/golden), keyed also by the golden's own sources -- it is slow to build and
    the same for every scheduling variant a DSE exports."""
    import hashlib
    import pickle
    src = hashlib.sha1()
    for f in sorted(glob.glob(os.path.join(dsv2_dir, "util", "*.py"))):
        with open(f, "rb") as fh:
            src.update(fh.read())
    root = os.environ.get("BINGO_GOLDEN_CACHE", os.path.expanduser("~/.cache/hemaia/golden"))
    path = os.path.join(root, f"s{seed}_L{L}_w{wbits}_bc{BC}_{src.hexdigest()[:12]}.pkl")
    try:
        with open(path, "rb") as fh:
            return pickle.load(fh)
    except (OSError, EOFError, pickle.UnpicklingError):
        pass
    g = golden.make(seed=seed, L=L, bc=BC, wbits=wbits) if wbits != 8 else \
        golden.make(seed=seed, L=L, bc=BC)
    os.makedirs(root, exist_ok=True)
    tmp = f"{path}.{os.getpid()}"
    with open(tmp, "wb") as fh:
        pickle.dump(g, fh, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)
    return g


def generate(p, dsv2_dir):
    golden, hwmodel, layout, fp, pack = import_dsv2(dsv2_dir)
    seed, L = int(p.get("seed", 1)), int(p.get("L", 511))
    # params wbits: 8, or 4 -- every weight but the router's INT4 (snax util/pack.py), its
    # blob in the paired nibble-packed layout the GEMV's w4 reads (Layout.B_W4).
    wbits = int(p.get("wbits", 8))
    if wbits not in (8, 4):
        raise ValueError(f"params wbits={wbits}: 8 or 4")
    if p.get("export_only", False):
        g = _golden_cached(golden, dsv2_dir, seed, L, wbits)
    else:
        g = golden.make(seed=seed, L=L, bc=BC, wbits=wbits) if wbits != 8 else \
            golden.make(seed=seed, L=L, bc=BC)
    # params att_split: the attention split over every cluster by key (att_split_plan); its
    # golden differs from the one-cluster attention's by rounding, and so does all after it
    if p.get("att_split", False):
        if int(p.get("tokens", 1)) != 1:
            raise ValueError("att_split: one token per pass")
        g = dataclasses.replace(g, hw=split_attention_golden(g, hwmodel, p))
    d, H, S, P = g.dims, g.hw, g.scales, g.pack
    if (d.hidden, d.heads, d.q_head, d.kv_rank + d.q_rope, d.n_routed, d.top_k,
            d.moe_inter, d.shared_inter) != (D_MODEL, HEADS, Q_HEAD, KV, N_EXP, TOP_K, I_EXP,
                                            I_SH):
        raise ValueError("the golden's config is not DeepSeek-V2-Lite's layer shape")
    keys = L + 1
    # A pass of several tokens (params tokens) attends to L + tokens keys (_pass); one token
    # to L + 1, which must be whole tiles: the attention block masks no partial last tile.
    multi = int(p.get("tokens", 1)) > 1
    if keys % BC and not multi:
        raise ValueError(f"L={L}: {keys} keys are not whole tiles of {BC}; the attention "
                         f"block does not mask a partial last tile yet")
    cap = int(p.get("cap", keys if not keys % BC else keys + (-keys % BC)))
    if cap < keys or cap % BC:
        raise ValueError(f"cap={cap}: at least L+1={keys} and a multiple of {BC}")
    mesh = layout.MESH
    ks = {k: int(v) for k, v in H["ks"].items()}
    J = H                              # run() merges the joint values into the token's dict
    att = J["att"]
    ids = [int(e) for e in H["ids"]]
    f32 = fp.f32bits

    # ---- the weights, exactly as the device streams them -------------------------------
    blobs = {"wdkv": P.wdkv.blob(mesh), "wq": P.wq.blob(mesh), "wuk": P.wuk.blob(mesh),
             "wuv": P.wuv.blob(mesh), "wo": P.wo.blob(mesh), "wr": P.wr.blob(mesh),
             "sh_gu": P.shared_gu.blob(mesh), "sh_dn": P.shared_down.blob(mesh)}
    for e in ids:
        gu, dn = P.expert(e)
        blobs[f"e{e}_gu"] = gu.blob(mesh)
        blobs[f"e{e}_dn"] = dn.blob(mesh)
    blobs = {k: np.ascontiguousarray(v.astype(np.int8)) for k, v in blobs.items()}

    # ---- factors ------------------------------------------------------------------------
    factors = {"s_kv": f16(H["kv_s"]), "s_q": f16(H["q_s"]),
               "s_uk": f16(H["qt_s"]).reshape(-1), "s_uv": f16(H["oh_s"]).reshape(-1),
               "s_o": f16(H["a_s"]), "s_r": f16(H["lg_s"]),
               "s_sh_gu": f16(H["shared"]["g_s"]), "s_sh_dn": f16(H["shared"]["y_s"])}
    expert_f = {}
    for i, e in enumerate(ids):
        expert_f[e] = (f16(H["slots"][i]["g_s"]), f16(H["slots"][i]["y_s"]),
                       int(f32(S.expert_a[e].inv)))

    # ---- RoPE tables at the token's position: cos repeated per pair, sin signed ----------
    c_rep = np.repeat(f16(H["cos16"]), 2)
    s_sgn = np.empty(ROPE, dtype=np.float16)
    s_sgn[0::2] = -f16(H["sin16"])
    s_sgn[1::2] = f16(H["sin16"])
    rope = {n: (np.tile(c_rep, n), np.tile(s_sgn, n)) for n in (HPC, HPC + 1)}

    # ---- the cache: both copies with the L cached rows, and after the append --------------
    rows0 = np.concatenate([g.c8, g.kpe8], axis=1).astype(np.int8)
    new_row = np.concatenate([H["c8_new"], H["kpe8_new"]]).astype(np.int8)
    rows1 = np.concatenate([rows0, new_row[None]], axis=0)
    key0 = layout.key_copy(rows0, cap, mesh).astype(np.int8)
    val0 = layout.value_copy(rows0[:, :KV_RANK], cap, mesh).astype(np.int8)
    key1 = layout.key_copy(rows1, cap, mesh).astype(np.int8)
    val1 = layout.value_copy(rows1[:, :KV_RANK], cap, mesh).astype(np.int8)
    if not np.array_equal(layout.from_a(key1, cap, KV, mesh)[:keys], J["Kc"]):
        raise AssertionError("the appended key copy is not the golden's key rows")

    # ---- goldens, per stage and cluster ----------------------------------------------------
    O16 = f16(J["O16"])                                       # [512, 32], O^T
    ot = f16(fp.mul16(O16, f16(J["rsc16"])[None, :]))          # [512, 32], o~^T
    if not np.array_equal(ot.T[:HEADS].view(np.uint16), f16(H["ot16"]).view(np.uint16)):
        raise AssertionError("o~ from O16 and c / l is not the golden's ot16")
    q8 = layout.to_a(np.asarray(J["Q8"], dtype=np.int8), mesh).astype(np.int8)
    gold = {
        "xn": f16(H["xn"]), "kv16": f16(H["ckv16"]), "cn": f16(H["cn16"]),
        "c8": np.asarray(H["c8_new"], dtype=np.int8),
        "kpe8": np.asarray(H["kpe8_new"], dtype=np.int8),
        "q8": q8, "m": f16(att["m"]), "l": f16(att["l"]), "o16": O16, "ot": ot,
        "xt": np.ascontiguousarray(ot.T), "key_t": key1, "val": val1,
        "hn": f16(H["hn"]), "lg": f16(H["logits16"]), "p": f16(H["p16"]),
        "sh_g": f16(H["shared"]["g16"]),
        "sh_a": layout.gemv_a(np.asarray(H["shared"]["a8"], dtype=np.int8), mesh).astype(np.int8),
        "ys": f16(H["shared"]["y16"]), "out": f16(H["out16"]),
    }
    for c in range(NCL):
        hs = slice(HPC * c, HPC * (c + 1))
        gold[f"q16_c{c}"] = f16(H["q16"][HPC * Q_HEAD * c: HPC * Q_HEAD * (c + 1)])
        rot = [f16(H["qpe_rot"][hs])]
        if c == 0:
            rot.append(f16(H["kpe_rot"])[None])
        gold[f"rot_c{c}"] = np.concatenate(rot, axis=0)
        gold[f"qt16_c{c}"] = f16(H["qt16"][hs]).reshape(-1)
        gold[f"oh_c{c}"] = f16(H["o16"][hs]).reshape(-1)
        cs = slice(512 * c, 512 * (c + 1))
        gold[f"attn_c{c}"] = f16(H["attn16"][cs])
        gold[f"h_c{c}"] = f16(H["h16"][cs])
    for s in range(TOP_K):
        sl = H["slots"][s]
        gold[f"g_s{s}"] = f16(sl["g16"])
        gold[f"a_s{s}"] = layout.gemv_a(np.asarray(sl["a8"], dtype=np.int8), mesh).astype(np.int8)
        gold[f"e_s{s}"] = f16(sl["y16"])

    # ---- the device's own arithmetic, re-derived where a slice could be the wrong one -----
    # The GEMV of cluster c's W_Q heads, from the packed weight, must be the golden's q_raw:
    # that proves the byte range cluster c streams is the one it should.
    xq = np.asarray(H["xq"], dtype=np.int64)
    for c in range(NCL):
        cols = slice(HPC * Q_HEAD * c, HPC * Q_HEAD * (c + 1))
        mine = fp.d_port(xq @ P.wq.q[:, cols].astype(np.int64), ks["x"])
        if not np.array_equal(f16(mine).view(np.uint16), f16(H["q_raw"][cols]).view(np.uint16)):
            raise AssertionError(f"cluster {c}: the recomputed W_Q GEMV differs")

    # ---- more tokens per pass (params tokens): what stages 1-3 compute for each -----------
    # Token 0 is the golden token; the others are fresh draws, as snax util/golden.py
    # spec_passes draws a speculative pass's tokens, at positions L + t. Up to the
    # projections every token is its own computation -- the same norm, quantiser, weights and
    # factors -- so hwmodel.mla_token gives each one exactly (the attention, which mixes them,
    # is what run_tokens limits to two).
    T = int(p.get("tokens", 1))
    if T < 1:
        raise ValueError(f"params tokens={T}: at least one")
    rng = np.random.default_rng([seed, 17])
    toks, raw = [], []
    for t in range(T):
        x_t = g.x16 if t == 0 else golden._draw_states(rng, 1, d.hidden)[0]
        tk = H if t == 0 else hwmodel.mla_token(P, S, x_t, L + t)
        raw.append(tk)
        # RoPE tables at the token's position: cos repeated per pair, sin signed
        cr = np.repeat(f16(tk["cos16"]), 2)
        sg = np.empty(ROPE, dtype=np.float16)
        sg[0::2], sg[1::2] = -f16(tk["sin16"]), f16(tk["sin16"])
        toks.append({"x16": f16(tk["x16"]), "xn": f16(tk["xn"]),
                     "x8": np.asarray(tk["xq"], dtype=np.int8), "q16": f16(tk["q16"]),
                     "kv16": f16(tk["ckv16"]), "cn": f16(tk["cn16"]),
                     "c8": np.asarray(tk["c8_new"], dtype=np.int8),
                     "kpe8": np.asarray(tk["kpe8_new"], dtype=np.int8),
                     "qpe_rot": f16(tk["qpe_rot"]), "kpe_rot": f16(tk["kpe_rot"]),
                     "qt16": f16(tk["qt16"]), "cos": cr, "sin": sg})

    # ---- the whole layer for the pass (stages 5-8), when its keys are whole tiles ---------
    pas = None
    if T > 1 and (L + T) % BC == 0:
        pas = _pass(g, hwmodel, layout, fp, raw, L, T, mesh)
        for e in pas["order"]:
            if f"e{e}_gu" not in blobs:
                gu, dn = P.expert(e)
                blobs[f"e{e}_gu"] = np.ascontiguousarray(gu.blob(mesh).astype(np.int8))
                blobs[f"e{e}_dn"] = np.ascontiguousarray(dn.blob(mesh).astype(np.int8))
            if e not in expert_f:
                expert_f[e] = pas["expert_f"][e]

    report = [
        f"DeepSeek-V2-Lite layer 1, seed {seed}: one token at position {L} ({keys} keys, "
        f"{keys // BC} tiles of {BC}; cache capacity {cap}); golden draw {g.tries}",
        f"D-port shifts {ks}, O's k_o = {J['k_o']}; a' = {J['a_exp']:.6g}",
        f"top-{TOP_K}: {ids}, weights {[float(w) for w in f16(H['w16'])]}",
        f"weights {sum(b.nbytes for b in blobs.values()) / 2**20:.2f} MiB in the HBM",
    ]
    return dict(seed=seed, L=L, wbits=wbits, keys=keys, cap=cap, ks=ks, k_o=int(J["k_o"]),
                a_exp=float(J["a_exp"]), a_n=int(f32(J["a_n"])), ids=ids,
                w16=f16(H["w16"]), inv={
                    "x": int(f32(S.x.inv)), "qn": int(f32(S.qn.inv)), "c": int(f32(S.c.inv)),
                    "kpe": int(f32(S.kpe.inv)), "qt": int(f32(S.qt.inv)),
                    "qpe": int(f32(S.qpe.inv)), "ot": int(f32(S.ot.inv)),
                    "o": int(f32(S.o.inv)), "h": int(f32(S.h.inv)),
                    "sh_a": int(f32(S.shared_a.inv))},
                x16=f16(H["x16"]), toks=toks, blobs=blobs, factors=factors, expert_f=expert_f,
                rope=rope, key0=key0, val0=val0, gold=gold, report=report, pass_=pas,
                # the whole one-token hardware model, for a mapping that cuts its own slices
                hw=H)


def _pass(g, hwmodel, layout, fp, raw, L, T, mesh):
    """Layer 1 for a speculative pass of T tokens at positions L .. L + T - 1 (hwmodel.
    run_tokens, which stops at two, for any even T): every token appends its row, and the
    attention runs as T / 2 GROUPS of two tokens -- 32 query lanes each, token 2 i + u in
    lanes 16 u .. 16 u + 15 -- over the same L + T keys, token t masked from keys L + u,
    u > t. The MoE runs the UNION of the tokens' experts; token t's combine adds every union
    slot with its own weight, 0 for an expert it did not pick -- what the device computes --
    and that sum is checked against moe_experts' own (it only differs by the sign of a zero)."""
    d, S, P = g.dims, g.scales, g.pack
    F32 = hwmodel.F32
    Hh = d.heads
    ks = hwmodel.shifts(d)
    c8 = np.concatenate([g.c8] + [tk["c8_new"][None] for tk in raw], axis=0)
    kpe8 = np.concatenate([g.kpe8] + [tk["kpe8_new"][None] for tk in raw], axis=0)
    Kc = np.concatenate([c8, kpe8], axis=1)
    keys = L + T
    a_exp = F32(d.softmax_scale * S.qt.s * S.c.s * 2.0 ** ks["s"])
    k_o = fp.d_shift_for_depth(keys)
    a_n = F32(hwmodel.P8_SCALE / (2.0 ** k_o * S.c.s))
    groups, ot_tok = [], [None] * T
    for gi in range(T // 2):
        tt = (2 * gi, 2 * gi + 1)
        Q8 = np.zeros((2 * Hh, d.latent), dtype=np.int8)
        for u, t in enumerate(tt):
            Q8[u * Hh:(u + 1) * Hh, :d.kv_rank] = fp.quant_i8(raw[t]["qt16"], S.qt.inv)
            Q8[u * Hh:(u + 1) * Hh, d.kv_rank:] = fp.quant_i8(raw[t]["qpe_rot"], S.qpe.inv)
        masked = [(L + v, slice(u * Hh, (u + 1) * Hh)) for u, t in enumerate(tt)
                  for v in range(t + 1, T)]
        att = hwmodel.mla_attention(Q8, Kc, c8, a_exp, ks["s"], BC, masked)
        O16 = fp.d_port(att["o"], k_o)
        t16 = hwmodel.simd.stream_map(att["l"], a_n, 0.0, hwmodel.simd.RSQRT)
        rsc16 = fp.mul16(t16, t16)
        ot = f16(fp.mul16(O16, rsc16[None, :]))                        # [512, 32], o~^T
        for u, t in enumerate(tt):
            ot_tok[t] = ot.T[u * Hh:(u + 1) * Hh]
        groups.append(dict(tokens=tt, q8=layout.to_a(Q8, mesh).astype(np.int8), ot=ot,
                           xt=np.ascontiguousarray(ot.T),
                           masks=[(L + v, 16 * u, 16) for u, t in enumerate(tt)
                                  for v in range(t + 1, T)]))
    toks = []
    for t, tk in enumerate(raw):
        r = dict(hwmodel.mla_output(P, S, tk["x16"], ot_tok[t]))
        r.update(hwmodel.moe_route(P, S, r["h16"]))
        toks.append(r)
    order = hwmodel.union_order([r["ids"] for r in toks])
    for r in toks:
        w_of = {int(e): w for e, w in zip(r["ids"], r["w16"])}
        full = dict(hq=r["hq"], ids=np.array(order),
                    w16=f16([w_of.get(e, 0.0) for e in order]))
        m = hwmodel.moe_experts(P, S, r["h16"], full)
        own = hwmodel.moe_experts(P, S, r["h16"], r, [e for e in order if e in list(r["ids"])])
        if not np.array_equal(np.abs(f16(m["out16"])), np.abs(f16(own["out16"]))):
            raise AssertionError("the union combine with zero weights is not the token's own")
        r.update(slots=m["slots"], shared=m["shared"], out16=f16(m["out16"]))
    expert_f = {e: (f16(toks[0]["slots"][i]["g_s"]), f16(toks[0]["slots"][i]["y_s"]),
                    int(fp.f32bits(S.expert_a[e].inv))) for i, e in enumerate(order)}
    rows = np.concatenate([c8, kpe8], axis=1).astype(np.int8)
    cap = keys
    return dict(T=T, L=L, keys=keys, cap=cap, k_o=int(k_o), a_exp=float(a_exp),
                a_n=int(fp.f32bits(a_n)), groups=groups, toks=toks, order=order,
                expert_f=expert_f,
                key0=layout.key_copy(rows[:L], cap, mesh).astype(np.int8),
                val0=layout.value_copy(rows[:L, :d.kv_rank], cap, mesh).astype(np.int8),
                key1=layout.key_copy(rows, cap, mesh).astype(np.int8),
                val1=layout.value_copy(rows[:, :d.kv_rank], cap, mesh).astype(np.int8))


def stage(st, data, l4=(), skip=()):
    """Stage every array; returns the handles (and the expert table's golden record).

    `l4`: weight blobs to place in the memory chiplet's SRAM instead of the HBM -- the
    steady state of a memchip-side HBM -> L4 prefetch of the weights every token uses.
    `skip`: weight blobs the caller stages itself (a per-cluster weight image, say)."""
    from libs.blocks import expert_table, record_bytes
    h = {"x": st.put("dsv2_x", "uint16_t", data["x16"].view(np.uint16))}
    unknown = sorted(set(l4) - set(data["blobs"]))
    if unknown:
        raise ValueError(f"l4_weights {unknown}: not weight blobs ({sorted(data['blobs'])})")
    for k, b in data["blobs"].items():
        if k in skip:
            continue
        h[k] = st.put_l4(f"dsv2_{k}", b) if k in l4 else st.put_hbm(f"dsv2_{k}", b)
    for k, v in data["factors"].items():
        h[k] = st.put(f"dsv2_{k}", "uint16_t", v.view(np.uint16))
    for n, (c, s) in data["rope"].items():
        h[f"cos{n}"] = st.put(f"dsv2_cos{n}", "uint16_t", c.view(np.uint16))
        h[f"sin{n}"] = st.put(f"dsv2_sin{n}", "uint16_t", s.view(np.uint16))
    # the cache: written by the device (the append), so it is a writable array
    h["key"] = st.put("dsv2_key", "int8_t", data["key0"])
    h["val"] = st.put("dsv2_val", "int8_t", data["val0"])
    # the routed experts' factors in the HBM too: the table holds their numeric addresses
    entries = {}
    for e, (gus, dns, inv) in data["expert_f"].items():
        a_gus = st.put_hbm(f"dsv2_e{e}_gu_s", gus.view(np.int8))
        a_dns = st.put_hbm(f"dsv2_e{e}_dn_s", dns.view(np.int8))
        entries[e] = (h[f"e{e}_gu"].address, a_gus.address, h[f"e{e}_dn"].address,
                      a_dns.address, inv)
    table = expert_table(N_EXP, entries)
    h["table"] = st.put("dsv2_expert_table", "int8_t", table)
    h["table_bytes"] = table
    rec = record_bytes(data["ids"], data["w16"].view(np.uint16), table)
    for k, v in list(data["gold"].items()) + [("rec", rec)]:
        v = np.ascontiguousarray(v)
        ctype = "uint16_t" if v.dtype == np.float16 else "int8_t"
        arr = v.view(np.uint16) if v.dtype == np.float16 else v.view(np.int8)
        h[f"gold_{k}"] = st.put(f"dsv2_gold_{k}", ctype, arr.reshape(-1))
    return h
