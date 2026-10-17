#!/usr/bin/env python3
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
"""estimate_prefill.py: the MAC utilisation of the FA prefill (T = 256) ladder, estimated before the
generator exists. NOTHING HERE IS MEASURED ON THE PREFILL: every figure it prints is an estimate, and the
plan page shows measured cells as TBD until a run fills them.

The workload (workloads/fa/common/fa_prefill_mc.py): DeepSeek-V2-Lite layer 1's MLA attention for T = 256
tokens over the 8,192-token latent cache, head parallel (chip k = head k, cluster = token half), in chunks of
C tokens (default 64 = 4 chunks), every chunk sweeping the whole cache in tiles of BC keys with the
cluster's C / 2 rows as C / 32 query tiles of 16 (the SIMD beat). MAC utilisation is the plan page's:
useful MACs / (window x 32 clusters x 1,024 MAC a cycle x clock); the FFN's 82.0 % is this quantity.

The model, per chunk and cluster, is a max over three engines per K/V tile (the GEMM core, the SIMD core,
the DM core's iDMA), plus the chunk's projections, a seam, and the arm's exchanges. Its inputs, set below:

  taken from earlier runs (the value sits beside its name, the run in the comment there)
  CLK_NS, CPP        the cluster clock and the INT8 x INT8 cycles a pass
  SSW_US             a GEMM task's software set-up before the array starts
  FA_US              the array's own fixed cost a task
  BLK_QK/BLK_PV      today's blocking gemm_fa_qk / gemm_fa_pv task time on the decode run
  SM64_US            the FA softmax on a 64-key tile, 16 lanes
  DMA_GBPS           iDMA L3 -> L1
  LINK_GBPS          one D2D push
  ROOT_PULL_US       the reduce with the root loading all partials and folding on the SIMD, 128 KiB a chip, P = 16
  CHAIN_US           the reduce as one xDMA ChainGather, same point
  BARRIER_US         one chip barrier at P = 16
  X_PULL / X_MCAST   the x rows pulled vs multicast for 32 tokens (linear in C)

  assumed (each is a knob of the sensitivity tables)
  PROLOGUE_US        the first chunk's x and W_q landing before the first GEMM
  SEAM_US            a chunk's non-GEMM steps on the critical path (q quantise, Q8 relayout, O requant)
  FILL_US            a barrier arm re-fills its K/V prefetch after each fence
  TAIL_US            the last chunk's normalise and store after its reduce

Run: python3 estimate_prefill.py [--json out.json]. The arms are the plan page's: base, bcast,
bcast_reduce, all (the ladder) and the three leave-one-out arms of all."""

import argparse
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path

HERE = Path(__file__).resolve().parent

# ---- the model: DeepSeek-V2-Lite layer 1, 16 heads, absorbed MLA ------------------------------------
HEADS, D_QK, D_V, KEYS = 16, 576, 512, 8192
Q_HEAD, NOPE, HID = 192, 128, 2048
MAC_TOKEN_HEAD = (HID * Q_HEAD + NOPE * D_V + KEYS * D_QK + KEYS * D_V + D_V * NOPE + NOPE * HID)
CACHE_B = KEYS * (D_QK + D_V)                   # K rows + V columns, INT8: 8.5 MiB
WEIGHT_B = HID * Q_HEAD + NOPE * D_V + D_V * NOPE + NOPE * HID   # one head: W_q, W_UK, W_UV, W_O
CLUSTERS, MAC_CC, BR = 32, 1024, 16

# ---- measured ---------------------------------------------------------------------------------------
CLK_NS = 2.002
PEAK_MAC_US = CLUSTERS * MAC_CC / (CLK_NS * 1e-3)
BLK_QK_US, BLK_PV_US, BLK_QK_PASSES, BLK_PV_PASSES = 2.797, 2.967, 576, 512
SM64_US = 1.654
ROOT_PULL_US, CHAIN_US, BARRIER_US = 441.2, 12.8, 3.3          # E2 at 128 KiB a chip, P = 16; E3
X_PULL_US_PER_TOKEN, X_MCAST_US_PER_TOKEN = 31 / 32, 4 / 32

# ---- assumed ----------------------------------------------------------------------------------------
PROLOGUE_US, SEAM_US, FILL_US, TAIL_US = 35.0, 12.0, 10.0, 8.0
L1_CAP = 990000                                                 # params l1_capacity of the paper cluster
PROJ_TASK_PASSES = 4096                                         # a projection GEMM is cut into tasks of this size


@dataclass(frozen=True)
class Scn:
    """One set of the model's knobs."""
    name: str = "nominal"
    tokens: int = 256
    chunk: int = 64                  # tokens a chunk; each cluster takes chunk / 2 rows
    bc: int = 256                    # keys a K/V tile
    gemm_async: bool = True          # FA GEMMs as start / wait pairs (the next task's set-up under this one)
    cpp: float = 1.0                 # cycles a pass: INT8 x INT8 measured 1.000
    fa_us: float = 0.35              # array fixed cost a task
    ssw_us: float = 2.55             # software set-up a task
    blk_us: float = 1.8              # blocking task overhead: the decode run's 2.797 - 576 passes, 2.967 - 512
    sm_scale: float = 1.0            # the softmax time vs 1.654 us a 64-key tile, linear in keys
    dma_gbps: float = 23.0
    link_gbps: float = 25.0


SCENARIOS = [
    Scn("nominal"),
    Scn("stress", cpp=1.05, fa_us=0.7, sm_scale=1.5, link_gbps=22.0),
    Scn("blocking FA GEMMs (today's kernels)", gemm_async=False),
]


def task_us(s, passes):
    run = passes * CLK_NS * 1e-3 * s.cpp
    return max(s.ssw_us, run + s.fa_us) if s.gemm_async else run + s.blk_us


def chunk_model(s):
    """Per chunk and cluster: the engines' time a K/V tile, the chunk's compute, and its array efficiency."""
    rows = s.chunk // 2
    nq = rows // BR
    if rows % BR or s.tokens % s.chunk or KEYS % s.bc:
        raise ValueError(f"chunk {s.chunk} / bc {s.bc} do not tile {s.tokens} tokens over {KEYS} keys")
    tiles = KEYS // s.bc
    qk, pv = task_us(s, (s.bc // 16) * (D_QK // 4)), task_us(s, D_V // 16 * (s.bc // 4))
    gemm = nq * (qk + pv)
    simd = nq * SM64_US * s.sm_scale * s.bc / 64
    dma = (D_QK + D_V) * s.bc / (s.dma_gbps * 1e3)
    tile = max(gemm, simd, dma)
    bound = ("GEMM", "SIMD", "iDMA")[[gemm, simd, dma].index(tile)]
    rb = rows // 16
    proj_passes = rb * ((HID // 4) * (Q_HEAD // 16) + (NOPE // 4) * (D_V // 16)
                        + (D_V // 4) * (NOPE // 16) + (NOPE // 4) * (HID // 16))
    ntask = math.ceil(proj_passes / PROJ_TASK_PASSES)
    proj = proj_passes * CLK_NS * 1e-3 * s.cpp + ntask * (s.fa_us if s.gemm_async else s.blk_us)
    compute = tiles * tile + proj + SEAM_US
    ideal = s.chunk * HEADS * MAC_TOKEN_HEAD / PEAK_MAC_US
    return dict(nq=nq, tiles=tiles, qk=qk, pv=pv, gemm=gemm, simd=simd, dma=dma, tile=tile, bound=bound,
                proj=proj, compute=compute, ideal=ideal, eta=ideal / compute,
                dma_gbps=(D_QK + D_V) * s.bc / tile / 1e3)


def windows(s):
    """The window of each arm, us. The ladder: base, bcast, bcast_reduce, all; then all without one mechanism."""
    m = chunk_model(s)
    n, c, comp = s.tokens // s.chunk, s.chunk, m["compute"]
    link = s.link_gbps * 1e3
    stream_base = (2 * CACHE_B + 2 * WEIGHT_B) / link         # two chips' copies on a row link
    stream_bc = (CACHE_B + 2 * WEIGHT_B) / link               # one multicast copy, two chips' weights
    x_pull, x_mc = X_PULL_US_PER_TOKEN * c, X_MCAST_US_PER_TOKEN * c
    scale = c * HID * 2 / 2 / (128 * 1024)                     # a cluster's y partial against E2's 128 KiB
    red_root, chain = ROOT_PULL_US * scale, CHAIN_US * scale
    bar = 2 * BARRIER_US
    w = {
        "base": PROLOGUE_US + n * (x_pull + max(comp, stream_base) + red_root + bar + FILL_US),
        "bcast": PROLOGUE_US + n * (x_mc + max(comp, stream_bc) + red_root + bar + FILL_US),
        "bcast_reduce": PROLOGUE_US + n * (x_mc + max(comp, stream_bc) + chain + bar + FILL_US),
        "all": PROLOGUE_US + n * max(comp, stream_bc) + chain + TAIL_US,
        # all without the multicast: x pulled once (the rest under the sweeps), every chip pushes its own cache copy
        "reduce_sync": PROLOGUE_US + (x_pull - x_mc) + n * max(comp, stream_base) + chain + TAIL_US,
        # all without the chain: the root's clusters load and fold every chunk's partials besides their own sweep
        "bcast_sync": PROLOGUE_US + n * (max(comp, stream_bc) + red_root) + TAIL_US,
    }
    ideal = s.tokens * HEADS * MAC_TOKEN_HEAD / PEAK_MAC_US
    return m, {k: (v, 100 * ideal / v) for k, v in w.items()}, ideal, dict(stream_base=stream_base,
                                                                             stream_bc=stream_bc)


def l1_budget(s):
    """A cluster's static L1 plan, bytes, with nothing overlaid (the plan may overlay lifetimes)."""
    rows, bc = s.chunk // 2, s.bc
    nq = rows // BR
    ring = 3 * bc * D_QK                                       # K slab is the larger chunk; 3 buffers
    parts = {
        "ring slabs (K / V / weights, 3)": ring,
        "Q8 (query tiles)": nq * BR * D_QK,
        "O accumulators (INT32)": nq * D_V * BR * 4,
        "score tiles S^T (FP16, 2 per tile)": nq * 2 * (32 + bc * BR * 2),
        "P8 (2 per tile)": nq * 2 * (bc * BR + 2 * 32),
        "softmax arenas": nq * 512,
        "O^T out (FP16) + l": nq * (D_V * BR * 2 + 64),
        "x rows (INT8)": rows * HID,
        "q / absorb / O requant (INT8, INT32)": rows * Q_HEAD * 4 + rows * D_V,
        "y partial (FP16)": rows * HID * 2,
        "flags, comm": 8192,
    }
    return parts


def break_even(s, target=80.0):
    """The array efficiency of a chunk at which `all` just holds `target` % (bisection over a flat eta)."""
    m, w, ideal, _ = windows(s)
    n = s.tokens // s.chunk
    chain = CHAIN_US * s.chunk * HID / (128 * 1024)
    # all = PROLOGUE + n * compute + chain + TAIL (compute-bound); compute = ideal_chunk / eta
    need = (100 * ideal / target - PROLOGUE_US - chain - TAIL_US) / n
    return m["ideal"] / need


def fmt(rows, head):
    out = [head]
    out += rows
    return "\n".join(out)


def report(emit):
    nom = SCENARIOS[0]
    m, w, ideal, st = windows(nom)
    emit(f"## FA prefill, T = {nom.tokens}: {nom.tokens // nom.chunk} chunks of {nom.chunk} tokens, "
         f"tiles of {nom.bc} keys, {m['nq']} query tiles a cluster\n")
    emit(f"Useful work {nom.tokens * HEADS * MAC_TOKEN_HEAD / 1e9:.2f} G MAC; ideal window "
         f"{ideal:,.0f} us on 32 clusters x 1,024 MAC at {CLK_NS} ns.\n")

    emit("### Tasks and the array efficiency of a chunk\n")
    emit("| scenario | QK task | PV task | tile: GEMM / SIMD / iDMA (us) | bound | iDMA GB/s | chunk compute (us) | chunk efficiency |")
    emit("|---|---|---|---|---|---|---|---|")
    for s in SCENARIOS:
        c = chunk_model(s)
        emit(f"| {s.name} | {c['qk']:.2f} us | {c['pv']:.2f} us | {c['gemm']:.1f} / {c['simd']:.1f} / {c['dma']:.1f}"
             f" | {c['bound']} | {c['dma_gbps']:.1f} | {c['compute']:.0f} | {100 * c['eta']:.1f} % |")
    emit("")

    emit("### The ladder, MAC utilisation (estimate; measured TBD)\n")
    emit("| scenario | base | + broadcast | + reduce | all | all without broadcast | without reduce | without per-edge sync |")
    emit("|---|---|---|---|---|---|---|---|")
    for s in SCENARIOS:
        _, ww, _, _ = windows(s)
        emit(f"| {s.name} | " + " | ".join(f"{ww[k][1]:.1f} %" for k in
                                         ("base", "bcast", "bcast_reduce", "all", "reduce_sync", "bcast_sync", "bcast_reduce")) + " |")
    emit("")
    emit("Windows, us (nominal): " + ", ".join(f"{k} {v[0]:,.0f}" for k, v in w.items()) + "\n")
    emit(f"Row-link stream a chunk (nominal, {nom.link_gbps} GB/s): base {st['stream_base']:.0f} us, "
         f"with the multicast {st['stream_bc']:.0f} us, against {m['compute']:.0f} us of compute.\n")

    emit("### Where 80 % is lost: `all` against the chunk's array efficiency\n")
    emit(f"`all` holds 80 % while a chunk runs at or above **{100 * break_even(nom):.1f} %** array efficiency "
         f"(nominal {100 * m['eta']:.1f} %).\n")
    emit("| GEMM fixed cost a task (us) \\ cycles a pass | 1.00 | 1.05 | 1.10 | 1.125 |")
    emit("|---|---|---|---|---|")
    for fa in (0.35, 0.7, 1.0, 1.5):
        row = []
        for cpp in (1.0, 1.05, 1.10, 1.125):
            _, ww, _, _ = windows(replace(nom, fa_us=fa, cpp=cpp))
            row.append(f"{ww['all'][1]:.1f} %")
        emit(f"| {fa} | " + " | ".join(row) + " |")
    emit("")
    emit("| softmax time vs the decode run's 1.654 us a 64 keys | 1x | 1.5x | 2x | 3x |")
    emit("|---|---|---|---|---|")
    row = []
    for sc in (1.0, 1.5, 2.0, 3.0):
        c = chunk_model(replace(nom, sm_scale=sc))
        _, ww, _, _ = windows(replace(nom, sm_scale=sc))
        row.append(f"{ww['all'][1]:.1f} % ({c['bound']})")
    emit("| `all` | " + " | ".join(row) + " |")
    emit("")

    emit("### The chunk size (tokens a chunk), `all` and base, nominal\n")
    emit("| chunk | query tiles a cluster | tile bound | iDMA GB/s | chunk efficiency | all | base | base / all | L1 plan |")
    emit("|---|---|---|---|---|---|---|---|---|")
    for c in (32, 64, 128, 256):
        try:
            s = replace(nom, chunk=c)
            mm, ww, _, _ = windows(s)
        except ValueError as e:
            emit(f"| {c} | {e} |")
            continue
        emit(f"| {c} ({nom.tokens // c} chunk{'s' if nom.tokens // c > 1 else ''}) | {mm['nq']} | {mm['bound']} | {mm['dma_gbps']:.1f} | "
             f"{100 * mm['eta']:.1f} % | {ww['all'][1]:.1f} % | {ww['base'][1]:.1f} % | "
             f"{ww['base'][0] / ww['all'][0]:.2f}x | {sum(l1_budget(s).values()):,} B |")
    emit("")
    emit("| tile (keys) | QK task | PV task | chunk efficiency | all | L1 plan |")
    emit("|---|---|---|---|---|---|")
    for bc in (64, 128, 256, 512):
        try:
            s = replace(nom, bc=bc)
            mm, ww, _, _ = windows(s)
        except ValueError as e:
            emit(f"| {bc} | {e} |")
            continue
        l1 = sum(l1_budget(s).values())
        emit(f"| {bc} | {mm['qk']:.2f} us | {mm['pv']:.2f} us | {100 * mm['eta']:.1f} % | {ww['all'][1]:.1f} % | "
             f"{l1:,} B{'' if l1 <= L1_CAP else ' (over ' + format(L1_CAP, ',') + ')'} |")
    emit("")

    emit("### A cluster's L1 (static plan, nothing overlaid)\n")
    parts = l1_budget(nom)
    emit("| buffer | bytes |")
    emit("|---|---|")
    for k, v in parts.items():
        emit(f"| {k} | {v:,} |")
    emit(f"| **total** | **{sum(parts.values()):,}** of {L1_CAP:,} (`l1_capacity`) |")
    chunks, tiles = {}, {}
    for c in (32, 64, 128, 256):
        s = replace(nom, chunk=c)
        mm, ww, _, _ = windows(s)
        chunks[c] = dict(nq=mm["nq"], bound=mm["bound"], eta=round(100 * mm["eta"], 1), all=round(ww["all"][1], 1),
                         base=round(ww["base"][1], 1), l1=sum(l1_budget(s).values()), dma_gbps=round(mm["dma_gbps"], 1))
    for bc in (64, 128, 256, 512):
        s = replace(nom, bc=bc)
        mm, ww, _, _ = windows(s)
        tiles[bc] = dict(qk=round(mm["qk"], 2), pv=round(mm["pv"], 2), eta=round(100 * mm["eta"], 1),
                         all=round(ww["all"][1], 1), l1=sum(l1_budget(s).values()))
    return dict(chunks=chunks, tiles=tiles, break_even_eta=round(100 * break_even(nom), 1),
                window_us={k: v[0] for k, v in w.items()}, mac_pct={k: round(v[1], 1) for k, v in w.items()},
                chunk=m, scenarios=[s.name for s in SCENARIOS],
                scenario_mac_pct={s.name: {k: round(v[1], 1) for k, v in windows(s)[1].items()} for s in SCENARIOS})


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--json", default=None)
    ap.add_argument("--md", default=str(HERE / "results" / "prefill_estimate.md"))
    a = ap.parse_args()
    lines = []
    res = report(lines.append)
    text = "\n".join(lines)
    print(text)
    if a.md:
        Path(a.md).parent.mkdir(parents=True, exist_ok=True)
        Path(a.md).write_text("# FA prefill estimate (T = 256), generated by estimate_prefill.py\n\n" + text + "\n")
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
