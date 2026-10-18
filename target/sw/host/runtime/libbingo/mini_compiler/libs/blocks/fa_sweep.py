# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""The FlashAttention sweep of one cluster: `nq` query tiles of `br` rows against `nt` key tiles of
`bc` keys, an online softmax per query tile, the K and V tiles supplied by the caller.

A function, not a block: the tiles come from the caller's stream (a weight ring, a LoadStream), whose
slabs and release rules the caller owns, and the buffers persist from one sweep to the next (a chunk
of queries after another), so the caller keeps them (`fa_sweep_alloc`) and hands the previous sweep's
result in (`prev`) for the reuse edges.

PER TILE j, for every query tile q (the kernels are attention transposed, gemm_fa.h / simd.h):

    QK(j, q)   S^T = K_j . Q_q^T, FP16 at k_s, into s16[j % 2][q] behind its latch beat
    SM(j, q)   the online softmax (simd_fa_softmax, EXACT): m and l carried in arena[q], P8 into
               p8[j % 2][q] in PV's B layout, corr exported behind the row sum
    PV(j, q)   O^T = corr (.) O^T + V_j^T . P_q^T, INT32 in place in oacc[q]; the last tile leaves as
               FP16 at k_o (GEMM_FA_D_FP16) into part[q]

then per query tile c / l into o~^T (simd_mla_normalise) and its transpose (xDMA): xt[q], one query
per row, FP16, in the accumulator's bytes. One resident K_j and V_j serve every query tile.

THE GEMM CORE'S PROGRAM is QK(j, *) then PV(j - 1, *), so a tile's softmax runs under the next
tile's score matmuls. With `gemm_async` every matmul is a start / wait pair (blocks/gemm_async.py): a
matmul's consumers depend on its wait, a slab or buffer it reads is free after its wait, and the
caller orders the cores with GemmQueue.program_order before compiling.

THE TILE SOURCE. `src.fetch(kind, j)` -> (ready task, slab handle, token) for K ('K') or V ('V') tile j
and `src.release(token, task)` once `task` is the last reader of the slab. fetch is called in the
ring's order, K0 V0 K1 V1 ..., and a slab's next fetch is created right after the tasks that free
it, so the caller's loads stay in the DM core's program order and their reuse edges exist when they
are made.
"""

from dataclasses import dataclass, field

from bingo_kernel_args import (
    SnaxBingoKernelGemmFaPvArgs,
    SnaxBingoKernelGemmFaQkArgs,
    SnaxBingoKernelSimdFaSoftmaxArgs,
    SnaxBingoKernelSimdMlaNormaliseArgs,
    SnaxBingoKernelXdmaTranspose2dArgs,
)

from ..comm.ports import at_offset
from .gemm_async import GemmQueue

FA = SnaxBingoKernelSimdFaSoftmaxArgs
PV = SnaxBingoKernelGemmFaPvArgs


@dataclass(frozen=True)
class FaSweepCfg:
    bc: int                    # keys a tile
    nt: int                    # tiles a sweep
    k_s: int                   # the score matmul's D-port shift (depth d_qk)
    k_o: int                   # the last PV's D-port shift (depth keys)
    a_exp: float               # a' = softmax_scale s_q~ s_c 2^k_s
    a_n_f32bits: int           # 1 / c, c = 2^k_o s_c / 127
    nq: int = 1                # query tiles sharing each K / V tile
    br: int = 16               # query rows a tile: the SIMD's FP16 lanes
    d_qk: int = 576
    d_v: int = 512
    gemm_async: bool = True
    phase: str = "fa"          # the GemmQueue phase of the sweep's matmuls

    def __post_init__(self):
        if self.bc % 16 or self.bc % 4 or self.d_qk % 4 or self.d_v % 16 or self.br % 16:
            raise ValueError(f"FaSweepCfg: bc={self.bc} d_qk={self.d_qk} d_v={self.d_v} "
                             f"br={self.br} do not tile the (16, 4, 16) array")
        if self.nt < 2 or self.nq < 1:
            raise ValueError(f"FaSweepCfg: nt={self.nt} (at least 2: the pipeline runs two tiles "
                             f"deep) nq={self.nq}")

    @property
    def beat(self):
        return FA.beat()

    @property
    def qk_passes(self):
        return (self.bc // 16) * (self.d_qk // 4) * (self.br // 16)

    @property
    def pv_passes(self):
        return (self.d_v // 16) * (self.bc // 4) * (self.br // 16)


@dataclass
class FaSweepBufs:
    arena: list
    s16: list                  # [parity][q]
    p8: list                   # [parity][q]
    oacc: list
    part: list                 # [q]: a latch beat, then O^T FP16
    layout: dict = field(default_factory=dict)


def fa_sweep_alloc(l1, cfg: FaSweepCfg) -> FaSweepBufs:
    """The sweep's buffers. `l1(name, nbytes)` -> a handle (the caller's namer, so a name is unique on
    its cluster). The softmax arenas are named to sort first: two of its tasks reach into s16 and p8 with
    unsigned strides (PLACEMENT_ORDER)."""
    beat = cfg.beat
    s16_b = beat + cfg.bc * cfg.br * 2
    p8_b = FA.p8_bytes(cfg.bc, cfg.br, FA.EXACT)
    return FaSweepBufs(
        arena=[l1(f"fas_arena{q}", FA.arena_bytes(cfg.bc, 1)) for q in range(cfg.nq)],
        s16=[[l1(f"fas_s16_{p}_{q}", s16_b) for q in range(cfg.nq)] for p in range(2)],
        p8=[[l1(f"fas_p8_{p}_{q}", p8_b) for q in range(cfg.nq)] for p in range(2)],
        oacc=[l1(f"fas_oacc{q}", cfg.d_v * cfg.br * 4) for q in range(cfg.nq)],
        part=[l1(f"fas_part{q}", beat + cfg.d_v * cfg.br * 2) for q in range(cfg.nq)],
        layout=FA.layout(cfg.bc, 1))


def fa_sweep(g, cfg: FaSweepCfg, bufs: FaSweepBufs, src, q8, roles, *, prev=None, oacc_free=None,
             name="s", early=2):
    """Build one sweep on the cluster `g` names. g: an object with node(name, core, kernel, args,
    after), dfg, gemm, chiplet, cluster (a Ctx, or the chiplet driver's adapter). q8: {q: (handle, ready
    tasks)}, the query tiles in the A layout. roles: {"gemm", "simd", "xdma"} core ids.
    prev: the previous sweep's result on these buffers (its tasks are the reuse edges);
    oacc_free: {q: [tasks]}, what must finish before oacc[q] is overwritten (the readers of the
    previous sweep's xt[q]). early: the tiles fetched before the first matmul (2, or 1 when the source
    has fewer than four slabs). Returns a dict: qk, sm, pv (per [q][j], the tasks consumers depend on),
    norm, tr ([q]), xt ([q] handles)."""
    gq = GemmQueue.of(g.dfg, cfg.gemm_async)
    gm, simd, xdma = roles["gemm"], roles["simd"], roles["xdma"]
    nq, nt, beat = cfg.nq, cfg.nt, cfg.beat
    corr = FA.corr_offset(cfg.bc, cfg.br)
    lay = bufs.layout
    prev = prev or {}
    oacc_free = oacc_free or {}
    qk = [[None] * nt for _ in range(nq)]
    sm = [[None] * nt for _ in range(nq)]
    pv = [[None] * nt for _ in range(nq)]
    kt, vt = {}, {}                            # tile -> (ready, slab, token)

    def matmul(kind, nm, args, after):
        if gq.enabled:
            return (gq.qk if kind == "qk" else gq.pv)(g, nm, args, after, cfg.phase)
        kernel = ("__snax_bingo_kernel_gemm_fa_qk" if kind == "qk"
                  else "__snax_bingo_kernel_gemm_fa_pv")
        return g.node(nm, gm, kernel, args, after)

    def fetch_k(j):
        if j < nt and j not in kt:
            kt[j] = src.fetch("K", j)

    def fetch_v(j):
        if j < nt and j not in vt:
            vt[j] = src.fetch("V", j)

    # The tiles are fetched in the ring's order, K0 V0 K1 V1 K2 V2 ...: `early` tiles before the first
    # matmul, then K_{j+1} and V_{j+1} right after PV(j-1) is created, so every fetch comes after the
    # tasks that free its slab. Two early tiles (K0 V0 K1 V1) need four slabs free of each other; with
    # one, tile 1 is fetched after tile 0's score matmuls, which free K0's slab.
    fetch_k(0)
    fetch_v(0)
    if early >= 2:
        fetch_k(1)
        fetch_v(1)

    def make_qk(j):
        k_ready, k_slab, _ = kt[j]
        for q in range(nq):
            ar, s16 = bufs.arena[q], bufs.s16[j % 2][q]
            deps = [k_ready] + (list(q8[q][1]) if j < 2 else [])
            if j >= 2:
                deps.append(sm[q][j - 2])
            elif prev:
                deps.append(prev["sm"][q][-1])
            qk[q][j] = matmul("qk", f"{name}qk{j}q{q}", SnaxBingoKernelGemmFaQkArgs(
                k_slab, q8[q][0], 0, at_offset(s16, beat), M=cfg.bc // 16, K=cfg.d_qk // 4,
                N=cfg.br // 16, d_shift=cfg.k_s), deps)
        # the slab frees when every query tile's score matmul is done: the last one in the core's
        # program, whose wait follows the earlier waits
        src.release(kt[j][2], qk[nq - 1][j])

    def make_sm(j):
        for q in range(nq):
            s16, p8 = bufs.s16[j % 2][q], bufs.p8[j % 2][q]
            deps = [qk[q][j]]
            if j:
                deps.append(sm[q][j - 1])
            elif prev:
                deps.append(prev["norm"][q])
            if j >= 2:
                deps.append(pv[q][j - 2])
            elif prev:
                deps.append(prev["pv"][q][nt - 2 + j])
            sm[q][j] = g.node(f"{name}sm{j}q{q}", simd, "__snax_bingo_kernel_simd_fa_softmax",
                              FA(at_offset(s16, beat), p8, bufs.arena[q], cfg.bc, 1, tile_idx=j,
                                 seed_state=1, geom_mode=FA.GEOM_SELF, score_scale=cfg.a_exp,
                                 p_mode=FA.EXACT, p8_pitch=0), deps)

    def make_pv(j):
        v_ready, v_slab, _ = vt[j]
        first, last = j == 0, j == nt - 1
        for q in range(nq):
            flags = PV.B_KMAJOR | (0 if first else PV.C_COLSCALE) | (PV.D_FP16 if last else 0)
            p8, oacc, part = bufs.p8[j % 2][q], bufs.oacc[q], bufs.part[q]
            deps = [sm[q][j], v_ready]
            if j:
                deps.append(pv[q][j - 1])
            else:
                deps += list(oacc_free.get(q, ()))
                if prev:
                    deps.append(prev["norm"][q])
            pv[q][j] = matmul("pv", f"{name}pv{j}q{q}", PV(
                v_slab, p8, 0 if first else oacc, at_offset(part, beat) if last else oacc,
                M=cfg.d_v // 16, K=cfg.bc // 4, N=cfg.br // 16, flags=flags,
                corr_addr=0 if first else at_offset(p8, corr), d_shift=cfg.k_o if last else 0), deps)
        src.release(vt[j][2], pv[nq - 1][j])

    for j in range(nt):
        make_qk(j)
        make_sm(j)
        if j == 0:
            fetch_k(1)
            fetch_v(1)
        if j >= 1:
            make_pv(j - 1)
            fetch_k(j + 1)
            fetch_v(j + 1)
    make_pv(nt - 1)
    norm, tr, xt = [], [], []
    for q in range(nq):
        oacc, part = bufs.oacc[q], bufs.part[q]
        ot16 = at_offset(oacc, cfg.d_v * cfg.br * 2)
        x = at_offset(oacc, 0)
        n = g.node(f"{name}norm{q}", simd, "__snax_bingo_kernel_simd_mla_normalise",
                   SnaxBingoKernelSimdMlaNormaliseArgs(
                       at_offset(bufs.arena[q], lay["lrun"]), part, at_offset(part, beat), ot16,
                       cfg.d_v, cfg.a_n_f32bits), [pv[q][nt - 1], sm[q][nt - 1]])
        t = g.node(f"{name}tr{q}", xdma, "__snax_bingo_kernel_xdma_transpose_2d",
                   SnaxBingoKernelXdmaTranspose2dArgs(ot16, x, cfg.d_v, cfg.br, 2), [n])
        norm.append(n)
        tr.append(t)
        xt.append(x)
    return dict(qk=qk, sm=sm, pv=pv, norm=norm, tr=tr, xt=xt)
