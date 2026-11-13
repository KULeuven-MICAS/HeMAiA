#!/usr/bin/env python3
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""Llama-3-8B's attention (synthetic weights) over a fresh prompt on sixteen chiplets, stages 1-4 alone: every pass's x
multicast and fetch, its projections and the K/V trade inside each chip pair. The window ends when every cluster holds
its query tiles of every pass and its key and value tiles of the whole prompt.

THE SPLIT (the whole block's, workloads/llama3/common/llama3_attn_mc.py). Cluster cl runs query head h(cl); a key/value
group's four query heads sit on a chip pair (x, 2y), (x, 2y + 1) of one column. Each chip projects half of its group's
K (cluster 0) and half of its V (cluster 1), whole RoPE chunk pairs (the q and k columns in chunk-pair order), and every
cluster of the pair takes the halves it lacks from its own chip's L3 and, by a remote read, from the mate's.

THE TASKS of pass i (names without their _k<chip>_cl<c> suffix; j a weight chunk, t a key tile):

    S0_start  -> x_hbm_bcast_i<i>                  root host: its memory chip's push engine multicasts x_i from HBM
                     :                             into every chip's L3 (a flag a 32 KiB tile); the root's ring runs
                     :                             its W_KVA chunk ahead of the casts, the rest after the last landed
                 x_wait_i<i>                       DM: x_i into L1 once its last tile's flag is there
                 rope_ld_i<i>                      DM: the pass's RoPE tables
                 Ld_w<j>_i<i> -> gq<j>_i<i>        DM ring load, GEMM: x . W_q chunk (D layout, FP16)
                 dqq<j>_i<i> -> qr*<j>_i<i>        SIMD: dequantise, RoPE of each chunk pair
                 d2aq<j>_i<i>                      xDMA: D -> A
                 q8_<u>_i<i>                       SIMD: a query tile into INT8
                 gkv<j>, dqkv<j>, ...              the cluster's K half (RoPE, A layout, k8_i<i>) or V half (row-major,
                                                   each key tile transposed into V^T's A layout, vq<j><t>_i<i>)
                 kx_put / vx_put<t>                DM: the chip's halves into its L3
                 kx_get / kx_rget / vx_get / vx_rget   DM: the halves a cluster lacks, its own chip's and the mate's
    -> fin<chip> -> S1_end                         one probe a chip behind its last tasks

Checks: the last pass's first and last query tile on both clusters of the root and of the fold order's middle chip,
and every key and value tile of the whole prompt on those clusters.

params.hjson: the prompt and its passes (tokens, rows, bc), the weights (wbits, w_chunk_bytes, w_buffers, weight_ring),
the platform and compile switches."""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(__file__).split("/target/sw/host/")[0]
sys.path.insert(0, os.path.join(REPO, "util", "sim", "multichip"))
sys.path.insert(0, os.path.join(REPO, "util", "sim", "common"))
sys.path.insert(0, os.path.join(REPO, "target", "sw", "host", "runtime", "libbingo", "mini_compiler"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "llama3", "common"))
import _bingo_paths  # noqa: F401,E402
from bingo_kernel_args import (  # noqa: E402
    INT4_B_LAYOUT_PAIRED, HostBingoKernelHbmBcastArgs, SnaxBingoKernelGemmFullArgs, SnaxBingoKernelIdma1dCopyArgs,
    SnaxBingoKernelIdma2dCopyArgs, SnaxBingoKernelIdmaFetchFlaggedArgs, SnaxBingoKernelSimdAddF16Args,
    SnaxBingoKernelSimdFp16ToInt8Args, SnaxBingoKernelSimdMulF16Args, SnaxBingoKernelXdmaTranspose2dArgs,
    simd_beats, simd_fp16_lanes)
from bingo_mem_handle import BingoMemSymbol  # noqa: E402
from driver import run_driver  # noqa: E402
from libs.blocks import FaSweepCfg, fa_sweep_alloc  # noqa: E402
from libs.comm.nest import convert_args, d_to_a_args  # noqa: E402

import chiplet_mc as cm  # noqa: E402
import llama3_data  # noqa: E402

BLK = 16                      # a GEMM column block (meshCol)
BR = 16                       # query rows a tile: the SIMD's FP16 lanes
QC = 32                       # columns a projection weight chunk
SLC = 128                     # y's slices: a cluster's columns in one slice (the block's wo_slice)
XT = 32768                    # the x multicast's tile, a flag after each
X_PF_CHUNKS = 1               # the root's ring chunks ahead of the x multicasts in its memory chip's queue


class Stage1234(cm.ChipletBuild):
    """Stages 1-4 of the attention (the module's docstring). run_driver makes the program in four steps: __init__ (the
    parameters), mem_handles() (the data and the memory, no task), build() (the task graph) and finish() (the
    compile)."""
    TAG = "la"

    def __init__(self, args, p, plat, hw):
        super().__init__(args, p, plat, hw)
        if simd_fp16_lanes() != BR:
            raise ValueError(f"the SIMD has {simd_fp16_lanes()} FP16 lanes; the softmax takes one query a lane")
        self.R = int(p.get("rows", 64))
        self.bc = int(p.get("bc", 32))
        self.int4_b = INT4_B_LAYOUT_PAIRED
        self.seed = int(p.get("seed", 1))               # the synthetic weights' and prompt's draws

    # ---- the data and the memory ------------------------------------------------------------------------------------
    def mem_handles(self):
        """The data and its golden (llama3_data.LlamaAttnData), the shapes and the split, then every input staged
        (_stage), the load streams' slabs, the weight rings and the L1 buffers (_buffers)."""
        d = self.d = llama3_data.LlamaAttnData(tokens=self.T, rows=self.R, bc=self.bc, seed=self.seed,
                                               wbits=self.wbits, mask_c=True)
        self.lay, self.mesh = d.layout, d.mesh
        self.NP, self.HD, self.H = d.NP, d.HD, d.H
        self.NQ = self.R // BR                          # query tiles a pass
        self.TPP = self.R // self.bc                    # key tiles a pass
        self.NQC = self.HD // QC                        # weight chunks a projection
        if self.NQC % 4:
            raise ValueError(f"{self.NQC} chunks a head: two halves of whole RoPE chunk pairs needed")
        # the head's dimensions in chunk-pair order: chunk 2t the first dimension of 32 pairs, chunk 2t + 1 the second
        first = np.arange(QC) * 2
        self.perm = np.concatenate([np.concatenate([t * 2 * QC + first, t * 2 * QC + first + 1])
                                    for t in range(self.NQC // 2)])
        self.HALF = self.H // self.NC
        if self.HALF % SLC:
            raise ValueError(f"y's slices of {SLC}: whole slices of a cluster's {self.HALF} outputs")
        self.WOC = self.H // (len(self.chips) * self.NC)  # y's columns a cluster
        self.NWS = self.WOC // 32                       # its W_O GEMMs of 32 columns
        if len(self.chips) * self.NC != d.NH:
            raise ValueError(f"{len(self.chips) * self.NC} clusters: the attention has {d.NH} query heads, one a cluster")
        pairs = sorted({(k, k + 1) for k in self.chips if (k & 15) % 2 == 0 and k + 1 in self.chips})
        if len(pairs) != d.NKV:
            raise ValueError(f"{len(pairs)} chip pairs (x, 2y), (x, 2y + 1): the attention has {d.NKV} key/value groups")
        self.group, self.head, self.kv_half, self.kv_mate = {}, {}, {}, {}
        for m, (ka, kb) in enumerate(pairs):
            for n, (k, c) in enumerate(((ka, 0), (ka, 1), (kb, 0), (kb, 1))):
                cl = self.chips.index(k) * self.NC + c
                self.group[k] = m
                self.head[cl] = m * d.GQ + n
            self.kv_half[ka], self.kv_half[kb] = 0, 1
            self.kv_mate[ka], self.kv_mate[kb] = kb, ka
        self.kind = {cl: "K" if self.local(cl) == 0 else "V" for cl in range(self.N_CL)}     # the projection besides q
        self.NKVC = self.NQC // 2                       # the k or v chunks a cluster projects: its chip's half
        self.WO0 = self.NQC + self.NKVC                 # its first W_O chunk in its weight stream
        self.gold_y = d.part(list(range(d.NH)))
        for line in d.report(d.y16(self.gold_y)):
            print(f"[{self.TAG}] {line}")
        print(f"[{self.TAG}] {self.T} tokens in {self.NP} passes of {self.R} ({self.NP} built): {self.NQ} query tiles, "
              f"{self.TPP} key tiles of {self.bc} a pass; W_O by columns, {self.WOC} a cluster in slices of {SLC}; "
              f"fold order {[f'{k:02x}' for k in self.order]}; key/value groups "
              f"{[(f'{a:02x}', f'{b:02x}') for a, b in pairs]}")
        self.XB = self.R * self.H                       # a pass's x, INT8
        self.YB = self.R * self.H * 2                   # a pass's y, FP16
        self.SLB = self.R * SLC * 2                     # one slice of y, FP16
        self.TILE_B = self.bc * self.HD                 # a key or value tile, INT8
        self.HKB = self.TILE_B * self.NKVC // self.NQC  # the half of one a cluster projects
        self._stage()
        self.make_streams()
        self.make_rings()
        self._buffers()

    def kv_cc(self, cl):
        """The k or v chunks cluster cl projects: its chip's half."""
        h = self.kv_half[self.chip(cl)]
        return range(h * self.NKVC, (h + 1) * self.NKVC)

    def xland(self, off=0):
        """Where x lands in every chip's L3, at byte `off`: after its pass's source in the overlay of x and y."""
        return BingoMemSymbol(f"{self.TAG}_xy", offset=off // self.XB * self.YB + self.XB + off % self.XB)

    def _pack_b(self, q):
        return self.lay.pack_int4(self.lay.to_b_pairs(q, self.mesh))

    def _d_rows(self, vec, rows):
        """A per-column vector on `rows` rows in the D layout, FP16."""
        return self.lay.to_d(np.tile(np.asarray(vec, dtype=np.float16), (rows, 1)), self.mesh).astype(np.float16)

    def _stage(self):
        """x in the HBM of the root's memory chip (a flag a tile), the overlay of every pass's x landing and y in L3,
        the flags, the trades' areas, the RoPE tables, and per cluster its weights in the order it streams them and its
        constants (the chunks' dequantisation factors, the mask patterns) in its memory chip's HBM."""
        st, d, T, lay, R, HD = self.st, self.d, self.TAG, self.lay, self.R, self.HD
        xs = [lay.to_a(d.hq[i * R:(i + 1) * R], self.mesh).astype(np.int8) for i in range(self.NP)]
        if self.YB != 2 * self.XB:
            raise ValueError("a pass's y must be as long as its x source and landing area (the overlay)")
        m = self.mem[self.ROOT]
        self.h_xhbm = st.put_hbm(f"{T}_xhbm", np.concatenate(xs), mem_chip=(m >> 4, m & 15))
        if self.XB % XT:
            raise ValueError(f"the x multicast's tile of {XT} B: a pass's x ({self.XB} B) in whole tiles")
        self.NXT = self.XB // XT
        st.put_zeros(f"{T}_xtflag", "uint8_t", self.NP * self.NXT * 64)
        st.put_zeros(f"{T}_xy", "int8_t", self.NP * self.YB, preload=True)
        st.put_zeros(f"{T}_xflag", "uint8_t", self.NP * 64)
        self.KVX = 2 * self.TPP * self.HKB
        st.put_zeros(f"{T}_kvx", "uint8_t", self.NP * self.KVX)
        st.put_zeros(f"{T}_oflag", "uint8_t", self.NP * self.d.NH * 64)
        st.put_zeros(f"{T}_omflag", "uint8_t", self.NP * 64)
        st.put_zeros(f"{T}_yflag", "uint8_t", self.NP * self.N_CL * 64)
        st.put_zeros(f"{T}_ydum", "uint8_t", self.N_CL * 64)
        rope = []
        for i in range(self.NP):
            for c in range(self.NQC):
                cols = self.perm[c * QC:(c + 1) * QC]
                rope += [lay.to_d(d.cos_rep[i * R:(i + 1) * R][:, cols], self.mesh),
                         lay.to_d(d.sin_sgn[i * R:(i + 1) * R][:, cols], self.mesh)]
        self.ROPE_B = 2 * self.NQC * R * QC * 2         # a pass's tables
        st.put(f"{T}_rope", "uint16_t", np.concatenate(rope).astype(np.float16).view(np.uint16))
        self.img, self.chunk_off, self.cst = {}, {}, {}
        for cl in range(self.N_CL):
            k, h = self.chip(cl), self.head[cl]
            m = self.group[k]
            loc = (self.mem[k] >> 4, self.mem[k] & 15)
            wkv, fkv = (d.wk_q, d.k_f[m]) if self.kind[cl] == "K" else (d.wv_q, d.v_f[m])
            qcol = lambda cc: self.perm[cc * QC:(cc + 1) * QC]                          # noqa: E731
            kcol = qcol if self.kind[cl] == "K" else (lambda cc: np.arange(cc * QC, (cc + 1) * QC))
            parts = [self._pack_b(d.wq_q[:, h * HD + qcol(cc)]) for cc in range(self.NQC)]
            parts += [self._pack_b(wkv[:, m * HD + kcol(cc)]) for cc in self.kv_cc(cl)]
            for s in range(self.NWS):
                o0 = self.WOC * cl + s * 32
                parts.append(self._pack_b(d.wo_q[:, o0:o0 + 32]))
            offs, off = [], 0
            for b in parts:
                if b.size > self.chunk:
                    raise ValueError(f"a weight chunk of {b.size} B; params w_chunk_bytes is {self.chunk}")
                offs.append((off, b.size))
                off += b.size
            self.img[cl] = st.put_hbm(f"{T}_w_cl{cl}", np.concatenate(parts).astype(np.int8), mem_chip=loc)
            self.chunk_off[cl] = offs
            sc = [self._d_rows(d.q_f[h][qcol(cc)], R) for cc in range(self.NQC)]
            sc += [self._d_rows(fkv[kcol(cc)], R) for cc in self.kv_cc(cl)]
            blob = np.concatenate([np.asarray(a, dtype=np.float16).reshape(-1).view(np.int8) for a in sc]
                                  + [mk.reshape(-1).view(np.int8) for mk in d.mask_bias_patterns()])
            self.cst[cl] = (st.put_hbm(f"{T}_c_cl{cl}", blob, mem_chip=loc), blob.nbytes)
        self.SC_B = R * QC * 2                          # one chunk's factors

    def _buffers(self):
        """Every cluster's L1: the exchange buffer, every key and value tile of the prompt, the constants, o8, y's
        columns and the sweep's buffers (the whole block's, so the plan is the block's)."""
        d, R = self.d, self.R
        self.xch_zero = False
        self.xch_bytes = 2 * self.SLB
        self.xch = {(k, c): self.cx_chip(k).at(c).l1(f"AAAA_{self.TAG}_xch", self.xch_bytes)
                    for k in self.chips for c in range(self.NC)}
        self.p1c = {c: self.cx_chip(self.ROOT).at(c).l1(f"{self.TAG}_p1c", self.xch_bytes) for c in range(self.NC)}
        self.kbuf, self.vbuf, self.consts = {}, {}, {}
        for cl in range(self.N_CL):
            l1 = self.l1_of(cl)
            self.kbuf[cl] = l1("kt", self.NP * self.TPP * self.TILE_B)     # every key tile of the prompt, A layout
            self.vbuf[cl] = l1("vt", self.NP * self.TPP * self.TILE_B)     # every value tile, V^T in the A layout
            self.consts[cl] = l1("consts", self.cst[cl][1])
            l1("o8", R * self.HD)
            l1("ywo", self.R * self.WOC * 2)
            cfg = FaSweepCfg(bc=self.bc, nt=self.TPP, nq=self.NQ, br=BR, k_s=d.k_s, k_o=d.k_o, a_exp=d.a_exp,
                             a_n_f32bits=d.a_n_bits, d_qk=self.HD, d_v=self.HD)
            fa_sweep_alloc(lambda name, nb, l1=l1: l1(name, nb), cfg)
        self._pb = {}

    def l1_of(self, cl):
        k, c = self.chip(cl), self.local(cl)
        return lambda name, nb: self.cx_chip(k).at(c).l1(f"{self.TAG}_{name}", nb)

    def pb(self, cl, i):
        """Pass i's own buffers on cluster cl: the static L1 planner lays those of different passes over each other."""
        if (cl, i) in self._pb:
            return self._pb[cl, i]
        R, l1 = self.R, self.l1_of(cl)
        b = dict(x=l1(f"x_i{i}", self.XB), rope=l1(f"rope_i{i}", self.ROPE_B),
                 a16q=l1(f"a16q_i{i}", R * self.HD * 2), q8=[l1(f"q8_{u}_i{i}", BR * self.HD) for u in range(self.NQ)],
                 a16o=l1(f"a16o_i{i}", R * self.HD * 2))
        b["ofull"] = l1(f"x_i{i}o", R * self.H)
        for side, ccs in (("q", range(self.NQC)), ("kv", self.kv_cc(cl))):
            for cc in ccs:
                b[f"{side}d{cc}"] = l1(f"{side}d{cc}_i{i}", R * QC * 2)
                b[f"{side}s{cc}"] = l1(f"{side}s{cc}_i{i}", R * QC * 2)       # x sin of the other chunk
                b[f"{side}t{cc}"] = l1(f"{side}t{cc}_i{i}", R * QC * 2)       # x cos
        if self.kind[cl] == "K":
            b["a16k"] = l1(f"a16k_i{i}", R * self.NKVC * QC * 2)
            b["k8h"] = l1(f"k8h_i{i}", R * self.NKVC * QC)                    # the chip's half of the new key tiles
        else:
            for cc in self.kv_cc(cl):
                b[f"vr{cc}"] = l1(f"vr{cc}_i{i}", R * QC * 2)                 # row-major v
                for t in range(self.TPP):
                    b[f"vt{cc}{t}"] = l1(f"vt{cc}{t}_i{i}", QC * self.bc * 2)  # its tile t transposed
                    b[f"va{cc}{t}"] = l1(f"va{cc}{t}_i{i}", QC * self.bc * 2)  # in the A layout
        self._pb[cl, i] = b
        return b

    # ---- one task ---------------------------------------------------------------------------------------------------
    def _load(self, cl, i, j):
        """Weight chunk j of cluster cl into the next slab, a ring load (every pass streams its chunks again)."""
        k, c, dm = self.chip(cl), self.local(cl), self.roles["dm"]
        ls = self.ls[cl]
        off, nb = self.chunk_off[cl][j]
        si, slab, war = ls.take(nb)
        ld, _ = ls.rings.load_chunk(
            lambda n, core, kern, args, after: self.node(k, c, core, n, kern, args, after), c,
            f"Ld_w{j}_i{i}", src=self.img[cl].address + off, nbytes=nb, slab=slab, war=war, deps=(), dm=dm)
        return ld, si, slab

    def gmm(self, cl, name, a, b, dst, M, K, N, k_shift, deps):
        k, c = self.chip(cl), self.local(cl)
        args = SnaxBingoKernelGemmFullArgs(
            input_A_addr=a, input_B_addr=b, input_C_addr=0, output_D_addr=dst, M=M, K=K, N=N, array_shape_idx=0,
            transpose_A=0, transpose_B=0, accumPrevC=0, int32tofp16_enable=1, int4_b_enable=self.int4_b, d_shift=k_shift)
        return self.node(k, c, self.roles["gemm"], name, "__snax_bingo_kernel_gemm_full", args, list(deps))

    def nd(self, cl, role, name, kernel, args, deps):
        return self.node(self.chip(cl), self.local(cl), self.roles[role], name, kernel, args, list(dict.fromkeys(deps)))

    def mul(self, cl, name, a, b, out, n, deps):
        return self.nd(cl, "simd", name, "__snax_bingo_kernel_simd_stream_elementwise",
                       SnaxBingoKernelSimdMulF16Args(a, b, out, n // 32, 32), deps)

    def add(self, cl, name, a, b, out, n, deps):
        return self.nd(cl, "simd", name, "__snax_bingo_kernel_simd_stream_elementwise",
                       SnaxBingoKernelSimdAddF16Args(a, b, out, n // 32, 32), deps)

    def quant(self, cl, name, src, dst, nbytes, inv, deps):
        bits = int(np.asarray(inv, dtype=np.float32).reshape(()).view(np.uint32))
        return self.nd(cl, "simd", name, "__snax_bingo_kernel_simd_fp16_to_int8",
                       SnaxBingoKernelSimdFp16ToInt8Args(src, dst, beats=simd_beats(nbytes), rows=1, inv_scale_f32bits=bits),
                       deps)

    def xd(self, cl, name, args, deps):
        return self.nd(cl, "xdma", name, "__snax_bingo_kernel_xdma_6d", args, deps)

    def copy(self, cl, name, src, dst, nb, deps):
        return self.nd(cl, "dm", name, "__snax_bingo_kernel_idma_1d_copy", SnaxBingoKernelIdma1dCopyArgs(src, dst, nb),
                       deps)

    def copy2d(self, cl, name, src, dst, size, sst, dstr, reps, deps):
        return self.nd(cl, "dm", name, "__snax_bingo_kernel_idma_2d_copy",
                       SnaxBingoKernelIdma2dCopyArgs(src, dst, size, sst, dstr, reps), deps)

    # ---- x ----------------------------------------------------------------------------------------------------------
    def x_puts(self, go):
        """Every pass's x multicast, one after the other on the root's host; the root's ring runs its first
        X_PF_CHUNKS chunks ahead of them in its memory chip's queue and the rest once the last has landed."""
        R, T, m = self.ROOT, self.TAG, self.mem[self.ROOT]
        puts = []
        for i in range(self.NP):
            puts.append(self.hbm_cast(
                R, f"x_hbm_bcast_i{i}",
                HostBingoKernelHbmBcastArgs(self.h_xhbm.address + i * self.XB, self.xland(i * self.XB),
                                            BingoMemSymbol(f"{T}_xtflag", offset=i * self.NXT * 64),
                                            self.h_seq[m].address + 64, 1, XT, self.NXT, m, self.engine[R]),
                after=[go[R]] + puts[-1:], ahead_of_rings=False))
        if not 0 < X_PF_CHUNKS <= self.n_slots:
            raise ValueError(f"{X_PF_CHUNKS} ring chunks ahead of x: 1..{self.n_slots} (the ring's slots)")
        ring = self.rings[R]
        for r in range(ring.n_rings):
            ring.boundary(r, X_PF_CHUNKS, before=[self.first_of(puts[0])], after=[self.done_of(puts[-1])])
        return puts

    def stage_x(self, cl, i, put, deps):
        """Pass i's x into cluster cl's L1 once the multicast's last tile has landed (its flag); the post orders the
        fetch, the flag proves the landing."""
        T = self.TAG
        flag = BingoMemSymbol(f"{T}_xtflag", offset=(i * self.NXT + self.NXT - 1) * 64)
        nd = self.nd(cl, "dm", f"x_wait_i{i}", "__snax_bingo_kernel_idma_fetch_flagged",
                     SnaxBingoKernelIdmaFetchFlaggedArgs(self.xland(i * self.XB), self.pb(cl, i)["x"], self.XB, flag, 1, 1),
                     list(deps))
        self.dfg.bingo_add_order_edge(self.queued_of(put), nd)
        return nd

    # ---- the projections --------------------------------------------------------------------------------------------
    def _rope_pair(self, cl, i, side, ca, cb, after):
        """RoPE of chunks ca (the pairs' first dimensions) and cb (their second), [R, 32] D layout each, in place:
        a = a cos_a + b sin_a, b = b cos_b + a sin_b (sin signed). Returns the two adds."""
        b, n = self.pb(cl, i), self.R * QC
        tab = lambda cc, k: b["rope"].view((2 * cc + k) * self.SC_B)            # noqa: E731
        xa, xb = b[f"{side}d{ca}"], b[f"{side}d{cb}"]
        dep = list(after) + [self.rope_ld[cl, i]]
        ta = self.mul(cl, f"{side}rc{ca}_i{i}", xa, tab(ca, 0), b[f"{side}t{ca}"], n, dep)
        sa = self.mul(cl, f"{side}rs{ca}_i{i}", xb, tab(ca, 1), b[f"{side}s{ca}"], n, dep)
        tb = self.mul(cl, f"{side}rc{cb}_i{i}", xb, tab(cb, 0), b[f"{side}t{cb}"], n, dep)
        sb = self.mul(cl, f"{side}rs{cb}_i{i}", xa, tab(cb, 1), b[f"{side}s{cb}"], n, dep)
        ra = self.add(cl, f"{side}ra{ca}_i{i}", b[f"{side}t{ca}"], b[f"{side}s{ca}"], xa, n, [ta, sa, sb])
        rb = self.add(cl, f"{side}ra{cb}_i{i}", b[f"{side}t{cb}"], b[f"{side}s{cb}"], xb, n, [tb, sb, sa])
        return ra, rb

    def _v_tiles(self, cl, i, cc, dq, vq):
        """A v chunk: D -> row-major, each key tile transposed into V^T's A layout and quantised into its place (in
        turn, so the last stands for every one)."""
        b, d = self.pb(cl, i), self.d
        vr = self.xd(cl, f"vrm{cc}_i{i}", convert_args("D", "row_major", self.R, QC, self.mesh, 2, b[f"kvd{cc}"],
                                                       b[f"vr{cc}"]), [dq])
        for t in range(self.TPP):
            tr = self.nd(cl, "xdma", f"vtr{cc}{t}_i{i}", "__snax_bingo_kernel_xdma_transpose_2d",
                         SnaxBingoKernelXdmaTranspose2dArgs(b[f"vr{cc}"].view(t * self.bc * QC * 2),
                                                            b[f"vt{cc}{t}"], self.bc, QC, 2), [vr])
            va = self.xd(cl, f"vta{cc}{t}_i{i}",
                         convert_args("row_major", "A", QC, self.bc, self.mesh, 2, b[f"vt{cc}{t}"], b[f"va{cc}{t}"]), [tr])
            j = i * self.TPP + t
            vq.append(self.quant(cl, f"vq{cc}{t}_i{i}", b[f"va{cc}{t}"],
                                 self.vbuf[cl].view(j * self.TILE_B + cc * QC * self.bc),
                                 QC * self.bc * 2, d.s_v.inv, [va] + vq[-1:]))

    def stage_proj(self, cl, i, x):
        """Pass i's q (into its query tiles) and the cluster's k or v half. Returns ({u: (q8 handle, ready)}, every
        projection GEMM (each reads x), the task after which the cluster's own new tiles are there)."""
        d, b, R, HD = self.d, self.pb(cl, i), self.R, self.HD
        mt, kb = R // 16, HD // 4                     # row tiles; 4-column blocks of a head in the A layout
        self.rope_ld[cl, i] = self.copy(cl, f"rope_ld_i{i}", BingoMemSymbol(f"{self.TAG}_rope", offset=i * self.ROPE_B),
                                        b["rope"], self.ROPE_B, [])
        cst = self.consts[cl]
        x_readers, d2a, vq = [], {"q": [], "kv": []}, []
        # (side, its index in the cluster's weight stream and constants, chunk)
        items = ([("q", cc, cc) for cc in range(self.NQC)]
                 + [("kv", self.NQC + n, cc) for n, cc in enumerate(self.kv_cc(cl))])
        first_dq = None
        for side, wi, cc in items:
            ld, si, slab = self._load(cl, i, wi)
            dd = b[f"{side}d{cc}"]
            gm = self.gmm(cl, f"g{side}{cc}_i{i}", b["x"], slab, dd, mt, self.H // 4, QC // BLK, d.k_x, [ld, x])
            self.ls[cl].release(si, gm)
            x_readers.append(gm)
            dq = self.mul(cl, f"dq{side}{cc}_i{i}", dd, cst.view(wi * self.SC_B), dd, R * QC, [gm])
            if side == "kv" and self.kind[cl] == "V":
                self._v_tiles(cl, i, cc, dq, vq)
                continue
            dst = b["a16q"] if side == "q" else b["a16k"]
            # the chunk's place in the A layout: q's of the head; k's of the columns the cluster projects
            lo, kbw = (0, kb) if side == "q" else (self.kv_cc(cl)[0], self.NKVC * QC // 4)
            to_a = lambda c_, rt_: self.xd(cl, f"d2a{side}{c_}_i{i}",                       # noqa: E731
                                           d_to_a_args(self.mesh, mt, QC // BLK, kbw, b[f"{side}d{c_}"],
                                                       dst.view((c_ - lo) * (QC // 4) * 16 * 4 * 2)), [rt_])
            if cc % 2 == 0:
                first_dq = dq                     # the pair's first chunk waits for its second
            else:
                ra, rb = self._rope_pair(cl, i, side, cc - 1, cc, [first_dq, dq])
                d2a[side] += [to_a(cc - 1, ra), to_a(cc, rb)]
        q8 = {}
        for u in range(self.NQ):
            qa = self.quant(cl, f"q8_{u}_i{i}", b["a16q"].view(u * BR * HD * 2), b["q8"][u], BR * HD * 2, d.s_q.inv,
                            d2a["q"])
            q8[u] = (b["q8"][u], [qa])
        if self.kind[cl] == "K":
            own = self.quant(cl, f"k8_i{i}", b["a16k"], b["k8h"], R * self.NKVC * QC * 2, d.s_k.inv, d2a["kv"])
        else:
            own = vq[-1]
        return q8, x_readers, own

    # ---- the K/V trade ----------------------------------------------------------------------------------------------
    def stage_trade(self, i, own):
        """Each chip's half of the pass's new key tiles (cluster 0, compact) and value tiles (cluster 1, in place) into
        its own L3; every cluster of the pair then copies the halves it lacks into its tiles, from its own chip's L3
        and, by a remote read, from the mate's (a half is only ever written where it is made). A key tile's half is a
        run of each 16-row tile of its A layout; a value tile's (V^T) is contiguous."""
        T, HD, TPP, bc = self.TAG, self.HD, self.TPP, self.bc
        hk = self.NKVC * QC * BR                # a key half's run in a 16-row tile (INT8 A layout)
        hv = self.NKVC * QC * bc                # a value half of one tile
        rt = TPP * bc // BR                     # the pass's 16-row tiles of keys
        base, voff = i * self.KVX, TPP * self.HKB
        kvx = lambda off, chip=None: BingoMemSymbol(f"{T}_kvx", offset=base + off, chip_id=chip)    # noqa: E731
        kput, vput, kpl = {}, {}, {}
        for k in self.chips:
            cl0, cl1 = self.clusters(k)
            h = self.kv_half[k]
            kput[k] = self.copy(cl0, f"kx_put_i{i}", self.pb(cl0, i)["k8h"], kvx(0), TPP * self.HKB, [own[cl0]])
            kpl[k] = self.copy2d(cl0, f"kx_pl_i{i}", self.pb(cl0, i)["k8h"],
                                 self.kbuf[cl0].view(i * TPP * self.TILE_B + h * hk), hk, hk, HD * BR, rt, [own[cl0]])
            vput[k] = [self.copy(cl1, f"vx_put{t}_i{i}", self.vbuf[cl1].view((i * TPP + t) * self.TILE_B + h * hv),
                                 kvx(voff + t * hv), hv, [own[cl1]]) for t in range(TPP)]
        for k in self.chips:
            cl0, cl1 = self.clusters(k)
            h, m = self.kv_half[k], self.kv_mate[k]
            kdst = lambda cl, hh: self.kbuf[cl].view(i * TPP * self.TILE_B + hh * hk)              # noqa: E731
            vdst = lambda cl, t, hh: self.vbuf[cl].view((i * TPP + t) * self.TILE_B + hh * hv)     # noqa: E731
            k1 = self.copy2d(cl1, f"kx_get_i{i}", kvx(0), kdst(cl1, h), hk, hk, HD * BR, rt, [kput[k]])
            k0 = self.copy2d(cl0, f"kx_rget_i{i}", kvx(0, m), kdst(cl0, 1 - h), hk, hk, HD * BR, rt, [kput[m], kpl[k]])
            k1 = self.copy2d(cl1, f"kx_rget_i{i}", kvx(0, m), kdst(cl1, 1 - h), hk, hk, HD * BR, rt, [kput[m], k1])
            v0, v1 = None, own[cl1]
            for t in range(TPP):
                v0 = self.copy(cl0, f"vx_get{t}_i{i}", kvx(voff + t * hv), vdst(cl0, t, h), hv,
                               [vput[k][t]] + ([v0] if v0 is not None else []))
                v0 = self.copy(cl0, f"vx_rget{t}_i{i}", kvx(voff + t * hv, m), vdst(cl0, t, 1 - h), hv,
                               [vput[m][t], v0])
                v1 = self.copy(cl1, f"vx_rget{t}_i{i}", kvx(voff + t * hv, m), vdst(cl1, t, 1 - h), hv,
                               [vput[m][t], v1])
            for t in range(TPP):
                j = i * TPP + t
                self.tile_ready[(cl0, "K", j)], self.tile_ready[(cl1, "K", j)] = k0, k1
                self.tile_ready[(cl0, "V", j)], self.tile_ready[(cl1, "V", j)] = v0, v1

    # ---- the window -------------------------------------------------------------------------------------------------
    def _consts(self):
        """Every cluster's constants from its memory chip's HBM into L1, before the window (cluster 1 first)."""
        pre = {k: [] for k in self.chips}
        for k in self.chips:
            for c in reversed(range(self.NC)):
                cl = self.chips.index(k) * self.NC + c
                h, nb = self.cst[cl]
                pre[k] = [self.copy(cl, "const_ld", h.address, self.consts[cl], nb, pre[k])]
        return pre

    def build(self):
        R = self.ROOT
        self.tile_ready, self.rope_ld = {}, {}
        for ls in self.ls:
            ls.strict = True
        go = self.setup(extra=self._consts())
        self.dfg.relay_one_consumer_cores = [self.roles["xdma"]]
        puts = self.x_puts(go)
        last_proj, q8s, last = {}, {}, {k: [] for k in self.chips}
        for i in range(self.NP):
            x = {cl: self.stage_x(cl, i, puts[i], [go[self.chip(cl)]] + list(last_proj.get(cl, [])))
                 for cl in range(self.N_CL)}
            own = {}
            for cl in range(self.N_CL):
                q8s[(cl, i)], last_proj[cl], own[cl] = self.stage_proj(cl, i, x[cl])
            self.stage_trade(i, own)
            for cl in range(self.N_CL):
                k = self.chip(cl)
                last[k] += [t for u in range(self.NQ) for t in q8s[(cl, i)][u][1]]
                last[k] += [self.tile_ready[(cl, kind, i * self.TPP + t)] for kind in ("K", "V") for t in range(self.TPP)]
        last = {k: list(dict.fromkeys(v)) for k, v in last.items()}
        ends = [self.probe(k, f"fin{k:02x}", last[k]) for k in self.chips if k != R]
        ends.append(self.probe(R, f"fin{R:02x}", last[R]))
        self.end_window(ends)
        self._checks()

    def _tiles(self, cl):
        """Cluster cl's key tiles (A layout, chunk-pair column order) and value tiles (V^T, A layout) of the whole
        prompt, as kbuf and vbuf hold them."""
        d, m = self.d, self.group[self.chip(cl)]
        kt = [self.lay.to_a(d.k8[m][j * self.bc:(j + 1) * self.bc][:, self.perm], self.mesh).astype(np.int8)
              for j in range(self.NP * self.TPP)]
        vt = [self.lay.to_a(d.v8[m][j * self.bc:(j + 1) * self.bc].T, self.mesh).astype(np.int8)
              for j in range(self.NP * self.TPP)]
        return np.concatenate([t.reshape(-1) for t in kt]), np.concatenate([t.reshape(-1) for t in vt])

    def _checks(self):
        """The last pass's first and last query tile on both clusters of the root and of the fold order's middle chip;
        every key and value tile of the whole prompt on those clusters."""
        R, d, checks = self.ROOT, self.d, []
        far = self.order[len(self.order) // 2]
        i = self.NP - 1
        for k in (R, far):
            for c in range(self.NC):
                cl = self.chips.index(k) * self.NC + c
                h = self.head[cl]
                for u in sorted({0, self.NQ - 1}):
                    r0 = i * self.R + u * BR
                    q8 = self.lay.to_a(d.q8[h][r0:r0 + BR][:, self.perm], self.mesh).astype(np.int8)
                    nm = f"q8_k{k:02x}c{c}u{u}"
                    checks.append((nm, self.stash(nm, k, c, self.pb(cl, i)["q8"][u], q8.size, []), q8.view(np.uint8),
                                   q8.size))
                kt, vt = self._tiles(cl)
                for nm, buf, img in ((f"k8_k{k:02x}c{c}", self.kbuf[cl], kt), (f"v8_k{k:02x}c{c}", self.vbuf[cl], vt)):
                    checks.append((nm, self.stash(nm, k, c, buf, img.size, []), img.view(np.uint8), img.size))
        self.run_checks(checks)


def _app_name(S):
    return (f"Llama-3-8B attention (synthetic weights), fresh {S.T}-token prompt, {len(S.chips)} chips x {S.NC} clusters, "
            f"passes of {S.R}")


if __name__ == "__main__":
    run_driver("llama3 attention stage 1234 workload generator", Stage1234, _app_name)
