#!/usr/bin/env python3
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""Llama-3-8B's attention (synthetic weights) over a fresh prompt on sixteen chiplets, stage 5 alone: every pass's
sweep. Every cluster's query tiles of every pass and its key and value tiles of the whole prompt are in its L1 before
the window (copied from an image in its memory chip's HBM); the window ends when every pass's o~ (o8, A layout: the
sweep's normalisation quantises it) has been copied out of the sweep's buffers into the chip's L3.

THE TASKS of pass i (names without their _k<chip>_cl<c> suffix; u a query tile):

    before the window   const_ld, kt_pre, vt_pre, q8_pre_i<i>u<u>    DM: the constants, every key and value tile, every
                                                                   query tile, HBM -> L1
    S0_start
    s<i>_*              fa_sweep over key tiles 0 .. the pass's last: S^T = K Q^T with the causal mask as the score
                        matmul's INT32 C, the online softmax (each arena's geometry on the xDMA core), O^T += V^T P^T,
                        then o~ = O / l quantised into o8, A layout (libs/blocks/fa_sweep.py)
    ot_out<u>_i<i>      DM: the query tile's o8 into the chip's L3
    -> fin<chip> -> S1_end

Checks: the last pass's first and last query tile's o8 on both clusters of the root and of the fold order's middle
chip, in their chip's L3.

params.hjson: the prompt and its passes (tokens, rows, bc), the weights (wbits, w_chunk_bytes, w_buffers, weight_ring;
staged as the whole block's), the platform and compile switches."""

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
from bingo_kernel_args import INT4_B_LAYOUT_PAIRED, SnaxBingoKernelIdma1dCopyArgs, simd_fp16_lanes  # noqa: E402
from bingo_mem_handle import BingoMemSymbol  # noqa: E402
from driver import run_driver  # noqa: E402
from libs.blocks import FaSweepCfg, fa_sweep, fa_sweep_alloc  # noqa: E402

import chiplet_mc as cm  # noqa: E402
import llama3_data  # noqa: E402

BR = 16                       # query rows a tile: the SIMD's FP16 lanes
QC = 32                       # columns a projection weight chunk
SLC = 128                     # y's slices: a cluster's columns in one slice (the block's wo_slice)
XT = 32768                    # the x multicast's tile, a flag after each


class L1Tiles:
    """fa_sweep's tile source over a cluster's L1-resident key and value tiles: tile j is ready after the task that
    put it there, and nothing is ever released (the tiles stay for the later passes)."""

    def __init__(self, ab, cl):
        self.ab, self.cl = ab, cl

    def fetch(self, kind, j):
        ab = self.ab
        buf = ab.kbuf[self.cl] if kind == "K" else ab.vbuf[self.cl]
        return ab.tile_ready[(self.cl, kind, j)], buf.view(j * ab.TILE_B), None

    def release(self, token, task):
        return None


class Stage5(cm.ChipletBuild):
    """Stage 5 of the attention (the module's docstring). run_driver makes the program in four steps: __init__ (the
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
        self._inputs()

    def kv_cc(self, cl):
        """The k or v chunks cluster cl projects: its chip's half."""
        h = self.kv_half[self.chip(cl)]
        return range(h * self.NKVC, (h + 1) * self.NKVC)

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
        self.C_MASK = (self.NQC + self.NKVC) * self.SC_B
        self.MASK_B = self.bc * BR * 4                  # a mask pattern: the score matmul's INT32 C

    def _buffers(self):
        """Every cluster's L1: the exchange buffer, every key and value tile of the prompt, the constants, o8, y's
        columns and the sweep's buffers (the whole block's, so the plan is the block's)."""
        R = self.R
        self.xch_zero = False
        self.xch_bytes = 2 * self.SLB
        self.xch = {(k, c): self.cx_chip(k).at(c).l1(f"AAAA_{self.TAG}_xch", self.xch_bytes)
                    for k in self.chips for c in range(self.NC)}
        self.p1c = {c: self.cx_chip(self.ROOT).at(c).l1(f"{self.TAG}_p1c", self.xch_bytes) for c in range(self.NC)}
        self.kbuf, self.vbuf, self.consts, self.sweep_bufs = {}, {}, {}, {}
        for cl in range(self.N_CL):
            l1 = self.l1_of(cl)
            self.kbuf[cl] = l1("kt", self.NP * self.TPP * self.TILE_B)     # every key tile of the prompt, A layout
            self.vbuf[cl] = l1("vt", self.NP * self.TPP * self.TILE_B)     # every value tile, V^T in the A layout
            self.consts[cl] = l1("consts", self.cst[cl][1])
            l1("o8", R * self.HD)
            l1("ywo", self.R * self.WOC * 2)
            self.sweep_bufs[cl] = fa_sweep_alloc(lambda name, nb, l1=l1: l1(name, nb), self.sweep_cfg(1))
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

    # ---- one task ----------------------------------------------------------------------------------------------------
    def nd(self, cl, role, name, kernel, args, deps):
        return self.node(self.chip(cl), self.local(cl), self.roles[role], name, kernel, args, list(dict.fromkeys(deps)))

    def copy(self, cl, name, src, dst, nb, deps):
        return self.nd(cl, "dm", name, "__snax_bingo_kernel_idma_1d_copy", SnaxBingoKernelIdma1dCopyArgs(src, dst, nb),
                       deps)

    def sweep_cfg(self, i_plus_1):
        d = self.d
        return FaSweepCfg(bc=self.bc, nt=i_plus_1 * self.TPP, nq=self.NQ, br=BR, k_s=d.k_s, k_o=d.k_o,
                          a_exp=d.a_exp, a_n_f32bits=d.a_n_bits, d_qk=self.HD, d_v=self.HD)

    # ---- the stage's inputs and outputs ------------------------------------------------------------------------------
    def _tiles(self, cl):
        """Cluster cl's key tiles (A layout, chunk-pair column order) and value tiles (V^T, A layout) of the whole
        prompt, as kbuf and vbuf hold them."""
        d, m = self.d, self.group[self.chip(cl)]
        kt = [self.lay.to_a(d.k8[m][j * self.bc:(j + 1) * self.bc][:, self.perm], self.mesh).astype(np.int8)
              for j in range(self.NP * self.TPP)]
        vt = [self.lay.to_a(d.v8[m][j * self.bc:(j + 1) * self.bc].T, self.mesh).astype(np.int8)
              for j in range(self.NP * self.TPP)]
        return np.concatenate([t.reshape(-1) for t in kt]), np.concatenate([t.reshape(-1) for t in vt])

    def _q8_tiles(self, cl):
        """Cluster cl's query tiles of every pass, one after the other (A layout, chunk-pair column order)."""
        d, h = self.d, self.head[cl]
        return np.concatenate([self.lay.to_a(d.q8[h][r0:r0 + BR][:, self.perm], self.mesh).astype(np.int8).reshape(-1)
                               for r0 in range(0, self.T, BR)])

    def _inputs(self):
        """Per cluster its key, value and query tiles in its memory chip's HBM, and the L3 area every pass's o8 is
        copied into: per chip, its clusters' tiles, pass by pass."""
        T = self.TAG
        self.h_in = {}
        for cl in range(self.N_CL):
            kt, vt = self._tiles(cl)
            m = self.mem[self.chip(cl)]
            self.h_in[cl] = self.st.put_hbm(f"{T}_s5in_cl{cl}", np.concatenate([kt, vt, self._q8_tiles(cl)]),
                                           mem_chip=(m >> 4, m & 15))
        self.OTB = BR * self.HD                         # a query tile's o8
        self.st.put_zeros(f"{T}_otout", "uint8_t", self.NP * self.NC * self.NQ * self.OTB)

    def _ot_tile(self, h, r0):
        """Query tile r0's o8 of head h as the sweep leaves it (A layout)."""
        return self.lay.to_a(self.d.ot8[h][r0:r0 + BR], self.mesh).astype(np.int8).reshape(-1).view(np.uint8)

    def mask(self, i):
        """The sweep's mask of pass i: key tile j, query tile u -> the pattern's handle in the cluster's constants."""
        d, first = self.d, i * self.TPP

        def m(cl):
            def f(j, u):
                if j < first:
                    return None
                pi = d.mask_tiles[(j - first, u)]
                return None if pi is None else self.consts[cl].view(self.C_MASK + pi * self.MASK_B)
            return f
        return m

    # ---- the window -------------------------------------------------------------------------------------------------
    def _preloads(self):
        """Every cluster's constants, key, value and query tiles from its memory chip's HBM into L1, before the
        window, one chain a chip (cluster 1 first)."""
        nk, qb = self.NP * self.TPP * self.TILE_B, BR * self.HD
        pre = {k: [] for k in self.chips}
        for k in self.chips:
            for c in reversed(range(self.NC)):
                cl = self.chips.index(k) * self.NC + c
                h, nb = self.cst[cl]
                pre[k] = [self.copy(cl, "const_ld", h.address, self.consts[cl], nb, pre[k])]
        for k in self.chips:
            for c in range(self.NC):
                cl = self.chips.index(k) * self.NC + c
                a = self.h_in[cl].address
                pre[k] = [self.copy(cl, "kt_pre", a, self.kbuf[cl], nk, pre[k])]
                pre[k] = [self.copy(cl, "vt_pre", a + nk, self.vbuf[cl], nk, pre[k])]
                for i in range(self.NP):
                    for u in range(self.NQ):
                        off = 2 * nk + (i * self.NQ + u) * qb
                        pre[k] = [self.copy(cl, f"q8_pre_i{i}u{u}", a + off, self.pb(cl, i)["q8"][u], qb, pre[k])]
        return pre

    def build(self):
        T, R = self.TAG, self.ROOT
        self.tile_ready = {}
        for ls in self.ls:
            ls.strict = True
        go = self.setup(extra=self._preloads())
        self.dfg.relay_one_consumer_cores = [self.roles["xdma"]]
        for cl in range(self.N_CL):
            for kind in ("K", "V"):
                for j in range(self.NP * self.TPP):
                    self.tile_ready[(cl, kind, j)] = go[self.chip(cl)]
        ot8 = int(np.asarray(self.d.s_ot.inv, dtype=np.float32).reshape(()).view(np.uint32))
        prev_sw, xt_rd, last = {}, {}, {k: [] for k in self.chips}
        for i in range(self.NP):
            for cl in range(self.N_CL):
                k, c = self.chip(cl), self.local(cl)
                q8 = {u: (self.pb(cl, i)["q8"][u], [go[k]]) for u in range(self.NQ)}
                sw = fa_sweep(self.at(k, c), self.sweep_cfg(i + 1), self.sweep_bufs[cl], L1Tiles(self, cl), q8,
                              self.roles, prev=prev_sw.get(cl), oacc_free=xt_rd.get(cl), name=f"s{i}_", early=1,
                              mask=self.mask(i)(cl), mask_c=True, ot8=ot8, geom_core="xdma", sm_batch=False)
                outs = [self.copy(cl, f"ot_out{u}_i{i}", sw["xt"][u],
                                  BingoMemSymbol(f"{T}_otout", offset=((i * self.NC + c) * self.NQ + u) * self.OTB),
                                  self.OTB, [sw["tr"][u]]) for u in range(self.NQ)]
                xt_rd[cl] = {u: [outs[u]] for u in range(self.NQ)}
                prev_sw[cl] = sw
                last[k] += outs
        ends = [self.probe(k, f"fin{k:02x}", last[k]) for k in self.chips if k != R]
        ends.append(self.probe(R, f"fin{R:02x}", last[R]))
        self.end_window(ends)
        self._checks()

    def _checks(self):
        """The last pass's first and last query tile's o8 on both clusters of the root and of the fold order's middle
        chip, in their chip's L3."""
        T, R, checks = self.TAG, self.ROOT, []
        far = self.order[len(self.order) // 2]
        i = self.NP - 1
        for k in (R, far):
            for c in range(self.NC):
                cl = self.chips.index(k) * self.NC + c
                h = self.head[cl]
                for u in sorted({0, self.NQ - 1}):
                    r0 = i * self.R + u * BR
                    off = ((i * self.NC + c) * self.NQ + u) * self.OTB
                    checks.append((f"ot_k{k:02x}c{c}u{u}", BingoMemSymbol(f"{T}_otout", offset=off, chip_id=k),
                                   self._ot_tile(h, r0), self.OTB))
        self.run_checks(checks)


def _app_name(S):
    return (f"Llama-3-8B attention (synthetic weights), fresh {S.T}-token prompt, {len(S.chips)} chips x {S.NC} clusters, "
            f"passes of {S.R}")


if __name__ == "__main__":
    run_driver("llama3 attention stage 5 workload generator", Stage5, _app_name)
