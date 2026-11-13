#!/usr/bin/env python3
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""Llama-3-8B's attention (synthetic weights) over a fresh prompt on sixteen chiplets, stages 6-8 alone: every pass's
o8 into its head's place, the root's gather of every head's o8 and its multicast, W_O by output columns and y scattered
into the owners of its row tiles. Every cluster's o8 of every pass (the sweep's normalised, quantised output, A layout)
is in its L1 before the window; the window ends when every chip holds its tiles of y.

W_O BY COLUMNS (the whole block's, workloads/llama3/common/llama3_attn_mc.py). Every head's o8 goes into the root's L3,
the root multicasts all of them, and each cluster's GEMMs make its 128 columns of y over the full depth: no reduce.
Each cluster's columns of a pass go, row tile by row tile, straight into the L3 of the chip that owns the tile (tile q
on the q-th chip) behind a flag there. The root's multicast of a pass goes out before its own gather, each cluster's
first W_O chunk loads right behind its put, the gathers run on the xDMA, and the even and odd passes have a W_O output
buffer each, so pass i's front runs ahead of pass i - 1's y stores.

THE TASKS of pass i (names without their _k<chip>_cl<c> suffix; u a query tile, s a W_O chunk, m a row tile):

    before the window   const_ld, ot_pre_i<i>u<u>                  DM: the constants and every o8 tile, HBM -> L1
    S0_start
    o8cp<u>_i<i>        xDMA: a query tile's o8 into the cluster's o8
    o_put_i<i>          DM: the cluster's o8 into the root's L3, behind its head's flag
    Ld_w<WO0>_i<i>      DM: the first W_O chunk
    o_nwait_i<i>        root DM: every head's flag
    o_bput_i<i>         root DM: the multicast of every head's o8 into every chip's L3 (behind the previous pass's)
    o_get_i<i>          xDMA: every head's o8 into L1 (root: after its wait; others: after o_wait_i<i> on the
                        multicast's flag)
    Ld_w<j>_i<i> -> gwo<s>_i<i>    DM ring load, GEMM: o8 . W_O chunk, 32 columns of y (FP16, D layout)
    y_sc_i<i>m<m>, y_sf_i<i>m<m>   DM: the cluster's columns of row tile m into its owner's L3, then its flag
    -> yt_wait (one a chip, on its tiles' flags) -> S1_end

Checks: the first, a middle and the last tile of y in their owners' L3 (head, middle and tail of each).

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
    INT4_B_LAYOUT_PAIRED, SnaxBingoKernelGemmFullArgs, SnaxBingoKernelIdma1dCopyArgs, SnaxBingoKernelIdma2dCopyArgs,
    SnaxBingoKernelIdmaPutFlaggedArgs, SnaxBingoKernelWaitFlagsArgs, SnaxBingoKernelXdma1dCopyArgs,
    SnaxBingoKernelXdma6dArgs, simd_fp16_lanes)
from bingo_mem_handle import BingoMemSymbol  # noqa: E402
from driver import run_driver  # noqa: E402
from libs.blocks import FaSweepCfg, fa_sweep_alloc  # noqa: E402

import chiplet_mc as cm  # noqa: E402
import llama3_data  # noqa: E402

BLK = 16                      # a GEMM column block (meshCol)
BR = 16                       # query rows a tile: the SIMD's FP16 lanes
QC = 32                       # columns a projection weight chunk
SLC = 128                     # y's slices: a cluster's columns in one slice (the block's wo_slice)
XT = 32768                    # the x multicast's tile, a flag after each
TILE16 = 16 * 16 * 2          # one D-layout FP16 block


class Stage678(cm.ChipletBuild):
    """Stages 6-8 of the attention (the module's docstring). run_driver makes the program in four steps: __init__ (the
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
        self.YT_B = BR * self.H * 2                     # a 16-row tile of y, D layout, FP16
        self.NTL = -(-self.T // BR // len(self.chips))  # the tiles of y a chip owns
        self._stage()
        self.make_streams()
        self.make_rings()
        self._buffers()
        self._inputs()

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
        st.put_zeros(f"{T}_yt", "uint8_t", self.NTL * self.YT_B, preload=True)     # the chip's tiles of y
        st.put_zeros(f"{T}_ytflag", "uint8_t", self.NTL * self.N_CL * 64)          # a cluster's columns of one landed
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
        self.kbuf, self.vbuf, self.consts, self.o8, self.ywo, self.ywo_b = {}, {}, {}, {}, {}, {}
        for cl in range(self.N_CL):
            l1 = self.l1_of(cl)
            self.kbuf[cl] = l1("kt", self.NP * self.TPP * self.TILE_B)     # every key tile of the prompt, A layout
            self.vbuf[cl] = l1("vt", self.NP * self.TPP * self.TILE_B)     # every value tile, V^T in the A layout
            self.consts[cl] = l1("consts", self.cst[cl][1])
            self.o8[cl] = l1("o8", R * self.HD)                            # every head's o8 put comes from here
            self.ywo[cl] = l1("ywo", self.R * self.WOC * 2)                # the cluster's columns of y, even passes
            self.ywo_b[cl] = l1("ywo_b", self.R * self.WOC * 2)            # odd passes
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

    def copy(self, cl, name, src, dst, nb, deps):
        return self.nd(cl, "dm", name, "__snax_bingo_kernel_idma_1d_copy", SnaxBingoKernelIdma1dCopyArgs(src, dst, nb),
                       deps)

    def y_owner(self, q):
        """(chip, its tile index) of 16-row tile q of y."""
        return self.chips[q % len(self.chips)], q // len(self.chips)

    def yt_sym(self, q, off=0, chip_id=None):
        """Tile q of y in its owner's L3 at byte `off` (chip_id: as another chip names it)."""
        return BingoMemSymbol(f"{self.TAG}_yt", offset=self.y_owner(q)[1] * self.YT_B + off, chip_id=chip_id)

    def _yw(self, cl, i):
        """Cluster cl's W_O output buffer for pass i: one for the even passes, one for the odd."""
        return self.ywo_b[cl] if i % 2 else self.ywo[cl]

    # ---- the stage's inputs ------------------------------------------------------------------------------------------
    def _ot_tile(self, h, r0):
        """Query tile r0's o8 of head h as the sweep leaves it (A layout)."""
        return self.lay.to_a(self.d.ot8[h][r0:r0 + BR], self.mesh).astype(np.int8).reshape(-1).view(np.uint8)

    def _inputs(self):
        """Per cluster its head's o8 of every pass, tile by tile, in its memory chip's HBM."""
        T = self.TAG
        self.OTB = BR * self.HD                         # a query tile's o8
        self.h_in = {}
        for cl in range(self.N_CL):
            k, h = self.chip(cl), self.head[cl]
            m = self.mem[k]
            ot = np.concatenate([self._ot_tile(h, r0) for r0 in range(0, self.T, BR)])     # tile by tile, pass by pass
            self.h_in[cl] = self.st.put_hbm(f"{T}_s678in_cl{cl}", ot.view(np.int8), mem_chip=(m >> 4, m & 15))

    # ---- stage 6: o8 into its head's place ---------------------------------------------------------------------------
    def stage_post(self, cl, i, ot):
        """Pass i's o8 tiles (ot[u], in L1) into the cluster's o8 on the xDMA, each after the previous pass's put of
        that o8. Returns the copies."""
        HD = self.HD
        cv = []
        for u in range(self.NQ):
            cv.append(self.nd(cl, "xdma", f"o8cp{u}_i{i}", "__snax_bingo_kernel_xdma_1d_copy",
                              SnaxBingoKernelXdma1dCopyArgs(ot[u], self.o8[cl].view(u * BR * HD), BR * HD),
                              [self.go[self.chip(cl)]] + self._o8_war.get(cl, [])))
        return cv

    # ---- stage 7: every head's o8 on every cluster --------------------------------------------------------------------
    def _o8_puts(self, i, post_last):
        """Every cluster's o8 into the root's landing area of pass i, behind its head's flag there; the root's wait for
        every flag. Returns the wait."""
        T, R, HD, NH = self.TAG, self.ROOT, self.HD, self.d.NH
        OB, base = self.R * HD, i * self.XB
        on_root = lambda sym, k: sym if k == R else BingoMemSymbol(sym.symbol_name, offset=sym.offset, chip_id=R)  # noqa
        rcl = [cl for cl in range(self.N_CL) if self.chip(cl) == R]
        puts = {}
        for cl in range(self.N_CL):
            k, h = self.chip(cl), self.head[cl]
            puts[cl] = self.nd(cl, "dm", f"o_put_i{i}", "__snax_bingo_kernel_idma_put_flagged",
                               SnaxBingoKernelIdmaPutFlaggedArgs(
                                   self.o8[cl], on_root(self.xland(base + h * OB), k), OB,
                                   on_root(BingoMemSymbol(f"{T}_oflag", offset=(i * NH + h) * 64), k), value=1),
                               list(post_last[cl]) + [self.go[R] for _ in rcl])
            self._o8_war[cl] = [puts[cl]]
        self.first_ld = {cl: self._load(cl, i, self.WO0) for cl in range(self.N_CL)}
        w = self.node(R, 0, self.roles["dm"], f"o_nwait_i{i}", "__snax_bingo_kernel_wait_flags",
                      SnaxBingoKernelWaitFlagsArgs(BingoMemSymbol(f"{T}_oflag", offset=i * NH * 64), NH, 1),
                      [puts[cl] for cl in rcl])
        for cl in range(self.N_CL):
            if self.chip(cl) != R:
                self.dfg.bingo_add_order_edge(puts[cl], w)
        return w

    def _gather(self, cl, i, deps):
        """Every head's o8 of pass i from this chip's landing area into the cluster's L1 (A layout [R, H]: head h's
        [R, 128] to columns 128 h of every row tile), one xDMA pass of 64-byte beats."""
        HD, NH = self.HD, self.d.NH
        OB, run = self.R * HD, HD // 4 * 64              # a head's o8; a head's row tile, contiguous on both sides
        return self.nd(cl, "xdma", f"o_get_i{i}", "__snax_bingo_kernel_xdma_6d",
                       SnaxBingoKernelXdma6dArgs(self.xland(i * self.XB), self.pb(cl, i)["ofull"], 8, 8,
                                                 [64, run, OB], [run // 64, self.R // 16, NH],
                                                 [64, self.H // 4 * 64, run], [run // 64, self.R // 16, NH]),
                       list(deps))

    def stage_o8(self, i, post_last):
        """Stage 7 of pass i: the o8 puts and the root's wait, its multicast (behind the previous pass's), then the
        gathers into every cluster's L1 (the root's after its wait, the others' after the multicast's flag). Returns
        {cluster: its gather}, the multicast."""
        T, R, NH = self.TAG, self.ROOT, self.d.NH
        OB, base = self.R * self.HD, i * self.XB
        w = self._o8_puts(i, post_last)
        prev = [self._o_mc_last] if self._o_mc_last is not None else []
        self._o_mc_last = mc = self.broadcast_put(
            R, 0, f"o_bput_i{i}", SnaxBingoKernelIdmaPutFlaggedArgs(
                self.xland(base), self.xland(base), NH * OB, BingoMemSymbol(f"{T}_omflag", offset=i * 64), value=1,
                bcast=True), [w] + prev)
        gets = {cl: self._gather(cl, i, [w]) for cl in range(self.N_CL) if self.chip(cl) == R}
        for cl in range(self.N_CL):
            if self.chip(cl) == R:
                continue
            ow = self.nd(cl, "dm", f"o_wait_i{i}", "__snax_bingo_kernel_wait_flags",
                         SnaxBingoKernelWaitFlagsArgs(BingoMemSymbol(f"{T}_omflag", offset=i * 64), 1, 1), [])
            self.dfg.bingo_add_order_edge(mc, ow)
            gets[cl] = self._gather(cl, i, [ow])
        return gets, mc

    # ---- stage 8: W_O by columns and y to its owners --------------------------------------------------------------------
    def stage_wo(self, i, gets):
        """Each cluster's W_O GEMMs of pass i over the full depth, 32 columns each (its first chunk loaded behind its
        put), then its columns scattered to the row tiles' owners."""
        GB = self.R * 32 * 2                              # one GEMM's [R, 32] of y, FP16, D layout
        for cl in range(self.N_CL):
            k, gms = self.chip(cl), []
            for s in range(self.NWS):
                ld, si, slab = self.first_ld[cl] if s == 0 else self._load(cl, i, self.WO0 + s)
                gm = self.gmm(cl, f"gwo{s}_i{i}", self.pb(cl, i)["ofull"], slab, self._yw(cl, i).view(s * GB),
                              self.R // 16, self.H // 4, 32 // BLK, self.d.k_wo,
                              [ld, gets[cl]] + self._ywo_war.get((cl, s, i % 2), []))
                self.ls[cl].release(si, [gm])
                gms.append(gm)
            self._y_scatter_stores(cl, i, k, gms)

    def _y_scatter_stores(self, cl, i, k, gms):
        """Cluster cl's [R, 128] of pass i (the W_O GEMMs' [R, 32] D blocks one after the other), row tile by row tile
        into the tile's owner: the GEMMs' 2 column tiles of that row tile, 4 KiB apart, as 8 consecutive column tiles at
        the cluster's place in the tile's row (D layout); then a 64-byte put that raises the cluster's flag there (the
        in-order DM core runs it after the copy)."""
        T, GB = self.TAG, self.R * 32 * 2
        mine, flags = [], []
        for m in range(self.R // BR):
            q = i * (self.R // BR) + m
            own, lt = self.y_owner(q)
            far = None if own == k else own
            cp = self.nd(cl, "dm", f"y_sc_i{i}m{m}", "__snax_bingo_kernel_idma_2d_copy",
                         SnaxBingoKernelIdma2dCopyArgs(self._yw(cl, i).view(m * 2 * TILE16),
                                                       self.yt_sym(q, cl * self.WOC * BR * 2, chip_id=far),
                                                       2 * TILE16, GB, 2 * TILE16, self.NWS),
                         list(gms))
            fl = self.nd(cl, "dm", f"y_sf_i{i}m{m}", "__snax_bingo_kernel_idma_put_flagged",
                         SnaxBingoKernelIdmaPutFlaggedArgs(
                             self._yw(cl, i), BingoMemSymbol(f"{T}_ydum", offset=cl * 64, chip_id=far), 64,
                             BingoMemSymbol(f"{T}_ytflag", offset=(lt * self.N_CL + cl) * 64, chip_id=far), value=1),
                         [cp])
            self._yt_puts.setdefault(own, []).append(fl)
            mine.append(cp)
            flags.append(fl)
        for s in range(self.NWS):
            # the buffer's readers: the copies, and the flag puts (their 64-byte payload is read from it)
            self._ywo_war[(cl, s, i % 2)] = list(mine) + flags

    def _y_waits(self):
        """On each chip's cluster 0, one wait on its tiles' flags (every cluster's columns of each)."""
        waits = []
        for k in self.chips:
            puts = self._yt_puts.get(k, [])
            w = self.node(k, 0, self.roles["dm"], "yt_wait", "__snax_bingo_kernel_wait_flags",
                          SnaxBingoKernelWaitFlagsArgs(BingoMemSymbol(f"{self.TAG}_ytflag"), len(puts), 1),
                          [t for t in puts if t.assigned_chiplet_id == k])
            for t in puts:
                if t.assigned_chiplet_id != k:
                    self.dfg.bingo_add_order_edge(t, w)
            waits.append(w)
        return waits

    # ---- the window -------------------------------------------------------------------------------------------------
    def _preloads(self):
        """Every cluster's constants and its o8 tiles of every pass from its memory chip's HBM into L1, before the
        window, one chain a chip (the constants cluster 1 first). Returns them and {(cl, pass, tile): L1 buffer}."""
        pre = {k: [] for k in self.chips}
        for k in self.chips:
            for c in reversed(range(self.NC)):
                cl = self.chips.index(k) * self.NC + c
                h, nb = self.cst[cl]
                pre[k] = [self.copy(cl, "const_ld", h.address, self.consts[cl], nb, pre[k])]
        ot_buf = {}
        for k in self.chips:
            for c in range(self.NC):
                cl = self.chips.index(k) * self.NC + c
                a = self.h_in[cl].address
                for i in range(self.NP):
                    for u in range(self.NQ):
                        buf = self.cx_chip(k).at(c).l1(f"{self.TAG}_otp{u}_i{i}", self.OTB)
                        pre[k] = [self.copy(cl, f"ot_pre_i{i}u{u}", a + (i * self.NQ + u) * self.OTB, buf, self.OTB,
                                            pre[k])]
                        ot_buf[(cl, i, u)] = buf
        return pre, ot_buf

    def build(self):
        self._o8_war, self._ywo_war, self._yt_puts, self._o_mc_last = {}, {}, {}, None
        for ls in self.ls:
            ls.strict = True
        pre, ot_buf = self._preloads()
        self.go = self.setup(extra=pre)
        self.dfg.relay_one_consumer_cores = [self.roles["xdma"]]
        for i in range(self.NP):
            post_last = {cl: self.stage_post(cl, i, [ot_buf[(cl, i, u)] for u in range(self.NQ)])
                         for cl in range(self.N_CL)}
            gets, _ = self.stage_o8(i, post_last)
            self.stage_wo(i, gets)
        self.end_window(self._y_waits())
        self._checks()

    def _checks(self):
        """The first, a middle and the last tile of y in their owners' L3: head, middle and tail of each."""
        n, checks = 2048, []
        nt = self.NP * self.R // BR
        for q in (0, nt // 2, nt - 1):
            img = self.lay.to_d(self.gold_y[q * BR:(q + 1) * BR], self.mesh).astype(np.float16).reshape(-1).view(np.uint8)
            for nm, off in (("a", 0), ("m", self.YT_B // 2 - n // 2), ("b", self.YT_B - n)):
                checks.append((f"yt_q{q}{nm}", self.yt_sym(q, off, chip_id=self.y_owner(q)[0]), img[off:off + n], n))
        self.run_checks(checks)


def _app_name(S):
    return (f"Llama-3-8B attention (synthetic weights), fresh {S.T}-token prompt, {len(S.chips)} chips x {S.NC} clusters, "
            f"passes of {S.R}")


if __name__ == "__main__":
    run_driver("llama3 attention stage 678 workload generator", Stage678, _app_name)
