#!/usr/bin/env python3
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""Llama-3-8B's FFN (synthetic weights), stages 4-5 of one pass of the 256-token prefill on sixteen chiplets alone: the
down slices over the weight stream, the ChainGathers into the root and the stores.

The chip's A8 (both clusters' columns) is in every cluster's L1 before the window (copied from an image of it in its
memory chip's HBM). A chip's two clusters split the down projection by output columns: each takes the chip's
intermediate columns for its half of the outputs, slice by slice as its weight chunks land (the first slice one GEMM per
post block), into an output slot of its symmetric buffer, so a slot holds the chip's whole partial; one ChainGather a
group of slices through every chip, each hop's xDMA adding its partial in transit, folds them into the root's fold slot,
and the root stores each group to Y_raw in its L3 a gather later. The window ends with the last stores. Checks: Y_raw
(its first and last 2 KiB) against the golden of the device's arithmetic in the fold order.

params.hjson: the whole block's mapping (model, rows, the columns a cluster, the down slices and their reduce, the
weight ring) for one pass (tokens = rows), the platform and compile switches."""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(__file__).split("/target/sw/host/")[0]
sys.path.insert(0, os.path.join(REPO, "util", "sim", "multichip"))
sys.path.insert(0, os.path.join(REPO, "util", "sim", "common"))
sys.path.insert(0, os.path.join(REPO, "target", "sw", "host", "runtime", "libbingo", "mini_compiler"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "llama3", "common"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "common"))
import _bingo_paths  # noqa: F401,E402
from bingo_kernel_args import (  # noqa: E402
    INT4_B_LAYOUT_PAIRED, SnaxBingoKernelGemmFullArgs, SnaxBingoKernelIdma1dCopyArgs,
    SnaxBingoKernelXdmaChainGatherArgs)
from bingo_mem_handle import BingoMemSymbol  # noqa: E402
from driver import run_driver  # noqa: E402
from libs.blocks import chain_gather  # noqa: E402

import chiplet_mc as cm  # noqa: E402
import ffn_data  # noqa: E402
import llama3_data  # noqa: E402

CACHE = os.path.join(REPO, "target", "sim", "automation", "sweep", "llm_blocks", "ffn", "cache")
CHK = 2048                                  # bytes a checked window
TILE = 16 * 16 * 2                         # one D-layout FP16 tile
BLK = 16                                   # a GEMM column block (meshCol)
POST_BLOCKS = 4                            # the first down slice's GEMMs, one per row block of A8


def checked(chips):
    """The clusters whose L1 the checks read: the root's two, the middle chip's cluster 0, the last chip's cluster 1."""
    return [(chips[0], 0), (chips[0], 1), (chips[len(chips) // 2], 0), (chips[-1], 1)]


def gather_check(S, name, items):
    """items [(chip, cluster, L1 source, golden)]: after the end stamp each cluster's DM core copies the first CHK
    bytes of its source into its own slice of one L3 symbol on the root, whose host compares the whole symbol in one
    check (the first bad offset // CHK names the item). Returns the check for run_checks."""
    sym = f"{S.TAG}_chk_{name}"
    S.st.put_zeros(sym, "uint8_t", len(items) * CHK)
    for j, (k, c, src, _) in enumerate(items):
        dst = BingoMemSymbol(sym, offset=j * CHK, chip_id=S.ROOT)
        S.check_copies.append(S.node(k, c, S.roles["dm"], f"chk_{name}_{j}", "__snax_bingo_kernel_idma_1d_copy",
                                     SnaxBingoKernelIdma1dCopyArgs(src, dst, CHK), [S.end]))
    gold = np.concatenate([np.ascontiguousarray(g).view(np.uint8).reshape(-1)[:CHK] for *_, g in items])
    return (name, BingoMemSymbol(sym, chip_id=S.ROOT), gold, len(items) * CHK)


def pack_b(d, q):
    """An INT4 weight chunk [K, N] in the paired layout (blocks of 32 columns, layout.py to_b_pairs) the GEMMs' B reader
    walks 64 bytes a pass."""
    if q.shape[1] % (2 * d.mesh[2]):
        raise ValueError(f"a weight chunk of {q.shape[1]} columns is not whole pairs of {d.mesh[2]}-column tiles")
    return d.layout.pack_int4(d.layout.to_b_pairs(q, d.mesh))


def split_columns(S, p):
    """{g: (first column, columns)} of the intermediate columns, in whole GEMM column blocks, the root chip's clusters
    taking the smaller share (in the whole block its cluster 0 also runs every gather and stores Y); params root_cols:
    exactly that many on the root's clusters (one number for both, or one each). Sets S.kcols {chip: (first column,
    columns)}: a chip's two clusters' columns, which follow each other."""
    rc = p.get("root_cols", 0)
    rc = [int(v) for v in rc] if isinstance(rc, (list, tuple)) else [int(rc)] * S.NC
    cols = S.split_columns(S.I, last=S.gs(S.ROOT), unit=BLK,
                           fixed=dict(zip(S.gs(S.ROOT), rc)) if any(rc) else None)
    S.kcols = {}
    for k in S.chips:
        (a0, an), (b0, bn) = (cols[g] for g in S.gs(k))
        if b0 != a0 + an:
            raise ValueError(f"chip {k:02x}'s columns are not contiguous")
        S.kcols[k] = (a0, an + bn)
    return cols


class Stage45(cm.ChipletBuild):
    """Stages 4-5 of the FFN prefill block (../full) for one pass, alone. The block runs, for each pass of `rows`
    tokens:

        stage 0     X from HBM into every cluster                                     ../stage0
        stages 1-3  gate|up GEMMs, dequantise and SwiGLU, quantise, the A8 exchange   ../stage123
        stages 4-5  down GEMMs, the ChainGathers into the root, the stores of Y       this app

    Stage 3's result, each chip's A8 in both its clusters' L1, is loaded before the window instead, so the window
    (S0_start .. S1_end) is these stages' time. run_driver makes the program in three steps: __init__ (the mapping,
    the data, the golden and the memory), build() (the task graph: every task and edge) and finish() (the buffer-hazard
    check, then the compile into the C header and the data image, in which the compiler orders every core's tasks and
    reuses a repeated GEMM configuration)."""

    TAG = "pf"      # prefill FFN, as in the whole block: the prefix of the symbols, task names and printed lines

    def __init__(self, args, p, plat, hw):
        """The mapping, the data, the golden and the memory. ChipletBuild.__init__ first: the chips, each one's memory
        chip and its push engine there, the fold order, the empty DFG, the data staging (self.st) and the stamp buffer.
        Then: each chip's intermediate columns (split_columns) and each cluster's output columns as down slices
        (_dn_out_split); the golden Y, added in the fold order the ChainGathers add in; _stage(): each chip's A8 and
        every cluster's down chunks into HBM, Y_raw's place in the root's L3; make_streams(): each cluster's L1 weight
        slabs; make_rings(): each chip's weight rings in its L3 and its host's prefetcher task
        (weight_prefetch_k<chip>); _buffers(): the output slots of every cluster's symmetric buffer, the root's fold
        slots, every cluster's A8. The prefetchers are the only tasks made here."""
        super().__init__(args, p, plat, hw)
        d = self.d = ffn_data.FFNData(CACHE, p["model"], wbits=self.wbits, tokens=self.T)
        R = self.ROWS = int(p["rows"])
        if self.T != R or R % (16 * POST_BLOCKS):
            raise ValueError(f"params tokens={self.T} rows={R}: one pass (tokens = rows), rows whole blocks of 16")
        self.MT, self.D, self.I = R // 16, d.H, d.I
        self.YB = R * self.D * 2                # Y_raw of the pass, FP16
        # params dn_cols: output columns a down slice (whole pairs of column tiles)
        self.DNC = int(p.get("dn_cols", 512))
        if self.DNC % (2 * BLK) or self.D % self.DNC or (self.D // self.DNC) % self.NC:
            raise ValueError(f"params dn_cols={self.DNC}: whole pairs of column tiles, an even number of slices")
        self.NDN = self.D // self.DNC
        self.cols = split_columns(self, p)
        h = self.NDN // self.NC
        self.dn_sl = self._dn_out_split(bool(p.get("dn_balance", False)))
        # params chain_group: one ChainGather carries this many consecutive slices of a cluster (back to back in its
        # output slots) into a fold slot of the root; out_slots: the output slots of each cluster's symmetric buffer the
        # slices go through (slice j into slot j mod out_slots), so the down GEMMs run that many slices ahead of the
        # serial gathers
        self.chain_group = GS = int(p.get("chain_group", 1))
        self.out_slots = int(p.get("out_slots", 2))
        if self.out_slots < 2 or h % GS or self.out_slots % GS:
            raise ValueError(f"params chain_group={GS} out_slots={self.out_slots}: at least 2 slots, the group "
                             f"dividing them and the {h} slices a cluster")
        # params chain_alt: cluster 1's ChainGathers run the chips in the reverse order, so a gather's data follows the
        # previous gather's finish back along the ring instead of meeting it mid-chain
        self.chain_alt = bool(p.get("chain_alt", False))
        self.gold = d.golden([self.kcols[k] for k in self.chips], {k: [i] for i, k in enumerate(self.chips)},
                             self.fold_order())
        if self.chain_alt:
            loc, add16 = self.gold["loc"], d.fp.add16
            rev = self.order[1:][::-1] + self.order[:1]
            y = loc[rev[0]]
            for k in rev[1:]:
                y = add16(y, loc[k])
            yr = self.gold["y_raw"].copy()
            for c0, w in self.dn_sl[1]:
                yr[:, c0:c0 + w] = y[:, c0:c0 + w]
            self.gold["y_raw"], self.gold["y16"] = yr, d.fp.mul16(yr, d.dn_f[None, :])
        for line in d.report(self.gold["y16"]):
            print(f"[{self.TAG}] {line}")
        print(f"[{self.TAG}] one pass of {R} tokens; fold order {[f'{k:02x}' for k in self.order]}; {h} slices a "
              f"cluster, {h // GS} gathers a cluster")
        self._stage()
        self.make_streams()
        self.make_rings()
        self._buffers()

    def _dn_out_split(self, balance):
        """{cluster: [(first output column, columns)]}, NDN / 2 slices each. Default: the two halves of the outputs in
        dn_cols-column slices. params dn_balance: cluster 0's share of the outputs moves (whole pairs of column tiles,
        the same on every chip, since a gather takes the same slice from every chip) to where the busiest cluster's
        array steps (gate|up of its own columns + down of its chip's) are fewest, and each share is cut into slices as
        even as possible."""
        h = self.NDN // self.NC
        if not balance:
            return {c: [(j * self.DNC, self.DNC) for j in range(c * h, (c + 1) * h)] for c in range(self.NC)}
        mt, kd, u = self.MT, self.D // 4, 2 * BLK

        def worst(S):
            w = 0
            for k in self.chips:
                kn = self.kcols[k][1]
                for c, g in enumerate(self.gs(k)):
                    out = S if c == 0 else self.D - S
                    w = max(w, mt * kd * 2 * self.cols[g][1] // BLK + mt * (kn // 4) * out // BLK)
            return w
        S = min(range(u, self.D, u), key=lambda v: (worst(v), abs(v - self.D // 2)))

        def cut(o0, width):
            nb = width // u
            sizes = [nb // h + (1 if i < nb % h else 0) for i in range(h)]
            out, c0 = [], o0
            for b in sizes:
                out.append((c0, b * u))
                c0 += b * u
            return out
        print(f"[{self.TAG}] dn_balance: cluster 0 takes outputs 0..{S - 1}, cluster 1 {S}..{self.D - 1}; busiest "
              f"cluster {worst(S):,} array steps (even halves: {worst(self.D // 2):,})")
        return {0: cut(0, S), 1: cut(S, self.D - S)}

    def _stage(self):
        """Each chip's A8 (A layout) in its memory chip's HBM (the image its clusters' L1 is loaded from); per cluster
        its down chunks (the chip's columns of its slices, paired INT4) in its memory chip's HBM; Y_raw in the root's
        L3."""
        st, d, T, lay = self.st, self.d, self.TAG, self.d.layout
        self.h_a8 = {}
        for k in self.chips:
            kc0, kn = self.kcols[k]
            m = self.mem[k]
            a8 = lay.to_a(d.a8[:self.ROWS, kc0:kc0 + kn], d.mesh).astype(np.int8)
            self.h_a8[k] = st.put_hbm(f"{T}_s45a8_k{k:02x}", a8.reshape(-1), mem_chip=(m >> 4, m & 15))
        st.put_zeros(f"{T}_y", "uint8_t", self.YB, preload=True)
        self.img, self.chunk_off = {}, {}
        for g in self.cols:
            k = self.chip(g)
            kc0, kn = self.kcols[k]
            parts = [pack_b(d, d.dn_q[kc0:kc0 + kn, o0:o0 + w]) for o0, w in self.dn_sl[self.c(g)]]
            offs, off = [], 0
            for b in parts:
                offs.append((off, b.size))
                off += b.size
            self.img[g] = st.put_hbm(f"{T}_w_g{g}", np.concatenate(parts).astype(np.int8),
                                     mem_chip=(self.mem[k] >> 4, self.mem[k] & 15))
            self.chunk_off[g] = offs

    def _buffers(self):
        """Every cluster's symmetric buffer (out_slots output slots of a slice), the root clusters' fold buffers (two
        fold slots of chain_group slices), and every cluster's copy of its chip's A8."""
        T, R = self.TAG, self.ROOT
        self.SLICE = self.ROWS * max(w for v in self.dn_sl.values() for _, w in v) * 2
        self.sym_bytes = self.out_slots * self.SLICE
        self.sym = {(k, c): self.cx_chip(k).at(c).l1(f"AAAA_{T}_sym", self.sym_bytes)
                    for k in self.chips for c in range(self.NC)}
        self.p1c = {c: self.cx_chip(R).at(c).l1(f"{T}_p1c", 2 * self.chain_group * self.SLICE) for c in range(self.NC)}
        self.a8 = {g: self.cx_chip(self.chip(g)).at(self.c(g)).l1(f"{T}_a8_0", self.ROWS * self.kcols[self.chip(g)][1])
                   for g in range(self.G)}

    def pre_window(self):
        """{chip: [task]} the start waits for: every cluster's copy of its chip's A8 from its memory chip's HBM."""
        pre = {k: [] for k in self.chips}
        self.a8_pre = {}
        for g in range(self.G):
            k, c = self.chip(g), self.c(g)
            self.a8_pre[g] = self.node(k, c, self.roles["dm"], f"a8_pre_g{g}", "__snax_bingo_kernel_idma_1d_copy",
                                       SnaxBingoKernelIdma1dCopyArgs(self.h_a8[k].address, self.a8[g],
                                                                     self.ROWS * self.kcols[k][1]), [])
            pre[k].append(self.a8_pre[g])
        return pre

    def _load(self, g, jl):
        """Down chunk jl of cluster g into the next slab: a ring load behind the chunk's push. Returns (the load, the
        slab index, the slab)."""
        k, c = self.chip(g), self.c(g)
        ls = self.ls[g]
        off, nb = self.chunk_off[g][jl]
        si, slab, war = ls.take(nb)
        ld, _ = ls.rings.load_chunk(lambda n, core, kern, args, after: self.node(k, c, core, n, kern, args, after), c,
                                    f"Ld_c{jl}_i0_g{g}", src=self.img[g].address + off, nbytes=nb, slab=slab, war=war,
                                    dm=self.roles["dm"])
        return ld, si, slab

    def build(self):
        """The task graph (task names without their _k<chip> suffix; g a cluster, c its index on its chip, j a down
        slice, s a fold slot); every cluster runs the same chain on its own output columns:

          before the window     a8_pre_g<g>                        DM: the chip's A8, HBM -> L1
                                     |
                                 S0_start  ---------->  weight_prefetch_k<chip>   host: its memory chip pushes the
                                     :                                            chunks into the L3 ring
          every slice j         Ld_c<j>_i0_g<g>                    DM: the chunk, ring slot -> L1 slab, once landed
                                     |
                                dn_c<j>_i0[b<b>]_g<g>              GEMM: A8 x the chunk, the chip's FP16 partial of
                                     |                             the slice, into output slot j % out_slots
          every chain_group slices, per c, on the root chip:
                                y_chain_j<j>                       cluster c's xDMA: one ChainGather through every
                                     |                             chip, each hop adding its partial, into fold slot s
                                y_store_j<j>                       DM: fold slot s -> Y_raw in the root's L3
                                     |
                   the last stores  ->  S1_end  ->  pf_report  ->  Check_pf_y_i0_a  ->  Check_pf_y_i0_b

        The first slice's GEMM is one a post block of A8 rows (POST_BLOCKS), the others one GEMM each. A down GEMM
        waits for the gather that last read its output slot; a gather waits for its group's down GEMMs on every chip,
        the previous gather and the stores that last read its fold slot. A group's stores are made after the next
        group's gathers, so the root's DM core stores one gather behind. After the checks every cluster's sym_hold keeps
        its symmetric buffer live to the end; check_hazards() (in finish) proves the reused slots safe."""
        R, dm, xdma, gemm = self.ROOT, self.roles["dm"], self.roles["xdma"], self.roles["gemm"]
        # dep-tag relays of the gathers go to the SIMD cores first, idle in the down phase, each serving that one
        # gather, so a gather waits for its own slices' GEMMs only
        self.dfg.relay_prefer_cores = {xdma: [self.roles["simd"]]}
        self.dfg.relay_one_consumer_cores = [xdma]
        # params gather_relay_only: a dep-tag cell of the gathers over budget relays its leftover edges instead of
        # ordering them, so no down GEMM waits for an earlier gather
        if self.p.get("gather_relay_only", False):
            self.dfg.relay_not_order_cores = [xdma]
        # the symmetric buffers are not zeroed: every slot is written whole by its down GEMMs before it is read
        self.sym_zero = False
        self.setup(extra=self.pre_window())
        SL, others, GS = self.SLICE, self.order[1:], self.chain_group
        h = self.NDN // self.NC
        p1x = {c: f"((uint64_t)({self.p1c[c].get_c_var_name()}))" for c in range(self.NC)}
        sym_reader, p1_reader, pending, last_st = {}, {}, [], {}
        self.dn, self.gathers, self.stores = {}, {}, {}
        last_red = None

        def store(c, j, red, s):
            # the group's slices from the root's fold slot s into Y_raw in L3 (slice by slice, a slice's rows x w at its
            # first column x rows): one copy when the slices fill their fold slots and their columns follow each other
            sl = [self.dn_sl[c][j - c * h + n] for n in range(GS)]
            whole = all(self.ROWS * w * 2 == SL for _, w in sl[:-1]) and \
                all(sl[n][0] == sl[n - 1][0] + sl[n - 1][1] for n in range(1, GS))
            runs = [(0, sl[0][0], (GS - 1) * SL + self.ROWS * sl[-1][1] * 2)] if whole else \
                [(n * SL, o0, self.ROWS * w * 2) for n, (o0, w) in enumerate(sl)]
            sts = [self.node(R, c, dm, f"y_store_j{j}" if n == 0 else f"y_store_j{j}n{n}",
                             "__snax_bingo_kernel_idma_1d_copy",
                             SnaxBingoKernelIdma1dCopyArgs(self.p1c[c].view(s * GS * SL + off),
                                                           BingoMemSymbol(f"{self.TAG}_y", offset=o0 * self.ROWS * 2),
                                                           nb), [red])
                   for n, (off, o0, nb) in enumerate(runs)]
            last_st[c] = p1_reader[(c, s)] = self.stores[(c, (j - c * h) // GS)] = sts

        lds = {g: self._load(g, 0) for g in range(self.G)}
        prev_dn = {}
        for jl in range(h):
            so = jl % self.out_slots            # the clusters' output slot
            dn = {}
            for g in range(self.G):
                k, c = self.chip(g), self.c(g)
                ld, si, slab = lds[g]
                kn = self.kcols[k][1]
                nt = self.dn_sl[c][jl][1] // BLK    # this slice's column tiles
                kargs = dict(input_B_addr=slab, input_C_addr=0, K=kn // 4, N=nt, array_shape_idx=0, transpose_A=0,
                             transpose_B=0, accumPrevC=0, int32tofp16_enable=1, int4_b_enable=INT4_B_LAYOUT_PAIRED,
                             d_shift=self.d.k_dn)
                war = sym_reader.get((k, c, so), [])
                if jl == 0:                     # one GEMM per post block of A8 rows, so the first slice starts sooner
                    mb = self.MT // POST_BLOCKS
                    dn[g] = [self.node(k, c, gemm, f"dn_c{c * h}_i0b{b}_g{g}", "__snax_bingo_kernel_gemm_full",
                                       SnaxBingoKernelGemmFullArgs(
                                           input_A_addr=self.a8[g].view(b * mb * 16 * kn),
                                           output_D_addr=self.sym[(k, c)].view(so * SL + b * mb * nt * TILE), M=mb,
                                           **kargs),
                                       [ld, self.a8_pre[g]] + war)
                             for b in range(POST_BLOCKS)]
                else:
                    dn[g] = [self.node(k, c, gemm, f"dn_c{c * h + jl}_i0_g{g}", "__snax_bingo_kernel_gemm_full",
                                       SnaxBingoKernelGemmFullArgs(input_A_addr=self.a8[g],
                                                                   output_D_addr=self.sym[(k, c)].view(so * SL),
                                                                   M=self.MT, **kargs),
                                       [ld, self.a8_pre[g]] + war)]
                self.dn[(g, jl)] = dn[g][-1]
                self.ls[g].release(si, dn[g])   # every GEMM that reads the slab
                if jl % GS:
                    # a gather waits for its group's last down GEMM on each chip only: each slice's GEMMs follow the
                    # previous slice's (order edges, no dependency tag), so the last implies the group
                    for p in prev_dn[g]:
                        self.dfg.bingo_add_order_edge(p, dn[g][0])
                prev_dn[g] = dn[g]
            if jl + 1 < h:
                lds = {g: self._load(g, jl + 1) for g in range(self.G)}
            while len(pending) > self.NC:       # a gather's stores follow the next gather (two fold slots)
                store(*pending.pop(0))
            if (jl + 1) % GS:
                continue
            j0 = jl + 1 - GS                    # the group's gathers: its slices' output slots into fold slot s
            s, so0 = (j0 // GS) % 2, j0 % self.out_slots
            for c in range(self.NC):
                j = c * h + j0
                ordr = others[::-1] if self.chain_alt and c == 1 else others
                ga = SnaxBingoKernelXdmaChainGatherArgs.fold_into(
                    cm.remote, R, c, ordr, src=self.sx(c, so0 * SL), dst=f"({p1x[c]} + {s * GS * SL})",
                    size=(GS - 1) * SL + self.ROWS * self.dn_sl[c][jl][1] * 2,
                    junction=self.jct["Elementwise"], jct_csr0=cm.EW_F16_ADD, holds=[self.p1c[c]])
                ins = [t for g in range(self.G) if self.c(g) == c for t in dn[g]]
                after = ins + p1_reader.get((c, s), []) + ([last_red] if last_red is not None else [])
                red = chain_gather(lambda n, kernel, a, deps, c=c: self.node(R, c, xdma, n, kernel, a, deps),
                                   f"y_chain_j{j}", ga, after)
                self.gathers[(c, j0 // GS)] = last_red = red
                for k in self.chips:
                    for o in range(so0, so0 + GS):
                        sym_reader[(k, c, o)] = [red]
                pending.append((c, j, red, s))
        while pending:
            store(*pending.pop(0))
        self.end_window([t for c in sorted(last_st) for t in last_st[c]])
        y = self.gold["y_raw"][:self.ROWS]
        img = np.concatenate([self.d.layout.to_d(y[:, c0:c0 + w], self.d.mesh) for c in range(self.NC)
                              for c0, w in self.dn_sl[c]]).astype(np.float16).view(np.uint8)
        self.run_checks([(name, BingoMemSymbol(f"{self.TAG}_y", offset=off, chip_id=R), img[off:off + CHK], CHK)
                         for name, off in (("y_i0_a", 0), ("y_i0_b", self.YB - CHK))])

    def check_hazards(self):
        """The slots the slices reuse: a cluster's output slot read by its gather before the down GEMM out_slots slices
        later writes it again, and a root fold slot stored before the gather two groups later folds into it."""
        import networkx as nx
        h, GS, bad = self.NDN // self.NC, self.chain_group, []
        for c in range(self.NC):
            for jl in range(h - self.out_slots):
                for g in range(self.G):
                    if self.c(g) == c and not nx.has_path(self.dfg, self.gathers[(c, jl // GS)],
                                                          self.dn[(g, jl + self.out_slots)]):
                        bad.append(f"output slot g{g}: slice {jl} gathered before slice {jl + self.out_slots}")
            for n in range(h // GS - 2):
                for st in self.stores[(c, n)]:
                    if not nx.has_path(self.dfg, st, self.gathers[(c, n + 2)]):
                        bad.append(f"fold slot c{c}: group {n} stored before group {n + 2}")
        if bad:
            raise ValueError(f"{len(bad)} buffer hazards, e.g. " + "; ".join(bad[:4]))
        print(f"[{self.TAG}] buffer hazards: none (output and fold slots)")

    def finish(self, app_name, x_prefixes=("a8_pre_",)):
        self.check_hazards()
        self.dfg.split_chip_functions = True     # one host function a chip
        super().finish(app_name, x_prefixes)


def _app_name(S):
    return (f"{S.p['model']} FFN prefill, stage 45, {S.T} tokens in {S.T // S.ROWS} x {S.ROWS}, "
            f"{len(S.chips)} chips x {S.NC} clusters, X from HBM")


if __name__ == "__main__":
    llama3_data.register_ffn()
    run_driver("ffn prefill stage 45 workload generator", Stage45, _app_name)
