#!/usr/bin/env python3
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""DeepSeek-V2-Lite's dense FFN (layer 0), stages 1-3 of one pass of the 256-token prefill on sixteen chiplets alone:
gate|up over the weight stream, dequantise and SwiGLU, the reshape, the quantise and the A8 exchange.

X is in every cluster's L1 before the window (copied from an image of it in each memory chip's HBM), as are the gate|up
dequantisation factors. Each cluster takes its intermediate columns, the same columns of gate and of up; each weight
chunk, pushed by the cluster's memory chip into its L3 ring, meets all of the pass's rows in one GEMM as it lands (the
first chunk one GEMM per row block of X, as the whole block runs it while X lands, the last one GEMM per post block).
Every chunk's output is dequantised and through SwiGLU one row tile at a time into A16 (D layout); after the last chunk
A16 is reshaped to the A layout on the xDMA and quantised into the chip's A8 post block by post block, each row tile
then copied into the other cluster's A8. The window ends when every cluster holds its chip's whole A8. Checks: the
chip's A8 (its first and last 2 KiB) on the root's two clusters, the middle chip's cluster 0 and the last chip's cluster
1.

params.hjson: the whole block's mapping (model, rows, the columns a cluster and a chunk, the weight ring) for one pass
(tokens = rows), the platform and compile switches."""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(__file__).split("/target/sw/host/")[0]
sys.path.insert(0, os.path.join(REPO, "util", "sim", "multichip"))
sys.path.insert(0, os.path.join(REPO, "util", "sim", "common"))
sys.path.insert(0, os.path.join(REPO, "target", "sw", "host", "runtime", "libbingo", "mini_compiler"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "common"))
import _bingo_paths  # noqa: F401,E402
from bingo_gemm_reuse import reuse_gemm_config  # noqa: E402
from bingo_kernel_args import (  # noqa: E402
    INT4_B_LAYOUT_PAIRED, SnaxBingoKernelGemmFullArgs, SnaxBingoKernelIdma1dCopyArgs, SnaxBingoKernelSimdFp16ToInt8Args,
    SnaxBingoKernelSimdMulF16Args, SnaxBingoKernelSimdSwigluF16F16Args, simd_beats)
from bingo_mem_handle import BingoMemSymbol  # noqa: E402
from driver import run_driver  # noqa: E402
from libs.blocks import GemmQueue  # noqa: E402
from libs.comm.nest import d_to_a_args  # noqa: E402

import chiplet_mc as cm  # noqa: E402
import ffn_data  # noqa: E402

CACHE = os.path.join(REPO, "target", "sim", "automation", "sweep", "llm_blocks", "ffn", "cache")
CHK = 2048                                  # bytes a checked window
TILE = 16 * 16 * 2                         # one D-layout FP16 tile
BLK = 16                                   # a GEMM column block (meshCol)
X_BLOCKS = 4                               # the first chunk's GEMMs, one per row block of X
POST_BLOCKS = 4                            # the last chunk's GEMMs and the A16 reshape and quantise, per row block


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


def gu_chunks(n, blocks, first):
    """[(first gate block, blocks)] of a cluster's n columns: the first chunk `first` gate blocks wide (0: `blocks`),
    then chunks of `blocks`."""
    nb = n // BLK
    if not first:
        return [(b0, min(blocks, nb - b0)) for b0 in range(0, nb, blocks)]
    if first > blocks or first > nb:
        raise ValueError(f"params first_blocks={first}: at most blocks={blocks} and the {nb} blocks of a cluster")
    return [(0, first)] + [(b0, min(blocks, nb - b0)) for b0 in range(first, nb, blocks)]


class Stage123(cm.ChipletBuild):
    TAG = "pf"

    def __init__(self, args, p, plat, hw):
        super().__init__(args, p, plat, hw)
        d = self.d = ffn_data.FFNData(CACHE, p["model"], wbits=self.wbits, tokens=self.T)
        R = self.ROWS = int(p["rows"])
        if self.T != R or R % (16 * X_BLOCKS) or R % (16 * POST_BLOCKS):
            raise ValueError(f"params tokens={self.T} rows={R}: one pass (tokens = rows), rows whole blocks of 16")
        self.MT, self.D, self.I = R // 16, d.H, d.I
        self.XB = R * self.D                    # X of the pass, INT8
        # params blocks: 16-column gate blocks a gate|up chunk; first_blocks: the first chunk's (a narrow first chunk is
        # pushed sooner, so the first GEMM starts sooner)
        self.BPC = int(p.get("blocks", 3))
        self.cols = split_columns(self, p)
        self.chunks = {g: gu_chunks(n, self.BPC, int(p.get("first_blocks", 0))) for g, (_, n) in self.cols.items()}
        print(f"[{self.TAG}] one pass of {R} tokens; columns per cluster {sorted({n for _, n in self.cols.values()})}; "
              f"gate|up chunks per cluster {sorted({len(c) for c in self.chunks.values()})}")
        self._stage()
        self.make_streams()
        self.make_rings()
        self._buffers()

    def _stage(self):
        """X in every memory chip's HBM (the image every cluster's L1 is loaded from), and per cluster its gate|up
        chunks ([gate blocks | the same up blocks], paired INT4) and their dequantisation factors (one row tile's) in
        its memory chip's HBM."""
        st, d, T, lay = self.st, self.d, self.TAG, self.d.layout
        x = lay.to_a(d.hq[:self.ROWS]).astype(np.int8).reshape(-1)
        self.h_x = {m: st.put_hbm(f"{T}_s123x_m{m:02x}", x, mem_chip=(m >> 4, m & 15))
                    for m in sorted(set(self.mem.values()))}
        self.img, self.chunk_off, self.sdup = {}, {}, {}
        for g, (c0, n) in self.cols.items():
            k = self.chip(g)
            loc = (self.mem[k] >> 4, self.mem[k] & 15)
            parts, offs, sc, off = [], [], [], 0
            for b0, nb_ in self.chunks[g]:
                gate = np.arange(c0 + BLK * b0, c0 + BLK * (b0 + nb_))
                cols = np.concatenate([gate, self.I + gate])
                parts.append(pack_b(d, d.gu_q[:, cols]))
                offs.append((off, parts[-1].size))
                off += parts[-1].size
                sc.append(lay.to_d(np.tile(d.gu_f[cols].astype(np.float16), (16, 1)), d.mesh))
            self.img[g] = st.put_hbm(f"{T}_w_g{g}", np.concatenate(parts).astype(np.int8), mem_chip=loc)
            self.chunk_off[g] = offs
            s = np.concatenate(sc).astype(np.float16)
            self.sdup[g] = (st.put_hbm(f"{T}_sgu_g{g}", s.view(np.int8), mem_chip=loc), s.nbytes)

    def _buffers(self):
        """Per cluster: X, two gate|up output buffers (a chunk's GEMM writes one while the other's is dequantised), the
        factors, A16 in the D and in the A layout, and the chip's A8 (both clusters' columns)."""
        R, MT, T = self.ROWS, self.MT, self.TAG
        self.xbuf, self.gbuf, self.sbuf, self.a16d, self.a16a, self.a8 = {}, {}, {}, {}, {}, {}
        for g, (_, n) in self.cols.items():
            l1 = lambda name, nb: self.cx_chip(self.chip(g)).at(self.c(g)).l1(f"{T}_{name}", nb)
            self.xbuf[g] = l1("x0", self.XB)
            self.gbuf[g] = [l1(f"g{b}", MT * 2 * self.BPC * TILE) for b in range(2)]
            self.sbuf[g] = l1("sgu", self.sdup[g][1])
            self.a16d[g] = l1("a16d", R * n * 2)
            self.a16a[g] = l1("a16a", R * n * 2)
            self.a8[g] = l1("a8_0", R * self.kcols[self.chip(g)][1])

    def pre_window(self):
        """{chip: [task]} the start waits for: every cluster's factors and X into L1 from its memory chip's HBM. params
        start_per_chip: a chip's factor loads one after the other, cluster 0's last."""
        dm = self.roles["dm"]
        per_chip = bool(self.p.get("start_per_chip", False))
        pre = {k: [] for k in self.chips}
        for g in sorted(range(self.G), key=lambda g: (self.chip(g), -self.c(g))) if per_chip else range(self.G):
            k, c = self.chip(g), self.c(g)
            h, nb = self.sdup[g]
            t = self.node(k, c, dm, f"sgu_load_g{g}", "__snax_bingo_kernel_idma_1d_copy",
                          SnaxBingoKernelIdma1dCopyArgs(h.address, self.sbuf[g], nb), pre[k][-1:] if per_chip else [])
            pre[k] = [t] if per_chip else pre[k] + [t]
        self.x_pre = {}
        for g in range(self.G):
            k, c = self.chip(g), self.c(g)
            src = self.h_x[self.mem[k]].address
            self.x_pre[g] = self.node(k, c, dm, f"x_pre_g{g}", "__snax_bingo_kernel_idma_1d_copy",
                                      SnaxBingoKernelIdma1dCopyArgs(src, self.xbuf[g], self.XB), [])
            pre[k].append(self.x_pre[g])
        return pre

    def _load(self, g, j, deps=()):
        """Gate|up chunk j of cluster g into the next slab: a ring load behind the chunk's push. Returns (the load, the
        slab index, the slab)."""
        k, c = self.chip(g), self.c(g)
        ls = self.ls[g]
        off, nb = self.chunk_off[g][j]
        si, slab, war = ls.take(nb)
        ld, _ = ls.rings.load_chunk(lambda n, core, kern, args, after: self.node(k, c, core, n, kern, args, after), c,
                                    f"Ld_c{j}_i0_g{g}", src=self.img[g].address + off, nbytes=nb, slab=slab, war=war,
                                    deps=deps, dm=self.roles["dm"])
        return ld, si, slab

    def build(self):
        # dep-tag relays of xDMA tasks go to the SIMD cores first, each serving that one task (the whole block's)
        self.dfg.relay_prefer_cores = {self.roles["xdma"]: [self.roles["simd"]]}
        self.dfg.relay_one_consumer_cores = [self.roles["xdma"]]
        # params gather_relay_only: a dep-tag cell of the xDMA tasks over budget relays its leftover edges
        if self.p.get("gather_relay_only", False):
            self.dfg.relay_not_order_cores = [self.roles["xdma"]]
        self.sym_zero = False                   # no symmetric buffer
        self.setup(extra=self.pre_window())
        self.ex_push = {g: [] for g in range(self.G)}
        for g in range(self.G):
            self._gate_up(g)
        self.end_window([t for g in range(self.G) for t in self.ex_push[g]])
        items = []
        for k, c in checked(self.chips):
            g = self.gs(k)[c]
            kc0, kn = self.kcols[k]
            a8 = self.d.layout.to_a(self.d.a8[:self.ROWS, kc0:kc0 + kn], self.d.mesh)
            items += [(k, c, self.a8[g], a8), (k, c, self.a8[g].view(a8.size - CHK), a8.reshape(-1)[-CHK:])]
        self.run_checks([gather_check(self, "a8", items)])

    def _gate_up(self, g):
        """Cluster g's gate|up chunks in turn: each a GEMM over X as it lands (into the gate|up buffer the chunk before
        last used, once its reader is done), dequantised and through SwiGLU row tile by row tile into A16; the last
        chunk's per post block (_tail)."""
        k, c = self.chip(g), self.c(g)
        gemm, simd = self.roles["gemm"], self.roles["simd"]
        n = self.cols[g][1]
        nb, xb, x = n // BLK, self.xbuf[g], self.x_pre[g]
        reader = [None, None]                   # each gate|up buffer's last reader
        soff = 0                                # the chunk's factors in sbuf
        last_j = len(self.chunks[g]) - 1
        for j, (b0, nb_) in enumerate(self.chunks[g]):
            ld, si, slab = self._load(g, j, [x] if j == 0 else ())
            b = j % 2
            gb, war = self.gbuf[g][b], ([reader[b]] if reader[b] is not None else [])
            nblk, half = 2 * nb_, nb_ * TILE        # its gate blocks and its up blocks; one row block's gate tiles
            kargs = dict(input_B_addr=slab, input_C_addr=0, K=self.D // 4, N=nblk, array_shape_idx=0, transpose_A=0,
                         transpose_B=0, accumPrevC=0, int32tofp16_enable=1, int4_b_enable=INT4_B_LAYOUT_PAIRED,
                         d_shift=self.d.k_x)
            last = j == last_j and j != 0
            if j == 0 or last:                  # one GEMM per X block (first) or per post block (last): its row tiles
                nsp = POST_BLOCKS if last else X_BLOCKS
                mb = self.MT // nsp
                gms = [self.node(k, c, gemm, f"gu_c{j}_i0b{bb}_g{g}", "__snax_bingo_kernel_gemm_full",
                                 SnaxBingoKernelGemmFullArgs(input_A_addr=xb.view(bb * mb * (self.XB // self.MT)),
                                                             output_D_addr=gb.view(bb * mb * 2 * half), M=mb, **kargs),
                                 [ld, x] + war)
                       for bb in range(nsp)]
                gm_of = lambda m, gms=gms, mb=mb: gms[m // mb]
                gm = gms[-1]
            else:
                gm = self.node(k, c, gemm, f"gu_c{j}_i0_g{g}", "__snax_bingo_kernel_gemm_full",
                               SnaxBingoKernelGemmFullArgs(input_A_addr=xb, output_D_addr=gb, M=self.MT, **kargs),
                               [ld, x] + war)
                gm_of = lambda m, gm=gm: gm
            self.ls[g].release(si, gm)
            if last:
                reader[b] = self._tail(g, j, b0, nb_, gb, soff, gm_of)
                continue
            el = nblk * 256
            dqs = [self.node(k, c, simd, f"deq_c{j}m{m}_i0_g{g}", "__snax_bingo_kernel_simd_stream_elementwise",
                             SnaxBingoKernelSimdMulF16Args(gb.view(m * 2 * half), self.sbuf[g].view(soff),
                                                           gb.view(m * 2 * half), el // 32, 32), [gm_of(m)])
                   for m in range(self.MT)]
            soff += el * 2
            for m in range(self.MT):
                out = self.a16d[g].view((m * nb + b0) * TILE)
                sw = self.node(k, c, simd, f"swi_c{j}m{m}_i0_g{g}", "__snax_bingo_kernel_simd_swiglu_f16_f16",
                               SnaxBingoKernelSimdSwigluF16F16Args(gb.view(m * 2 * half), gb.view(m * 2 * half + half),
                                                                   out, 1, nb_ * 256), [dqs[m]])
            reader[b] = sw

    def _tail(self, g, j, b0, nb_, gb, soff, gm_of):
        """The last chunk's dequantise and SwiGLU row tile by row tile; behind each post block's last SwiGLU its reshape
        (xDMA) and the previous block's quantise (SIMD, whose reshape is done by then). Returns the last SwiGLU."""
        k, c, simd = self.chip(g), self.c(g), self.roles["simd"]
        nb, half, el = self.cols[g][1] // BLK, nb_ * TILE, 2 * nb_ * 256
        pmb = self.MT // POST_BLOCKS
        rs, sw = [], None
        for m in range(self.MT):
            dq = self.node(k, c, simd, f"deq_c{j}m{m}_i0_g{g}", "__snax_bingo_kernel_simd_stream_elementwise",
                           SnaxBingoKernelSimdMulF16Args(gb.view(m * 2 * half), self.sbuf[g].view(soff),
                                                         gb.view(m * 2 * half), el // 32, 32), [gm_of(m)])
            sw = self.node(k, c, simd, f"swi_c{j}m{m}_i0_g{g}", "__snax_bingo_kernel_simd_swiglu_f16_f16",
                           SnaxBingoKernelSimdSwigluF16F16Args(gb.view(m * 2 * half), gb.view(m * 2 * half + half),
                                                               self.a16d[g].view((m * nb + b0) * TILE), 1, nb_ * 256),
                           [dq])
            if (m + 1) % pmb == 0:
                bb = m // pmb
                rs.append(self._d2a(g, bb, sw))
                if bb:
                    self._quant(g, bb - 1, rs[bb - 1])
        self._quant(g, POST_BLOCKS - 1, rs[-1])
        return sw

    def _d2a(self, g, b, after):
        """Post block b of A16 (its row tiles, every column) reshaped to the A layout on the xDMA."""
        k, c, n = self.chip(g), self.c(g), self.cols[g][1]
        mb, nb = self.MT // POST_BLOCKS, n // BLK
        m0 = b * mb
        return self.node(k, c, self.roles["xdma"], f"d2a_i0b{b}_g{g}", "__snax_bingo_kernel_xdma_6d",
                         d_to_a_args(self.d.mesh, mb, nb, n // 4, self.a16d[g].view(m0 * nb * TILE),
                                     self.a16a[g].view(m0 * 16 * n * 2)), [after])

    def _quant(self, g, b, rs):
        """Post block b of A16a quantised, one row tile at a time, into this cluster's run of the chip's A8, each run
        then copied into the other cluster's A8 (self.ex_push[g])."""
        k, c, n = self.chip(g), self.c(g), self.cols[g][1]
        mb = self.MT // POST_BLOCKS
        kn = self.kcols[k][1]
        other = self.gs(k)[1 - c]
        coff = 16 * (kn - n) if c else 0        # this cluster's columns sit after the first cluster's
        for m in range(b * mb, (b + 1) * mb):
            off = m * 16 * kn + coff
            q = self.node(k, c, self.roles["simd"], f"quant_i0b{b}m{m}_g{g}", "__snax_bingo_kernel_simd_fp16_to_int8",
                          SnaxBingoKernelSimdFp16ToInt8Args(self.a16a[g].view(m * 16 * n * 2), self.a8[g].view(off),
                                                            beats=simd_beats(16 * n * 2), rows=1,
                                                            inv_scale_f32bits=self.d.inv_a), [rs])
            cp = SnaxBingoKernelIdma1dCopyArgs(self.a8[g].view(off), self.a8[other].view(off), 16 * n)
            self.ex_push[g].append(self.node(k, c, self.roles["dm"], f"a8_push_m{m}_g{g}",
                                             "__snax_bingo_kernel_idma_1d_copy", cp, [q]))

    def finish(self, app_name, x_prefixes=("x_pre_",)):
        # every device core runs its tasks in the order they are made; a GEMM whose configuration repeats its core's
        # previous one runs as gemm_minimal; one host function a chip
        GemmQueue.of(self.dfg, False).program_order(self.dfg, [self.roles[r] for r in ("gemm", "simd", "xdma", "dm")],
                                                    tag=self.TAG)
        print(f"[{self.TAG}] gemm_reuse: {reuse_gemm_config(self.dfg)} GEMMs run as gemm_minimal")
        self.dfg.split_chip_functions = True
        super().finish(app_name, x_prefixes)


def _app_name(S):
    return (f"{S.p['model']} FFN prefill, stage 123, {S.T} tokens in {S.T // S.ROWS} x {S.ROWS}, "
            f"{len(S.chips)} chips x {S.NC} clusters, X from HBM")


if __name__ == "__main__":
    run_driver("ffn prefill stage 123 workload generator", Stage123, _app_name)
