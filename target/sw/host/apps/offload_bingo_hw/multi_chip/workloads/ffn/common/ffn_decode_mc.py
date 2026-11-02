# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""The shared driver of the ffn decode workloads (workloads/dsv2/layer0/decode/ffn, workloads/llama3/decode/ffn): one
LLM's dense SwiGLU FFN for ONE token, on the prefill's mapping (ffn_prefill_mc.PBuild) run as one pass of one row tile,
with INT4 weights, tensor-parallel over every cluster of the 16 + 4 chiplet platform. The platform, the weight rings, the
anchors and the setup are chiplet_mc.py's; the data and the golden are ffn_data.py's (one token).

THE TILE. The (16, 4, 16) GEMM shape takes its rows 16 at a time, so the token is row 0 of one 16-row tile whose other 15
rows are zero: one pass of 16 rows, one row tile, one post block. Every buffer is the prefill's at 16 rows. The rows of
a GEMM, of SwiGLU, of the quantise and of the reduce never mix, so row 0 of every result is the token's and the zero rows
stay zero (silu(0) * 0 = 0) through every stage.

THE MAPPING is the prefill's. Cluster g takes the intermediate columns J_g, the same columns of gate and of up; each
gate|up weight chunk meets the tile in one GEMM as it streams in. A chip's two clusters split the down projection by
output columns: each takes the chip's columns (K) for its half of the outputs, so a slice of the down output holds the
chip's whole partial, and the chips' partials are reduced by ChainGathers into the root:

    X        the token's INT8 row from the HBM of the root's memory chip (params x_row, the default): the root's host
             queues that memory chip's push engine, which multicasts the row into every chip's L3 behind one flag
             (__host_bingo_kernel_hbm_bcast, the prefill's x_hbm path); every cluster waits for the flag and copies the
             row into row 0 of its X tile in one 2-D copy (4 bytes of every 64-byte block of the A layout), the tile's
             other rows zeroed before the window. x_row false: the HBM holds the whole tile, A layout (16 x the
             bytes), multicast and fetched as the prefill fetches a pass
    gate|up  per weight chunk: GEMM X . [gate | up] chunk -> G (D layout, FP16), dequantised (x the chunk's column
             scales), SwiGLU into A16 (D layout)
    A8       A16 relaid out D -> A on the xDMA, quantised to INT8 into the chip's A8, and the cluster's columns of it
             copied into the other cluster's A8
    down     per weight chunk (a slice of the cluster's outputs): GEMM A8 . down chunk -> the chip's partial of the slice
             (16 rows, D layout), into an output slot of the cluster's symmetric buffer. params dn_gemv: the GEMV of
             VersaCore's (1, 4, 32) shape instead (__snax_bingo_kernel_gemv: row 0 of the same A8, the same paired INT4
             chunk, twice the weights a pass), whose partial is the token's row alone, FP16, contiguous
    reduce   per group of params chain_group slices: one ChainGather through every chip, each hop's xDMA adding its
             partial in transit, into the root's fold slots; the gathers run one at a time, alternating between the
             two clusters' xDMAs, the first behind X's multicast (a multicast never meets a gather)
    Y        the root's clusters copy the token's row of each folded slice (row 0 of every 16 x 16 tile; with dn_gemv
             the slice itself) into one L1 row on the root's cluster 0, Y_raw in column order; once both clusters'
             last copies are done (every gather has ended, every weight has landed) the root's DM core WRITES the row
             to the HBM of the root's memory chip, so no cross-chip write of Y meets a ChainGather or a weight push.
             Y_raw is FP16 before the final dequantise (x s_dn per output column), as the prefill leaves it: a layer
             applies s_dn where it adds the residual. params y_dequant: the root's SIMD applies s_dn to the row first
             (one task), so the HBM holds Y16

Every SIMD task is one of the prefill's per-row-tile kernels (dequantise, SwiGLU, quantise), which run on this cluster's
32-byte SIMD beat; the one-row kernels of the old decode driver (ffn_mc.py: SwiGLU into a_row) assume a 64-byte beat.

THE WEIGHTS stream as in the prefill's first pass: each cluster's gate|up and down chunks sit in its memory chip's HBM;
the memory chip pushes them into the chip's L3 ring and the DM core loads each into an L1 slab behind its landed flag.

THE PROGRAM ORDER. Every device core runs its tasks in the order this driver creates them (chain_program_order).

TIMING. The start anchor after every chip's setup (the scales loaded, the X tiles zeroed); the end anchor after the HBM
write of Y. X's multicast and the weight stream are inside the window.

CHECKS, byte for byte against the golden of the device's arithmetic (ffn_data.py, the reduce's fold order): the Y_raw
(y_dequant: Y16) row in the root memory chip's HBM, every column; the first 2 KiB of the root's A8 (row 0 the token's,
the zero rows).

PARAMS: ffn_prefill_mc.py's, for one pass of one row tile; tokens 1. The decode refuses what needs more than one pass,
row tile or post block (x_device, l3_overlay, y_scatter, split_last, split_first_dn, x_early, restream, x_blocks,
x_buffers, x_tile_bytes) and the ablation arms (sync barrier, reduce root, x_cast unicast).
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ffn_prefill_mc as fp  # noqa: E402
from driver import run_driver  # noqa: E402

from bingo_kernel_args import (  # noqa: E402
    SnaxBingoKernelGemmFullArgs, SnaxBingoKernelGemvArgs, SnaxBingoKernelIdma1dCopyArgs, SnaxBingoKernelIdma2dCopyArgs,
    SnaxBingoKernelSimdMulF16Args, SnaxBingoKernelWaitFlagsArgs, SnaxBingoKernelXdmaChainGatherArgs,
    SnaxBingoKernelXdmaMemsetArgs)
from bingo_mem_handle import BingoMemSymbol  # noqa: E402

import chiplet_mc as cm  # noqa: E402
import ffn_data  # noqa: E402

ROWS = 16                                  # the GEMM shape's row granule: the token's row and 15 zero rows
TILE, BLK = fp.TILE, fp.BLK                # one D-layout FP16 tile, a GEMM column block


class DBuild(fp.PBuild):
    TAG = "fd"
    MIN_POST_BLOCKS = 1

    def __init__(self, args, p, plat, hw, shared=None):
        if int(p.get("tokens", 1)) != 1:
            raise ValueError(f"params tokens={p['tokens']}: the decode runs one token")
        bad = [k for k in ("x_device", "l3_overlay", "y_scatter", "split_last", "split_first_dn", "x_early", "restream")
               if p.get(k, False)]
        bad += [k for k, v in (("rows", ROWS), ("post_blocks", 1), ("x_blocks", 1), ("x_buffers", 1))
                if int(p.get(k, v)) != v] + [k for k in ("x_tile_bytes",) if k in p]
        bad += [f"{k} {p[k]}" for k, v in (("sync", "edge"), ("reduce", "chain"), ("x_cast", "multicast"))
                if str(p.get(k, v)) != v]
        if bad:
            raise ValueError(f"params {bad}: the decode is one pass of one {ROWS}-row tile, one post block, X from HBM "
                             f"by one multicast, the ChainGather reduce, edge sync")
        # params x_row (default true): the HBM holds the token's INT8 row, which every cluster scatters into row 0 of its
        # zeroed X tile; false: the HBM holds the whole tile in the A layout
        self.x_row = bool(p.get("x_row", True))
        # params dn_gemv: the down slices on VersaCore's one-row GEMV shape (the paired INT4 chunks, w4_paired). With
        # gemm_reuse: every GEMV of a GEMM core follows its last gemm_full, so no gemm_minimal inherits a GEMV's set-up
        self.dn_gemv = bool(p.get("dn_gemv", False))
        if self.dn_gemv and not p.get("w4_paired", False):
            raise ValueError("params dn_gemv: the GEMV reads INT4 weights in the paired layout (w4_paired)")
        # params y_dequant: the root's SIMD multiplies Y_raw's row by s_dn (one task, before the HBM write), so the HBM
        # holds Y16; default false: Y_raw, as the prefill leaves it
        self.y_dequant = bool(p.get("y_dequant", False))
        # the prefill's builder runs the tile as one pass of ROWS tokens
        super().__init__(args, dict(p, tokens=ROWS, rows=ROWS, post_blocks=1, x_blocks=1), plat, hw, shared=shared)
        self.T = 1
        # x_hbm_bcasts multicasts what the HBM holds as one tile behind one flag
        self.XT, self.TPP = self.XR, 1
        print(f"[{self.TAG}] decode: one token, row 0 of a {ROWS}-row tile; X from HBM as "
              f"{'the row' if self.x_row else 'the tile'} ({self.XR} B), {'Y16' if self.y_dequant else 'Y_raw'} "
              f"({2 * self.D} B) into HBM; down on the "
              f"{'GEMV' if self.dn_gemv else 'GEMM'}; {self.NDN // self.NC} slices a cluster, "
              f"{self.NDN // self.NC // self.chain_group} gathers a cluster")

    def _data(self, args, p):
        """The FFN's data and golden of the one token."""
        cache = os.path.join(fp.REPO, p.get("weights", "target/sim/automation/sweep/llm_blocks/ffn/cache"))
        return ffn_data.FfnData(cache, p.get("model", "dsv2-lite-l0"), args.dsv2_dir, seed=int(p.get("seed", 1)),
                                wbits=self.wbits, tokens=1, inter=p.get("inter"))

    # ---- data -----------------------------------------------------------------------------
    def x_tile(self):
        """X as every cluster's GEMMs read it: the 16-row tile in the A layout, the token's row 0, the other rows zero."""
        t = np.zeros((ROWS, self.D), dtype=np.int8)
        t[0] = self.d.hq[0]
        return self.lay.to_a(t, self.d.mesh).astype(np.int8)

    def x_a(self):
        """X as the HBM holds it: the token's INT8 row (x_row) or the whole tile."""
        return self.d.hq[0].astype(np.int8) if self.x_row else self.x_tile()

    def _stage(self):
        """X in the root memory chip's HBM, its landing area in every chip's L3 and its flag; Y_raw's row in the same
        HBM; every cluster's weight chunks and gate|up scales in its memory chip's HBM, as PBuild._stage stages them."""
        st, d, T, lay = self.st, self.d, self.TAG, self.lay
        m = self.mem[self.ROOT]
        x = self.x_a().reshape(-1)
        self.XR = x.size                        # bytes of X in the HBM and in every chip's L3
        st.put_zeros(f"{T}_xland", "int8_t", self.XR, preload=True)
        self.h_xhbm = st.put_hbm(f"{T}_xhbm", x, mem_chip=(m >> 4, m & 15))
        st.put_zeros(f"{T}_xtflag", "uint8_t", 64)
        self.h_yhbm = st.put_hbm(f"{T}_yhbm", np.zeros(2 * self.D, dtype=np.uint8), mem_chip=(m >> 4, m & 15))
        if self.y_dequant:                      # s_dn, FP16 a column
            self.h_sdn = st.put_hbm(f"{T}_sdn", d.dn_f.astype(np.float16).view(np.int8), mem_chip=(m >> 4, m & 15))
        self.img, self.chunk_off, self.sdup = {}, {}, {}
        for g, (c0, n) in self.cols.items():
            k = self.chip(g)
            loc = (self.mem[k] >> 4, self.mem[k] & 15)
            parts, offs, sc = [], [], []
            for b0, nb_ in self.chunks[g]:
                cols = self._gu_cols(g, b0, nb_)
                parts.append(self._pack_b(d.gu_q[:, cols]))
                s = np.tile(d.gu_f[cols].astype(np.float16), (16, 1))
                sc.append(lay.to_d(s, d.mesh))  # one row tile: its scale tile
            kc0, kn = self.kcols[k]             # the chip's columns, this cluster's share of the outputs
            for o0, w in self.dn_sl[self.c(g)]:
                parts.append(self._pack_b(d.dn_q[kc0:kc0 + kn, o0:o0 + w]))
            off = 0
            for b in parts:
                offs.append((off, b.size))
                off += b.size
            self.img[g] = st.put_hbm(f"{T}_w_g{g}", np.concatenate(parts).astype(np.int8), mem_chip=loc)
            self.chunk_off[g] = offs
            s = np.concatenate(sc).astype(np.float16)
            self.sdup[g] = (st.put_hbm(f"{T}_sgu_g{g}", s.view(np.int8), mem_chip=loc), s.nbytes)

    def _buffers(self):
        """The prefill's buffers at 16 rows, and Y_raw's row on the root's cluster 0. dn_gemv: an output slot (and a
        fold slot) holds a row of the slice, so the slots are SLICE = a row apart in the same buffers."""
        super()._buffers()
        self.yrow = self.cx_chip(self.ROOT).at(0).l1(f"{self.TAG}_yrow", 2 * self.D)
        if self.y_dequant:
            self.sdn = self.cx_chip(self.ROOT).at(0).l1(f"{self.TAG}_sdn", 2 * self.D)
        if self.dn_gemv:
            self.SLICE = 2 * max(w for v in self.dn_sl.values() for _, w in v)

    def _xland(self, off, chip_id=None):
        """Where X lands in every chip's L3, at byte `off`."""
        return BingoMemSymbol(f"{self.TAG}_xland", offset=off, chip_id=chip_id)

    def slice_bytes(self, w):
        """The bytes of a w-column slice of the down output: 16 rows in the D layout, or (dn_gemv) the token's row."""
        return (1 if self.dn_gemv else ROWS) * w * 2

    # ---- X --------------------------------------------------------------------------------
    def pre_window(self):
        """The prefill's (the scales into L1), with x_row every X tile zeroed on its cluster's xDMA, with y_dequant s_dn
        into the root's cluster 0."""
        pre = super().pre_window()
        if self.y_dequant:
            R = self.ROOT
            pre[R].append(self.node(R, 0, self.roles["dm"], "sdn_load", "__snax_bingo_kernel_idma_1d_copy",
                                    SnaxBingoKernelIdma1dCopyArgs(self.h_sdn, self.sdn, 2 * self.D), pre[R][-1:]))
        for g in range(self.G if self.x_row else 0):
            k, c = self.chip(g), self.c(g)
            pre[k].append(self.node(k, c, self.roles["xdma"], f"x_zero_g{g}", "__snax_bingo_kernel_xdma_memset",
                                    SnaxBingoKernelXdmaMemsetArgs(self.xbuf[g][0], self.XB,
                                                                  SnaxBingoKernelXdmaMemsetArgs.PATTERN_ZERO), []))
        return pre

    def stage_x(self, i, src, after):
        """x_row: the token's row into row 0 of every cluster's X tile, behind the multicast's flag: a wait on the flag,
        then one 2-D copy from the chip's L3, a 4-byte run (the row's 4 values of a 4-column block) into each 64-byte
        block of the A layout. Without x_row the prefill's flagged fetch of the tile. {g: the task after which it is
        there}."""
        if not self.x_row:
            return super().stage_x(i, src, after)
        T, dm, ts = self.TAG, self.roles["dm"], self.d.mesh[1]
        out = {}
        for g in range(self.G):
            k, c = self.chip(g), self.c(g)
            w = self.node(k, c, dm, f"x_flag_g{g}", "__snax_bingo_kernel_wait_flags",
                          SnaxBingoKernelWaitFlagsArgs(BingoMemSymbol(f"{T}_xtflag"), 1, 1), list(after.get(g, [])))
            # the multicast's flag says the row has landed: an order edge, not a dependency
            self.dfg.bingo_add_order_edge(src, w)
            out[g] = self.node(k, c, dm, f"x_wait_i{i}_g{g}", "__snax_bingo_kernel_idma_2d_copy",
                               SnaxBingoKernelIdma2dCopyArgs(self._xland(0), self.xbuf[g][0], ts, ts, ROWS * ts,
                                                             self.D // ts), [w])
        return out

    # ---- the pass ---------------------------------------------------------------------------
    def _passes(self, go, after_x):
        """The one pass: X from HBM, the gate|up chunks and the post-processing (PBuild.stage_gu, stage_post), then the
        down slices, each cluster's in output slots j mod out_slots, one ChainGather a group of chain_group slices, the
        token's row of each folded group into Y_raw's row (params store_lag gathers late), and Y_raw into HBM."""
        R, dm, xdma, gemm = self.ROOT, self.roles["dm"], self.roles["xdma"], self.roles["gemm"]
        self.x_bc = self.x_hbm_bcasts(go)
        x = self.stage_x(0, self.x_bc[0], after_x)
        sw, _ = self.stage_gu(0, x)
        post = self.stage_post(0, sw)
        other = lambda g: self.gs(self.chip(g))[1 - self.c(g)]
        ex_last = {g: self.ex_push[(g, 0)][-1] for g in range(self.G)}
        # params first_gather_waits_copies (default true): the first gather also waits for every A8 copy; false: only
        # the copies into cluster c precede cluster c's gathers, through cluster c's own down GEMMs
        copies = list(ex_last.values()) if self.p.get("first_gather_waits_copies", True) else []
        SL, others, h, GS = self.SLICE, self.order[1:], self.NDN // self.NC, self.chain_group
        p1x = {c: f"((uint64_t)({self.p1c[c].get_c_var_name()}))" for c in range(self.NC)}
        lag = int(self.p.get("store_lag", 0))
        sym_reader, p1_reader, pending, last_st = {}, {}, [], {}
        last_red = None

        def load(jl):
            if jl == 0 and self.dn_first_load:
                return dict(self.dn_first_load)
            return {g: self._load(g, 0, len(self.chunks[g]) + jl) for g in range(self.G)}

        def store(c, j, red, s):
            # the token's row of slices j .. j + GS - 1 (one gather, slice n at n x SL of the root's fold slot s) into
            # Y_raw's row at their columns: one copy when the slices fill their fold slots and their columns follow each
            # other, else one a slice. GEMM: row 0 of each 16 x 16 tile (32 B a tile, TILE apart); GEMV: the slice
            sl = [self.dn_sl[c][j - c * h + n] for n in range(GS)]
            base = s * GS * SL
            whole = all(self.slice_bytes(w) == SL for _, w in sl[:-1]) and \
                all(sl[n][0] == sl[n - 1][0] + sl[n - 1][1] for n in range(1, GS))
            runs = [(0, sl[0][0], sum(w for _, w in sl))] if whole else \
                [(n * SL, o0, w) for n, (o0, w) in enumerate(sl)]
            sts = []
            for n, (off, o0, w) in enumerate(runs):
                src, dst = self.p1c[c].view(base + off), self.yrow.view(2 * o0)
                kern, kargs = ("__snax_bingo_kernel_idma_1d_copy", SnaxBingoKernelIdma1dCopyArgs(src, dst, 2 * w)) \
                    if self.dn_gemv else \
                    ("__snax_bingo_kernel_idma_2d_copy", SnaxBingoKernelIdma2dCopyArgs(src, dst, 2 * BLK, TILE, 2 * BLK,
                                                                                       w // BLK))
                sts.append(self.node(R, c, dm, f"y_store_j{j}" if n == 0 else f"y_store_j{j}n{n}", kern, kargs, [red]))
            last_st[c] = sts
            p1_reader[(c, s)] = sts

        lds = load(0)
        prev_dn = {}
        for jl in range(h):
            so = jl % self.out_slots                # the clusters' output slot
            dn = {}
            for g in range(self.G):
                k, c = self.chip(g), self.c(g)
                ld, si, slab = lds[g]
                kn = self.kcols[k][1]
                w = self.dn_sl[c][jl][1]
                deps = [ld, post[g], ex_last[other(g)]] + sym_reader.get((k, c, so), [])
                dst = self.sym[(k, c)].view(so * SL)
                if self.dn_gemv:
                    dn[g] = self.node(k, c, gemm, f"dn_c{c * h + jl}_i0_g{g}", "__snax_bingo_kernel_gemv",
                                      SnaxBingoKernelGemvArgs(self.a8[g][0], slab, dst, K=kn, N=w, d_shift=self.d.k_dn,
                                                              w4=True), deps)
                else:
                    dn[g] = self.node(k, c, gemm, f"dn_c{c * h + jl}_i0_g{g}", "__snax_bingo_kernel_gemm_full",
                                      SnaxBingoKernelGemmFullArgs(
                                          input_A_addr=self.a8[g][0], input_B_addr=slab, input_C_addr=0,
                                          output_D_addr=dst, M=self.MT, K=kn // 4, N=w // BLK, array_shape_idx=0,
                                          transpose_A=0, transpose_B=0, accumPrevC=0, int32tofp16_enable=1,
                                          int4_b_enable=self.int4_b, d_shift=self.d.k_dn), deps)
                self.ls[g].release(si, dn[g])
                if jl % GS:
                    # params chain_group: a gather waits for its group's last down GEMM on each chip only; the GEMM core
                    # runs its queue in order, so the group's GEMMs go into it in slice order
                    self.dfg.bingo_add_order_edge(prev_dn[g], dn[g])
                prev_dn[g] = dn[g]
            if jl + 1 < h:
                lds = load(jl + 1)
            while len(pending) > lag * self.NC:
                store(*pending.pop(0))
            if (jl + 1) % GS:
                continue
            # the group's gathers: slices j0 .. jl, in output slots so0 .. so (back to back), into the root's fold slot s
            j0 = jl + 1 - GS
            s, so0 = (j0 // GS) % 2, j0 % self.out_slots
            for c in range(self.NC):
                j = c * h + j0
                # params chain_alt: cluster 1's gathers run the ring the other way
                ordr = others[::-1] if self.chain_alt and c == 1 else others
                ga = SnaxBingoKernelXdmaChainGatherArgs.fold_into(
                    cm.remote, R, c, ordr, src=self.sx(c, so0 * SL), dst=f"({p1x[c]} + {s * GS * SL})",
                    size=(GS - 1) * SL + self.slice_bytes(self.dn_sl[c][jl][1]),
                    junction=self.jct["Elementwise"], jct_csr0=fp.EW_F16_ADD, holds=[self.p1c[c]])
                first = ([last_red] if last_red is not None else []) + (list(copies) if j0 == 0 else [])
                if j0 == 0 and c == 0:
                    first.append(self.x_bc[-1])     # X's multicast before the first gather
                ins = [dn[g] for g in range(self.G) if self.c(g) == c]
                red = self.node(R, c, xdma, f"y_chain_j{j}", "__snax_bingo_kernel_xdma_chain_gather", ga,
                                ins + p1_reader.get((c, s), []) + first)
                last_red = red
                for k in self.chips:
                    for o in range(so0, so0 + GS):
                        sym_reader[(k, c, o)] = [red]
                pending.append((c, j, red, s))
        while pending:
            store(*pending.pop(0))
        # Y's row into HBM once both root clusters' last copies are done: every gather has ended by then
        ready = [t for c in sorted(last_st) for t in last_st[c]]
        if self.y_dequant:                      # Y16 = Y_raw (.) s_dn, in place, the prefill's dequantise kernel
            ready = [self.node(R, 0, self.roles["simd"], "y_deq", "__snax_bingo_kernel_simd_stream_elementwise",
                               SnaxBingoKernelSimdMulF16Args(self.yrow, self.sdn, self.yrow, self.D // 32, 32), ready)]
        self.y_hbm = self.y_out(ready)
        self.end_window([self.y_hbm])
        self._checks()

    def y_out(self, ready):
        """The root's DM core writes the row (yrow, on the root's cluster 0) into the HBM of the root's memory chip behind
        `ready`. Returns the task the end follows. A whole layer (llama3/common/llama3_decode_layer_mc.py) puts its final
        residual here instead."""
        return self.node(self.ROOT, 0, self.roles["dm"], "y_hbm", "__snax_bingo_kernel_idma_1d_copy",
                         SnaxBingoKernelIdma1dCopyArgs(self.yrow, self.h_yhbm, 2 * self.D), ready)

    def _checks(self):
        """Y_raw's row in the HBM of the root's memory chip, every column (FP16, column order); the first 2 KiB of the
        root's own A8 (A layout: the token's row 0, the zero rows)."""
        R = self.ROOT
        y = np.ascontiguousarray(self.gold["y16" if self.y_dequant else "y_raw"][0].astype(np.float16)).view(np.uint8)
        checks = [("y_hbm", self.h_yhbm, y, y.size)]
        g = self.gs(R)[0]
        c0, ncol = self.kcols[R]                # the chip's A8, both clusters
        a8 = np.zeros((ROWS, ncol), dtype=np.int8)
        a8[0] = self.d.a8[0, c0:c0 + ncol]
        img = self.lay.to_a(a8, self.d.mesh).view(np.uint8)
        n = min(2048, img.size)
        got = self.stash("a8_root", R, 0, self.a8[g][0], n, [])
        checks.append(("a8_root", got, img[:n], n))
        self.run_checks(checks)


def _app_name(S):
    return (f"{S.p.get('model', 'dsv2-lite-l0')} FFN decode, one token as a {ROWS}-row tile, {len(S.chips)} chips x "
            f"{S.NC} clusters, X from HBM ({'row' if S.x_row else 'tile'}), Y_raw to HBM"
            + (", down on the GEMV" if S.dn_gemv else ""))


def run():
    run_driver("ffn decode workload generator", DBuild, _app_name)
