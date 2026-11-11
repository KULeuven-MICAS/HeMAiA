# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""The FFN block of the decode layers (DBuild, which ../../dsv2/common/dsv2_decode_mc.py and
../../llama3/common/llama3_decode_layer_mc.py extend as their FfnPart): one
LLM's dense SwiGLU FFN for ONE token, on the prefill's mapping (ffn_prefill_mc.PBuild) run as one pass of one row tile,
with INT4 weights, tensor-parallel over every cluster of the 16 + 4 chiplet platform. The platform, the weight rings, the
stamps and the setup are chiplet_mc.py's; the data and the golden are ffn_data.py's (one token).

THE TILE. The (16, 4, 16) GEMM shape takes its rows 16 at a time, so the token is row 0 of one 16-row tile whose other 15
rows are zero: one pass of 16 rows, one row tile, one post block. Every buffer is the prefill's at 16 rows. The rows of
a GEMM, of SwiGLU, of the quantise and of the reduce never mix, so row 0 of every result is the token's and the zero rows
stay zero (silu(0) * 0 = 0) through every stage.

THE MAPPING is the prefill's. Cluster cl takes the intermediate columns J_cl, the same columns of gate and of up; each
gate|up weight chunk meets the tile in one GEMM as it streams in. A chip's two clusters split the down projection by
output columns: each takes the chip's columns (K) for its half of the outputs, so a slice of the down output holds the
chip's whole partial, and the chips' partials are reduced by ChainGathers into the root:

    X        the token's INT8 row from the HBM of the root's memory chip (params x_row, the default): the root's host
             queues that memory chip's push engine, which multicasts the row into every chip's L3 behind one flag
             (__host_bingo_kernel_hbm_bcast, as the prefill's X); every cluster waits for the flag and copies the
             row into row 0 of its X tile in one 2-D copy (4 bytes of every 64-byte block of the A layout), the tile's
             other rows zeroed before the window. x_row false: the HBM holds the whole tile, A layout (16 x the
             bytes), multicast and fetched as the prefill fetches a pass
    gate|up  per weight chunk: GEMM X . [gate | up] chunk -> G (D layout, FP16), dequantised (x the chunk's column
             scales), SwiGLU into A16 (D layout)
    A8       A16 relaid out D -> A on the xDMA, quantised to INT8 into the chip's A8, and the cluster's columns of it
             copied into the other cluster's A8
    down     per weight chunk (a slice of the cluster's outputs): GEMM A8 . down chunk -> the chip's partial of the slice
             (16 rows, D layout), into an output slot of the cluster's exchange buffer. params dn_gemv: the GEMV of
             VersaCore's (1, 4, 32) shape instead (__snax_bingo_kernel_gemv: row 0 of the same A8, the same paired INT4
             chunk, twice the weights a pass), whose partial is the token's row alone, FP16, contiguous
    reduce   per group of params chain_group slices: one ChainGather through every chip, each hop's xDMA adding its
             partial in transit, into the root's fold slots; the gathers run one at a time, alternating between the
             two clusters' xDMAs, the first behind X's multicast (a multicast never meets a gather). params
             chains_after_stream: the first gather also behind every cluster's last down GEMM, so every gather runs
             after the last weight push of every memory chip has landed (no gather beside a push; out_slots must hold
             every slice of a cluster)
    Y        the root's clusters copy the token's row of each folded slice (row 0 of every 16 x 16 tile; with dn_gemv
             the slice itself) into one L1 row on the root's cluster 0, Y_raw in column order; once both clusters'
             last copies are done (every gather has ended, every weight has landed) the root's DM core WRITES the row
             to the HBM of the root's memory chip, so no cross-chip write of Y meets a ChainGather or a weight push.
             Y_raw is FP16 before the final dequantise (x s_dn per output column), as the prefill leaves it: a layer
             applies s_dn where it adds the residual. params y_dequant: the root's SIMD applies s_dn to the row first
             (one task), so the HBM holds Y16

Every SIMD task is one of the prefill's per-row-tile kernels (dequantise, SwiGLU, quantise), which run on this cluster's
32-byte SIMD beat.

params gu_gemv (with dn_gemv): EVERY weight multiply on the GEMV, no 16-row tile anywhere; the same distribution
(columns over every cluster, the down split by output columns, the rings, the paired INT4 chunks, the same HBM weight
images), only rows in L1:

    X        the token's INT8 row in a_row (THE X DELIVERY below), multicast from the root memory chip's HBM into every
             chip's L3 behind one flag as above; every cluster fetches it into L1 in one flagged copy
    gate|up  per weight chunk as it lands: GEMV X . [gate | up] chunk (the same paired INT4 chunk, a_row A) -> the
             chunk's FP16 row, its gate columns then its up columns, contiguous; dequantised (x the chunk's column
             factors, one FP16 row: the scales are rows, not D tiles); SwiGLU of the row's two halves into the cluster's
             A16 row (FP16, its columns in order)
    A8       the A16 row spread into a_row's FP16 form on the xDMA (each 8-byte atom, 4 values, 16 bytes apart in a
             row zeroed before the window), quantised on the SIMD (Fp16ToInt8 keeps every value's place, so the INT8
             lands in a_row) into the cluster's run of the chip's A8 row (a_row, both clusters' columns, 2 bytes a
             column), and that run copied into the other cluster's A8 row
    down     as dn_gemv, its A the chip's A8 row (a_row)
    reduce, Y  as above
The SIMD tasks are the 32-byte-beat stream kernels on whole beats: the dequantise (MulF16, a beat every 16 columns of
the chunk's row), SwiGLU (a beat a gate block), the quantise. The xDMA moves whole beats of 8 atoms (32 values), so the spread
reads the A16 row and writes its a_row form up to a whole beat; both buffers are padded to that and zeroed with it, and
the quantise reads only the row's own bytes. No task here is a GEMM, so the GEMM-config reuse rewrites nothing
(_passes checks that every array task of the block is a GEMV).

THE X DELIVERY (gu_gemv). X is the token's INT8 row in a_row: value c at byte (c // 4) * 8 + c % 4, 2 x hidden
bytes (gemv_utils.a_row_bytes); the GEMV reads the first 4 bytes of each 8-byte word, so the other 4 are free (zero
here). It lands at _xland(0), XR = 2 x hidden bytes, behind the flag <TAG>_xtflag, as the row does without gu_gemv. A
producer holding the FP16 row makes it with two existing kernels: an xDMA spread of the row's 8-byte atoms 16 bytes
apart into an FP16 row zeroed once (__snax_bingo_kernel_xdma_6d; the gaps stay zero across uses), then
simd_fp16_to_int8 over that row (4 x hidden bytes in, a_row out); a producer holding an INT8 A-layout tile with the
token in row 0 compacts it with one xDMA copy of the first 8-byte word of every 64-byte block. A contiguous INT8 row
has no cheap conversion (4-byte runs; the xDMA moves 8-byte atoms).

THE WEIGHTS stream as in the prefill's first pass: each cluster's gate|up and down chunks sit in its memory chip's HBM;
the memory chip pushes them into the chip's L3 ring and the DM core loads each into an L1 slab behind its landed flag.

THE PROGRAM ORDER. The compiler orders every device core's program (mini_compiler passes/bingo_dfg_schedule.py); the
edges this driver adds are every ordering the data needs.

TIMING. The start stamp after every chip's setup (the scales loaded, the X tiles zeroed); the end stamp after the HBM
write of Y. X's multicast and the weight stream are inside the window.

CHECKS, byte for byte against the golden of the device's arithmetic (ffn_data.py, the reduce's fold order): the Y_raw
(y_dequant: Y16) row in the root memory chip's HBM, every column; the first 2 KiB of the root's A8 (row 0 the token's,
the zero rows; gu_gemv: the root's A8 row in a_row, its free bytes zero). The golden is per row and elementwise, so the
GEMV path's is the tile path's row 0.

PARAMS: ffn_prefill_mc.py's, for one pass of one row tile; tokens 1. The decode refuses what needs more than one pass,
row tile or post block (l3_overlay, y_scatter, split_last, split_first_dn, restream, x_blocks) and the
ablation arms (sync barrier, reduce root, x_cast unicast).
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ffn_prefill_mc as fp  # noqa: E402

from bingo_kernel_args import (  # noqa: E402
    SnaxBingoKernelGemmFullArgs, SnaxBingoKernelGemvArgs, SnaxBingoKernelIdma1dCopyArgs, SnaxBingoKernelIdma2dCopyArgs,
    SnaxBingoKernelIdmaFetchFlaggedArgs, SnaxBingoKernelSimdFp16ToInt8Args, SnaxBingoKernelSimdMulF16Args,
    SnaxBingoKernelSimdSwigluF16F16Args, SnaxBingoKernelWaitFlagsArgs, SnaxBingoKernelXdma6dArgs,
    SnaxBingoKernelXdmaChainGatherArgs, SnaxBingoKernelXdmaMemsetArgs, simd_beats)
from bingo_mem_handle import BingoMemSymbol  # noqa: E402

import chiplet_mc as cm  # noqa: E402
import ffn_data  # noqa: E402
import gemv_utils  # noqa: E402  (util/sim/gemm: ffn_data puts it on sys.path)

ROWS = 16                                  # the GEMM shape's row granule: the token's row and 15 zero rows
TILE, BLK = fp.TILE, fp.BLK                # one D-layout FP16 tile, a GEMM column block
ATOM, LANES = 8, 8                         # gu_gemv: the xDMA's 8-byte atom (4 FP16 values), atoms a beat


class DBuild(fp.PBuild):
    """The FFN decode block (the module docstring), the prefill's PBuild run as one pass of one 16-row tile. run_driver
    makes the program in four steps: __init__ (the parameters), mem_handles() (the data, the golden and the memory),
    build() (the task graph: every task and edge) and finish() (the buffer-hazard check, then the compile into the C
    header and the data image)."""
    TAG = "fd"
    MIN_POST_BLOCKS = 1

    def __init__(self, args, p, plat, hw, shared=None):
        """The parameters; no data, memory or task yet: the decode's refusals of what needs more than one pass, row
        tile or post block, its switches (x_row, dn_gemv, gu_gemv, y_dequant), then PBuild.__init__ with the params of
        one pass of one 16-row tile; the decode's token count is one."""
        if int(p.get("tokens", 1)) != 1:
            raise ValueError(f"params tokens={p['tokens']}: the decode runs one token")
        bad = [k for k in ("l3_overlay", "y_scatter", "split_last", "split_first_dn", "restream")
               if p.get(k, False)]
        bad += [k for k, v in (("rows", ROWS), ("post_blocks", 1), ("x_blocks", 1)) if int(p.get(k, v)) != v]
        bad += [f"{k} {p[k]}" for k, v in (("sync", "edge"), ("reduce", "chain"), ("x_cast", "multicast"))
                if str(p.get(k, v)) != v]
        if bad:
            raise ValueError(f"params {bad}: the decode is one pass of one {ROWS}-row tile, one post block, X from HBM "
                             f"by one multicast, the ChainGather reduce, edge sync")
        # params x_row (default true): the HBM holds the token's INT8 row, which every cluster scatters into row 0 of its
        # zeroed X tile; false: the HBM holds the whole tile in the A layout
        self.x_row = bool(p.get("x_row", True))
        # params dn_gemv: the down slices on VersaCore's one-row GEMV shape (the paired INT4 chunks). Every GEMV of a GEMM
        # core follows its last gemm_full, so no gemm_minimal inherits a GEMV's set-up
        self.dn_gemv = bool(p.get("dn_gemv", False))
        # params gu_gemv: gate|up on the GEMV too, X and every activation a row (the module docstring)
        self.gu_gemv = bool(p.get("gu_gemv", False))
        if self.gu_gemv and not (self.dn_gemv and self.x_row):
            raise ValueError("params gu_gemv: with dn_gemv (the A8 row feeds the down GEMVs) and x_row (X is the "
                             "token's row, in a_row)")
        self.a_blk = 8 if self.gu_gemv else 64      # the down GEMVs' A: the A8 row (a_row) or row 0 of the A8 tile
        # params y_dequant: the root's SIMD multiplies Y_raw's row by s_dn (one task, before the HBM write), so the HBM
        # holds Y16; default false: Y_raw, as the prefill leaves it
        self.y_dequant = bool(p.get("y_dequant", False))
        # the prefill's builder runs the tile as one pass of ROWS tokens
        super().__init__(args, dict(p, tokens=ROWS, rows=ROWS, post_blocks=1, x_blocks=1), plat, hw, shared=shared)
        self.T = 1

    def mem_handles(self):
        """PBuild's data, golden and memory of the one-token tile (this class's _data, _stage and _buffers: X and Y_raw's
        row in the root memory chip's HBM, the 16-row or, with gu_gemv, the one-row L1 buffers); then X's multicast as
        what the HBM holds, one tile behind one flag."""
        super().mem_handles()
        # x_hbm_bcasts multicasts what the HBM holds as one tile behind one flag
        self.XT, self.TPP = self.XR, 1
        if self.gu_gemv:
            print(f"[{self.TAG}] decode: one token as rows, gate|up and down on the GEMV; X from HBM as the row in "
                  f"a_row ({self.XR} B), {'Y16' if self.y_dequant else 'Y_raw'} ({2 * self.D} B) into HBM; "
                  f"{self.NDN // self.NC} slices a cluster, "
                  f"{self.NDN // self.NC // self.chain_group} gathers a cluster")
            return
        print(f"[{self.TAG}] decode: one token, row 0 of a {ROWS}-row tile; X from HBM as "
              f"{'the row' if self.x_row else 'the tile'} ({self.XR} B), {'Y16' if self.y_dequant else 'Y_raw'} "
              f"({2 * self.D} B) into HBM; down on the "
              f"{'GEMV' if self.dn_gemv else 'GEMM'}; {self.NDN // self.NC} slices a cluster, "
              f"{self.NDN // self.NC // self.chain_group} gathers a cluster")

    def _data(self, args, p):
        """The FFN's data and golden of the one token."""
        cache = os.path.join(fp.REPO, p.get("weights", "target/sim/automation/sweep/llm_blocks/ffn/cache"))
        return ffn_data.FFNData(cache, p.get("model", "dsv2-lite-l0"), seed=int(p.get("seed", 1)),
                                wbits=self.wbits, tokens=1, inter=p.get("inter"))

    # ---- data -----------------------------------------------------------------------------
    def x_tile(self):
        """X as every cluster's GEMMs read it: the 16-row tile in the A layout, the token's row 0, the other rows zero."""
        t = np.zeros((ROWS, self.D), dtype=np.int8)
        t[0] = self.d.hq[0]
        return self.lay.to_a(t, self.d.mesh).astype(np.int8)

    def x_a(self):
        """X as the HBM holds it: the token's INT8 row (x_row; gu_gemv: in a_row) or the whole tile."""
        if self.gu_gemv:
            return gemv_utils.a_row_bytes(self.d.hq[0])
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
        for cl, (c0, n) in self.cols.items():
            k = self.chip(cl)
            loc = (self.mem[k] >> 4, self.mem[k] & 15)
            parts, offs, sc = [], [], []
            for b0, nb_ in self.chunks[cl]:
                cols = self._gu_cols(cl, b0, nb_)
                parts.append(self._pack_b(d.gu_q[:, cols]))
                if self.gu_gemv:                # the GEMV's row: the chunk's column factors, [gate | up]
                    sc.append(d.gu_f[cols].astype(np.float16))
                    continue
                s = np.tile(d.gu_f[cols].astype(np.float16), (16, 1))
                sc.append(lay.to_d(s, d.mesh))  # one row tile: its scale tile
            kc0, kn = self.kcols[k]             # the chip's columns, this cluster's share of the outputs
            for o0, w in self.dn_sl[self.local(cl)]:
                parts.append(self._pack_b(d.dn_q[kc0:kc0 + kn, o0:o0 + w]))
            off = 0
            for b in parts:
                offs.append((off, b.size))
                off += b.size
            self.img[cl] = st.put_hbm(f"{T}_w_cl{cl}", np.concatenate(parts).astype(np.int8), mem_chip=loc)
            self.chunk_off[cl] = offs
            s = np.concatenate(sc).astype(np.float16)
            self.sdup[cl] = (st.put_hbm(f"{T}_sgu_cl{cl}", s.view(np.int8), mem_chip=loc), s.nbytes)

    def _buffers(self):
        """The prefill's buffers at 16 rows, and Y_raw's row on the root's cluster 0. dn_gemv: an output slot (and a
        fold slot) holds a row of the slice, so the slots are SLICE = a row apart in the same buffers. gu_gemv: rows
        only (_row_buffers)."""
        if self.gu_gemv:
            self._row_buffers()
        else:
            super()._buffers()
        self.yrow = self.cx_chip(self.ROOT).at(0).l1(f"{self.TAG}_yrow", 2 * self.D)
        if self.y_dequant:
            self.sdn = self.cx_chip(self.ROOT).at(0).l1(f"{self.TAG}_sdn", 2 * self.D)
        if self.dn_gemv:
            self.SLICE = 2 * max(w for v in self.dn_sl.values() for _, w in v)

    def _row_buffers(self):
        """gu_gemv: every buffer a row. The exchange buffer's output slots and the root's fold slots a row of a slice
        each (SLICE); per cluster X (a_row), its gate|up rows (FP16, chunk j's at its first gate block x 64 B, so no
        chunk's row overlaps another's), their factors, its A16 row and the row's a_row form in one padded buffer (the
        spread's whole beats), and the chip's A8 row (a_row)."""
        self.SLICE = 2 * max(w for v in self.dn_sl.values() for _, w in v)
        self.xch_slots = self.out_slots
        self.xch_bytes = self.xch_slots * self.SLICE
        self.xch = {(k, c): self.cx_chip(k).at(c).l1(f"AAAA_{self.TAG}_xch", self.xch_bytes)
                    for k in self.chips for c in range(self.NC)}
        self.p1c = {c: self.cx_chip(self.ROOT).at(c).l1(f"{self.TAG}_p1c", 2 * self.chain_group * self.SLICE)
                    for c in range(self.NC)}
        self.xbuf, self.sbuf, self.grow, self.rowp, self.a16r, self.ar16, self.a8 = {}, {}, {}, {}, {}, {}, {}
        for cl, (_, n) in self.cols.items():
            l1 = lambda name, nb: self.cx(cl).l1(f"{self.TAG}_{name}", nb)
            self.xbuf[cl] = [l1("xr", self.XR)]
            self.sbuf[cl] = l1("sgu", self.sdup[cl][1])
            self.grow[cl] = l1("gr", 2 * 2 * n)
            a16 = -(-2 * n // (LANES * ATOM)) * LANES * ATOM    # the A16 row, whole xDMA beats
            self.rowp[cl] = l1("rowp", 3 * a16)                  # [A16 row | its a_row form, twice as long]
            self.a16r[cl], self.ar16[cl] = self.rowp[cl].view(0), self.rowp[cl].view(a16)
            a8 = l1("a8r", 2 * self.kcols[self.chip(cl)][1])
            self.a8[cl] = [a8, a8]

    def _xland(self, off, chip_id=None):
        """Where X lands in every chip's L3, at byte `off`."""
        return BingoMemSymbol(f"{self.TAG}_xland", offset=off, chip_id=chip_id)

    def slice_bytes(self, w):
        """The bytes of a w-column slice of the down output: 16 rows in the D layout, or (dn_gemv) the token's row."""
        return (1 if self.dn_gemv else ROWS) * w * 2

    # ---- X --------------------------------------------------------------------------------
    def pre_window(self):
        """The prefill's (the scales into L1), with x_row every X tile zeroed on its cluster's xDMA, with y_dequant s_dn
        into the root's cluster 0. gu_gemv: no X tile; every cluster's A16 row and its a_row form zeroed instead (the
        a_row form's free bytes stay zero, so the A8 row's do)."""
        pre = super().pre_window()
        if self.y_dequant:
            R = self.ROOT
            pre[R].append(self.node(R, 0, self.roles["dm"], "sdn_load", "__snax_bingo_kernel_idma_1d_copy",
                                    SnaxBingoKernelIdma1dCopyArgs(self.h_sdn, self.sdn, 2 * self.D), pre[R][-1:]))
        if self.gu_gemv:
            for cl in range(self.N_CL):
                k, c = self.chip(cl), self.local(cl)
                pre[k].append(self.node(k, c, self.roles["xdma"], f"row_zero", "__snax_bingo_kernel_xdma_memset",
                                        SnaxBingoKernelXdmaMemsetArgs(self.rowp[cl], self.rowp[cl].size,
                                                                      SnaxBingoKernelXdmaMemsetArgs.PATTERN_ZERO), []))
            return pre
        for cl in range(self.N_CL if self.x_row else 0):
            k, c = self.chip(cl), self.local(cl)
            pre[k].append(self.node(k, c, self.roles["xdma"], f"x_zero", "__snax_bingo_kernel_xdma_memset",
                                    SnaxBingoKernelXdmaMemsetArgs(self.xbuf[cl][0], self.XB,
                                                                  SnaxBingoKernelXdmaMemsetArgs.PATTERN_ZERO), []))
        return pre

    def stage_x(self, i, src, after):
        """x_row: the token's row into row 0 of every cluster's X tile, behind the multicast's flag: a wait on the flag,
        then one 2-D copy from the chip's L3, a 4-byte run (the row's 4 values of a 4-column block) into each 64-byte
        block of the A layout. Without x_row the prefill's flagged fetch of the tile. gu_gemv: the a_row row into L1 in
        one flagged fetch. {cl: the task after which it is there}."""
        if not self.x_row:
            return super().stage_x(i, src, after)
        T, dm, ts = self.TAG, self.roles["dm"], self.d.mesh[1]
        if self.gu_gemv:
            out = {}
            for cl in range(self.N_CL):
                k, c = self.chip(cl), self.local(cl)
                out[cl] = self.node(k, c, dm, f"x_wait_i{i}", "__snax_bingo_kernel_idma_fetch_flagged",
                                   SnaxBingoKernelIdmaFetchFlaggedArgs(self._xland(0), self.xbuf[cl][0], self.XR,
                                                                       BingoMemSymbol(f"{T}_xtflag"), 1, 1),
                                   list(after.get(cl, [])))
                # the multicast's flag says the row has landed: an order edge, not a dependency
                self.dfg.bingo_add_order_edge(src, out[cl])
            return out
        out = {}
        for cl in range(self.N_CL):
            k, c = self.chip(cl), self.local(cl)
            w = self.node(k, c, dm, f"x_flag", "__snax_bingo_kernel_wait_flags",
                          SnaxBingoKernelWaitFlagsArgs(BingoMemSymbol(f"{T}_xtflag"), 1, 1), list(after.get(cl, [])))
            # the multicast's flag says the row has landed: an order edge, not a dependency
            self.dfg.bingo_add_order_edge(src, w)
            out[cl] = self.node(k, c, dm, f"x_wait_i{i}", "__snax_bingo_kernel_idma_2d_copy",
                               SnaxBingoKernelIdma2dCopyArgs(self._xland(0), self.xbuf[cl][0], ts, ts, ROWS * ts,
                                                             self.D // ts), [w])
        return out

    # ---- gu_gemv: gate|up and the A8 as rows ------------------------------------------------
    def stage_gu(self, i, x):
        """gu_gemv: every cluster's gate|up chunks, each a GEMV of the X row into the chunk's FP16 row as it lands, the
        row dequantised and its gate half through SwiGLU with its up half into the chunk's columns of the A16 row.
        Without gu_gemv the prefill's. Returns {cl: the last SwiGLU}, {cl: the last GEMV}."""
        if not self.gu_gemv:
            return super().stage_gu(i, x)
        gemm, simd = self.roles["gemm"], self.roles["simd"]
        last_sw, last_gemm = {}, {}
        for cl in self.cols:
            k, c = self.chip(cl), self.local(cl)
            sw = None
            for j, (b0, nb_) in enumerate(self.chunks[cl]):
                ld, si, slab = self._load(cl, i, j, [x[cl]] if j == 0 else ())
                N, off = 2 * nb_ * BLK, 4 * BLK * b0    # its gate columns then its up columns; 64 B a gate block
                row = self.grow[cl].view(off)
                gm = self.node(k, c, gemm, f"gu_c{j}_i{i}", "__snax_bingo_kernel_gemv",
                               SnaxBingoKernelGemvArgs(self.xbuf[cl][0], slab, row, K=self.D, N=N, d_shift=self.d.k_x,
                                                       a_blk=8, w4=True), [ld, x[cl]])
                self.ls[cl].release(si, gm)
                last_gemm[cl] = gm
                dq = self.node(k, c, simd, f"deq_c{j}_i{i}", "__snax_bingo_kernel_simd_stream_elementwise",
                               SnaxBingoKernelSimdMulF16Args(row, self.sbuf[cl].view(off), row, N // 32, 32), [gm])
                sw = self.node(k, c, simd, f"swi_c{j}_i{i}", "__snax_bingo_kernel_simd_swiglu_f16_f16",
                               SnaxBingoKernelSimdSwigluF16F16Args(row, self.grow[cl].view(off + 2 * nb_ * BLK),
                                                                   self.a16r[cl].view(2 * BLK * b0), 1, nb_ * BLK),
                               [dq] + ([sw] if sw is not None else []))     # in turn: the last stands for every one
            last_sw[cl] = sw
        return last_sw, last_gemm

    def stage_post(self, i, sw):
        """gu_gemv: behind each cluster's last SwiGLU its A16 row spread into a_row's FP16 form (xDMA: 8-byte atoms
        16 bytes apart, whole beats), quantised into the cluster's run of the chip's A8 row (SIMD), and that run copied
        into the other cluster's A8 row (DM; self.ex_push, named as the prefill's row-tile copy). Without gu_gemv the
        prefill's. {cl: the quantise}."""
        if not self.gu_gemv:
            return super().stage_post(i, sw)
        out = {}
        for cl, (_, n) in self.cols.items():
            k, c = self.chip(cl), self.local(cl)
            kn = self.kcols[k][1]
            other = self.clusters(k)[1 - c]
            off = 2 * (kn - n) if c else 0     # this cluster's columns sit after the first cluster's, 2 B a column
            beats = -(-n // (4 * LANES))
            sp = self.node(k, c, self.roles["xdma"], f"a2r_i{i}", "__snax_bingo_kernel_xdma_6d",
                           SnaxBingoKernelXdma6dArgs(self.a16r[cl], self.ar16[cl], ATOM, 2 * ATOM, [LANES * ATOM],
                                                     [beats], [LANES * 2 * ATOM], [beats]), [sw[cl]])
            run = self.a8[cl][0].view(off)
            q = self.node(k, c, self.roles["simd"], f"quant_i{i}", "__snax_bingo_kernel_simd_fp16_to_int8",
                          SnaxBingoKernelSimdFp16ToInt8Args(self.ar16[cl], run, beats=simd_beats(4 * n), rows=1,
                                                            inv_scale_f32bits=self.d.inv_a), [sp])
            self.ex_push.setdefault((cl, i), []).append(
                self.node(k, c, self.roles["dm"], f"a8_push_m0" if i == 0 else f"a8_push_i{i}m0",
                          "__snax_bingo_kernel_idma_1d_copy",
                          SnaxBingoKernelIdma1dCopyArgs(run, self.a8[other][0].view(off), 2 * n), [q]))
            self.quant[(cl, i)] = out[cl] = q
        return out

    def _check_gemv_only(self, first):
        """gu_gemv: every array task this block made (node ids from `first`) is a GEMV, so the GEMM-config reuse (which
        tracks only GEMM kernels) finds no GEMM here to turn into a gemm_minimal behind a GEMV."""
        gemm = self.roles["gemm"]
        arr = [t for t in self.dfg.node_list if t.node_id >= first and t.assigned_core_id == gemm
               and (t.kernel_name or "").startswith("__snax")]
        bad = [t.node_name for t in arr if t.kernel_name != "__snax_bingo_kernel_gemv"]
        if bad:
            raise ValueError(f"params gu_gemv: array tasks that are not GEMVs: {bad[:4]}")
        print(f"[{self.TAG}] gu_gemv: {len(arr)} array tasks, every one a GEMV (gate|up and down)")

    # ---- the pass ---------------------------------------------------------------------------
    def _passes(self, go, after_x):
        """The one pass: X from HBM, the gate|up chunks and the post-processing (PBuild.stage_gu, stage_post), then the
        down slices, each cluster's in output slots j mod out_slots, one ChainGather a group of chain_group slices, the
        token's row of each folded group into Y_raw's row (a gather late), and Y_raw into HBM. gu_gemv:
        X, gate|up and the A8 as rows (stage_x, stage_gu, stage_post here), the down GEMVs on the A8 row."""
        R, dm, xdma, gemm = self.ROOT, self.roles["dm"], self.roles["xdma"], self.roles["gemm"]
        block0 = max((t.node_id for t in self.dfg.node_list), default=-1) + 1     # this block's first node id
        self.x_bc = self.x_hbm_bcasts(go)
        x = self.stage_x(0, self.x_bc[0], after_x)
        sw, _ = self.stage_gu(0, x)
        post = self.stage_post(0, sw)
        other = lambda cl: self.clusters(self.chip(cl))[1 - self.local(cl)]
        ex_last = {cl: self.ex_push[(cl, 0)][-1] for cl in range(self.N_CL)}
        SL, others, h, GS = self.SLICE, self.order[1:], self.NDN // self.NC, self.chain_group
        p1x = {c: f"((uint64_t)({self.p1c[c].get_c_var_name()}))" for c in range(self.NC)}
        lag = 1                                 # a gather's stores follow the next gather (two fold slots)
        xch_reader, p1_reader, pending, last_st, all_st = {}, {}, [], {}, []
        last_red = None
        # params chains_after_stream: the gathers are made after every down GEMM, the first behind every cluster's last
        # one, which follows the cluster's last ring load: every gather runs after every chunk of every memory chip's
        # push has landed. No slot is then reused before its gather, so out_slots holds every slice of a cluster
        after_stream = bool(self.p.get("chains_after_stream", False))
        if after_stream and self.out_slots < h:
            raise ValueError(f"params chains_after_stream: out_slots={self.out_slots} must hold the {h} slices a "
                             f"cluster (no output slot is read by a gather before the last down GEMM)")

        def load(jl):
            return {cl: self._load(cl, 0, len(self.chunks[cl]) + jl) for cl in range(self.N_CL)}

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
            all_st.extend(sts)
            p1_reader[(c, s)] = sts

        def gathers(jl, dn, last_dn):
            # the group's gathers: slices j0 .. jl, in output slots so0 .. so (back to back), into the root's fold slot s;
            # last_dn: more tasks the first gather follows
            nonlocal last_red
            j0 = jl + 1 - GS
            s, so0 = (j0 // GS) % 2, j0 % self.out_slots
            for c in range(self.NC):
                j = c * h + j0
                # params chain_alt: cluster 1's gathers run the ring the other way
                ordr = others[::-1] if self.chain_alt and c == 1 else others
                ga = SnaxBingoKernelXdmaChainGatherArgs.fold_into(
                    cm.remote, R, c, ordr, src=self.xch_addr(c, so0 * SL), dst=f"({p1x[c]} + {s * GS * SL})",
                    size=(GS - 1) * SL + self.slice_bytes(self.dn_sl[c][jl][1]),
                    junction=self.jct["Elementwise"], jct_csr0=fp.EW_F16_ADD, holds=[self.p1c[c]])
                # the A8 copies into cluster c precede its gathers through its own down GEMMs
                first = [last_red] if last_red is not None else []
                if j0 == 0 and c == 0:
                    first.append(self.x_bc[-1])     # X's multicast before the first gather
                    first += list(last_dn)
                # chains_after_stream: a gather after the first follows it (last_red), and so every down GEMM already
                ins = [] if after_stream and last_red is not None else [dn[cl] for cl in range(self.N_CL)
                                                                        if self.local(cl) == c]
                red = self.node(R, c, xdma, f"y_chain_j{j}", "__snax_bingo_kernel_xdma_chain_gather", ga,
                                ins + p1_reader.get((c, s), []) + first)
                last_red = red
                for k in self.chips:
                    for o in range(so0, so0 + GS):
                        xch_reader[(k, c, o)] = [red]
                pending.append((c, j, red, s))

        lds = load(0)
        prev_dn, groups = {}, []
        for jl in range(h):
            so = jl % self.out_slots                # the clusters' output slot
            dn = {}
            for cl in range(self.N_CL):
                k, c = self.chip(cl), self.local(cl)
                ld, si, slab = lds[cl]
                kn = self.kcols[k][1]
                w = self.dn_sl[c][jl][1]
                deps = [ld, post[cl], ex_last[other(cl)]] + xch_reader.get((k, c, so), [])
                dst = self.xch[(k, c)].view(so * SL)
                if self.dn_gemv:
                    dn[cl] = self.node(k, c, gemm, f"dn_c{c * h + jl}_i0", "__snax_bingo_kernel_gemv",
                                      SnaxBingoKernelGemvArgs(self.a8[cl][0], slab, dst, K=kn, N=w, d_shift=self.d.k_dn,
                                                              a_blk=self.a_blk, w4=True), deps)
                else:
                    dn[cl] = self.node(k, c, gemm, f"dn_c{c * h + jl}_i0", "__snax_bingo_kernel_gemm_full",
                                      SnaxBingoKernelGemmFullArgs(
                                          input_A_addr=self.a8[cl][0], input_B_addr=slab, input_C_addr=0,
                                          output_D_addr=dst, M=self.MT, K=kn // 4, N=w // BLK, array_shape_idx=0,
                                          transpose_A=0, transpose_B=0, accumPrevC=0, int32tofp16_enable=1,
                                          int4_b_enable=self.int4_b, d_shift=self.d.k_dn), deps)
                self.ls[cl].release(si, dn[cl])
                if jl % GS:
                    # params chain_group: a gather waits for its group's last down GEMM on each chip only; the GEMM core
                    # runs its queue in order, so the group's GEMMs go into it in slice order
                    self.dfg.bingo_add_order_edge(prev_dn[cl], dn[cl])
                prev_dn[cl] = dn[cl]
            if jl + 1 < h:
                lds = load(jl + 1)
            while len(pending) > lag * self.NC:
                store(*pending.pop(0))
            if (jl + 1) % GS:
                continue
            if after_stream:
                groups.append((jl, dn))
                continue
            gathers(jl, dn, [])
        # chains_after_stream: the gathers in the loop's order, the first behind every cluster's last down GEMM
        for jl, dn in groups:
            while len(pending) > lag * self.NC:
                store(*pending.pop(0))
            gathers(jl, dn, [prev_dn[cl] for cl in range(self.N_CL)])
        while pending:
            store(*pending.pop(0))
        # Y's row into HBM once every copy into it is done (every gather has ended by then)
        ready = list(all_st)
        if self.y_dequant:                      # Y16 = Y_raw (.) s_dn, in place, the prefill's dequantise kernel
            ready = [self.node(R, 0, self.roles["simd"], "y_deq", "__snax_bingo_kernel_simd_stream_elementwise",
                               SnaxBingoKernelSimdMulF16Args(self.yrow, self.sdn, self.yrow, self.D // 32, 32), ready)]
        self.y_hbm = self.y_out(ready)
        if self.gu_gemv:
            self._check_gemv_only(block0)
        self.end_window([self.y_hbm])
        self._checks()

    def y_out(self, ready):
        """The root's DM core writes the row (yrow, on the root's cluster 0) into the HBM of the root's memory chip behind
        `ready`. Returns the task the end follows. A whole layer (llama3/common/llama3_decode_layer_mc.py) puts its final
        residual here instead."""
        return self.node(self.ROOT, 0, self.roles["dm"], "y_hbm", "__snax_bingo_kernel_idma_1d_copy",
                         SnaxBingoKernelIdma1dCopyArgs(self.yrow, self.h_yhbm, 2 * self.D), ready)

    def a8_root(self):
        """(the root's cluster-0 A8 buffer, its golden bytes, the bytes checked): the first 2 KiB of the chip's A8 in
        the A layout (the token's row 0, the zero rows), or with gu_gemv the chip's A8 row in a_row (its free bytes
        zero). A layer's block checks the same."""
        R = self.ROOT
        cl = self.clusters(R)[0]
        c0, ncol = self.kcols[R]                # the chip's A8, both clusters
        if self.gu_gemv:
            img = gemv_utils.a_row_bytes(self.d.a8[0, c0:c0 + ncol]).view(np.uint8)
        else:
            a8 = np.zeros((ROWS, ncol), dtype=np.int8)
            a8[0] = self.d.a8[0, c0:c0 + ncol]
            img = self.lay.to_a(a8, self.d.mesh).view(np.uint8)
        n = min(2048, img.size)
        return self.a8[cl][0], img[:n], n

    def _checks(self):
        """Y_raw's row in the HBM of the root's memory chip, every column (FP16, column order); the root's own A8
        (a8_root)."""
        R = self.ROOT
        y = np.ascontiguousarray(self.gold["y16" if self.y_dequant else "y_raw"][0].astype(np.float16)).view(np.uint8)
        checks = [("y_hbm", self.h_yhbm, y, y.size)]
        buf, img, n = self.a8_root()
        got = self.stash("a8_root", R, 0, buf, n, [])
        checks.append(("a8_root", got, img, n))
        self.run_checks(checks)
