# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""The driver of the full-layer prefill workloads (workloads/llama3/full_layer_prefill/<variant>): one Llama-3-8B decoder
layer (synthetic weights of the model's shapes, ../../common/llama3_data.py) over a fresh causal prompt on the 16 + 4
chiplet platform, in ONE graph, from the two blocks' mappings:

    x -> RMSNorm (input) -> attention -> + x = h -> RMSNorm (post-attention) -> SwiGLU FFN -> + h -> output

THE BLOCKS. The attention is ../../common/llama3_attn_mc.py's block (params attn): a query head a cluster, passes of `rows`
tokens, each pass's x multicast from the root memory chip's HBM by its push engine, y scattered row tile by row tile
into the tiles' owners. The FFN is ../../../ffn/common/ffn_prefill_mc.py's block (params ffn): X of each pass multicast
from the root memory chip's HBM by the same engine, the down projection split by output columns, ChainGathers into the
root, which scatters Y_raw row tile by row tile into the tiles' owners (ffn.y_scatter). params attn.passes and
layer.ffn_passes build only the first passes of the prompt (a proof of concept): the data, the scales, the causal mask
and the key/value buffers stay the whole prompt's.

THE BOUNDARIES GO THROUGH HBM. The layer's input x (FP16) sits in the memory chips' HBM. The input norm writes the
attention's x8 into the HBM region the attention's x multicast reads; the glue writes h (FP16) into its chip's memory chip
and the FFN's X (INT8, the FFN's A layout) into the HBM region the FFN's X multicast reads; the final residual writes the
output (FP16) into its chip's memory chip. Nothing of a pass's activations is kept whole in an L3. Every write into HBM is
a flagged put (the data, then a flag beside it in the same memory chip's HBM, one in-order path); the writer then waits
for its own flag there (a flagged fetch), and that wait is what a reader of the data follows: a multicast of x8 or X
waits for its pass's four waits, the final residual's read of h polls h's flag itself.

THE STEPS, one 16-row tile q at a time on the tile's owner chip (chips[q % 16], where both blocks scatter y), on cluster
params layer.cluster:
    norm   x16 rows from HBM, RMSNorm, times the input gain, reshaped to the A layout (xDMA), quantised (s_x), into the
           attention's x region (the root memory chip's HBM)
    glue   the tile's y (D layout) from the chip's L3 -> row major (xDMA), dequantised (W_O's factors), plus x16: h, into
           the chip's memory chip's HBM; RMSNorm of h, times the post-attention gain, reshaped, quantised (s_h), into the
           FFN's X region (the root memory chip's HBM)
    final  the tile's Y_raw (D layout) from the chip's L3 -> row major, dequantised (the down projection's factors), plus
           h (from HBM): the output, into the chip's memory chip's HBM

PASS BY PASS. A step's tasks are placed in each core's program (order keys) at the blocks' pass boundaries: the
attention's iteration i begins before sweep i ("top", after pass i's projections and K/V trade) and reaches W_O after
sweep i ("wo"); the FFN's pass i begins with its gate|up ("top") and reaches down slice j ("dn"). A step made later than
the point of one of its parts takes that point's place in the cores' programs.
    norm of pass p   at the top of attention iteration p - layer.norm_lead (iteration 0's top is after pass 0's trade;
                     a pass p < norm_lead before the attention's first tasks: the head)
    glue of pass p   its loads, the conversion, the dequantisation, the residual, the RMSNorm and the gain from the "wo"
                     point of iteration p + layer.glue_lag, behind the cluster's last softmax there (the SIMD then runs
                     them under the W_O GEMMs); the reshape, the quantise and the HBM writes from the top of iteration
                     p + glue_lag + 1 (or the attention's end)
    final of pass p  everything but the output's write from FFN pass p + layer.final_lag's down slice layer.final_slice
                     (Y_raw of pass p is complete once the root has stored that pass's last gathers, which the FFN's
                     store lag makes during the next pass's first down slices); the write from the top of pass
                     p + final_lag + 1 (or the FFN's end)

THE ROOT'S HOST runs every HBM multicast of both blocks and the root's weight prefetchers, one task after the other.
The attention's prefetcher on the root runs in parts (weight_prefetch_range) around each pass's x multicast: part 0
pushes pass 0's projection chunks, then x_0 is multicast (behind its four norms); part i pushes up to pass i - 1's first
W_O chunk, then x_i is multicast. A part ends once its last chunk is pushed, which needs the chunk n_slots before it read
from the ring: that read never depends on x_i (proved in finish, _check_prefetch_parts), so a late norm only holds the
part after it. The FFN's X multicasts follow the attention's last part, each behind its pass's glue; with more than one
FFN pass the FFN's prefetcher on the root pushes its first layer.ffn_pf_head chunks of each ring before the last pass's
multicast, the rest after it. On every other chip the FFN's prefetcher follows the attention's (their rings are separate
L3 regions), so the FFN's first chunks land under the attention's last pass.

THE COLLECTIVES. The multicasts in time: x_0, x_1, o8 of pass 0, x_2, o8 of pass 1, ..., o8 of the last pass, then the FFN's
X multicasts, then the FFN's ChainGathers. Each o8 multicast waits for the next pass's x multicast, each x multicast for the
o8 multicast two passes back; _check_collectives proves every pair of multicasts and ChainGathers ordered by a path.

CHECKS, byte for byte against the device's arithmetic end to end, sampled: a few 2 KiB windows of x8 and X in HBM, of y and
Y_raw in their owners' L3, and whole rows of h and of the output in HBM, spread over the passes and chips. After the end
stamp each sample is copied into one L3 buffer on the root (from HBM behind its flag), where the root's host compares it.

PARAMETERS (params.hjson): the platform and graph switches both blocks share (and args_phase_bytes: each cluster's task
args in phases of at most this many bytes), then `attn` (llama3_attn_mc.py's params), `ffn` (ffn_prefill_mc.py's) and
`layer`: cluster (0 | 1, the steps' cluster), ffn_passes (the FFN's built passes, at most attn.passes), norm_lead,
glue_lag, final_lag, final_slice, ffn_pf_head, inplace_norm (the input norm's RMSNorm in place: half its L1), piece_rows,
norm_cluster, glue_cluster, final_cluster, final_step, final_root_rows, x_slots, alias_l3, rope_hbm, ffn_part_chunk.
"""

import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WL = os.path.join(HERE, "..", "..", "..")
sys.path.insert(0, os.path.join(os.path.abspath(__file__).split("/target/sw/host/")[0], "util", "sim", "multichip"))
sys.path.insert(0, os.path.join(os.path.abspath(__file__).split("/target/sw/host/")[0], "util", "sim", "common"))
from driver import repo_root, run_driver  # noqa: E402
REPO = repo_root(__file__)
sys.path.insert(0, f"{REPO}/target/sw/host/runtime/libbingo/mini_compiler")
sys.path.insert(0, os.path.join(WL, "ffn", "common"))
sys.path.insert(0, os.path.join(WL, "llama3", "common"))

import _bingo_paths  # noqa: F401,E402
from bingo_kernel_args import (  # noqa: E402
    HostBingoKernelHbmBcastArgs, HostBingoKernelWeightPrefetchRangeArgs, SnaxBingoKernelIdma1dCopyArgs,
    SnaxBingoKernelIdma2dCopyArgs, SnaxBingoKernelIdmaFetchFlaggedArgs, SnaxBingoKernelIdmaPutFlaggedArgs,
    SnaxBingoKernelSimdAddF16Args,
    SnaxBingoKernelSimdFp16ToInt8Args, SnaxBingoKernelSimdMulTilesArgs, SnaxBingoKernelSimdRmsnormArgs,
    SnaxBingoKernelWaitFlagsArgs, simd_beats)
from bingo_mem_handle import BingoMemFixedAddr, BingoMemSymbol  # noqa: E402
from libs.comm.nest import convert_args  # noqa: E402

import chiplet_mc as cm  # noqa: E402
import ffn_data  # noqa: E402
import ffn_prefill_mc as fpm  # noqa: E402
import llama3_attn_mc as lam  # noqa: E402
import llama3_data  # noqa: E402

TR = 16                       # rows a tile: one A-layout row block
COPIES = ("__snax_bingo_kernel_idma_1d_copy", "__snax_bingo_kernel_idma_2d_copy", "__snax_bingo_kernel_idma_put_flagged")


class LayerStaging:
    """The layer's DataStaging, with changes for the blocks' arrays: put_hbm of a name in `zero_hbm` stages zeros of the
    array's size (the layer writes it inside the window, so the image must not hold the golden); put_zeros of a name in
    `drop` stages nothing (a buffer of a block the layer never reads or writes); put_zeros of a name in `sizes` stages that
    many bytes instead (a ring of a block's per-pass buffer); put_zeros of a name in `alias` stages nothing and the name
    becomes its region (finish stages each region once, at `sizes[region]` or the largest extent asked; lines are the
    defines); put of the name `rope[0]` stages the array in the HBM of each memory chip in `rope[1]` instead (rope_h)."""

    def __init__(self, st, zero_hbm=(), drop=(), alias=None, sizes=None, rope=None):
        self._st, self.zero_hbm, self.drop = st, set(zero_hbm), set(drop)
        self.zeroed, self.dropped = [], []
        self.alias, self.sizes, self.asked = dict(alias or {}), dict(sizes or {}), {}
        self.rope, self.rope_h = rope, {}

    def __getattr__(self, name):
        return getattr(self._st, name)

    def put_hbm(self, name, arr, mem_chip=None):
        if name in self.zero_hbm:
            arr = np.zeros(np.asarray(arr).nbytes, dtype=np.uint8)
            self.zeroed.append(name)
        return self._st.put_hbm(name, arr, mem_chip=mem_chip)

    def put(self, name, ctype, arr, mem_chip=None):
        if self.rope is not None and name == self.rope[0]:
            raw = np.ascontiguousarray(arr).view(np.uint8).reshape(-1)
            for m in self.rope[1]:
                self.rope_h[m] = self._st.put_hbm(f"{name}_m{m:02x}", raw, mem_chip=(m >> 4, m & 15))
            return BingoMemSymbol(name)
        return self._st.put(name, ctype, arr, mem_chip=mem_chip)

    def put_zeros(self, name, ctype, count, mem_chip=None, preload=False):
        if name in self.drop:
            self.dropped.append(name)
            return BingoMemSymbol(name)
        width = {"uint8_t": 1, "int8_t": 1, "uint16_t": 2, "int16_t": 2, "uint32_t": 4, "int32_t": 4}[ctype]
        if name in self.alias:
            self.asked[name] = count * width
            return BingoMemSymbol(name)
        if name in self.sizes:
            count = self.sizes[name] // width
        return self._st.put_zeros(name, ctype, count, mem_chip=mem_chip, preload=preload)

    def finish(self):
        for region in sorted(set(self.alias.values())):
            ext = [self.asked[n] for n, r in self.alias.items() if r == region and n in self.asked]
            if ext:
                self._st.put_zeros(region, "uint8_t", self.sizes.get(region) or max(ext), preload=True)

    def lines(self):
        names = [(n, r) for n, r in sorted(self.alias.items()) if n in self.asked]
        if not names:
            return []
        return ["// one L3 region, several names (llama3_layer_mc.py LayerStaging)"] + [f"#define {n} {r}" for n, r in names]


def _hbm(h, off=0):
    return BingoMemFixedAddr(h.address + off, mem_level="HBM")


def _range_args(b, lo, hi):
    """The weight prefetcher `b` (its args) over chunks [lo, hi) of each ring (hi 0: to the ring's end)."""
    return HostBingoKernelWeightPrefetchRangeArgs(
        b.sched_addr, b.ring_addr, b.flag_addr, b.release_addr, b.seq_addr, b.rec_addr, b.rec_slots, b.n_rings,
        b.n_slots, b.slot_bytes, b.memchip_id, batch=b.batch, policy=b.policy, engine=b.engine,
        batch_routed=b.batch_routed, queued=b.queued, queued_inflight_routed=b.queued_inflight_routed, head=b.head,
        batch_head=b.batch_head, chunk_lo=lo, chunk_hi=hi)


# ---- the blocks, over the layer's graph ------------------------------------------------------------------------------
class AttnPart(lam.ABuild):
    """The attention block in the layer: its window opened by the layer, its x multicasts the layer's (behind the
    norms, between parts of the root's prefetcher), its pass boundaries the layer's step points, its checks the
    layer's."""
    stream_name = "zs_la_stream"

    def __init__(self, layer, args, p, plat, hw):
        self.layer = layer
        super().__init__(args, p, plat, hw, shared=layer)

    def setup(self, extra=None):
        self.go = self.layer.open_window(extra)
        return dict(self.go)

    def x_puts_up_front(self, go):
        return self.layer.attn_x_puts(go)

    def pass_start(self, i):
        self.layer.at(("attn_top", i))

    def stage_trade_kv_split(self, i, own):
        super().stage_trade_kv_split(i, own)
        if i == 0:
            # iteration 0's top: after pass 0's K/V trade, before sweep 0 (the later iterations' is pass_start)
            self.layer.at(("attn_top", 0))

    def stage_wo_cols(self, i, post_last, x, stores, between=None):
        L = self.layer
        L.at(("attn_wo", i))
        # pass i's o8 multicast follows pass i + 1's x multicast (the one before it in time)
        self._x_put_last = L.x_bc[min(i + 1, self.NP - 1)]
        return super().stage_wo_cols(i, post_last, x, stores, between)

    def end_window(self, after):
        self.layer.attn_ends = list(after)

    def run_checks(self, checks):
        pass

    def xland(self, off=0):
        # params layer.x_slots: pass i's x lands in slot i mod x_slots of a ring
        S = self.layer.x_slots
        if not S:
            return super().xland(off)
        return BingoMemSymbol(f"{self.TAG}_xland", offset=off // self.XB % S * self.XB + off % self.XB)

    def copy(self, cl, name, src, dst, nb, deps):
        # params layer.rope_hbm: the RoPE tables from the cluster's memory chip's HBM instead of L3
        L = self.layer
        if L.rope_hbm and isinstance(src, BingoMemSymbol) and src.symbol_name == f"{self.TAG}_rope":
            src = BingoMemFixedAddr(L.st.rope_h[self.mem[self.chip(cl)]].address + src.offset, mem_level="HBM")
        return super().copy(cl, name, src, dst, nb, deps)


class FfnPart(fpm.PBuild):
    """The FFN block in the layer: its rows the layer's h, its window opened by the layer's, its X multicasts the
    layer's (behind the glue), its pass boundaries the layer's step points, its end and checks the layer's."""
    stream_name = "zs_pf_stream"

    def __init__(self, layer, args, p, plat, hw):
        self.layer = layer
        super().__init__(args, p, plat, hw, shared=layer)

    def _data(self, args, p):
        L = self.layer
        # the FFN's rows: h of the built passes; its static scales calibrated on h of the whole prompt
        return ffn_data.FFNData("", llama3_data.MODEL, seed=int(p.get("seed", 1)), wbits=self.wbits,
                                tokens=self.T, inter=p.get("inter"), rows=dict(x16=L.h16[:self.T], cal=L.h16))

    def setup(self, extra=None):
        self.go = self.layer.ffn_window(extra)
        return dict(self.go)

    def x_hbm_bcasts(self, go):
        return self.layer.ffn_x_puts(go)

    def pass_start(self, i):
        self.layer.at(("ffn_top", i))

    def _load(self, cl, i, j, deps=()):
        if cl == 0 and j >= len(self.chunks[0]):
            self.layer.at(("ffn_dn", i, j - len(self.chunks[0])))
        return super()._load(cl, i, j, deps)

    def end_window(self, after):
        L = self.layer
        L.close_window(list(after))
        self.end, self.check_copies = L.end, L.check_copies

    def _checks(self):
        pass

    def _xland(self, off, chip_id=None):
        # params layer.x_slots: pass i's X lands in slot i mod x_slots of a ring
        S = self.layer.x_slots
        if not S:
            return super()._xland(off, chip_id)
        return BingoMemSymbol(f"{self.TAG}_xland", offset=off // self.XB % S * self.XB + off % self.XB, chip_id=chip_id)


# ---- the layer ---------------------------------------------------------------------------------------------------------
class LayerBuild(cm.ChipletBuild):
    """The whole prefill layer (the module docstring), its blocks parts of the layer's graph. run_driver makes the
    program in four steps: __init__ (the parameters, the layer's staging, the attention part made), mem_handles() (the
    data, the golden and the memory: the attention's, the FFN part made on the passes the attention builds and its
    own, then the layer's), build() (the task graph: the attention's with the norms and the glue at its step points,
    the FFN's with the final, the checks) and finish() (the checks of the graph, then the compile into the C header
    and the data image)."""
    TAG = "ll"

    def __init__(self, args, p, plat, hw):
        """The parameters; no data, memory or task yet. ChipletBuild.__init__ with the switches both blocks share;
        params layer (the steps' clusters, pieces and lags, x_slots, alias_l3, rope_hbm); the layer's staging
        (LayerStaging: the HBM regions the steps write, the FFN's dropped root Y, the x rings and the aliased L3
        regions, the RoPE tables in HBM); the synthetic FFN registered (llama3_data.register_ffn); the attention part
        (AttnPart), checked for what the layer needs of it; the FFN's params. The FFN part is made in mem_handles: its
        passes default to the attention's built ones, which the attention's data gives."""
        common = {k: v for k, v in p.items() if k not in ("attn", "ffn", "layer")}
        lp = self.lp = dict(p.get("layer", {}))
        super().__init__(args, common, plat, hw)
        self.SC = int(lp.get("cluster", 1))
        if self.SC not in range(self.NC):
            raise ValueError(f"params layer.cluster={self.SC}: a cluster index")
        # params layer.x_slots: both blocks' x landing areas a ring of this many passes (0: one area a pass); alias_l3:
        # one L3 region for the two blocks' x rings, one for their y tiles, one for their weight rings (the FFN's
        # prefetchers then follow the attention's last ring loads); rope_hbm: the attention's RoPE tables in HBM
        self.x_slots = int(lp.get("x_slots", 0))
        self.alias_l3 = bool(lp.get("alias_l3", False))
        self.rope_hbm = bool(lp.get("rope_hbm", False))
        if self.x_slots and self.x_slots < 3:
            raise ValueError(f"params layer.x_slots={self.x_slots}: at least 3 (pass i's x lands while pass i - 2's o8 "
                             f"gathers read their slot)")
        if self.alias_l3 and not self.x_slots:
            raise ValueError("params layer.alias_l3: needs x_slots (the blocks' x areas share one ring)")
        hidden = llama3_data.LLAMA3_8B["hidden"]
        xa, xf = int(p["attn"].get("rows", 64)) * hidden, int(p["ffn"].get("rows", 64)) * hidden
        alias, sizes = {}, {}
        if self.x_slots and self.alias_l3:
            alias.update(la_xland="ll_xland", pf_xland="ll_xland")
            sizes["ll_xland"] = self.x_slots * max(xa, xf)
        elif self.x_slots:
            sizes.update(la_xland=self.x_slots * xa, pf_xland=self.x_slots * xf)
        if self.alias_l3:
            alias.update(la_yt="ll_yt", pf_yt="ll_yt", la_wring_slots="ll_wring", pf_wring_slots="ll_wring")
        rope = ("la_rope", sorted(set(self.mem.values()))) if self.rope_hbm else None
        # the blocks' input regions in HBM are written in the window; the FFN's root Y buffer is never used (y_scatter)
        self.st = LayerStaging(self.st, zero_hbm=("la_xhbm", "pf_xhbm"), drop=("pf_y",), alias=alias, sizes=sizes,
                               rope=rope)
        self.xch_zero = False
        # params layer.piece_rows (16 | 8 | 4): rows a step processes at a time inside the window (its L1 buffers that
        # many rows); norm_cluster, glue_cluster, final_cluster: each step's cluster (default layer.cluster);
        # final_step: down slices between two final pieces
        self.PR = int(lp.get("piece_rows", TR))
        if self.PR not in (TR, 8, 4):
            raise ValueError(f"params layer.piece_rows={self.PR}: 16, 8 or 4")
        self.NPC = TR // self.PR
        self.cl_norm = int(lp.get("norm_cluster", self.SC))
        self.cl_glue = int(lp.get("glue_cluster", self.SC))
        self.cl_final = int(lp.get("final_cluster", self.SC))
        self.final_step = int(lp.get("final_step", 2))
        # params layer.final_root_rows: rows of a final piece on the root chip (default piece_rows); fewer rows there, each
        # piece's buffers one after the other, keep the piece inside what the root's FFN leaves free in L1
        self.PR_root = int(lp.get("final_root_rows", self.PR))
        if TR % self.PR_root or self.PR_root > self.PR:
            raise ValueError(f"params layer.final_root_rows={self.PR_root}: divides {TR}, at most piece_rows")
        self.ffn_part_chunk = int(lp.get("ffn_part_chunk", 8))
        self.norm_lead = int(lp.get("norm_lead", 2))
        self.glue_lag = int(lp.get("glue_lag", 1))
        self.final_lag = int(lp.get("final_lag", 1))
        self.final_slice = int(lp.get("final_slice", 8))
        self.ffn_pf_head = int(lp.get("ffn_pf_head", 16))
        self.inplace_norm = bool(lp.get("inplace_norm", False))
        if self.norm_lead < 1 or self.glue_lag < 1 or self.final_lag < 1:
            raise ValueError("params layer.norm_lead, glue_lag, final_lag: at least 1 (a pass's step beside another pass)")
        if self.NPC > 1 and self.norm_lead < self.NPC + 1:
            # a piece a pass: the last piece's write follows that iteration's W_O loads, which the root's prefetcher pushes
            # before pass p's x multicast only up to iteration p - 2
            raise ValueError(f"params layer.norm_lead={self.norm_lead} with {self.NPC} pieces a tile: at least {self.NPC + 1}")
        self.ps = {}
        self.marks, self.pending, self.made = {}, [], set()
        self.seed = int(common.get("seed", 1))
        llama3_data.register_ffn(seed=self.seed)

        # ---- the attention ---------------------------------------------------------------------------------------
        pa = dict(common, **p["attn"])
        if pa.get("x_device"):
            raise ValueError("params attn.x_device: the layer's attention takes x8 from HBM (the input norm writes it)")
        A = self.attn = AttnPart(self, args, pa, plat, hw)
        if not (A.y_scatter and A.wo_cols and A.x_mem and A.kv_split):
            raise ValueError("params attn: the layer needs y_scatter, wo_cols, kv_split and x from HBM")
        if A.R % TR or self.T != A.T:
            raise ValueError(f"{A.T} tokens in passes of {A.R}: whole {TR}-row tiles, and the layer's tokens")
        self.NQ = A.R // TR                                       # tiles a pass

        # ---- the FFN's params (its part made in mem_handles) --------------------------------------------------------
        self.ffn_p = dict(common, **p["ffn"])
        if not self.ffn_p.get("y_scatter") or self.ffn_p.get("l3_overlay"):
            raise ValueError("params ffn: y_scatter, no l3_overlay (Y_raw goes to the tiles' owners, no pass of it whole "
                             "in the root's L3)")

    def mem_handles(self):
        """The data, the golden and the memory; the only tasks made are the blocks' prefetchers. The attention's
        (AttnPart's mem_handles), then the golden on its data: x and h. The FFN part (FfnPart), on params
        layer.ffn_passes of the attention's built passes (default all of them), then its data, golden and memory (its
        mem_handles, on h), its gain checked against the layer's, the output's golden. The layer's own data: per memory
        chip, in its HBM, x of the tiles its chips own, the regions the steps write (h, the output) and their flags, the
        constants (the gains, W_O's and the down projection's factors); in the root memory chip's HBM the flags of the
        x8 and X tiles; with pieces of fewer than 16 rows, the INT8 staging tiles in L3."""
        A, args = self.attn, self.args
        A.mem_handles()
        d = A.d
        fp = d.fp
        self.H = d.H
        self.x16 = np.asarray(d.x16, dtype=np.float16)
        self.h16 = fp.add16(d.y16(A.gold_y), self.x16)           # the residual stream after the attention

        # ---- the FFN ----------------------------------------------------------------------------------------------
        nf = int(self.lp.get("ffn_passes", A.NP))
        if not 0 < nf <= A.NP:
            raise ValueError(f"params layer.ffn_passes={nf}: 1 .. attn.passes ({A.NP}), the passes the glue makes")
        pf = dict(self.ffn_p)
        pf["model"] = llama3_data.MODEL
        pf["tokens"] = nf * int(pf.get("rows", A.R))
        F = self.ffn = FfnPart(self, args, pf, self.plat, self.hw)
        if F.ROWS != A.R:
            raise ValueError("params ffn.rows: the attention's rows (a pass's tiles are the same in both blocks)")
        F.mem_handles()                                           # its data on h
        if not np.array_equal(F.d.gain16, llama3_data.ffn_gain(self.seed).astype(np.float16)):
            raise AssertionError("the FFN block's gain and the layer's differ")
        self.out16 = fp.add16(fp.mul16(F.gold["y_raw"], F.d.dn_f[None, :]), self.h16[:F.T])

        # ---- the layer's own data, in HBM ---------------------------------------------------------------------------
        st, T, H = self.st, self.TAG, self.H
        self.NT, self.NTF = A.NP * self.NQ, F.T // TR             # tiles of the built attention / FFN passes
        self.TB = TR * H * 2                                      # a tile, FP16
        self.XTB = TR * H                                         # a tile, INT8
        self.CB = H * 2                                           # one constant vector, FP16
        self.tq, per_m = {}, {}                                   # tile -> (owner, its memory chip, index there)
        for q in range(self.NT):
            k = A.y_owner(q)[0]
            if k != F.y_owner(q)[0]:
                raise AssertionError(f"tile {q}: the blocks scatter y to different chips")
            m = self.mem[k]
            self.tq[q] = (k, m, len(per_m.setdefault(m, [])))
            per_m[m].append(q)
        loc = lambda m: (m >> 4, m & 15)
        cvec = np.concatenate([d.gain16, np.asarray(d.wo_f, dtype=np.float16), F.d.gain16,
                               np.asarray(F.d.dn_f, dtype=np.float16)]).astype(np.float16)
        self.h_x16, self.h_h, self.h_hf, self.h_out, self.h_of, self.h_c = {}, {}, {}, {}, {}, {}
        for m, qs in sorted(per_m.items()):
            rows = np.concatenate([self.x16[q * TR:(q + 1) * TR] for q in qs])
            self.h_x16[m] = st.put_hbm(f"{T}_x16_m{m:02x}", rows.view(np.uint8).reshape(-1), mem_chip=loc(m))
            self.h_h[m] = st.put_hbm(f"{T}_h_m{m:02x}", np.zeros(len(qs) * self.TB, np.uint8), mem_chip=loc(m))
            self.h_hf[m] = st.put_hbm(f"{T}_hflag_m{m:02x}", np.zeros(len(qs) * 64, np.uint8), mem_chip=loc(m))
            self.h_out[m] = st.put_hbm(f"{T}_out_m{m:02x}", np.zeros(len(qs) * self.TB, np.uint8), mem_chip=loc(m))
            self.h_of[m] = st.put_hbm(f"{T}_oflag_m{m:02x}", np.zeros(len(qs) * 64, np.uint8), mem_chip=loc(m))
        for m in sorted(set(self.mem.values())):
            self.h_c[m] = st.put_hbm(f"{T}_c_m{m:02x}", cvec.view(np.uint8), mem_chip=loc(m))
        R0 = self.mem[self.ROOT]
        if A.mem[A.ROOT] != R0 or F.mem[F.ROOT] != R0:
            raise AssertionError("the blocks' x multicasts are not the root memory chip's")
        self.h_nf = st.put_hbm(f"{T}_nflag", np.zeros(self.NT * 64, np.uint8), mem_chip=loc(R0))   # x8 tiles landed
        self.h_gf = st.put_hbm(f"{T}_gflag", np.zeros(self.NT * 64, np.uint8), mem_chip=loc(R0))   # X tiles landed
        self.n_conf = {i: [] for i in range(A.NP)}               # pass -> the norms' waits on their flags
        self.g_conf = {i: [] for i in range(A.NP)}               # pass -> the glue's
        self.f_out = []                                           # the output writes
        self.waits = []                                           # (wait task, block, tile) of the glue and the final
        if self.NPC > 1:
            # the input norm's and the glue's INT8 tile, row block by row block, in the chip's L3 before its put into HBM
            st.put_zeros(f"{T}_n8stg", "uint8_t", self.XTB)
            st.put_zeros(f"{T}_g8stg", "uint8_t", self.XTB)
            h = F.NDN // F.NC
            for k in (self.ROOT, self.chips[-1]):
                rows, n, step = self._final_shape(k)
                if self.final_slice + step * (n - 1) + 1 >= h:
                    raise ValueError(f"params layer.final_slice={self.final_slice}, final_step={self.final_step}: the last "
                                     f"{rows}-row piece's write past the {h} down slices of a pass")
            if A.XB != F.XB and self.alias_l3:
                raise ValueError("params layer.alias_l3: the blocks' passes differ in X bytes")
        print(f"[{T}] layer: {self.T} tokens, attention {A.NP} of {A.NP_T} passes, FFN {F.NI} passes of {F.ROWS}; steps on "
              f"cluster {self.SC} of each tile's owner; norm_lead {self.norm_lead}, glue_lag {self.glue_lag}, final_lag "
              f"{self.final_lag} (slice {self.final_slice}); |h| max {float(np.max(np.abs(self.h16))):.3g}")
        if self.NPC > 1 or self.x_slots or self.alias_l3 or self.rope_hbm:
            print(f"[{T}] pieces of {self.PR} rows (norm cluster {self.cl_norm}, glue {self.cl_glue}, final {self.cl_final}, "
                  f"final slices {self.final_slice}+{self.final_step}n); x ring {self.x_slots or 'off'}; alias_l3 "
                  f"{self.alias_l3}; rope_hbm {self.rope_hbm}")

    # ---- the step points -------------------------------------------------------------------------------------------
    def at(self, key):
        """A step point is reached (the blocks call it as they make their tasks): its place in the programs, then every
        pending step whose last part is placed here."""
        if key in self.marks:
            return
        self.marks[key] = self.dfg.id + 1
        todo = [s for s in self.pending if s[0] == key]
        self.pending = [s for s in self.pending if s[0] != key]
        for _, make in todo:
            make()

    def _point(self, block, kind, i, *rest):
        """The step point `kind` of pass/iteration i of `block` (attn | ffn), or the block's end past its last pass."""
        n = self.attn.NP if block == "attn" else self.ffn.NI
        if i < 0:
            return ("head",)
        if i >= n:
            return (f"{block}_end",)
        return (f"{block}_{kind}", i) + tuple(rest)

    def _schedule(self, last, make):
        if last in self.marks:
            make()
        else:
            self.pending.append((last, make))

    def _placed(self, tasks, key):
        """Tasks made later than the point `key` that belong there: the point must have been reached (the compiler
        places them in the cores' programs)."""
        if key not in self.marks:
            raise ValueError(f"step point {key}: not reached when a step placed there is made")

    # ---- helpers ----------------------------------------------------------------------------------------------------
    def _t(self, k, role, name, kernel, args, deps, c=None):
        return self.node(k, self.SC if c is None else c, self.roles[role], name, kernel, args, list(dict.fromkeys(deps)))

    def _l1(self, k, name, nb, c=None):
        """A step's L1 buffer, named to sort after the blocks' (the packer places by name)."""
        return self.cx_chip(k).at(self.SC if c is None else c).l1(f"zt_{self.TAG}_{name}", nb)

    def _scr(self, rows=TR):
        return SnaxBingoKernelSimdRmsnormArgs(0, 0, rows, self.H).scratch_bytes()

    def _rms(self, k, name, src, dst, scr, deps, rows=TR, c=None):
        return self._t(k, "simd", name, "__snax_bingo_kernel_simd_rmsnorm",
                       SnaxBingoKernelSimdRmsnormArgs(src, dst, rows, self.H, scratch_addr=scr), deps, c)

    def _mul_rows(self, k, name, a, vec, deps, rows=TR, c=None):
        H = self.H
        return self._t(k, "simd", name, "__snax_bingo_kernel_simd_mul_tiles",
                       SnaxBingoKernelSimdMulTilesArgs(a, vec, a, rows, H // 128, 2 * H, 2 * H), deps, c)

    def _quant(self, k, name, src, dst, inv, deps, nb=None, c=None):
        bits = int(np.asarray(inv, dtype=np.float32).reshape(()).view(np.uint32))
        return self._t(k, "simd", name, "__snax_bingo_kernel_simd_fp16_to_int8",
                       SnaxBingoKernelSimdFp16ToInt8Args(src, dst, beats=simd_beats(nb or self.TB), rows=1,
                                                         inv_scale_f32bits=bits), deps, c)

    def _cvt(self, k, name, src_l, dst_l, src, dst, deps, rows=TR, c=None):
        mesh = cm.MESH if rows == TR else (rows,) + tuple(cm.MESH[1:])
        return self._t(k, "xdma", name, "__snax_bingo_kernel_xdma_6d",
                       convert_args(src_l, dst_l, rows, self.H, mesh, 2, src, dst), deps, c)

    def _copy(self, k, name, src, dst, nb, deps, c=None):
        return self._t(k, "dm", name, "__snax_bingo_kernel_idma_1d_copy", SnaxBingoKernelIdma1dCopyArgs(src, dst, nb),
                       deps, c)

    def _put_hbm(self, k, name, src, dst, nb, flag, deps, c=None):
        """A flagged put into HBM: the data, then the flag beside it in the same memory chip (one in-order path)."""
        return self._t(k, "dm", name, "__snax_bingo_kernel_idma_put_flagged",
                       SnaxBingoKernelIdmaPutFlaggedArgs(src, dst, nb, flag, value=1), deps, c)

    def _landed(self, k, name, flag, buf, deps, c=None):
        """The writer's wait for its flag in HBM (a flagged fetch of it into a 64-byte L1 buffer): once it ends, the data
        it put before the flag is in HBM."""
        return self._t(k, "dm", name, "__snax_bingo_kernel_idma_fetch_flagged",
                       SnaxBingoKernelIdmaFetchFlaggedArgs(flag, buf, 64, flag, 1, 1), deps, c)

    def _x16(self, q):
        k, m, j = self.tq[q]
        return _hbm(self.h_x16[m], j * self.TB)

    def _hrow(self, q, kind):
        """Tile q's h or output in its owner's memory chip's HBM, and its flag there."""
        k, m, j = self.tq[q]
        data, flag = (self.h_h, self.h_hf) if kind == "h" else (self.h_out, self.h_of)
        return _hbm(data[m], j * self.TB), _hbm(flag[m], j * 64)

    def _cld(self, k, name, which, n, buf, deps, c=None):
        """Constant vectors which .. which + n - 1 (input gain, W_O's factors, the post-attention gain, the down
        projection's factors) from the chip's memory chip into `buf`."""
        return self._copy(k, name, _hbm(self.h_c[self.mem[k]], which * self.CB), buf, n * self.CB, deps, c)

    # ---- the steps ----------------------------------------------------------------------------------------------------
    def norm(self, q, after, c=None):
        """The input norm of tile q (16 rows at once, on cluster c, default layer.cluster) into the attention's x region in
        HBM; its wait on the landed flag joins n_conf."""
        A, H = self.attn, self.H
        k = self.tq[q][0]
        nm = f"n{q}"
        b = dict(x=self._l1(k, f"{nm}x", self.TB, c), c=self._l1(k, f"{nm}c", self.CB, c),
                 s=self._l1(k, f"{nm}s", self._scr(), c), e=self._l1(k, f"{nm}e", self.TB, c),
                 q8=self._l1(k, f"{nm}q", self.XTB, c), f=self._l1(k, f"{nm}f", 64, c))
        b["y"] = b["x"] if self.inplace_norm else self._l1(k, f"{nm}y", self.TB, c)
        ld = self._copy(k, f"n_ld_q{q}", self._x16(q), b["x"], self.TB, after, c)
        cl = self._cld(k, f"n_cld_q{q}", 0, 1, b["c"], after, c)
        rn = self._rms(k, f"n_rms_q{q}", b["x"], b["y"], b["s"], [ld], c=c)
        gm = self._mul_rows(k, f"n_gain_q{q}", b["y"], b["c"], [rn, cl], c=c)
        cv = self._cvt(k, f"n_toa_q{q}", "row_major", "A", b["y"], b["e"], [gm], c=c)
        qz = self._quant(k, f"n_q_q{q}", b["e"], b["q8"], A.d.s_x.inv, [cv], c=c)
        flag = _hbm(self.h_nf, q * 64)
        put = self._put_hbm(k, f"n_put_q{q}", b["q8"], _hbm(A.h_xhbm, q * self.XTB), self.XTB, flag, [qz], c)
        conf = self._landed(k, f"n_conf_q{q}", flag, b["f"], [put], c)
        self.n_conf[q // self.NQ].append(conf)

    def glue(self, q, key_a):
        """The glue of tile q: h into HBM, the FFN's X into its region in HBM (its wait joins g_conf). The parts before
        the reshape are placed at the point key_a."""
        A, F, H, N_CL = self.attn, self.ffn, self.H, self.attn.N_CL
        k, m, _ = self.tq[q]
        own, lt = A.y_owner(q)
        nm = f"g{q}"
        b = dict(d=self._l1(k, f"{nm}d", self.TB), a=self._l1(k, f"{nm}a", self.TB), x=self._l1(k, f"{nm}x", self.TB),
                 c=self._l1(k, f"{nm}c", 2 * self.CB), s=self._l1(k, f"{nm}s", self._scr()), e=self._l1(k, f"{nm}e", self.TB),
                 q8=self._l1(k, f"{nm}q", self.XTB), f=self._l1(k, f"{nm}f", 64))
        # every cluster's columns of tile q of y have landed in the chip's L3; placed beside an iteration's W_O, the glue
        # also follows this cluster's last softmax of that iteration, so its buffers live only once the sweep's are free
        after = [A._last_sm[self.chips.index(k) * self.NC + self.SC, key_a[1]]] if key_a and key_a[0] == "attn_wo" else []
        w = self._t(k, "dm", f"g_wait_q{q}", "__snax_bingo_kernel_wait_flags",
                    SnaxBingoKernelWaitFlagsArgs(BingoMemSymbol(f"{A.TAG}_ytflag", offset=lt * N_CL * 64), N_CL, 1),
                    after)
        self.waits.append((w, "attn", q))
        ly = self._copy(k, f"g_ldy_q{q}", A.yt_sym(q), b["d"], self.TB, [w])
        cl = self._cld(k, f"g_cld_q{q}", 1, 2, b["c"], [w])
        rw = self._cvt(k, f"g_rows_q{q}", "D", "row_major", b["d"], b["a"], [ly])
        # x16 once y's tile is converted (its buffer lives from the residual on)
        lx = self._copy(k, f"g_ldx_q{q}", self._x16(q), b["x"], self.TB, [rw])
        dq = self._mul_rows(k, f"g_deq_q{q}", b["a"], b["c"], [rw, cl])
        ad = self._t(k, "simd", f"g_add_q{q}", "__snax_bingo_kernel_simd_stream_elementwise",
                     SnaxBingoKernelSimdAddF16Args(b["a"], b["x"], b["x"], self.TB // 64, 32), [dq, lx])
        rn = self._rms(k, f"g_rms_q{q}", b["x"], b["a"], b["s"], [ad])
        gm = self._mul_rows(k, f"g_gain_q{q}", b["a"], b["c"].view(self.CB), [rn])
        part_a = [w, ly, lx, cl, rw, dq, ad, rn, gm]
        cv = self._cvt(k, f"g_toa_q{q}", "row_major", "A", b["a"], b["e"], [gm])
        qz = self._quant(k, f"g_q_q{q}", b["e"], b["q8"], F.d.s_h.inv, [cv])
        hd, hf = self._hrow(q, "h")
        sh = self._put_hbm(k, f"g_sth_q{q}", b["x"], hd, self.TB, hf, [ad])
        flag = _hbm(self.h_gf, q * 64)
        put = self._put_hbm(k, f"g_put_q{q}", b["q8"], _hbm(F.h_xhbm, q * self.XTB), self.XTB, flag, [qz])
        conf = self._landed(k, f"g_conf_q{q}", flag, b["f"], [put])
        self.g_conf[q // self.NQ].append(conf)
        if key_a is not None:
            self._placed(part_a, key_a)

    def final(self, q, key_a):
        """The final residual of tile q into HBM. The parts before the write are placed at the point key_a."""
        F, H = self.ffn, self.H
        k, m, _ = self.tq[q]
        own, lt = F.y_owner(q)
        nm = f"f{q}"
        b = dict(d=self._l1(k, f"{nm}d", self.TB), a=self._l1(k, f"{nm}a", self.TB), h=self._l1(k, f"{nm}h", self.TB),
                 c=self._l1(k, f"{nm}c", self.CB))
        # every slice of tile q of Y_raw has landed in the chip's L3
        w = self._t(k, "dm", f"f_wait_q{q}", "__snax_bingo_kernel_wait_flags",
                    SnaxBingoKernelWaitFlagsArgs(BingoMemSymbol(f"{F.TAG}_ytflag", offset=lt * F.NDN * 64), F.NDN, 1), [])
        self.waits.append((w, "ffn", q))
        ly = self._copy(k, f"f_ldy_q{q}", F.yt_sym(q), b["d"], self.TB, [w])
        cl = self._cld(k, f"f_cld_q{q}", 3, 1, b["c"], [w])
        rw = self._cvt(k, f"f_rows_q{q}", "D", "row_major", b["d"], b["a"], [ly])
        hd, hf = self._hrow(q, "h")
        # h behind its flag in HBM (the glue's put)
        lh = self._t(k, "dm", f"f_ldh_q{q}", "__snax_bingo_kernel_idma_fetch_flagged",
                     SnaxBingoKernelIdmaFetchFlaggedArgs(hd, b["h"], self.TB, hf, 1, 1), [rw])
        dq = self._mul_rows(k, f"f_deq_q{q}", b["a"], b["c"], [rw, cl])
        ad = self._t(k, "simd", f"f_add_q{q}", "__snax_bingo_kernel_simd_stream_elementwise",
                     SnaxBingoKernelSimdAddF16Args(b["a"], b["h"], b["h"], self.TB // 64, 32), [dq, lh])
        part_a = [w, ly, cl, rw, lh, dq, ad]
        od, of = self._hrow(q, "out")
        self.f_out.append(self._put_hbm(k, f"f_st_q{q}", b["h"], od, self.TB, of, [ad]))
        if key_a is not None:
            self._placed(part_a, key_a)

    # ---- the steps in pieces (params layer.piece_rows below 16) ------------------------------------------------------
    # A tile's piece n is its rows 16 n / NPC ..; it has L1 buffers of its own, live for that piece only, so the static
    # plan fits them into what the block leaves free where the piece runs. A step's pieces on one chip form a chain: each
    # piece's first writer of a buffer follows the previous piece's last reader of its counterpart, so successive
    # pieces may take the same bytes. A piece's loads, its conversion to rows, its SIMD work, h's write, the reshape, the
    # quantise and the INT8 rows' copy into the tile's L3 staging (an A-layout tile, row block by row block); after the
    # last piece one put of the staged tile into HBM. Where each runs in the cores' programs is the compiler's
    # (passes/bingo_dfg_schedule.py).

    def _place(self, k, name, src, stg, r0, deps, c):
        """A piece's INT8 rows (A layout of PR rows: 4 rows' 4 bytes a 4-column block) into an L3 staging tile (A layout of
        16 rows: 64 bytes a block), at the piece's rows of every block."""
        PR, H = self.PR, self.H
        return self._t(k, "dm", name, "__snax_bingo_kernel_idma_2d_copy",
                       SnaxBingoKernelIdma2dCopyArgs(src, BingoMemSymbol(stg, offset=r0 * 4), PR * 4, PR * 4, 64, H // 4),
                       deps, c)

    def _final_shape(self, k):
        """Chip k's final pieces: rows a piece, pieces a tile, down slices between two pieces."""
        rows = self.PR_root if k == self.ROOT else self.PR
        return rows, TR // rows, max(1, self.final_step * rows // self.PR)

    def _d_rows(self, k, name, src_sym, r0, dst, deps, c, rows=None):
        """A piece's rows of a D-layout tile (16 rows, 16-column blocks of 512 bytes) in the chip's L3 into L1 as the D
        layout of that many rows (default PR)."""
        PR, H = rows or self.PR, self.H
        sym = BingoMemSymbol(src_sym.symbol_name, offset=src_sym.offset + r0 * 32, chip_id=src_sym.chip_id)
        return self._t(k, "dm", name, "__snax_bingo_kernel_idma_2d_copy",
                       SnaxBingoKernelIdma2dCopyArgs(sym, dst, PR * 32, 512, PR * 32, H // 16), deps, c)

    def _piece_bufs(self, step, q, n, k, c, spec):
        """Piece n of tile q's L1 buffers, and the chain of `step` on chip k (the previous piece's last readers)."""
        return ({name: self._l1(k, f"{step}{q}p{n}{name}", nb, c) for name, nb in spec.items()},
                self.ps.setdefault((step, k), {}))

    def norm_piece(self, q, n, j):
        """Piece n of tile q's input norm at attention iteration j (cluster layer.norm_cluster): x16 rows, RMSNorm in place,
        the gain, the A layout, quantised (s_x) into the L3 staging; after the last piece the tile into the attention's x
        region in HBM, its wait on the landed flag joining n_conf."""
        A, H, PR = self.attn, self.H, self.PR
        k, c = self.tq[q][0], self.cl_norm
        PB, r0, sfx = PR * H * 2, n * PR, f"q{q}p{n}"
        last = n == self.NPC - 1
        b, prev = self._piece_bufs("n", q, n, k, c, dict(x=PB, s=self._scr(PR), e=PB, q8=PR * H, c=self.CB,
                                                         **(dict(f=64) if last else {})))
        after = [A._last_sm[self.chips.index(k) * self.NC + c, j]]
        ld = self._copy(k, f"n_ld_{sfx}", BingoMemFixedAddr(self._x16(q).address + r0 * H * 2, mem_level="HBM"), b["x"],
                        PB, after + prev.get("toa", []), c)
        cl = self._cld(k, f"n_cld_{sfx}", 0, 1, b["c"], after + prev.get("gain", []), c)
        rn = self._rms(k, f"n_rms_{sfx}", b["x"], b["x"], b["s"], [ld], rows=PR, c=c)
        gm = self._mul_rows(k, f"n_gain_{sfx}", b["x"], b["c"], [rn, cl], rows=PR, c=c)
        cv = self._cvt(k, f"n_toa_{sfx}", "row_major", "A", b["x"], b["e"], [gm] + prev.get("q", []), rows=PR, c=c)
        qz = self._quant(k, f"n_q_{sfx}", b["e"], b["q8"], A.d.s_x.inv, [cv] + prev.get("place", []), nb=PB, c=c)
        pl = self._place(k, f"n_pl_{sfx}", b["q8"], f"{self.TAG}_n8stg", r0, [qz], c)
        if last:
            flag = _hbm(self.h_nf, q * 64)
            put = self._put_hbm(k, f"n_put_q{q}", BingoMemSymbol(f"{self.TAG}_n8stg"), _hbm(A.h_xhbm, q * self.XTB),
                                self.XTB, flag, [pl], c)
            conf = self._landed(k, f"n_conf_q{q}", flag, b["f"], [put], c)
            self.n_conf[q // self.NQ].append(conf)
        prev.update(toa=[cv], q=[qz], place=[pl], gain=[gm])

    def glue_piece(self, q, n, j):
        """Piece n of tile q's glue at attention iteration j (None: at the attention's end; cluster layer.glue_cluster): y's
        rows from the chip's L3 -> row major, dequantised, plus x16: h, into HBM (the last piece's behind a flag); RMSNorm
        in place, the post-attention gain, the A layout, quantised (s_h) into the L3 staging; after the last piece the
        tile into the FFN's X region in HBM, its wait on the landed flag joining g_conf."""
        A, F, H, N_CL, PR = self.attn, self.ffn, self.H, self.attn.N_CL, self.PR
        k, c = self.tq[q][0], self.cl_glue
        own, lt = A.y_owner(q)
        PB, r0, sfx = PR * H * 2, n * PR, f"q{q}p{n}"
        last = n == self.NPC - 1
        b, prev = self._piece_bufs("g", q, n, k, c, dict(d=PB, a=PB, x=PB, s=self._scr(PR), e=PB, q8=PR * H, c=2 * self.CB,
                                                         **(dict(f=64) if last else {})))
        if n == 0:
            # every cluster's columns of tile q of y have landed in the chip's L3 (and, inside the attention, this
            # cluster's sweep of iteration j is done, so the buffers live once the sweep's are free)
            after = [A._last_sm[self.chips.index(k) * self.NC + c, j]] if j is not None else []
            prev["w"] = self._t(k, "dm", f"g_wait_q{q}", "__snax_bingo_kernel_wait_flags",
                                SnaxBingoKernelWaitFlagsArgs(BingoMemSymbol(f"{A.TAG}_ytflag", offset=lt * N_CL * 64),
                                                             N_CL, 1),
                                after, c)
            self.waits.append((prev["w"], "attn", q))
        w = prev["w"]
        cl = self._cld(k, f"g_cld_{sfx}", 1, 2, b["c"], [w] + prev.get("gain", []), c)
        ly = self._d_rows(k, f"g_ldy_{sfx}", A.yt_sym(q), r0, b["d"], [w] + prev.get("rows", []), c)
        rw = self._cvt(k, f"g_rows_{sfx}", "D", "row_major", b["d"], b["a"], [ly] + prev.get("add", []), rows=PR, c=c)
        dq = self._mul_rows(k, f"g_deq_{sfx}", b["a"], b["c"], [rw, cl], rows=PR, c=c)
        # x after the conversion: d is free by then, so x may take its bytes
        lx = self._copy(k, f"g_ldx_{sfx}", BingoMemFixedAddr(self._x16(q).address + r0 * H * 2, mem_level="HBM"), b["x"],
                        PB, [rw] + prev.get("toa", []), c)
        ad = self._t(k, "simd", f"g_add_{sfx}", "__snax_bingo_kernel_simd_stream_elementwise",
                     SnaxBingoKernelSimdAddF16Args(b["a"], b["x"], b["x"], PB // 64, 32), [dq, lx], c)
        hd, hf = self._hrow(q, "h")
        hrow = BingoMemFixedAddr(hd.address + r0 * H * 2, mem_level="HBM")
        sh = (self._put_hbm(k, f"g_sth_{sfx}", b["x"], hrow, PB, hf, [ad], c) if last else
              self._copy(k, f"g_sth_{sfx}", b["x"], hrow, PB, [ad], c))
        rn = self._rms(k, f"g_rms_{sfx}", b["x"], b["x"], b["s"], [sh], rows=PR, c=c)
        gm = self._mul_rows(k, f"g_gain_{sfx}", b["x"], b["c"].view(self.CB), [rn], rows=PR, c=c)
        cv = self._cvt(k, f"g_toa_{sfx}", "row_major", "A", b["x"], b["e"], [gm] + prev.get("q", []), rows=PR, c=c)
        qz = self._quant(k, f"g_q_{sfx}", b["e"], b["q8"], F.d.s_h.inv, [cv] + prev.get("place", []), nb=PB, c=c)
        pl = self._place(k, f"g_pl_{sfx}", b["q8"], f"{self.TAG}_g8stg", r0, [qz], c)
        if last:
            flag = _hbm(self.h_gf, q * 64)
            put = self._put_hbm(k, f"g_put_q{q}", BingoMemSymbol(f"{self.TAG}_g8stg"), _hbm(F.h_xhbm, q * self.XTB),
                                self.XTB, flag, [pl], c)
            conf = self._landed(k, f"g_conf_q{q}", flag, b["f"], [put], c)
            self.g_conf[q // self.NQ].append(conf)
        prev.update(rows=[rw], add=[ad], toa=[cv], q=[qz], place=[pl], gain=[gm])

    def final_piece(self, q, n, at):
        """Piece n of tile q's final residual at FFN pass / down slice `at` = (i, s) (None: at the FFN's end; cluster
        layer.final_cluster): Y_raw's rows from the chip's L3 -> row major, dequantised, plus h (from HBM behind its
        flag): the output, into HBM (the last piece's behind a flag)."""
        F, H = self.ffn, self.H
        k, c = self.tq[q][0], self.cl_final
        PR, npc, _ = self._final_shape(k)
        last, tight = n == npc - 1, PR != self.PR
        own, lt = F.y_owner(q)
        PB, r0, sfx = PR * H * 2, n * PR, f"q{q}p{n}"
        b, prev = self._piece_bufs("f", q, n, k, c, dict(d=PB, a=PB, h=PB, c=self.CB))
        if n == 0:
            # every slice of tile q of Y_raw has landed in the chip's L3
            prev["w"] = self._t(k, "dm", f"f_wait_q{q}", "__snax_bingo_kernel_wait_flags",
                                SnaxBingoKernelWaitFlagsArgs(BingoMemSymbol(f"{F.TAG}_ytflag", offset=lt * F.NDN * 64),
                                                             F.NDN, 1), [], c)
            self.waits.append((prev["w"], "ffn", q))
        w = prev["w"]
        ly = self._d_rows(k, f"f_ldy_{sfx}", F.yt_sym(q), r0, b["d"], [w] + prev.get("rows", []), c, rows=PR)
        rw = self._cvt(k, f"f_rows_{sfx}", "D", "row_major", b["d"], b["a"], [ly] + prev.get("add", []), rows=PR, c=c)
        # tight: the factors after the conversion (d is free then) and h after the multiply (c is free then), so at most
        # two of the piece's buffers are live
        cl = self._cld(k, f"f_cld_{sfx}", 3, 1, b["c"], [w] + prev.get("deq", []) + ([rw] if tight else []), c)
        dq = self._mul_rows(k, f"f_deq_{sfx}", b["a"], b["c"], [rw, cl], rows=PR, c=c)
        hd, hf = self._hrow(q, "h")
        lh = self._t(k, "dm", f"f_ldh_{sfx}", "__snax_bingo_kernel_idma_fetch_flagged",
                     SnaxBingoKernelIdmaFetchFlaggedArgs(BingoMemFixedAddr(hd.address + r0 * H * 2, mem_level="HBM"),
                                                         b["h"], PB, hf, 1, 1),
                     [dq if tight else rw] + prev.get("st", []), c)
        ad = self._t(k, "simd", f"f_add_{sfx}", "__snax_bingo_kernel_simd_stream_elementwise",
                     SnaxBingoKernelSimdAddF16Args(b["a"], b["h"], b["h"], PB // 64, 32), [dq, lh], c)
        od, of = self._hrow(q, "out")
        orow = BingoMemFixedAddr(od.address + r0 * H * 2, mem_level="HBM")
        st = (self._put_hbm(k, f"f_st_{sfx}", b["h"], orow, PB, of, [ad], c) if last else
              self._copy(k, f"f_st_{sfx}", b["h"], orow, PB, [ad], c))
        self.f_out.append(st)
        prev.update(rows=[rw], add=[ad], st=[st], deq=[dq])

    def _plan_pieces(self):
        """params layer.piece_rows below 16: every piece of every step, made at the point after its last part."""
        A, F = self.attn, self.ffn
        for p in range(A.NP):
            for u in range(self.NQ):
                q = p * self.NQ + u
                for n in range(self.NPC if p >= self.norm_lead else 0):
                    j = p - self.norm_lead + n
                    self._schedule(self._point("attn", "top", j + 1), lambda q=q, n=n, j=j: self.norm_piece(q, n, j))
                for n in range(self.NPC):
                    j = p + self.glue_lag + n
                    jj = j if j < A.NP else None
                    self._schedule(self._point("attn", "top", j + 1), lambda q=q, n=n, jj=jj: self.glue_piece(q, n, jj))
        for p in range(F.NI):
            i = p + self.final_lag
            for u in range(self.NQ):
                q = p * self.NQ + u
                _, npc, step = self._final_shape(self.tq[q][0])
                for n in range(npc):
                    at = (i, self.final_slice + step * n) if i < F.NI else None
                    self._schedule(self._point("ffn", "top", i + 1), lambda q=q, n=n, at=at: self.final_piece(q, n, at))

    def _plan_steps(self):
        """Every step of every tile, made at the point of its last part (pending until the block reaches it)."""
        A, F = self.attn, self.ffn
        if self.NPC > 1:
            return self._plan_pieces()
        for p in range(A.NP):
            if p >= self.norm_lead:
                key = self._point("attn", "top", p - self.norm_lead)
                for u in range(self.NQ):
                    self._schedule(key, lambda q=p * self.NQ + u: self.norm(q, [self.go[self.ROOT]]))
            ka = self._point("attn", "wo", p + self.glue_lag)
            kb = self._point("attn", "top", p + self.glue_lag + 1)
            for u in range(self.NQ):
                self._schedule(kb, lambda q=p * self.NQ + u, ka=ka, kb=kb: self.glue(q, ka if ka != kb else None))
        for p in range(F.NI):
            ka = self._point("ffn", "dn", p + self.final_lag, self.final_slice)
            kb = self._point("ffn", "top", p + self.final_lag + 1)
            for u in range(self.NQ):
                self._schedule(kb, lambda q=p * self.NQ + u, ka=ka, kb=kb: self.final(q, ka if ka != kb else None))

    # ---- the windows -------------------------------------------------------------------------------------------------
    def open_window(self, extra):
        """The start stamp (after every chip's setup), the attention's prefetchers behind it, and the head: the norms
        of the first norm_lead passes, before the attention's tasks."""
        A = self.attn
        ends = [t for k in self.chips for t in (extra or {}).get(k, [])]
        start = self.stamp("start", ends)
        self.go = {k: start for k in self.chips}
        for k in self.chips:
            self.edge(start, A.prefetch[k])
        self.marks[("head",)] = self.dfg.id + 1
        for p in range(min(self.norm_lead, A.NP)):
            for u in range(self.NQ):
                q = p * self.NQ + u
                # with pieces the head's tiles alternate clusters every 16 tiles (more than a tile a chip there)
                c = (self.cl_norm + q // len(self.chips)) % self.NC if self.NPC > 1 else None
                self.norm(q, [start], c)
        self._plan_steps()
        return dict(self.go)

    def attn_x_puts(self, go):
        """The root's host: the attention's prefetcher in parts, each pass's x multicast after its part. The parts'
        chunk ranges are set once the rings are known (_set_attn_parts)."""
        A, R = self.attn, self.ROOT
        pf = A.prefetch[R]
        self.a_pf_args = pf.kernel_args
        self.a_parts, self.x_bc, prev = [], [], [go[R]]
        for i in range(A.NP):
            part = self.cx_chip(R).host_node(f"la_prefetch_part{i}_k{R:02x}", HostBingoKernelWeightPrefetchRangeArgs.KERNEL_NAME,
                                             _range_args(self.a_pf_args, 0, 1), after=prev)
            self.a_parts.append(part)
            self.x_bc.append(A.stage_x_put(i, [part]))
            prev = [self.x_bc[-1]]
        pf.kernel_args = _range_args(self.a_pf_args, 0, 0)
        pf.kernel_name = pf.kernel_args.KERNEL_NAME
        self.edge(self.x_bc[-1], pf)
        A._x_put_last = self.x_bc[-1]
        return list(self.x_bc)

    def _set_attn_parts(self):
        """Each part's chunk range on the root's rings: part 0 pass 0's projection chunks [0, WO0), part i from the end of
        part i - 1 to pass i - 1's first W_O chunk, the last part (the prefetcher) from there to the ring's end; and the
        edges that order each x multicast after the o8 multicast two passes back."""
        A, R = self.attn, self.ROOT
        first_wo = {}
        for t in self.dfg.node_list:
            m = re.match(r"Ld_w(\d+)_i(\d+)_k", t.node_name)
            if m and t.assigned_chiplet_id == R and getattr(t, "exec_chunk", None) is not None:
                ring, idx = t.exec_chunk
                j, i = int(m.group(1)), int(m.group(2))
                if j >= A.WO0:
                    first_wo[(ring, i)] = min(first_wo.get((ring, i), idx), idx)
        bounds = [A.WO0]
        for i in range(1, A.NP):
            b = {first_wo[(r, i - 1)] for r in range(self.NC)}
            if len(b) != 1:
                raise ValueError(f"pass {i - 1}'s first W_O chunk sits at ring indices {sorted(b)} on the root's rings")
            bounds.append(b.pop())
        if bounds != sorted(bounds) or len(set(bounds)) != len(bounds):
            raise ValueError(f"the root prefetcher's part bounds {bounds} are not increasing")
        lo = 0
        for i, part in enumerate(self.a_parts):
            part.kernel_args = _range_args(self.a_pf_args, lo, bounds[i])
            lo = bounds[i]
        A.prefetch[R].kernel_args = _range_args(self.a_pf_args, lo, 0)
        self.a_bounds = bounds
        # each x multicast after the o8 multicast two passes back
        mc = {int(re.match(r"o_bput_i(\d+)_k", t.node_name).group(1)): t for t in self.dfg.node_list
              if t.node_name.startswith("o_bput_i") and t.assigned_chiplet_id == R}
        for i in range(2, A.NP):
            self.edge(mc[i - 2], self.x_bc[i])
        self.o_bput_last = mc[A.NP - 1]
        print(f"[{self.TAG}] the root's attention prefetcher in {len(self.a_parts) + 1} parts, x multicast i after the part "
              f"ending at ring index {bounds} (n_slots {A.n_slots})")

    def ffn_window(self, extra):
        """The FFN on each chip: its setup loads after the chip's attention end (its wait on its tiles of y); each chip's
        FFN prefetcher after its attention prefetcher (the root's after its X multicasts, ffn_x_puts)."""
        A, F = self.attn, self.ffn
        go = {}
        for k in self.chips:
            ends = [t for t in self.attn_ends if t.assigned_chiplet_id == k]
            if len(ends) != 1:
                raise ValueError(f"chip {k:02x}: {len(ends)} attention end tasks")
            go[k] = ends[0]
            for t in (extra or {}).get(k, []):
                self.edge(go[k], t)
            if k != self.ROOT:
                self.edge(A.prefetch[k], F.prefetch[k])
        return go

    def ffn_x_puts(self, go):
        """The root's host after the attention's last prefetcher part: the FFN's X multicasts, each behind its pass's glue
        and the attention's last o8 multicast; the FFN's prefetcher split around the last one (ffn_pf_head). With an X
        ring of fewer slots than passes (layer.x_slots) the first x_slots multicasts come here and each later one after
        a part of the prefetcher (_ffn_ring places it between the gathers)."""
        A, F, R, T = self.attn, self.ffn, self.ROOT, self.ffn.TAG
        m = F.mem[R]
        one = F.h_seq[m].address + 64
        prev = [A.prefetch[R], self.o_bput_last]
        out = []
        self.f_head, self.f_parts = None, []

        def bcast(i, after):
            args = HostBingoKernelHbmBcastArgs(F.h_xhbm.address + i * F.XB, F._xland(i * F.XB),
                                               BingoMemSymbol(f"{T}_xtflag", offset=64 * i * F.TPP), one, 1, F.XT, F.TPP,
                                               m, F.engine[R])
            return self.cx_chip(R).host_node(f"pf_x_hbm_bcast_i{i}_k{R:02x}", "__host_bingo_kernel_hbm_bcast", args,
                                             after=after + self.g_conf[i] + self._slot_free(i))

        if self.x_slots and F.NI > self.x_slots:
            S = self.x_slots
            for i in range(S):
                out.append(bcast(i, prev))
                prev = out[-1:]
            b = self.f_pf_args = F.prefetch[R].kernel_args
            for i in range(S, F.NI):
                part = self.cx_chip(R).host_node(f"pf_prefetch_part{i - S}_k{R:02x}",
                                                 HostBingoKernelWeightPrefetchRangeArgs.KERNEL_NAME, _range_args(b, 0, 1),
                                                 after=prev)
                self.f_parts.append(part)
                out.append(bcast(i, [part]))
                prev = out[-1:]
            F.prefetch[R].kernel_args = _range_args(b, 0, 0)
            F.prefetch[R].kernel_name = F.prefetch[R].kernel_args.KERNEL_NAME
            self.edge(out[-1], F.prefetch[R])
            self.f_bc = out
            return out
        for i in range(F.NI):
            if i == F.NI - 1 and F.NI > 1 and self.ffn_pf_head:
                b = F.prefetch[R].kernel_args
                self.f_pf_args = b
                self.f_head = self.cx_chip(R).host_node(f"pf_prefetch_head_k{R:02x}",
                                                        HostBingoKernelWeightPrefetchRangeArgs.KERNEL_NAME,
                                                        _range_args(b, 0, self.ffn_pf_head), after=prev)
                F.prefetch[R].kernel_args = _range_args(b, self.ffn_pf_head, 0)
                F.prefetch[R].kernel_name = F.prefetch[R].kernel_args.KERNEL_NAME
                prev = [self.f_head]
            out.append(bcast(i, prev))
            prev = out[-1:]
        self.edge(out[-1], F.prefetch[R])
        self.f_bc = out
        return out

    def _slot_free(self, i):
        """params layer.alias_l3: the FFN's X_i lands in the slot the attention's last pass there used; that pass's glue
        (each tile's wait follows every cluster's y of it, so every cluster's o8 gather from the slot) comes first."""
        if not self.alias_l3:
            return []
        S, NP = self.x_slots, self.attn.NP
        last = [p for p in range(NP) if p % S == i % S]
        return list(self.g_conf[last[-1]]) if last and i < S else []

    def _ffn_ring(self):
        """The FFN's X ring (more passes than slots): X_i (i >= x_slots) multicast after pass i - 2's last gather and
        before pass i - 1's first (every multicast ordered against every gather; its slot's previous X, pass i - x_slots,
        was fetched before pass i - 2's down GEMMs), the first gather after the last multicast made up front, and the
        root's prefetcher parts: part i - x_slots ends at pass i - 1's chunk layer.ffn_part_chunk."""
        F, R, S = self.ffn, self.ROOT, self.x_slots
        chains, idx = {}, {}
        for t in self.dfg.nodes:
            if t.assigned_chiplet_id != R:
                continue
            mm = re.match(r"y_chain_(?:i(\d+))?j\d+_k", t.node_name)
            if mm:
                chains.setdefault(int(mm.group(1) or 0), []).append(t)
            mm = re.match(r"Ld_c(\d+)_i(\d+)_k", t.node_name)
            if mm and getattr(t, "exec_chunk", None) is not None:
                idx[(t.exec_chunk[0], int(mm.group(2)), int(mm.group(1)))] = t.exec_chunk[1]
        first = {i: min(ts, key=lambda t: t.node_id) for i, ts in chains.items()}
        last = {i: max(ts, key=lambda t: t.node_id) for i, ts in chains.items()}
        xb = self.f_bc
        if self.dfg.has_edge(xb[-1], first[0]):
            self.dfg.remove_edge(xb[-1], first[0])
        self.edge(xb[S - 1], first[0])
        bounds = []
        for i in range(S, F.NI):
            self.edge(last[i - 2], xb[i])
            self.edge(xb[i], first[i - 1])
            b = {idx[(r, i - 1, self.ffn_part_chunk)] for r in range(self.NC)}
            if len(b) != 1:
                raise ValueError(f"FFN pass {i - 1}'s chunk {self.ffn_part_chunk} at ring indices {sorted(b)} on the root")
            bounds.append(b.pop())
        lo = 0
        for part, hi in zip(self.f_parts, bounds):
            part.kernel_args = _range_args(self.f_pf_args, lo, hi)
            lo = hi
        F.prefetch[R].kernel_args = _range_args(self.f_pf_args, lo, 0)
        self.f_bounds = list(zip(xb[S:], bounds))
        print(f"[{self.TAG}] FFN X ring of {S}: {len(bounds)} multicasts between the gathers, the root's FFN prefetcher in "
              f"{len(bounds) + 1} parts ending at ring index {bounds}")

    def _ring_alias_edges(self):
        """params layer.alias_l3: the blocks' weight rings are one region, so each chip's FFN pushes follow both its
        clusters' last attention ring loads (the slots read)."""
        A, F, R = self.attn, self.ffn, self.ROOT
        last = {}
        for t in self.dfg.nodes:
            if t.node_name.startswith("Ld_w") and getattr(t, "exec_chunk", None) is not None:
                key = (t.assigned_chiplet_id, t.assigned_cluster_id)
                if key not in last or t.exec_chunk[1] > last[key].exec_chunk[1]:
                    last[key] = t
        for k in self.chips:
            first = F.prefetch[k] if k != R else (self.f_parts[0] if self.f_parts else self.f_head or F.prefetch[R])
            for c in range(self.NC):
                self.edge(last[(k, c)], first)
        self.ring_first = {k: (F.prefetch[k] if k != R else (self.f_parts[0] if self.f_parts else self.f_head or
                                                              F.prefetch[R])) for k in self.chips}
        self.ring_last = last

    def close_window(self, after):
        """The FFN's end: the finals still pending, then the end stamp (behind the FFN's ends and every output write) and
        the root's report."""
        self.at(("ffn_end",))
        if self.pending:
            raise ValueError(f"steps never made: {[s[0] for s in self.pending][:4]}")
        self.end_window(list(after) + self.f_out)

    # ---- the layer ----------------------------------------------------------------------------------------------------
    def build(self):
        A, F = self.attn, self.ffn
        A.build()
        self.at(("attn_end",))
        self._set_attn_parts()
        for i in range(A.NP):
            if len(self.n_conf[i]) != self.NQ:
                raise ValueError(f"pass {i}: {len(self.n_conf[i])} norms made")
            for t in self.n_conf[i]:
                self.edge(t, self.x_bc[i])
        F.build()
        if self.f_parts:
            self._ffn_ring()
        if self.alias_l3:
            self._ring_alias_edges()
        self._wait_edges()
        self.xch = dict(F.xch)        # the FFN's exchange buffers: the gathers name them by the root's address
        self.run_checks(self.layer_checks())

    def _wait_edges(self):
        """Each step's wait on a tile's flags follows (order edges) every put that raises one of them."""
        A, F, R = self.attn, self.ffn, self.ROOT
        ya = {}                                  # tile -> the attention's flag puts of its y
        for k, puts in A._yt_puts.items():
            for t in puts:
                mm = re.match(r"y_sf_i(\d+)m(\d+)_k", t.node_name)
                ya.setdefault(int(mm.group(1)) * self.NQ + int(mm.group(2)), []).append(t)
        yf = {}                                  # tile -> the FFN's puts of its Y_raw slices (the root's)
        for t in self.dfg.node_list:
            mm = re.match(r"y_sc_(?:i(\d+))?j\d+m(\d+)_k", t.node_name)
            if mm and t.assigned_chiplet_id == R and t.kernel_name == "__snax_bingo_kernel_idma_put_flagged":
                yf.setdefault(int(mm.group(1) or 0) * self.NQ + int(mm.group(2)), []).append(t)
        for w, blk, q in self.waits:
            puts, n = (ya.get(q, []), A.N_CL) if blk == "attn" else (yf.get(q, []), F.NDN)
            if len(puts) != n:
                raise ValueError(f"tile {q}: {len(puts)} {blk} puts raise the {n} flags its step waits for")
            for t in puts:
                if t.assigned_chiplet_id == w.assigned_chiplet_id:
                    self.edge(t, w)
                else:
                    self.dfg.bingo_add_order_edge(t, w)

    # ---- checks -------------------------------------------------------------------------------------------------------
    def _gather(self, name, items):
        """items [(chip, source, golden bytes, HBM flag or None)]: after the end stamp each source is copied into its
        slice of one L3 buffer on the root (from HBM through a 64 KiB L1 buffer of its chip, behind the data's flag);
        returns the check."""
        R, T = self.ROOT, self.TAG
        n = sum(ref.size for _, _, ref, _ in items)
        sym = f"{T}_chk_{name}"
        self.st.put_zeros(sym, "uint8_t", n)
        off = 0
        for j, (k, src, gold, flag) in enumerate(items):
            dst = BingoMemSymbol(sym, offset=off, chip_id=None if k == R else R)
            if flag is None:
                cp = self._copy(k, f"chk_{name}{j}", src, dst, gold.size, [self.end], c=0)
            else:
                buf = self._l1(k, f"chk_{name}{j}", gold.size, c=0)
                ld = self._t(k, "dm", f"chk_{name}{j}_ld", "__snax_bingo_kernel_idma_fetch_flagged",
                             SnaxBingoKernelIdmaFetchFlaggedArgs(src, buf, gold.size, flag, 1, 1), [self.end], c=0)
                cp = self._copy(k, f"chk_{name}{j}", buf, dst, gold.size, [ld], c=0)
            self.check_copies.append(cp)
            off += gold.size
        gold = np.concatenate([ref.reshape(-1) for _, _, ref, _ in items])
        return (name, BingoMemSymbol(sym, chip_id=R), gold, n)

    def layer_checks(self):
        A, F, lay, mesh = self.attn, self.ffn, self.attn.lay, self.attn.mesh
        u8 = lambda a: np.ascontiguousarray(a).view(np.uint8).reshape(-1)    # noqa: E731
        n, NT, NTF, TB, XTB = 2048, self.NT, self.NTF, self.TB, self.XTB
        tile = lambda a, q: a[q * TR:(q + 1) * TR]                            # noqa: E731
        last, lastf = NT - 1, NTF - 1
        mid = NT // 2 + 1 if NT > 2 else 0
        checks = []
        # x8 (the input norm's): the second tile's head, the last tile's tail; in the attention's x region
        x8 = lambda q: u8(lay.to_a(tile(A.d.hq, q), mesh).astype(np.int8))   # noqa: E731
        items = [(self.tq[q][0], _hbm(A.h_xhbm, q * XTB + off), x8(q)[off:off + n], _hbm(self.h_nf, q * 64))
                 for q, off in ((1 % NT, 0), (last, XTB - n))]
        checks.append(self._gather("x8", items))
        # y (the attention's, raw, D layout) in its owners' L3: the first tile's head, a middle one's middle, the last's tail
        yd = lambda q: u8(lay.to_d(tile(A.gold_y, q), mesh).astype(np.float16))   # noqa: E731
        items = [(self.tq[q][0], A.yt_sym(q, off), yd(q)[off:off + n], None)
                 for q, off in ((0, 0), (mid, TB // 2 - n // 2), (last, TB - n))]
        if not self.alias_l3:
            # (params layer.alias_l3: the FFN's Y_raw tiles land over y's; h below checks y through the glue)
            checks.append(self._gather("y", items))
        # h (FP16, row major) in HBM: two whole rows, the first pass's third tile, the last pass's
        rb = self.H * 2
        items = []
        for q, r in ((2 % NT, 9), (min(last, 6), 3)):
            hd, hf = self._hrow(q, "h")
            items.append((self.tq[q][0], _hbm(self.h_h[self.tq[q][1]], self.tq[q][2] * TB + r * rb),
                          u8(self.h16[q * TR + r].astype(np.float16)), hf))
        checks.append(self._gather("h", items))
        # X (the glue's, the FFN's A layout) in the FFN's X region: a first-pass tile's head, the last FFN tile's tail
        xf = lambda q: u8(lay.to_a(tile(F.d.hq, q), mesh).astype(np.int8))   # noqa: E731
        items = [(self.tq[q][0], _hbm(F.h_xhbm, q * XTB + off), xf(q)[off:off + n], _hbm(self.h_gf, q * 64))
                 for q, off in ((3 % NTF, 0), (lastf, XTB - n))]
        checks.append(self._gather("X", items))
        # Y_raw (the FFN's, D layout) in its owners' L3: the first tile's head, the last FFN tile's tail
        yr = lambda q: u8(lay.to_d(tile(F.gold["y_raw"], q), mesh).astype(np.float16))   # noqa: E731
        items = [(self.tq[q][0], F.yt_sym(q, off), yr(q)[off:off + n], None) for q, off in ((0, 0), (lastf, TB - n))]
        checks.append(self._gather("Y", items))
        # the output (FP16, row major) in HBM: whole rows of tiles on four chips over both FFN passes
        items = []
        rows = {(0, 0), (min(3, lastf), 15), (min(4, lastf), 7), (lastf, 15)}
        if self.PR_root != self.PR:
            rows.add((0, 15))                                  # the root's pieces of fewer rows: its last, flagged one too
        for q, r in sorted(rows):
            od, of = self._hrow(q, "out")
            items.append((self.tq[q][0], _hbm(self.h_out[self.tq[q][1]], self.tq[q][2] * TB + r * rb),
                          u8(self.out16[q * TR + r].astype(np.float16)), of))
        checks.append(self._gather("out", items))
        tot = sum(c[3] for c in checks)
        print(f"[{self.TAG}] checks: {len(checks)} on the root's L3, {tot:,} B compared (" +
              ", ".join(f"{c[0]} {c[3]:,} B" for c in checks) + ")")
        return checks

    # ---- proofs -------------------------------------------------------------------------------------------------------
    def _check_collectives(self):
        """Every multicast (an HBM multicast, a device bcast put) and every ChainGather ordered against every other by a
        path of dependency and order edges: sorted by a topological order, each one reaches the next."""
        import networkx as nx
        G = self.dfg.bingo_order_view()
        items = [t for t in self.dfg.node_list
                 if (t.kernel_name == "__snax_bingo_kernel_idma_put_flagged" and t.kernel_args.bcast & 1)
                 or t.kernel_name == "__host_bingo_kernel_hbm_bcast"
                 or t.kernel_name == "__snax_bingo_kernel_xdma_chain_gather"]
        pos = {t: i for i, t in enumerate(nx.topological_sort(G))}
        items.sort(key=lambda t: pos[t])
        # a split ChainGather runs from its start to its wait: the one before it must end before that start
        bad = [(a.node_name, b.node_name) for a, b in zip(items, items[1:]) if not nx.has_path(G, a, b)]
        if bad:
            raise ValueError(f"{len(bad)} collective pairs not ordered by a path, e.g. {bad[:4]}")
        kinds = {}
        for t in items:
            kinds[t.kernel_name.split("kernel_")[-1]] = kinds.get(t.kernel_name.split("kernel_")[-1], 0) + 1
        print(f"[{self.TAG}] collectives: {len(items)} ({kinds}), each ordered before the next by a path")

    def _check_prefetch_parts(self):
        """Each prefetcher part that a multicast follows on the root's host ends once its last chunk is pushed, which
        needs the chunk n_slots before it read (its ring load done). That ring load must not follow (by any path) the
        multicast after the part, or the host would wait for it forever: checked for the attention's parts (x_i) and
        the FFN's head part (the last X)."""
        import networkx as nx
        G = self.dfg.bingo_order_view()
        A, F, R = self.attn, self.ffn, self.ROOT
        loads = {}
        for t in self.dfg.node_list:
            if t.assigned_chiplet_id == R and getattr(t, "exec_chunk", None) is not None:
                blk = "attn" if t.node_name.startswith("Ld_w") else "ffn" if t.node_name.startswith("Ld_c") else None
                if blk:
                    loads[(blk,) + tuple(t.exec_chunk)] = t
        checks = [("attn", A.n_slots, hi, self.x_bc[i]) for i, hi in enumerate(self.a_bounds) if i]
        # the attention's last part, before the FFN's first X multicast on the same host
        first = [t for t in self.dfg.node_list if t.node_name.startswith("pf_x_hbm_bcast_i0_")][0]
        checks.append(("attn", A.n_slots, len(A.rings[R].entries[0]), first))
        if self.f_head is not None:
            last = [t for t in self.dfg.node_list if t.node_name.startswith(f"pf_x_hbm_bcast_i{F.NI - 1}_")][0]
            checks.append(("ffn", F.n_slots, self.ffn_pf_head, last))
        for mc, hi in getattr(self, "f_bounds", []):
            checks.append(("ffn", F.n_slots, hi, mc))
        bad = []
        for blk, ns, hi, mc in checks:
            desc = nx.descendants(G, mc)
            need = [loads[(blk, r, k)] for r in range(self.NC) for k in range(max(0, hi - ns))
                    if (blk, r, k) in loads]
            bad += [(blk, hi, t.node_name) for t in need if t in desc]
        if bad:
            raise ValueError(f"prefetcher parts that wait for a ring load behind their own multicast: {bad[:4]}")
        print(f"[{self.TAG}] prefetcher parts: {len(checks)} parts before a multicast, none waits for a load behind it")

    def _check_rings(self):
        """params layer.x_slots / alias_l3: every buffer reuse the rings and aliases make is ordered by a path. A slot's next
        multicast follows every task of its previous pass that reads or writes it (the attention: its x fetches, the o8
        puts into the root's slot, the root's o8 multicast, every o8 gather; the FFN: its X fetches); with alias_l3 the
        FFN's first multicasts follow the attention's last pass in each slot, the FFN's Y_raw puts into a tile follow the
        glue's loads of that tile's y, and each chip's first FFN push follows both clusters' last attention ring loads."""
        import networkx as nx
        G = self.dfg.bingo_order_view()
        A, F, S, R = self.attn, self.ffn, self.x_slots, self.ROOT
        by = {}
        for t in self.dfg.nodes:
            by.setdefault(re.sub(r"_k[0-9a-f]{2}_cl\d$", "", t.node_name), []).append(t)

        def users_attn(i):
            pat = re.compile(rf"^((x_wait|x_xw|x_xcp)_i{i}t\d+$|o_put_i{i}$|o_bput_i{i}$|o_get_i{i}r)")
            return [t for n, ts in by.items() if pat.match(n) for t in ts]

        def users_ffn(i):
            pat = re.compile(rf"^x_wait_i{i}(b\d+)?$")
            return [t for n, ts in by.items() if pat.match(n) for t in ts]
        bad, n = [], 0
        for i in range(S, A.NP):
            anc = nx.ancestors(G, self.x_bc[i])
            us = users_attn(i - S)
            n += len(us)
            bad += [(t.node_name, self.x_bc[i].node_name) for t in us if t not in anc]
        for i in range(S, F.NI):
            anc = nx.ancestors(G, self.f_bc[i])
            us = users_ffn(i - S)
            n += len(us)
            bad += [(t.node_name, self.f_bc[i].node_name) for t in us if t not in anc]
        if self.alias_l3:
            for i in range(min(S, F.NI)):
                last = [p for p in range(A.NP) if p % S == i]
                if not last:
                    continue
                anc = nx.ancestors(G, self.f_bc[i])
                us = users_attn(last[-1])
                n += len(us)
                bad += [(t.node_name, self.f_bc[i].node_name) for t in us if t not in anc]
            for q in range(self.NTF):
                puts = sorted((t for t in self.dfg.nodes if t.assigned_chiplet_id == R
                               and re.match(rf"y_sc_(?:i\d+)?j\d+m{q % self.NQ}_k", t.node_name)
                               and int((re.match(r"y_sc_(?:i(\d+))?", t.node_name).group(1) or 0)) == q // self.NQ),
                              key=lambda t: t.node_id)[:1]
                loads = [t for nm, ts in by.items() if re.match(rf"^g_ldy_q{q}(p\d+)?$", nm) for t in ts]
                for p in puts:
                    anc = nx.ancestors(G, p)
                    n += len(loads)
                    bad += [(t.node_name, p.node_name) for t in loads if t not in anc]
            for k, first in self.ring_first.items():
                anc = nx.ancestors(G, first)
                for c in range(self.NC):
                    n += 1
                    if self.ring_last[(k, c)] not in anc:
                        bad.append((self.ring_last[(k, c)].node_name, first.node_name))
        if bad:
            raise ValueError(f"{len(bad)} ring or alias reuses not ordered, e.g. {bad[:4]}")
        print(f"[{self.TAG}] rings and aliases: {n} earlier users each before the reuse of their slot or region")

    def _check_fold_slots(self):
        """ffn.y_scatter: the puts of a root fold slot (the root's Y_raw scatter after a gather) all precede the next
        gather into the same slot (the block's hazard check names the stores the scatter replaces)."""
        import networkx as nx
        F, R = self.ffn, self.ROOT
        G = self.dfg.bingo_order_view()
        h, GS = F.NDN // F.NC, F.chain_group
        gathers = {}
        for t in self.dfg.node_list:
            mm = re.match(r"y_chain_(?:i(\d+))?j(\d+)_k", t.node_name)
            if mm and t.assigned_chiplet_id == R:
                i, j = int(mm.group(1) or 0), int(mm.group(2))
                c = t.assigned_cluster_id
                gathers.setdefault(c, []).append(((i, j - c * h), t))
        n = 0
        for c, gs in gathers.items():
            gs.sort()
            for a_n, ((i, jl), ga) in enumerate(gs):
                s = (jl // GS) % 2
                nxt = next((gb for (i2, jl2), gb in gs[a_n + 1:] if (jl2 // GS) % 2 == s), None)
                if nxt is None:
                    continue
                readers = [v for v in self.dfg.successors(ga) if v.kernel_name == "__snax_bingo_kernel_idma_put_flagged"
                           and v.assigned_chiplet_id == R and v.node_name.startswith("y_sc_")]
                if not readers:
                    raise ValueError(f"{ga.node_name}: no Y_raw puts read its fold slot")
                n += len(readers)
                late = [v.node_name for v in readers if not nx.has_path(G, v, nxt)]
                if late:
                    raise ValueError(f"fold slot of {ga.node_name}: {late[:3]} not before {nxt.node_name}")
        print(f"[{self.TAG}] fold slots: {n} Y_raw puts each before the next gather into their slot")

    # ---- emission -----------------------------------------------------------------------------------------------------
    def emit_ring_schedules(self):
        for part in (self.attn, self.ffn):
            part.emit_ring_schedules()

    def finish(self, app_name, x_prefixes=("x_wait_", "x_pull_", "x_copy_")):
        A, F = self.attn, self.ffn
        bad = [b for b in F.check_hazards() if "root fold slot" not in b]
        if bad:
            raise ValueError(f"{len(bad)} FFN buffer hazards, e.g. " + "; ".join(bad[:4]))
        print(f"[{self.TAG}] FFN buffer hazards: none (A16, A16a, A8, X, output slots; every cluster)")
        self._check_fold_slots()
        self._check_collectives()
        self._check_prefetch_parts()
        writes = [t for t in self.dfg.node_list if t.kernel_name in COPIES
                  and isinstance(getattr(t.kernel_args, "dst_addr", None), BingoMemFixedAddr)]
        print(f"[{self.TAG}] {len(writes)} device writes into HBM (" + ", ".join(
            f"{p} {sum(1 for t in writes if t.node_name.startswith(p))}" for p in ("n_put", "g_sth", "g_put", "f_st")) + ")"
            + f"; LayerStaging zeroed {self.st.zeroed}, dropped {self.st.dropped}")
        if self.x_slots or self.alias_l3:
            self._check_rings()
        self.dfg.split_chip_functions = True
        self.st.finish()
        # the compiler's dep-tag relays come after the args phases: a relay takes no phase, so its args stay resident, as
        # the compiler's exit tasks' do, and no phase load can take their region
        add_node = self.dfg.bingo_add_node

        def add_resident(node):
            if node.node_name.startswith("relay"):
                node.args_resident = True
            add_node(node)

        self.dfg.bingo_add_node = add_resident
        cm.ChipletBuild.finish(self, app_name, x_prefixes)
        self._check_step_l1()

    def data_h_extra(self):
        return self.st.lines()

    def _check_step_l1(self):
        """The static plan's peak per cluster, and where the steps' buffers sit."""
        pl = getattr(self.dfg, "static_l1_placement", None) or {}
        print(f"[{self.TAG}] static L1 placement: {len(pl)} buffers placed")


def _app_name(S):
    return (f"Llama-3-8B layer (synthetic weights), fresh {S.T}-token prompt, attention {S.attn.NP} passes, FFN "
            f"{S.ffn.NI} passes of {S.ffn.ROWS}, {len(S.chips)} chips x {S.NC} clusters")


def run():
    run_driver("llama3 full-layer prefill workload generator", LayerBuild, _app_name)
