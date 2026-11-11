# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""The driver of the whole-layer prefill workloads at the final block mappings (workloads/dsv2/layer{0,1}/prefill/<app>):
one DeepSeek-V2-Lite decoder layer over a fresh causal prompt (empty cache) on the 16 + 4 chiplet platform
(hemaia_16c4m_8MBL3_2cluster), every step on the device, in ONE graph:

    layer 1:  x -> input RMSNorm -> MLA attention -> + x = h -> post-attention RMSNorm -> MoE           -> + h
    layer 0:  x -> input RMSNorm -> MLA attention -> + x = h -> post-attention RMSNorm -> dense SwiGLU FFN -> + h

THE BLOCKS are the tuned block builders, each with its own block's params: the attention is fa_stages_mc.py's stage
"full" (params attn), the MoE moe_prefill_mc.py's (params moe, layer 1), the dense FFN ffn_prefill_mc.py's (params ffn,
layer 0). Each block reads its input from HBM exactly as it does alone: the attention's x (INT8, A layout) from the
root memory chip's HBM, multicast by that memory chip's push engine into every chip's L3 (x_hbm, x_bcast memchip); the
MoE's and the FFN's X (INT8; row major for the MoE, A layout for the FFN) from the root memory chip's HBM, multicast the
same way (their x_hbm). The layer's steps write those HBM regions, so the blocks meet THROUGH HBM: no activation tensor
is held whole in one chip's L3.

THE LAYER'S DATA: the layer input x (FP16, the hidden states entering the checkpoint's layer 1; fa_prefill_data.py) once
in each memory chip's HBM; the layer's FP16 constants (the attention's output scale, the post-attention gain, the input
gain, the FFN's down scale) once in each memory chip's HBM; the weights stay where each block puts them (HBM). THE
ATTENTION IS ALWAYS LAYER 1'S: its data (its checkpoint weights, input_layernorm gain and the cache of hidden states)
exists for layer 1 only, so layer 0 runs layer 1's attention (the same shapes, so the same timing) in front of layer 0's
dense FFN with layer 0's post_attention_layernorm gain (the FFN data's, model dsv2-lite-l0).

THE STEPS, a 16-row tile at a time (tile q on chip chips[q mod 16], cluster (q // 16) mod 2):
    input norm   x's rows from the chip's memory chip, RMSNorm (simd_rmsnorm, row major), times the input gain
                 (simd_mul_tiles), reshaped to the A layout (xDMA), quantised (simd_fp16_to_int8), into the attention's
                 x region in HBM; each of the attention's x multicasts (chunk 0's tiles, chunk 1's, the rest) waits for
                 the tiles it carries. Every tile of the prompt runs at the window's head: the attention's keys need
                 all of x, and its multicasts run ahead of the root's weight prefetcher on the same engine
    glue         once the attention is done: y's rows from the two collectors of their chunk (the D layout of each W_O
                 block), D -> row major (xDMA), dequantised (W_O's factors), plus x: h; h into HBM (layer 1: in the MoE
                 output's slice layout, where the residual chip reads it; layer 0: row major, where the final reads it);
                 RMSNorm, times the post-attention gain, quantised (layer 0: reshaped to the A layout first) into the
                 MoE's / FFN's X region in HBM; the block's X multicast (layer 0: each pass's) waits for its tiles
    MoE          layer 1: moe_prefill_mc.py's block on that X; one chip (params layer.resid_chip) adds h, read from HBM
                 a piece at a time, into its partial of every output slice, so the in-transit reduce returns MoE(h) + h,
                 which the root stores slice by slice into HBM: the layer's output
    FFN, final   layer 0: ffn_prefill_mc.py's block on that X, its Y_raw stored slice by slice into the root memory
                 chip's HBM (params layer.ffn_y); then per tile the final residual: Y_raw's rows (its slices' D layout),
                 D -> row major, times the down scale, plus h from HBM: the layer's output rows into HBM
The glue and the final run where a block's L1 is free: the blocks' static plans fill every cluster's L1 (the
attention's peaks at 92-94 % of l1_capacity), and a 16-row tile of the glue holds about 290 KB, so neither overlaps
the block beside it.

THE GOLDEN is the device's arithmetic end to end: the attention block's y (each W_O block folded in its collector's
chain order), the steps' operators (hwmodel rmsnorm, mul16, add16, quant_i8), the MoE / FFN block's golden on the
glue's h (its scales calibrated on that h), with h in the residual chip's partial (layer 1) or the final's add
(layer 0).

A PROOF OF CONCEPT (params attn.chunks): the attention at the full geometry makes every token's keys but only the first
chunks' queries, as fa_stages_mc.py does; the MoE / FFN then runs on those chunks' rows (params moe.tokens / ffn.tokens).
The keys of the later chunks run after the window (behind its end stamp, last in every core's program).

L1: each block's static plan overlays the blocks' buffers by lifetime (the steps' tiles take the gaps, a tile's buffers
its own); every device task's args come in phases of at most params args_phase_bytes, two alternating L1 regions
(mini_compiler passes/bingo_args_phases.py), cut over the whole layer. Every block has its own symmetric buffer; their lifetimes
are disjoint, so each sits at offset 0 of its cluster's arena.

L3: the attention's activations (its x landing, the y blocks over it, the keys) and the MoE's / FFN's (its X landing,
its ring of weight slots) are one region, dl_act (LayerStaging): the second block's prefetchers start after its X
multicast, which follows the glue, which follows the attention. With params layer.l3_arena (layer 1) the attention's
weight ring and L3 scratch join that region, past its activations, and the router's weights land past the MoE's ring,
over them; each ring schedule stays in its own chip's image. _check_l3_arena proves the order the overlays rely on.

PARAMETERS (params.hjson): the platform and the graph switches every block shares (num_chiplets, num_clusters,
static_l1, l1_capacity, l1_order, compact_id_maps, gid_lists, per_chip_image, desc_in_image, fast_host_init, d2d_ddr,
d2d_rx_yield, relay_per_gather, args_phase_bytes), then `attn` (fa_stages_mc.py's params), `moe` or `ffn` (that block's
params) and `layer`: layer (1: the MoE, 0: the dense FFN), resid_chip (layer 1: hex chip id), resid_piece (layer 1: bytes
the residual comes in at a time), moe_passes (layer 1: the MoE in that many passes over consecutive rows, each its own
block, moe.tokens rows each), ffn_y (layer 0: hbm | l3, where the FFN's Y_raw is stored), l3_arena (layer 1: the L3
overlays above, for a graph whose image does not fit its L3 without them; default off).
"""

import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WL = os.path.join(HERE, "..", "..")
sys.path.insert(0, os.path.join(os.path.abspath(__file__).split("/target/sw/host/")[0], "util", "sim", "multichip"))
sys.path.insert(0, os.path.join(os.path.abspath(__file__).split("/target/sw/host/")[0], "util", "sim", "common"))
from driver import repo_root, run_driver  # noqa: E402
REPO = repo_root(__file__)
sys.path.insert(0, f"{REPO}/target/sw/host/runtime/libbingo/mini_compiler")
for sub in ("ffn", "fa", "moe"):
    sys.path.insert(0, os.path.join(WL, sub, "common"))
sys.path.insert(0, HERE)

import _bingo_paths  # noqa: F401,E402
from bingo_kernel_args import (  # noqa: E402
    SnaxBingoKernelIdma1dCopyArgs, SnaxBingoKernelIdma2dCopyArgs, SnaxBingoKernelSimdAddF16Args,
    SnaxBingoKernelSimdFp16ToInt8Args, SnaxBingoKernelSimdMulTilesArgs, SnaxBingoKernelSimdRmsnormArgs, simd_beats)
from bingo_mem_handle import BingoMemFixedAddr, BingoMemSymbol  # noqa: E402
from libs.comm.nest import convert_args  # noqa: E402

import chiplet_mc as cm  # noqa: E402
import fa_stages_mc as fas  # noqa: E402
import ffn_data  # noqa: E402
import ffn_prefill_mc as fpm  # noqa: E402
import moe_prefill_data as mpd  # noqa: E402
import moe_prefill_mc as mom  # noqa: E402
from dsv2_prefill_mc import AliasStaging, fa_d_to_rows, rows_to_slices  # noqa: E402

HIDDEN = 2048
TR = 16                        # rows a step tile: one A-layout row block
TB = TR * HIDDEN * 2           # a tile of FP16 rows
XT = TR * HIDDEN               # a tile of INT8 rows (either layout)
CB = HIDDEN * 2                # one FP16 constant row
PASS_TAGS = ("mp", "mq", "mr", "ms")   # the MoE passes' tags (params layer.moe_passes)
# params layer.l3_arena: what of the attention's joins dl_act, past its activations
ARENA_ATTN = ("fp_wring_slots", "fp_kvraw", "fp_wkvc3", "fp_kc3", "fp_wkvx3")
ARENA_ALIGN = 4096
ATTN_PUSHES = ("__host_bingo_kernel_weight_prefetch", "__host_bingo_kernel_weight_prefetch_range",
               "__host_bingo_kernel_hbm_bcast")


def arg_symbols(args, field=""):
    """The names of the L3 symbols a task's args hold, in the fields with `field` in their name."""
    out = set()
    for k, v in (vars(args).items() if args is not None else ()):
        for x in (v if isinstance(v, (list, tuple)) else (v,)):
            if field in k and isinstance(x, BingoMemSymbol):
                out.add(x.symbol_name)
    return out


class LayerStaging(AliasStaging):
    """AliasStaging (names that are one L3 region), and HBM regions the layer's steps write: put_hbm of such a name
    stages zeros of the same size, so a step that never wrote its output fails its check."""

    def __init__(self, st, aliases, zero_hbm=(), chip_arrays=False):
        super().__init__(st, aliases)
        self.zero_hbm = set(zero_hbm)
        self.chip_arrays = [] if chip_arrays else None

    def put(self, name, ctype, arr, mem_chip=None):
        """With chip_arrays a ring schedule goes into its own chip's image only (lines()): only that chip's prefetcher
        reads it, and in the shared data every image would hold every chip's."""
        m = re.fullmatch(r"\w+_ring_sched_k([0-9a-f]{2})", name)
        if self.chip_arrays is None or m is None:
            return self._st.put(name, ctype, arr, mem_chip=mem_chip)
        self.chip_arrays.append((int(m.group(1), 16), name, ctype, np.asarray(arr).reshape(-1)))
        return BingoMemSymbol(name)

    def lines(self):
        out = super().lines()
        for chip, name, ctype, arr in self.chip_arrays or ():
            vals = [f"{int(v)}ULL" for v in arr]
            out += [f"#if !defined(BINGO_ONLY_CHIP) || BINGO_ONLY_CHIP == 0x{chip:02x}",
                    f'{ctype} {name}[{arr.size}] __attribute__ ((aligned (64), section (".bingo_chip.data"))) = {{']
            out += ["    " + ", ".join(vals[i:i + 8]) + "," for i in range(0, len(vals), 8)]
            out += ["};", "#endif"]
        return out

    def put_hbm(self, name, arr, mem_chip=None):
        if name in self.zero_hbm:
            arr = np.zeros_like(np.asarray(arr))
        return self._st.put_hbm(name, arr, mem_chip=mem_chip)


# ---- the blocks, over the layer's graph ----------------------------------------------------------------------------
class AttnPart(fas.FaStages):
    """The attention block (fa_stages_mc.py, stage full) in the layer: its window the layer's, its x the input norm's
    (its multicasts wait for their tiles), its end the glue's start; its keys after the window (a proof of concept's
    later chunks) behind the layer's end stamp; its checks the layer's (h stands for y)."""
    stream_name = "zs_fp_stream"

    def __init__(self, layer, args, p, plat, hw):
        self.layer = layer
        self.post_heads, self.post_tasks = [], []
        super().__init__(args, p, plat, hw, shared=layer)

    def setup(self, extra=None):
        self.go = self.layer.open_window(extra)
        return self.go

    def stage_x_bcast(self, go):
        puts = super().stage_x_bcast(go)
        self.layer.x_after_norm(puts)
        return puts

    def end_window(self, after):
        self.layer.attn_ends = list(after)

    def _checks(self, swp_res, post_res):
        """The layer checks the attention through h (y dequantised plus x): its L3 regions are the second block's by the
        end."""

    def _keys_by_chunk_order(self):
        """fa_stages_mc.py's keys after the window, recorded instead of tied to an end stamp: the layer's end stamp comes
        later (relocate_post_keys). The key tasks of the chunks after the last one made, and those of them that no other
        one of them precedes (they follow the end stamp)."""
        runs = sorted(self._key_runs, key=lambda r: r[0])
        for (a, h), (b, _) in zip(runs, runs[1:]):
            if h is None:
                continue
            for t in self.dfg.node_list:
                if a <= t.node_id < b:
                    t._key_chunk = h
        post = {t for t in self.dfg.node_list if getattr(t, "_key_chunk", 0) >= self.NCH}
        preds = {t: set(self.dfg.predecessors(t)) for t in post}
        for x, y in self.dfg.order_edges:
            if y in preds:
                preds[y].add(x)
        self.post_tasks = sorted(post, key=lambda t: t.node_id)
        self.post_heads = [t for t in self.post_tasks if not preds[t] & post]
        print(f"[{self.TAG}] keys_by_chunk: {len(self.post_tasks)} key tasks of the chunks from {self.NCH} on after the "
              f"layer's window, {len(self.post_heads)} behind its end stamp")

    def relocate_post_keys(self, end):
        """The keys after the window behind the layer's end stamp."""
        for t in self.post_heads:
            self.dfg.bingo_add_edge(end, t)


class MoePart(mom.MBuild):
    """The MoE block (moe_prefill_mc.py) in the layer: its data the glue's h, its X multicast behind the glue's X, its
    window opened per chip after the glue, the residual read from HBM, its output stored into HBM."""
    stream_name = "zs_mp_stream"

    def __init__(self, layer, args, p, plat, hw, n=0, rows=None):
        """Pass n of the MoE (params layer.moe_passes): its rows of h (rows), its output and h at its offset (base)."""
        self.layer, self.n = layer, n
        self.rows = rows
        self.base = n * int(p["tokens"]) * HIDDEN * 2
        self.resid = dict(layer.resid, h16=rows["x16"])
        super().__init__(args, p, plat, hw, shared=layer)

    def _data(self, args, p, hw):
        cache = os.path.join(REPO, p.get("weights", "target/sim/automation/sweep/common/weights/cache"))
        return mpd.MoePrefillData(cache, hw=hw, seed=int(p.get("seed", 1)),
                                  n_calib=int(p.get("n_calib", 8)), n_cal_act=int(p.get("n_cal_act", 32)),
                                  tokens=self.T, inter_e=p.get("inter_e"), inter_sh=p.get("inter_sh"),
                                  rows=self.rows)

    def last(self):
        return self.n == len(self.layer.mlps) - 1

    def setup(self, extra=None):
        self.go = self.layer.mlp_window(extra, self.prefetch, self.n)
        return self.go

    def hbm_bcasts(self, go):
        rw, x = super().hbm_bcasts(go)
        for t in self.layer.glue_done.values():
            self.edge(t, x)
        return rw, x

    def ysym(self, off=0, chip_id=None):
        # the layer's output, slice by slice, into the root memory chip's HBM (this pass's rows)
        return BingoMemFixedAddr(self.layer.h_out.address + self.base + off, mem_level="HBM")

    def _resid_add(self, g, jl, y, after):
        """moe_prefill_mc.py's residual add, h read from its HBM region (the glue's, in the output's slice layout)."""
        k, c = self.chip(g), self.c(g)
        dm, simd = self.roles["dm"], self.roles["simd"]
        pc, hb = self.resid["piece"], self.B[g]["hres"]
        prev = after
        for o in range(0, self.SL, pc):
            war = [self._hres_reader[g]] if self._hres_reader.get(g) is not None else []
            ld = self.node(k, c, dm, f"res_ld_j{jl}o{o}_g{g}", "__snax_bingo_kernel_idma_1d_copy",
                           SnaxBingoKernelIdma1dCopyArgs(
                               BingoMemFixedAddr(self.layer.h_h3.address + self.base + self.y_off(c, jl) + o,
                                                 mem_level="HBM"),
                               hb, pc), war)
            prev = self.node(k, c, simd, f"res_add_j{jl}o{o}_g{g}", "__snax_bingo_kernel_simd_stream_elementwise",
                             SnaxBingoKernelSimdAddF16Args(y.view(o), hb, y.view(o), pc // 64, 32), [ld, prev])
            self._hres_reader[g] = prev
        return prev

    def end_window(self, after):
        """The last pass ends the layer's window; an earlier one hands its end to the next pass's window."""
        self.ends = list(after)
        if self.last():
            self.layer.end_window(after)
            self.end, self.check_copies = self.layer.end, self.layer.check_copies

    def _checks(self):
        """An earlier pass's checks wait for the layer's end stamp (the layer makes them after the last pass)."""
        if self.last():
            super()._checks()
        else:
            self.layer.deferred.append(self)

    def stash(self, name, k, c, src, nbytes, after):
        # an earlier pass's L1 holds the later passes' buffers by the end: its stash checks are left out
        return None if not self.last() else super().stash(f"{self.TAG}_{name}", k, c, src, nbytes, after)

    def run_checks(self, checks):
        self.layer.block_checks += [(f"{self.TAG}_{nm}", got, gold, nb) for nm, got, gold, nb in checks
                                    if got is not None]


class FfnPart(fpm.PBuild):
    """The dense FFN block (ffn_prefill_mc.py) in the layer: its data the glue's h, each pass's X multicast behind the
    glue's tiles of that pass, its window opened per chip after the glue, then the layer's final residual."""
    stream_name = "zs_pf_stream"

    def __init__(self, layer, args, p, plat, hw):
        self.layer = layer
        super().__init__(args, p, plat, hw, shared=layer)

    def _data(self, args, p):
        cache = os.path.join(REPO, p.get("weights", "target/sim/automation/sweep/llm_blocks/ffn/cache"))
        return ffn_data.FFNData(cache, p.get("model", "dsv2-lite-l0"), seed=int(p.get("seed", 1)),
                                wbits=self.wbits, tokens=self.T, inter=p.get("inter"), rows=self.layer.mlp_rows)

    def setup(self, extra=None):
        self.go = self.layer.mlp_window(extra, self.prefetch, 0)
        return self.go

    def x_hbm_bcasts(self, go):
        out = super().x_hbm_bcasts(go)
        # no multicast beside the glue's transfers: the first (the others follow it) behind every chip's glue probe,
        # which follows every X store of the chip; one edge a chip into the root's host
        for t in self.layer.glue_done.values():
            self.edge(t, out[0])
        return out

    def end_window(self, after):
        fin = self.layer.final(list(after))
        self.layer.end_window(fin)
        self.end, self.check_copies = self.layer.end, self.layer.check_copies

    def _ysym(self, off, **kw):
        # params layer.ffn_y hbm: Y_raw, slice by slice, into the root memory chip's HBM, where the final reads it
        if self.layer.ffn_y_hbm:
            return BingoMemFixedAddr(self.layer.h_ffn_y.address + off, mem_level="HBM")
        return super()._ysym(off, **kw)

    def _xland(self, off, chip_id=None):
        # params layer.ffn_y hbm: no Y_raw over the X buffers, so each pass's X lands at its own offset, back to back
        if self.layer.ffn_y_hbm and self.l3_overlay:
            return BingoMemSymbol(f"{self.TAG}_xy", offset=off, chip_id=chip_id)
        return super()._xland(off, chip_id)

    def run_checks(self, checks):
        # params layer.ffn_y hbm: Y_raw of the first pass's head and the last pass's tail (HBM reads are slow)
        keep = {"y_i0_a", f"y_i{self.NI - 1}_b", "a8_root"} if self.layer.ffn_y_hbm else None
        self.layer.block_checks += [ch for ch in checks if keep is None or ch[0] in keep]


# ---- the layer -------------------------------------------------------------------------------------------------------
class LayerBuild(cm.ChipletBuild):
    TAG = "dl"

    def __init__(self, args, p, plat, hw):
        common = {k: v for k, v in p.items() if k not in ("attn", "moe", "ffn", "layer")}
        lp = self.lp = dict(p.get("layer", {}))
        super().__init__(args, common, plat, hw)
        self.which = int(lp.get("layer", 1))
        if self.which not in (0, 1):
            raise ValueError(f"params layer.layer={self.which}: 1 (the MoE) or 0 (the dense FFN)")
        if self.which == 1 and "moe" not in p or self.which == 0 and "ffn" not in p:
            raise ValueError("params: layer 1 needs a moe section, layer 0 an ffn section")
        mtag = "mp" if self.which == 1 else "pf"
        # one L3 region for both blocks' activations (the module docstring): the attention's x landing, the y blocks over
        # it and the keys first; the second block's X landing (and Y) and its weight slots after the attention
        pa = dict(common, **p["attn"])
        T_attn = int(pa.get("tokens", 0))
        XS = T_attn * HIDDEN
        # params layer.moe_passes (layer 1): the MoE in that many passes over consecutive rows, one after the other, each
        # its own block (its own route, placement and weight stream); their tags
        self.n_pass = int(lp.get("moe_passes", 1)) if self.which == 1 else 1
        if not 1 <= self.n_pass <= len(PASS_TAGS):
            raise ValueError(f"params layer.moe_passes={self.n_pass}: 1..{len(PASS_TAGS)}")
        tags = PASS_TAGS[:self.n_pass] if self.which == 1 else ("pf",)
        aliases = {"fp_xsrc": ("dl_act", 0), "fp_xland": ("dl_act", XS), "fp_kv": ("dl_act", 2 * XS)}
        aliases.update({f"{t}_xy": ("dl_act", 0) for t in tags})
        # every pass multicasts the same router weights into the same landing
        aliases.update({f"{t}_rwland": "dl_rwland" for t in tags if self.which == 1})
        # params layer.l3_arena (layer 1): the L3 overlays of the module docstring; _place_l3_arena sets the offsets once
        # every block has asked for its sizes
        self.l3_arena = bool(lp.get("l3_arena", False))
        if self.l3_arena:
            if self.which != 1:
                raise ValueError("params layer.l3_arena: layer 1")
            aliases.update({n: ("dl_act", 0) for n in ARENA_ATTN})
            aliases.update({f"{t}_rwland": ("dl_act", 0) for t in tags})
        self.st = LayerStaging(self.st, aliases, zero_hbm=("fp_xhbm",) + tuple(f"{t}_xhbm" for t in tags),
                               chip_arrays=self.l3_arena)
        self.sym_zero = False
        self.phase_starts = []
        self.attn_ends, self.block_checks, self.deferred = [], [], []

        # ---- the attention ------------------------------------------------------------------------------
        if pa.get("stage") != "full" or not pa.get("x_hbm", True) or pa.get("x_bcast") != "memchip":
            raise ValueError("params attn: stage full, x from HBM multicast by the memory chip (x_hbm, x_bcast memchip)")
        A = self.attn = AttnPart(self, args, pa, plat, hw)
        d, fp = A.d, A.d.fp
        self.fp = fp
        self.NB = int(A.chunks_built) if A.chunks_built is not None else A.NCH      # the chunks made
        self.T_mlp = self.NB * A.C
        self.x16 = np.asarray(d.x1, dtype=np.float16)                                 # [T, 2048]
        self.oscale = np.asarray(d.out_scale(), dtype=np.float16)
        self.in_gain = np.asarray(d.in_norm, dtype=np.float16)
        # y of the chunks made, each W_O block folded in its collector's chain order
        y = np.zeros((self.T_mlp, HIDDEN), dtype=np.float16)
        half = HIDDEN // 2
        for i in range(self.NB):
            for c in range(self.NC):
                r0 = A.r0(i, c)
                yh = [A.y_raw(r0, A.ROWS, A.y_fold_heads(A.y_root(i, h)) if A.y_roots else None) for h in range(2)]
                y[r0:r0 + A.ROWS, :half] = yh[0][:, :half]
                y[r0:r0 + A.ROWS, half:] = yh[1][:, half:]
        self.y_raw = y
        self.h16 = fp.add16(fp.mul16(y, self.oscale[None, :]), self.x16[:self.T_mlp])     # the residual stream
        n_cal_act = int(p.get("moe", {}).get("n_cal_act", 32))
        self.mlp_rows = dict(x16=self.h16, cal=self.h16, cal_act=self.h16[-n_cal_act:])

        # ---- the MoE (layer 1) or the dense FFN (layer 0) --------------------------------------------------
        self.resid = None
        if self.which == 1:
            pm = dict(common, **p["moe"])
            Tp = self.T_mlp // self.n_pass
            if int(pm.get("tokens", 0)) * self.n_pass != self.T_mlp:
                raise ValueError(f"params moe.tokens: a pass's rows, the {self.T_mlp} rows of the attention's chunks made "
                                 f"over {self.n_pass} pass(es)")
            for k_ in ("x_device", "args_phases"):
                if pm.get(k_):
                    raise ValueError(f"params moe.{k_}: the layer reads the MoE's X from HBM and cuts the args phases")
            pm["x_hbm"] = True
            self.resid_chip = int(str(lp.get("resid_chip", "33")), 16)
            if self.resid_chip not in self.chips or self.resid_chip == self.ROOT:
                raise ValueError(f"params layer.resid_chip={lp.get('resid_chip')}: a compute chip other than the root")
            self.resid = dict(chip=self.resid_chip, src=None, piece=int(lp.get("resid_piece", 16384)), h16=self.h16)
            self.SW = int(pm.get("slice", 128))
            self.Tp = Tp
            SL = Tp * self.SW * 2
            if SL % self.resid["piece"] or self.resid["piece"] % 64:
                raise ValueError(f"params layer.resid_piece={self.resid['piece']}: whole 64-byte rows dividing a {SL}-byte "
                                 f"slice")
            self.mlps = []
            for n, tag in enumerate(tags):
                # its weight slots behind its X landing (l3_overlay: [source | landing | zero row]); a later pass's over
                # the pass before's (it starts once that pass is done). The input scale is calibrated on every row of
                # the layer: one static scale for every pass
                self.st.aliases[f"{tag}_wring_slots"] = ("dl_act", (2 * Tp + 1) * HIDDEN)
                cls = type(f"MoePart_{tag}", (MoePart,), {"TAG": tag, "stream_name": f"zs_{tag}_stream"})
                rows = dict(self.mlp_rows, x16=self.h16[n * Tp:(n + 1) * Tp])
                self.mlps.append(cls(self, args, pm, plat, hw, n=n, rows=rows))
            M = self.mlp = self.mlps[-1]
            self.post_gain = np.asarray(M.d.gain16, dtype=np.float16)
            self.s_h = M.d.s_h
            self.out16 = np.concatenate([m.y_raw for m in self.mlps])
            for m in self.mlps:
                if m.d.s_h.s != self.s_h.s:
                    raise AssertionError("the MoE passes' input scales differ")
        else:
            pf = dict(common, **p["ffn"])
            if int(pf.get("tokens", 0)) != self.T_mlp:
                raise ValueError(f"params ffn.tokens: the {self.T_mlp} rows of the attention's chunks made")
            if pf.get("y_scatter") or not pf.get("l3_overlay", False):
                raise ValueError("params ffn: Y_raw on the root over its X (l3_overlay, no y_scatter): the final reads it "
                                 "there")
            pf.pop("args_phase_bytes", None)
            # params layer.ffn_y (hbm | l3): the FFN's Y_raw into the root memory chip's HBM (each pass's X landing then
            # back to back in L3), or in the root's L3 over its X buffers, as the block alone (l3_overlay)
            self.ffn_y_hbm = str(lp.get("ffn_y", "hbm")) == "hbm"
            if str(lp.get("ffn_y", "hbm")) not in ("hbm", "l3"):
                raise ValueError(f"params layer.ffn_y={lp.get('ffn_y')}: hbm or l3")
            # its weight slots behind its X landing (l3_overlay: a pass's [source | landing], Y_raw over both; ffn_y hbm:
            # the passes' landings)
            self.st.aliases["pf_wring_slots"] = ("dl_act", (1 if self.ffn_y_hbm else 2) * self.T_mlp * HIDDEN)
            M = self.mlp = FfnPart(self, args, pf, plat, hw)
            self.mlps = [M]
            if M.ROWS != A.C:
                raise ValueError("params ffn.rows: the attention's chunk (a pass a chunk)")
            self.post_gain = np.asarray(M.d.gain16, dtype=np.float16)
            self.s_h = M.d.s_h
            self.dn_f = np.asarray(M.d.dn_f, dtype=np.float16)
            self.out16 = fp.add16(fp.mul16(M.gold["y_raw"], self.dn_f[None, :]), self.h16)
        self.x8 = fp.quant_i8(fp.mul16(d.hm.rmsnorm(self.h16), self.post_gain[None, :]), self.s_h.inv)
        if not np.array_equal(self.x8, np.concatenate([m.d.hq for m in self.mlps])):
            raise AssertionError("the glue's X and the block's golden X differ")
        if not np.array_equal(np.asarray(d.hq), fp.quant_i8(fp.mul16(d.hm.rmsnorm(self.x16), self.in_gain[None, :]),
                                                             d.s_x.inv)):
            raise AssertionError("the input norm's x and the attention's golden x differ")

        # ---- the layer's own data ------------------------------------------------------------------------
        st, Tg = self.st, self.TAG
        mems = sorted(set(self.mem.values()))
        self.h_x1 = {m: st.put_hbm(f"{Tg}_x1_m{m:02x}", self.x16.view(np.uint8).reshape(-1), mem_chip=(m >> 4, m & 15))
                     for m in mems}
        consts = np.concatenate([self.oscale, self.post_gain, self.in_gain,
                                 self.dn_f if self.which == 0 else np.zeros(HIDDEN, np.float16)])
        self.h_c = {m: st.put_hbm(f"{Tg}_c_m{m:02x}", consts.view(np.uint8).reshape(-1), mem_chip=(m >> 4, m & 15))
                    for m in mems}
        nb = self.T_mlp * HIDDEN * 2
        if self.which == 1:
            rm, R = self.mem[self.resid_chip], self.mem[self.ROOT]
            self.h_h3 = st.put_hbm(f"{Tg}_h3", np.zeros(nb, np.uint8), mem_chip=(rm >> 4, rm & 15))
            self.h_out = st.put_hbm(f"{Tg}_out", np.zeros(nb, np.uint8), mem_chip=(R >> 4, R & 15))
        else:
            # h and the output: each tile in the HBM of its owner's memory chip
            self.h_h = {m: st.put_hbm(f"{Tg}_h_m{m:02x}", np.zeros(nb, np.uint8), mem_chip=(m >> 4, m & 15))
                        for m in mems}
            self.h_outm = {m: st.put_hbm(f"{Tg}_out_m{m:02x}", np.zeros(nb, np.uint8), mem_chip=(m >> 4, m & 15))
                           for m in mems}
            if self.ffn_y_hbm:
                R = self.mem[self.ROOT]
                self.h_ffn_y = st.put_hbm(f"{Tg}_ffn_y", np.zeros(nb, np.uint8), mem_chip=(R >> 4, R & 15))
        print(f"[{Tg}] layer {self.which}: attention {A.T} tokens, {self.NB} of {A.NCH} chunks made; "
              f"{'MoE' if self.which == 1 else 'FFN'} {self.T_mlp} tokens; |h| max "
              f"{float(np.max(np.abs(self.h16))):.3g}, rms {float(np.sqrt(np.mean(self.h16.astype(np.float32) ** 2))):.3g}")

    # ---- the tiles ---------------------------------------------------------------------------------------------
    def owner(self, q):
        """(chip, cluster) of 16-row tile q of every step."""
        return self.chips[q % len(self.chips)], (q // len(self.chips)) % self.NC

    def per_owner(self, n_tiles):
        out = {}
        for q in range(n_tiles):
            out.setdefault(self.owner(q), []).append(q)
        return out

    def hbm(self, h, off=0):
        return BingoMemFixedAddr(h.address + off, mem_level="HBM")

    def _on(self, sym, k, chip):
        return sym if k == chip else BingoMemSymbol(sym.symbol_name, offset=sym.offset, chip_id=chip)

    def _bufs(self, k, c, name, kinds):
        """A tile's own L1 buffers on chip k cluster c (named to sort after the blocks' buffers)."""
        sizes = dict(a=TB, b=TB, d=TB, x8=XT, scr=SnaxBingoKernelSimdRmsnormArgs(0, 0, TR, HIDDEN).scratch_bytes())
        return {kd: self.cx_chip(k).at(c).l1(f"zt_{self.TAG}_{name}_{kd}", sizes[kd]) for kd in kinds}

    def _consts(self, k, c, name, off, nb, deps):
        """nb bytes of the layer's constants from offset `off` into chip k cluster c's L1, from its memory chip."""
        buf = self.cx_chip(k).at(c).l1(f"zt_{self.TAG}_{name}c", nb)
        ld = self.node(k, c, self.roles["dm"], f"{name}_c_ld", "__snax_bingo_kernel_idma_1d_copy",
                       SnaxBingoKernelIdma1dCopyArgs(self.hbm(self.h_c[self.mem[k]], off), buf, nb), list(deps))
        return buf, ld

    def _rms_gain(self, k, c, name, src, dst, scr, gain, deps):
        simd = self.roles["simd"]
        rn = self.node(k, c, simd, f"{name}_rms", "__snax_bingo_kernel_simd_rmsnorm",
                       SnaxBingoKernelSimdRmsnormArgs(src, dst, TR, HIDDEN, scratch_addr=scr), list(deps))
        return self.node(k, c, simd, f"{name}_gain", "__snax_bingo_kernel_simd_mul_tiles",
                         SnaxBingoKernelSimdMulTilesArgs(dst, gain, dst, TR, HIDDEN // 128, 2 * HIDDEN, 2 * HIDDEN), [rn])

    # ---- the window ------------------------------------------------------------------------------------------
    def open_window(self, extra):
        """The start stamp (after every chip's setup), the attention's prefetchers behind it, and the input norm."""
        ends = [t for k in self.chips for t in (extra or {}).get(k, [])]
        start = self.stamp("start", ends)
        for k in self.chips:
            self.edge(start, self.attn.prefetch[k])
        self.go = {k: start for k in self.chips}
        self.input_norm(start)
        return self.go

    def input_norm(self, start):
        """Every 16-row tile of the prompt's x, normed, gained and quantised (A layout) into the attention's x region
        in HBM; self.norm_st {tile: its store}."""
        A, dm, xdma, simd = self.attn, self.roles["dm"], self.roles["xdma"], self.roles["simd"]
        inv = self.fp.f32bits(A.d.s_x.inv)
        self.norm_st = {}
        for (k, c), qs in sorted(self.per_owner(A.T // TR).items()):
            gain, cld = self._consts(k, c, "n", 2 * CB, CB, [start])
            prev = None
            for n, q in enumerate(qs):
                b = self._bufs(k, c, f"n{n}", ("a", "b", "x8", "scr"))
                # the tile's rows land once the tile before has been normed (its rows read)
                ld = self.node(k, c, dm, f"n_ld_q{q}", "__snax_bingo_kernel_idma_1d_copy",
                               SnaxBingoKernelIdma1dCopyArgs(self.hbm(self.h_x1[self.mem[k]], q * TB), b["b"], TB),
                               [start] + ([prev] if prev is not None else []))
                gm = self._rms_gain(k, c, f"n_q{q}", b["b"], b["a"], b["scr"], gain, [ld, cld])
                prev = gm
                cv = self.node(k, c, xdma, f"n_toa_q{q}", "__snax_bingo_kernel_xdma_6d",
                               convert_args("row_major", "A", TR, HIDDEN, cm.MESH, 2, b["a"], b["b"]), [gm])
                qz = self.node(k, c, simd, f"n_quant_q{q}", "__snax_bingo_kernel_simd_fp16_to_int8",
                               SnaxBingoKernelSimdFp16ToInt8Args(b["b"], b["x8"], beats=simd_beats(TB), rows=1,
                                                                 inv_scale_f32bits=inv), [cv])
                self.norm_st[q] = self.node(k, c, dm, f"n_st_q{q}", "__snax_bingo_kernel_idma_1d_copy",
                                            SnaxBingoKernelIdma1dCopyArgs(b["x8"], self.hbm(A.h_xhbm, q * XT), XT), [qz])

    def x_after_norm(self, puts):
        """Each of the attention's x multicasts behind the input norm's tiles it carries: per cluster the last of them
        (a cluster's stores run in order on its DM core)."""
        A = self.attn
        P = A.C // 16
        cuts = ([0, P] + ([2 * P] if A.NX > 2 * P else []) + [A.NX]) if A.keys_by_chunk else [0, A.NX]
        if len(cuts) - 1 != len(puts):
            raise ValueError(f"{len(puts)} x multicasts, {len(cuts) - 1} cuts of x")
        for put, a, b in zip(puts, cuts, cuts[1:]):
            last = {}
            for q in range(a, b):
                key = self.owner(q)
                if key not in last or self.norm_st[q].node_id > last[key].node_id:
                    last[key] = self.norm_st[q]
            for t in last.values():
                self.edge(t, put)

    def glue(self):
        """Once the attention is done, every tile of the chunks made: h = x + dequantised y into HBM, and the block's X
        from h into its X region in HBM. self.glue_x {tile: its X store}, self.glue_done {chip: the probe after its
        tiles}."""
        A = self.attn
        dm, xdma, simd = self.roles["dm"], self.roles["xdma"], self.roles["simd"]
        inv = self.fp.f32bits(self.s_h.inv)
        # every y block's store on its collector (chunk, cluster, W_O block): a tile's loads follow its own two, which the
        # attention's end (the last chunk's last stores) need not
        y_st = {}
        for t in self.dfg.node_list:
            m = re.match(r"y_store_c(\d)_i(\d+)_h(\d)_k[0-9a-f]{2}(?:_cl\d)?$", t.node_name)
            if m:
                y_st[tuple(int(v) for v in m.groups())] = t
        self.glue_x, last = {}, {}
        tp = self.T_mlp // self.n_pass // TR                                 # tiles a MoE / FFN pass's X region holds
        SL = tp * TR * getattr(self, "SW", 0) * 2
        for (k, c), qs in sorted(self.per_owner(self.T_mlp // TR).items()):
            cbuf, cld = self._consts(k, c, "g", 0, 2 * CB, self.attn_ends)
            prev = None
            for n, q in enumerate(qs):
                i, j = q // (A.C // TR), q % (A.C // TR)
                hh, tt = j // (A.ROWS // TR), j % (A.ROWS // TR)            # the half of the chunk's rows, the tile in it
                cy = next(cc for cc in range(self.NC) if A.xh(i, cc) == hh)  # the cluster whose rows they are
                b = self._bufs(k, c, f"g{n}", ("a", "b", "d", "x8", "scr"))
                deps = list(self.attn_ends) + ([prev] if prev is not None else [])
                ldy = [self.node(k, c, dm, f"g_ldy{h}_q{q}", "__snax_bingo_kernel_idma_1d_copy",
                                 SnaxBingoKernelIdma1dCopyArgs(
                                     self._on(A.y_sym(i, cy, h * A.YB // 2 + tt * TB // 2), k, A.y_root(i, h)),
                                     b["d"].view(h * TB // 2), TB // 2), deps + [y_st[(cy, i, h)]]) for h in range(2)]
                ldx = self.node(k, c, dm, f"g_ldx_q{q}", "__snax_bingo_kernel_idma_1d_copy",
                                SnaxBingoKernelIdma1dCopyArgs(self.hbm(self.h_x1[self.mem[k]], q * TB), b["b"], TB), deps)
                cv = self.node(k, c, xdma, f"g_rows_q{q}", "__snax_bingo_kernel_xdma_6d",
                               fa_d_to_rows(b["d"], b["a"], TR), ldy)
                dq = self.node(k, c, simd, f"g_deq_q{q}", "__snax_bingo_kernel_simd_mul_tiles",
                               SnaxBingoKernelSimdMulTilesArgs(b["a"], cbuf, b["a"], TR, HIDDEN // 128, 2 * HIDDEN,
                                                               2 * HIDDEN), [cv, cld])
                ad = self.node(k, c, simd, f"g_add_q{q}", "__snax_bingo_kernel_simd_stream_elementwise",
                               SnaxBingoKernelSimdAddF16Args(b["a"], b["b"], b["b"], TB // 64, 32), [dq, ldx])
                prev = ad
                M, ql = self.mlps[q // tp], q % tp                           # its pass, its tile there
                if self.which == 1:
                    # h in the MoE output's slice layout, slice-major, where the residual chip reads it
                    h3 = self.node(k, c, xdma, f"g_h3_q{q}", "__snax_bingo_kernel_xdma_6d",
                                   rows_to_slices(b["b"], b["d"], TR, self.SW), [ad, cv])
                    sth = self.node(k, c, dm, f"g_sth_q{q}", "__snax_bingo_kernel_idma_2d_copy",
                                    SnaxBingoKernelIdma2dCopyArgs(b["d"], self.hbm(self.h_h3, M.base + ql * TR * self.SW * 2),
                                                                  TR * self.SW * 2, TR * self.SW * 2, SL,
                                                                  HIDDEN // self.SW), [h3])
                    gm = self._rms_gain(k, c, f"g_q{q}", b["b"], b["a"], b["scr"], cbuf.view(CB), [ad])
                    q_src, q_dep, xoff, xh = b["a"], gm, ql * XT, M.h_xhbm
                else:
                    # h row major, in the HBM of the tile's memory chip, where the final reads it
                    sth = self.node(k, c, dm, f"g_sth_q{q}", "__snax_bingo_kernel_idma_1d_copy",
                                    SnaxBingoKernelIdma1dCopyArgs(b["b"], self.hbm(self.h_h[self.mem[k]], q * TB), TB),
                                    [ad])
                    gm = self._rms_gain(k, c, f"g_q{q}", b["b"], b["a"], b["scr"], cbuf.view(CB), [ad])
                    q_dep = self.node(k, c, xdma, f"g_toa_q{q}", "__snax_bingo_kernel_xdma_6d",
                                      convert_args("row_major", "A", TR, HIDDEN, cm.MESH, 2, b["a"], b["d"]), [gm, cv])
                    q_src, xoff, xh = b["d"], q * XT, M.h_xhbm
                qz = self.node(k, c, simd, f"g_quant_q{q}", "__snax_bingo_kernel_simd_fp16_to_int8",
                               SnaxBingoKernelSimdFp16ToInt8Args(q_src, b["x8"], beats=simd_beats(TB), rows=1,
                                                                 inv_scale_f32bits=inv), [q_dep])
                self.glue_x[q] = self.node(k, c, dm, f"g_stx_q{q}", "__snax_bingo_kernel_idma_1d_copy",
                                           SnaxBingoKernelIdma1dCopyArgs(b["x8"], self.hbm(xh, xoff), XT), [qz, sth])
                last.setdefault(k, []).append(self.glue_x[q])
        # per chip, the probe its block's window opens behind: every tile of the chip done (its X store follows its h
        # store), the attention's end
        self.glue_done = {k: self.probe(k, "glue_done", list(self.attn_ends) + last.get(k, [])) for k in self.chips}

    def mlp_window(self, extra, prefetch, n):
        """Where the MoE / FFN (pass n) starts on each chip: behind the chip's glue probe (pass n > 0: behind the pass
        before's end on every chip, the root's last stores), then the block's own setup tasks; its prefetchers behind
        it."""
        go = {}
        for k in self.chips:
            if n:
                opened = self.probe(k, f"{self.mlps[n].TAG}_open", self.mlps[n - 1].ends)
            elif k == getattr(self, "resid_chip", None):
                # the residual chip reads every chip's h: behind every chip's glue
                opened = self.probe(k, "resid_open", list(self.glue_done.values()))
            else:
                opened = self.glue_done[k]
            pre = list((extra or {}).get(k, []))
            for t in pre:
                self.edge(opened, t)
            go[k] = self.probe(k, f"{self.mlps[n].TAG}_go", [opened] + pre) if pre else opened
            self.edge(go[k], prefetch[k])
        return go

    def final(self, after):
        """Layer 0: per tile the FFN's Y_raw rows from the root's L3, D -> row major, times the down scale, plus h: the
        layer's output rows into HBM. Returns the stores."""
        F, R = self.mlp, self.ROOT
        dm, xdma, simd = self.roles["dm"], self.roles["xdma"], self.roles["simd"]
        out = []
        for (k, c), qs in sorted(self.per_owner(self.T_mlp // TR).items()):
            dnf, cld = self._consts(k, c, "f", 3 * CB, CB, after)
            prev = None
            for n, q in enumerate(qs):
                i, tt = q // F.MT, q % F.MT
                b = self._bufs(k, c, f"f{n}", ("a", "b", "d"))
                deps = list(after) + ([prev] if prev is not None else [])
                lds = []
                for o0, w in F.y_slices():
                    lds.append(self.node(k, c, dm, f"f_ldy_q{q}o{o0}", "__snax_bingo_kernel_idma_1d_copy",
                                         SnaxBingoKernelIdma1dCopyArgs(
                                             F._ysym(i * F.YB + o0 * F.ROWS * 2 + tt * TR * w * 2,
                                                     **({} if k == R else {"chip_id": R})),
                                             b["d"].view(o0 * TR * 2), TR * w * 2), deps))
                ldh = self.node(k, c, dm, f"f_ldh_q{q}", "__snax_bingo_kernel_idma_1d_copy",
                                SnaxBingoKernelIdma1dCopyArgs(self.hbm(self.h_h[self.mem[k]], q * TB), b["b"], TB), deps)
                cv = self.node(k, c, xdma, f"f_rows_q{q}", "__snax_bingo_kernel_xdma_6d",
                               fa_d_to_rows(b["d"], b["a"], TR), lds)
                dq = self.node(k, c, simd, f"f_deq_q{q}", "__snax_bingo_kernel_simd_mul_tiles",
                               SnaxBingoKernelSimdMulTilesArgs(b["a"], dnf, b["a"], TR, HIDDEN // 128, 2 * HIDDEN,
                                                               2 * HIDDEN), [cv, cld])
                ad = self.node(k, c, simd, f"f_add_q{q}", "__snax_bingo_kernel_simd_stream_elementwise",
                               SnaxBingoKernelSimdAddF16Args(b["a"], b["b"], b["b"], TB // 64, 32), [dq, ldh])
                prev = ad
                out.append(self.node(k, c, dm, f"f_st_q{q}", "__snax_bingo_kernel_idma_1d_copy",
                                     SnaxBingoKernelIdma1dCopyArgs(b["b"], self.hbm(self.h_outm[self.mem[k]], q * TB), TB),
                                     [ad]))
        return out

    # ---- the layer ----------------------------------------------------------------------------------------------
    def build(self):
        self.attn.build()
        self.glue()
        for m in self.mlps:
            m.build()
        self.close_checks()

    def close_checks(self):
        """The layer's checks, upstream first, then the blocks': sampled windows, byte for byte against the golden (an
        earlier MoE pass's checks made now, behind the layer's end stamp)."""
        for m in self.deferred:
            m.end, m.check_copies = self.end, self.check_copies
            mom.MBuild._checks(m)
        A, M, R = self.attn, self.mlp, self.ROOT
        n = 1024
        checks = []
        # the input norm's x in HBM: the head of the first tile, the middle of a tile of the rest, the tail of the last
        xa = A.xsrc.view(np.uint8)
        nx = A.T // TR
        for nm, off in (("x8_a", 0), ("x8_m", (nx // 2 + 5) * XT + XT // 2), ("x8_b", nx * XT - n)):
            checks.append((nm, self.hbm(A.h_xhbm, off), xa[off:off + n], n))
        nt = self.T_mlp // TR
        if self.which == 1:
            # h in the slice layout: slice 0 of the first tile, the last slice of the last tile (16 rows x 64 columns)
            h3 = np.concatenate([A.lay.to_d(self.h16[:, s * self.SW:(s + 1) * self.SW], mom.MESH3)
                                 for s in range(HIDDEN // self.SW)]).astype(np.float16).view(np.uint8)
            # (each MoE pass's rows a region of their own, slice-major)
            h3 = np.concatenate([np.concatenate([A.lay.to_d(self.h16[p * self.Tp:(p + 1) * self.Tp,
                                                                     s * self.SW:(s + 1) * self.SW], mom.MESH3)
                                                 for s in range(HIDDEN // self.SW)]) for p in range(self.n_pass)]
                                ).astype(np.float16).view(np.uint8)
            SL, tb = self.Tp * self.SW * 2, TR * self.SW * 2
            last = (self.n_pass - 1) * self.Tp * HIDDEN * 2 + (HIDDEN // self.SW - 1) * SL + (self.Tp // TR - 1) * tb
            for nm, off in (("h3_a", 0), ("h3_b", last)):
                checks.append((nm, self.hbm(self.h_h3, off), h3[off:off + tb], tb))
            # X in HBM, row major: the head of the first row, the tail of the last (the first and the last pass)
            x8 = self.x8.astype(np.int8).view(np.uint8).reshape(-1)
            xp = self.Tp * HIDDEN
            for nm, m, off in (("xm_a", self.mlps[0], 0), ("xm_b", self.mlps[-1], xp - n)):
                o = m.n * xp + off
                checks.append((nm, self.hbm(m.h_xhbm, off), x8[o:o + n], n))
        else:
            h = self.h16.view(np.uint8).reshape(-1)
            for nm, q, off in (("h_a", 0, 0), ("h_b", nt - 1, TB - n)):
                o = q * TB + off
                checks.append((nm, self.hbm(self.h_h[self.mem[self.owner(q)[0]]], o), h[o:o + n], n))
            xa8 = np.concatenate([A.lay.to_a(self.x8[q * TR:(q + 1) * TR]) for q in range(nt)]).astype(np.int8).view(np.uint8)
            for nm, off in (("xf_a", 0), ("xf_b", xa8.size - n)):
                checks.append((nm, self.hbm(M.h_xhbm, off), xa8[off:off + n], n))
        checks += list(self.block_checks)
        if self.which == 0:
            o16 = self.out16.view(np.uint8).reshape(-1)
            for nm, q, off in (("out_a", 0, 0), ("out_m", nt // 2, TB // 2), ("out_b", nt - 1, TB - n)):
                o = q * TB + off
                checks.append((nm, self.hbm(self.h_outm[self.mem[self.owner(q)[0]]], o), o16[o:o + n], n))
        # the hold that keeps the second block's symmetric buffers live to the end
        self.sym = M.sym
        cm.ChipletBuild.run_checks(self, checks)

    # ---- emission -----------------------------------------------------------------------------------------------
    def emit_ring_schedules(self):
        for part in [self.attn] + self.mlps:
            part.emit_ring_schedules()

    def data_h_extra(self):
        return self.st.lines()

    def check_core_order(self, header, x_prefixes):
        self.attn.check_core_order(header, x_prefixes)

    def finish(self, app_name, x_prefixes=("x_wait_", "x_pull_", "x_copy_", "kx_ld_")):
        A, M = self.attn, self.mlp
        A.relocate_post_keys(self.end)
        self.prune_post_edges()
        if A.keys_by_chunk:
            A._order_collectives()
            A._check_multicasts()
        if self.which == 0:
            bad = M.check_hazards()
            if bad:
                raise ValueError(f"{len(bad)} FFN buffer hazards, e.g. " + "; ".join(bad[:4]))
            print(f"[{self.TAG}] FFN buffer hazards: none")
        self.dfg.split_chip_functions = True
        if self.l3_arena:
            self._place_l3_arena()
            self._check_l3_arena()
        self.st.finish()
        super().finish(app_name, x_prefixes)
        # every block's symmetric buffer at offset 0 of its cluster's arena (super checked the second block's)
        pl = self.dfg.static_l1_placement
        for part in [A] + self.mlps:
            bad = [(k, c, pl[id(h)][1]) for (k, c), h in part.sym.items() if id(h) not in pl or pl[id(h)][1] != 0]
            if bad:
                raise ValueError(f"the {part.TAG} block's symmetric buffers not at offset 0: {bad[:4]}")
        print(f"[{self.TAG}] symmetric buffers: every block's ({', '.join(p.TAG for p in [A] + self.mlps)}) at offset 0")

    def _place_l3_arena(self):
        """params layer.l3_arena: ARENA_ATTN past the attention's activations in dl_act, the router's weights past the
        MoE's X and ring."""
        al, asked = self.st.aliases, self.st.asked
        up = lambda n: -(-n // ARENA_ALIGN) * ARENA_ALIGN
        ends = lambda names: max(al[n][1] + asked[n] for n in names)
        rw = [n for n in al if n.endswith("_rwland") and n in asked]
        fixed = [n for n, (r, _) in al.items() if r == "dl_act" and n in asked and n not in ARENA_ATTN and n not in rw]
        at = up(ends([n for n in fixed if n.startswith("fp_")]))
        for n in ARENA_ATTN:
            al[n] = ("dl_act", at)
            at += up(asked[n])
        moe_end = up(ends([n for n in fixed if not n.startswith("fp_")]))
        for n in rw:
            al[n] = ("dl_act", moe_end)
        print(f"[{self.TAG}] l3_arena: dl_act {max(at, moe_end + max(asked[n] for n in rw)):,} B; attention to {at:,} B, "
              f"the MoE's X and ring to {moe_end:,} B, then its router weights")

    def _check_l3_arena(self):
        """params layer.l3_arena: every task that reads or writes a region of dl_act for the attention follows, on a
        path of dependency and order edges, before every write of the MoE into an overlapping part: the host tasks that
        fill the MoE's ring, X and router weights. Each writer is checked against each attention task on any chip, by
        address span. The attention's own pushes end before its last read of them: each ring chunk has a load."""
        import networkx as nx
        G = self.dfg.bingo_order_view()
        st, A = self.st, self.attn
        span = {n: (o, o + st.asked[n]) for n, (r, o) in st.aliases.items() if r == "dl_act" and n in st.asked}
        mlp = tuple(f"{m.TAG}_" for m in self.mlps)
        users = {}
        for t in self.dfg.node_list:
            names = arg_symbols(getattr(t, "kernel_args", None)) & set(span)
            if names:
                users[t] = names
        attn = {t: {n for n in v if n.startswith("fp_")} for t, v in users.items()
                if any(n.startswith("fp_") for n in v) and t.kernel_name not in ATTN_PUSHES}
        # the MoE's writes: its host tasks, and any device task with a name of its as a destination
        writers = [t for t, v in users.items() if any(n.startswith(mlp) for n in v) and
                   (t.kernel_name.startswith("__host") or
                    any(n.startswith(mlp) for n in arg_symbols(t.kernel_args, "dst") & set(span)))]
        kinds = {}
        for t in writers:
            kinds[t.kernel_name] = kinds.get(t.kernel_name, 0) + 1
        over = lambda a, b: span[a][0] < span[b][1] and span[b][0] < span[a][1]
        pairs, bad = 0, []
        for w in writers:
            anc = nx.ancestors(G, w)
            hit = {a for a in span if a.startswith("fp_") and any(over(a, b) for b in users[w] if b.startswith(mlp))}
            for t, v in attn.items():
                if v & hit:
                    pairs += 1
                    if t not in anc:
                        bad.append((t.node_name, w.node_name))
        if bad:
            raise ValueError(f"l3_arena: {len(bad)} attention tasks not ordered before a MoE write over their L3, "
                             f"e.g. {bad[:4]}")
        for k, rings in A.rings.items():
            need = {(r, i) for r, e in enumerate(rings.entries) for i in range(len(e))}
            have = {tuple(t.exec_chunk[:2]) for t, v in attn.items() if t.assigned_chiplet_id == k and "fp_wring_slots" in v
                    and getattr(t, "exec_chunk", None) is not None}
            if need - have:
                raise ValueError(f"l3_arena: chip 0x{k:02x}: {len(need - have)} attention ring chunks without a load")
        print(f"[{self.TAG}] l3_arena: {len(attn)} attention tasks before {len(writers)} MoE writers ({pairs} ordered pairs "
              f"over overlapping L3), every ring chunk loaded; the writers: {kinds}")

    def prune_post_edges(self):
        """The keys after the window (a proof of concept's later chunks) follow the layer's end stamp: a dependency on
        them from another core's task that the stamp already follows is implied (every one of them is behind the stamp,
        directly or through another of them), and it would hold a dep tag across the whole MoE / FFN. Such edges go."""
        post = set(self.attn.post_tasks)
        if not post:
            return
        import networkx as nx
        before = nx.ancestors(self.dfg, self.end)
        core = lambda t: (t.assigned_chiplet_id, t.assigned_cluster_id, t.assigned_core_id)
        rm = [(u, v) for v in post for u in list(self.dfg.predecessors(v))
              if u not in post and u in before and core(u) != core(v)]
        self.dfg.remove_edges_from(rm)
        print(f"[{self.TAG}] keys after the window: {len(rm)} dependencies the end stamp implies removed")


class _LayerBuild(LayerBuild):
    """LayerBuild with the attention's last node id recorded (the reuse passes' scope)."""

    def build(self):
        self.attn.build()
        self.attn_last_id = max(t.node_id for t in self.dfg.node_list)
        relay = dict(getattr(self.dfg, "relay_prefer_cores", None) or {})
        self.glue()
        for m in self.mlps:
            m.build()
        self.close_checks()
        # the dep-tag relays' cores are one choice for the whole graph: the attention's (params attn.relay_prefer), whose
        # key loop shares cells with its sweep's SIMD core; the FFN's relay_on_simd would put the attention's relays there
        if relay and getattr(self.dfg, "relay_prefer_cores", None) != relay:
            print(f"[{self.TAG}] dep-tag relays: the attention's core preference for the whole layer")
            self.dfg.relay_prefer_cores = relay


def _app_name(S):
    return (f"DeepSeek-V2-Lite layer {S.which} prefill, attention {S.attn.T} tokens ({S.NB} of {S.attn.NCH} chunks), "
            f"{'MoE' if S.which == 1 else 'dense FFN'} {S.T_mlp} tokens, {len(S.chips)} chips x {S.NC} clusters")


def run():
    run_driver("dsv2 whole-layer prefill workload generator (final block mappings)", _LayerBuild, _app_name)
