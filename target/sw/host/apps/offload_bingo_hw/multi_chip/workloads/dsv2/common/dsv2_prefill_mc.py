# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""The driver of the whole-layer prefill workloads (workloads/dsv2/layer1/<app>/): DeepSeek-V2-Lite's
layer 1 over a prompt of T tokens on the 16 + 4 chiplet platform (hemaia_16c4m_8MBL3_2cluster), every step on the
device, built from the two blocks of workloads/fa and workloads/moe in ONE graph:

    x1 -> RMSNorm (input_layernorm) -> MLA attention -> + x1 = h -> RMSNorm (post_attention_layernorm) -> MoE -> + h

THE INPUT is real: the hidden states x1 of T positions of the 8K-token sequence the attention's cache holds
(fa_prefill_data.py), FP16 in the root's L3: the last T, which attend to the whole cache, or (params attn.fresh) the
first T, a fresh prompt: no cache, causal, the keys made on the device from the input norm's tiles (each chip's tiles
in its cluster 0's L1 go on to the attention's stage_kv, so params layer.tiles chips).

THE STEPS, in the order they run:
    input norm   16 rows at a time (params layer.tiles: on the root's clusters or the chips' tiles): x1's rows into
                 L1, RMSNorm (simd_rmsnorm, row major), times the input_layernorm gain (simd_mul_tiles, a row a
                 tile), reshaped to the attention's A layout (xDMA), quantised (simd_fp16_to_int8) into the
                 attention's x source -- what the attention block's data otherwise stages from the host (params
                 layer.in_norm host: that)
    attention    fa_prefill_mc.py's block (params attn): head parallel, the token chunks one after the other, the
                 partials folded into the root, which stores the raw y in its L3
    glue         16 rows at a time, where the input norm's tiles run: y's rows into L1, reshaped D -> row major (xDMA),
                 dequantised (simd_mul_tiles), plus x1 (the residual: h), h stored in the MoE output's slice layout
                 (for the last residual), RMSNorm, times the post_attention_layernorm gain, quantised into the MoE's
                 X source -- what the MoE block's data otherwise stages
    MoE          moe_prefill_mc.py's block (params moe), on that X; one chip (params layer.resid_chip) adds h into its
                 partial of every output slice, so the in-transit reduce returns MoE(h) + h: the layer's output

THE GOLDEN is the device's arithmetic end to end: the attention block's y (its fold order), the glue's operators
(hwmodel: mul16, add16, rmsnorm, quant_i8), the MoE block's golden on the glue's X, with h in the resid chip's
partial. The MoE's input scale is calibrated on the prompt's own h (as the attention's are on the prompt).

L1. A task's args struct lives in its cluster's L1 for the whole run, so a layer would hold both blocks' args at
once; the layer splits its tasks into ARGS PHASES (bingo_dfg_emit.py): the input norm and the attention's first
chunk, each further chunk with the glue after the last, the MoE. A cluster's args of one phase come into L1 by one
DM task (PhaseArgsLoadArgs) after every task of the previous phase on the cluster, before every task of its own;
the static L1 plan overlays the phases' buffers, which the same order makes disjoint in time. Both blocks share
one symmetric buffer (each block's partials).

L3. The two blocks' weight rings are one region (the MoE's prefetchers start after the attention's are done: the
MoE starts after the glue), aliased in the data header.

PARAMETERS (params.hjson): the platform and the graph switches every block shares (num_chiplets, num_clusters,
tokens, compact_id_maps, gid_lists, per_chip_image, static_l1, l1_capacity, d2d_ddr, relay_per_gather), then
`attn` (fa_prefill_mc.py's params), `moe` (moe_prefill_mc.py's) and `layer`: in_norm (device | host),
resid_chip (hex), resid_piece (bytes the residual comes in at a time), glue_rows (rows a glue or input-norm tile,
16), x1 (l3 | hbm: where x1 is staged), tiles (root | chips: the input norm's and the glue's tiles on the root's two
clusters, or tile q on chip q mod 16, each chip reading x1 from its own copy and putting its rows and a landed flag into
the root's L3), moe_prefetch (start | early:
a chip's MoE ring fills from the MoE's start, or from the chip's last attention weight load), stop (full | glue: the
MoE left out, the glue's outputs checked), l3_arena (explained where it is read).

A LONGER PROMPT (T = 256 n): each chip runs n input-norm and glue tiles, 16 apart (tile q on chip q mod 16), and with
attn.fresh makes the keys of those n tiles; the attention takes T / 128 chunks and the MoE routes n 8-token tiles a
cluster. What else scales with T is set in the params (each explained where it is read): L3 with layer.x1 hbm and
layer.l3_arena (the activations' 13 T KiB in 8 T KiB), the dependency tags with attn.o_land, L1 with output slices of
32 K / T columns (moe.slice: a slice of every row stays 64 KiB), moe.expert_rows (the busiest experts' gate|up in row
blocks) and layer.resid_piece, and the descriptor lists with desc_in_image (chiplet_mc.py).
"""

import os
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

import _bingo_paths  # noqa: F401,E402
from bingo_kernel_args import (  # noqa: E402
    PhaseArgsLoadArgs, SnaxBingoKernelIdma1dCopyArgs, SnaxBingoKernelIdma2dCopyArgs, SnaxBingoKernelIdmaPutFlaggedArgs,
    SnaxBingoKernelWaitFlagsArgs,
    SnaxBingoKernelSimdAddF16Args, SnaxBingoKernelSimdFp16ToInt8Args, SnaxBingoKernelSimdMulTilesArgs,
    SnaxBingoKernelSimdRmsnormArgs, SnaxBingoKernelXdma6dArgs, simd_beats)
from bingo_mem_handle import BingoMemFixedAddr, BingoMemSymbol  # noqa: E402
from libs.comm.nest import convert_args  # noqa: E402

import chiplet_mc as cm  # noqa: E402
import fa_prefill_mc as fam  # noqa: E402
import moe_prefill_data as mpd  # noqa: E402
import moe_prefill_mc as mom  # noqa: E402

HIDDEN = 2048
MESH3 = (8, 8, 16)            # the MoE's array shape: its output's D layout
ATOM, LANES = 8, 8            # the xDMA: bytes a lane moves a beat, lanes


# ---- the xDMA nests the glue needs ------------------------------------------------------------------------
def xdma_nest(src, dst, axes, expect):
    """One xDMA pass over `axes` [(bound, src stride, dst stride)] in bytes, the innermost one the 8-byte atoms of
    a run (its strides 8, its bound the run's atoms); one axis whose bound is a multiple of 8 takes the lanes.
    `expect` (src element offset -> dst element offset, FP16 elements) is the mapping it must make: the nest is
    walked here and compared with it element by element."""
    atom, rest = axes[0], list(axes[1:])
    if atom[1] != ATOM or atom[2] != ATOM:
        raise ValueError(f"the innermost axis {atom} is not the 8-byte atoms of a run")
    lane = next(i for i, a in enumerate(rest) if a[0] % LANES == 0)
    b, ss, ds = rest.pop(lane)
    temporal = [atom] + rest + ([(b // LANES, ss * LANES, ds * LANES)] if b > LANES else [])
    if len(temporal) > 5:
        raise ValueError(f"{len(temporal)} temporal dimensions: the xDMA has 5")
    args = SnaxBingoKernelXdma6dArgs(
        src, dst, spatial_stride_src=ss, spatial_stride_dst=ds,
        temporal_strides_src=[t[1] for t in temporal], temporal_bounds_src=[t[0] for t in temporal],
        temporal_strides_dst=[t[2] for t in temporal], temporal_bounds_dst=[t[0] for t in temporal])
    # walk it: every lane of every temporal point moves 4 FP16 elements
    grids = np.meshgrid(*[np.arange(t[0]) for t in temporal], np.arange(LANES), indexing="ij")
    so = sum(g * t[1] for g, t in zip(grids, temporal)) + grids[-1] * ss
    do = sum(g * t[2] for g, t in zip(grids, temporal)) + grids[-1] * ds
    so, do = so.reshape(-1) // 2, do.reshape(-1) // 2
    so = (so[:, None] + np.arange(4)).reshape(-1)
    do = (do[:, None] + np.arange(4)).reshape(-1)
    if so.size != expect.size or not np.array_equal(np.sort(so), np.arange(expect.size)) \
            or not np.array_equal(expect[so], do):
        raise ValueError("the xDMA nest does not make the mapping it is meant to")
    return args


def fa_d_to_rows(src, dst, rows):
    """`rows` rows of the attention's y, D layout of its (16, 4, 16) array in two halves of 1024 columns
    ([half][column tile][16 rows][16 columns], the halves at 32 KiB a 16-row tile), into row major [rows, 2048]."""
    if rows != 16:
        raise ValueError("a glue tile is one 16-row tile of the attention's D layout")
    h, n, r, cc = np.meshgrid(np.arange(2), np.arange(64), np.arange(16), np.arange(16), indexing="ij")
    src_el = ((h * 64 + n) * 16 + r) * 16 + cc
    dst_el = r * HIDDEN + 1024 * h + 16 * n + cc
    expect = np.empty(src_el.size, dtype=np.int64)
    expect[src_el.reshape(-1)] = dst_el.reshape(-1)
    return xdma_nest(src, dst, [(4, 8, 8), (16, 32, 2 * HIDDEN), (128, 512, 32)], expect)


def rows_to_slices(src, dst, rows, sw):
    """`rows` row-major FP16 rows [rows, 2048] into the MoE output's slice layout, slice-major: per slice of `sw`
    columns the rows' row tiles of 8, each D layout of (8, 8, 16) ([slice][row tile][column tile][8][16])."""
    ns, nt, nm = HIDDEN // sw, sw // 16, rows // 8
    s, m, n, rr, cc = np.meshgrid(np.arange(ns), np.arange(nm), np.arange(nt), np.arange(8), np.arange(16),
                                  indexing="ij")
    src_el = (8 * m + rr) * HIDDEN + sw * s + 16 * n + cc
    dst_el = (((s * nm + m) * nt + n) * 8 + rr) * 16 + cc
    expect = np.empty(src_el.size, dtype=np.int64)
    expect[src_el.reshape(-1)] = dst_el.reshape(-1)
    return xdma_nest(src, dst, [(4, 8, 8), (8, 2 * HIDDEN, 32), (nm, 16 * HIDDEN, nt * 256), (nt, 32, 256),
                                (ns, 2 * sw, nm * nt * 256)], expect)


# ---- staging with aliases ------------------------------------------------------------------------------------
class AliasStaging:
    """The layer's DataStaging, with names that are one region: put_zeros of an aliased name stages nothing and
    returns its symbol; finish() stages the region at the largest extent asked and lines() are the defines that make
    each name the region. An alias is a region's name, or (region, offset): the name that many bytes into it."""

    def __init__(self, st, aliases):
        self._st, self.asked = st, {}
        self.aliases = {n: (a if isinstance(a, tuple) else (a, 0)) for n, a in aliases.items()}

    def __getattr__(self, name):
        return getattr(self._st, name)

    def put_zeros(self, name, ctype, count, mem_chip=None, preload=False):
        if name not in self.aliases:
            return self._st.put_zeros(name, ctype, count, mem_chip=mem_chip, preload=preload)
        width = {"uint8_t": 1, "int8_t": 1, "uint16_t": 2, "int16_t": 2, "uint32_t": 4, "int32_t": 4}[ctype]
        self.asked[name] = count * width
        return BingoMemSymbol(name)

    def finish(self):
        for region in sorted({r for r, _ in self.aliases.values()}):
            ext = [o + self.asked[n] for n, (r, o) in self.aliases.items() if r == region and n in self.asked]
            if ext:
                self._st.put_zeros(region, "uint8_t", max(ext), preload=True)

    def lines(self):
        out = ["// one L3 region, several names (dsv2_prefill_mc.py AliasStaging)"]
        out += [f"#define {n} {r}" if not o else f"#define {n} ((uint8_t *){r} + {o})"
                for n, (r, o) in sorted(self.aliases.items()) if n in self.asked]
        return out


# ---- the blocks, over the layer's graph ----------------------------------------------------------------------
class AttnPart(fam.FaBuild):
    """The attention block (fa_prefill_mc.py) in the layer: its window is the layer's, its chunks the layer's args
    phases, its checks the layer's."""
    stream_name = "zs_fp_stream"     # sorts after the block's buffers, as "stream" does alone (the packer's order)

    def __init__(self, layer, args, p, plat, hw):
        self.layer = layer
        super().__init__(args, p, plat, hw, shared=layer)

    def setup(self, extra=None):
        return self.layer.open_window(extra)

    def chunk_start(self, i):
        # params layer.chunk_phases (default true): false keeps the attention's chunks in one args phase
        if self.layer.chunk_phases:
            self.layer.phase_boundary(f"attention chunk {i}")

    def end_window(self, after):
        self.layer.attn_ends = list(after)

    def run_checks(self, checks):
        self.layer.part_checks += list(checks)

    def stash(self, *a, **kw):
        raise ValueError("the attention's stage checks need the end stamp: params attn.checks y, stop full")


class MoePart(mom.MBuild):
    """The MoE block (moe_prefill_mc.py) in the layer: its data the glue's rows, its window opened by the layer, its
    end the layer's."""
    stream_name = "zs_mp_stream"

    def __init__(self, layer, args, p, plat, hw):
        self.layer = layer
        self.resid = layer.resid
        super().__init__(args, p, plat, hw, shared=layer)

    def _data(self, args, p, hw):
        cache = os.path.join(REPO, p.get("weights", "target/sim/automation/sweep/common/weights/cache"))
        return mpd.MoePrefillData(cache, hw=hw, n_calib=int(p.get("n_calib", 8)),
                                  n_cal_act=int(p.get("n_cal_act", 32)), tokens=self.T, inter_e=p.get("inter_e"),
                                  inter_sh=p.get("inter_sh"), rows=self.layer.moe_rows)

    def setup(self, extra=None):
        return self.layer.moe_window(extra)

    def args_phase(self, name):
        self.layer.phase_boundary(name)

    def end_window(self, after):
        self.layer.end_window(after)
        self.end, self.check_copies = self.layer.end, self.layer.check_copies

    def run_checks(self, checks):
        cm.ChipletBuild.run_checks(self.layer, self.layer.part_checks + list(checks))


# ---- the layer -------------------------------------------------------------------------------------------------
class LayerBuild(cm.ChipletBuild):
    TAG = "dl"

    def __init__(self, args, p, plat, hw):
        common = {k: v for k, v in p.items() if k not in ("attn", "moe", "layer")}
        lp = self.lp = dict(p.get("layer", {}))
        super().__init__(args, common, plat, hw)
        aliases = {"fp_wring_slots": "dl_wring", "mp_wring_slots": "dl_wring"}
        # params layer.l3_arena (default false): the prompt's activations in one L3 region, dl_act, each where the
        # phases that use it leave it free (XS = T x 2048 bytes): the attention's x source and landing (the root's y
        # over them, attn.l3_overlay) at 0 and XS, its keys at 2 XS; h (the glue's, read by the resid chip's adds)
        # over the first two, on the resid chip only; the MoE's X (the root's source, every chip's landing, the root's
        # Y over both, moe.l3_overlay) from 2 XS, its source over the keys and the attention's o8 area (attn.o_land)
        # after them. 4 XS of L3 instead of 6.5 (and the o8 area's)
        self.l3_arena = bool(lp.get("l3_arena", False))
        if self.l3_arena:
            XS, KS = self.T * HIDDEN, -(-(self.T // 16) * fam.KVP // 64) * 64
            aliases.update({"fp_xsrc": ("dl_act", 0), "fp_xland": ("dl_act", XS), "fp_kv": ("dl_act", 2 * XS),
                            "fp_oland": ("dl_act", 2 * XS + KS), "dl_h3": ("dl_act", 0), "mp_xy": ("dl_act", 2 * XS)})
        self.st = AliasStaging(self.st, aliases)
        self.sym_zero = False
        self.in_norm = str(lp.get("in_norm", "device"))
        if self.in_norm not in ("device", "host"):
            raise ValueError(f"params layer.in_norm={self.in_norm!r}: device or host")
        self.stop = str(lp.get("stop", "full"))
        if self.stop not in ("full", "glue"):
            raise ValueError(f"params layer.stop={self.stop!r}: full or glue")
        self.R = int(lp.get("glue_rows", 16))
        # params layer.tiles (root | chips): where the input norm's and the glue's row tiles run (_owner)
        self.tiles = str(lp.get("tiles", "root"))
        # params layer.chunk_phases (default true): an args phase a chunk of the attention; false: one for all of them
        self.chunk_phases = bool(lp.get("chunk_phases", True))
        if self.tiles not in ("root", "chips"):
            raise ValueError(f"params layer.tiles={self.tiles!r}: root or chips")
        # params layer.moe_prefetch (start | early): each chip's MoE weight ring fills from the MoE's start, or from the
        # chip's last attention weight load (the two blocks' rings are one L3 region)
        self.moe_prefetch = str(lp.get("moe_prefetch", "start"))
        if self.moe_prefetch not in ("start", "early"):
            raise ValueError(f"params layer.moe_prefetch={self.moe_prefetch!r}: start or early")
        if self.T % (16 * self.NC * 2):
            raise ValueError(f"{self.T} tokens: whole 16-row tiles on each root cluster")
        # the args phases: each a run of the tasks made one after the other (node ids from phase_starts[p - 1])
        self.phase_starts, self.phase_names, self.loaders = [], ["input norm, attention chunk 0"], {}
        self.part_checks = []

        # ---- the attention, and the glue's golden --------------------------------------------------------
        pa = dict(common, **p["attn"])
        pa["x_device"] = self.in_norm == "device"
        if int(pa.get("tokens", self.T)) != self.T:
            raise ValueError("params attn.tokens: the layer's tokens")
        A = self.attn = AttnPart(self, args, pa, plat, hw)
        d = A.d
        fp = d.fp
        self.x1 = np.asarray(d.x1, dtype=np.float16)                                # [T, 2048]
        self.oscale = np.asarray(d.out_scale(), dtype=np.float16)
        self.in_gain = np.asarray(d.in_norm, dtype=np.float16)
        y_raw = np.concatenate([A.y_raw(i * A.C + c * A.ROWS, A.ROWS) for i in range(A.NCH) for c in range(A.NC)])
        self.h16 = fp.add16(fp.mul16(y_raw, self.oscale[None, :]), self.x1)        # the residual stream
        n_cal_act = int(p["moe"].get("n_cal_act", 32))
        self.moe_rows = dict(x16=self.h16, cal=self.h16, cal_act=self.h16[-n_cal_act:])
        self.resid_chip = int(str(lp.get("resid_chip", "33")), 16)
        if self.resid_chip not in self.chips:
            raise ValueError(f"params layer.resid_chip={lp.get('resid_chip')}: not a compute chip")
        # h in the output's slice layout lives where the residual is added (tiles chips: every chip writes its rows
        # there) or on the root (tiles root: where the glue runs)
        self.h3_chip = self.resid_chip if self.tiles == "chips" else self.ROOT
        if self.l3_arena and (self.h3_chip == self.ROOT or self.stop != "full" or not pa.get("l3_overlay")
                              or not p["moe"].get("l3_overlay")):
            raise ValueError("params layer.l3_arena: tiles chips with the resid chip not the root, stop full, and "
                             "l3_overlay in attn and moe")
        if self.l3_arena and A.fresh:
            # the root's keys lie under the MoE's X source by the end: the root's key checks read another chip's copy
            far = A.order[len(A.order) // 2]
            A.kv_checks_at = next(k for k in A.order if k not in (self.ROOT, self.resid_chip, far))
        self.resid = (dict(chip=self.resid_chip,
                           src=BingoMemSymbol("dl_h3", chip_id=None if self.h3_chip == self.resid_chip else self.h3_chip),
                           piece=int(lp.get("resid_piece", 65536)),
                           h16=self.h16) if self.stop == "full" else None)

        # ---- the MoE ----------------------------------------------------------------------------------------
        pm = dict(common, **p["moe"])
        pm["x_device"] = True
        self.moe = MoePart(self, args, pm, plat, hw) if self.stop == "full" else None
        md = self.moe.d if self.moe else None
        self.post_gain = np.asarray(md.gain16 if md else self._post_gain(args, pm), dtype=np.float16)
        self.s_h = md.s_h if md else None
        if self.moe is None:
            amax = float(np.max(np.abs(fp.mul16(d.hm.rmsnorm(self.h16), self.post_gain[None, :]).astype(np.float32))))
            self.s_h = d.hm.QScale.for_amax(amax)
        self.x8 = fp.quant_i8(fp.mul16(d.hm.rmsnorm(self.h16), self.post_gain[None, :]), self.s_h.inv)
        if md is not None and not np.array_equal(self.x8, md.hq):
            raise AssertionError("the glue's X and the MoE block's golden X differ")
        self.SW = int(pm.get("slice", 128))

        # ---- one symmetric buffer for both blocks ------------------------------------------------------------
        sizes = [A.sym_bytes] + ([self.moe.NOUT * self.moe.SL] if self.moe else [])
        self.sym_bytes = max(sizes)
        self.sym = {(k, c): self.cx_chip(k).at(c).l1(f"AAAA_{self.TAG}_sym", self.sym_bytes)
                    for k in self.chips for c in range(self.NC)}
        A.sym = self.sym
        if self.moe:
            self.moe.sym = self.sym

        # ---- the layer's own data, in the root's L3 ------------------------------------------------------------
        st, T = self.st, self.TAG
        # params layer.x1 (l3 | hbm): x1 in every chip's L3 image, or once in each memory chip's HBM, where each chip
        # reads its rows from its own memory chip (_x1)
        self.x1_at = str(lp.get("x1", "l3"))
        if self.x1_at not in ("l3", "hbm"):
            raise ValueError(f"params layer.x1={self.x1_at!r}: l3 or hbm")
        if self.x1_at == "l3":
            st.put(f"{T}_x1", "uint16_t", self.x1.view(np.uint16).reshape(-1))
        else:
            self.h_x1 = {m: st.put_hbm(f"{T}_x1_m{m:02x}", self.x1.view(np.uint8).reshape(-1), mem_chip=(m >> 4, m & 15))
                         for m in sorted(set(self.mem.values()))}
        st.put(f"{T}_c", "uint16_t", np.concatenate([self.oscale, self.post_gain, self.in_gain]).view(np.uint16))
        st.put_zeros(f"{T}_h3", "uint8_t", self.T * HIDDEN * 2, preload=True)
        if self.tiles == "chips":                           # each tile's landed flag in the root's L3, per step
            st.put_zeros(f"{T}_nflag", "uint8_t", 64 * (self.T // self.R), preload=True)
            st.put_zeros(f"{T}_gflag", "uint8_t", 64 * (self.T // self.R), preload=True)
        if self.moe is None:
            st.put_zeros(f"{T}_x8", "int8_t", self.T * HIDDEN, preload=True)
        print(f"[{T}] layer: {self.T} tokens; input norm on the {self.in_norm}; residual on chip "
              f"{self.resid_chip:02x}; stop {self.stop}; |h| max {float(np.max(np.abs(self.h16))):.3g}, "
              f"rms {float(np.sqrt(np.mean(self.h16.astype(np.float32) ** 2))):.3g}")

    def _post_gain(self, args, pm):
        cache = os.path.join(REPO, pm.get("weights", "target/sim/automation/sweep/common/weights/cache"))
        import json
        from model_data import read_bf16
        with open(os.path.join(cache, "manifest.json")) as f:
            m = json.load(f)["dsv2-lite-bf16"]
        name = "model.layers.1.post_attention_layernorm.weight"
        return read_bf16(os.path.join(cache, m["repo"].replace("/", "__"), name + ".bin"), m["tensors"][name]["shape"])

    # ---- the args phases ---------------------------------------------------------------------------------------
    def _dev(self, n):
        return (n.kernel_name or "").startswith("__snax")

    def phase_boundary(self, name):
        """The current args phase ends on every cluster: each cluster's DM core gets the task that loads the next
        phase's args, after every task of the phase on the chip (a block's task may write the other cluster's L1, so a
        chip's clusters leave a phase together): after each of them that no other one of them follows. The tasks made
        from here on are of the next phase."""
        lo = self.phase_starts[-1] if self.phase_starts else 0
        mine = {}
        for n in self.dfg.node_list:
            if n.node_id >= lo and self._dev(n) and not getattr(n, "args_resident", False):
                mine.setdefault(n.assigned_chiplet_id, set()).add(n)
        p = len(self.phase_starts) + 1
        for k in self.chips:
            ts = mine.get(k, set())
            deps = sorted((t for t in ts if not any(v in ts for v in self.dfg.successors(t))), key=lambda t: t.node_id)
            for c in range(self.NC):
                ld = self.node(k, c, self.roles["dm"], f"args_p{p}", "__snax_bingo_kernel_idma_1d_copy",
                               PhaseArgsLoadArgs(k, c, p), deps)
                ld.args_resident = True
                self.loaders[(p, k, c)] = ld
        self.phase_starts.append(max(n.node_id for n in self.dfg.node_list) + 1)
        self.phase_names.append(name)

    def _assign_phases(self):
        """Every device task its args phase; each task of a phase that no task of the phase on its cluster precedes
        waits for its cluster's load."""
        starts = self.phase_starts
        phase_of = {}
        for n in self.dfg.node_list:
            if not self._dev(n) or getattr(n, "args_resident", False):
                continue
            phase_of[n] = n.args_phase = sum(1 for s in starts if n.node_id >= s)
        cl = lambda t: (t.assigned_chiplet_id, t.assigned_cluster_id)
        added = 0
        for n, p in phase_of.items():
            if not p or any(phase_of.get(u) == p and cl(u) == cl(n) for u in self.dfg.predecessors(n)):
                continue
            ld = self.loaders[(p, *cl(n))]
            if not self.dfg.has_edge(ld, n):
                self.edge(ld, n)
                added += 1
        counts = {}
        for n in self.dfg.node_list:
            if self._dev(n) and n.assigned_chiplet_id == self.ROOT and n.assigned_cluster_id == 0:
                counts[getattr(n, "args_phase", "r")] = counts.get(getattr(n, "args_phase", "r"), 0) + 1
        print(f"[{self.TAG}] args phases: " + "; ".join(f"{p} {nm}" for p, nm in enumerate(self.phase_names))
              + f"; root cluster 0 tasks a phase {dict(sorted(counts.items(), key=str))}; {added} load edges")

    # ---- the window ---------------------------------------------------------------------------------------------
    def open_window(self, extra):
        """The start stamp (after every chip's setup), the attention's prefetchers behind it, and the input norm."""
        ends = [t for k in self.chips for t in (extra or {}).get(k, [])]
        start = self.stamp("start", ends)
        go = {k: start for k in self.chips}
        for k in self.chips:
            self.edge(start, self.attn.prefetch[k])
        self.go = go
        if self.in_norm == "device":
            self.attn.x_ready = self.input_norm(start)
        return go

    def moe_window(self, extra):
        """Where the MoE starts on each chip: its args loads (the root's both, in one probe); the MoE's setup tasks
        and prefetchers behind them."""
        go = dict(self.moe_go)
        A = self.attn
        for k in self.chips:
            for t in (extra or {}).get(k, []):
                self.edge(go[k], t)
            if self.moe_prefetch == "early":
                for g in self.gs(k):
                    self.edge(A.last_ld[g], self.moe.prefetch[k])
            else:
                self.edge(go[k], self.moe.prefetch[k])
        return go

    # ---- the row tiles: input norm and glue ------------------------------------------------------------------
    def _owner(self, q):
        """(chip, cluster) that runs 16-row tile q of the prompt: params layer.tiles root (the root's clusters; cluster c
        the tiles of the rows the attention gives it, 4c .. 4c + 3 of each chunk) or chips (tile q on chip q mod 16 in
        the platform's order, its cluster 0; with attn.kv_clusters the cluster that makes the tile's keys)."""
        if self.tiles == "root":
            return self.ROOT, (q % 8) // 4
        k = self.chips[q % len(self.chips)]
        return k, (self.attn.kv_cluster(k, q) if self.attn.fresh else 0)

    def _per_owner(self):
        out = {}
        for q in range(self.T // self.R):
            out.setdefault(self._owner(q), []).append(q)
        return out

    def _x1(self, k, off):
        """x1 at byte `off` as chip k reads it: its own L3 copy, or its memory chip's HBM copy (params layer.x1)."""
        if self.x1_at == "l3":
            return BingoMemSymbol(f"{self.TAG}_x1", offset=off)
        return BingoMemFixedAddr(self.h_x1[self.mem[k]].address + off, mem_level="HBM")

    def _on(self, sym, k, chip):
        """`sym` (a staged symbol) as chip k names chip `chip`'s copy of it."""
        return sym if k == chip else BingoMemSymbol(sym.symbol_name, offset=sym.offset, chip_id=chip)

    def _tile_bufs(self, k, c, kind, slot, x8=True):
        """A tile slot's buffers on chip k cluster c; x8 false: without the quantised rows (each tile has its own)."""
        Tg = self.TAG
        # named to sort after the blocks' buffers: the static packer places buffers in name order, and these live in
        # the gaps the blocks' long-lived ones leave (the slabs' region before the attention, its working set after)
        l1 = lambda name, nb: self.cx_chip(k).at(c).l1(f"zt_{Tg}_{kind}{slot}_{name}", nb)
        rows = self.R
        b = dict(a=l1("a", rows * HIDDEN * 2), b=l1("b", rows * HIDDEN * 2))
        if x8:
            b["x8"] = l1("x8", rows * HIDDEN)
        rn = SnaxBingoKernelSimdRmsnormArgs(0, 0, rows, HIDDEN)
        b["scr"] = l1("scr", rn.scratch_bytes())
        if kind == "g":
            b["d"] = l1("d", rows * HIDDEN * 2)
        return b

    def _consts(self, k, c, kind, deps):
        """The gains and the output scale into chip k cluster c's L1, from the chip's own copy."""
        Tg = self.TAG
        buf = self.cx_chip(k).at(c).l1(f"zt_{Tg}_{kind}c", 3 * HIDDEN * 2)
        ld = self.node(k, c, self.roles["dm"], f"{kind}_c_ld", "__snax_bingo_kernel_idma_1d_copy",
                       SnaxBingoKernelIdma1dCopyArgs(BingoMemSymbol(f"{Tg}_c"), buf, 3 * HIDDEN * 2), list(deps))
        return buf, ld

    def _norm_rows(self, k, c, name, src, dst8, gain, inv_bits, bufs, deps, to_a=None):
        """rows of `src` (row-major FP16 in buffer bufs['b']) normalised, times `gain`, quantised (to_a: reshaped to
        the attention's A layout first) into dst8. Returns the quantisation."""
        simd, xdma = self.roles["simd"], self.roles["xdma"]
        rows = self.R
        rn = self.node(k, c, simd, f"{name}_rms", "__snax_bingo_kernel_simd_rmsnorm",
                       SnaxBingoKernelSimdRmsnormArgs(src, bufs["a"], rows, HIDDEN, scratch_addr=bufs["scr"]),
                       list(deps))
        gm = self.node(k, c, simd, f"{name}_gain", "__snax_bingo_kernel_simd_mul_tiles",
                       SnaxBingoKernelSimdMulTilesArgs(bufs["a"], gain, bufs["a"], rows, HIDDEN // 128,
                                                       2 * HIDDEN, 2 * HIDDEN), [rn])
        q_src, before = bufs["a"], [gm]
        if to_a is not None:
            cv = self.node(k, c, xdma, f"{name}_toa", "__snax_bingo_kernel_xdma_6d",
                           convert_args("row_major", "A", rows, HIDDEN, cm.MESH, 2, bufs["a"], to_a), [gm])
            q_src, before = to_a, [cv]
        return self.node(k, c, simd, f"{name}_q", "__snax_bingo_kernel_simd_fp16_to_int8",
                         SnaxBingoKernelSimdFp16ToInt8Args(q_src, dst8, beats=simd_beats(rows * HIDDEN * 2), rows=1,
                                                           inv_scale_f32bits=inv_bits), before)

    def input_norm(self, start):
        """The attention's x from x1, a 16-row tile at a time on the tiles' owners (_owner), into the root's x source:
        {chunk: [the stores of its tiles]}."""
        A, R, Tg, dm = self.attn, self.ROOT, self.TAG, self.roles["dm"]
        rows, fp = self.R, A.d.fp
        inv = fp.f32bits(A.d.s_x.inv)
        per_chunk = A.C // rows
        ready = {i: [] for i in range(A.NCH)}
        for (k, c), qs in self._per_owner().items():
            consts, cld = self._consts(k, c, "n", [start])
            # a fresh prompt's tiles stay in L1 as x8 until the chip's W_KVA GEMMs have read them (stage_kv, made after
            # these): with more than one tile a chip, each tile's x8 its own buffer instead of its slot's
            own_x8 = A.fresh and len(qs) > 1
            bufs = [self._tile_bufs(k, c, "n", s, x8=not own_x8) for s in range(min(2, len(qs)))]
            if own_x8:
                x8s = [self.cx_chip(k).at(c).l1(f"zt_{Tg}_nx{n}_x8", rows * HIDDEN) for n in range(len(qs))]
            free = [[] for _ in bufs]                      # each slot's last readers
            lds = {}

            def load(n):
                """tile n's rows into its slot: made before tile n - 1's store, so the in-order DM core loads a
                tile while the one before it is computed"""
                q, s = qs[n], n % len(bufs)
                lds[n] = self.node(k, c, dm, f"n_ld_q{q}", "__snax_bingo_kernel_idma_1d_copy",
                                   SnaxBingoKernelIdma1dCopyArgs(self._x1(k, q * rows * HIDDEN * 2),
                                                                 bufs[s]["b"], rows * HIDDEN * 2), [start] + free[s])

            load(0)
            for n, q in enumerate(qs):
                i = q // per_chunk
                b, s = bufs[n % len(bufs)], n % len(bufs)
                x8 = x8s[n] if own_x8 else b["x8"]
                ld = lds[n]
                qz = self._norm_rows(k, c, f"n_q{q}", b["b"], x8, consts.view(4 * HIDDEN), inv, b, [ld, cld],
                                     to_a=b["scr"])
                if n + 1 < len(qs):
                    load(n + 1)
                dst = self._on(BingoMemSymbol("fp_xsrc", offset=q * rows * HIDDEN), k, R)
                if A.owner_bcast:                           # the rows and the tile's flag into every chip's landing
                    stv = self.node(k, c, dm, f"n_put_q{q}", "__snax_bingo_kernel_idma_put_flagged",
                                    SnaxBingoKernelIdmaPutFlaggedArgs(
                                        x8, BingoMemSymbol("fp_xland", offset=q * rows * HIDDEN), rows * HIDDEN,
                                        BingoMemSymbol("fp_xtflag", offset=64 * q), value=1, bcast=True,
                                        also_here=True), [qz])
                elif self.tiles == "chips":                 # the rows, then the tile's flag in the root's L3
                    stv = self.node(k, c, dm, f"n_put_q{q}", "__snax_bingo_kernel_idma_put_flagged",
                                    SnaxBingoKernelIdmaPutFlaggedArgs(
                                        x8, dst, rows * HIDDEN,
                                        self._on(BingoMemSymbol(f"{Tg}_nflag", offset=64 * q), k, R), value=1), [qz])
                else:
                    stv = self.node(k, c, dm, f"n_st_q{q}", "__snax_bingo_kernel_idma_1d_copy",
                                    SnaxBingoKernelIdma1dCopyArgs(x8, dst, rows * HIDDEN), [qz])
                free[s] = [stv, qz]
                ready[i].append(stv)
                if A.fresh:                                 # the tile's keys are made from its x8, where it is
                    if self.tiles != "chips" or q not in A.kv_tiles[k] or c != A.kv_cluster(k, q):
                        raise ValueError("params attn.fresh: the input-norm tiles where their keys are made, tile q on "
                                         "chip q mod 16 (params layer.tiles chips)")
                    A.kv_src[q] = (x8, [qz])
        if A.owner_bcast:
            return ready                                    # the clusters' x waits count the tiles' own flags
        if self.tiles == "chips":
            # each chunk's tiles landed in the root's x source: one wait on their flags, on the DM core that multicasts
            # the chunk (the puts are ordering edges, as every flagged collective's)
            pc = A.x_put_cluster
            waits = {}
            for i, puts in ready.items():
                waits[i] = [self.node(R, pc, dm, f"n_wait_i{i}", "__snax_bingo_kernel_wait_flags",
                                      SnaxBingoKernelWaitFlagsArgs(BingoMemSymbol(f"{Tg}_nflag", offset=64 * i * per_chunk),
                                                                   len(puts), 1),
                                      [start] + [t for t in puts if A.flag_dep(t, R, pc)])]
                for t in puts:
                    if not A.flag_dep(t, R, pc):
                        self.dfg.bingo_add_order_edge(t, waits[i][0])
            return waits
        return ready

    def glue(self):
        """h = x1 + dequantised y and the MoE's X from it, a 16-row tile at a time on the tiles' owners (_owner); h
        stored in the MoE output's slice layout on chip h3_chip. Returns the stores."""
        A, R, Tg = self.attn, self.ROOT, self.TAG
        dm, simd, xdma = self.roles["dm"], self.roles["simd"], self.roles["xdma"]
        rows, fp = self.R, A.d.fp
        inv = fp.f32bits(self.s_h.inv)
        sl = self.T * self.SW * 2                         # one output slice, all rows
        x8sym = (lambda off: BingoMemSymbol(f"{Tg}_x8", offset=off)) if self.moe is None else self.moe.xsrc
        out = []
        for (k, c), qs in self._per_owner().items():
            consts, cld = self._consts(k, c, "g", self.attn_ends)
            bufs = [self._tile_bufs(k, c, "g", s) for s in range(min(2, len(qs)))]
            free = [[] for _ in bufs]
            lds = {}

            def load(n):
                """tile n's y and x1 rows into its slot, made before tile n - 1's stores (as the input norm's)"""
                q, s = qs[n], n % len(bufs)
                # the chunk, the attention cluster whose rows, the tile in them
                i, cy, tt = q // (A.C // rows), (q % (A.C // rows)) // (A.ROWS // rows), q % (A.ROWS // rows)
                b = bufs[s]
                # y's rows: the two 1024-column halves of this 16-row tile, 32 KiB each, 128 KiB apart
                lds[n] = (self.node(k, c, dm, f"g_ldy_q{q}", "__snax_bingo_kernel_idma_2d_copy",
                                    SnaxBingoKernelIdma2dCopyArgs(self._on(A.y_sym(i, cy, tt * rows * 1024 * 2), k, R),
                                                                  b["d"], rows * 1024 * 2, A.YB // 2, rows * 1024 * 2, 2),
                                    list(self.attn_ends) + free[s]),
                          self.node(k, c, dm, f"g_ldx_q{q}", "__snax_bingo_kernel_idma_1d_copy",
                                    SnaxBingoKernelIdma1dCopyArgs(self._x1(k, q * rows * HIDDEN * 2),
                                                                  b["b"], rows * HIDDEN * 2), list(self.attn_ends) + free[s]))

            load(0)
            for n, q in enumerate(qs):
                b, s = bufs[n % len(bufs)], n % len(bufs)
                ld_y, ld_x = lds[n]
                cv = self.node(k, c, xdma, f"g_rows_q{q}", "__snax_bingo_kernel_xdma_6d",
                               fa_d_to_rows(b["d"], b["a"], rows), [ld_y])
                dq = self.node(k, c, simd, f"g_deq_q{q}", "__snax_bingo_kernel_simd_mul_tiles",
                               SnaxBingoKernelSimdMulTilesArgs(b["a"], consts, b["a"], rows, HIDDEN // 128,
                                                               2 * HIDDEN, 2 * HIDDEN), [cv, cld])
                ad = self.node(k, c, simd, f"g_add_q{q}", "__snax_bingo_kernel_simd_stream_elementwise",
                               SnaxBingoKernelSimdAddF16Args(b["a"], b["b"], b["b"], rows * HIDDEN * 2 // 64, 32),
                               [dq, ld_x])
                h3 = self.node(k, c, xdma, f"g_h3_q{q}", "__snax_bingo_kernel_xdma_6d",
                               rows_to_slices(b["b"], b["d"], rows, self.SW), [ad, cv])
                if n + 1 < len(qs):
                    load(n + 1)
                sh = self.node(k, c, dm, f"g_sth_q{q}", "__snax_bingo_kernel_idma_2d_copy",
                               SnaxBingoKernelIdma2dCopyArgs(
                                   b["d"], self._on(BingoMemSymbol(f"{Tg}_h3", offset=q * rows * self.SW * 2), k,
                                                    self.h3_chip),
                                   rows * self.SW * 2, rows * self.SW * 2, sl, HIDDEN // self.SW), [h3])
                qz = self._norm_rows(k, c, f"g_q{q}", b["b"], b["x8"], consts.view(2 * HIDDEN), inv, b, [ad])
                if self.tiles == "chips":                   # X's rows, then the tile's flag in the root's L3 (after h)
                    sx = self.node(k, c, dm, f"g_putx_q{q}", "__snax_bingo_kernel_idma_put_flagged",
                                   SnaxBingoKernelIdmaPutFlaggedArgs(
                                       b["x8"], self._on(x8sym(q * rows * HIDDEN), k, R), rows * HIDDEN,
                                       self._on(BingoMemSymbol(f"{Tg}_gflag", offset=64 * q), k, R), value=1), [qz, sh])
                else:
                    sx = self.node(k, c, dm, f"g_stx_q{q}", "__snax_bingo_kernel_idma_1d_copy",
                                   SnaxBingoKernelIdma1dCopyArgs(b["x8"], self._on(x8sym(q * rows * HIDDEN), k, R),
                                                                 rows * HIDDEN), [qz])
                free[s] = [sx, sh, qz]
                out += [sx, sh]
        if self.tiles == "chips":
            # every tile's X (and before it its h) has landed: one wait on the root, on its flags
            w = self.node(R, 0, dm, "g_wait", "__snax_bingo_kernel_wait_flags",
                          SnaxBingoKernelWaitFlagsArgs(BingoMemSymbol(f"{Tg}_gflag"), self.T // rows, 1),
                          list(self.attn_ends) + [t for t in out if A.flag_dep(t, R, 0)])
            for t in out:
                if not A.flag_dep(t, R, 0):
                    self.dfg.bingo_add_order_edge(t, w)
            return [w]
        return out

    # ---- the layer ----------------------------------------------------------------------------------------------
    def build(self):
        self.attn.build()
        self.glue_end = self.glue()
        if self.moe is None:
            self.end_window(self.glue_end)
            self.run_checks(self.part_checks + self._glue_checks(full=False))
            return
        self.phase_boundary("MoE")
        # where the MoE starts on each chip, made before the MoE's own tasks (each core runs its tasks in the order
        # they are made): its args loads, the root's both in one probe
        p, R = len(self.phase_starts), self.ROOT
        self.moe_go = {k: self.loaders[(p, k, 0)] for k in self.chips}
        self.moe_go[R] = self.probe(R, "moe_go", [self.loaders[(p, R, c)] for c in range(self.NC)])
        self.part_checks += self._glue_checks(full=True)
        self.moe.build()

    def _glue_checks(self, full):
        """h (the residual, in the output's slice layout) and, without the MoE, X: windows at the head, the middle
        and the tail, from the root's L3."""
        R, Tg, n = self.ROOT, self.TAG, 2048
        h3 = np.concatenate([self.attn.d.layout.to_d(self.h16[:, s * self.SW:(s + 1) * self.SW], MESH3)
                             for s in range(HIDDEN // self.SW)]).astype(np.float16).view(np.uint8)
        out = [(f"h3_{nm}", BingoMemSymbol(f"{Tg}_h3", offset=off, chip_id=self.h3_chip), h3[off:off + n], n)
               for nm, off in (("a", 0), ("m", h3.size // 2), ("b", h3.size - n))]
        if not full:
            x8 = self.x8.astype(np.int8).view(np.uint8).reshape(-1)
            out += [(f"x8_{nm}", BingoMemSymbol(f"{Tg}_x8", offset=off, chip_id=R), x8[off:off + n], n)
                    for nm, off in (("a", 0), ("m", x8.size // 2), ("b", x8.size - n))]
        return out

    # ---- emission -----------------------------------------------------------------------------------------------
    def emit_ring_schedules(self):
        for part in (self.attn, self.moe):
            if part is not None:
                part.emit_ring_schedules()

    def data_h_extra(self):
        return self.st.lines()

    def finish(self, app_name, x_prefixes=("x_wait_", "x_pull_", "x_copy_")):
        if self.attn.fresh:
            # a fresh prompt's W_KVA loads run on cluster 0 before its x fetch, behind the chip's input-norm put: the
            # cluster's first part of x, and after the start stamp and (on the root) the x multicasts the pushes wait
            # for, so no ring load spins ahead of what its chunks depend on (chiplet_mc.py check_core_order)
            x_prefixes = tuple(x_prefixes) + ("n_put_",)
        if self.l3_arena:                                   # every name within its place in dl_act
            XS, asked = self.T * HIDDEN, self.st.asked
            KS = -(-(self.T // 16) * fam.KVP // 64) * 64
            room = {"fp_xsrc": XS, "fp_xland": XS, "fp_kv": KS, "fp_oland": 2 * XS + HIDDEN - KS, "dl_h3": 2 * XS,
                    "mp_xy": 2 * XS + HIDDEN}
            bad = {n: asked[n] for n, b in room.items() if asked.get(n, 0) > b}
            if bad:
                raise ValueError(f"params layer.l3_arena: {bad} overrun their places in dl_act ({room})")
        self._assign_phases()
        self.dfg.split_chip_functions = True
        self.st.finish()
        super().finish(app_name, x_prefixes)


def _app_name(S):
    return (f"DeepSeek-V2-Lite layer 1 prefill, {S.T} tokens, {len(S.chips)} chips x {S.NC} clusters"
            + ("" if S.moe else ", attention and glue only"))


def run():
    run_driver("dsv2 whole-layer prefill workload generator", LayerBuild, _app_name)
