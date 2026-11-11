# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""The driver of the decode whole-layer workloads (workloads/llama3/decode/half, workloads/llama3/decode/layer): one decoder
layer of Llama-3-8B (synthetic weights of the model's shapes, INT4) for ONE token at position ctx over a ctx-position KV
cache, on the 16 + 4 chiplet platform, every step on the device, the two decode blocks in ONE graph:

    x -> RMSNorm (input) -> GQA decode -> + x = h -> RMSNorm (post-attention) -> FFN decode -> + h -> the output row

THE BLOCKS are the decode modes of the prefill mappings, built with shared=the layer: the attention is
llama3_decode_mc.py's (a query head a cluster, a key/value group on a column's chip pair, every projection a one-row
GEMV, each chip's half of its group's cache from its memory chip's HBM, the new key and value appended there); the FFN is
ffn/common/ffn_decode_mc.py's (the prefill's single pass on one 16-row tile, the token's row 0; down by output columns,
ChainGathers into the root).

THE I/O. x, the residual stream entering the layer, is one FP16 row in the HBM of the root's memory chip (0x20); the
layer's output row goes into that HBM. Weights and the KV cache stream from the memory chips' HBM as in the blocks; the
new key and value halves are appended to the cache there. The gains and the blocks' output dequantisation factors (one
FP16 row each) sit in the same HBM and come into the root's L1 before the window.

THE STEPS. The single-row work runs on the root's (C00) cluster 0, which holds the blocks' row ends already (the
attention gathers y in the root's L3, the FFN folds Y into the root):
    input norm   x's row from HBM (DM), RMSNorm (simd_rmsnorm, one row), times the input gain (SIMD), its 8-byte atoms
                 (4 values) 16 bytes apart into a zeroed FP16 row (xDMA), quantised (SIMD): x8 in a_row, the GEMV's A;
                 the root's DM core multicasts it from L1 into every chip's L3 behind one flag (also into its own), where
                 the attention's x fetches wait for it
    attention    the GQA decode block; its y (W_O's raw FP16 row) stays in the root's L3
    glue         y's row into L1 (DM), times W_O's column factors, plus x: h (kept in L1 for the final residual);
                 RMSNorm, times the post-attention gain, quantised: the FFN's X row (INT8, 4 KiB). The root's DM core
                 multicasts it from L1 into every chip's L3 behind one flag, where the FFN's x_row fetch scatters it into
                 row 0 of each cluster's zeroed X tile. params layer.stop attn: h and X into the root memory chip's HBM
                 instead, and the layer ends there
    FFN          the FFN decode block on that X; its Y_raw row ends in the root's cluster 0 L1
    final        Y_raw times the down projection's column factors, plus h: the output row, written into the HBM

THE FFN's X PATH. The glue makes X on the root's cluster 0, so its source is that L1; the root's DM core multicasts it
(idma_put_flagged, bcast with also_here) into every chip's L3 at the FFN block's landing area and flag, the receiving end
the FFN block's own (a wait on the flag, a 2-D copy into row 0). Not through the HBM: that is a round trip, and the
memory chip's push engine is driven by the root's host, which runs the FFN's weight prefetcher by then (with
ffn_prefetch early it is inside it for the whole FFN: an X push queued behind it never runs). The root's also_here copy is
L1 -> own L3, local, so no remote read meets a push into that L3 (the HBM -> own-L3 trap). Deadlock-free by order: the
layer's collectives are three multicasts from one source (the root's cluster-0 DM core: x, the attention's o rows, X) and
the FFN's ChainGathers, which run one at a time; each is a dependency descendant of the one before (x -> the sweeps -> o;
o -> W_O -> y -> the glue -> X; X -> every chip's X wait -> the down GEMMs -> the first gather -> the next gathers), so no
two of them are ever in flight together. _check_collectives proves it on the graph (every pair ordered by a path).

THE WEIGHT STREAM. The decode is bound by the memory-chip links, and most of their bytes are the FFN's weights. Each
chip's host runs its FFN prefetcher after its attention prefetcher; params layer.ffn_prefetch:
    after   (default) and once that chip's attention part has ended (its appends, behind its last weight load and y put);
            the two blocks' weight rings are one L3 region, since no attention chunk is read after then
    early   right behind its attention prefetcher, so the FFN's first ring-full of chunks (params ffn.weight_ring.slots a
            cluster) streams while the attention still computes (the sweep, the o rows, W_O, the glue); the rings are two
            L3 regions
The window report prints each block's bytes over the memory links, the busiest link, and the FFN bytes the early rings
take ahead.

ORDER AND PLACEMENT. Every device core runs its tasks in creation order (one program order over the layer), except the
attention's k / v projection (its kv_at) and the FFN's set-up (its scales into L1, its X tiles zeroed), which runs before
the window on each core, as in the FFN block alone. The layer's glue on the root runs behind the attention's tasks there;
each chip's FFN tasks behind its attention tasks.

THE GOLDEN is the device's arithmetic end to end: the attention data's prompt row ctx and input norm (LlamaDecodeData),
its y; the glue's operators (hwmodel rmsnorm, mul16, add16, quant_i8); the FFN block's golden on the glue's h (its static
scales calibrated on that row); the final residual.

CHECKS, byte for byte: stop full -- the output row in HBM, h (from the root's L1), the input norm's x8 and the glue's X in
the root's and the last chip's L3, the attention's y in the root's L3 and its appended key and value halves, the FFN's
Y_raw row and the root's A8; stop attn -- h and X in HBM, x8 in L3, the attention's checks (params attn.checks).

THE KEY-SPLIT ATTENTION (params layer.attn ks; hp, the default, is the head-parallel block above). The attention is
llama3_decode_ks_mc.py's (KsBuild: chip i holds key tile i, its cluster c the query half c; the q pieces gathered on the
root and multicast as two halves, the row maxima met by one ChainGather and m* multicast, the partials reduced by a
second ChainGather, o multicast, W_O by output columns; each chip's W_O pushes wait for the reduce). What changes:
    x            the same input norm and a_row multicast, into the block's landing area lk_xland behind lk_xtflag
    set-up       every cluster's exchange buffer of the block zeroed before the window (its q halves, new-key tiles and
                 partials rely on zero elsewhere), and held to the layer's end beside the FFN's (the remote writers and
                 readers name it by address): the two sit at one offset on every chip, check_xch_offsets proves it
    y            the root's L3 row lk_yrow, behind the block's y wait
    append       made at the layer's end (KsBuild.kv_append): the new key's chip's cluster 1 writes the new K and V^T
                 tiles into its memory chip's HBM behind the FFN's last Y copies (stop attn: behind the y wait)
The proof (_check_collectives, attn ks) adds to the pairwise order of every multicast and ChainGather (the x, q halves,
m*, o and X multicasts; the statistics', the reduce's and the FFN's ChainGathers): every ChainGather runs after every
push of a memory chip's push part has landed (the part's ring loads or cache fetches among its ancestors) or before the
part starts (the part's host task among its descendants), for every part of both blocks (the attention's W_q, cache,
k / v and W_O parts, the FFN's prefetcher); and every device write into HBM (the appends, the output row, stop attn's h
and X) follows every multicast, every ChainGather and every chip's last ring load and cache fetch. With wo_push
after_reduce and ffn_prefetch after, the FFN's own gathers need ffn.chains_after_stream (every gather behind every
cluster's last down GEMV).

X FOR THE ALL-GEMV FFN (ffn.gu_gemv, or stop attn's params layer.x_arow): X in a_row, value c at byte (c // 4) * 8 + c % 4
(2 x hidden bytes): the glue spreads the normed FP16 row's 8-byte atoms 16 bytes apart into the input norm's zeroed FP16
a_row row (xDMA, its gaps still zero) and quantises it (SIMD, the gaps stay zero); the root's DM core multicasts the
8 KiB into every chip's L3 at the FFN's landing area (fd_xland) behind fd_xtflag, where the FFN's flagged fetch takes it
into L1. The FFN's Y_raw row ends in the root's cluster 0 L1 as before; the final residual reads it there.

PARAMETERS (params.hjson): the platform and the graph switches both blocks share, then `attn` (llama3_decode_mc.py's
params, or with layer.attn ks llama3_decode_ks_mc.py's), `ffn` (ffn_decode_mc.py's, ffn_prefill_mc.py's) and `layer`:
attn (hp | ks), stop (full | attn), ffn_prefetch (after | early), x_arow (stop attn: X into HBM in a_row; stop full
takes ffn.gu_gemv), args_phase_bytes (at the top level: each cluster's task args in phases of at most this many bytes).
"""

import itertools
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
for sub in ("ffn", "fa", "dsv2"):
    sys.path.insert(0, os.path.join(WL, sub, "common"))
sys.path.insert(0, HERE)

import _bingo_paths  # noqa: F401,E402
import networkx as nx  # noqa: E402
from bingo_kernel_args import (  # noqa: E402
    SnaxBingoKernelIdma1dCopyArgs, SnaxBingoKernelIdmaPutFlaggedArgs, SnaxBingoKernelSimdAddF16Args,
    SnaxBingoKernelSimdFp16ToInt8Args, SnaxBingoKernelSimdMulF16Args, SnaxBingoKernelSimdRmsnormArgs,
    SnaxBingoKernelWaitFlagsArgs, SnaxBingoKernelXdma6dArgs, SnaxBingoKernelXdmaMemsetArgs, simd_beats)
from bingo_mem_handle import BingoMemFixedAddr, BingoMemSymbol  # noqa: E402
from libs.verify import checks as vchecks  # noqa: E402

import chiplet_mc as cm  # noqa: E402
import ffn_data  # noqa: E402
import ffn_decode_mc as fdm  # noqa: E402
import gemv_utils  # noqa: E402  (util/sim/gemm: ffn_data puts it on sys.path)
import llama3_data  # noqa: E402
import llama3_decode_mc as lmd  # noqa: E402
import llama3_decode_ks_mc as lks  # noqa: E402
from chiplet_mc import AliasStaging  # noqa: E402

ATOM, LANES = lmd.ATOM, lmd.LANES          # the xDMA's 8-byte lane, lanes a beat
LINK_GBPS = 231.2                          # the memory chips' links together, GB/s
COPIES = ("__snax_bingo_kernel_idma_1d_copy", "__snax_bingo_kernel_idma_2d_copy",
          "__snax_bingo_kernel_idma_fetch_flagged", "__snax_bingo_kernel_xdma_1d_copy")


class _AttnHooks:
    """The attention block in the layer: its window opened by the layer, x made by the layer's input norm, y kept in the
    root's L3, its checks the layer's (made after the layer's end stamp: BLOCK._checks)."""
    x_on_host = False
    y_to_hbm = False

    def __init__(self, layer, args, p, plat, hw):
        self.layer = layer
        super().__init__(args, p, plat, hw, shared=layer)

    def setup(self, extra=None):
        return self.layer.open_window(extra)

    def x_multicast(self, go):
        return self.layer.x_put

    def y_out(self, yw):
        self.y_ready = yw
        return yw

    def end_window(self, after):
        self.layer.attn_fin = dict(zip(self.chips, after))

    def _checks(self):
        """Made by the layer once its end stamp exists (a stage check copies after it)."""

    def run_checks(self, checks):
        self.layer.part_checks += list(checks)


class AttnPart(_AttnHooks, lmd.DBuild):
    """The head-parallel GQA decode block (llama3_decode_mc.py) in the layer."""
    stream_name = "zs_ld_stream"
    BLOCK = lmd.DBuild


class AttnPartKs(_AttnHooks, lks.KsBuild):
    """The key-split GQA decode block (llama3_decode_ks_mc.py) in the layer: its append made at the layer's end."""
    stream_name = "zs_lk_stream"
    BLOCK = lks.KsBuild

    def append_now(self, deps):
        return []


class FfnPart(fdm.DBuild):
    """The FFN decode block in the layer: its data the glue's h, its X the glue's multicast, its set-up before the
    layer's window, its Y_raw row the final residual's input, its checks the layer's."""
    stream_name = "zs_fd_stream"

    def __init__(self, layer, args, p, plat, hw):
        self.layer = layer
        super().__init__(args, p, plat, hw, shared=layer)

    def _data(self, args, p):
        h = self.layer.h16[None, :]
        return ffn_data.FFNData("", llama3_data.MODEL, seed=int(p.get("seed", 1)), wbits=self.wbits,
                                tokens=1, inter=p.get("inter"), rows=dict(x16=h, cal=h))

    def pre_window(self):
        lo = max(t.node_id for t in self.dfg.node_list) + 1
        pre = super().pre_window()
        self.layer.ffn_pre = [t for t in self.dfg.node_list if t.node_id >= lo]
        return pre

    def setup(self, extra=None):
        return self.layer.ffn_window(extra)

    def x_hbm_bcasts(self, go):
        return [self.layer.ffn_x_put]

    def y_out(self, ready):
        return self.layer.final(ready)

    def end_window(self, after):
        L = self.layer
        L.end_window(list(after))
        self.end, self.check_copies = L.end, L.check_copies

    def run_checks(self, checks):
        """The block's checks, its Y_raw row read from the root's L1 (the layer writes the output row into HBM instead),
        then the layer's and the attention's."""
        out = []
        for name, got, gold, nb in checks:
            if name == "y_hbm":
                name, got = "y_raw", self.stash("y_raw", self.ROOT, 0, self.yrow, nb, [])
            out.append((name, got, gold, nb))
        self.layer.all_checks(out)


class LayerBuild(cm.ChipletBuild):
    """The whole decode layer (the module docstring), its blocks parts of the layer's graph. run_driver makes the
    program in four steps: __init__ (the parameters, the parts made), mem_handles() (the data, the golden and the
    memory: each part's in turn, between them the layer's golden that the next one reads, then the layer's own),
    build() (the task graph: the attention's, the glue's, the FFN's with the final) and finish() (the checks of the
    graph, then the compile into the C header and the data image)."""
    TAG = "dl"

    def __init__(self, args, p, plat, hw):
        """The parameters; no data, memory or task yet. ChipletBuild.__init__ with the switches both blocks share;
        params layer (stop, ffn_prefetch, attn, x_arow); with one L3 region for both weight rings, the layer's staging
        (AliasStaging); the synthetic FFN registered (llama3_data.register_ffn); then the parts, each a block over the
        layer's graph and staging with its own params: the attention (AttnPart, or AttnPartKs with attn ks) and, stop
        full, the FFN (FfnPart). X's form follows the FFN's (in a_row with ffn.gu_gemv)."""
        common = {k: v for k, v in p.items() if k not in ("attn", "ffn", "layer")}
        lp = dict(p.get("layer", {}))
        super().__init__(args, common, plat, hw)
        self.stop = str(lp.get("stop", "full"))
        if self.stop not in ("full", "attn"):
            raise ValueError(f"params layer.stop={self.stop!r}: full or attn")
        self.ffn_prefetch = str(lp.get("ffn_prefetch", "after"))
        if self.ffn_prefetch not in ("after", "early"):
            raise ValueError(f"params layer.ffn_prefetch={self.ffn_prefetch!r}: after or early")
        if self.stop == "attn" and "ffn_prefetch" in lp:
            raise ValueError("params layer.ffn_prefetch: the half layer (stop attn) has no FFN")
        # params layer.attn: the attention block, hp (head parallel, llama3_decode_mc.py) or ks (split by key,
        # llama3_decode_ks_mc.py)
        self.attn_kind = str(lp.get("attn", "hp"))
        if self.attn_kind not in ("hp", "ks"):
            raise ValueError(f"params layer.attn={self.attn_kind!r}: hp or ks")
        self.ks = self.attn_kind == "ks"
        Part = AttnPartKs if self.ks else AttnPart
        if self.stop == "full" and "x_arow" in lp:
            raise ValueError("params layer.x_arow: stop full takes X's form from the FFN (ffn.gu_gemv)")
        # after: one L3 region for both blocks' weight rings (no attention chunk is read once a chip's FFN pushes start)
        self.ring_alias = self.stop == "full" and self.ffn_prefetch == "after"
        if self.ring_alias:
            self.st = AliasStaging(self.st, {f"{Part.TAG}_wring_slots": f"{self.TAG}_wring",
                                             "fd_wring_slots": f"{self.TAG}_wring"})
        self.xch_zero = False
        self.part_checks, self.ffn, self.ffn_pre = [], None, []
        self.xch_more, self.kv_app = [], []      # attn ks: the attention's exchange buffers, its append
        self.seed = int(common.get("seed", 1))
        llama3_data.register_ffn(seed=self.seed)

        # ---- the parts --------------------------------------------------------------------------------------------
        self.attn = Part(self, args, dict(common, **p["attn"]), plat, hw)
        if self.stop == "full":
            pf = dict(common, **p["ffn"])
            pf["model"] = llama3_data.MODEL
            self.ffn = FfnPart(self, args, pf, plat, hw)
        # X's form: the INT8 row, or (x_arow) a_row, the all-GEMV FFN's X (stop full: ffn.gu_gemv)
        self.x_arow = bool(self.ffn.gu_gemv) if self.ffn is not None else bool(lp.get("x_arow", False))

    def mem_handles(self):
        """The data, the golden and the memory; the only tasks made are the blocks' prefetchers. The attention's
        (AttnPart's or AttnPartKs's mem_handles), then the golden on its data: x (its input norm checked against the
        block's row), x8 in a_row, h, and the post-attention gain. Stop full: the FFN's (FfnPart's mem_handles, on h),
        its gain checked against the layer's, s_h, X and the output's golden, its exchange buffers held to the end;
        stop attn: s_h and X from h. X's image (x_arow: in a_row); attn ks: the attention's exchange buffers held to
        the end too. The layer's own data in the HBM of the root's memory chip (x, the constants, the output or stop
        attn's h and X) and the root's cluster-0 L1 buffers."""
        # ---- the attention, and the glue's golden ------------------------------------------------------------------
        A = self.attn
        A.mem_handles()
        d = A.d
        fp, hm = d.fp, d.hm
        self.H = H = d.H
        F16 = np.float16
        self.x16 = np.asarray(d.x16[A.ctx], dtype=F16)                       # the row entering the layer
        xg = fp.mul16(hm.rmsnorm(self.x16[None, :]), d.gain16[None, :])
        if not np.array_equal(fp.quant_i8(xg, d.s_x.inv)[0], d.hq[A.ctx]):
            raise AssertionError("the input norm of one row differs from the attention data's row")
        self.x8_attn = A._a_row(d.hq[A.ctx])                                  # what the GEMVs read, a_row
        self.h16 = fp.add16(fp.mul16(d.y_raw, d.wo_f), self.x16)             # the residual stream after the attention
        self.ffn_gain = llama3_data.ffn_gain(self.seed).astype(F16)

        # ---- the FFN ----------------------------------------------------------------------------------------------
        if self.stop == "full":
            F = self.ffn
            F.mem_handles()                     # its data on h
            if not np.array_equal(F.d.gain16, self.ffn_gain):
                raise AssertionError("the FFN block's gain and the layer's differ")
            self.s_h = F.d.s_h
            self.x8_ffn = np.asarray(F.d.hq[0], dtype=np.int8)
            self.dn_f = np.asarray(F.d.dn_f, dtype=F16)
            self.out16 = fp.add16(fp.mul16(F.gold["y_raw"][0], self.dn_f), self.h16)
            self.xch = dict(F.xch)              # the FFN's exchange buffers: at offset 0, held to the end
        else:
            xg = fp.mul16(hm.rmsnorm(self.h16[None, :]), self.ffn_gain[None, :])
            self.s_h = hm.QScale.for_amax(float(np.max(np.abs(xg.astype(np.float32)))))
            self.x8_ffn = np.asarray(fp.quant_i8(xg, self.s_h.inv)[0], dtype=np.int8)
        self.x_img = gemv_utils.a_row_bytes(self.x8_ffn) if self.x_arow else self.x8_ffn
        self.XN = self.x_img.size
        if self.ffn is not None and self.XN != self.ffn.XR:
            raise AssertionError(f"the glue's X is {self.XN} B, the FFN lands {self.ffn.XR} B")
        if self.ks:
            # the attention's exchange buffers, zeroed before the window and held to the end with the layer's
            if self.ffn is None:
                self.xch = dict(A.xch)
            else:
                self.xch_more = [dict(A.xch)]

        # ---- the layer's own data: in the root memory chip's HBM, and the root's cluster-0 buffers -----------------
        st, T, R = self.st, self.TAG, self.ROOT
        m = self.mem[R]
        loc = (m >> 4, m & 15)
        self.RB = 2 * H                                                       # one FP16 row
        self.h_x16 = st.put_hbm(f"{T}_x16", self.x16.view(np.uint8), mem_chip=loc)
        cs = [d.gain16, np.asarray(d.wo_f, dtype=F16), self.ffn_gain] + ([self.dn_f] if self.ffn else [])
        self.CB = len(cs) * self.RB
        self.h_c = st.put_hbm(f"{T}_c", np.concatenate(cs).astype(F16).view(np.uint8), mem_chip=loc)
        if self.ffn:
            self.h_out = st.put_hbm(f"{T}_out", np.zeros(self.RB, dtype=np.uint8), mem_chip=loc)
        else:
            self.h_hhbm = st.put_hbm(f"{T}_h", np.zeros(self.RB, dtype=np.uint8), mem_chip=loc)
            self.h_x8hbm = st.put_hbm(f"{T}_x8", np.zeros(self.XN, dtype=np.uint8), mem_chip=loc)
        l1 = lambda name, nb: self.cx_chip(R).at(0).l1(f"zt_{T}_{name}", nb)
        scr = SnaxBingoKernelSimdRmsnormArgs(0, 0, 1, H).scratch_bytes()
        self.b = dict(c=l1("c", self.CB), x16=l1("x16", self.RB), n16=l1("n16", self.RB), nscr=l1("nscr", scr),
                      ar16=l1("ar16", 2 * self.RB), ar8=l1("ar8", self.RB), y16=l1("y16", self.RB), h=l1("h", self.RB),
                      g16=l1("g16", self.RB), gscr=l1("gscr", scr), x8=l1("x8", self.XN))
        if self.ffn:
            self.b["o16"] = l1("o16", self.RB)
        rings = (f"; ffn_prefetch {self.ffn_prefetch} ({'one L3 region' if self.ring_alias else 'two L3 regions'} for "
                 f"the weight rings)") if self.ffn else ""
        kind = ("; attention split by key, X in a_row" if self.x_arow else "; attention split by key") if self.ks \
            else ("; X in a_row" if self.x_arow else "")
        print(f"[{T}] layer: one token at position {A.ctx}; stop {self.stop}{rings}{kind}; |h| max "
              f"{float(np.max(np.abs(self.h16.astype(np.float32)))):.3g}; s_h {self.s_h.s:.4g}")

    # ---- node helpers ------------------------------------------------------------------------------------------------
    def _r(self, role, name, kernel, args, deps):
        """A task on the root's cluster 0."""
        return self.node(self.ROOT, 0, self.roles[role], name, kernel, args, list(dict.fromkeys(deps)))

    def _copy(self, name, src, dst, nb, deps):
        return self._r("dm", name, "__snax_bingo_kernel_idma_1d_copy", SnaxBingoKernelIdma1dCopyArgs(src, dst, nb), deps)

    def _ew(self, name, cls, a, b, out, deps):
        """out = a (.) b over one FP16 row (cls: the SIMD's add or multiply)."""
        return self._r("simd", name, "__snax_bingo_kernel_simd_stream_elementwise", cls(a, b, out, self.H // 32, 32), deps)

    def _rms(self, name, src, dst, scr, deps):
        return self._r("simd", name, "__snax_bingo_kernel_simd_rmsnorm",
                       SnaxBingoKernelSimdRmsnormArgs(src, dst, 1, self.H, scratch_addr=scr), deps)

    def _quant(self, name, src, dst, nbytes, inv, deps):
        bits = int(np.asarray(inv, dtype=np.float32).reshape(()).view(np.uint32))
        return self._r("simd", name, "__snax_bingo_kernel_simd_fp16_to_int8",
                       SnaxBingoKernelSimdFp16ToInt8Args(src, dst, beats=simd_beats(nbytes), rows=1, inv_scale_f32bits=bits),
                       deps)

    def _bcast(self, name, src, dst, nb, flag, deps):
        """The root's DM core multicasts `src` (its L1) into `dst` of every chip's L3, its own included, then `flag`."""
        return self._r("dm", name, "__snax_bingo_kernel_idma_put_flagged",
                       SnaxBingoKernelIdmaPutFlaggedArgs(src, dst, nb, flag, value=1, bcast=True, also_here=True), deps)

    # ---- the window and the input norm --------------------------------------------------------------------------------
    def open_window(self, extra):
        """Before the window: the layer's constants into the root's L1, its FP16 a_row zeroed. The start stamp after
        every chip's set-up, the attention's prefetchers behind it, the input norm. Returns {chip: the start}."""
        b = self.b
        pre = [self._copy("c_ld", self.h_c, b["c"], self.CB, []),
               self._r("xdma", "ar_zero", "__snax_bingo_kernel_xdma_memset",
                       SnaxBingoKernelXdmaMemsetArgs(b["ar16"], 2 * self.RB, SnaxBingoKernelXdmaMemsetArgs.PATTERN_ZERO), [])]
        if self.ks:
            # the key-split block's exchange buffers zeroed, as ChipletBuild.setup zeroes a block's own
            A = self.attn
            for k in self.chips:
                prev = []
                for c in range(self.NC):
                    prev = [self.node(k, c, self.roles["xdma"], f"xch_zero_c{c}", "__snax_bingo_kernel_xdma_memset",
                                      SnaxBingoKernelXdmaMemsetArgs(A.xch[(k, c)], A.xch_bytes,
                                                                    SnaxBingoKernelXdmaMemsetArgs.PATTERN_ZERO), prev)]
                pre += prev
        ends = [t for k in self.chips for t in (extra or {}).get(k, [])] + pre
        self.start = self.stamp("start", ends)
        self.go = {k: self.start for k in self.chips}
        for k in self.chips:
            self.edge(self.start, self.attn.prefetch[k])
        self.x_put = self.input_norm()
        return dict(self.go)

    def input_norm(self):
        """x's row from HBM, RMSNorm, the gain, a_row (FP16, then quantised), multicast into every chip's L3 at the
        attention's x landing area and flag. Returns the multicast."""
        A, b = self.attn, self.b
        ld = self._copy("in_ld", self.h_x16, b["x16"], self.RB, [self.start])
        rn = self._rms("in_rms", b["x16"], b["n16"], b["nscr"], [ld])
        gn = self._ew("in_gain", SnaxBingoKernelSimdMulF16Args, b["n16"], b["c"].view(0), b["n16"], [rn])
        ar = self._toar("in_toar", b["n16"], [gn])
        qz = self._quant("in_q", b["ar16"], b["ar8"], 2 * self.RB, A.d.s_x.inv, [ar])
        return self._bcast("in_put", b["ar8"], A.l3("xland"), A.XB, A.l3("xtflag"), [qz])

    def _toar(self, name, src, deps):
        """An FP16 row into the root's FP16 a_row (ar16): value c at (c // 4) * 8 + c % 4, each 8-byte atom (4 values)
        16 bytes apart, the other 8 bytes zero (zeroed before the window, never written); the quantiser keeps every
        value's place."""
        n = self.H // 4 // LANES
        return self._r("xdma", name, "__snax_bingo_kernel_xdma_6d",
                       SnaxBingoKernelXdma6dArgs(src, self.b["ar16"], ATOM, 2 * ATOM, [LANES * ATOM], [n],
                                                 [LANES * 2 * ATOM], [n]), deps)

    # ---- the glue, the FFN's window, the final ------------------------------------------------------------------------
    def glue(self):
        """y's row from the root's L3, W_O's factors, plus x: h; RMSNorm, the post-attention gain, quantised: X (x_arow:
        the normed row spread into the FP16 a_row first, so X is a_row). stop full: X multicast into every chip's L3 at
        the FFN's landing area and flag (returns the multicast); stop attn: h and X into HBM (returns the two writes)."""
        A, b = self.attn, self.b
        ly = self._copy("gl_ldy", A.l3("yrow"), b["y16"], self.RB, [A.y_ready])
        dq = self._ew("gl_deq", SnaxBingoKernelSimdMulF16Args, b["y16"], b["c"].view(self.RB), b["y16"], [ly])
        self.h_task = self._ew("gl_add", SnaxBingoKernelSimdAddF16Args, b["y16"], b["x16"], b["h"], [dq])
        rn = self._rms("gl_rms", b["h"], b["g16"], b["gscr"], [self.h_task])
        gn = self._ew("gl_gain", SnaxBingoKernelSimdMulF16Args, b["g16"], b["c"].view(2 * self.RB), b["g16"], [rn])
        if self.x_arow:
            ar = self._toar("gl_toar", b["g16"], [gn])
            qz = self._quant("gl_q", b["ar16"], b["x8"], 2 * self.RB, self.s_h.inv, [ar])
        else:
            qz = self._quant("gl_q", b["g16"], b["x8"], self.RB, self.s_h.inv, [gn])
        if self.ffn is None:
            return [self._copy("gl_sth", b["h"], self.h_hhbm, self.RB, [self.h_task]),
                    self._copy("gl_stx", b["x8"], self.h_x8hbm, self.XN, [qz])]
        F = self.ffn
        return self._bcast("gl_put", b["x8"], F._xland(0), F.XR, BingoMemSymbol(f"{F.TAG}_xtflag"), [qz])

    def ffn_window(self, extra):
        """The FFN's set-up (its scales into L1, its X tiles zeroed) before the window, first on its cores; its
        prefetchers per params layer.ffn_prefetch. Returns {chip: the task its X waits follow} (the chip's attention
        end)."""
        F = self.ffn
        for t in self.ffn_pre:
            self.edge(t, self.start)            # before the window
        for k in self.chips:
            for t in (extra or {}).get(k, []):
                self.edge(t, self.start)
            # the chip's host runs its FFN prefetcher after its attention prefetcher; after: also after the chip's
            # attention end, so no FFN chunk lands in a slot of the shared region an attention load still reads
            self.edge(self.attn.prefetch[k], F.prefetch[k])
            if self.ffn_prefetch == "after":
                self.edge(self.attn_fin[k], F.prefetch[k])
        print(f"[{self.TAG}] FFN: {len(self.ffn_pre)} set-up tasks before the window; its prefetchers "
              + ("behind the attention's on each host" if self.ffn_prefetch == "early" else
                 "behind each chip's attention end"))
        return dict(self.attn_fin)

    def final(self, ready):
        """Y_raw's row (the FFN's, the root's cluster 0) times the down projection's factors, plus h, into HBM. attn ks:
        the attention's append behind `ready` (the FFN's last Y copies: every gather and every push before them). Returns
        the write."""
        b, F = self.b, self.ffn
        dq = self._ew("fi_deq", SnaxBingoKernelSimdMulF16Args, F.yrow, b["c"].view(3 * self.RB), b["o16"], ready)
        ad = self._ew("fi_add", SnaxBingoKernelSimdAddF16Args, b["o16"], b["h"], b["o16"], [dq, self.h_task])
        st = self._copy("fi_st", b["o16"], self.h_out, self.RB, [ad])
        if self.ks:
            self.kv_app = self.attn.kv_append(list(ready))
        return st

    def end_window(self, after):
        """The end stamp after `after` and (attn ks) the attention's append."""
        super().end_window(list(after) + self.kv_app)

    def run_checks(self, checks):
        """ChipletBuild.run_checks: the root's host compares the checks one at a time, then holds keep the layer's
        exchange buffers live to the end; attn ks with the FFN holds the attention's too (xch_more)."""
        if not self.xch_more:
            return super().run_checks(checks)
        rcx = self.cx_chip(self.ROOT)
        prev = [self.report] + self.check_copies
        for name, got, golden, nb in checks:
            a = np.ascontiguousarray(golden)
            gh = self.st.put(f"{self.TAG}_gold_{name}", "uint8_t", a.view(np.uint8).reshape(-1))
            prev = [vchecks.check_bytes(rcx, f"Check_{self.TAG}_{name}", golden=gh, got=got,
                                        nbytes=nb, after=prev, label=f"{self.TAG}_{name}")]
        for i, xch in enumerate([self.xch] + self.xch_more):
            for (k, c), h in xch.items():
                self.node(k, c, self.roles["dm"], f"xch_hold{i or ''}_c{c}", "__snax_bingo_kernel_wait_flags",
                          SnaxBingoKernelWaitFlagsArgs(h, 0, 0), prev if k == self.ROOT else [self.end])

    def check_xch_offsets(self):
        """ChipletBuild's (the layer's exchange buffers at offset 0 of each cluster's arena); attn ks with the FFN, the
        attention's at one offset on every chip for each cluster index (its remote addresses are the root's)."""
        super().check_xch_offsets()
        pl = self.dfg.static_l1_placement
        for xch in self.xch_more:
            for c in range(self.NC):
                offs = {pl[id(xch[(k, c)])][1] for k in self.chips}
                if len(offs) != 1:
                    raise ValueError(f"the attention's exchange buffers of cluster {c} at {len(offs)} offsets: "
                                     f"{sorted(offs)[:4]}")
            print(f"[{self.TAG}] exchange buffers: the FFN's at offset 0, the attention's at "
                  f"{sorted({pl[id(h)][1] for h in xch.values()})} on every chip")

    # ---- the layer ----------------------------------------------------------------------------------------------------
    def build(self):
        self.attn.build()
        if self.ffn is None:
            sts = self.glue()
            if self.ks:
                # the append behind the root's y wait: after every multicast, ChainGather and push of the attention
                self.kv_app = self.attn.kv_append([self.attn.y_ready])
            self.end_window(sts + [self.attn_fin[k] for k in self.chips])
            self.all_checks([])
            return
        self.ffn_x_put = self.glue()
        self.ffn_first = max(t.node_id for t in self.dfg.node_list) + 1
        self.ffn.build()

    def all_checks(self, ffn_checks):
        """The attention's checks (made now, after the end stamp), the layer's, the FFN's; then compared."""
        A, R = self.attn, self.ROOT
        A.end, A.check_copies = self.end, self.check_copies
        A.BLOCK._checks(A)
        u8 = lambda a: np.ascontiguousarray(a).view(np.uint8).reshape(-1)
        last = self.chips[-1]
        xa = u8(self.x8_attn)
        checks = [(f"x8_k{k:02x}", A.l3("xland", 0, k), xa, xa.size) for k in (R, last)]
        x8 = u8(self.x_img)
        if self.ffn is None:
            h = u8(self.h16.astype(np.float16))
            checks += [("h_hbm", self.h_hhbm, h, h.size), ("x_hbm", self.h_x8hbm, x8, x8.size)]
        else:
            F = self.ffn
            o, h = u8(self.out16.astype(np.float16)), u8(self.h16.astype(np.float16))
            checks += [("out_hbm", self.h_out, o, o.size),
                       ("h", self.stash("h", R, 0, self.b["h"], h.size, []), h, h.size)]
            checks += [(f"x_k{k:02x}", F._xland(0, chip_id=k), x8, x8.size) for k in (R, last)]
        self.run_checks(self.part_checks + checks + list(ffn_checks))

    # ---- proofs and reports -------------------------------------------------------------------------------------------
    def _check_collectives(self):
        """Every multicast (a device bcast put, a memory chip's HBM multicast) and every ChainGather of the graph is
        ordered against every other by a path of dependency and order edges, so no two are ever in flight together.
        No device copy reads HBM into an L3 buffer. attn ks: also every ChainGather against every push part of both
        blocks and every device write into HBM after all of them (_check_pushes). Returns (multicasts, gathers)."""
        G = self.dfg.bingo_order_view()
        mc = [t for t in self.dfg.node_list
              if (t.kernel_name == "__snax_bingo_kernel_idma_put_flagged" and t.kernel_args.bcast & 1)
              or t.kernel_name == "__host_bingo_kernel_hbm_bcast"]
        ch = [t for t in self.dfg.node_list if t.kernel_name == "__snax_bingo_kernel_xdma_chain_gather"]
        items = sorted(mc + ch, key=lambda t: t.node_id)
        desc = {t: nx.descendants(G, t) for t in items}
        bad = [(a.node_name, b.node_name) for a, b in itertools.combinations(items, 2)
               if b not in desc[a] and a not in desc[b]]
        if bad:
            raise ValueError(f"{len(bad)} unordered collective pairs, e.g. {bad[:4]}")
        hbm_l3 = [t.node_name for t in self.dfg.node_list if t.kernel_name in COPIES
                  and isinstance(t.kernel_args.dst_addr, BingoMemSymbol)
                  and (isinstance(t.kernel_args.src_addr, int)
                       or (isinstance(t.kernel_args.src_addr, BingoMemFixedAddr)
                           and t.kernel_args.src_addr.mem_level in (None, "HBM")))]
        if hbm_l3:
            raise ValueError(f"device copies from HBM into L3: {hbm_l3[:4]}")
        more = self._check_pushes(G, mc, ch, desc) if self.ks else ""
        print(f"[{self.TAG}] collectives: {len(mc)} multicasts and {len(ch)} ChainGathers, every pair ordered by a path; "
              f"no device copy from HBM into L3{more}")
        return len(mc), len(ch)

    def _push_parts(self):
        """attn ks: {(chip, part): (the part's host task, the tasks after which every push of it has landed)}: the
        attention's W_q, cache, k / v and W_O parts and the FFN's prefetcher. A chunk has landed once its ring load (which
        waits for the chunk's landed flag) has run, the cache once both of each cluster's flagged fetches have."""
        A, F = self.attn, self.ffn
        ring = {}
        for blk, S in (("attn", A), ("ffn", F)):
            for (cl, j), (_, ld) in (S.slot.items() if S is not None else ()):
                ring.setdefault((blk, self.chip(cl)), []).append((j, ld))
        parts = {}
        for k in self.chips:
            head, mid, wo = A.pf_parts[k]
            at = ring[("attn", k)]
            parts[k, "W_q"] = (head, [ld for j, ld in at if j < A.NQC])
            parts[k, "cache"] = (A.ucast[k], [t for cl in self.clusters(k) for t in (A.kc_ready[cl], A.vc_ready[cl])])
            parts[k, "k/v"] = (mid, [ld for j, ld in at if A.NQC <= j < A.WO0])
            parts[k, "W_O"] = (wo, [ld for j, ld in at if j >= A.WO0])
            want = {"attn": sum(len(A.chunk_off[cl]) for cl in self.clusters(k))}
            if F is not None:
                parts[k, "FFN"] = (F.prefetch[k], [ld for _, ld in ring[("ffn", k)]])
                want["ffn"] = sum(len(F.chunk_off[cl]) for cl in self.clusters(k))
            for blk, n in want.items():
                if len(ring[(blk, k)]) != n:
                    raise ValueError(f"chip {k:#04x}: {len(ring[(blk, k)])} {blk} ring loads of its {n} chunks")
        return parts

    def _check_pushes(self, G, mc, ch, desc):
        """attn ks: every ChainGather runs after every push of each push part has landed or before the part starts
        (its host task a descendant), for every part of both blocks on every chip; every device write into HBM (dst a
        global address) follows every multicast, every ChainGather and every chip's last push of every part. Returns
        the report's tail."""
        parts = self._push_parts()
        beside = []
        for t in ch:
            anc = nx.ancestors(G, t)
            beside += [(t.node_name, f"{k:02x}", nm) for (k, nm), (host, landed) in parts.items()
                       if host not in desc[t] and not all(x in anc for x in landed)]
        if beside:
            raise ValueError(f"{len(beside)} ChainGather / push-part pairs may run at once, e.g. {beside[:4]}")
        hbm = lambda h: isinstance(h, (int, np.integer)) or (isinstance(h, BingoMemFixedAddr)   # noqa: E731
                                                             and h.mem_level in (None, "HBM"))
        writes = [t for t in self.dfg.node_list if t.kernel_name in COPIES and hbm(t.kernel_args.dst_addr)]
        landed = [x for _, ls in parts.values() for x in ls]
        late = []
        for w in writes:
            anc = nx.ancestors(G, w)
            late += [(w.node_name, t.node_name) for t in mc + ch + landed if t not in anc]
        if late:
            raise ValueError(f"{len(late)} HBM writes not after a collective or a push, e.g. {late[:4]}")
        return (f"; every ChainGather after or before each of the {len(parts)} push parts (both blocks, every chip); "
                f"{len(writes)} HBM writes ({', '.join(sorted({w.node_name.split('_k')[0] for w in writes}))}) after "
                f"every collective and every push")

    def _check_array_ks(self):
        """attn ks: the attention's array tasks are GEMVs and FA matmuls only (no gemm_full / gemm_minimal, as KsBuild
        checks alone), and with the all-GEMV FFN none in the whole graph, so no GEMM configuration is reused across
        another array task on any GEMM core. Prints the array tasks by kernel."""
        gemm = self.roles["gemm"]
        arr = [t for t in self.dfg.node_list if t.assigned_core_id == gemm and (t.kernel_name or "").startswith("__snax")]
        full = ("__snax_bingo_kernel_gemm_full", "__snax_bingo_kernel_gemm_minimal")
        bad = [t.node_name for t in arr if t.kernel_name in full
               and (t.node_id < self.ffn_first if self.ffn is not None else True)]
        if self.ffn is not None and self.ffn.gu_gemv:
            bad += [t.node_name for t in arr if t.kernel_name in full and t.node_id >= self.ffn_first]
        if bad:
            raise ValueError(f"gemm_full / gemm_minimal tasks where every array task is a GEMV or an FA matmul: {bad[:4]}")
        n = {}
        for t in arr:
            kn = t.kernel_name.replace("__snax_bingo_kernel_", "")
            n[kn] = n.get(kn, 0) + 1
        print(f"[{self.TAG}] array tasks: " + ", ".join(f"{v} {kn}" for kn, v in sorted(n.items())))

    def _stream_report(self):
        """Bytes over the memory links: each block's, the busiest link's, the floors at LINK_GBPS (all links) and its
        share per link, and the FFN bytes the rings take ahead under ffn_prefetch early."""
        A, F = self.attn, self.ffn
        links = sorted(set(self.link.values()))
        per_link = LINK_GBPS / len(links)
        us = lambda nb, bw: nb / (bw * 1e3)
        att = {L: 0 for L in links}
        ffn = {L: 0 for L in links}
        ahead = {L: 0 for L in links}
        for k in self.chips:
            L = self.link[k]
            att[L] += sum(n for cl in self.clusters(k) for _, n in A.chunk_off[cl]) + A.P * A.PB
            if F is not None:
                for cl in self.clusters(k):
                    ffn[L] += sum(n for _, n in F.chunk_off[cl])
                    ahead[L] += sum(n for _, n in F.chunk_off[cl][:F.n_slots])
        att[self.link[self.ROOT]] += 2 * self.RB                      # x in, the output row out (the constants load before)
        MiB = 2 ** 20
        tot = lambda dct: sum(dct.values())
        busy = lambda dct: max(dct.values())
        lines = [f"attention {tot(att) / MiB:.2f} MiB (busiest link {busy(att) / MiB:.3f} MiB): floor "
                 f"{us(tot(att), LINK_GBPS):.1f} us, busiest link {us(busy(att), per_link):.1f} us"]
        if F is not None:
            lay = {L: att[L] + ffn[L] for L in links}
            rest = {L: ffn[L] - ahead[L] for L in links}
            lines += [f"FFN {tot(ffn) / MiB:.2f} MiB (busiest link {busy(ffn) / MiB:.3f} MiB): floor "
                      f"{us(tot(ffn), LINK_GBPS):.1f} us, busiest link {us(busy(ffn), per_link):.1f} us",
                      f"layer {tot(lay) / MiB:.2f} MiB: floor {us(tot(lay), LINK_GBPS):.1f} us, busiest link "
                      f"{us(busy(lay), per_link):.1f} us",
                      f"ffn_prefetch early: the FFN rings ({F.n_slots} slots a cluster) take {tot(ahead) / MiB:.2f} MiB "
                      f"({us(tot(ahead), LINK_GBPS):.1f} us of the links) while the attention computes; the FFN's "
                      f"stream after the glue {us(tot(rest), LINK_GBPS):.1f} us (busiest link "
                      f"{us(busy(rest), per_link):.1f} us) vs {us(tot(ffn), LINK_GBPS):.1f} us with after"]
        for ln in lines:
            print(f"[{self.TAG}] stream: {ln}")

    def _args_report(self):
        """Each cluster's task args (the compiler's arena estimate: each struct 64-byte aligned) beside its static L1
        peak."""
        from bingo_args_phases import _args_bytes
        dev = {self.roles[r] for r in ("gemm", "simd", "xdma", "dm")}
        per = {}
        for t in self.dfg.node_list:
            if t.assigned_core_id in dev and (t.kernel_name or "").startswith("__snax"):
                key = (t.assigned_chiplet_id, t.assigned_cluster_id)
                per[key] = per.get(key, 0) + _args_bytes(t)
        (k, c), nb = max(per.items(), key=lambda kv: kv[1])
        print(f"[{self.TAG}] task args: at most {nb:,} B a cluster (chip {k:#04x} cluster {c}), estimate")

    # ---- emission -----------------------------------------------------------------------------------------------------
    def data_h_extra(self):
        return self.st.lines() if self.ring_alias else []

    def finish(self, app_name, x_prefixes=("x_wait_",)):
        if self.ffn is not None:
            bad = self.ffn.check_hazards()
            if bad:
                raise ValueError(f"{len(bad)} FFN buffer hazards, e.g. " + "; ".join(bad[:4]))
            print(f"[{self.TAG}] FFN buffer hazards: none")
        if self.ks:
            self._check_array_ks()
            self.attn._check_collectives()
        self._check_collectives()
        self._stream_report()
        self._args_report()
        self.dfg.split_chip_functions = True
        if self.ring_alias:
            self.st.finish()
        super().finish(app_name, x_prefixes)


def _app_name(S):
    return (f"Llama-3-8B decode layer (synthetic weights), one token over a {S.attn.ctx}-position cache, "
            f"{len(S.chips)} chips x {S.NC} clusters" + ("" if S.ffn else ", input norm, attention and glue only")
            + (", attention split by key" if S.ks else "") + (", X in a_row" if S.x_arow else ""))


def run():
    run_driver("llama3 decode whole-layer workload generator", LayerBuild, _app_name)
