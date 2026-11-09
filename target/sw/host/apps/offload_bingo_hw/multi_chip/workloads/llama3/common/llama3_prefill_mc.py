# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""The driver of the whole-layer workloads (workloads/llama3/<size>/layer): one decoder layer of Llama-3-8B with synthetic
weights of the model's shapes over a fresh prompt of T tokens on the 16 + 4 chiplet platform, every step on the device,
built from two blocks in ONE graph:

    x -> RMSNorm (input) -> attention -> + x = h -> RMSNorm (post-attention) -> SwiGLU FFN -> + h

THE INPUT is the attention data's prompt (llama3_data.LlamaAttnData: x16, seeded), FP16, once in each memory chip's HBM.

THE STEPS, in the order they run; the row tiles (16 rows each) run one a chip, on its cluster 0:
    input norm   x's rows from the chip's memory chip, RMSNorm (simd_rmsnorm, row major), times the input gain
                 (simd_mul_tiles), reshaped to the A layout (xDMA), quantised, put into the root's attention x source with a
                 landed flag; the root multicasts a pass once its tiles' flags are in
    attention    llama3_attn_mc.py's block (params attn): a query head a cluster, the passes one after the other, the
                 output partials folded into the root, which stores the raw y in its L3
    glue         y's rows from the root, D layout -> row major (xDMA), dequantised (W_O's factors), plus x: h, kept in the
                 chip's own L3 for the last residual; RMSNorm, times the post-attention gain, reshaped, quantised, put into
                 the root's FFN X source with a flag
    FFN          the ffn driver's block (params ffn, ../../ffn/common/ffn_prefill_mc.py) on the glue's X
    final        the FFN's raw Y rows from the root, D layout -> row major, dequantised, plus h: the layer's output, in the
                 chip's L3

THE GOLDEN is the device's arithmetic end to end: the attention block's y (its fold order), the glue's operators (hwmodel
mul16, add16, rmsnorm, quant_i8), the FFN block's golden on the glue's h (its scales calibrated on that h), the final
residual.

L1. The blocks' buffers overlay (the static planner, by lifetime), and the tasks' args come in ARGS PHASES
(bingo_dfg_emit.py): the input norm and the attention's first passes; its last passes; the glue and the FFN's first
passes; its last passes and the final tiles. A cluster's args of a phase come into L1 by one DM task after every task of
the previous phase on the chip.

L3. The two blocks' weight rings are one region (the FFN's prefetchers start after the attention is done); each block
keeps its y over its x buffers (l3_overlay); h and the output stay in the chip that made them.

PARAMETERS (params.hjson): the platform and the graph switches both blocks share, then `attn` (llama3_attn_mc.py's
params), `ffn` (ffn_prefill_mc.py's) and `layer`: stop (full | attn: the FFN and the final step left out, the glue's
X checked), phases (the args phase boundaries, a subset of attn2, glue, ffn2).
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
for sub in ("ffn", "fa", "dsv2"):
    sys.path.insert(0, os.path.join(WL, sub, "common"))
sys.path.insert(0, HERE)

import _bingo_paths  # noqa: F401,E402
from bingo_kernel_args import (  # noqa: E402
    PhaseArgsLoadArgs, SnaxBingoKernelIdma1dCopyArgs, SnaxBingoKernelIdma2dCopyArgs, SnaxBingoKernelIdmaPutFlaggedArgs,
    SnaxBingoKernelSimdAddF16Args, SnaxBingoKernelSimdFp16ToInt8Args, SnaxBingoKernelSimdMulTilesArgs,
    SnaxBingoKernelSimdRmsnormArgs, SnaxBingoKernelWaitFlagsArgs, simd_beats)
from bingo_mem_handle import BingoMemFixedAddr, BingoMemSymbol  # noqa: E402
from bingo_gemm_reuse import reuse_gemm_config  # noqa: E402
from libs.blocks import GemmQueue  # noqa: E402
from libs.comm.nest import convert_args  # noqa: E402

import chiplet_mc as cm  # noqa: E402
import ffn_data  # noqa: E402
import ffn_prefill_mc as fpm  # noqa: E402
import llama3_attn_mc as lam  # noqa: E402
import llama3_data  # noqa: E402
from dsv2_prefill_mc import AliasStaging  # noqa: E402

TR = 16                       # rows a tile: one A-layout row block


class LazyWaits(dict):
    """{pass: [the root's wait]} whose waits are made when a block's multicast first asks for them: the in-order DM
    core then runs pass i's wait just before pass i's multicast, not every pass's wait ahead of the first one."""

    def __init__(self, make, n):
        super().__init__()
        self._make, self._n = make, n

    def __missing__(self, i):
        if not 0 <= i < self._n:
            raise KeyError(i)
        v = self[i] = self._make(i)
        return v

    def get(self, i, default=None):
        return self[i] if 0 <= i < self._n else default

    def every(self):
        """Every pass's wait, those not asked for yet made now."""
        return [t for i in range(self._n) for t in self[i]]

    def check_made(self, name):
        if len(self) != self._n:
            raise ValueError(f"{name}: the waits of passes {sorted(set(range(self._n)) - set(self))} were never asked "
                             f"for, so those multicasts do not wait for their tiles")


class AttnPart(lam.ABuild):
    """The attention block in the layer: its window opened by the layer, its passes the layer's args phases, its checks
    the layer's."""
    stream_name = "zs_la_stream"

    def __init__(self, layer, args, p, plat, hw):
        self.layer = layer
        super().__init__(args, p, plat, hw, shared=layer)

    def setup(self, extra=None):
        return self.layer.open_window(extra)

    def pass_start(self, i):
        self.layer.at_pass("attn", i)

    def end_window(self, after):
        self.layer.attn_ends = list(after)

    def run_checks(self, checks):
        self.layer.part_checks += list(checks)


class FfnPart(fpm.PBuild):
    """The FFN block in the layer: its data the glue's h, its window opened by the layer, its end the final step's."""
    stream_name = "zs_pf_stream"

    def __init__(self, layer, args, p, plat, hw):
        self.layer = layer
        super().__init__(args, p, plat, hw, shared=layer)

    def _data(self, args, p):
        L = self.layer
        return ffn_data.FfnData("", llama3_data.MODEL, seed=int(p.get("seed", 1)), wbits=self.wbits,
                                tokens=self.T, inter=p.get("inter"), rows=dict(x16=L.h16, cal=L.h16))

    def setup(self, extra=None):
        return self.layer.ffn_window(extra)

    def pass_start(self, i):
        self.layer.at_pass("ffn", i)

    def end_window(self, after):
        L = self.layer
        L.end_window(L.final(list(after)))
        self.end, self.check_copies = L.end, L.check_copies

    def run_checks(self, checks):
        L = self.layer
        cm.ChipletBuild.run_checks(L, L.part_checks + list(checks) + L.final_checks())


class LayerBuild(cm.ChipletBuild):
    TAG = "ll"

    def __init__(self, args, p, plat, hw):
        common = {k: v for k, v in p.items() if k not in ("attn", "ffn", "layer")}
        lp = self.lp = dict(p.get("layer", {}))
        super().__init__(args, common, plat, hw)
        self.st = AliasStaging(self.st, {"la_wring_slots": "ll_wring", "pf_wring_slots": "ll_wring"})
        self.sym_zero = False
        self.stop = str(lp.get("stop", "full"))
        if self.stop not in ("full", "attn"):
            raise ValueError(f"params layer.stop={self.stop!r}: full or attn")
        self.phases = set(lp.get("phases", ["attn2", "glue", "ffn2"]))
        if not self.phases <= {"attn2", "glue", "ffn2"}:
            raise ValueError(f"params layer.phases={sorted(self.phases)}: a subset of attn2, glue, ffn2")
        self.phase_starts, self.phase_names, self.loaders = [], ["input norm, attention"], {}
        self.part_checks = []
        llama3_data.register_ffn(seed=int(common.get("seed", 1)))

        # ---- the attention, and the glue's golden ----------------------------------------------------------
        pa = dict(common, **p["attn"])
        pa["x_device"] = True
        A = self.attn = AttnPart(self, args, pa, plat, hw)
        d = A.d
        fp = d.fp
        self.H = d.H
        if self.T % TR or self.T // TR != len(self.chips) or A.R % TR:
            raise ValueError(f"{self.T} tokens: one {TR}-row tile a chip, whole tiles a pass")
        self.x16 = np.asarray(d.x16, dtype=np.float16)
        self.h16 = fp.add16(d.y16(A.gold_y), self.x16)                    # the residual stream after the attention
        self.ffn_gain = llama3_data.ffn_gain(int(common.get("seed", 1))).astype(np.float16)

        # ---- the FFN ----------------------------------------------------------------------------------------
        self.ffn = None
        if self.stop == "full":
            pf = dict(common, **p["ffn"])
            pf["x_device"] = True
            pf["model"] = llama3_data.MODEL
            if pf.get("dn_balance") or pf.get("chain_alt") or not pf.get("l3_overlay") or str(pf.get("dn_split")) != "cols":
                raise ValueError("params ffn: the final step reads Y as dn_split cols, l3_overlay, without dn_balance and "
                                 "chain_alt (uniform slices in column order)")
            F = self.ffn = FfnPart(self, args, pf, plat, hw)
            if not np.array_equal(F.d.gain16, self.ffn_gain):
                raise AssertionError("the FFN block's gain and the layer's differ")
            if F.ROWS != A.R:
                raise ValueError("params ffn.rows: the attention's rows (a pass a tile group)")
            self.s_h = F.d.s_h
            self.x8_ffn = F.d.hq
            self.out16 = fp.add16(fp.mul16(F.gold["y_raw"], F.d.dn_f[None, :]), self.h16)
            self.dn_f = np.asarray(F.d.dn_f, dtype=np.float16)
        else:
            amax = float(np.max(np.abs(fp.mul16(d.hm.rmsnorm(self.h16), self.ffn_gain[None, :]).astype(np.float32))))
            self.s_h = d.hm.QScale.for_amax(amax)
            self.x8_ffn = fp.quant_i8(fp.mul16(d.hm.rmsnorm(self.h16), self.ffn_gain[None, :]), self.s_h.inv)
            self.dn_f = np.zeros(self.H, dtype=np.float16)

        # ---- the layer's own data ---------------------------------------------------------------------------
        st, T = self.st, self.TAG
        self.h_x16 = {m: st.put_hbm(f"{T}_x16_m{m:02x}", self.x16.view(np.uint8).reshape(-1), mem_chip=(m >> 4, m & 15))
                      for m in sorted(set(self.mem.values()))}
        self.CB = self.H * 2
        st.put(f"{T}_c", "uint16_t", np.concatenate([d.gain16, np.asarray(d.wo_f, dtype=np.float16), self.ffn_gain,
                                                     self.dn_f]).view(np.uint16))
        self.TB = TR * self.H * 2                                         # a tile, FP16
        st.put_zeros(f"{T}_h", "uint8_t", self.TB, preload=True)          # each chip's tile of h
        st.put_zeros(f"{T}_out", "uint8_t", self.TB, preload=True)        # each chip's tile of the output
        st.put_zeros(f"{T}_nflag", "uint8_t", 64 * (self.T // TR), preload=True)
        st.put_zeros(f"{T}_gflag", "uint8_t", 64 * (self.T // TR), preload=True)
        if self.ffn is None:
            st.put_zeros(f"{T}_x8", "int8_t", self.T * self.H, preload=True)
        print(f"[{T}] layer: {self.T} tokens, one {TR}-row tile a chip; stop {self.stop}; |h| max "
              f"{float(np.max(np.abs(self.h16))):.3g}, rms {float(np.sqrt(np.mean(self.h16.astype(np.float32) ** 2))):.3g}")

    # ---- the args phases (as dsv2_prefill_mc.py's) ---------------------------------------------------------------
    def _dev(self, n):
        return (n.kernel_name or "").startswith("__snax")

    def phase_boundary(self, name):
        lo = self.phase_starts[-1] if self.phase_starts else 0
        last = {}
        for n in self.dfg.node_list:
            if n.node_id >= lo and self._dev(n) and not getattr(n, "args_resident", False):
                key = (n.assigned_chiplet_id, n.assigned_cluster_id, n.assigned_core_id)
                if key not in last or n.node_id > last[key].node_id:
                    last[key] = n
        p = len(self.phase_starts) + 1
        for k in self.chips:
            for c in range(self.NC):
                deps = [t for (kk, cc, _), t in sorted(last.items(), key=lambda x: x[0]) if kk == k]
                ld = self.node(k, c, self.roles["dm"], f"args_p{p}", "__snax_bingo_kernel_idma_1d_copy",
                               PhaseArgsLoadArgs(k, c, p), deps)
                ld.args_resident = True
                self.loaders[(p, k, c)] = ld
        self.phase_starts.append(max(n.node_id for n in self.dfg.node_list) + 1)
        self.phase_names.append(name)
        return p

    def _assign_phases(self):
        starts = self.phase_starts
        first = {}
        for n in self.dfg.node_list:
            if not self._dev(n) or getattr(n, "args_resident", False):
                continue
            p = sum(1 for s in starts if n.node_id >= s)
            n.args_phase = p
            key = (p, n.assigned_chiplet_id, n.assigned_cluster_id, n.assigned_core_id)
            if key not in first or n.node_id < first[key].node_id:
                first[key] = n
        for (p, k, c, core), n in first.items():
            if p and not self.dfg.has_edge(self.loaders[(p, k, c)], n):
                self.edge(self.loaders[(p, k, c)], n)
        counts = {}
        for n in self.dfg.node_list:
            if self._dev(n) and n.assigned_chiplet_id == self.ROOT and n.assigned_cluster_id == 0:
                counts[getattr(n, "args_phase", "r")] = counts.get(getattr(n, "args_phase", "r"), 0) + 1
        print(f"[{self.TAG}] args phases: " + "; ".join(f"{p} {nm}" for p, nm in enumerate(self.phase_names))
              + f"; root cluster 0 tasks a phase {dict(sorted(counts.items(), key=str))}")

    def at_pass(self, block, i):
        """A block's pass i (i >= 1) begins: the args phase boundary there, if params layer.phases has it."""
        if i == 2 and f"{block}2" in self.phases:
            self.phase_boundary(f"{block} passes 2 ..")

    # ---- the windows -------------------------------------------------------------------------------------------
    def open_window(self, extra):
        """The start stamp (after every chip's setup), the attention's prefetchers behind it, and the input norm."""
        ends = [t for k in self.chips for t in (extra or {}).get(k, [])]
        start = self.stamp("start", ends)
        self.go = {k: start for k in self.chips}
        for k in self.chips:
            self.edge(start, self.attn.prefetch[k])
        self.attn.x_ready = self.input_norm(start)
        return dict(self.go)

    def ffn_window(self, extra):
        """Where the FFN starts on each chip: behind the glue's phase loads; its setup tasks and prefetchers after."""
        go = dict(self.ffn_go)
        for k in self.chips:
            for t in (extra or {}).get(k, []):
                self.edge(go[k], t)
            self.edge(go[k], self.ffn.prefetch[k])
        return go

    # ---- the row tiles ------------------------------------------------------------------------------------------
    def _x16(self, k, q):
        """Tile q of the prompt's x as chip k reads it: from its memory chip's HBM."""
        return BingoMemFixedAddr(self.h_x16[self.mem[k]].address + q * self.TB, mem_level="HBM")

    def _on(self, sym, k, chip):
        return sym if k == chip else BingoMemSymbol(sym.symbol_name, offset=sym.offset, chip_id=chip)

    def _bufs(self, k, step):
        """A step's (n: input norm, g: glue, f: final) tile buffers on chip k's cluster 0: each step its own, so they
        live only while it runs and the static planner lays them over the blocks' dead ones; named to sort after the
        blocks' buffers (the packer places by name)."""
        if not hasattr(self, "_tb"):
            self._tb = {}
        if (k, step) not in self._tb:
            l1 = lambda name, nb: self.cx_chip(k).at(0).l1(f"zt_{self.TAG}_{step}{name}", nb)
            b = dict(a=l1("a", self.TB), b=l1("b", self.TB), d=l1("d", self.TB), x8=l1("x8", TR * self.H),
                     c=l1("c", 4 * self.CB))
            b["scr"] = l1("scr", SnaxBingoKernelSimdRmsnormArgs(0, 0, TR, self.H).scratch_bytes())
            self._tb[k, step] = b
        return self._tb[k, step]

    def _tile(self, q):
        return self.chips[q], q // (self.attn.R // TR), q % (self.attn.R // TR)     # chip, pass, row block in the pass

    def _node(self, k, role, name, kernel, args, deps):
        return self.node(k, 0, self.roles[role], name, kernel, args, list(dict.fromkeys(deps)))

    def _rms_q(self, k, name, src, gain, inv, b, deps):
        """Rows of `src` (row major FP16) normalised, times `gain`, reshaped to the A layout, quantised into b['x8']."""
        H = self.H
        rn = self._node(k, "simd", f"{name}_rms", "__snax_bingo_kernel_simd_rmsnorm",
                        SnaxBingoKernelSimdRmsnormArgs(src, b["a"], TR, H, scratch_addr=b["scr"]), deps)
        gm = self._node(k, "simd", f"{name}_gain", "__snax_bingo_kernel_simd_mul_tiles",
                        SnaxBingoKernelSimdMulTilesArgs(b["a"], gain, b["a"], TR, H // 128, 2 * H, 2 * H), [rn])
        cv = self._node(k, "xdma", f"{name}_toa", "__snax_bingo_kernel_xdma_6d",
                        convert_args("row_major", "A", TR, H, cm.MESH, 2, b["a"], b["d"]), [gm])
        bits = int(np.asarray(inv, dtype=np.float32).reshape(()).view(np.uint32))
        return self._node(k, "simd", f"{name}_q", "__snax_bingo_kernel_simd_fp16_to_int8",
                          SnaxBingoKernelSimdFp16ToInt8Args(b["d"], b["x8"], beats=simd_beats(self.TB), rows=1,
                                                            inv_scale_f32bits=bits), [cv])

    def _consts(self, k, step, after):
        b = self._bufs(k, step)
        return self._node(k, "dm", f"{step}_c_ld", "__snax_bingo_kernel_idma_1d_copy",
                          SnaxBingoKernelIdma1dCopyArgs(BingoMemSymbol(f"{self.TAG}_c"), b["c"], 4 * self.CB), after)

    def _gather_flags(self, name, flag, puts, per, deps):
        """One wait on the root for each group of `per` tiles' flags: {group: [the wait]}, each made when first asked
        for (LazyWaits)."""
        R, dm = self.ROOT, self.roles["dm"]

        def make(i):
            grp = puts[i * per:(i + 1) * per]
            w = self.node(R, 0, dm, f"{name}_wait_i{i}", "__snax_bingo_kernel_wait_flags",
                          SnaxBingoKernelWaitFlagsArgs(BingoMemSymbol(f"{self.TAG}_{flag}", offset=64 * i * per), per, 1),
                          list(deps) + [t for t in grp if t.assigned_chiplet_id == R])
            for t in grp:
                if t.assigned_chiplet_id != R:
                    self.dfg.bingo_add_order_edge(t, w)
            return [w]
        return LazyWaits(make, len(puts) // per)

    def input_norm(self, start):
        """The attention's x, a tile a chip, into the root's x source: {pass: [the root's wait on its tiles' flags]}."""
        A, R, H = self.attn, self.ROOT, self.H
        inv = A.d.s_x.inv
        puts = []
        for q in range(self.T // TR):
            k, i, m = self._tile(q)
            b = self._bufs(k, "n")
            cld = self._consts(k, "n", [start])
            ld = self._node(k, "dm", f"n_ld_q{q}", "__snax_bingo_kernel_idma_1d_copy",
                            SnaxBingoKernelIdma1dCopyArgs(self._x16(k, q), b["b"], self.TB), [start])
            qz = self._rms_q(k, f"n_q{q}", b["b"], b["c"].view(0), inv, b, [ld, cld])
            puts.append(self._node(k, "dm", f"n_put_q{q}", "__snax_bingo_kernel_idma_put_flagged",
                                   SnaxBingoKernelIdmaPutFlaggedArgs(
                                       b["x8"], self._on(A.xsrc(i * A.XB + m * TR * H), k, R), TR * H,
                                       self._on(BingoMemSymbol(f"{self.TAG}_nflag", offset=64 * q), k, R), value=1), [qz]))
        return self._gather_flags("n", "nflag", puts, A.R // TR, [start])

    def glue(self):
        """h = x + dequantised y, kept in the chip's L3, and the FFN's X from it, a tile a chip, into the root's X
        source: {pass: [the root's wait on its tiles' flags]}."""
        A, R, H = self.attn, self.ROOT, self.H
        puts = []
        for q in range(self.T // TR):
            k, i, m = self._tile(q)
            b = self._bufs(k, "g")
            # attn.y_scatter: the attention ends with one wait a chip, and this tile waits for its own chip's only
            ends = [t for t in self.attn_ends if t.assigned_chiplet_id == k] if A.y_scatter else list(self.attn_ends)
            after = ends + ([self.glue_go[k]] if hasattr(self, "glue_go") else [])
            cld = self._consts(k, "g", after)
            if A.y_scatter:
                if A.y_owner(q)[0] != k:
                    raise ValueError(f"attn.y_scatter: tile {q} of y lands on chip {A.y_owner(q)[0]:02x}, its glue runs on {k:02x}")
                # the tile's row of y, already in the chip's own L3 (D layout)
                ly = self._node(k, "dm", f"g_ldy_q{q}", "__snax_bingo_kernel_idma_1d_copy",
                                SnaxBingoKernelIdma1dCopyArgs(A.yt_sym(q), b["d"], self.TB), after)
            else:
                # y's rows of this tile: in each of the pass's slices (cluster half by half), its row block
                ly = self._node(k, "dm", f"g_ldy_q{q}", "__snax_bingo_kernel_idma_2d_copy",
                                SnaxBingoKernelIdma2dCopyArgs(self._on(A.y_sym(i * A.YB + m * TR * A.SLC * 2), k, R),
                                                              b["d"], TR * A.SLC * 2, A.SLB, TR * A.SLC * 2, H // A.SLC),
                                after)
            lx = self._node(k, "dm", f"g_ldx_q{q}", "__snax_bingo_kernel_idma_1d_copy",
                            SnaxBingoKernelIdma1dCopyArgs(self._x16(k, q), b["b"], self.TB), after)
            cv = self._node(k, "xdma", f"g_rows_q{q}", "__snax_bingo_kernel_xdma_6d",
                            convert_args("D", "row_major", TR, H, cm.MESH, 2, b["d"], b["a"]), [ly])
            dq = self._node(k, "simd", f"g_deq_q{q}", "__snax_bingo_kernel_simd_mul_tiles",
                            SnaxBingoKernelSimdMulTilesArgs(b["a"], b["c"].view(self.CB), b["a"], TR, H // 128, 2 * H, 2 * H),
                            [cv, cld])
            ad = self._node(k, "simd", f"g_add_q{q}", "__snax_bingo_kernel_simd_stream_elementwise",
                            SnaxBingoKernelSimdAddF16Args(b["a"], b["b"], b["b"], self.TB // 64, 32), [dq, lx])
            sh = self._node(k, "dm", f"g_sth_q{q}", "__snax_bingo_kernel_idma_1d_copy",
                            SnaxBingoKernelIdma1dCopyArgs(b["b"], BingoMemSymbol(f"{self.TAG}_h"), self.TB), [ad])
            qz = self._rms_q(k, f"g_q{q}", b["b"], b["c"].view(2 * self.CB), self.s_h.inv, b, [ad])
            if self.ffn is not None:
                F = self.ffn
                dst = self._on(F._xsrc(i * F.XB + m * TR * H), k, R)
            else:
                dst = self._on(BingoMemSymbol(f"{self.TAG}_x8", offset=q * TR * H), k, R)
            puts.append(self._node(k, "dm", f"g_put_q{q}", "__snax_bingo_kernel_idma_put_flagged",
                                   SnaxBingoKernelIdmaPutFlaggedArgs(
                                       b["x8"], dst, TR * H,
                                       self._on(BingoMemSymbol(f"{self.TAG}_gflag", offset=64 * q), k, R), value=1),
                                   [qz, sh]))
        return self._gather_flags("g", "gflag", puts, A.R // TR, list(self.attn_ends))

    def final(self, after):
        """The layer's output, a tile a chip: the FFN's raw Y rows dequantised, plus h, into the chip's L3. Returns the
        stores."""
        F, R, H = self.ffn, self.ROOT, self.H
        w = F.DNC
        out = []
        for q in range(self.T // TR):
            k, i, m = self._tile(q)
            b = self._bufs(k, "f")
            # ffn.y_scatter: the FFN ends with one wait a chip, and this tile waits for its own chip's only
            mine = [t for t in after if t.assigned_chiplet_id == k] if F.y_scatter else list(after)
            cld = self._consts(k, "f", mine)
            if F.y_scatter:
                if F.y_owner(q)[0] != k:
                    raise ValueError(f"ffn.y_scatter: tile {q} of Y lands on chip {F.y_owner(q)[0]:02x}, its final step "
                                     f"runs on {k:02x}")
                # the tile's row of Y_raw, already in the chip's own L3 (D layout)
                ly = self._node(k, "dm", f"f_ldy_q{q}", "__snax_bingo_kernel_idma_1d_copy",
                                SnaxBingoKernelIdma1dCopyArgs(F.yt_sym(q), b["d"], self.TB), mine)
            else:
                ly = self._node(k, "dm", f"f_ldy_q{q}", "__snax_bingo_kernel_idma_2d_copy",
                                SnaxBingoKernelIdma2dCopyArgs(self._on(F._ysym(i * F.YB + m * TR * w * 2), k, R), b["d"],
                                                              TR * w * 2, F.ROWS * w * 2, TR * w * 2, H // w), mine)
            lh = self._node(k, "dm", f"f_ldh_q{q}", "__snax_bingo_kernel_idma_1d_copy",
                            SnaxBingoKernelIdma1dCopyArgs(BingoMemSymbol(f"{self.TAG}_h"), b["b"], self.TB), mine)
            cv = self._node(k, "xdma", f"f_rows_q{q}", "__snax_bingo_kernel_xdma_6d",
                            convert_args("D", "row_major", TR, H, cm.MESH, 2, b["d"], b["a"]), [ly])
            dq = self._node(k, "simd", f"f_deq_q{q}", "__snax_bingo_kernel_simd_mul_tiles",
                            SnaxBingoKernelSimdMulTilesArgs(b["a"], b["c"].view(3 * self.CB), b["a"], TR, H // 128, 2 * H, 2 * H),
                            [cv, cld])
            ad = self._node(k, "simd", f"f_add_q{q}", "__snax_bingo_kernel_simd_stream_elementwise",
                            SnaxBingoKernelSimdAddF16Args(b["a"], b["b"], b["b"], self.TB // 64, 32), [dq, lh])
            out.append(self._node(k, "dm", f"f_st_q{q}", "__snax_bingo_kernel_idma_1d_copy",
                                  SnaxBingoKernelIdma1dCopyArgs(b["b"], BingoMemSymbol(f"{self.TAG}_out"), self.TB), [ad]))
        return out

    # ---- the layer ----------------------------------------------------------------------------------------------
    def build(self):
        self.attn.build()
        self.attn.x_ready.check_made("input norm")
        if "glue" in self.phases:
            p = self.phase_boundary("glue, FFN")
            self.glue_go = {k: self.loaders[(p, k, 0)] for k in self.chips}
            self.ffn_go = dict(self.glue_go)
            self.ffn_go[self.ROOT] = self.probe(self.ROOT, "ffn_go", [self.loaders[(p, self.ROOT, c)] for c in range(self.NC)])
        else:
            self.ffn_go = {k: self.probe(k, f"ffn_go{k:02x}", list(self.attn_ends)) for k in self.chips}
        ready = self.glue()
        if self.ffn is None:
            self.end_window(ready.every())
            self.run_checks(self.part_checks + self.glue_checks())
            return
        self.ffn.x_ready = ready
        self.ffn.build()
        ready.check_made("glue")

    def glue_checks(self):
        """Without the FFN: X (the FFN's INT8 input) on the root, head, middle and tail."""
        n = 2048
        x8 = self.x8_ffn.astype(np.int8).view(np.uint8).reshape(-1)
        return [(f"x8_{nm}", BingoMemSymbol(f"{self.TAG}_x8", offset=off, chip_id=self.ROOT), x8[off:off + n], n)
                for nm, off in (("a", 0), ("m", x8.size // 2), ("b", x8.size - n))]

    def final_checks(self):
        """The output's tile (row major FP16) on the first, a middle and the last chip: the head and the tail of each."""
        n = 2048
        out = []
        for q in (0, self.T // TR // 2, self.T // TR - 1):
            k = self.chips[q]
            img = self.out16[q * TR:(q + 1) * TR].astype(np.float16).view(np.uint8).reshape(-1)
            out += [(f"out_q{q}{nm}", BingoMemSymbol(f"{self.TAG}_out", offset=off, chip_id=k), img[off:off + n], n)
                    for nm, off in (("a", 0), ("b", self.TB - n))]
        return out

    # ---- emission -----------------------------------------------------------------------------------------------
    def emit_ring_schedules(self):
        for part in (self.attn, self.ffn):
            if part is not None:
                part.emit_ring_schedules()

    def data_h_extra(self):
        return self.st.lines()

    def finish(self, app_name, x_prefixes=("x_wait_", "x_pull_", "x_copy_")):
        self._assign_phases()
        dev = [self.roles[r] for r in ("gemm", "simd", "xdma", "dm")]
        GemmQueue.of(self.dfg, True).program_order(self.dfg, dev, tag=self.TAG)
        if self.ffn is not None and self.ffn.p.get("gemm_reuse"):
            # params ffn.gemm_reuse: the FFN's blocking GEMMs that repeat their core's previous configuration run as
            # gemm_minimal (the block's own finish, which the layer's replaces, would do it)
            print(f"[{self.TAG}] gemm_reuse: {reuse_gemm_config(self.dfg)} GEMMs run as gemm_minimal")
        self.dfg.split_chip_functions = True
        self.st.finish()
        super().finish(app_name, x_prefixes)


def _app_name(S):
    return (f"Llama-3-8B layer (synthetic weights), fresh {S.T}-token prompt, {len(S.chips)} chips x {S.NC} clusters"
            + ("" if S.ffn else ", attention and glue only"))


def run():
    run_driver("llama3 whole-layer prefill workload generator", LayerBuild, _app_name)
