# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""Llama-3-8B's attention (llama3_attn_mc.py) split into stages that each run as their own app, the method of the DSv2 FA
block's fa_stages_mc.py: a stage's inputs are in place before the window, its outputs are checked, so its window is the
stage alone. params stage:

  "1234"  every pass's x multicast from the root and fetch into L1, its projections (q; the cluster's half of k or v) and
          the K/V trade inside each chip pair; the window ends when every cluster's query tiles of every pass and its key
          and value tiles are in L1
  "5"     every pass's sweep; every cluster's query tiles of every pass and its key and value tiles of the whole prompt
          in L1 before the window; the window ends when every pass's o~ has been copied out of the sweep's buffers into
          the chip's L3
  "678"   every pass's o~ -> o8, the root's gather of every head's o8 and its multicast, W_O by columns and the y stores;
          every cluster's o~ of every pass in L1 before the window; the window ends with y on the root
  "full"  (default) the whole attention, ABuild.build

The other params are llama3_attn_mc.py's.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import llama3_attn_mc as la  # noqa: E402
from llama3_attn_mc import BR, ABuild, run_driver  # noqa: E402
from bingo_kernel_args import SnaxBingoKernelIdma1dCopyArgs, SnaxBingoKernelWaitFlagsArgs  # noqa: E402
from bingo_mem_handle import BingoMemSymbol  # noqa: E402
from libs.blocks import fa_sweep  # noqa: E402


class LStages(ABuild):
    """ABuild's block stage by stage (the module's docstring), in ABuild's four steps: __init__ (the parameters),
    mem_handles() (the data and the memory, no task), build() (the stage's task graph) and finish() (the compile)."""

    def __init__(self, args, p, plat, hw, shared=None):
        """The parameters: the stage, then ABuild's."""
        self.stage = str(p.get("stage", "full"))
        if self.stage not in ("full", "1234", "5", "678"):
            raise ValueError(f"params stage={self.stage!r}: full, 1234, 5 or 678")
        super().__init__(args, p, plat, hw, shared=shared)

    def mem_handles(self):
        """ABuild's data and memory, then the stage's inputs in the memory chips' HBM (stages 5 and 678) and, for
        stage 5, the L3 area its outputs are copied into."""
        super().mem_handles()
        if self.stage == "5":
            self._stage5_images()
        elif self.stage == "678":
            self._stage678_images()

    # ---- the inputs of a stage alone, in the memory chips' HBM ----------------------------------------------------------
    def _tiles(self, cl):
        """Cluster cl's key tiles (A layout, rope_pairs' column order) and value tiles (V^T, A layout) of the whole
        prompt, as kbuf and vbuf hold them."""
        d, m = self.d, self.group[self.chip(cl)]
        kt = [self.lay.to_a(d.k8[m][j * self.bc:(j + 1) * self.bc][:, self.perm], self.mesh).astype(np.int8)
              for j in range(self.NP * self.TPP)]
        vt = [self.lay.to_a(d.v8[m][j * self.bc:(j + 1) * self.bc].T, self.mesh).astype(np.int8)
              for j in range(self.NP * self.TPP)]
        return np.concatenate([t.reshape(-1) for t in kt]), np.concatenate([t.reshape(-1) for t in vt])

    def _q8_tiles(self, cl):
        """Cluster cl's query tiles of every pass, one after the other (A layout, rope_pairs' column order)."""
        d, h = self.d, self.head[cl]
        return np.concatenate([self.lay.to_a(d.q8[h][r0:r0 + BR][:, self.perm], self.mesh).astype(np.int8).reshape(-1)
                               for r0 in range(0, self.T, BR)])

    def _stage5_images(self):
        T = self.TAG
        self.h_in = {}
        for cl in range(self.N_CL):
            k = self.chip(cl)
            kt, vt = self._tiles(cl)
            m = self.mem[k]
            self.h_in[cl] = self.st.put_hbm(f"{T}_s5in_cl{cl}", np.concatenate([kt, vt, self._q8_tiles(cl)]),
                                           mem_chip=(m >> 4, m & 15))
        # every pass's o~ out of the sweep: per chip, its clusters' tiles, pass by pass (ot8_fused: o8, A layout)
        self.OTB = BR * self.HD * (1 if self.ot8_fused else 2)
        self.st.put_zeros(f"{T}_otout", "uint8_t", self.NP * self.NC * self.NQ * self.OTB)

    def _ot_tile(self, h, r0):
        """Query tile r0's o~ of head h as the sweep leaves it: FP16 rows, or (ot8_fused) o8 in the A layout."""
        d = self.d
        if self.ot8_fused:
            return self.lay.to_a(d.ot8[h][r0:r0 + BR], self.mesh).astype(np.int8).reshape(-1).view(np.uint8)
        return np.asarray(d.ot16[h][r0:r0 + BR], dtype=np.float16).reshape(-1).view(np.uint8)

    def _stage678_images(self):
        T = self.TAG
        self.OTB = BR * self.HD * (1 if self.ot8_fused else 2)
        self.h_in = {}
        for cl in range(self.N_CL):
            k, h = self.chip(cl), self.head[cl]
            m = self.mem[k]
            ot = np.concatenate([self._ot_tile(h, r0) for r0 in range(0, self.T, BR)])     # tile by tile, pass by pass
            self.h_in[cl] = self.st.put_hbm(f"{T}_s678in_cl{cl}", ot.view(np.int8), mem_chip=(m >> 4, m & 15))

    def _consts(self):
        pre = {k: [] for k in self.chips}
        for k in self.chips:
            for c in reversed(range(self.NC)):
                cl = self.chips.index(k) * self.NC + c
                h, nb = self.cst[cl]
                pre[k] = [self.copy(cl, f"const_ld", h.address, self.consts[cl], nb, pre[k])]
        return pre

    def _init(self):
        self.slot, self.tile_ready, self.rope_ld, self._rope_war, self._o8_war = {}, {}, {}, {}, {}
        self._first_ld, self._proj_lds, self._x_first, self._x_last, self._last_sm = {}, {}, {}, {}, {}
        self._post_first, self._x_xdma = {}, {}
        self._ofull_war, self._ywo_war, self._y_last, self._yt_puts, self._y_first = {}, {}, [], {}, {}
        for ls in self.ls:
            ls.strict = True

    def _end_per_chip(self, last):
        """One probe a chip behind its tasks in `last` (the root's last), then the end stamp."""
        R = self.ROOT
        ends = [self.probe(k, f"fin{k:02x}", last[k]) for k in self.chips if k != R]
        ends.append(self.probe(R, f"fin{R:02x}", last[R]))
        self.end_window(ends)

    # ---- the stages --------------------------------------------------------------------------------------------------
    def build(self):
        if self.stage == "full":
            return super().build()
        getattr(self, f"build_{self.stage}")()

    def build_1234(self):
        R = self.ROOT
        self._init()
        go = self.setup(extra=self._consts())
        self.dfg.relay_one_consumer_cores = [self.roles["xdma"]]
        puts = self.x_puts_up_front(go)
        last_proj, q8s, last = {}, {}, {k: [] for k in self.chips}
        for i in range(self.NP):
            if i:
                self.pass_start(i)
            x = {cl: self.stage_x(cl, i, puts[i], [go[self.chip(cl)]] + list(last_proj.get(cl, [])))
                 for cl in range(self.N_CL)}
            own = {}
            for cl in range(self.N_CL):
                q8s[(cl, i)], last_proj[cl], own[cl] = self.stage_proj(cl, i, x[cl])
            if i + 1 < self.NP and not self.x_mem:
                puts.append(self.stage_x_put(i + 1, [go[R]] + puts[-1:]))
            if self.kv_split:
                self.stage_trade_kv_split(i, own)
            else:
                for k in self.chips:
                    self.stage_trade_kv(k, i, own)
            for cl in range(self.N_CL):
                k = self.chip(cl)
                last[k] += [t for u in range(self.NQ) for t in q8s[(cl, i)][u][1]]
                last[k] += [self.tile_ready[(cl, kind,
                                             i * self.TPP + t)] for kind in ("K", "V") for t in range(self.TPP)]
        self._end_per_chip({k: list(dict.fromkeys(v)) for k, v in last.items()})
        # checks: the last pass's first and last query tile on both clusters of the root and of the fold order's middle
        # chip; every cluster of both chips' key and value tiles of the whole prompt
        d, checks = self.d, []
        far = self.order[len(self.order) // 2]
        i = self.NP - 1
        for k in (R, far):
            for c in range(self.NC):
                cl = self.chips.index(k) * self.NC + c
                h = self.head[cl]
                for u in sorted({0, self.NQ - 1}):
                    r0 = i * self.R + u * BR
                    q8 = self.lay.to_a(d.q8[h][r0:r0 + BR][:, self.perm], self.mesh).astype(np.int8)
                    nm = f"q8_k{k:02x}c{c}u{u}"
                    checks.append((nm, self.stash(nm, k, c, self.pb(cl, i)["q8"][u], q8.size, []), q8.view(np.uint8),
                                   q8.size))
                kt, vt = self._tiles(cl)
                for nm, buf, img in ((f"k8_k{k:02x}c{c}", self.kbuf[cl], kt), (f"v8_k{k:02x}c{c}", self.vbuf[cl], vt)):
                    checks.append((nm, self.stash(nm, k, c, buf, img.size, []), img.view(np.uint8), img.size))
        self.run_checks(checks)

    def build_5(self):
        T, R = self.TAG, self.ROOT
        self._init()
        pre = self._consts()
        nk = self.NP * self.TPP * self.TILE_B
        qb = BR * self.HD
        for k in self.chips:
            for c in range(self.NC):
                cl = self.chips.index(k) * self.NC + c
                a = self.h_in[cl].address
                pre[k] = [self.copy(cl, "kt_pre", a, self.kbuf[cl], nk, pre[k])]
                pre[k] = [self.copy(cl, "vt_pre", a + nk, self.vbuf[cl], nk, pre[k])]
                for i in range(self.NP):
                    for u in range(self.NQ):
                        off = 2 * nk + (i * self.NQ + u) * qb
                        pre[k] = [self.copy(cl, f"q8_pre_i{i}u{u}", a + off, self.pb(cl, i)["q8"][u], qb, pre[k])]
        go = self.setup(extra=pre)
        self.dfg.relay_one_consumer_cores = [self.roles["xdma"]]
        for cl in range(self.N_CL):
            for kind in ("K", "V"):
                for j in range(self.NP * self.TPP):
                    self.tile_ready[(cl, kind, j)] = go[self.chip(cl)]
        prev_sw, xt_rd, last = {}, {}, {k: [] for k in self.chips}
        for i in range(self.NP):
            if i:
                self.pass_start(i)
            for cl in range(self.N_CL):
                k, c = self.chip(cl), self.local(cl)
                q8 = {u: (self.pb(cl, i)["q8"][u], [go[k]]) for u in range(self.NQ)}
                sw = fa_sweep(self.at(k, c), self.sweep_cfg(i + 1), self.sweep_bufs[cl], la.L1Tiles(self, cl), q8,
                              self.roles, prev=prev_sw.get(cl), oacc_free=xt_rd.get(cl), name=f"s{i}_", early=1,
                              mask=self.mask(i)(cl), mask_c=self.mask_c, ot8=self.ot8_bits(), geom_core=self.sm_geom,
                              sm_batch=self.sm_batch)
                outs = [self.copy(cl, f"ot_out{u}_i{i}", sw["xt"][u],
                                  BingoMemSymbol(f"{T}_otout", offset=((i * self.NC + c) * self.NQ + u) * self.OTB),
                                  self.OTB, [sw["tr"][u]]) for u in range(self.NQ)]
                xt_rd[cl] = {u: [outs[u]] for u in range(self.NQ)}
                prev_sw[cl] = sw
                last[k] += outs
        self._end_per_chip(last)
        # checks: the last pass's first and last query tile's o~ (FP16 rows) on both clusters of the root and of the fold
        # order's middle chip, in their chip's L3
        d, checks = self.d, []
        far = self.order[len(self.order) // 2]
        i = self.NP - 1
        for k in (R, far):
            for c in range(self.NC):
                cl = self.chips.index(k) * self.NC + c
                h = self.head[cl]
                for u in sorted({0, self.NQ - 1}):
                    r0 = i * self.R + u * BR
                    ot = self._ot_tile(h, r0)
                    off = ((i * self.NC + c) * self.NQ + u) * self.OTB
                    checks.append((f"ot_k{k:02x}c{c}u{u}", BingoMemSymbol(f"{T}_otout", offset=off, chip_id=k), ot,
                                   self.OTB))
        self.run_checks(checks)

    def build_678(self):
        R, dm = self.ROOT, self.roles["dm"]
        if not self.wo_cols:
            raise ValueError("params stage 678: W_O by columns (wo_cols) only")
        if self.wo_under_proj:
            raise ValueError("params stage 678: no projections to run under W_O (wo_under_proj)")
        self._init()
        pre = self._consts()
        ot_buf = {}
        for k in self.chips:
            for c in range(self.NC):
                cl = self.chips.index(k) * self.NC + c
                a = self.h_in[cl].address
                for i in range(self.NP):
                    for u in range(self.NQ):
                        buf = self.cx_chip(k).at(c).l1(f"{self.TAG}_otp{u}_i{i}", self.OTB)
                        off = (i * self.NQ + u) * self.OTB
                        pre[k] = [self.copy(cl, f"ot_pre_i{i}u{u}", a + off, buf, self.OTB, pre[k])]
                        ot_buf[(cl, i, u)] = buf
        go = self.setup(extra=pre)
        self.dfg.relay_one_consumer_cores = [self.roles["xdma"]]
        stores = []
        x = {cl: go[self.chip(cl)] for cl in range(self.N_CL)}
        for i in range(self.NP):
            if i:
                self.pass_start(i)
            post_last = {}
            for cl in range(self.N_CL):
                k = self.chip(cl)
                sw = {"xt": [ot_buf[(cl, i, u)] for u in range(self.NQ)], "tr": [go[k]] * self.NQ}
                _, post_last[cl] = self.stage_post(cl, i, sw)
            self.stage_wo_cols(i, post_last, x, stores, None)
        if self.y_scatter:
            self.end_window(self._y_scatter_waits())
        else:
            yw = self.node(R, 0, dm, "y_wait", "__snax_bingo_kernel_wait_flags",
                           SnaxBingoKernelWaitFlagsArgs(BingoMemSymbol(f"{self.TAG}_yflag"), self.NP * self.N_CL, 1),
                           [t for t in self._y_last if t.assigned_chiplet_id == R])
            for t in self._y_last:
                if t.assigned_chiplet_id != R:
                    self.dfg.bingo_add_order_edge(t, yw)
            self.end_window([yw])
        self.checks_kind = "y"
        self._checks()

    def finish(self, app_name, x_prefixes=("x_wait_", "x_pull_", "x_copy_")):
        # a stage without x: its preload is the first thing every DM core runs, ahead of any weight load
        prefixes = {"5": ("kt_pre",), "678": ("ot_pre_",)}.get(self.stage, x_prefixes)
        super().finish(app_name, prefixes)


def run():
    run_driver("llama3 attention stages workload generator", LStages, la._app_name)
