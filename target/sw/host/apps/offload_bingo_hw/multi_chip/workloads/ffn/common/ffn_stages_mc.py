# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""A dense FFN prefill (ffn_prefill_mc.PBuild) split into stages that each run as their own app
(workloads/ffn/<model>_prefill/stage<N>/, params stage), the method of the FA and MoE blocks: a stage's inputs are in
place before the window (in L1, copied from an image in the memory chips' HBM), its outputs are checked, so its window
is the stage alone. The stages follow the FFN Prefill Tracker's "The stages, step by step":

  "0"     X from HBM: the root's host queues its memory chip's push engine, which multicasts every pass's X a tile at a
          time into every chip's L3, each tile behind a flag; every cluster copies the first pass's X into L1, block by
          block behind its tiles' flags. No weight chunk is pushed. The window ends when every cluster holds the first
          pass's X. Checks: X in L1 on the root's clusters and two others, the last pass's X in the last chip's L3.
  "123"   one pass (tokens = rows): X in every cluster's L1 before the window; gate|up GEMMs over the weight stream (1),
          dequantise and SwiGLU (2), the reshape D -> A, the quantise and the A8 exchange between a chip's clusters (3).
          The window ends when every cluster holds its chip's whole A8. Checks: the chip's A8 on the same clusters.
  "45"    one pass: the chip's A8 in every cluster's L1 before the window; the down slices over the weight stream (4),
          the ChainGathers into the root and the stores (5). The window ends with Y_raw in the root's L3. Checks: Y_raw.
  "full"  (default) the whole block, PBuild.build: every pass, X from HBM.

The other params are ffn_prefill_mc.py's; a stage's workload uses its full workload's params (stages 123 and 45 with
tokens = rows: one pass).
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ffn_prefill_mc as fp  # noqa: E402
from bingo_kernel_args import SnaxBingoKernelIdma1dCopyArgs  # noqa: E402
from driver import run_driver  # noqa: E402

STAGES = ("0", "123", "45", "full")
CHK = 2048                                  # bytes a checked window


class FStages(fp.PBuild):

    def __init__(self, args, p, plat, hw, shared=None):
        self.stage = str(p.get("stage", "full"))
        if self.stage not in STAGES:
            raise ValueError(f"params stage={self.stage!r}: one of {STAGES}")
        if p.get("x_device"):
            raise ValueError("the stage workloads take X from HBM (params x_hbm)")
        super().__init__(args, p, plat, hw, shared=shared)
        if self.stage in ("123", "45") and self.NI != 1:
            raise ValueError(f"params stage {self.stage}: one pass, tokens = rows (not {self.T} = {self.NI} x {self.ROWS})")
        self.h_in = {}
        if self.stage == "0":                           # X alone: neither half of a pass
            self.do_up = self.do_down = False
        elif self.stage == "123":
            self.do_down = False
            for m in sorted(set(self.mem.values())):    # X of the pass on every memory chip, for the clusters it feeds
                self.h_in[m] = self.st.put_hbm(f"{self.TAG}_s123x_m{m:02x}", self.x_a().reshape(-1),
                                               mem_chip=(m >> 4, m & 15))
        elif self.stage == "45":
            self.do_up = False
            for k in self.chips:                        # the chip's A8 (both clusters' columns, A layout)
                kc0, kn = self.kcols[k]
                a8 = self.lay.to_a(self.d.a8[:self.ROWS, kc0:kc0 + kn], self.d.mesh).astype(np.int8)
                m = self.mem[k]
                self.h_in[k] = self.st.put_hbm(f"{self.TAG}_s45a8_k{k:02x}", a8.reshape(-1), mem_chip=(m >> 4, m & 15))

    def checked(self):
        """The clusters whose L1 the checks read: the root's two, the middle chip's cluster 0, the last chip's cluster 1."""
        k = self.chips
        return [(k[0], 0), (k[0], 1), (k[len(k) // 2], 0), (k[-1], 1)]

    def pre_window(self):
        """The full block's (the scales into L1) and the stage's inputs, each cluster's from its memory chip's HBM."""
        pre = super().pre_window()
        dm = self.roles["dm"]
        for g in range(self.G if self.stage in ("123", "45") else 0):
            k, c = self.chip(g), self.c(g)
            if self.stage == "123":
                t = self.node(k, c, dm, f"x_pre_g{g}", "__snax_bingo_kernel_idma_1d_copy",
                              SnaxBingoKernelIdma1dCopyArgs(self.h_in[self.mem[k]].address, self.xbuf[g][0], self.XB), [])
                self.x_pre[g] = t
            else:
                t = self.node(k, c, dm, f"a8_pre_g{g}", "__snax_bingo_kernel_idma_1d_copy",
                              SnaxBingoKernelIdma1dCopyArgs(self.h_in[k].address, self.a8[g][0],
                                                            self.ROWS * self.kcols[k][1]), [])
                self.a8_pre[g] = t
            pre[k].append(t)
        return pre

    def build(self):
        if self.stage == "0":
            return self._build_0()
        super().build()
        if self.stage == "123":
            self.end_window(self.up_ends)
            items = []
            for k, c in self.checked():
                g = self.gs(k)[c]
                kc0, kn = self.kcols[k]
                a8 = self.lay.to_a(self.d.a8[:self.ROWS, kc0:kc0 + kn], self.d.mesh)
                items += [(k, c, self.a8[g][0], a8), (k, c, self.a8[g][0].view(a8.size - CHK), a8.reshape(-1)[-CHK:])]
            self.run_checks([self.gather_check("a8", items, CHK)])

    def _build_0(self):
        self.sym_zero = bool(self.p.get("sym_zero", True))
        go = self.setup()
        self.x_bc = self.x_hbm_bcasts(go)
        ends = []
        for g in range(self.G):
            prev = [go[self.chip(g)]]
            for b in range(self.x_blocks):          # the first pass's blocks, one after the other on the DM core
                prev = [self.x_wait_block(g, b, self.x_bc[0], prev)]
            ends += prev
        R = self.ROOT
        ends.append(self.probe(R, "x_bcast_done", [self.x_bc[-1]]))
        self.end_window(ends)
        x0 = self.x_a()[:self.XB]
        items = []
        for k, c in self.checked():
            xb = self.xbuf[self.gs(k)[c]][0]
            items += [(k, c, xb, x0), (k, c, xb.view(self.XB - CHK), x0[-CHK:])]
        last = self.x_a()[-CHK:].view(np.uint8)
        k = self.chips[-1]
        checks = [self.gather_check("x", items, CHK),
                  ("x_land_last", self._xland(self.NI * self.XB - CHK, chip_id=k), last, CHK)]
        self.run_checks(checks)

    def finish(self, app_name, x_prefixes=("x_wait_",)):
        # a stage without X fetches: its preload is the first thing every DM core runs, ahead of any weight load
        super().finish(app_name, {"123": ("x_pre_",), "45": ("a8_pre_",)}.get(self.stage, x_prefixes))


def _app_name(S):
    return (f"{S.p.get('model', 'dsv2-lite-l0')} FFN prefill, stage {S.stage}, {S.T} tokens in {S.NI} x {S.ROWS}, "
            f"{len(S.chips)} chips x {S.NC} clusters, X from HBM")


def run():
    run_driver("ffn prefill stage workload generator", FStages, _app_name)
