#!/usr/bin/env python3
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""DeepSeek-V2-Lite's layer-1 MLA attention, stage 5 of the 256-token prefill on sixteen chiplets alone: the sweep
over the keys.

Before the window every chip copies every piece of the keys and their landed flags (stage 3's result) from its
memory chip's HBM into its L3, and every cluster its query tiles Q8 (stage 4's result) and its constants (the causal
masks are in them) into L1. Per chunk and cluster (in every odd chunk the clusters take the other halves: the late half
sweeps more tiles): fa_sweep over the key tiles up to the chunk's end, no tile whose every score is masked: per tile
its K and V from L3 (K in slab 0, V in slabs 1 and 2 in turn), S^T = K Q^T with the causal masks as the score matmul's
INT32 bias, the online softmax, O^T += V^T P^T, then o~ = O / l quantised straight into W_UV's A operand (INT8).

          before the window     const_ld, kv_pre, kvf_pre, kvnf_pre, q8_pre_i<i>q<q>    DM: HBM -> L3 / L1
          S0_start
          every cluster, chunk i: kv_wait_i<i> (the keys' flags), then fa_sweep's tasks s<i>_*: Ld_K<j>_i<i>,
                                  Ld_V<j>_i<i> (DM), the QK and PV matmuls, the softmax and the normalisation
          P_fin<chip> -> S1_end -> fp_report, the checks

Checks: o~ of the first and the last query tile of both clusters of the chip in the middle of the fold order, the last
chunk.

params.hjson: the platform, the tokens, the chunk and key-tile sizes, the weight chunk and ring sizes, the weights'
width."""

import dataclasses
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from stage_common import BR, D_QK, D_V, KB, KVP, BingoMemSymbol, StageBase, app_name  # noqa: E402
from bingo_kernel_args import SnaxBingoKernelIdma2dCopyArgs, SnaxBingoKernelWaitFlagsArgs  # noqa: E402
from driver import run_driver  # noqa: E402
from libs.blocks import fa_sweep  # noqa: E402


class FreshTileSource:
    """fa_sweep's tiles: key and value tile j of chunk i from the chip's own L3, where the keys' multicasts leave the
    16-token pieces [key rows | value columns] side by side, behind one wait a sweep on the flags of the pieces it
    reads. A key tile is its pieces' key rows one after the other (a 2D copy) into slab 0; a value tile is the A layout
    of [512, bc], each piece's 256-byte runs one 16-row block apart (a 3D copy), into slabs 1 and 2 in turn."""

    def __init__(self, fb, cl, i, nt):
        self.fb, self.cl, self.i, self.nt = fb, cl, i, nt
        self.wait = None

    def fetch(self, kind, j):
        fb = self.fb
        k, c = fb.chip(self.cl), fb.local(self.cl)
        if self.wait is None:
            self.wait = fb.kv_wait(self.cl, self.i)
        n = fb.bc // 16
        src = BingoMemSymbol(f"{fb.TAG}_kv", offset=j * n * KVP + (KB if kind == "V" else 0))
        if kind == "K":
            si, slab, war = fb.ls[self.cl].take_at(0, fb.bc * D_QK)
            args = SnaxBingoKernelIdma2dCopyArgs(src, slab, KB, KVP, KB, n)
        else:
            si, slab, war = fb.ls[self.cl].take_at(1 + j % 2, D_V * fb.bc)
            args = SnaxBingoKernelIdma2dCopyArgs(src, slab, 256, 256, 16 * fb.bc, D_V // 16, outer=n, src_outer=KVP,
                                                 dst_outer=256)
        ld = fb.node(k, c, fb.roles["dm"], f"Ld_{kind}{j}_i{self.i}", "__snax_bingo_kernel_idma_2d_copy", args,
                     war + [self.wait])
        return ld, slab, si

    def release(self, token, task):
        self.fb.ls[self.cl].release(token, task)


class Stage5(StageBase):
    """Stage 5 alone (the module docstring)."""
    STAGE = 5
    X_HBM = False
    KVA_IN_RING = 0
    PER_CHUNK = 0                                  # its key and value tiles come from L3
    ROW_SWAP = True
    SKIP_MASKED = True

    def stage_inputs(self):
        """Every chip's query tiles (its head, both chunks, both clusters: the reference's Q8, A layout) and every
        piece of the keys with every chunk's and every piece's landed flag, in its memory chip's HBM."""
        d, T, n = self.d, self.TAG, self.T // 16
        kv = np.concatenate([np.asarray(d.kv_piece(16 * q)).view(np.uint8).reshape(-1) for q in range(n)])
        flags = np.zeros((self.NCH + n) * 64, dtype=np.uint8)
        flags.view(np.uint32).reshape(-1, 16)[:, 0] = 1
        self.h_kvall, self.h_q8 = {}, {}
        for m in sorted(set(self.mem.values())):
            self.h_kvall[m] = self.st.put_hbm(f"{T}_kvall_m{m:02x}", np.concatenate([kv, flags]).view(np.int8),
                                              mem_chip=(m >> 4, m & 15))
        for k in self.chips:
            h = self.head[k]
            img = np.concatenate([d.q8_a(h, self.r0(i, c) + q * BR).view(np.uint8).reshape(-1)
                                  for i in range(self.NCH) for c in range(self.NC) for q in range(self.NQ)])
            m = self.mem[k]
            self.h_q8[k] = self.st.put_hbm(f"{T}_q8_k{k:02x}", img.view(np.int8), mem_chip=(m >> 4, m & 15))

    def build(self):
        self.last_ld = {}
        for ls in self.ls:
            ls.strict = True
        go = self.setup(extra=self.stage5_preload())
        self.relays()
        self.x_puts = []
        last = {k: [] for k in self.chips}
        self.stage_sweep(go, last)
        self.window_end(last)
        self.run_checks(self.xt_checks())

    def stage5_preload(self):
        """Before the window, per chip: every cluster's constants (the causal masks are in them), the keys and the flags
        into its L3, every cluster's query tiles into its L1."""
        T, n, qb = self.TAG, self.T // 16, D_QK * BR
        pre = {k: [] for k in self.chips}
        for k in self.chips:
            cl0 = self.chips.index(k) * self.NC
            for c in reversed(range(self.NC)):
                pre[k] = [self.dm_copy(cl0 + c, "const_ld", self.h_c[k].address, self.consts[cl0 + c],
                                       self.C_BYTES, pre[k])]
            src = self.h_kvall[self.mem[k]].address
            pre[k] = [self.dm_copy(cl0, "kv_pre", src, BingoMemSymbol(f"{T}_kv"), n * KVP, pre[k])]
            pre[k] = [self.dm_copy(cl0, "kvf_pre", src + n * KVP, BingoMemSymbol(f"{T}_kvflag"), self.NCH * 64, pre[k])]
            pre[k] = [self.dm_copy(cl0, "kvnf_pre", src + n * KVP + self.NCH * 64, BingoMemSymbol(f"{T}_kvnflag"),
                                   n * 64, pre[k])]
            for i in range(self.NCH):
                for c in range(self.NC):
                    cl = cl0 + c
                    for q in range(self.NQ):
                        off = ((i * self.NC + c) * self.NQ + q) * qb
                        pre[k] = [self.dm_copy(cl, f"q8_pre_i{i}q{q}", self.h_q8[k].address + off,
                                               self.bufs(cl, i)["q8"][q], qb, pre[k])]
        return pre

    # ---- the sweep ----------------------------------------------------------------------------------------------
    def stage_sweep(self, go, last):
        """Per chunk and cluster the sweep; the keys' flags are set before the window, so kv_wait's puts are the
        start."""
        R = self.ROOT
        self.kv_put = {q: go[R] for q in range(self.T // 16)}
        self.kv_mc = [go[R]] * self.NCH
        self.sw_of = {}
        for i in range(self.NCH):
            for cl in range(self.N_CL):
                k = self.chip(cl)
                q8 = {q: (self.bufs(cl, i)["q8"][q], [go[k]]) for q in range(self.NQ)}
                sw = self.sweep(cl, i, q8)
                self.sw_of[(cl, i)] = sw
                last[k] += list(sw["tr"])

    def sweep(self, cl, i, q8):
        """Cluster cl's sweep of chunk i over the key tiles up to the chunk's end (FreshTileSource), the causal masks on
        the scores, o~ quantised into W_UV's A operand; each softmax arena's geometry built on the cluster's xDMA core
        at the sweep's head."""
        k, c = self.chip(cl), self.local(cl)
        nt, plan = self.d.tile_plan(self.r0(i, c))
        early = 2 if len(self.ls[cl].slabs) >= 4 else 1
        return fa_sweep(self.at(k, c), dataclasses.replace(self.sweep_cfg, nt=nt), self.bufs(cl, i)["swp"],
                        FreshTileSource(self, cl, i, nt), q8, self.roles, name=f"s{i}_", early=early,
                        mask=self.mask_of(cl, plan), mask_c=True, ot8=self.fbits(self.d.s_ot.inv), geom_core="xdma")

    def mask_of(self, cl, plan):
        """fa_sweep's mask hook for cluster cl: a masked pair's pattern in the cluster's constants, else None."""
        return lambda j, q: (None if plan.get((j, q)) is None
                             else self.consts[cl].view(self.C_MASK + plan[(j, q)] * self.MASK_B))

    def kv_wait(self, cl, i):
        """Cluster cl's wait, before chunk i's sweep, until the keys of chunks 0 .. i have landed in its chip's L3: the
        root multicasts' flags, or on the root its pieces' own (set before the window)."""
        k, c, R, T = self.chip(cl), self.local(cl), self.ROOT, self.TAG
        if k == R:
            n = (i + 1) * (self.C // 16)
            puts = [self.kv_put[q] for q in range(n)]
            args = SnaxBingoKernelWaitFlagsArgs(BingoMemSymbol(f"{T}_kvnflag"), n, 1)
        else:
            puts = self.kv_mc[:i + 1]
            args = SnaxBingoKernelWaitFlagsArgs(BingoMemSymbol(f"{T}_kvflag"), i + 1, 1)
        w = self.node(k, c, self.roles["dm"], f"kv_wait_i{i}", "__snax_bingo_kernel_wait_flags", args,
                      [t for t in puts if self.flag_dep(t, k, c)])
        for t in puts:
            if not self.flag_dep(t, k, c):
                self.dfg.bingo_add_order_edge(t, w)
        return w

    # ---- checks -------------------------------------------------------------------------------------------------
    def xt_checks(self):
        """o~ of the first and the last query tile of both clusters of the chip in the middle of the fold order, the
        last chunk (the checks copy after the window, when an earlier chunk's buffers may hold the next chunk's)."""
        d = self.d
        far = self.order[len(self.order) // 2]
        h, i = self.head[far], self.NCH - 1
        checks = []
        for c in range(self.NC):
            cl = self.chips.index(far) * self.NC + c
            sw = self.sw_of[(cl, i)]
            for q in sorted({0, self.NQ - 1}):
                r0 = self.r0(i, c) + q * BR
                tag = f"i{i}c{c}q{q}"
                ot = d.layout.to_a(d.head(h)["ot8"][r0:r0 + BR], d.mesh).astype(np.int8).view(np.uint8)
                checks.append((f"xt_{tag}", self.stash(f"xt_{tag}", far, c, sw["xt"][q], D_V * BR, [sw["tr"][q]]),
                               ot, D_V * BR))
        return checks


if __name__ == "__main__":
    run_driver("fa prefill stage 5 workload generator", Stage5, app_name)
