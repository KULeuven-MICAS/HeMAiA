#!/usr/bin/env python3
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""Llama-3-8B's FFN (synthetic weights), stage 0 of the 256-token prefill on sixteen chiplets alone: every pass's X from
HBM into every cluster.

The root's host queues its memory chip's push engine, which multicasts every pass's X (INT8, A layout) a tile at a time
into every chip's L3, each tile behind a flag; every cluster then copies the first pass's X into L1, row block by row
block, each block behind its tile's flag. No weight is streamed. The window ends when every cluster holds the first
pass's X and the last multicast has been issued. Checks: X in L1 (its first and last 2 KiB) on the root's two clusters,
the middle chip's cluster 0 and the last chip's cluster 1, and the last pass's X in the last chip's L3.

params.hjson: the model, tokens and rows a pass (the whole block's), the platform and compile switches."""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(__file__).split("/target/sw/host/")[0]
sys.path.insert(0, os.path.join(REPO, "util", "sim", "multichip"))
sys.path.insert(0, os.path.join(REPO, "util", "sim", "common"))
sys.path.insert(0, os.path.join(REPO, "target", "sw", "host", "runtime", "libbingo", "mini_compiler"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "llama3", "common"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "common"))
import _bingo_paths  # noqa: F401,E402
from bingo_kernel_args import (  # noqa: E402
    HostBingoKernelHbmBcastArgs, SnaxBingoKernelIdma1dCopyArgs, SnaxBingoKernelIdmaFetchFlaggedArgs)
from bingo_mem_handle import BingoMemSymbol  # noqa: E402
from driver import run_driver  # noqa: E402

import chiplet_mc as cm  # noqa: E402
import ffn_data  # noqa: E402
import llama3_data  # noqa: E402

CACHE = os.path.join(REPO, "target", "sim", "automation", "sweep", "llm_blocks", "ffn", "cache")
CHK = 2048                                  # bytes a checked window
X_BLOCKS = 4                                # the first pass's X in row blocks, each behind its own tile flag


def checked(chips):
    """The clusters whose L1 the checks read: the root's two, the middle chip's cluster 0, the last chip's cluster 1."""
    return [(chips[0], 0), (chips[0], 1), (chips[len(chips) // 2], 0), (chips[-1], 1)]


def gather_check(S, name, items):
    """items [(chip, cluster, L1 source, golden)]: after the end stamp each cluster's DM core copies the first CHK
    bytes of its source into its own slice of one L3 symbol on the root, whose host compares the whole symbol in one
    check (the first bad offset // CHK names the item). Returns the check for run_checks."""
    sym = f"{S.TAG}_chk_{name}"
    S.st.put_zeros(sym, "uint8_t", len(items) * CHK)
    for j, (k, c, src, _) in enumerate(items):
        dst = BingoMemSymbol(sym, offset=j * CHK, chip_id=S.ROOT)
        S.check_copies.append(S.node(k, c, S.roles["dm"], f"chk_{name}_{j}", "__snax_bingo_kernel_idma_1d_copy",
                                     SnaxBingoKernelIdma1dCopyArgs(src, dst, CHK), [S.end]))
    gold = np.concatenate([np.ascontiguousarray(ref).view(np.uint8).reshape(-1)[:CHK] for *_, ref in items])
    return (name, BingoMemSymbol(sym, chip_id=S.ROOT), gold, len(items) * CHK)


class Stage0(cm.ChipletBuild):
    """Stage 0 of the FFN prefill block (../full), alone. The block runs, for each pass of `rows` tokens:

        stage 0     X from HBM into every cluster                                     this app
        stages 1-3  gate|up GEMMs, dequantise and SwiGLU, quantise, the A8 exchange   ../stage123
        stages 4-5  down GEMMs, the ChainGathers into the root, the stores of Y       ../stage45

    Nothing follows it here, so the window (S0_start .. S1_end) is stage 0's time. run_driver makes the program in
    four steps: __init__ (the parameters), mem_handles() (the data and the memory, no task), build() (the task graph:
    every task and edge) and finish() (the compile into the C header and the data image, in which the compiler orders
    every core's tasks)."""

    TAG = "pf"      # prefill FFN, as in the whole block: the prefix of the symbols, task names and printed lines

    def __init__(self, args, p, plat, hw):
        """The parameters; no data, memory or task yet. ChipletBuild.__init__ first: the platform's and the graph's
        switches, the chips, each one's memory chip and its push engine there, the empty DFG, the data staging (self.st)
        and the stamp buffer. Then this stage's: the model (params model), and the rows a pass (params rows), whole
        X_BLOCKS blocks of 16 rows dividing the tokens, so NI passes."""
        super().__init__(args, p, plat, hw)
        self.model = p["model"]
        R = self.ROWS = int(p["rows"])
        if R % (16 * X_BLOCKS) or self.T % R:
            raise ValueError(f"params tokens={self.T} rows={R}: rows whole {X_BLOCKS} blocks of 16, dividing tokens")
        self.NI = self.T // R

    def mem_handles(self):
        """The data and the memory; no task yet: the model's FFN data (FFNData) and every pass's X from it (quantised
        INT8, A layout) staged in the root's memory chip's HBM (pf_xhbm); in every chip's L3 the landing area for it
        (pf_xland) and a flag a tile (pf_xtflag); the flags' value in HBM (pf_one); every cluster's L1 buffer for the
        first pass's X (pf_x0)."""
        d = self.d = ffn_data.FFNData(CACHE, self.model, wbits=self.wbits, tokens=self.T)
        R, self.D = self.ROWS, d.H
        self.XB = R * self.D                    # X of a pass, INT8
        self.XT = self.XB // X_BLOCKS           # a multicast tile: a row block of X
        T, st, m = self.TAG, self.st, self.mem[self.ROOT]
        self.x_img = np.concatenate([d.layout.to_a(d.hq[i * R:(i + 1) * R]) for i in range(self.NI)]).astype(
            np.int8).reshape(-1)
        print(f"[{T}] {self.T} tokens in {self.NI} passes of {R}: X {self.NI} x {self.XB} B, {X_BLOCKS} tiles a pass")
        st.put_zeros(f"{T}_xland", "int8_t", self.NI * self.XB, preload=True)     # every chip's landing area
        self.h_xhbm = st.put_hbm(f"{T}_xhbm", self.x_img, mem_chip=(m >> 4, m & 15))
        st.put_zeros(f"{T}_xtflag", "uint8_t", 64 * self.NI * X_BLOCKS)          # a flag a tile
        one = np.zeros(8, dtype=np.uint64)
        one[0] = 1
        self.h_one = st.put_hbm(f"{T}_one", one.view(np.int8), mem_chip=(m >> 4, m & 15))   # the flags' value
        self.xbuf = {cl: self.cx(cl).l1(f"{T}_x0", self.XB) for cl in range(self.N_CL)}

    def build(self):
        """The task graph (task names without their _k<chip>_cl<c> suffix; i a pass, b a row block of X):

                                 S0_start                         the start stamp: root chip, cluster 0, DM core
                     _______________|________________
                    |                                |
             x_hbm_bcast_i0  - - - - - - - - > x_wait_i0b0        left: the root's host; right: every cluster's
                    |         (order edge)           |            DM core, one chain a cluster
             x_hbm_bcast_i1                    x_wait_i0b1
                    |                                |
                   ...                              ...
                    |                                |
             P_x_bcast_done                    x_wait_i0b3
                    |________________________________|
                                    |
                                  S1_end  ->  pf_report, chk_x_<n>  ->  Check_pf_x  ->  Check_pf_x_land_last

        x_hbm_bcast_i<i>: the root's host queues its memory chip's push engine to multicast pass i's X, a tile (a row
        block) at a time, into every chip's L3, each tile followed by its flag. x_wait_i0b<b>: each cluster's DM core
        spins on tile b's flag, then copies the tile into its L1. The order edges (from the first multicast to every
        wait) only dispatch the multicast ahead of the waits; the flag is the real dependency. P_x_bcast_done: a join
        point after the last multicast. After S1_end the root's host prints the window (pf_report), the checked
        clusters copy their X into the root's L3 (chk_x_<n>), and the root's host compares (Check_pf_*)."""
        T, R, dm = self.TAG, self.ROOT, self.roles["dm"]
        start = self.stamp("start")
        m = self.mem[R]
        # the root's host queues its memory chip's push engine, one task a pass: the pass's X a tile at a time into
        # every chip's L3, each tile behind its flag
        bc = []
        for i in range(self.NI):
            bc.append(self.cx_chip(R).host_node(
                f"x_hbm_bcast_i{i}_k{R:02x}", "__host_bingo_kernel_hbm_bcast",
                HostBingoKernelHbmBcastArgs(self.h_xhbm.address + i * self.XB,
                                            BingoMemSymbol(f"{T}_xland", offset=i * self.XB),
                                            BingoMemSymbol(f"{T}_xtflag", offset=64 * i * X_BLOCKS), self.h_one.address,
                                            1, self.XT, X_BLOCKS, m, self.engine[R]),
                after=[start] + bc[-1:]))
        # every cluster's DM core copies the first pass's blocks, one after the other, each behind its tile's flag (the
        # multicast's flag says the tile has landed: an order edge, not a dependency)
        ends = []
        for cl in range(self.N_CL):
            k, c = self.chip(cl), self.local(cl)
            prev = [start]
            for b in range(X_BLOCKS):
                off = b * self.XT
                t = self.node(k, c, dm, f"x_wait_i0b{b}", "__snax_bingo_kernel_idma_fetch_flagged",
                              SnaxBingoKernelIdmaFetchFlaggedArgs(BingoMemSymbol(f"{T}_xland", offset=off),
                                                                  self.xbuf[cl].view(off), self.XT,
                                                                  BingoMemSymbol(f"{T}_xtflag", offset=64 * b), 1, 1),
                              prev)
                self.dfg.bingo_add_order_edge(bc[0], t)
                prev = [t]
            ends += prev
        ends.append(self.probe(R, "x_bcast_done", [bc[-1]]))
        self.end_window(ends)
        x0 = self.x_img[:self.XB]
        items = []
        for k, c in checked(self.chips):
            xb = self.xbuf[self.clusters(k)[c]]
            items += [(k, c, xb, x0), (k, c, xb.view(self.XB - CHK), x0[-CHK:])]
        last = self.chips[-1]
        self.run_checks([gather_check(self, "x", items),
                         ("x_land_last", BingoMemSymbol(f"{T}_xland", offset=self.NI * self.XB - CHK, chip_id=last),
                          self.x_img[-CHK:].view(np.uint8), CHK)])

    def finish(self, app_name, x_prefixes=("x_wait_",)):
        self.dfg.split_chip_functions = True     # one host function a chip
        super().finish(app_name, x_prefixes)


def _app_name(S):
    return (f"{S.p['model']} FFN prefill, stage 0, {S.T} tokens in {S.T // S.ROWS} x {S.ROWS}, "
            f"{len(S.chips)} chips x {S.NC} clusters, X from HBM")


if __name__ == "__main__":
    llama3_data.register_ffn()
    run_driver("ffn prefill stage 0 workload generator", Stage0, _app_name)
