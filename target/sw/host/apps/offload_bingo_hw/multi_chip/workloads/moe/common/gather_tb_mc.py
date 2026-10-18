# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""The MoE prefill's reduce on its own (moe_prefill_mc.py, "reduce"): the ChainGathers of the down
projection's output slices, against the weight stream the down projection pulls through the rings.
A testbench for running cluster 0's and cluster 1's gathers at the same time.

PER SLICE j, PER CLUSTER g (chip k, cluster c), on the cluster's DM core, as the MoE runs it:
    partial    the cluster's partial of slice j into slot j of the cluster's symmetric buffer (the
               MoE's down projection writes it there; the MoE reuses two slots, here every slice
               has its own, so no gather is a remote predecessor of a later partial), from its
               memory chip's HBM (params partials: hbm) or from its own L3 (l3: the copies stay off
               the memory chips, which the weight stream keeps busy)
    loads      params loads_per_slice weight chunks from the cluster's ring into an L1 slab, each
               slab released at once (the MoE's down projection: 1 shared + 4 expert chunks a slice)
THEN, PER SLICE j, PER CLUSTER c, on the root's cluster c:
    reduce     one ChainGather through every chip's cluster c into the root (chain_order(c), the
               root's partial last), from slot j into slot j % 2 of the root's p1c[c], after every cluster c's
               partial and loads of slice j; then the root stores the slot to y in its L3

params chain_mode: concurrent (a gather after its own cluster's previous gather only, so cluster 0's
and cluster 1's run at the same time; each passes only its own cluster's xDMAs: the MoE's default) or
serial (every gather after the previous one, either cluster). Both clusters' chains run the ring the
same way: two chains crossing a D2D link in opposite directions at the same time can deadlock.

THE DATA: every partial value is a multiple of 1/4 between -2 and 2, so the 16 chips' sum is exact
in FP16 whatever the order; the golden still adds in the chain order. With partials l3 every chip
runs the same image, so every chip holds the same partials (a cluster's and a slice's still differ).
CHECKS: the first and the last 256 B of every slice of y.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "common"))     # workloads/common
from driver import repo_root, run_driver  # noqa: E402
REPO = repo_root(__file__)
sys.path.insert(0, f"{REPO}/target/sw/host/runtime/libbingo/mini_compiler")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "ffn", "common"))

import _bingo_paths  # noqa: F401,E402
from bingo_kernel_args import (  # noqa: E402
    SnaxBingoKernelIdma1dCopyArgs, SnaxBingoKernelXdmaChainGatherArgs)
from bingo_mem_handle import BingoMemFixedAddr, BingoMemSymbol  # noqa: E402

import chiplet_mc as cm  # noqa: E402

CHECK_BYTES = 256             # bytes checked at each end of a slice


class GatherTb(cm.ChipletBuild):
    TAG = "gt"

    def __init__(self, args, p, plat, hw):
        super().__init__(args, p, plat, hw)
        if self.crest or self.reduce_impl != "chain" or self.sync != "edge":
            raise ValueError("this testbench builds reduce chain, sync edge, no CREST")
        self.mode = str(p.get("chain_mode", "concurrent"))
        if self.mode not in ("serial", "concurrent"):
            raise ValueError(f"params chain_mode={self.mode!r}: serial or concurrent")
        self.NSLC = int(p.get("slices", 8))                 # slices a cluster
        self.SL = int(p.get("slice_bytes", 32768))          # one slice, FP16
        if self.SL % 64 or self.SL < 2 * CHECK_BYTES:
            raise ValueError(f"params slice_bytes={self.SL}: whole 64-B beats, at least {2 * CHECK_BYTES}")
        # params loads_per_slice: one count for every cluster, or a list per cluster ([cluster 0, cluster 1]): a
        # cluster with more loads a slice reaches its gathers later every slice, so the two clusters' gathers drift
        # against each other and start at every offset, the same time included
        lps = p.get("loads_per_slice", 5)
        self.LPS_C = [int(x) for x in lps] if isinstance(lps, list) else [int(lps)] * self.NC
        if len(self.LPS_C) != self.NC:
            raise ValueError(f"params loads_per_slice={lps}: one count or {self.NC}")
        self.LPS = max(self.LPS_C)
        # params first_slice_extra_loads: [cluster 0, cluster 1] weight chunks a cluster loads on top of its count in
        # slice 0 only, which shifts that cluster's every later gather by the same amount (the two clusters' gathers then
        # start at a fixed offset in every slice, the same time included)
        fx = p.get("first_slice_extra_loads", [0] * self.NC)
        self.FX = [int(x) for x in fx]
        if len(self.FX) != self.NC:
            raise ValueError(f"params first_slice_extra_loads={fx}: {self.NC} counts")
        if "chain_alt" in p:
            raise ValueError("params chain_alt: removed, every ChainGather runs the ring the same way")
        self.part_src = str(p.get("partials", "hbm"))
        if self.part_src not in ("hbm", "l3"):
            raise ValueError(f"params partials={self.part_src!r}: hbm or l3")
        rng = np.random.default_rng(int(p.get("seed", 1)))
        n = self.SL // 2
        # parts[(k, c)]: [slice, values]
        if self.part_src == "l3":
            same = {c: (rng.integers(-8, 9, size=(self.NSLC, n)) / 4).astype(np.float16) for c in range(self.NC)}
            self.parts = {(k, c): same[c] for k in self.chips for c in range(self.NC)}
        else:
            self.parts = {(k, c): (rng.integers(-8, 9, size=(self.NSLC, n)) / 4).astype(np.float16)
                          for k in self.chips for c in range(self.NC)}
        self._golden()
        self._stage()
        self.make_streams()
        self.make_rings({})
        self.sym = {(k, c): self.cx_chip(k).at(c).l1(f"AAAA_{self.TAG}_sym", self.NSLC * self.SL)
                    for k in self.chips for c in range(self.NC)}
        self.sym_zero = False                    # each slot is written whole by its partial copy first
        self.p1c = {c: self.cx_chip(self.ROOT).at(c).l1(f"{self.TAG}_p1c", 2 * self.SL) for c in range(self.NC)}

    def chain_order(self, c):
        """The chips a ChainGather of cluster c passes, the root last; both clusters' the same way, as
        moe_prefill_mc.py."""
        return self.order[1:] + [self.ROOT]

    def _golden(self):
        self.y = {}
        for c in range(self.NC):
            order = self.chain_order(c)
            for j in range(self.NSLC):
                acc = self.parts[(order[0], c)][j]
                for k in order[1:]:
                    acc = (acc + self.parts[(k, c)][j]).astype(np.float16)
                self.y[(c, j)] = acc

    def _stage(self):
        """Each memory chip's HBM: the weight chunks the rings of the chips it feeds stream (one run
        of chunks a memory chip, which every cluster it feeds reads), and with partials hbm those
        chips' partials; with partials l3 the image every chip runs holds them."""
        st, T = self.st, self.TAG
        self.h_part, self.w_src = {}, {}
        n_w = max(1, self.NSLC * self.LPS + max(self.FX))
        if self.part_src == "l3":
            st.put(f"{T}_part", "uint8_t",
                   np.concatenate([self.parts[(self.ROOT, c)].reshape(-1) for c in range(self.NC)]).view(np.uint8))
        for m in sorted(set(self.mem.values())):
            fed = [k for k in self.chips if self.mem[k] == m]
            if self.part_src == "hbm":
                blob = np.concatenate([self.parts[(k, c)].reshape(-1) for k in fed for c in range(self.NC)])
                h = st.put_hbm(f"{T}_part_m{m:02x}", blob.view(np.uint8), mem_chip=(m >> 4, m & 15))
                for i, k in enumerate(fed):
                    for c in range(self.NC):
                        self.h_part[(k, c)] = h.address + (i * self.NC + c) * self.NSLC * self.SL
            w = np.random.default_rng(m).integers(-128, 128, size=n_w * self.chunk, dtype=np.int8)
            self.w_src[m] = st.put_hbm(f"{T}_w_m{m:02x}", w, mem_chip=(m >> 4, m & 15)).address
        self.h_y = st.put_zeros(f"{T}_y", "uint8_t", self.NC * self.NSLC * self.SL)

    def _load(self, g, j, deps):
        """Weight chunk j of cluster g from its ring into the next slab, released at once."""
        k, c = self.chip(g), self.c(g)
        ls = self.ls[g]
        si, slab, war = ls.take(self.chunk)
        ld, _, _ = ls.rings.load_chunk(
            lambda n, core, kern, args, after: self.node(k, c, core, n, kern, args, after), c,
            f"Ld_w{j}_g{g}", src=self.w_src[self.mem[k]] + j * self.chunk, nbytes=self.chunk,
            slab=slab, war=war, deps=deps, dm=self.roles["dm"], trailer=True)
        ls.release(si, ld)
        return ld

    def y_off(self, c, j):
        return (c * self.NSLC + j) * self.SL

    def build(self):
        R, Tg = self.ROOT, self.TAG
        dm, xdma = self.roles["dm"], self.roles["xdma"]
        if self.p.get("relay_per_gather", True):
            self.dfg.relay_one_consumer_cores = [xdma]
        self.setup()
        G = self.G
        p1_reader = {c: [None, None] for c in range(self.NC)}
        last_red = {c: None for c in range(self.NC)}
        last_any = None
        nload = {g: 0 for g in range(G)}
        for j in range(self.NSLC):
            s2 = j % 2
            ready = {g: [] for g in range(G)}
            for g in range(G):
                k, c = self.chip(g), self.c(g)
                if self.part_src == "l3":
                    src = BingoMemSymbol(f"{Tg}_part", offset=(c * self.NSLC + j) * self.SL)
                else:
                    src = BingoMemFixedAddr(self.h_part[(k, c)] + j * self.SL, mem_level="HBM")
                cp = self.node(k, c, dm, f"x_copy_j{j}_g{g}", "__snax_bingo_kernel_idma_1d_copy",
                               SnaxBingoKernelIdma1dCopyArgs(src, self.sym[(k, c)].view(j * self.SL), self.SL))
                ready[g].append(cp)
                prev = cp
                for _ in range(self.LPS_C[c] + (self.FX[c] if j == 0 else 0)):
                    prev = self._load(g, nload[g], [prev])
                    nload[g] += 1
                if self.LPS_C[c] + (self.FX[c] if j == 0 else 0):
                    ready[g].append(prev)
            for c in range(self.NC):
                ordr = self.chain_order(c)[:-1]
                p1x = f"((uint64_t)({self.p1c[c].get_c_var_name()}))"
                ga = SnaxBingoKernelXdmaChainGatherArgs.fold_into(
                    cm.remote, R, c, ordr, src=self.sx(c, j * self.SL), dst=f"({p1x} + {s2 * self.SL})",
                    size=self.SL, junction=self.jct["Elementwise"], jct_csr0=cm.EW_F16_ADD,
                    holds=[self.p1c[c]])
                before = last_any if self.mode == "serial" else last_red[c]
                red = self.node(R, c, xdma, f"y_chain_c{c}j{j}", "__snax_bingo_kernel_xdma_chain_gather", ga,
                                [t for g in range(G) if self.c(g) == c for t in ready[g]]
                                + ([before] if before is not None else [])
                                + ([p1_reader[c][s2]] if p1_reader[c][s2] is not None else []))
                last_red[c] = last_any = red
                st = self.node(R, c, dm, f"y_store_c{c}j{j}", "__snax_bingo_kernel_idma_1d_copy",
                               SnaxBingoKernelIdma1dCopyArgs(self.p1c[c].view(s2 * self.SL),
                                                             BingoMemSymbol(f"{Tg}_y", offset=self.y_off(c, j)),
                                                             self.SL),
                               [red])
                p1_reader[c][s2] = st
        self.end_window([p1_reader[c][(self.NSLC - 1) % 2] for c in range(self.NC)])
        checks = []
        for c in range(self.NC):
            for j in range(self.NSLC):
                img = self.y[(c, j)].view(np.uint8)
                for end, off in (("a", 0), ("b", self.SL - CHECK_BYTES)):
                    checks.append((f"y_c{c}j{j}{end}",
                                   BingoMemSymbol(f"{Tg}_y", offset=self.y_off(c, j) + off, chip_id=R),
                                   img[off:off + CHECK_BYTES].copy(), CHECK_BYTES))
        self.run_checks(checks)


def _app_name(S):
    return (f"MoE reduce testbench, {S.NSLC} slices of {S.SL} B a cluster, chains {S.mode}, "
            f"{S.LPS} weight loads a slice, {len(S.chips)} chips x {S.NC} clusters")


def run():
    run_driver("MoE reduce testbench generator", GatherTb, _app_name,
               dsv2_dir={"default": None, "help": "unused (moe.mk passes it)"}, x_prefixes=("x_copy_",))
