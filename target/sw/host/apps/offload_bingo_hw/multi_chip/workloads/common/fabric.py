# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""What the multi-chiplet LLM workload builds (workloads/ffn, moe, fa) share on the 16 + 4 chiplet
platform: the mesh, the address of a local buffer on another chip, the fold order over the compute
chips, and FabricBase, the topology and node helpers of ChipletBuild (ffn/common/chiplet_mc.py) and of
the decode FFN's FBuild (ffn/common/ffn_mc.py).

It imports the compiler's modules, so a caller puts mini_compiler on sys.path (and imports
_bingo_paths) before importing it.
"""

import importlib.util
import os

from bingo_kernel_args import SnaxBingoKernelSyncMarkArgs, SnaxBingoKernelSyncProbeArgs
from libs import Ctx

L1_CAPACITY = 990000
MESH = (16, 4, 16)
MASK40 = "0xFFFFFFFFFFULL"


def remote(chip, local_expr):
    """`local_expr` (a local address, the same on every chip) on chip `chip`."""
    return f"(chiplet_addr_transform_full(0x{chip:02x}, ((uint64_t)({local_expr}) & {MASK40})))"


def dist(a, b):
    return abs((a >> 4) - (b >> 4)) + abs((a & 15) - (b & 15))


def fold_cycle(chips, root):
    """A Hamiltonian cycle over `chips` through `root` (root first), each step to a chip at
    Manhattan distance 1, or 2 where the grid leaves no choice: the fewest distance-2 steps,
    found by backtracking (16 chips: instant). Falls back to a path if no cycle exists."""
    nb = {k: sorted((m for m in chips if m != k and dist(k, m) <= 2),
                    key=lambda m: (dist(k, m), m)) for k in chips}

    def search(path, seen, longs, budget, cycle):
        k = path[-1]
        if len(path) == len(chips):
            if not cycle:
                return path
            d = dist(k, root)
            if d == 1 or (d == 2 and longs < budget):
                return path
            return None
        free = [m for m in nb[k] if m not in seen]
        # fewest onward choices first (Warnsdorff), the near ones before the far ones
        free.sort(key=lambda m: (dist(k, m), sum(1 for q in nb[m] if q not in seen), m))
        for m in free:
            ln = longs + (dist(k, m) == 2)
            if ln > budget:
                continue
            seen.add(m)
            got = search(path + [m], seen, ln, budget, cycle)
            if got:
                return got
            seen.discard(m)
        return None

    for cycle in (True, False):
        for budget in range(0, 5):
            got = search([root], {root}, 0, budget, cycle)
            if got:
                return got
    raise ValueError(f"no fold order over {[hex(c) for c in chips]}")


def crest_codec(dsv2_dir):
    """snax_cluster's CREST reference codec (hw/chisel/doc/crest_decompressor/crest_codec.py)."""
    root = os.path.abspath(dsv2_dir)
    while root != os.path.dirname(root) and not os.path.isdir(os.path.join(root, "hw", "chisel")):
        root = os.path.dirname(root)
    path = os.path.join(root, "hw", "chisel", "doc", "crest_decompressor", "crest_codec.py")
    spec = importlib.util.spec_from_file_location("crest_codec", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class FabricBase:
    """The topology and node helpers of a build. The subclass sets chips, NC, ROOT, dfg, roles, hw,
    n_anchor, stamps, mark and sym (the symmetric buffers, {(chip, cluster): handle})."""

    # ---- topology helpers -----------------------------------------------------------------
    def chip(self, g):
        return self.chips[g // self.NC]

    def c(self, g):
        return g % self.NC

    def gs(self, k):
        i = self.chips.index(k)
        return list(range(i * self.NC, (i + 1) * self.NC))

    def cx(self, g):
        return Ctx(dfg=self.dfg, mesh=MESH, roles=self.roles, hw=self.hw,
                   chiplet=self.chip(g)).at(self.c(g))

    def cx_chip(self, k):
        return Ctx(dfg=self.dfg, mesh=MESH, roles=self.roles, hw=self.hw, chiplet=k)

    # ---- nodes ----------------------------------------------------------------------------
    def node(self, k, c, core, name, kernel, kargs, after=()):
        return self.cx_chip(k).node(f"{name}_k{k:02x}", core, kernel, kargs, after, cluster=c)

    def edge(self, a, b):
        self.dfg.bingo_add_edge(a, b)

    def anchor(self, tag, after=()):
        nd = self.node(self.ROOT, 0, self.roles["dm"], f"A{self.n_anchor}_{tag}",
                       "__snax_bingo_kernel_sync_mark",
                       SnaxBingoKernelSyncMarkArgs(stamp_buf=self.stamps, slot=self.n_anchor,
                                                   mark_addr=self.mark), after)
        self.n_anchor += 1
        return nd

    def probe(self, k, tag, after=()):
        """A DM-core task on chip k's cluster 0 that does nothing but order."""
        return self.node(k, 0, self.roles["dm"], f"P_{tag}", "__snax_bingo_kernel_sync_probe",
                         SnaxBingoKernelSyncProbeArgs(stamp_buf=0, slot=0), after)

    def sv(self, k, c, off):
        """A view of chip k cluster c's symmetric buffer at `off`."""
        return self.sym[(k, c)].view(off)

    def sx(self, c, off):
        """The C expression of the root's own symmetric buffer at `off`: the same local
        address on every chip."""
        return f"((uint64_t)({self.sym[(self.ROOT, c)].get_c_var_name()}) + {off})"
