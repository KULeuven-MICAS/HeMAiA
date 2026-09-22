# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""The node and handle factory a block builds through.

One Ctx is bound to one cluster and one name prefix. The prefix is what lets the same
block be instantiated twice in one graph: handle names must be unique per chip across all
clusters and levels, because the emitter dedups by object identity and two same-named
handles emit two C declarations that do not compile.
"""

from dataclasses import dataclass, replace

from bingo_mem_handle import BingoMemAlloc
from bingo_node import BingoNode

# ======================================================================================

@dataclass
class Ctx:
    """Node and handle factory, bound to one cluster and one name prefix.

    Every handle carries its cluster id, so a handle from another cluster already resolves
    to a full (chip | cluster | offset) address -- which is what lets one block's push name
    its destination on another cluster directly.
    """
    dfg: object
    mesh: tuple                       # (meshRow, tileSize, meshCol)
    roles: dict                       # {"gemm","simd","xdma","dm","host"} -> core id
    cluster: int = 0
    chiplet: int = 0
    prefix: str = ""

    @property
    def gemm(self): return self.roles["gemm"]
    @property
    def simd(self): return self.roles["simd"]
    @property
    def xdma(self): return self.roles["xdma"]
    @property
    def dm(self): return self.roles["dm"]
    @property
    def host(self): return self.roles["host"]

    def at(self, cluster: int) -> "Ctx":
        return replace(self, cluster=cluster)

    def scope(self, prefix: str) -> "Ctx":
        """A child namespace. Two instances of one block need different prefixes or their
        handles collide -- and a handle collision is a C redefinition, not a Python error:
        the emitter dedups by object identity and filters only by chip, so two same-named
        handles emit two `uint64_t ptr_X = ...` lines.

        The prefix is COMMON to every handle a block makes, which matters more than it
        looks: the heap is laid out in name-sorted order, so a shared prefix preserves the
        relative order a block may depend on, while a per-handle prefix would not.
        """
        if not prefix:
            # An empty namespace is no namespace. A single-block graph wants the names it
            # would have had without the library, which is also what makes a refactor
            # verifiable by diffing the generated header.
            return self
        return replace(self, prefix=f"{self.prefix}{prefix}.")

    def l1(self, name: str, size: int, cluster: int = None) -> BingoMemAlloc:
        cl = self.cluster if cluster is None else cluster
        return BingoMemAlloc(f"{self.prefix}{name}_cl{cl}", size=size, mem_level="L1",
                             chip_id=self.chiplet, cluster_id=cl)

    def l3(self, name: str, size: int) -> BingoMemAlloc:
        return BingoMemAlloc(f"{self.prefix}{name}", size=size, mem_level="L3",
                             chip_id=self.chiplet)

    def node(self, name, core, kname, kargs, after=(), cluster=None) -> BingoNode:
        cl = self.cluster if cluster is None else cluster
        nd = BingoNode(assigned_chiplet_id=self.chiplet, assigned_cluster_id=cl,
                       assigned_core_id=core, node_name=f"{self.prefix}{name}_cl{cl}",
                       kernel_name=kname, kernel_args=kargs)
        self.dfg.bingo_add_node(nd)
        for pred in (after if isinstance(after, (list, tuple)) else [after]):
            if pred is not None:
                self.dfg.bingo_add_edge(pred, nd)
        return nd

    def host_node(self, name, kname, kargs, after=()) -> BingoNode:
        """A host kernel. They are validated onto cluster 0's host core and nowhere else,
        so the cluster is not a choice -- and because `_cl0` is then the same on every one,
        any distinguishing id has to be in `name` itself."""
        return self.node(name, self.host, kname, kargs, after, cluster=0)


# ======================================================================================
# The linker
