# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
"""Reachability in a DAG for many membership tests: what the tag allocator and the hang check ask of
the happens-before graph, thousands of times.

    reach = reach_sets(G)          # every node's descendants at once
    b in reach[a]                  # a reaches b (b is not a itself)

Each node gets one bitset over a topological order of G (bit i = its i-th node), the union of its
successors' and theirs, built in reverse order: a test is a bit test, where a networkx descendants()
walk per node made those passes quadratic in the graph.
"""
import networkx as nx


class Reach:
    """The nodes reachable from one node, as a set to test membership in."""
    __slots__ = ("bits", "pos")

    def __init__(self, bits: int, pos: dict):
        self.bits, self.pos = bits, pos

    def __contains__(self, n) -> bool:
        i = self.pos.get(n)
        return i is not None and (self.bits >> i) & 1 == 1


def reach_sets(G) -> dict:
    """node -> Reach: what each node of the DAG G reaches."""
    topo = list(nx.topological_sort(G))
    pos = {n: i for i, n in enumerate(topo)}
    bits: dict = {}
    for n in reversed(topo):
        b = 0
        for s in G.successors(n):
            b |= bits[s] | (1 << pos[s])
        bits[n] = b
    return {n: Reach(b, pos) for n, b in bits.items()}
