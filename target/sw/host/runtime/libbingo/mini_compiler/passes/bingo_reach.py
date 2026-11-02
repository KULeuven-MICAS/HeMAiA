# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
"""Reachability in a DAG for many membership tests: what the tag allocator, the liveness planner and the hang check
ask of the happens-before graph, thousands of times.

    reach = reach_sets(G)          # every node's descendants at once
    b in reach[a]                  # a reaches b (b is not a itself)

The graph is covered by chains (each chain's consecutive nodes joined by an edge, so a node reaches every later node of
its chain), built greedily in topological order, a node joining the chain of a predecessor that ends there (its own
core's first). Every chain of two or more nodes is a column: each node keeps, for each column, the earliest position it
reaches there, and a reaches b on a column exactly when that position is at most b's. A node left on a chain of its own
(a leaf: the dummy set and check nodes the lowering hangs off a task, see below) has no column; a reaches a leaf when it is, or
reaches, one of the leaf's predecessors. A BINGO graph is close to one chain a core, so memory is N x (a few hundred
columns), not N x N bits, and a membership test is one lookup.

The column form also answers "which nodes is a concurrent with" without testing every pair (ReachMap.concurrent): on a
column, the nodes a reaches are a suffix and the nodes that reach a a prefix, so the rest of the column is.

Leaves are the nodes of a chain of one node or of a short chain of dummy nodes (a dummy set feeding a dummy check): a
graph has many, and each a column would make the labels N x N again.

A graph whose cover needs too many columns (not the case for a task graph) falls back to one bitset a node over a
topological order (bit i = its i-th node), the union of its successors' and theirs.
"""
import networkx as nx
import numpy as np

_INF = np.iinfo(np.int32).max
LEAF = (-1, 0)                      # where[] of a node without a column
MIN_DUMMY_COLUMN = 8                # a chain of dummy nodes only is a column from this length on


class Reach:
    """The nodes reachable from one node (bitset form), as a set to test membership in."""
    __slots__ = ("bits", "pos")

    def __init__(self, bits: int, pos: dict):
        self.bits, self.pos = bits, pos

    def __contains__(self, n) -> bool:
        i = self.pos.get(n)
        return i is not None and (self.bits >> i) & 1 == 1


def _cover(G, topo, pos):
    """({node: (column, position) or LEAF}, columns): a greedy chain cover in topological order. Each node names the
    successor its chain should go on with: on its own core (BingoNode.assigned_*), the earliest one that is not a dummy
    node left without a later task of that core; with none on its core, its earliest successor. A node joins the chain
    of such a predecessor that is still its chain's end (its own core's first, then a task with a kernel, then the
    longest chain, then the latest). So a core's tasks stay one chain, also where the lowering hung a dummy set node off a task; those dummies
    are left over. A chain of one node, or of a few dummy nodes only, is leaves; the others are numbered as columns."""
    chain, tail_of, length, has_task = {}, [], [], []

    def core(n):
        return (getattr(n, "assigned_chiplet_id", None), getattr(n, "assigned_cluster_id", None),
                getattr(n, "assigned_core_id", None))

    def real(n):
        return bool(getattr(n, "kernel_name", None))
    def dead(s):                    # a dummy node no later task of its core follows
        return not real(s) and not any(core(x) == core(s) for x in G.successors(s))
    nxt = {}
    for p in topo:
        succ = list(G.successors(p))
        same = [s for s in succ if core(s) == core(p)]
        if same or succ:
            nxt[p] = min(same or succ, key=lambda s: (dead(s), pos[s]))
    for n in topo:
        best = None
        for p in G.predecessors(n):
            c, _i = chain[p]
            if nxt.get(p) is n and tail_of[c] is p and (
                    best is None or (core(p) == core(n), real(p), length[c], pos[p])
                    > (core(best) == core(n), real(best), length[chain[best][0]], pos[best])):
                best = p
        if best is None:
            chain[n] = (len(tail_of), 0)
            tail_of.append(n)
            length.append(1)
            has_task.append(real(n))
        else:
            c, i = chain[best]
            chain[n] = (c, i + 1)
            tail_of[c] = n
            length[c] += 1
            has_task[c] = has_task[c] or real(n)
    col, K = {}, 0
    for c, ln in enumerate(length):
        if ln > 1 and (has_task[c] or ln >= MIN_DUMMY_COLUMN):
            col[c] = K
            K += 1
    where = {n: (col[c], i) if c in col else LEAF for n, (c, i) in chain.items()}
    return where, K


def _labels(G, topo, pos, where, K):
    """[N, K]: each node's earliest reached position on each column (_INF: none), in reverse topological order."""
    L = np.full((len(topo), K), _INF, dtype=np.int32)
    for n in reversed(topo):
        row = L[pos[n]]
        for s in G.successors(n):
            np.minimum(row, L[pos[s]], out=row)
            c, i = where[s]
            if c >= 0 and i < row[c]:
                row[c] = i
    return L


def _leaf_ancestry(G, where, b):
    """For a leaf b: (its leaf ancestors reached through leaves only, the column nodes where that walk stops)."""
    leaves, cols, stack = set(), [], [b]
    while stack:
        for p in G.predecessors(stack.pop()):
            if p in leaves:
                continue
            if where[p][0] >= 0:
                cols.append(p)
            else:
                leaves.add(p)
                stack.append(p)
    return leaves, cols


def _reaches(G, L, pos, where, a, b) -> bool:
    """a reaches b (b is not a): on b's column by a's label; a leaf b when a is one of its leaf ancestors, is one of the
    column nodes above them, or reaches one."""
    c, i = where[b]
    if c >= 0:
        return L[pos[a], c] <= i
    leaves, cols = _leaf_ancestry(G, where, b)
    if a in leaves:
        return True
    row = L[pos[a]]
    return any(p is a or row[where[p][0]] <= where[p][1] for p in cols)


class ChainReach:
    """The nodes reachable from one node (column form), as a set to test membership in."""
    __slots__ = ("node", "m")

    def __init__(self, node, m):
        self.node, self.m = node, m

    def __contains__(self, n) -> bool:
        m = self.m
        return n in m.where and _reaches(m.G, m.L, m.pos, m.where, self.node, n)


class ChainLabels:
    """Reachability of a DAG that a pass keeps adding edges to (column form): reaches(a, b) as reach_sets' b in
    reach[a]; add_edge(a, b) after the graph gained a -> b. Exact as long as every added edge keeps the graph a DAG."""

    def __init__(self, G, topo=None):
        topo = list(nx.topological_sort(G)) if topo is None else topo
        self.G = G
        self.idx = {n: i for i, n in enumerate(topo)}
        self.where, K = _cover(G, topo, self.idx)
        self.L = _labels(G, topo, self.idx, self.where, K)

    def reaches(self, a, b) -> bool:
        return _reaches(self.G, self.L, self.idx, self.where, a, b)

    def _reachers(self, a):
        """bool [N]: a and every node that reaches a."""
        c, i = self.where[a]
        if c >= 0:
            rows = self.L[:, c] <= i
        else:
            leaves, cols = _leaf_ancestry(self.G, self.where, a)
            rows = np.zeros(len(self.L), dtype=bool)
            for p in cols:
                cp, ip = self.where[p]
                rows |= self.L[:, cp] <= ip
                rows[self.idx[p]] = True
            for p in leaves:
                rows[self.idx[p]] = True
        rows[self.idx[a]] = True
        return rows

    def add_edge(self, a, b) -> None:
        """a now reaches b (the graph already holds the edge): a and everything that reaches a reach what b reaches,
        and b (a leaf b is reached through its predecessors, which now include a)."""
        rows = self._reachers(a)
        gained = self.L[self.idx[b]].copy()
        cb, ib = self.where[b]
        if cb >= 0:
            gained[cb] = min(gained[cb], ib)
        self.L[rows] = np.minimum(self.L[rows], gained)


class ReachMap(dict):
    """reach_sets' node -> reach mapping; in the column form it also keeps the cover and answers concurrent(a): every
    column node neither reached from a nor reaching a (a excluded). Leaves are not listed."""

    def __init__(self, items, G=None, topo=None, pos=None, where=None, L=None):
        super().__init__(items)
        self.G, self.topo, self.pos, self.where, self.L = G, topo, pos, where, L
        self._P = self.columns = None
        if L is not None:
            self.columns = [[] for _ in range(L.shape[1])]
            for n in topo:
                c, _i = where[n]
                if c >= 0:
                    self.columns[c].append(n)

    @property
    def chain_form(self) -> bool:
        return self.L is not None

    def on_column(self, n) -> bool:
        return self.L is not None and n in self.where and self.where[n][0] >= 0

    def _latest(self):
        """[N, K]: each node's latest position on each column among the nodes that reach it (-1: none)."""
        if self._P is None:
            G, pos, where = self.G, self.pos, self.where
            P = np.full(self.L.shape, -1, dtype=np.int32)
            for n in self.topo:
                row = P[pos[n]]
                for p in G.predecessors(n):
                    np.maximum(row, P[pos[p]], out=row)
                    c, i = where[p]
                    if c >= 0 and i > row[c]:
                        row[c] = i
            self._P = P
        return self._P

    def matrix(self, srcs, dsts):
        """bool [len(srcs), len(dsts)]: entry (i, j) is dsts[j] in self[srcs[i]], one column or leaf of dsts at a
        time over all srcs."""
        M = np.zeros((len(srcs), len(dsts)), dtype=bool)
        if self.L is None:
            for i, a in enumerate(srcs):
                r = self[a]
                M[i] = [b in r for b in dsts]
            return M
        pos, where = self.pos, self.where
        at = np.array([pos[a] for a in srcs], dtype=np.int64)
        rows = self.L[at]
        on = [j for j, b in enumerate(dsts) if where[b][0] >= 0]
        if on:
            c = np.array([where[dsts[j]][0] for j in on])
            i = np.array([where[dsts[j]][1] for j in on])
            step = max(1, (16 << 20) // len(on))            # rows a block, so the gathered labels stay small
            for r in range(0, len(srcs), step):
                M[r:r + step, on] = rows[r:r + step, c] <= i
        for j, b in enumerate(dsts):
            if where[b][0] >= 0:
                continue
            leaves, cols = _leaf_ancestry(self.G, where, b)
            m = np.isin(at, [pos[p] for p in leaves] + [pos[p] for p in cols])
            for p in cols:
                m |= rows[:, where[p][0]] <= where[p][1]
            M[:, j] = m
        return M

    def concurrent(self, a, columns=None):
        """The column nodes a is unordered with, column by column in position order (only on `columns`, if given)."""
        P, k = self._latest(), self.pos[a]
        lo, hi = P[k], self.L[k]
        out = []
        for c in (range(len(self.columns)) if columns is None else columns):
            column = self.columns[c]
            b, e = int(lo[c]) + 1, (len(column) if hi[c] == _INF else int(hi[c]))
            if b < e:
                out += [x for x in column[b:e] if x is not a]
        return out


def find_cycle(G):
    """nx.find_cycle(G) for a DiGraph, the same edges, in time linear in the graph: networkx walks everything a start
    node reaches again for every start node (quadratic on a DAG), where this walk stops at the nodes an earlier start
    explored (no cycle is reachable from them, so the walk finds the same first cycle). Raises nx.NetworkXNoCycle."""
    explored = set()
    for start in G:
        if start in explored:
            continue
        edges, seen, active, previous_head = [], {start}, {start}, None
        its, stack = {start: G.successors(start)}, [start]
        while stack:
            tail = stack[-1]
            try:
                head = next(its[tail])
            except StopIteration:
                stack.pop()
                continue
            if head in explored:
                continue
            if head not in its:
                its[head] = G.successors(head)
            stack.append(head)
            if previous_head is not None and tail != previous_head:
                while True:                                 # backtracked: pop to the edge that ends at tail
                    if not edges:
                        active = {tail}
                        break
                    active.remove(edges.pop()[1])
                    if edges and edges[-1][1] == tail:
                        break
            edges.append((tail, head))
            if head in active:
                for i, (t, _h) in enumerate(edges):
                    if t == head:
                        break
                return edges[i:]
            seen.add(head)
            active.add(head)
            previous_head = head
        explored |= seen
    raise nx.NetworkXNoCycle("No cycle found.")


def reach_sets(G, max_label_bytes: int = 2 << 30, form: str = None) -> dict:
    """node -> what the node reaches in the DAG G (a set to test membership in), a ReachMap. form "chains" / "bits"
    forces one form (the tests); default: the column form when its labels are the smaller."""
    topo = list(nx.topological_sort(G))
    pos = {n: i for i, n in enumerate(topo)}
    where, K = _cover(G, topo, pos)
    N = len(topo)
    if form == "chains" or (form is None and N and K * N * 4 <= max_label_bytes and K * 32 < N):
        L = _labels(G, topo, pos, where, K)
        m = ReachMap((), G, topo, pos, where, L)
        for n in topo:
            m[n] = ChainReach(n, m)
        return m
    bits: dict = {}
    for n in reversed(topo):
        b = 0
        for s in G.successors(n):
            b |= bits[s] | (1 << pos[s])
        bits[n] = b
    return ReachMap((n, Reach(b, pos)) for n, b in bits.items())
