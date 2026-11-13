# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""Broadcasts and gathers kept apart by the compiler (libs/blocks/collective.py, THE RULE).

bingo_order_collectives runs on the graph the application built, before any pass derives an order from its edges. It
takes every collective of the graph (collective.spans()) in a topological order of the ORDER VIEW -- the dependencies,
the ordering-only edges, and each chip-barrier round as one point that every chip's tasks before it precede and every
chip's tasks after it follow -- the lowest node id first among the tasks ready at once. A collective is done after its
end task, or (a broadcast) after any receiver's wait on its data: a task one of its tasks orders by an ordering-only
edge. For each collective, every earlier one it may not run beside (collective.conflict) that is not done before it
starts, nor starts after it is done, gets an edge from the earlier one's end to its first task: an ordering-only edge
when the two tasks share a core (its in-order queue runs one after the other), a dependency otherwise; a broadcast
whose landing no task waits for yet gets that wait first (HbmCast.end). Then every such pair is proved ordered.

dfg.collectives_report = True only reports the pairs it would order, and adds nothing."""

import networkx as nx

from bingo_node import BingoNode


def _key(t):
    return t.node_id if isinstance(t, BingoNode) else -1


def _core(t):
    return (t.assigned_chiplet_id, t.assigned_cluster_id, t.assigned_core_id)


class BingoDFGCollectivesMixin:

    def _collectives_view(self):
        """A copy of the order view with a node a chip-barrier round, after every chip's barrier task of the round and
        before every successor of each."""
        g = self.bingo_order_view()
        g = nx.DiGraph(g) if g is self else g       # the order view is a copy already when ordering edges exist
        for n, rnd in enumerate(getattr(self, "barrier_rounds", None) or ()):
            v = ("barrier_round", n)
            for b in (rnd.values() if isinstance(rnd, dict) else rnd):
                succ = [s for s in g.successors(b) if s != v]
                g.add_edge(b, v)
                for s in succ:
                    g.add_edge(v, s)
        return g

    def _collectives_sync(self, g, t):
        """Task t (a landing wait just asked for) into the view g with every edge of it the graph now has."""
        g.add_node(t)
        for u in self.predecessors(t):
            g.add_edge(u, t)
        for v in self.successors(t):
            g.add_edge(t, v)
        for x, y in self.order_edges:
            if x is t or y is t:
                g.add_edge(x, y)

    def _collectives_done(self, sp):
        """Per span, the tasks after which it is done: its end (when made) and, a broadcast, every task one of its
        tasks orders by an ordering-only edge (a receiver's wait on its data)."""
        waits = {}
        for x, y in self.order_edges:
            waits.setdefault(x, []).append(y)
        done = []
        for s in sp:
            d = {s.done()} - {None}
            if s.kind == "broadcast":
                for t in set(s.tasks or [s.first]) | d:
                    d.update(waits.get(t, ()))
            done.append(d)
        return done

    @staticmethod
    def _collectives_reach(g, done):
        """{task: the bits of the spans done at or before it}."""
        bit = {}
        for i, d in enumerate(done):
            for t in d:
                bit[t] = bit.get(t, 0) | (1 << i)
        reach = {}
        for v in nx.topological_sort(g):
            r = bit.get(v, 0)
            for u in g.predecessors(v):
                r |= reach[u]
            reach[v] = r
        return reach

    def bingo_order_collectives(self) -> list:
        """Order every two collectives that may not run at once (the module docstring). Returns the edges added, as
        (earlier end, later first)."""
        from libs.blocks.collective import conflict, spans as find_spans
        sp = find_spans(self)
        if len(sp) < 2:
            return []
        report = bool(getattr(self, "collectives_report", False))
        g = self._collectives_view()
        pos = {t: n for n, t in enumerate(nx.lexicographical_topological_sort(g, key=_key))}
        sp.sort(key=lambda s: pos[s.first])
        done = self._collectives_done(sp)
        reach = self._collectives_reach(g, done)

        def ordered(i, j):
            return bool(reach[sp[j].first] >> i & 1 or reach[sp[i].first] >> j & 1)

        added, made = [], []
        for j in range(len(sp)):
            for i in range(j - 1, -1, -1):
                if not conflict(sp[i], sp[j]) or ordered(i, j):
                    continue
                if report:
                    added.append((sp[i], sp[j]))
                    continue
                a = sp[i].done()
                if a is None:
                    n0 = self.number_of_nodes()
                    a = sp[i].resolve()                # the broadcast's landing wait (made now if no task is it yet)
                    if self.number_of_nodes() != n0:
                        made.append(sp[i].name)
                    self._collectives_sync(g, a)
                    done[i].add(a)
                    reach = self._collectives_reach(g, done)
                    if ordered(i, j):
                        continue
                b = sp[j].first
                if _core(a) == _core(b):
                    self.bingo_add_order_edge(a, b)
                else:
                    self.bingo_add_edge(a, b)
                g.add_edge(a, b)
                added.append((sp[i], sp[j]))
                reach = self._collectives_reach(g, done)
        nb = sum(s.kind == "broadcast" for s in sp)
        head = f"[bingo] collectives: {nb} broadcasts, {len(sp) - nb} gathers"
        if report:
            print(f"{head}; {len(added)} pair(s) that may not run at once without an order"
                  + (f": {[f'{a.name} -> {b.name}' for a, b in added[:6]]}" if added else ""))
            return []
        bad = [(sp[i].name, sp[j].name) for j in range(len(sp)) for i in range(j)
               if conflict(sp[i], sp[j]) and not ordered(i, j)]
        if bad:
            raise ValueError(f"collectives: {len(bad)} pair(s) may run at once: {bad[:6]}")
        msg = f"{head}, every two that may not run at once ordered"
        if added:
            msg += f" ({len(added)} edge(s) added: {[f'{a.name} -> {b.name}' for a, b in added[:4]]})"
        if made:
            msg += f"; landing waits made for {made[:4]}"
        print(msg)
        return [(a.done(), b.first) for a, b in added]
