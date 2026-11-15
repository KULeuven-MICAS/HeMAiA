# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""Memory hazards: every pair of tasks of which one writes bytes the other touches is ordered by the graph.

WHY. The graph's edges are the only order the application states. A core's program order is the compiler's choice
(bingo_dfg_schedule.py), so two tasks that read and overwrite the same bytes with no path between them run in an order
nobody wrote down: the result depends on the schedule. Nothing hangs and nothing faults; the data is wrong.

WHAT IS CHECKED. Each task's kernel declares what it does at each address field (kernel_base.py Access: Read, Write,
ReadWrite with the bytes from the address, or runs of them a pitch apart; Sync; Hold). For every buffer -- an allocation (its views included), an L3
symbol of a chip -- every two tasks whose byte ranges overlap, one of them writing, must have a path between them over
the dependency edges and the ordering-only edges (bingo_add_order_edge). A dispatch that keeps reading after its task
returns (operands_held_by: a queued GEMM's start, an asynchronous softmax) touches its buffers until the task it names.
Sync accesses (a flag, a stamp, a weight ring behind its flags) are ordered by their protocol and Hold ones touch
nothing; neither is checked. An address the args hold as a plain integer or a C expression is no handle and not seen.

WHERE. First in bingo_compile_dfg, on the graph the application built. A graph whose program the compiler orders
(schedule_cores) must pass: a pair left unordered is an error naming both tasks, the buffer and the edge to add. A
graph without schedule_cores (a hand-built one, a test's) is reported on only.

Every kernel declares its accesses: a task whose args class declares none, or leaves an address field it holds out of
its declaration, is an error where the check is enforced."""

import os

import networkx as nx

import bingo_kernel_args as K
from bingo_mem_handle import BingoMemAlloc, BingoMemAllocView, BingoMemSymbol
from bingo_reach import reach_sets

INF = 1 << 62
SHOW = 12


SWEEP_MIN = 64                   # a buffer of more accesses than this is swept (bingo_check_hazards), not looped
RUNS_MAX = 1 << 16               # the runs of a nest enumerated against another one; past it, the bounds decide


def _norm(lo, dims):
    """A nest with every stride made non-negative (the first byte moves to its lowest run)."""
    out = []
    for c, s in dims:
        if s < 0:
            lo += s * (c - 1)
            s = -s
        out.append((c, s))
    return lo, tuple(out)


def _span(n, dims):
    return n + sum(s * (c - 1) for c, s in dims)


def _hit(lo, n, dims, b, nb):
    """Whether the runs of `n` bytes of the nest `dims` from `lo` (strides non-negative) touch the bytes [b, b + nb)."""
    if not dims:
        return lo < b + nb and b < lo + n
    (c, s), rest = dims[0], dims[1:]
    if s == 0:
        return _hit(lo, n, rest, b, nb)
    sub = _span(n, rest)
    first = max(0, (b - lo - sub) // s + 1)
    last = min(c - 1, (b + nb - 1 - lo) // s)
    return any(_hit(lo + i * s, n, rest, b, nb) for i in range(first, last + 1))


def _runs(lo, n, dims):
    """The first byte of every run of a nest."""
    if not dims:
        yield lo
        return
    (c, s), rest = dims[0], dims[1:]
    for i in range(c):
        yield from _runs(lo + i * s, n, rest)


def _overlap(a, b):
    """Whether two accesses (first byte, run bytes, nest) share a byte."""
    (la, na, da), (lb, nb, db) = a, b
    la, da = _norm(la, da)
    lb, db = _norm(lb, db)
    ca, cb = 1, 1
    for c, _ in da:
        ca *= c
    for c, _ in db:
        cb *= c
    if ca > cb:
        (la, na, da, ca), (lb, nb, db, cb) = (lb, nb, db, cb), (la, na, da, ca)
    if ca > RUNS_MAX:
        return True
    return any(_hit(lb, nb, db, r, na) for r in _runs(la, na, da))


def _handles(v):
    """The memory handles in a field value (lists, tuples and dicts walked)."""
    if isinstance(v, (BingoMemAlloc, BingoMemAllocView, BingoMemSymbol)):
        yield v
    elif isinstance(v, (list, tuple)):
        for x in v:
            yield from _handles(x)
    elif isinstance(v, dict):
        for x in v.values():
            yield from _handles(x)


def _where(h, task):
    """(buffer key, readable name, first byte, the buffer's size or INF) of a handle as `task` names it."""
    if isinstance(h, BingoMemAllocView):
        b = h.base
        return ("A", id(b)), _alloc_name(b), int(h.offset), int(b.size) if b.size else INF
    if isinstance(h, BingoMemAlloc):
        return ("A", id(h)), _alloc_name(h), 0, int(h.size) if h.size else INF
    chip = task.assigned_chiplet_id if h.chip_id is None else h.chip_id
    return ("S", h.symbol_name, chip), f"{h.symbol_name}@{chip:#04x}", int(h.offset), INF


def _alloc_name(h):
    return f"{h.name}@{h.mem_level}:{h.chip_id:#04x}.{h.cluster_id}"


class BingoDFGHazardsMixin:
    """bingo_check_hazards: every write ordered against every overlapping access (see the module)."""

    def bingo_check_hazards(self):
        enforce = self.schedule_cores
        undeclared, uncovered = {}, []
        bufs = {}                           # buffer key -> [(first byte, end, writes, task)]
        names = {}
        for t in self.node_list:
            a = t.kernel_args
            if a is None or getattr(t, "config_only", False):
                continue
            try:
                decl = a.accesses()
            except NotImplementedError:
                undeclared.setdefault(type(a).__name__, []).append(t)
                continue
            fields = {x.field for x in decl}
            for f, v in vars(a).items():
                if not f.startswith("_") and f not in fields and next(_handles(v), None) is not None:
                    uncovered.append((t, type(a).__name__, f))
            for x in decl:
                if x.mode in (K.SYNC, K.HOLD):
                    continue
                for h in _handles(getattr(a, x.field, None)):
                    key, name, lo, size = _where(h, t)
                    lo = max(0, lo + x.offset)
                    first, dims = (_norm(lo, x.dims) if x.dims else (lo, ()))
                    span = None if x.nbytes is None else _span(x.nbytes, dims)
                    lo, hi = first, min(size, INF if span is None else first + span)
                    if hi <= lo:
                        continue
                    names[key] = name
                    runs = (lo, x.nbytes, dims) if dims else None
                    bufs.setdefault(key, []).append((lo, hi, x.mode != K.READ, t, x.mode, runs))
        g = self.bingo_order_view()
        rounds = getattr(self, "barrier_rounds", None) or []
        if rounds:
            # chip barriers (dfg.barrier_rounds: per round, each chip's barrier task): a round orders every chip's
            # tasks before it against every chip's after it, which no edge between the chips states
            g = nx.DiGraph(g)
            for n, rnd in enumerate(rounds):
                v = ("barrier_round", n)
                for b in rnd.values():
                    succ = list(g.successors(b))
                    g.add_edge(b, v)
                    g.add_edges_from((v, s) for s in succ)
        reach = reach_sets(g)

        def end(t):
            return getattr(t, "operands_held_by", None) or t

        def ordered(a, b):
            return b in reach[end(a)] or a in reach[end(b)]

        # A buffer of more than SWEEP_MIN accesses, each of whose tasks reaches the task that ends its hold, is swept,
        # not looped. Two accesses that share a byte share the first byte of the later one, so they are both in the set
        # of accesses open there taken just before the next one closes (cover_sets: by first byte and end, a nest by its
        # bounds). One sweep of such a set in a topological order of the same view -- each writer after the writer
        # before it and every reader since (from the start, for the first), each reader after the last writer --
        # orders every pair with a writer exactly when the pairwise loop below would: along a topological order only
        # the forward direction can hold, and end(W1) -> W2 -> end(W2) -> W3 composes. Linear in each set, where the
        # loop is quadratic in the buffer. A nest's bounds may join two accesses whose runs share no byte, so a sweep
        # can ask more than the loop; a set the sweep finds unordered sends its buffer to the loop, which decides and
        # names the pairs as always. BINGO_HAZARD_CROSSCHECK=1: both; the sweep never passes what the loop fails, and
        # without nests they agree. The pairs counted for a swept buffer are by bounds.
        def cover_sets(acc):
            import heapq
            out, active, ends, grown = [], set(), [], False
            for i in sorted(range(len(acc)), key=lambda i: acc[i][0]):
                if ends and ends[0][0] <= acc[i][0]:
                    if grown:
                        out.append(list(active))
                        grown = False
                    while ends and ends[0][0] <= acc[i][0]:
                        active.discard(heapq.heappop(ends)[1])
                active.add(i)
                heapq.heappush(ends, (acc[i][1], i))
                grown = True
            if grown:
                out.append(list(active))
            return out

        def pairs_with_writer(acc):
            """The pairs of accesses (by bounds) that share a byte, one of them writing."""
            import bisect
            def overlapping(rs):
                rs = sorted(rs)
                los = [r[0] for r in rs]
                return sum(bisect.bisect_left(los, hi, i + 1) - (i + 1) for i, (_lo, hi) in enumerate(rs))
            return overlapping([(r[0], r[1]) for r in acc]) - overlapping([(r[0], r[1]) for r in acc if not r[2]])

        swept, rest = [], []
        for key, acc in bufs.items():
            if len(acc) > SWEEP_MIN and all(end(r[3]) is r[3] or end(r[3]) in reach[r[3]] for r in acc):
                sets = []
                for c in cover_sets(acc):
                    ts = {}
                    for i in c:
                        ts[acc[i][3]] = ts.get(acc[i][3], False) or acc[i][2]
                    if len(ts) > 1 and any(ts.values()):
                        sets.append(ts)
                swept.append((key, sets))
                continue
            rest.append(key)
        pos = {n: i for i, n in enumerate(nx.topological_sort(g))} if swept else {}

        def sweep(ts):
            """Whether the sweep orders every pair with a writer among the tasks `ts` ({task: writes})."""
            last, readers = None, []
            for t in sorted(ts, key=pos.__getitem__):
                if ts[t]:
                    if any(t not in reach[end(p)] for p in ([last] if last is not None else []) + readers):
                        return False
                    last, readers = t, []
                else:
                    if last is not None and t not in reach[end(last)]:
                        return False
                    readers.append(t)
            return True

        def loop_ok(acc):
            """The pairwise loop's verdict on one buffer."""
            acc = sorted(acc, key=lambda r: (r[0], r[1]))
            for i, (lo, hi, w, t, _m, ru) in enumerate(acc):
                for lo2, hi2, w2, t2, _m2, ru2 in acc[i + 1:]:
                    if lo2 >= hi:
                        break
                    if t2 is t or not (w or w2):
                        continue
                    if (ru or ru2) and not _overlap(ru or (lo, hi - lo, ()), ru2 or (lo2, hi2 - lo2, ())):
                        continue
                    if not ordered(t, t2):
                        return False
            return True

        cross = bool(os.environ.get("BINGO_HAZARD_CROSSCHECK"))
        looped, sweep_pairs = set(rest), 0
        for key, sets in swept:
            ok = all(sweep(ts) for ts in sets)
            if cross:
                acc = bufs[key]
                lok = loop_ok(acc)
                nests = any(r[5] is not None for r in acc)
                if (ok and not lok) or (not nests and ok != lok):
                    raise AssertionError(f"hazard sweep on {names[key]}: the sweep says {'ordered' if ok else 'unordered'}, "
                                         f"the pairwise check {'ordered' if lok else 'unordered'}")
            if ok:
                sweep_pairs += pairs_with_writer(bufs[key])
            else:
                looped.add(key)                 # the loop decides, and names the pairs
        if cross and swept:
            print(f"[hazards] cross-check: the sweep and the pairwise check agree on {len(swept)} swept buffers")
        loop_keys = [k for k in bufs if k in looped]

        bad, seen, pairs = [], set(), 0
        for key in loop_keys:
            acc = bufs[key]
            acc.sort(key=lambda r: (r[0], r[1]))
            n_acc = len(acc)
            for i, (lo, hi, w, t, m, ru) in enumerate(acc):
                for j in range(i + 1, n_acc):           # by index: a slice would copy the rest of the list each time
                    lo2, hi2, w2, t2, m2, ru2 = acc[j]
                    if lo2 >= hi:
                        break
                    if t2 is t or not (w or w2):
                        continue
                    if (ru or ru2) and not _overlap(ru or (lo, hi - lo, ()), ru2 or (lo2, hi2 - lo2, ())):
                        continue
                    pair = (t, t2) if t.node_id < t2.node_id else (t2, t)
                    if pair in seen:
                        continue
                    seen.add(pair)
                    pairs += 1
                    if not ordered(t, t2):
                        first = (t, m, lo, hi) if t.node_id < t2.node_id else (t2, m2, lo2, hi2)
                        then = (t2, m2, lo2, hi2) if first[0] is t else (t, m, lo, hi)
                        bad.append((names[key], first, then))
        pairs += sweep_pairs
        problems = []
        for cls, ts in sorted(undeclared.items()):
            problems.append(f"{cls} declares no accesses() ({len(ts)} tasks, e.g. {ts[0].node_name})")
        for t, cls, f in uncovered[:SHOW]:
            problems.append(f"{cls}.{f} holds a buffer its accesses() does not declare ({t.node_name})")

        def rng(lo, hi):
            return f"[{lo}:{'end' if hi >= INF else hi}]"

        for name, (a, ma, la, ha), (b, mb, lb, hb) in bad[:SHOW]:
            problems.append(f"{a.node_name} ({ma} {name}{rng(la, ha)}) and {b.node_name} ({mb} {name}{rng(lb, hb)}) "
                            f"are not ordered by the graph: add {a.node_name} -> {b.node_name}, or the edge that "
                            f"orders them the other way")
        n_bad = len(bad) + len(uncovered) + sum(len(v) for v in undeclared.values())
        if enforce and n_bad:
            more = f" (and {n_bad - len(problems)} more)" if n_bad > len(problems) else ""
            raise ValueError(f"memory hazards: {n_bad} problem(s){more}:\n  " + "\n  ".join(problems))
        tail = "" if enforce else " (report only: the compiler does not order this graph's cores)"
        print(f"[hazards] {pairs} overlapping pairs with a writer on {len(bufs)} buffers, {len(bad)} unordered; "
              f"{sum(len(v) for v in undeclared.values())} tasks of kernels without accesses(), {len(uncovered)} "
              f"undeclared fields{tail}")
        for p in problems[:4] if not enforce else []:
            print(f"[hazards]   {p}")
        return bad
