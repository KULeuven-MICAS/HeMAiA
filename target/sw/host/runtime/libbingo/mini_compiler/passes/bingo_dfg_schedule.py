# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""Each device core's program, chosen by the compiler, and the asynchronous execution of the accelerators.

WHY THE COMPILER ORDERS THE CORES. BINGO dispatches a core's tasks into an in-order queue, so the order of each core's
program is fixed when the graph is compiled, and two tasks with no path between them still run one after the other.
Which goes first is a question of time -- whose inputs land sooner, whose result the rest of the graph waits for -- so
it is answered with the cost model (bingo_cost_model.py), not left to the order a generator happened to make its tasks
in.

THE SCHEDULE: list scheduling over the cost model. A core's next task is the one of its ready tasks that can start
earliest; among those that can start at once, the one with the longest path to the end of the graph (its rank); and
the core waits for a task of a longer path whose inputs land while that one would hold it, where the two in that order
bound the end sooner (a long put does not take the DM core from the ring load the next GEMM needs). Both, with that
lookahead and without, are laid out, and the shorter kept: the lookahead's local bound can be wrong. Every
core runs in order and every task starts after its dependencies, its ordering-only edges (bingo_add_order_edge) and the
edge latency: the one timeline all cores share is a topological order of the graph, so the programs it yields close no
cycle. A weight-ring load also waits for its chunk, which its chip's prefetcher pushes from the time it starts, and a
cluster's ring loads keep the order they were made in (the ring's: a load ahead of an earlier one can block its DM core
on a chunk the prefetcher pushes only once that earlier one has freed its slot).

L1. With a static L1 plan (static_l1), the programs must leave the packer (bingo_l1_packer.py) room: two buffers share
space only when every user of one precedes every user of the other along the edges and the programs, and a schedule
that starts every load as early as it can leaves few such pairs. Where the plan overflows, the model is scheduled
again with a budget (_L1Gate): a task about to open a buffer where its cluster's buffers that cannot share with it
would pass the budget gets an edge from the last users of buffers that are done with (on its own cluster, the ones
that ended first), so the packer may give it their space; where those are not enough it waits. The budget starts at
the capacity and drops by what the packed plan still overflows (no less than 1/L1_STEPS of it, the packer's gaps).
A schedule that overflows by more than the capacity itself is not held back (its loads would have to wait for most of
what runs beside them), and the tries stop where one leaves more than half of what the try before overflowed by;
where no schedule fits, each core keeps the order its tasks were made in, as an in-order core runs it: a
task that waits is not passed by the core's later ones (only where that order closes a cycle), under the same budget.

ASYNCHRONOUS EXECUTION (dfg.async_exec). An accelerator driven through a CSR manager and a streamer -- VersaCore, the
SIMD core's, the xDMA -- can take a task as parts: a CONFIGURE (every CSR written, nothing started), a START (the start
write, once the operands are there), a WAIT (returns once the work is done). Which kernels can, and how, each kernel
declares (BingoKernelArgs.split(): the parts, how completion is tracked, how many works may be in flight, whether
another kernel may run meanwhile). The schedule places each part like any other task, under the rules every such
accelerator shares:

  - a configure follows the previous work's start: the CSR manager holds one pending configuration, and no other task
    that writes the configuration goes between a configure and its start;
  - at most `depth` works are started and not yet waited for;
  - an `exclusive` accelerator runs no other kernel's task while one of these is in flight;
  - a wait is ready when its work is done.

So a wait trails the next start exactly when that start can go before the accelerator finishes, and a configure runs
ahead of the waits before its start when the operands are later than the core is free. Parts the schedule left side
by side gain nothing and are fused again: a configure right before its start, run no earlier than the start's operands,
is one start that configures; a start right before its wait, configured in it, is the task as written, and configured
ahead, a start that waits for its own work where the kernel has one (Split.drain). Which accelerators' tasks run split
is the model's choice, one accelerator at a time (VersaCore, then the xDMA, then the SIMD core): its splits are kept
only where they make the whole program finish sooner (every part that stays its own task costs its core a task's
dispatch; parts the materialisation fuses, one: _Timeline.merged). async_exec "all" keeps every split (the tests of
the parts). Then:

  - each counter-tracked accelerator's works are numbered in its core's program (the wait compares the retired counter
    with the number), every work the counter counts taking a number, a whole one too (dispatch_accel());
  - a part that shares a per-cluster setup (the split's prologue: VersaCore's shape table) gets it, one task a cluster.

CSR REUSE. Along each accelerator core's program, a task whose configuration (config_key()) the CSRs already hold runs
as its cheaper form (config_reuse(): gemm_full -> gemm_minimal, a start's reuse flag); a softmax whose arena an earlier
softmax on its core set up skips the geometry, and one right after a softmax of the same geometry its shared CSRs; a
ChainGather wait right before the next one's configure skips its clean-up.

What a generator writes is one task per operation and its dependencies; this pass is where the queues, the numbers and
the configure / start / wait parts come from."""

import copy
import heapq

import networkx as nx

import bingo_cost_model as cm
from bingo_kernel_args import (KEEPS, HostBingoKernelWeightPrefetchArgs, SnaxBingoKernelSimdFaSoftmaxArgs,
                               SnaxBingoKernelXdmaChainWaitArgs)
from bingo_mem_handle import BingoMemAlloc, BingoMemAllocView
from bingo_node import BingoNode

GEMM_WAIT = "__snax_bingo_kernel_gemm_wait"
SOFTMAX = "__snax_bingo_kernel_simd_fa_softmax"
CHAIN_CFG = "__snax_bingo_kernel_xdma_chain_cfg"
ACCELS = ("gemm", "simd", "xdma")
L1_TRIES = 4            # schedules with the L1 budget, each lowered by what the static plan still overflows
L1_SLACK = 4096         # bytes below that overflow
L1_STEPS = 32           # and by no less than this fraction of the capacity (the packer's gaps)


class _Part:
    """A part of a split task inside the scheduler: "cfg", "start" or "wait" of the task `node`."""
    __slots__ = ("node", "kind")

    def __init__(self, node, kind):
        self.node, self.kind = node, kind

    def __repr__(self):
        return f"{self.node.node_name}:{self.kind}"


class _Model:
    """The graph the scheduler works on: every task a vertex, a split one its parts, with each vertex's place, time
    and role, and the edges with their latencies."""

    def __init__(self, dfg, roles, splits, keep_order=False):
        self.roles = roles
        self.host = roles["host"]
        self.dev = {roles[r] for r in ("gemm", "simd", "xdma", "dm")}
        self.accel_core = {roles[a]: a for a in ACCELS}
        self.splits = splits                # task -> Split
        self.parts = {}                     # task -> (cfg part or None, start part, wait part or None)
        self.vertices, self.idx, self.place, self.dur, self.eng, self.role = [], {}, {}, {}, {}, {}
        for n in sorted(dfg.node_list, key=lambda t: t.node_id):
            sp = splits.get(n)
            if sp is not None:
                ps = (_Part(n, "cfg") if sp.configure else None, _Part(n, "start"),
                      _Part(n, "wait") if sp.wait else None)
                self.parts[n] = ps
                for p in ps:
                    if p is not None:
                        self._add(p, n, sp)
            else:
                self._add(n, n, None)
        self.preds = {v: [] for v in self.vertices}
        self.succs = {v: [] for v in self.vertices}
        self.l1_edges = []                  # the edges the L1 gate added (schedule's l1)
        edges = list(dfg.edges()) + list(dfg.order_edges)
        for u, v in edges:
            self._edge(self._out(u), self.parts[v][1] if v in self.parts else v)
        # the ordering-only edges as vertex pairs: the static L1 plan orders their producer before the consumer's
        # successors only (bingo_liveness.with_spin_order)
        vx = lambda u, v: (self._out(u), self.parts[v][1] if v in self.parts else v)
        self.spin = {vx(u, v) for u, v in dfg.order_edges} - {vx(u, v) for u, v in dfg.edges()}
        self.phase_load = _phase_loads(dfg)
        for x, (c, s, w) in self.parts.items():
            if c is not None:
                self._edge(c, s, 0)
                # a configure reads its args too: after the load of their args phase (an application's phases)
                ld = self.phase_load.get((x.assigned_chiplet_id, x.assigned_cluster_id, getattr(x, "args_phase", 0)))
                if ld is not None:
                    self._edge(self._out(ld), c, 0)
            if w is not None:
                self._edge(s, w, 0)
        # a ring load waits for its chunk, which the memory chip pushes once the chip's prefetcher runs (the part whose
        # chunk range holds it, when the prefetcher runs in parts), or once the ring's cluster posts its run
        self.prefetcher, self.pf_of, self.arrival = {}, {}, {}
        for n in sorted(dfg.node_list, key=lambda t: t.node_id):
            if isinstance(n.kernel_args, HostBingoKernelWeightPrefetchArgs) or (n.kernel_name or "").startswith(
                    cm.WEIGHT_PREFETCH):
                self.prefetcher.setdefault(n.assigned_chiplet_id, []).append(n)

        def flags(a):
            """The rings' flag words a prefetcher fills or a ring load polls: which rings they are."""
            f = getattr(a, "flag_addr", None) if a is not None else None
            return getattr(getattr(f, "base", f), "symbol_name", None) or getattr(f, "name", None)

        def pushes(p, v):
            a = p.kernel_args
            if flags(a) is not None and flags(a) != flags(v.kernel_args):
                return False                    # another set of rings (another block's prefetcher)
            chunk = getattr(v, "exec_chunk", None)
            lo, hi = getattr(a, "chunk_lo", 0), getattr(a, "chunk_hi", 0)
            return chunk is None or (lo <= chunk[1] and (not hi or chunk[1] < hi))
        loads = {}
        for v in self.vertices:
            if not (isinstance(v, BingoNode) and v.kernel_name == cm.RING_LOAD):
                continue
            if getattr(v, "ring_start", None) is not None:
                # a ring whose cluster posts its own runs (WeightRings cluster mode): the load follows its run's push
                # by an edge, and the engine streams the ring's runs from the first push on
                self.pf_of[v] = [v.ring_start]
            elif v.assigned_chiplet_id in self.prefetcher:
                pfs = self.prefetcher[v.assigned_chiplet_id]
                self.pf_of[v] = [p for p in pfs if pushes(p, v)] or pfs[:1]
            else:
                continue
            loads.setdefault((v.assigned_chiplet_id, v.assigned_cluster_id), []).append(v)
        rings = {}
        for (k, _c) in loads:
            rings[k] = rings.get(k, 0) + 1
        for (k, c), ls in loads.items():
            sent = 0
            for i, v in enumerate(ls):
                sent += int(getattr(v.kernel_args, "size", 0))
                self.arrival[v] = cm.PUSH_LATENCY + sent * rings[k] // cm.PUSH_BYTES_PER_CYCLE
                for p in self.pf_of[v]:
                    self._edge(p, v, 0)
                if i:
                    # a cluster's ring loads in the ring's order: a load blocks its DM core until its chunk lands, and
                    # the prefetcher pushes a chunk only once the load of the chunk a ring's length before it has
                    # freed the slot, so a load ahead of an earlier one can wait for good
                    self._edge(ls[i - 1], v, 0)
        if keep_order:
            # every device core's tasks in the order they were made: each task starts after the start of the one
            # before it, so a split task's wait may still trail (_made_order)
            for vs in self._made_order().values():
                for a, b in zip(vs, vs[1:]):
                    self._edge(a, b, 0)

    def _made_order(self):
        """{device core: its tasks (a split task's start) in the order they were made}, as in-order cores run them: a
        core's next task waits until it is ready, and the core's later tasks do not pass it (they would open their
        buffers while the ones it waits behind are still live). Only where every core's next task waits for one a core
        made later (the made order closes a cycle with the edges, the rings' included) does the earliest made task that
        is ready pass the tasks ahead of it. A topological order of the edges, so the chains close no cycle."""
        node = lambda v: v.node if isinstance(v, _Part) else v       # noqa: E731
        key = {v: (node(v).node_id, self.idx[v]) for v in self.vertices}
        chained = lambda v: self.role[v] not in ("cfg", "wait") and self.on_core(v)   # noqa: E731
        queue = {}
        for v in sorted(self.vertices, key=key.get):
            if chained(v):
                queue.setdefault(self.place[v], []).append(v)
        head = {p: 0 for p in queue}
        left = {v: len(self.preds[v]) for v in self.vertices}
        done, free, ready = set(), [], set()        # ready: chained tasks whose edges are met
        hot = list(queue)                           # cores whose next task may have become ready
        for v in self.vertices:
            if not left[v]:
                (ready.add(v) if chained(v) else free.append(v))
        out = {p: [] for p in queue}

        def emit(v):
            done.add(v)
            if chained(v):
                ready.discard(v)
                out[self.place[v]].append(v)
                hot.append(self.place[v])
            for w, _ in self.succs[v]:
                left[w] -= 1
                if not left[w]:
                    if chained(w):
                        ready.add(w)
                        hot.append(self.place[w])
                    else:
                        free.append(w)

        def head_of(p):
            q, i = queue[p], head[p]
            while i < len(q) and q[i] in done:
                i += 1
            head[p] = i
            return q[i] if i < len(q) else None
        # what runs before every core's next task waits is the same in any order: free tasks and ready heads as they
        # come, each core looked at again when one of its tasks becomes ready or it runs one
        while len(done) < len(self.vertices):
            if free:
                emit(free.pop())
                continue
            if hot:
                h = head_of(hot.pop())
                if h is not None and h in ready:
                    emit(h)
                continue
            if not ready:
                raise ValueError("schedule: the task graph's edges close a cycle")
            emit(min(ready, key=key.get))           # every core's next task waits: the earliest ready one passes
        return out

    def fork(self):
        """A copy whose edges may grow (the L1 gate's) without touching this one's."""
        f = copy.copy(self)
        f.preds = {v: list(p) for v, p in self.preds.items()}
        f.succs = {v: list(p) for v, p in self.succs.items()}
        f.l1_edges = []
        return f

    def _out(self, n):
        """The vertex a task's successors follow: its wait, else its start."""
        if n not in self.parts:
            return n
        c, s, w = self.parts[n]
        return w if w is not None else s

    def _add(self, v, n, sp):
        self.idx[v] = len(self.vertices)
        self.vertices.append(v)
        self.place[v] = (n.assigned_chiplet_id, n.assigned_cluster_id, n.assigned_core_id)
        self.eng[v] = 0
        if isinstance(v, _Part):
            self.role[v] = v.kind
            if v.kind == "cfg":
                self.dur[v] = cm.configure_cycles(sp.configure[1] if sp.configure else None, sp.accel)
            elif v.kind == "start":
                self.dur[v] = cm.GEMM_START
                self.eng[v] = cm.engine_cycles(n.kernel_args, sp.accel)
            else:
                self.dur[v] = cm.GEMM_WAIT
        elif cm.is_blocking_gemm(n) and n.assigned_core_id in self.dev:
            self.eng[v] = cm.array_cycles(n.kernel_args)
            self.role[v] = "block"
            self.dur[v] = cm.GEMM_CONFIGURE
        else:
            self.role[v] = None
            self.dur[v] = cm.task_cycles(n)
            # a whole SIMD or xDMA task holds its accelerator for its work, as a blocking GEMM holds the array
            accel = self.accel_core.get(n.assigned_core_id)
            k = n.kernel_name or ""
            if accel in ("simd", "xdma") and k.startswith(f"__snax_bingo_kernel_{accel}"):
                work = cm.engine_cycles(n.kernel_args, accel)
                if work > 0:
                    self.role[v] = "block"
                    self.eng[v] = work
                    self.dur[v] = max(self.dur[v] - work, 0)
        if n.assigned_core_id in self.dev:
            self.dur[v] += cm.TASK_DISPATCH     # every device task's dispatch, its parts' each

    def _edge(self, u, v, lat=None):
        if lat is None:
            lat = cm.edge_latency(self.place[u], self.place[v], self.host)
        self.preds[v].append((u, lat))
        self.succs[u].append((v, lat))

    def on_core(self, v):
        """Whether `v` holds an in-order device core (a host task runs beside the rest of its chip's)."""
        return self.place[v][2] in self.dev

    def split_of(self, v):
        return self.splits[v.node] if isinstance(v, _Part) else None

    def drains(self, v):
        """A start that returns only once its work is done (a split without a wait)."""
        return self.role[v] == "start" and self.parts[v.node][2] is None

    def configures(self, v):
        """Whether `v` writes its accelerator's configuration: a configure, the start of a split that has none (the
        softmax's fire), a whole task on an accelerator core that does not declare it keeps the CSRs as they were."""
        if self.role[v] in ("cfg", "block"):
            return True
        if self.role[v] == "start":
            return self.parts[v.node][0] is None     # a split without a configure part: its start writes them
        if self.role[v] is not None or self.place[v][2] not in self.accel_core:
            return False
        a = v.kernel_args
        return a is None or a.config_key() != KEEPS

    def ranks(self):
        """Each vertex's longest path to the end of the graph, its own time and every edge's latency counted (a
        start's path to its wait: its work's time)."""
        g = nx.DiGraph()
        g.add_nodes_from(self.vertices)
        g.add_edges_from((u, v) for u in self.vertices for v, _ in self.succs[u])
        rank = {}
        for v in reversed(list(nx.topological_sort(g))):
            whole = self.role[v] == "block" or self.drains(v)
            own = self.dur[v] + (self.eng[v] if whole else 0)
            rank[v] = own + max((lat + (self.eng[v] if self.role[v] == "start" and not whole else 0) + rank[s]
                                 for s, lat in self.succs[v]), default=0)
        return rank


class _Timeline:
    """The machine's state while a schedule is laid out: when each core and each accelerator is free, the works
    started and not yet waited for, the configuration waiting for its start."""

    def __init__(self, model):
        self.m = model
        self.core_free, self.eng_free = {}, {}
        self.outstanding, self.pending, self.excl = {}, {}, {}
        self.start, self.fin, self.done = {}, {}, {}
        self.dr = {v: 0 for v in model.vertices}
        self.last = {}                      # core -> the vertex it ran last
        self.whole = set()                  # starts that will be one task with their configure

    def merged(self, v, c):
        """Whether `v` will be one task with the part before it on its core (what _materialize fuses), so costs its
        core no dispatch of its own: a start right after its configure that ran no earlier than the start's operands,
        a wait right after a start that configured in it (the task as written) or that can wait for its own work."""
        m, prev = self.m, self.last.get(c)
        if not isinstance(v, _Part) or not isinstance(prev, _Part) or prev.node is not v.node:
            return False
        cfg, s, w = m.parts[v.node]
        sp = m.split_of(v)
        if m.role[v] == "start":
            inputs = max((self.fin[u] + lat for u, lat in m.preds[v] if u is not cfg), default=0)
            if prev is cfg and self.start[cfg] >= inputs and sp.start(False) is not None:
                self.whole.add(v)
                return True
            return False
        if m.role[v] == "wait" and prev is s:
            return cfg is None or s in self.whole or sp.drain is not None
        return False

    def lay(self, v, est):
        """Run `v` from `est` (no earlier than its core is free and its inputs are there). Returns its end."""
        m, role, c = self.m, self.m.role[v], self.m.place[v]
        own = m.dur[v] - (cm.TASK_DISPATCH if m.on_core(v) and self.merged(v, c) else 0)
        if role in ("block", "start"):
            setup = est + own
            arr = max(setup, self.eng_free.get(c, 0))
            self.eng_free[c] = arr + m.eng[v]
            if role == "block" or m.drains(v):
                end = arr + m.eng[v]
                self.pending[c] = None
            else:
                end = setup
                self.done[v] = arr + m.eng[v]
                self.outstanding[c] = self.outstanding.get(c, 0) + 1
                self.pending[c] = None
                if m.split_of(v).exclusive:
                    self.excl[c] = (self.excl.get(c, (0, None))[0] + 1, v.node.kernel_name)
        else:
            end = est + own
            if role == "cfg":
                self.pending[c] = m.parts[v.node][1]
            elif role == "wait":
                self.outstanding[c] -= 1
                if m.split_of(v).exclusive:
                    n, k = self.excl[c]
                    self.excl[c] = (n - 1, k if n > 1 else None)
        self.start[v], self.fin[v] = est, end
        if m.on_core(v):
            self.core_free[c] = end
            self.last[c] = v
        return end

    def release(self, v, end):
        """What `v` ending at `end` tells its successors: a successor's inputs are there no earlier."""
        m = self.m
        for s, lat in m.succs[v]:
            t = end + lat
            if m.role[s] == "wait" and m.role[v] == "start":
                t = max(t, self.done[v])
            if s in m.arrival and v in m.pf_of.get(s, ()):
                t = self.start[v] + m.arrival[s]
            if t > self.dr[s]:
                self.dr[s] = t

    def makespan(self):
        return max(self.fin.values(), default=0)


def schedule_best(model, l1=None):
    """schedule() with and without its lookahead, the shorter (a model whose edges grow, the L1 gate's, is forked for
    each). Returns (seqs, timeline, model)."""
    runs = []
    for look in (False, True):
        m = model.fork() if l1 is not None else model
        sq, tl = schedule(m, l1, look)
        runs.append((tl.makespan(), look, sq, tl, m))
    _ms, _look, sq, tl, m = min(runs, key=lambda r: (r[0], r[1]))
    return sq, tl, m


def schedule(model, l1=None, lookahead=True):
    """List-schedule `model`: {device core: [vertex, ...]} in program order, and the timeline. With l1 = (budget,
    buffers) (_l1_buffers), a task that opens an L1 buffer waits while its cluster's buffers that cannot share space
    with it (see _L1Gate) would pass `budget` bytes; when nothing else can run, the waiting one of the longest path
    goes anyway. With lookahead, a core waits for a task of a longer path that lands while the ready one would hold it,
    where that bounds the end sooner (peek)."""
    m = model
    tl = _Timeline(m)
    gate = _L1Gate(m, *l1) if l1 is not None else None
    l1_wait = {}                            # vertex -> (core, clusters it opens a buffer in)
    forced = set()
    rank = m.ranks()
    npred = {v: len(m.preds[v]) for v in m.vertices}
    future, avail, version = {}, {}, {}
    gated = {}                              # core -> vertices that wait for its accelerator's state
    heap, seqs = [], {}

    def key(v):
        return (-rank[v], m.idx[v])

    def occ(v):
        """How long `v` holds its core: a whole accelerator task, or a start that drains, until its work is done."""
        return m.dur[v] + (m.eng[v] if m.role[v] == "block" or m.drains(v) else 0)

    def peek(c):
        f, a, free = future.get(c, []), avail.setdefault(c, []), tl.core_free.get(c, 0)
        while f and f[0][0] <= free:
            dr, _r, _i, v = heapq.heappop(f)
            heapq.heappush(a, (*key(v), dr, v))
        if a:
            v = a[0][-1]
            # a task of a longer path whose inputs land while v would hold the core: the core waits for it where the
            # two in that order bound the end sooner (each one's rank: its path to the end, its own time counted)
            end_v, best = free + occ(v), None
            for dr_u, _r, _i, u in (f if lookahead else ()):
                if dr_u < end_v and rank[u] > rank[v]:
                    vu = max(free + rank[v], end_v + rank[u])
                    uv = max(dr_u + rank[u], dr_u + occ(u) + rank[v])
                    if uv < vu and (best is None or uv < best[0]):
                        best = (uv, dr_u, u)
            if best is not None:
                return best[1], best[2]
            return free, v
        if f:
            return f[0][0], f[0][-1]
        return None

    def refresh(c):
        version[c] = version.get(c, 0) + 1
        p = peek(c)
        if p is not None:
            heapq.heappush(heap, (p[0], *key(p[1]), version[c], c))

    def accel_ok(v):
        """The accelerator's rules: the pending configuration, the works in flight, an exclusive kernel's."""
        c, role = m.place[v], m.role[v]
        if m.configures(v) and tl.pending.get(c) is not None:
            return False
        sp = m.split_of(v)
        if role == "start" and tl.outstanding.get(c, 0) >= sp.depth:
            return False                    # a configure may go ahead: only the start waits for room
        n, k = tl.excl.get(c, (0, None))
        if n and (sp is None or v.node.kernel_name != k):
            return False
        return True

    touched = set()
    work = []                               # vertices whose inputs are all decided, to offer
    state = {}                              # vertex -> "queued", "gated" or "done"

    def ready(v):
        """Every input of `v` decided: a configure, once its start waits for it alone."""
        if m.role[v] == "cfg":
            return npred[v] == 0 and npred[m.parts[v.node][1]] == 1 and v not in tl.fin
        return npred[v] == 0

    def hold(v):
        if state.get(v) != "gated":
            state[v] = "gated"
            gated.setdefault(m.place[v], []).append(v)

    def offer(v):
        """Queue `v` on its core, hold it for its accelerator, or run a host task now (its successors then follow)."""
        if state.get(v) in ("queued", "done"):
            return
        if m.on_core(v) and not accel_ok(v):
            hold(v)
            return
        state[v] = "queued"
        if not m.on_core(v):
            state[v] = "done"
            end = tl.lay(v, tl.dr[v])
            tl.release(v, end)
            decided(v)
            l1_ran(v, None)
            return
        dr = tl.dr[v]
        if m.role[v] == "cfg":
            dr = max(0, tl.dr[m.parts[v.node][1]] - m.dur[v])
        c = m.place[v]
        heapq.heappush(future.setdefault(c, []), (dr, *key(v), v))
        touched.add(c)

    def decided(v):
        for s, _ in m.succs[v]:
            npred[s] -= 1
            if m.role[s] == "start" and npred[s] == 1 and m.parts[s.node][0] is not None:
                cfg = m.parts[s.node][0]
                if cfg not in tl.fin:
                    work.append(cfg)
            elif m.role[s] == "cfg":
                if npred[s] == 0:
                    work.append(s)              # ready() checks its start too
            elif npred[s] == 0:
                work.append(s)

    def l1_ran(v, c):
        """`v` ran (on core c, None for a host task): the waiting tasks it may let go -- a waiting task's clock moves
        with its core's program, and a cluster's buffers free as their last users run."""
        if gate is None:
            return
        closed = gate.commit(v)
        for u, (uc, cls) in list(l1_wait.items()):
            if uc == c or cls & closed:
                del l1_wait[u]
                state.pop(u, None)
                work.append(u)

    def drain():
        while work:
            v = work.pop()
            if ready(v):
                offer(v)

    def ungate(c):
        for v in gated.pop(c, []):
            state.pop(v, None)
            work.append(v)

    work.extend(v for v in m.vertices if ready(v))
    drain()
    for c in list(touched):
        refresh(c)
    touched.clear()
    while heap or l1_wait:
        if not heap:
            u = min(l1_wait, key=key)       # nothing else can run: the one of the longest path goes over the budget
            del l1_wait[u]
            state.pop(u, None)
            forced.add(u)
            work.append(u)
            drain()
            for t in list(touched):
                refresh(t)
            touched.clear()
            continue
        est, _r, _i, ver, c = heapq.heappop(heap)
        if ver != version.get(c):
            continue
        p = peek(c)
        if p is None:
            continue
        est, v = p
        if avail[c] and avail[c][0][-1] is v:
            heapq.heappop(avail[c])
        elif future[c][0][-1] is v:
            heapq.heappop(future[c])
        else:                               # a later task the core waits for (peek)
            fl = future[c]
            fl.pop(next(i for i, e in enumerate(fl) if e[-1] is v))
            heapq.heapify(fl)
        if not accel_ok(v):
            state.pop(v, None)
            hold(v)                         # the accelerator's state changed since it was queued
            refresh(c)
            continue
        if gate is not None and v not in forced and not (m.role[v] == "start" and m.parts[v.node][0] in tl.fin):
            # a configure takes its accelerator for its start: it waits for the start's buffers, the start then goes
            t = m.parts[v.node][1] if m.role[v] == "cfg" else v
            plan = gate.plan(t, tl.fin)
            if plan is not None and plan[0] == "edges":
                for u in plan[1]:
                    m._edge(u, t)
                    m.l1_edges.append((u, t))
                    tl.dr[t] = max(tl.dr[t], tl.fin[u] + m.preds[t][-1][1])
                state.pop(v, None)
                work.append(v)              # offered again, no earlier than those users
                drain()
                touched.add(c)
                for x in list(touched):
                    refresh(x)
                touched.clear()
                continue
            if plan is not None:
                state[v] = "l1"
                l1_wait[v] = (c, plan[1])
                refresh(c)
                continue
        state[v] = "done"
        end = tl.lay(v, max(est, tl.dr[v]))
        seqs.setdefault(c, []).append(v)
        tl.release(v, end)
        decided(v)
        if m.role[v] in ("start", "wait", "block") or m.configures(v):
            ungate(c)
        l1_ran(v, c)
        drain()
        touched.add(c)
        for t in list(touched):
            refresh(t)
        touched.clear()
    tl.l1_forced = len(forced)
    left = [v for v in m.vertices if v not in tl.fin]
    if left:
        raise ValueError(f"schedule: {len(left)} tasks never became ready, e.g. {left[:4]}")
    return seqs, tl


def _phase_loads(dfg):
    """(chip, cluster, args phase) -> the task that loads that phase's args (an application's args phases)."""
    out = {}
    for n in dfg.node_list:
        a = n.kernel_args
        if getattr(n, "args_resident", False) and hasattr(a, "phase") and hasattr(a, "cluster"):
            out[(n.assigned_chiplet_id, n.assigned_cluster_id, int(a.phase))] = n
    return out


class _L1Gate:
    """The static L1 plan's view of a schedule being laid out. Two buffers share space only when every user of one
    precedes every user of the other along the graph's edges and the programs (bingo_liveness.can_share). Every
    device core's program is a chain, so what precedes a task is a prefix of each core's program: a clock, one
    position a core. A buffer whose users have all run frees its space for a task whose clock covers every one of
    them; a buffer with a user off the cores (a host task) never does. Where a task's clock does not cover them (no
    path from the last readers of the space to the next writer), an edge from each last user, on the task's own
    cluster, makes it so: the order a generator once wrote between them by hand."""

    def __init__(self, model, budget, buffers):
        import numpy as np
        self.np, self.m, self.budget = np, model, budget
        cores = sorted({model.place[v] for v in model.vertices if model.on_core(v)})
        self.ci = {c: i for i, c in enumerate(cores)}
        self.zero = np.zeros(len(cores), dtype=np.int64)
        self.cl, self.size = [b[0] for b in buffers], [b[1] for b in buffers]
        self.uses = {}
        for b, (_cl, _sz, users) in enumerate(buffers):
            for u in users:
                self.uses.setdefault(u, []).append(b)
        self.left = [len(b[2]) for b in buffers]
        self.opened = [False] * len(buffers)
        self.need = [{} for _ in buffers]          # core index -> the last position of a user on that core
        self.blind = [False] * len(buffers)
        self.alive = {}                            # cluster -> buffers opened
        self.clock, self.last, self.n = {}, {}, {}
        self.seq = {}                              # core index -> its vertices so far

    def _clock(self, v):
        """What precedes `v` if it runs next on its core: the latest position of each core."""
        np, m = self.np, self.m
        cs = [self.clock[p] for p, _ in m.preds[v] if p in self.clock]
        if m.on_core(v) and m.place[v] in self.last:
            cs.append(self.clock[self.last[m.place[v]]])
        return np.max(cs, axis=0) if cs else self.zero

    def _free(self, b, clk):
        return not self.left[b] and not self.blind[b] and all(clk[i] >= p for i, p in self.need[b].items())

    def plan(self, v, fin):
        """What `v`, opening its buffers now, needs to stay within the budget: None (it fits), ("edges", [u, ...]) --
        the last users, on v's cluster, of buffers whose users have all run, so that an edge from each frees their
        space for v (the ones that ended first) -- or ("wait", clusters) where even those do not make room."""
        new = [b for b in self.uses.get(v, ()) if not self.opened[b]]
        if not new:
            return None
        m = self.m
        clk, here = self._clock(v), m.place[v][:2]
        edges, waits = set(), set()
        for cl in {self.cl[b] for b in new}:
            held = [b for b in self.alive.get(cl, ()) if not self._free(b, clk)]
            live = sum(self.size[b] for b in new if self.cl[b] == cl) + sum(self.size[b] for b in held)
            if live <= self.budget:
                continue
            cands = []
            for b in held:
                if self.left[b] or self.blind[b]:
                    continue
                us = [self.seq[i][p - 1] for i, p in self.need[b].items() if clk[i] < p]
                if all(m.place[u][:2] == here for u in us):
                    cands.append((max(fin[u] for u in us), b, us))
            for _f, b, us in sorted(cands, key=lambda c: (c[0], c[1])):
                if live <= self.budget:
                    break
                live -= self.size[b]
                edges.update(us)
            if live > self.budget:
                waits.add(cl)
        if waits:
            return "wait", frozenset(waits)
        if edges:
            return "edges", sorted(edges, key=lambda u: m.idx[u])
        return None

    def commit(self, v):
        """`v` runs: its clock, and its buffers opened and counted down. Returns the clusters where a buffer's last
        user ran."""
        m = self.m
        clk = self._clock(v).copy()
        on = m.on_core(v)
        if on:
            c = m.place[v]
            i = self.ci[c]
            self.n[c] = self.n.get(c, 0) + 1
            clk[i] = self.n[c]
            self.last[c] = v
            self.seq.setdefault(i, []).append(v)
        self.clock[v] = clk
        closed = set()
        for b in self.uses.get(v, ()):
            if not self.opened[b]:
                self.opened[b] = True
                self.alive.setdefault(self.cl[b], []).append(b)
            if on:
                i = self.ci[m.place[v]]
                self.need[b][i] = max(self.need[b].get(i, 0), self.n[m.place[v]])
            else:
                self.blind[b] = True
            self.left[b] -= 1
            if not self.left[b]:
                closed.add(self.cl[b])
        return closed


def _verts(model, n):
    """A task's vertices that use its buffers: a split task's start and wait (its configure is no user)."""
    if n in model.parts:
        _c, s, w = model.parts[n]
        return [s] if w is None else [s, w]
    return [n]


def _l1_buffers(dfg, model, opts):
    """The static L1 plan's buffers, each [(chip, cluster), its packed footprint, {the vertices that use it}]."""
    from bingo_l1_packer import StaticL1Options, _footprint
    from bingo_liveness import collect_handle_users
    opts = opts if isinstance(opts, StaticL1Options) else StaticL1Options(enable=True)
    nodes = sorted(dfg.node_list, key=lambda t: t.node_id)
    out = []
    for _hid, (h, us) in collect_handle_users(nodes).items():
        if h.mem_level == "L1":
            out.append(((h.chip_id, h.cluster_id), _footprint(h.size, opts.alignment, opts.guard_bytes),
                        {v for n in us for v in _verts(model, n)}))
    return out


def _l1_cap(dfg, model):
    """The static L1 plan's capacity a cluster: the DFG's l1_capacity_bytes, or with l1_capacity_auto what the L1 heap
    leaves beside everything else a cluster keeps there with this model's tasks (bingo_args_phases.static_capacity)."""
    if not getattr(dfg, "l1_capacity_auto", False):
        return getattr(dfg, "l1_capacity_bytes", None)
    if getattr(model, "l1_cap", None) is None:
        from bingo_args_phases import static_capacity
        from bingo_platform import core_roles
        roles = getattr(dfg, "core_roles", None) or core_roles()
        counts = {}
        for v in model.vertices:
            if model.on_core(v):
                key = model.place[v][:2]
                counts[key] = counts.get(key, 0) + 1
        model.l1_cap = static_capacity(dfg, counts, {roles[r] for r in ("gemm", "simd", "xdma", "dm")})
    return model.l1_cap


def _l1_over(dfg, model, seqs, opts):
    """By how many bytes the static L1 plan (bingo_l1_packer.pack, with the DFG's capacity) overflows its fullest
    cluster with these programs (<= 0: it fits): a buffer's users as the scheduler's vertices (a split task's start
    and wait), ordered by the model's edges and the programs. 0 without a static plan or a capacity."""
    cap = _l1_cap(dfg, model)
    if not opts or not cap:
        return 0
    from bingo_l1_packer import StaticL1Options, collect_placement_order, pack
    from bingo_liveness import collect_handle_users
    from bingo_liveness import with_spin_order
    from bingo_reach import reach_sets
    opts = opts if isinstance(opts, StaticL1Options) else StaticL1Options(enable=True)
    nodes = sorted(dfg.node_list, key=lambda t: t.node_id)
    users = {hid: (h, {v for n in us for v in _verts(model, n)}) for hid, (h, us) in collect_handle_users(nodes).items()}
    spin = getattr(model, "spin", ())
    g = nx.DiGraph()
    g.add_nodes_from(model.vertices)
    g.add_edges_from((u, v) for u in model.vertices for v, _ in model.succs[u] if (u, v) not in spin)
    for seq in seqs.values():
        g.add_edges_from(zip(seq, seq[1:]))
    g = with_spin_order(g, spin, g.successors)
    desc = reach_sets(g)
    for v in model.vertices:
        desc.setdefault(v, frozenset())
    placement, stats = pack(users, desc, "L1", constraints=collect_placement_order(nodes), opts=opts, capacity=cap)
    model.l1_plan = ({hid: off for hid, (_h, off) in placement.items()}, stats)    # the static L1 plan's start
    return max((st["peak"] - cap for st in stats.values()), default=0)


def _l1_live_peak(dfg, model, tl, opts):
    """The most bytes of L1 buffers a cluster holds at one time of the timeline `tl`, the fullest cluster's: a lower
    bound of what the static L1 plan packs these programs into. Two buffers ordered by the edges and the programs are
    never live at one time (every user of the one ends before any user of the other starts), so buffers live at one time
    may not share space. 0 without a static plan."""
    if not opts:
        return 0
    events = {}
    for key, size, users in _l1_buffers(dfg, model, opts):
        us = [u for u in users if u in tl.fin]
        if us:
            ev = events.setdefault(key, [])
            ev.append((min(tl.start[u] for u in us), size))
            ev.append((max(tl.fin[u] for u in us), -size))
    peak = 0
    for ev in events.values():
        live = 0
        for _t, d in sorted(ev, key=lambda e: (e[0], e[1])):     # a buffer that ends frees before one that starts
            live += d
            peak = max(peak, live)
    return peak


def _l1_fits(dfg, model, seqs, opts):
    """Whether the static L1 plan fits every cluster with these programs (_l1_over)."""
    return _l1_over(dfg, model, seqs, opts) <= 0


def evaluate(model, seqs):
    """The timeline of fixed programs {device core: [vertex, ...]} under the same model, or None when the programs and
    the graph's edges close a cycle."""
    m = model
    tl = _Timeline(m)
    g = nx.DiGraph()
    g.add_nodes_from(m.vertices)
    g.add_edges_from((u, v) for u in m.vertices for v, _ in m.succs[u])
    for seq in seqs.values():
        g.add_edges_from(zip(seq, seq[1:]))
    try:
        order = list(nx.topological_sort(g))
    except nx.NetworkXUnfeasible:
        return None
    for v in order:
        est = tl.dr[v]
        if m.on_core(v):
            est = max(est, tl.core_free.get(m.place[v], 0))
        end = tl.lay(v, est)
        tl.release(v, end)
    return tl


class BingoDFGScheduleMixin:
    """bingo_schedule_programs: the compiler's order of every device core's program (see the module)."""

    def bingo_schedule_programs(self):
        from bingo_platform import core_roles
        roles = getattr(self, "core_roles", None) or core_roles()
        dev = {roles[r] for r in ("gemm", "simd", "xdma", "dm")}
        splits = {}
        if getattr(self, "async_exec", False):
            for n in self.node_list:
                a = n.kernel_args
                sp = a.split() if a is not None and n.assigned_core_id in dev else None
                if sp is not None and n.assigned_core_id == roles[sp.accel]:
                    splits[n] = sp
        static = getattr(self, "static_l1_opts", None)
        cap = getattr(self, "l1_capacity_bytes", None)
        self.l1_budget = None

        def fitted(c):
            """(candidate, None) when its buffers fit L1; else the same model scheduled again, its tasks that open
            buffers held back (schedule's l1), the budget lowered by what the plan still overflows: (that, budget);
            (None, None) if none fits."""
            m, sq, t = c
            cap = _l1_cap(self, m)
            live = _l1_live_peak(self, m, t, static) if static and cap else 0
            if cap and live - cap > cap:                # beyond what holding back can recover (below): no packing
                l1_log.append([live - cap])
                return None, None
            over = _l1_over(self, m, sq, static)
            tried = [over]
            l1_log.append(tried)
            if over <= 0:
                return c, None
            if not static or not cap or over > cap:
                return None, None
            bufs, budget = _l1_buffers(self, m, static), cap
            for _ in range(L1_TRIES):
                sq, t, mg = schedule_best(m, (budget, bufs))
                was, over = over, _l1_over(self, mg, sq, static)
                tried.append(over)
                if over <= 0:
                    return (mg, sq, t), budget
                if over > was // 2:
                    break                           # the budget no longer gains enough a try
                budget -= max(over + L1_SLACK, cap // L1_STEPS)
            return None, None

        l1_log = []                         # each candidate fitted: its overflow, then each try's (fitted)

        def best(keep_order):
            """The plain programs, and the asynchronous ones: each accelerator's splits added where the model says the
            program finishes sooner (async_exec "all": every split); the shortest that the static L1 plan fits (with
            static_l1). Returns (model, seqs, timeline, no_gain) or None."""
            plain = _Model(self, roles, {}, keep_order)
            cands = [(plain, *schedule_best(plain)[:2])]
            tried = None
            if splits and self.async_exec == "all":
                m = _Model(self, roles, splits, keep_order)
                cands = [(m, *schedule_best(m)[:2])] + cands
            elif splits:
                kept, cur = set(), cands[0]
                for accel in ("gemm", "xdma", "simd"):
                    sub = {n: sp for n, sp in splits.items() if sp.accel in kept | {accel}}
                    if len(sub) == sum(sp.accel in kept for sp in splits.values()):
                        continue                    # no task of this accelerator splits
                    m = _Model(self, roles, sub, keep_order)
                    c = (m, *schedule_best(m)[:2])
                    tried = c[2].makespan()
                    if c[2].makespan() < cur[2].makespan():
                        kept.add(accel)
                        cur = c
                        cands.append(c)
                cands.sort(key=lambda c: c[2].makespan())
            win = win_budget = None
            for c in cands:
                if win is not None and (self.async_exec == "all" or c[2].makespan() >= win[2].makespan()):
                    continue                    # held back, a schedule only gets longer
                f, budget = fitted(c)
                if f is not None and (win is None or f[2].makespan() < win[2].makespan()):
                    win, win_budget = f, budget
            if win is None:
                return None
            m, sq, t = win
            self.l1_budget = win_budget
            no_gain = (tried, len(splits)) if splits and not m.parts else None
            return m, sq, t, no_gain

        # the compiler's own order (where its buffers do not fit L1, its tasks that open buffers held back); where even
        # that does not fit, each core's tasks in the order they were made (held back the same way)
        picked = best(False)
        self.kept_order = picked is None
        if picked is None:
            picked = best(True)
            if picked is None:
                plain = _Model(self, roles, {}, True)
                picked = (plain, *schedule_best(plain)[:2], None)
        model, seqs, tl, no_gain = picked
        # the placement the schedule was fitted with: the static L1 plan keeps it where the final graph still allows it
        plan = getattr(model, "l1_plan", None)
        if getattr(self, "l1_capacity_auto", False) and _l1_cap(self, model):
            cap = self.l1_capacity_bytes = _l1_cap(self, model)
            print(f"[schedule] static L1 capacity {cap:,} B a cluster: what each cluster's L1 heap leaves beside its "
                  f"lists, scratch pools, scratchpads and args (the least of them)")
        cap_ok = plan is not None and cap and all(st["peak"] <= cap for st in plan[1].values())
        self.l1_plan_hint = plan if cap_ok else None
        plain = model if not model.parts and not model.l1_edges else _Model(self, roles, {})
        # the order the generator made its tasks in, every task whole, under the same model
        written = {}
        for n in sorted(self.node_list, key=lambda t: t.node_id):
            if n.assigned_core_id in dev:
                written.setdefault((n.assigned_chiplet_id, n.assigned_cluster_id, n.assigned_core_id), []).append(n)
        ref = evaluate(plain, written)
        programs, stats, made = self._materialize(model, seqs, tl)
        for u, v in model.l1_edges:
            a, b = made.get(u, u), made.get(v, v)
            if a is not b and not self.has_edge(a, b):
                self.bingo_add_edge(a, b)
        numbered = self._number_works(programs, roles) if model.parts else 0
        shared = self._prologues(programs) if model.parts else 0
        reused, folded = self._csr_reuse(programs, roles)
        if folded:
            self._renumber_dense()
        added = 0
        for seq in programs.values():
            for a, b in zip(seq, seq[1:]):
                if not self.has_edge(a, b):
                    self.bingo_add_edge(a, b)
                    added += 1
        line = (f"[schedule] {len(model.vertices)} tasks on {len(seqs)} device cores"
                + (", each core's tasks in the order made (the compiler's order overflows L1)" if self.kept_order else "")
                + (f", L1 fitted (budget {self.l1_budget:,} B a cluster: {len(model.l1_edges)} edges from the last "
                   "users of a space to its next writer, tasks that open buffers held back)"
                   if self.l1_budget is not None else "")
                + ", model makespan "
                f"{tl.makespan():,} cycles" + (f" (the order they were made in: {ref.makespan():,})" if ref else "")
                + f"; {added} program edges; CSR reuse: {reused} tasks")
        if model.parts:
            by = {}
            for sp in model.splits.values():
                by[sp.accel] = by.get(sp.accel, 0) + 1
            line += (f"; async: {len(model.parts)} tasks split ({', '.join(f'{a} {n}' for a, n in sorted(by.items()))})"
                     f", {stats['whole']} kept whole, {stats['ahead'] - folded} configured ahead ({folded} more only wrote "
                     f"addresses: in their start), {stats['trailing']} waits "
                     f"behind a later task, {numbered} numbered, {shared} shared setups")
        elif no_gain is not None:
            line += f"; async: no gain ({no_gain[0]:,} cycles split), the {no_gain[1]} tasks stay whole"
        print(line)
        if l1_log and any(t[0] > 0 for t in l1_log):
            print("[schedule] L1 overflow of each candidate tried (at least, where it was not packed), then of each "
                  "held-back try: "
                  + "; ".join(" -> ".join(f"{o:,}" for o in t) for t in l1_log))

    def _materialize(self, model, seqs, tl):
        """The programs as tasks: each split task's parts fused where the schedule left them side by side and the
        fused task runs no later (a configure that ran before the start's operands were there stays its own task: a
        start that configures waits for them first), the rest made real tasks. Returns ({core: [task, ...]}, counts)."""
        pos = {v: i for seq in seqs.values() for i, v in enumerate(seq)}
        made = {}                           # part -> the task it became
        stats = {"whole": 0, "ahead": 0, "trailing": 0}
        out_of = {}                         # start task -> its wait task, the source of its ordering-only edges
        self._split_parts = {}              # start task -> (configure task or None, wait task or None, Split)
        for x, (c, s, w) in model.parts.items():
            sp = model.splits[x]
            inputs = max((tl.fin[u] + lat for u, lat in model.preds[s] if u is not c), default=0)
            fused_cfg = c is None or (pos[c] + 1 == pos[s] and tl.start[c] >= inputs
                                      and sp.start(False) is not None)
            next_w = w is not None and pos[s] + 1 == pos[w]
            if fused_cfg and (w is None or next_w):
                for p in (c, s, w):
                    if p is not None:
                        made[p] = x
                stats["whole"] += 1
                continue
            drains = next_w and sp.drain is not None    # the start waits for its own work: no wait task
            base, orig = x.node_name, x.kernel_name    # a part's kernel None: the task's own
            cf = wt = None
            if w is not None and not drains:
                kname, args = sp.wait
                kname = kname or orig
                wt = BingoNode(x.assigned_chiplet_id, x.assigned_cluster_id, x.assigned_core_id,
                               node_name=base + sp.names[2], kernel_name=kname, kernel_args=args)
                if sp.names[2] == "":
                    x.node_name = base + sp.names[1]     # the wait takes the task's name
                self.bingo_add_node(wt)
                for _, succ, data in list(self.out_edges(x, data=True)):
                    self.remove_edge(x, succ)
                    self.add_edge(wt, succ, **data)
                out_of[x] = wt                             # its ordering-only edges leave from the wait (below)
                self.bingo_add_edge(x, wt)
                x.operands_held_by = wt
                stats["trailing"] += pos[w] > pos[s] + 1
            if not fused_cfg:
                kname, args = sp.configure
                kname = kname or orig
                cf = BingoNode(x.assigned_chiplet_id, x.assigned_cluster_id, x.assigned_core_id,
                               node_name=base + sp.names[0], kernel_name=kname, kernel_args=args)
                cf.config_only = True
                self.bingo_add_node(cf)
                self.bingo_add_edge(cf, x)
                ld = model.phase_load.get((x.assigned_chiplet_id, x.assigned_cluster_id, getattr(x, "args_phase", 0)))
                if ld is not None:
                    self.bingo_add_edge(ld, cf)     # its args' phase load
                stats["ahead"] += 1
            kname, args = (sp.drain if drains else sp.start)(not fused_cfg)
            x.kernel_name, x.kernel_args = kname or orig, args
            if sp.names[1] and sp.names[2] != "":
                x.node_name = base + sp.names[1]
            made[s] = x
            if c is not None:
                made[c] = cf if cf is not None else x
            if w is not None:
                made[w] = wt if wt is not None else x
            for part in (cf, wt):
                if part is not None and hasattr(x, "args_phase"):
                    part.args_phase = x.args_phase          # its args sit in the task's phase (an application's phases)
            self._split_parts[x] = (cf, wt, sp)
        if out_of:
            self.order_edges = [(out_of.get(a, a), b) for a, b in self.order_edges]
        programs = {}
        for core, seq in seqs.items():
            out = []
            for v in seq:
                t = made.get(v, v)
                if not out or out[-1] is not t:
                    out.append(t)
            programs[core] = out
        return programs, stats, made

    def _number_works(self, programs, roles):
        """Each counter-tracked accelerator's works 1, 2, ... in its core's program, every work its counter counts
        taking a number, a whole one too; a split work's parts carry its number. Returns the number of starts
        numbered."""
        accel_core = {roles[a]: a for a in ACCELS}
        counted = {sp.accel for _c, _w, sp in self._split_parts.values() if sp.counter}
        cfg_of = {cf: x for x, (cf, _w, _sp) in self._split_parts.items() if cf is not None}
        n_starts = 0
        for (chip, cluster, core), seq in programs.items():
            accel = accel_core.get(core)
            if accel not in counted:
                continue
            n = 0
            for t in seq:
                if t in cfg_of:
                    continue
                parts = self._split_parts.get(t)
                if parts is not None and parts[2].counter:
                    n += 1
                    for p in (t, parts[0], parts[1]):
                        if p is not None and hasattr(p.kernel_args, "seq"):
                            p.kernel_args.seq = n
                    n_starts += 1
                elif t.kernel_args is not None and t.kernel_args.dispatch_accel() == accel:
                    n += 1
            waits = {w for _c, w, _sp in self._split_parts.values() if w is not None}
            for t in seq:
                if t.kernel_name == GEMM_WAIT and t not in waits:
                    raise ValueError(f"chip {chip:#04x} cluster {cluster}: a gemm_wait the compiler did not make "
                                     f"({t.node_name}); write the dispatch as one task")
        return n_starts

    def _prologues(self, programs):
        """The per-cluster setup the configuring parts of a split share (Split.prologue): one task a core and key,
        first in the core's program, its L1 buffer named in every such part. Returns the number of setup tasks."""
        made = 0
        for core, seq in list(programs.items()):
            users = {}
            for t in seq:
                parts = self._split_parts.get(t)
                if parts is None or parts[2].prologue is None:
                    continue
                key, nbytes, init, field = parts[2].prologue
                # every part that has the field names the setup's buffer: the configure task reads it, and its start
                # names it too, since liveness counts no configure task as a user (config_only)
                for q in (parts[0], t):
                    if q is not None and hasattr(q.kernel_args, field):
                        users.setdefault(key, (nbytes, init, field, []))[3].append(q)
            first = []
            for key, (nbytes, init, field, ps) in users.items():
                chip, cluster, cid = core
                name = "_".join(str(k) for k in key)
                buf = BingoMemAlloc(f"{name}_k{chip:02x}_cl{cluster}", size=nbytes, mem_level="L1", chip_id=chip,
                                    cluster_id=cluster)
                buf.compiler_made = True       # placed after the application's buffers (bingo_l1_packer.py)
                kname, args = init(buf)
                t = BingoNode(chip, cluster, cid, node_name=f"{name}_init_k{chip:02x}_cl{cluster}",
                              kernel_name=kname, kernel_args=args)
                self.bingo_add_node(t)
                for p in ps:
                    setattr(p.kernel_args, field, buf)
                first.append(t)
                made += 1
            if first:
                programs[core] = first + seq
        return made

    def _renumber_dense(self):
        """Every task's id its position, as the emitter takes them (a global task id is a node's place in the list): the
        configure parts _fold_reuse dropped leave gaps otherwise. The order, and so every sort by id, is kept."""
        for i, n in enumerate(self.node_list):
            n.node_id = i
        self.id = len(self.node_list) - 1

    def _fold_reuse(self, cf, x):
        """A configure part whose configuration the CSRs hold already writes only addresses: it is dropped and its start
        `x` becomes the dispatch that writes them and starts (the split's unconfigured start, in its reuse form). Returns
        whether it was folded."""
        cf_, wt, sp = self._split_parts[x]
        form = sp.start(False)
        if form is None:
            return False
        kname, args = form
        _fold_reuse_fields(args, x.kernel_args)
        r = args.config_reuse()
        if r is None:
            return False
        x.kernel_name, x.kernel_args = (r[0] or kname or x.kernel_name), r[1]
        for pred, _, data in list(self.in_edges(cf, data=True)):
            if pred is not x and not self.has_edge(pred, x):
                self.add_edge(pred, x, **data)      # its phase load, its prologue: the start's own now
        self._folded[cf] = x                        # its ordering-only edges: the start's (_csr_reuse, once)
        self.remove_node(cf)
        self._split_parts[x] = (None, wt, sp)
        return True

    def _csr_reuse(self, programs, roles):
        """Along each accelerator core's program: a task whose configuration the CSRs hold already runs as its cheaper
        form, a split's configure part that would only write addresses folding into its start; then the softmax
        geometry and the ChainGather clean-up rules. Returns the number of tasks changed and of configure parts
        folded."""
        accel_cores = {roles[a] for a in ACCELS}
        cfg_of = {cf: x for x, (cf, _w, _sp) in getattr(self, "_split_parts", {}).items() if cf is not None}
        self._folded = {}
        changed = folded = 0
        for key, seq in programs.items():
            core = key[2]
            if core not in accel_cores:
                continue
            held, dropped = None, set()      # dropped: configure parts folded into their start
            for t in seq:
                a = t.kernel_args
                if a is None:
                    continue
                k = a.config_key()
                if k is KEEPS or k == KEEPS:
                    continue
                if k is None:
                    held = None
                    continue
                if k == held:
                    if t in cfg_of and self._fold_reuse(t, cfg_of[t]):
                        dropped.add(t)              # nothing left to configure ahead: its start reuses and starts
                    else:
                        r = a.config_reuse()
                        if r is not None:
                            t.kernel_name, t.kernel_args = r
                    changed += 1
                held = k
            if dropped:
                programs[key] = seq = [t for t in seq if t not in dropped]
                folded += len(dropped)
            if core == roles["simd"]:
                changed += _prime_softmax(seq)
            if core == roles["xdma"]:
                changed += _chain_teardown(seq)
        if self._folded:
            f = self._folded
            self.order_edges = [(f.get(a, a), f.get(b, b)) for a, b in self.order_edges if f.get(a, a) is not f.get(b, b)]
        return changed, folded


def _fold_reuse_fields(dst, src):
    """The fields the compiler set on a split part (its work's number, the shared setup's address) onto another
    form of the same dispatch."""
    for f in ("seq", "shape_addr"):
        if hasattr(src, f) and hasattr(dst, f):
            setattr(dst, f, getattr(src, f))


def _buffer(h):
    """A key for the L1 buffer a handle names, whatever the view."""
    if isinstance(h, BingoMemAllocView):
        return (id(h.base), h.offset)
    return h if isinstance(h, int) else (id(h), 0)


def _prime_softmax(seq):
    """A softmax whose arena an earlier softmax on this SIMD core used skips the geometry (GEOM_PRIMED), and its shared
    CSRs too right after a softmax of the same geometry (GEOM_CSR_PRIMED). Returns the number changed."""
    SM = SnaxBingoKernelSimdFaSoftmaxArgs
    used, prev, n = set(), None, 0
    for t in seq:
        if t.kernel_name == SOFTMAX:
            a = t.kernel_args
            key = _buffer(a.arena)
            if (key in used and a.geom_mode == SM.GEOM_SELF) or a.geom_mode == SM.GEOM_PRIMED:
                p = prev.kernel_args if prev is not None and prev.kernel_name == SOFTMAX else None
                if p is not None and (p.bc, p.dhead, p.p_mode, p.p8_pitch, p.score_scale) == \
                        (a.bc, a.dhead, a.p_mode, a.p8_pitch, a.score_scale):
                    a.geom_mode = SM.GEOM_CSR_PRIMED
                else:
                    a.geom_mode = SM.GEOM_PRIMED
                n += 1
            used.add(key)
        prev = t
    return n


def _chain_teardown(seq):
    """A ChainGather wait whose core's next task is a ChainGather configure skips its CSR clean-up: the configure
    rewrites every destination slot and clears the junctions. Returns the number changed."""
    n = 0
    for a, b in zip(seq, seq[1:]):
        if isinstance(a.kernel_args, SnaxBingoKernelXdmaChainWaitArgs) and b.kernel_name == CHAIN_CFG:
            a.kernel_args.teardown = False
            n += 1
    return n
