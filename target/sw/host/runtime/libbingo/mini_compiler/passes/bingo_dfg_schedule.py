# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""Each device core's program, chosen by the compiler, and the asynchronous VersaCore dispatch.

WHY THE COMPILER ORDERS THE CORES. BINGO dispatches a core's tasks into an in-order queue, so the
order of each core's program is fixed when the graph is compiled, and two tasks with no path between
them still run one after the other. Which goes first is a question of time -- whose inputs land
sooner, whose result the rest of the graph waits for -- so it is answered with the cost model
(bingo_cost_model.py), not left to the order a generator happened to make its tasks in.

THE SCHEDULE: list scheduling over the cost model. A core's next task is the one of its ready tasks
that can start earliest; among those that can start at once, the one with the longest path to the
end of the graph (its rank). Every core runs in order and every task starts after its dependencies,
its ordering-only edges (bingo_add_order_edge) and the edge latency: the one timeline all cores
share is a topological order of the graph, so the programs it yields close no cycle. A weight-ring
load also waits for its chunk, which its chip's prefetcher pushes from the time it starts.

ASYNCHRONOUS DISPATCH (dfg.async_exec). A VersaCore dispatch is then three parts on its cluster's
GEMM core: a CONFIGURE (every CSR written, nothing started), a START (the start write, once the
operands are there), and a WAIT (returns once the array has retired the dispatch). The schedule
places each part like any other task, under the array's rules:

  - a configure follows the previous dispatch's start: the managers' register files hold one
    pending configuration;
  - at most QUEUE_DEPTH dispatches are started and not yet waited for: the array runs one and
    queues the next;
  - a wait is ready when the array is done with its dispatch.

So a wait trails the next start exactly when that start can go before the array finishes, and a
configure runs ahead of the waits before its start when the operands are later than the core is
free. Parts the schedule left side by side gain nothing and are fused again: a configure right
before its start is one start that configures; a start right before its wait (and configured
in it) is the blocking gemm_full. The dispatches are then numbered on each cluster in program order
(the wait compares the array's retired counter with the number), every one that runs on the array
counting, a blocking one too.

CSR REUSE. Along each GEMM core's program, a dispatch whose configuration the array's CSRs already
hold writes only its operand addresses: gemm_full becomes gemm_minimal, a gemm_start takes `reuse`.

What a generator writes is one task per operation and its dependencies; this pass is where the
queue, the numbers and the configure / start / wait split come from."""

import heapq
import inspect

import networkx as nx

import bingo_cost_model as cm
from bingo_kernel_args import (SnaxBingoKernelGemmFullArgs, SnaxBingoKernelGemmMinimalArgs,
                               SnaxBingoKernelGemmStartArgs, SnaxBingoKernelGemmWaitArgs)
from bingo_node import BingoNode

QUEUE_DEPTH = 2
GEMM_START = SnaxBingoKernelGemmStartArgs.KERNEL_NAME
GEMM_WAIT = SnaxBingoKernelGemmWaitArgs.KERNEL_NAME
SHAPE_INIT = "__snax_bingo_kernel_gemm_shape_init"
CONFIGURE_AND_START = SnaxBingoKernelGemmStartArgs.CONFIGURE_AND_START
CONFIGURE_ONLY = SnaxBingoKernelGemmStartArgs.CONFIGURE_ONLY
START_ONLY = SnaxBingoKernelGemmStartArgs.START_ONLY
# the dispatches the array's retired counter counts, besides gemm_start and the typed blocking kernels
ARRAY_DISPATCHES = {cm.GEMM_FULL, cm.GEMM_MINIMAL, "__snax_bingo_kernel_gemv",
                    "__snax_bingo_kernel_gemm_fa_qk", "__snax_bingo_kernel_gemm_fa_pv",
                    "__snax_bingo_kernel_gemm_fa_qk_start", "__snax_bingo_kernel_gemm_fa_pv_start"}
_FULL_FIELDS = tuple(n for n in inspect.signature(SnaxBingoKernelGemmFullArgs.__init__).parameters if n != "self")
_ADDRS = ("input_A_addr", "input_B_addr", "input_C_addr", "output_D_addr")
_START_NOT_CONFIG = _ADDRS + ("seq", "shape_addr", "reuse", "stage")


class _Part:
    """A part of a split dispatch inside the scheduler: "cfg", "start" or "wait" of the task `node`."""
    __slots__ = ("node", "kind")

    def __init__(self, node, kind):
        self.node, self.kind = node, kind

    def __repr__(self):
        return f"{self.node.node_name}:{self.kind}"


class _Model:
    """The graph the scheduler works on: every task a vertex, a dispatch to split three, with each
    vertex's place, time and role, and the edges with their latencies."""

    def __init__(self, dfg, roles, split):
        self.roles = roles
        self.host = roles["host"]
        self.dev = {roles[r] for r in ("gemm", "simd", "xdma", "dm")}
        self.parts = {}                     # task -> (cfg, start, wait) parts, for a split dispatch
        self.vertices, self.idx, self.place, self.dur, self.role, self.mkn = [], {}, {}, {}, {}, {}
        for n in sorted(dfg.node_list, key=lambda t: t.node_id):
            if n in split:
                ps = tuple(_Part(n, k) for k in ("cfg", "start", "wait"))
                self.parts[n] = ps
                for p in ps:
                    self._add(p, n)
            else:
                self._add(n, n)
        self.preds = {v: [] for v in self.vertices}
        self.succs = {v: [] for v in self.vertices}
        edges = list(dfg.edges()) + list(dfg.order_edges)
        for u, v in edges:
            self._edge(self.parts[u][2] if u in self.parts else u, self.parts[v][1] if v in self.parts else v)
        for c, s, w in self.parts.values():
            self._edge(c, s, 0)
            self._edge(s, w, 0)
        # a ring load waits for its chunk, which the chip's prefetcher pushes once it runs
        self.prefetcher, self.arrival = {}, {}
        for n in dfg.node_list:
            if n.kernel_name == cm.WEIGHT_PREFETCH:
                k = n.assigned_chiplet_id
                if k not in self.prefetcher or n.node_id < self.prefetcher[k].node_id:
                    self.prefetcher[k] = n
        loads = {}
        for v in self.vertices:
            if isinstance(v, BingoNode) and v.kernel_name == cm.RING_LOAD and v.assigned_chiplet_id in self.prefetcher:
                loads.setdefault((v.assigned_chiplet_id, v.assigned_cluster_id), []).append(v)
        rings = {}
        for (k, _c) in loads:
            rings[k] = rings.get(k, 0) + 1
        for (k, c), ls in loads.items():
            sent = 0
            for v in ls:
                sent += int(getattr(v.kernel_args, "size", 0))
                self.arrival[v] = cm.PUSH_LATENCY + sent * rings[k] // cm.PUSH_BYTES_PER_CYCLE
                self._edge(self.prefetcher[k], v, 0)

    def _add(self, v, n):
        self.idx[v] = len(self.vertices)
        self.vertices.append(v)
        self.place[v] = (n.assigned_chiplet_id, n.assigned_cluster_id, n.assigned_core_id)
        if isinstance(v, _Part):
            self.mkn[v] = cm.array_cycles(n.kernel_args)
            self.role[v] = v.kind
            self.dur[v] = {"cfg": cm.GEMM_CONFIGURE, "start": cm.GEMM_START, "wait": cm.GEMM_WAIT}[v.kind]
        elif cm.is_blocking_gemm(n) and n.assigned_core_id in self.dev:
            self.mkn[v] = cm.array_cycles(n.kernel_args)
            self.role[v] = "block"
            self.dur[v] = cm.GEMM_CONFIGURE
        else:
            self.role[v] = None
            self.dur[v] = cm.task_cycles(n)

    def _edge(self, u, v, lat=None):
        if lat is None:
            lat = cm.edge_latency(self.place[u], self.place[v], self.host)
        self.preds[v].append((u, lat))
        self.succs[u].append((v, lat))

    def on_core(self, v):
        """Whether `v` holds an in-order device core (a host task runs beside the rest of its chip's)."""
        return self.place[v][2] in self.dev

    def array(self, v):
        return self.place[v][:2]

    def ranks(self):
        """Each vertex's longest path to the end of the graph, its own time and every edge's latency counted
        (a start's path to its wait: the array's time)."""
        g = nx.DiGraph()
        g.add_nodes_from(self.vertices)
        g.add_edges_from((u, v) for u in self.vertices for v, _ in self.succs[u])
        rank = {}
        for v in reversed(list(nx.topological_sort(g))):
            own = self.dur[v] + (self.mkn[v] if self.role[v] == "block" else 0)
            rank[v] = own + max((lat + (self.mkn[v] if self.role[v] == "start" else 0) + rank[s]
                                 for s, lat in self.succs[v]), default=0)
        return rank


class _Timeline:
    """The machine's state while a schedule is laid out: when each core and each array is free, the
    dispatches started and not yet waited for, the configuration waiting for its start."""

    def __init__(self, model):
        self.m = model
        self.core_free, self.array_free = {}, {}
        self.outstanding, self.pending = {}, {}
        self.start, self.fin, self.done = {}, {}, {}
        self.dr = {v: 0 for v in model.vertices}

    def lay(self, v, est):
        """Run `v` from `est` (no earlier than its core is free and its inputs are there). Returns its end."""
        m, role = self.m, self.m.role[v]
        if role in ("block", "start"):
            a = m.array(v)
            setup = est + m.dur[v]
            arr = max(setup, self.array_free.get(a, 0))
            self.array_free[a] = arr + m.mkn[v]
            if role == "block":
                end = arr + m.mkn[v]
            else:
                end = setup
                self.done[v] = arr + m.mkn[v]
                self.outstanding[a] = self.outstanding.get(a, 0) + 1
                self.pending[a] = None
        else:
            end = est + m.dur[v]
            if role == "cfg":
                self.pending[m.array(v)] = m.parts[v.node][1]
            elif role == "wait":
                a = m.array(v)
                self.outstanding[a] -= 1
        self.start[v], self.fin[v] = est, end
        if m.on_core(v):
            self.core_free[m.place[v]] = end
        return end

    def release(self, v, end):
        """What `v` ending at `end` tells its successors: a successor's inputs are there no earlier."""
        m = self.m
        for s, lat in m.succs[v]:
            t = end + lat
            if m.role[s] == "wait" and m.role[v] == "start":
                t = max(t, self.done[v])
            if s in m.arrival and m.prefetcher.get(m.place[s][0]) is v:
                t = self.start[v] + m.arrival[s]
            if t > self.dr[s]:
                self.dr[s] = t

    def makespan(self):
        return max(self.fin.values(), default=0)


def schedule(model):
    """List-schedule `model`: {device core: [vertex, ...]} in program order, and the timeline."""
    m = model
    tl = _Timeline(m)
    rank = m.ranks()
    npred = {v: len(m.preds[v]) for v in m.vertices}
    future, avail, version = {}, {}, {}
    gated = {}                              # array -> configure or blocking vertices that wait for it
    heap, seqs = [], {}

    def key(v):
        return (-rank[v], m.idx[v])

    def peek(c):
        f, a, free = future.get(c, []), avail.setdefault(c, []), tl.core_free.get(c, 0)
        while f and f[0][0] <= free:
            dr, _r, _i, v = heapq.heappop(f)
            heapq.heappush(a, (*key(v), dr, v))
        if a:
            return free, a[0][-1]
        if f:
            return f[0][0], f[0][-1]
        return None

    def refresh(c):
        version[c] = version.get(c, 0) + 1
        p = peek(c)
        if p is not None:
            heapq.heappush(heap, (p[0], *key(p[1]), version[c], c))

    def array_ok(v):
        a = m.array(v)
        if tl.pending.get(a) is not None:
            return False
        return m.role[v] != "cfg" or tl.outstanding.get(a, 0) < QUEUE_DEPTH

    touched = set()
    work = []                               # vertices whose inputs are all decided, to offer

    def ready(v):
        """Every input of `v` decided: a configure, once its start waits for it alone."""
        if m.role[v] == "cfg":
            return npred[m.parts[v.node][1]] == 1 and v not in tl.fin
        return npred[v] == 0

    state = {}                              # vertex -> "queued", "gated" or "done"

    def hold(v):
        if state.get(v) != "gated":
            state[v] = "gated"
            gated.setdefault(m.array(v), []).append(v)

    def offer(v):
        """Queue `v` on its core, hold it for its array, or run a host task now (its successors then follow)."""
        if state.get(v) in ("queued", "done"):
            return
        role = m.role[v]
        if role in ("cfg", "block") and not array_ok(v):
            hold(v)
            return
        state[v] = "queued"
        if not m.on_core(v):
            state[v] = "done"
            end = tl.lay(v, tl.dr[v])
            tl.release(v, end)
            decided(v)
            return
        dr = tl.dr[v]
        if role == "cfg":
            dr = max(0, tl.dr[m.parts[v.node][1]] - m.dur[v])
        c = m.place[v]
        heapq.heappush(future.setdefault(c, []), (dr, *key(v), v))
        touched.add(c)

    def decided(v):
        for s, _ in m.succs[v]:
            npred[s] -= 1
            if m.role[s] == "start" and npred[s] == 1:
                cfg = m.parts[s.node][0]
                if cfg not in tl.fin:
                    work.append(cfg)
            elif npred[s] == 0 and m.role[s] != "cfg":
                work.append(s)

    def drain():
        while work:
            v = work.pop()
            if ready(v):
                offer(v)

    def ungate(a):
        for v in gated.pop(a, []):
            state.pop(v, None)
            work.append(v)

    work.extend(v for v in m.vertices if ready(v))
    drain()
    for c in list(touched):
        refresh(c)
    touched.clear()
    while heap:
        est, _r, _i, ver, c = heapq.heappop(heap)
        if ver != version.get(c):
            continue
        p = peek(c)
        if p is None:
            continue
        est, v = p
        if avail[c] and avail[c][0][-1] is v:
            heapq.heappop(avail[c])
        else:
            heapq.heappop(future[c])
        if m.role[v] in ("cfg", "block") and not array_ok(v):
            state.pop(v, None)
            hold(v)                         # the array's state changed since it was queued
            refresh(c)
            continue
        state[v] = "done"
        end = tl.lay(v, max(est, tl.dr[v]))
        seqs.setdefault(c, []).append(v)
        tl.release(v, end)
        decided(v)
        if m.role[v] in ("start", "wait"):
            ungate(m.array(v))
        drain()
        touched.add(c)
        for t in list(touched):
            refresh(t)
        touched.clear()
    left = [v for v in m.vertices if v not in tl.fin]
    if left:
        raise ValueError(f"schedule: {len(left)} tasks never became ready, e.g. {left[:4]}")
    return seqs, tl


def evaluate(model, seqs):
    """The timeline of fixed programs {device core: [vertex, ...]} under the same model, or None when the
    programs and the graph's edges close a cycle."""
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


def _start_args(full, stage):
    return SnaxBingoKernelGemmStartArgs(**{f: getattr(full, f) for f in _FULL_FIELDS}, seq=1, stage=stage)


class BingoDFGScheduleMixin:
    """bingo_schedule_programs: the compiler's order of every device core's program (see the module)."""

    def bingo_schedule_programs(self):
        from bingo_platform import core_roles
        roles = getattr(self, "core_roles", None) or core_roles()
        dev = {roles[r] for r in ("gemm", "simd", "xdma", "dm")}
        plain = _Model(self, roles, set())
        model, (seqs, tl) = plain, schedule(plain)
        no_gain = None
        if getattr(self, "async_exec", False):
            split = {n for n in self.node_list if n.kernel_name == cm.GEMM_FULL and n.assigned_core_id == roles["gemm"]}
            if split:
                amodel = _Model(self, roles, split)
                aseqs, atl = schedule(amodel)
                # the asynchronous programs only where the model says they finish sooner than the blocking ones
                if atl.makespan() < tl.makespan():
                    model, seqs, tl = amodel, aseqs, atl
                else:
                    no_gain = (atl.makespan(), len(split))
        # the order the generator made its tasks in, every dispatch blocking, under the same model
        written = {}
        for n in sorted(self.node_list, key=lambda t: t.node_id):
            if n.assigned_core_id in dev:
                written.setdefault((n.assigned_chiplet_id, n.assigned_cluster_id, n.assigned_core_id), []).append(n)
        ref = evaluate(plain, written)
        programs, stats = self._materialize(model, seqs, tl)
        numbered = self._number_dispatches(programs, roles["gemm"]) if model.parts else 0
        reused = self._csr_reuse(programs, roles["gemm"])
        added = 0
        for seq in programs.values():
            for a, b in zip(seq, seq[1:]):
                if not self.has_edge(a, b):
                    self.bingo_add_edge(a, b)
                    added += 1
        self.program_ordered = True
        line = (f"[schedule] {len(model.vertices)} tasks on {len(seqs)} device cores, model makespan "
                f"{tl.makespan():,} cycles" + (f" (the order they were made in: {ref.makespan():,})" if ref else "")
                + f"; {added} program edges; CSR reuse: {reused} dispatches")
        if model.parts:
            line += (f"; async: {len(model.parts)} dispatches, {stats['blocking']} kept blocking, "
                     f"{stats['ahead']} configured ahead, {stats['trailing']} waits behind a later task, "
                     f"{numbered} numbered")
        elif no_gain is not None:
            line += (f"; async: no gain ({no_gain[0]:,} cycles split), the {no_gain[1]} dispatches stay blocking")
        print(line)

    def _materialize(self, model, seqs, tl):
        """The programs as tasks: each split dispatch's parts fused where the schedule left them side by side and
        the fused task runs no later (a configure that ran before the start's operands were there stays its own
        task: a start that configures waits for them first), the rest made real tasks. Returns ({core: [task, ...]},
        counts)."""
        pos = {v: i for seq in seqs.values() for i, v in enumerate(seq)}
        made = {}                           # part -> the task it became
        stats = {"blocking": 0, "ahead": 0, "trailing": 0}
        self._async_pairs = {}              # start task -> (configure task or None, wait task or None)
        for x, (c, s, w) in model.parts.items():
            inputs = max((tl.fin[u] + lat for u, lat in model.preds[s] if u is not c), default=0)
            with_cfg = pos[c] + 1 == pos[s] and tl.start[c] >= inputs
            with_wait = with_cfg and pos[s] + 1 == pos[w]
            full = x.kernel_args
            if with_wait:
                made[c] = made[s] = made[w] = x
                stats["blocking"] += 1
                continue
            wt = BingoNode(x.assigned_chiplet_id, x.assigned_cluster_id, x.assigned_core_id,
                           node_name=f"{x.node_name}_w", kernel_name=GEMM_WAIT,
                           kernel_args=SnaxBingoKernelGemmWaitArgs(1))
            self.bingo_add_node(wt)
            for _, succ, data in list(self.out_edges(x, data=True)):
                self.remove_edge(x, succ)
                self.add_edge(wt, succ, **data)
            self.order_edges = [(wt if a is x else a, b) for a, b in self.order_edges]
            self.bingo_add_edge(x, wt)
            cf = None
            if with_cfg:
                x.kernel_name, x.kernel_args = GEMM_START, _start_args(full, CONFIGURE_AND_START)
            else:
                cf = BingoNode(x.assigned_chiplet_id, x.assigned_cluster_id, x.assigned_core_id,
                               node_name=f"{x.node_name}_cfg", kernel_name=GEMM_START,
                               kernel_args=_start_args(full, CONFIGURE_ONLY))
                cf.config_only = True
                self.bingo_add_node(cf)
                self.bingo_add_edge(cf, x)
                x.kernel_name, x.kernel_args = GEMM_START, _start_args(full, START_ONLY)
                stats["ahead"] += 1
            x.operands_held_by = wt
            stats["trailing"] += pos[w] > pos[s] + 1
            made[s], made[w] = x, wt
            made[c] = cf if cf is not None else x
            self._async_pairs[x] = (cf, wt)
        programs = {}
        for core, seq in seqs.items():
            out = []
            for v in seq:
                t = made.get(v, v)
                if not out or out[-1] is not t:
                    out.append(t)
            programs[core] = out
        return programs, stats

    def _number_dispatches(self, programs, gemm_core):
        """Each cluster's VersaCore dispatches 1, 2, ... in its GEMM core's program, a start's configure and wait
        tasks on the start's number. Returns the number of starts numbered."""
        cfg_of = {cf: x for x, (cf, _w) in self._async_pairs.items() if cf is not None}
        n_starts = 0
        for (chip, cluster, core), seq in programs.items():
            if core != gemm_core:
                continue
            n = 0
            for t in seq:
                k = t.kernel_name or ""
                if t in cfg_of:
                    continue
                if k == GEMM_START or k in ARRAY_DISPATCHES or cm.is_blocking_gemm(t):
                    n += 1
                    if k == GEMM_START:
                        t.kernel_args.seq = n
                        cf, wt = self._async_pairs.get(t, (None, None))
                        for p in (cf, wt):
                            if p is not None:
                                p.kernel_args.seq = n
                        n_starts += 1
            for t in seq:
                if t.kernel_name == GEMM_WAIT and not any(t is w for _c, w in self._async_pairs.values()):
                    raise ValueError(f"chip {chip:#04x} cluster {cluster}: a gemm_wait the compiler did not make "
                                     f"({t.node_name}); write the dispatch as one gemm_full")
        return n_starts

    def _csr_reuse(self, programs, gemm_core):
        """Along each GEMM core's program, a dispatch whose configuration the array's CSRs hold already: a
        gemm_full becomes gemm_minimal, a configuring gemm_start takes `reuse`. Any other dispatch in between ends
        the run. Returns the number changed."""
        changed = 0
        for (_chip, _cluster, core), seq in programs.items():
            if core != gemm_core:
                continue
            held, prev_seq = None, None     # the configuration the CSRs hold, the seq of the start that wrote it
            for t in seq:
                k, a = t.kernel_name or "", t.kernel_args
                if k == cm.GEMM_FULL:
                    sig = ("full", tuple(sorted((f, v) for f, v in vars(a).items()
                                                if f not in _ADDRS and not f.startswith("_"))))
                    if sig == held:
                        t.kernel_name = cm.GEMM_MINIMAL
                        t.kernel_args = SnaxBingoKernelGemmMinimalArgs(*(getattr(a, f) for f in _ADDRS))
                        changed += 1
                    held, prev_seq = sig, None
                elif k == GEMM_START:
                    if a.stage == START_ONLY:
                        continue
                    sig = ("start", tuple(sorted((f, v) for f, v in vars(a).items()
                                                 if f not in _START_NOT_CONFIG and not f.startswith("_"))),
                           bool(a.input_C_addr))
                    if sig == held and prev_seq is not None and a.seq == prev_seq + 1:
                        a.reuse = 1
                        changed += 1
                    held, prev_seq = sig, a.seq
                elif k in (GEMM_WAIT, SHAPE_INIT, cm.GEMM_MINIMAL):
                    continue
                elif "gemm" in k or "gemv" in k:
                    held, prev_seq = None, None
        return changed
