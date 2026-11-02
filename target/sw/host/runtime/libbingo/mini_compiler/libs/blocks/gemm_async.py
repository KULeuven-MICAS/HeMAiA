# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""Queued GEMM dispatch for the blocks: a start task and a wait task in place of one blocking task.

A blocking GEMM task runs the core's software set-up with the array idle. VersaCore and its streamer
queue a configuration, so a start task (`gemm_start`, `gemm_fa_qk_start`, `gemm_fa_pv_start`)
configures and starts the cluster's n-th dispatch and returns, the core programs the next dispatch
while the array runs this one, and a wait task (`gemm_wait`) returns once the array has retired
dispatch n. Three rules come with it, and this module is where a block gets all three right:

  NUMBERING. A cluster's VersaCore dispatches are numbered from 1 in the order its GEMM core issues
  them, and every dispatch of the cluster carries a number: the wait compares the array's retired
  counter with it. `GemmQueue.dispatch` numbers them in creation order, so every GEMM a graph puts on
  a cluster goes through it, or none does.

  COMPLETION. The output is written when the wait is done, never when the start is. `dispatch`
  returns the task a consumer depends on, and a buffer a dispatch reads is free after that task too.
  A dispatch's INPUTS are read when the start runs (a PV's corr factors are read from L1 there), so
  their producers precede the start, which `after` gives.

  PROGRAM ORDER. A wait that sits between two starts of the core's program keeps the array idle for
  the second start's set-up. `GemmQueue.program_order` turns each device core's creation order into
  dependency edges (the compiler may reorder tasks of one core that have no path between them), with
  one exception: a wait trails the next start of its phase where nothing else sits between them and
  the program stays acyclic with every dependency and order edge. A phase names the dispatches whose
  waits may move behind each other's starts; a wait that releases something the next start waits for
  belongs to a phase of its own.

  CONFIGURATION AHEAD (`cfg_ahead`: gemm_start dispatches; `cfg_ahead_fa`: a FlashAttention
  dispatch that would not REUSE its cluster's previous one). Plain CSR writes go to the managers'
  register files at any time; only the start write waits for room in the queue. So a dispatch can be
  two tasks: a CONFIGURE task that writes every CSR and starts nothing, and a START task that waits
  for the operands and starts. The configure task has no dependency of its own, and the program
  order puts it right before its start, behind the waits there, so it delays no wait: the core runs
  it as soon as it gets there, while the start still waits for its operands or the array still runs
  the previous dispatch. It is left out when the waits' places are chosen, so a wait still trails
  the start after it. It names the operands' addresses without touching them, so liveness does not
  count it as a user (`config_only`).

  CONFIGURATION HOISTED (`cfg_hoist`): a gemm_start dispatch's configure task with no dependency of its
  own goes ahead of the waits right before its start instead (never ahead of a start, so a REUSE dispatch still finds the register
  file it expects). Where a phase changes (the sweep's last PV, then W_UV) the start cannot move ahead of
  those waits, so without this the configure runs after the array retired the previous dispatch; ahead
  of them it runs while the array works, and the waits behind it return that much later.

The queue is a registry on the DFG (`GemmQueue.of(dfg)`), because the blocks that build dispatches
and the application that orders the cores share nothing else.
"""

import inspect

from bingo_kernel_args import (
    SnaxBingoKernelGemmFaPvArgs,
    SnaxBingoKernelGemmFaPvStartArgs,
    SnaxBingoKernelGemmFaQkArgs,
    SnaxBingoKernelGemmFaQkStartArgs,
    SnaxBingoKernelGemmFullArgs,
    SnaxBingoKernelGemmStartArgs,
    SnaxBingoKernelGemmWaitArgs,
)

FA_QK, FA_PV, FULL = "fa_qk", "fa_pv", "full"

# The fields a gemm_start descriptor takes from gemm_full's: gemm_full's constructor parameters, so a
# field added there is copied without an edit here.
_FULL_FIELDS = tuple(n for n in inspect.signature(SnaxBingoKernelGemmFullArgs.__init__).parameters
                     if n != "self")


def _kind(args):
    if isinstance(args, SnaxBingoKernelGemmFaQkArgs):
        return FA_QK
    if isinstance(args, SnaxBingoKernelGemmFaPvArgs):
        return FA_PV
    if isinstance(args, SnaxBingoKernelGemmFullArgs):
        return FULL
    raise TypeError(f"{type(args).__name__} is not a GEMM args class the queue knows")


def start_args(args, seq: int, shape_addr=0):
    """The start-kernel descriptor of a dispatch given as its blocking kernel's args."""
    kind = _kind(args)
    if kind == FULL:
        return SnaxBingoKernelGemmStartArgs(**{f: getattr(args, f) for f in _FULL_FIELDS}, seq=seq,
                                            shape_addr=shape_addr)
    if args.perf_addr:
        raise ValueError("a queued FlashAttention matmul takes no perf_addr: the array's counters "
                         "reset on every configuration write")
    cls = SnaxBingoKernelGemmFaQkStartArgs if kind == FA_QK else SnaxBingoKernelGemmFaPvStartArgs
    return cls(args.input_A_addr, args.input_B_addr, args.input_C_addr, args.output_D_addr,
               M=args.M, K=args.K, N=args.N, flags=args.flags, corr_addr=args.corr_addr,
               d_shift=args.d_shift, b_pitch=args.b_pitch, seq=seq)


class GemmQueue:
    """The numbering and the start / wait pairs of every GEMM core of one DFG."""

    def __init__(self, enabled: bool = True):
        self.enabled = bool(enabled)
        self.seq = {}                 # (chip, cluster) -> dispatches numbered so far
        self.start_phase = {}         # start task -> phase
        self.wait_phase = {}          # wait task -> phase
        self.cfg_ahead = False        # gemm_start dispatches as a configure and a start task
        self.cfg_ahead_fa = False     # FA dispatches that would not REUSE, the same
        self.cfg_task = {}            # configure task -> its start task
        self.cfg_hoist = False        # gemm_start configure tasks ahead of the waits before their start
        self.last_sig = {}            # (chip, cluster) -> the previous dispatch's configuration

    @classmethod
    def of(cls, dfg, enabled: bool = None) -> "GemmQueue":
        """The DFG's queue, made on first use. `enabled` (first use only, or to check): False makes
        every dispatch a blocking task, so a block can offer the switch without a second code path."""
        q = getattr(dfg, "gemm_queue", None)
        if q is None:
            q = dfg.gemm_queue = cls(True if enabled is None else enabled)
        elif enabled is not None and q.enabled != bool(enabled):
            raise ValueError("one DFG mixes queued and blocking GEMM dispatches: every dispatch of "
                             "a cluster must be numbered, or none")
        return q

    def dispatch(self, g, name: str, args, after=(), phase: str = "gemm", shape_addr=0):
        """One GEMM on cluster `g` (a Ctx at that cluster) after `after`. `args`: the blocking
        kernel's descriptor (FlashAttention QK / PV, or gemm_full). Returns the task a consumer
        depends on: the wait of a queued dispatch, the blocking task otherwise."""
        kind = _kind(args)
        if not self.enabled:
            kernel = {FA_QK: "__snax_bingo_kernel_gemm_fa_qk", FA_PV: "__snax_bingo_kernel_gemm_fa_pv",
                      FULL: "__snax_bingo_kernel_gemm_full"}[kind]
            return g.node(name, g.gemm, kernel, args, after)
        key = (g.chiplet, g.cluster)
        seq = self.seq[key] = self.seq.get(key, 0) + 1
        sa = start_args(args, seq, shape_addr)
        # what bingo_cfg_reuse.reuse_gemm_fa_config compares: an FA dispatch that repeats the previous one REUSEs it
        sig = (("full",) if kind == FULL else
               (kind, sa.M, sa.K, sa.N, sa.flags, sa.d_shift, sa.b_pitch, bool(sa.input_C_addr)))
        repeat = sig == self.last_sig.get(key)
        self.last_sig[key] = sig
        cf = None
        if kind == FULL and self.cfg_ahead:
            ca = start_args(args, seq, shape_addr)
            ca.stage, sa.stage = ca.CONFIGURE_ONLY, sa.START_ONLY
            cf = g.node(f"{name}_cfg", g.gemm, ca.KERNEL_NAME, ca, [])
            cf.config_only = True
        elif kind != FULL and self.cfg_ahead_fa and not repeat:
            ca = start_args(args, seq, shape_addr)
            ca.flags |= ca.CONFIGURE
            sa.flags |= sa.START
            cf = g.node(f"{name}_cfg", g.gemm, ca.KERNEL_NAME, ca, [])
            cf.config_only = True
        st = g.node(name, g.gemm, sa.KERNEL_NAME, sa, after)
        if cf is not None:
            self.cfg_task[cf] = st
        wt = g.node(f"{name}_w", g.gemm, "__snax_bingo_kernel_gemm_wait",
                    SnaxBingoKernelGemmWaitArgs(seq), [st])
        # the array streams the start's operands until the wait: liveness counts the wait as their user
        # (bingo_liveness RULE 1), or the static L1 planner frees them when the start returns
        st.operands_held_by = wt
        self.start_phase[st] = self.wait_phase[wt] = phase
        return wt

    def note_blocking(self, g):
        """A blocking dispatch on cluster `g` that the queue did not make (another block's GEMM, a
        GEMV): it consumes a sequence number, since the array's retired counter counts it. Call it in
        the order the GEMM core issues it among the queued ones. Returns the number it took."""
        if not self.enabled:
            return 0
        key = (g.chiplet, g.cluster)
        seq = self.seq[key] = self.seq.get(key, 0) + 1
        return seq

    def qk(self, g, name, args, after=(), phase="fa"):
        if _kind(args) != FA_QK:
            raise TypeError("qk() takes SnaxBingoKernelGemmFaQkArgs")
        return self.dispatch(g, name, args, after, phase)

    def pv(self, g, name, args, after=(), phase="fa"):
        if _kind(args) != FA_PV:
            raise TypeError("pv() takes SnaxBingoKernelGemmFaPvArgs")
        return self.dispatch(g, name, args, after, phase)

    # ---- the program order -------------------------------------------------------------------
    def _renumber(self, groups):
        """NUMBERING in the program's order: `dispatch` numbered in creation order, and an order_key that
        moves a start moves the core's issue order, so each cluster's starts (with their configure and
        wait tasks) take 1, 2, ... in the order the GEMM core issues them. A wait numbered for creation
        order would wait for a dispatch the core issues after it."""
        cfg_of = {st: cf for cf, st in self.cfg_task.items()}
        for (chip, cluster, _), ts in groups.items():
            starts = [t for t in ts if t in self.start_phase]
            if not starts:
                continue
            if len(starts) != self.seq.get((chip, cluster), 0):
                raise ValueError(f"chip {chip:#04x} cluster {cluster}: {len(starts)} queued starts on one core of "
                                 f"{self.seq.get((chip, cluster), 0)} numbered dispatches; renumbering needs "
                                 f"every dispatch of the cluster queued on its GEMM core")
            for n, st in enumerate(starts, 1):
                st.kernel_args.seq = n
                st.operands_held_by.kernel_args.seq = n
                if st in cfg_of:
                    cfg_of[st].kernel_args.seq = n

    def program_order(self, dfg, device_cores, tag: str = "gemm", verbose: bool = True) -> int:
        """Each device core in `device_cores` (core ids) runs its tasks in creation order, as edges
        between consecutive tasks of every (chip, cluster, core), except that a queued dispatch's wait
        trails the next start of its phase. A move that would close a cycle with the dependency and
        order edges is undone. A queue with no dispatches (a disabled one) leaves the plain
        creation-order chain. Returns the number of edges added."""
        import networkx as nx
        from bingo_reach import find_cycle
        dev = set(device_cores)
        groups = {}
        for t in dfg.node_list:
            if t.assigned_core_id in dev:
                groups.setdefault((t.assigned_chiplet_id, t.assigned_cluster_id,
                                   t.assigned_core_id), []).append(t)
        # a task's place in its core's program: its creation order, or its order_key where a generator sets one
        # (a task made late that must run early on its core); the dependency edges still bind it
        for ts in groups.values():
            ts.sort(key=lambda t: (getattr(t, "order_key", t.node_id), t.node_id))
        self._renumber(groups)
        cfgs = {key: [t for t in ts if t in self.cfg_task] for key, ts in groups.items()}
        # cfg_hoist moves only a configure with no dependency of its own (a generator may give one, e.g. an end anchor)
        dep_free = {c for c in self.cfg_task if dfg.in_degree(c) == 0} - {b for _, b in dfg.order_edges}
        groups = {key: [t for t in ts if t not in self.cfg_task] for key, ts in groups.items()}
        movable = set()
        for ts in groups.values():
            for a, b in zip(ts, ts[1:]):
                if a in self.wait_phase and self.start_phase.get(b) == self.wait_phase[a]:
                    movable.add(a)

        def sequence(ts, moved):
            out, held = [], None
            for t in ts:
                if held is not None and t is not held:
                    out.append(t)
                    out.append(held)
                    held = None
                    continue
                if t in moved:
                    held = t
                    continue
                out.append(t)
            if held is not None:
                out.append(held)
            return out

        def placed(key, seq):
            # each configure task right before its start (cfg_hoist, a gemm_start's: before the waits there)
            out = list(seq)
            for c in cfgs[key]:
                i = out.index(self.cfg_task[c])
                if self.cfg_hoist and c in dep_free and c.kernel_name.endswith("_gemm_start"):
                    while i > 0 and out[i - 1] in self.wait_phase:
                        i -= 1
                out.insert(i, c)
            return out

        def program_edges(moved):
            edges = []
            for key, ts in groups.items():
                seq = placed(key, sequence(ts, moved))
                edges += list(zip(seq, seq[1:]))
            return edges

        moved = set(movable)
        # One graph, updated in place: undoing a move changes only its group's chain. The dependency and
        # order edges stay, and a task's one program edge stays the last of its successors, so every
        # find_cycle walks the graph in the order a rebuilt copy would and undoes the same moves.
        g = nx.DiGraph(dfg)
        g.add_edges_from(dfg.order_edges)
        fixed = set(g.edges)
        group_of = {t: key for key, ts in groups.items() for t in ts}
        chain = {}
        for key, ts in groups.items():
            seq = placed(key, sequence(ts, moved))
            chain[key] = list(zip(seq, seq[1:]))
            g.add_edges_from(chain[key])
        while True:
            try:
                cyc = find_cycle(g)
            except nx.NetworkXNoCycle:
                break
            back = [u for u, _ in cyc if u in moved]
            if not back:
                raise ValueError(f"the program order closes a cycle: "
                                 f"{[(a.node_name, b.node_name) for a, b in cyc[:6]]}")
            moved.discard(back[0])
            key = group_of[back[0]]
            g.remove_edges_from([e for e in chain[key] if e not in fixed])
            seq = placed(key, sequence(groups[key], moved))
            chain[key] = list(zip(seq, seq[1:]))
            g.add_edges_from(chain[key])
        added = 0
        for a, b in program_edges(moved):
            if not dfg.has_edge(a, b):
                dfg.bingo_add_edge(a, b)
                added += 1
        if not nx.is_directed_acyclic_graph(dfg):
            cyc = nx.find_cycle(dfg)
            raise ValueError(f"the program order closes a cycle: "
                             f"{[(a.node_name, b.node_name) for a, b in cyc[:6]]}")
        n_checked = self.check_numbering(dfg, device_cores)
        if verbose and self.cfg_task:
            print(f"[{tag}] program order: {len(self.cfg_task)} dispatches configured ahead"
                  + (", gemm_start configures ahead of the waits before their start" if self.cfg_hoist else ""))
        if verbose:
            print(f"[{tag}] program order: {added} core-order edges"
                  + (f"; {len(moved)} of {len(self.wait_phase)} GEMM waits behind the next start "
                     f"({len(movable) - len(moved)} kept in place by a cycle); numbering checked on "
                     f"{n_checked} GEMM cores" if self.enabled else ""))
        return added

    def check_numbering(self, dfg, device_cores) -> int:
        """NUMBERING on the finished program, read from the graph rather than from the sort that made it:
        each core's tasks form one chain of edges, and along it the starts carry 1, 2, ... with their
        configure and wait tasks on the same number. Returns the number of cores checked."""
        import networkx as nx
        dev = set(device_cores)
        groups = {}
        for t in dfg.node_list:
            if t.assigned_core_id in dev:
                groups.setdefault((t.assigned_chiplet_id, t.assigned_cluster_id,
                                   t.assigned_core_id), []).append(t)
        cfg_of = {st: cf for cf, st in self.cfg_task.items()}
        n_cores = 0
        for (chip, cluster, core), ts in groups.items():
            if not any(t in self.start_phase for t in ts):
                continue
            mine = set(ts)
            h = nx.DiGraph()
            h.add_nodes_from(ts)
            h.add_edges_from((a, b) for a in ts for b in dfg.successors(a) if b in mine)
            order = list(nx.topological_sort(h))
            for a, b in zip(order, order[1:]):
                if not dfg.has_edge(a, b):
                    raise ValueError(f"chip {chip:#04x} cluster {cluster} core {core}: no program edge "
                                     f"{a.node_name} -> {b.node_name}")
            n = 0
            for t in order:
                if t not in self.start_phase:
                    continue
                n += 1
                got = [t.kernel_args.seq, t.operands_held_by.kernel_args.seq]
                if t in cfg_of:
                    got.append(cfg_of[t].kernel_args.seq)
                if any(s != n for s in got):
                    raise ValueError(f"chip {chip:#04x} cluster {cluster}: {t.node_name} is the GEMM core's "
                                     f"dispatch {n} but is numbered {got} (start, wait[, configure])")
            n_cores += 1
        return n_cores
