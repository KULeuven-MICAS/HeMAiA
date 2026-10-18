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
        st = g.node(name, g.gemm, sa.KERNEL_NAME, sa, after)
        wt = g.node(f"{name}_w", g.gemm, "__snax_bingo_kernel_gemm_wait",
                    SnaxBingoKernelGemmWaitArgs(seq), [st])
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
    def program_order(self, dfg, device_cores, tag: str = "gemm", verbose: bool = True) -> int:
        """Each device core in `device_cores` (core ids) runs its tasks in creation order, as edges
        between consecutive tasks of every (chip, cluster, core), except that a queued dispatch's wait
        trails the next start of its phase. A move that would close a cycle with the dependency and
        order edges is undone. A queue with no dispatches (a disabled one) leaves the plain
        creation-order chain. Returns the number of edges added."""
        import networkx as nx
        dev = set(device_cores)
        groups = {}
        for t in dfg.node_list:
            if t.assigned_core_id in dev:
                groups.setdefault((t.assigned_chiplet_id, t.assigned_cluster_id,
                                   t.assigned_core_id), []).append(t)
        for ts in groups.values():
            ts.sort(key=lambda t: t.node_id)
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

        def program_edges(moved):
            edges = []
            for ts in groups.values():
                seq = sequence(ts, moved)
                edges += list(zip(seq, seq[1:]))
            return edges

        moved = set(movable)
        while True:
            g = nx.DiGraph(dfg)
            g.add_edges_from(dfg.order_edges)
            g.add_edges_from(program_edges(moved))
            try:
                cyc = nx.find_cycle(g)
            except nx.NetworkXNoCycle:
                break
            back = [u for u, _ in cyc if u in moved]
            if not back:
                raise ValueError(f"the program order closes a cycle: "
                                 f"{[(a.node_name, b.node_name) for a, b in cyc[:6]]}")
            moved.discard(back[0])
        added = 0
        for a, b in program_edges(moved):
            if not dfg.has_edge(a, b):
                dfg.bingo_add_edge(a, b)
                added += 1
        if not nx.is_directed_acyclic_graph(dfg):
            cyc = nx.find_cycle(dfg)
            raise ValueError(f"the program order closes a cycle: "
                             f"{[(a.node_name, b.node_name) for a, b in cyc[:6]]}")
        if verbose:
            print(f"[{tag}] program order: {added} core-order edges"
                  + (f"; {len(moved)} of {len(self.wait_phase)} GEMM waits behind the next start "
                     f"({len(movable) - len(moved)} kept in place by a cycle)" if self.enabled else ""))
        return added
