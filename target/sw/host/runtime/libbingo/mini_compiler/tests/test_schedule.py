#!/usr/bin/env python3
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
#
# Tests for the compiler's core programs (passes/bingo_dfg_schedule.py): the order the schedule picks
# from the cost model, the asynchronous VersaCore dispatch it splits and fuses, the dispatch numbers
# and the CSR reuse along a GEMM core's program.
#
#   python3 test_schedule.py

import os as _os
import sys as _sys

import networkx as nx

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _bingo_paths  # noqa: F401,E402
from bingo_dfg import BingoDFG                                        # noqa: E402
from bingo_kernel_args import (SnaxBingoKernelGemmFullArgs, SnaxBingoKernelIdma1dCopyArgs,   # noqa: E402
                               SnaxBingoKernelIdmaRingLoadArgs, SnaxBingoKernelSimdFp16ToInt8Args)
from bingo_node import BingoNode                                      # noqa: E402

FAILED = []
GEMM, SIMD, XDMA, DM, HOST = 0, 1, 2, 3, 4


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  {detail}" if not cond else ""))
    if not cond:
        FAILED.append(name)


def new_dfg(async_exec=False):
    d = BingoDFG(num_chiplets=1, num_clusters_per_chiplet=2, num_cores_per_cluster=4, is_host_as_acc=True,
                 chiplet_ids=[0], dep_tag_width=5)
    d.core_roles = {"gemm": GEMM, "simd": SIMD, "xdma": XDMA, "dm": DM, "host": HOST}
    d.schedule_cores, d.async_exec = True, async_exec
    return d


def task(d, core, name, kernel, args, after=(), cluster=0):
    n = BingoNode(0, cluster, core, node_name=name, kernel_name=kernel, kernel_args=args)
    d.bingo_add_node(n)
    for a in after:
        d.bingo_add_edge(a, n)
    return n


def gemm(d, name, after=(), m=8, k=64, n=4):
    return task(d, GEMM, name, "__snax_bingo_kernel_gemm_full",
                SnaxBingoKernelGemmFullArgs(0x100, 0x200, 0, 0x300, M=m, K=k, N=n, array_shape_idx=0,
                                            transpose_A=0, transpose_B=0, accumPrevC=0), after)


def copy(d, name, size, after=(), core=DM):
    return task(d, core, name, "__snax_bingo_kernel_idma_1d_copy", SnaxBingoKernelIdma1dCopyArgs(0x10, 0x20, size),
                after)


def quant(d, name, after=()):
    return task(d, SIMD, name, "__snax_bingo_kernel_simd_fp16_to_int8",
                SnaxBingoKernelSimdFp16ToInt8Args(0x10, 0x20, beats=64, rows=1), after)


def program(d, core, cluster=0):
    """The core's program as the schedule left it: its tasks along the program edges."""
    ts = [t for t in d.node_list if t.assigned_core_id == core and t.assigned_cluster_id == cluster]
    h = nx.DiGraph()
    h.add_nodes_from(ts)
    h.add_edges_from((a, b) for a in ts for b in d.successors(a) if b in ts)
    return [t.node_name for t in nx.topological_sort(h)]


print("\nthe order: the task that can start first, not the one made first")
d = new_dfg()
g = gemm(d, "g")
late = task(d, DM, "late", "__snax_bingo_kernel_stamp", None, [g])
early = copy(d, "early", 4096)
d.bingo_schedule_programs()
check("a ready copy runs before a task that waits for a GEMM, though made after it",
      program(d, DM) == ["early", "late"], program(d, DM))
check("the graph stays acyclic", nx.is_directed_acyclic_graph(d))

print("\nasync: a wait trails the next start when that start is ready before the array is done")
d = new_dfg(async_exec=True)
x0, x1 = copy(d, "x0", 1024), copy(d, "x1", 1024)
g0 = gemm(d, "g0", [x0])
g1 = gemm(d, "g1", [x1])
q0, q1 = quant(d, "q0", [g0]), quant(d, "q1", [g1])
d.bingo_schedule_programs()
p = program(d, GEMM)
check("g0's wait comes after g1's start", p.index("g0_w") > p.index("g1"), p)
check("the dispatches are numbered 1 and 2 in program order",
      (g0.kernel_args.seq, g1.kernel_args.seq) == (1, 2), (g0.kernel_args.seq, g1.kernel_args.seq))
w0 = next(t for t in d.node_list if t.node_name == "g0_w")
check("g0's wait carries g0's number", w0.kernel_args.seq == 1)
check("g0's start holds its operands until its wait", getattr(g0, "operands_held_by", None) is w0)
check("q0 depends on g0's wait, not on its start", d.has_edge(w0, q0) and not d.has_edge(g0, q0))
check("g1 repeats g0's configuration: it reuses the CSRs", getattr(g1.kernel_args, "reuse", 0) == 1
      or any(t.node_name == "g1_cfg" and t.kernel_args.reuse == 1 for t in d.node_list))
check("the graph stays acyclic", nx.is_directed_acyclic_graph(d))

print("\nasync: a dispatch with nothing to overlap stays blocking; one with a late operand configures ahead")
d = new_dfg(async_exec=True)
x1 = task(d, DM, "x1", "__snax_bingo_kernel_idma_1d_copy", SnaxBingoKernelIdma1dCopyArgs(0x10, 0x20, 1 << 21),
          cluster=1)                # lands long after g0 is done
g0 = gemm(d, "g0", [], m=2, k=8, n=2)
g1 = gemm(d, "g1", [x1], m=2, k=8, n=2)
d.bingo_schedule_programs()
p = program(d, GEMM)
check("g0, its inputs there at once and the next start far off, stays one blocking gemm_full",
      g0.kernel_name == "__snax_bingo_kernel_gemm_full" and "g0_w" not in p, (g0.kernel_name, p))
cf = next((t for t in d.node_list if t.node_name == "g1_cfg"), None)
check("g1 configures ahead: a configure task that only writes the CSRs, then a start that only starts",
      cf is not None and cf.kernel_args.stage == 1 and getattr(cf, "config_only", False)
      and g1.kernel_args.stage == 2 and d.has_edge(cf, g1), (p, cf and cf.kernel_args.stage))
check("the configure runs before x1 lands: it has no dependency of its own",
      cf is not None and d.in_degree(cf) <= 1 and not d.has_edge(x1, cf))

print("\nasync: at most two dispatches started and not waited for")
d = new_dfg(async_exec=True)
xs = [copy(d, f"x{i}", 256) for i in range(4)]
gs = [gemm(d, f"g{i}", [xs[i]], k=256) for i in range(4)]
for i, gg in enumerate(gs):
    quant(d, f"q{i}", [gg])
d.bingo_schedule_programs()
p = program(d, GEMM)
check("the asynchronous programs are the ones kept (they finish sooner)", any(n.endswith("_w") for n in p), p)
open_, worst = 0, 0
for name in p:
    if name.endswith("_w"):
        open_ -= 1
    elif not name.endswith("_cfg"):
        open_ += 1
        worst = max(worst, open_)
check("never more than two outstanding on the array", worst <= 2, p)
check("the four dispatches numbered 1..4", sorted(t.kernel_args.seq for t in gs if hasattr(t.kernel_args, "seq"))
      == [i + 1 for i in range(sum(hasattr(t.kernel_args, "seq") for t in gs))])

print("\nring loads wait for their chip's prefetcher")
d = new_dfg()
ld = task(d, DM, "ld", "__snax_bingo_kernel_idma_ring_load",
          SnaxBingoKernelIdmaRingLoadArgs(0x40, 1, 0x50, 0x60, 8192, 0x70))
x = copy(d, "x", 4096)
start = task(d, DM, "start", "__snax_bingo_kernel_stamp", None, [x])
task(d, HOST, "pf", "__host_bingo_kernel_weight_prefetch", None, [start])
gemm(d, "use", [ld, x])
d.bingo_schedule_programs()
check("the DM core runs x and the start before the ring load made first",
      program(d, DM) == ["x", "start", "ld"], program(d, DM))

print("\nwithout async_exec every dispatch stays blocking")
d = new_dfg()
x0, x1 = copy(d, "x0", 1024), copy(d, "x1", 1024)
g0, g1 = gemm(d, "g0", [x0]), gemm(d, "g1", [x1])
d.bingo_schedule_programs()
check("no wait and no start made", not any(t.kernel_name in ("__snax_bingo_kernel_gemm_wait",
                                                               "__snax_bingo_kernel_gemm_start") for t in d.node_list))
check("the repeated configuration is gemm_minimal", g1.kernel_name == "__snax_bingo_kernel_gemm_minimal"
      or g0.kernel_name == "__snax_bingo_kernel_gemm_minimal")

print(f"\n{'ALL PASS' if not FAILED else f'{len(FAILED)} FAILED: {FAILED}'}")
_sys.exit(1 if FAILED else 0)
