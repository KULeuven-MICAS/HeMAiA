#!/usr/bin/env python3
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
#
# Tests for the compiler's core programs (passes/bingo_dfg_schedule.py): the order the schedule picks
# from the cost model, the asynchronous VersaCore dispatch it splits and fuses, the dispatch numbers,
# the CSR reuse along a GEMM core's program and the edges that let the static L1 plan reuse space.
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
d = new_dfg(async_exec="all")
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
check("g1 repeats g0's configuration: its start writes the addresses and starts (reuse, configure and start)",
      g1.kernel_args.reuse == 1 and g1.kernel_args.stage == g1.kernel_args.CONFIGURE_AND_START,
      (g1.kernel_args.reuse, g1.kernel_args.stage))
check("g1 keeps no configure task: an address-only one folds into its start",
      "g1_cfg" not in {t.node_name for t in d.node_list} and "g1_cfg" not in program(d, GEMM), program(d, GEMM))
check("node ids stay their positions after the fold (the emitter's global task ids)",
      [t.node_id for t in d.node_list] == list(range(len(d.node_list))), [t.node_id for t in d.node_list])
check("g1's start carries its number and the shape's L1 copy", g1.kernel_args.seq == 2
      and g1.kernel_args.shape_addr is not None and g1.kernel_args.shape_addr == g0.kernel_args.shape_addr)
check("the graph stays acyclic", nx.is_directed_acyclic_graph(d))

print("\nasync: a dispatch with nothing to overlap stays blocking; one with a late operand configures ahead")
d = new_dfg(async_exec="all")
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
d = new_dfg(async_exec="all")
xs = [copy(d, f"x{i}", 256) for i in range(4)]
gs = [gemm(d, f"g{i}", [xs[i]], k=256) for i in range(4)]
for i, gg in enumerate(gs):
    quant(d, f"q{i}", [gg])
d.bingo_schedule_programs()
p = program(d, GEMM)
check("the asynchronous programs are the ones kept (they finish sooner)", any(n.endswith("_w") for n in p), p)
open_, worst = 0, 0
for name in p:
    if name.startswith("gemm_shape"):
        continue
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

print("\nthe VersaCore shape table: one setup a cluster, first on its GEMM core, read by every configuring part")
d = new_dfg(async_exec="all")
x0, x1 = copy(d, "x0", 1024), copy(d, "x1", 1024)
g0, g1 = gemm(d, "g0", [x0]), gemm(d, "g1", [x1])
quant(d, "q0", [g0])
quant(d, "q1", [g1])
d.bingo_schedule_programs()
p = program(d, GEMM)
init = next((t for t in d.node_list if t.node_name.startswith("gemm_shape_0_init")), None)
check("one shape setup, first in the GEMM core's program", init is not None and p[0] == init.node_name, p)
starts = [t for t in d.node_list if t.kernel_name == "__snax_bingo_kernel_gemm_start"]
check("every start part names the table (a configure task alone is no user to liveness)",
      starts and all(t.kernel_args.shape_addr is init.kernel_args.dst_addr for t in starts),
      [(t.node_name, t.kernel_args.stage) for t in starts])

print("\nSIMD configure-ahead (csr_mode): the configure runs while the start waits for its input")
d = new_dfg(async_exec="all")
xl = copy(d, "xl", 1 << 20)
q = quant(d, "q", [xl])
d.bingo_schedule_programs()
cf = next((t for t in d.node_list if t.node_name == "q_cfg"), None)
check("q_cfg configures (csr_mode 1), q starts what it left (csr_mode 2)",
      cf is not None and cf.kernel_args.csr_mode == 1 and q.kernel_args.csr_mode == 2 and d.has_edge(cf, q),
      (cf and cf.kernel_args.csr_mode, q.kernel_args.csr_mode))
check("no wait: the start drains", not any(t.node_name == "q_w" for t in d.node_list))

print("\nSIMD fire / wait: the next task's configure runs while this one's work is in flight")
d = new_dfg(async_exec="all")
xa, xb = copy(d, "xa", 512), copy(d, "xb", 512)
qa = task(d, SIMD, "qa", "__snax_bingo_kernel_simd_fp16_to_int8",
          SnaxBingoKernelSimdFp16ToInt8Args(0x10, 0x20, beats=1024, rows=1), [xa])
qb = task(d, SIMD, "qb", "__snax_bingo_kernel_simd_fp16_to_int8",
          SnaxBingoKernelSimdFp16ToInt8Args(0x30, 0x40, beats=1024, rows=1), [xb])
d.bingo_schedule_programs()
p = program(d, SIMD)
check("qb's configure goes between qa's launch and qa's wait",
      "qb_cfg" in p and "qa_w" in p and p.index("qa") < p.index("qb_cfg") < p.index("qa_w"), p)
check("qa launches without draining (csr_mode 3 or 4: fire), qa_w is simd_wait_all",
      qa.kernel_args.csr_mode in (3, 4) and next(t for t in d.node_list if t.node_name == "qa_w").kernel_name.endswith(
          "simd_wait_all"), (qa.kernel_args.csr_mode, p))

print("\nxDMA start / wait: the 6D transfer returns once started (NO_WAIT), xdma_wait_all retires it")
from bingo_kernel_args import SnaxBingoKernelXdma6dArgs                                 # noqa: E402
d = new_dfg(async_exec="all")
ya, yb = copy(d, "ya", 512), copy(d, "yb", 512)
ra = task(d, XDMA, "ra", "__snax_bingo_kernel_xdma_6d",
          SnaxBingoKernelXdma6dArgs(0x100, 0x200, 64, 64, [64], [256], [64], [256]), [ya])
rb = task(d, XDMA, "rb", "__snax_bingo_kernel_xdma_6d",
          SnaxBingoKernelXdma6dArgs(0x300, 0x400, 64, 64, [64], [256], [64], [256]), [yb])
d.bingo_schedule_programs()
p = program(d, XDMA)
waits = [t for t in d.node_list if t.kernel_name.endswith("xdma_wait_all")]
emitted = ra.kernel_args.get_c_field_assignments({})["num_temporal_dims"]
check("a start with NO_WAIT (bit 10) and an xdma_wait_all after it", waits and int(emitted) & (1 << 10), (emitted, p))

print("\nthe softmax fire / wait: two in flight, no other SIMD kernel between a fire and its wait")
from bingo_kernel_args import SnaxBingoKernelSimdFaSoftmaxArgs, set_simd_beat_bytes     # noqa: E402
from bingo_mem_handle import BingoMemAlloc                                             # noqa: E402
set_simd_beat_bytes(32)
d = new_dfg(async_exec="all")


def buf(name, n):
    return BingoMemAlloc(name, size=n, mem_level="L1", chip_id=0, cluster_id=0)


sms = []
for i in range(3):
    xi = copy(d, f"s{i}", 2048)
    sms.append(task(d, SIMD, f"sm{i}", "__snax_bingo_kernel_simd_fa_softmax",     # tiles long enough to overlap
                    SnaxBingoKernelSimdFaSoftmaxArgs(buf(f"s16_{i}", 65536), buf(f"p8_{i}", 32768),
                                                     buf(f"ar{i}", 65536), bc=1024, dhead=64, tile_idx=0), [xi]))
quant(d, "qq", [sms[0]])
d.bingo_schedule_programs()
p = program(d, SIMD)
fires = [n for n in p if n.startswith("sm") and not n.endswith("_w")]
check("the softmaxes fire with NO_DRAIN and are retired by their waits",
      all(t.kernel_args.p_mode & 4 for t in sms) and all(f"sm{i}_w" in p for i in range(3)), p)
ok_excl = True
for i in range(3):
    lo, hi = p.index(f"sm{i}"), p.index(f"sm{i}_w")
    ok_excl &= all(n.startswith("sm") for n in p[lo:hi + 1])
check("nothing but softmax parts between a fire and its wait", ok_excl, p)
check("at least one wait trails the next fire", any(p.index(f"sm{i}_w") > p.index(f"sm{i + 1}") for i in range(2)), p)
check("no softmax fire between a configure and its start (a fire writes the SIMD's CSRs itself)",
      all(not any(n.startswith("sm") and not n.endswith("_w") for n in p[p.index(c) + 1:p.index(c[:-4])])
          for c in p if c.endswith("_cfg")), p)

print("\nthe ChainGather configure / start / wait: the wait takes the gather's name")
from bingo_kernel_args import SnaxBingoKernelXdmaChainGatherArgs                       # noqa: E402
d = new_dfg(async_exec="all")
xl = copy(d, "xl", 1 << 20)
ga = task(d, XDMA, "y_chain", "__snax_bingo_kernel_xdma_chain_gather",
          SnaxBingoKernelXdmaChainGatherArgs(buf("loc", 4096), [0x1000, 0x2000], 4096, "0", 0), [xl])
quant(d, "after", [ga])
d.bingo_schedule_programs()
names = {t.node_name: t.kernel_name for t in d.node_list}
check("y_chain_cfg (chain_cfg), y_chain_st (chain_start), y_chain (chain_wait)",
      names.get("y_chain_cfg", "").endswith("chain_cfg") and names.get("y_chain_st", "").endswith("chain_start")
      and names.get("y_chain", "").endswith("chain_wait"), names)
w = next(t for t in d.node_list if t.node_name == "y_chain")
check("the consumer follows the wait", any(s.node_name == "after" for s in d.successors(w)))

print("\nFlashAttention QK as a configure / start / wait on the array's counter")
from bingo_kernel_args import SnaxBingoKernelGemmFaQkArgs                               # noqa: E402
d = new_dfg(async_exec="all")
qs = []
for i in range(2):
    xi = copy(d, f"k{i}", 1024)
    qs.append(task(d, GEMM, f"qk{i}", "__snax_bingo_kernel_gemm_fa_qk",
                   SnaxBingoKernelGemmFaQkArgs(buf(f"K{i}", 8192), buf(f"Q{i}", 8192), 0, buf(f"S{i}", 8192),
                                               M=4, K=32, N=2), [xi]))
    quant(d, f"sq{i}", [qs[-1]])
d.bingo_schedule_programs()
check("the starts are gemm_fa_qk_start, numbered 1 and 2",
      all(t.kernel_name.endswith("_qk_start") for t in qs) and sorted(t.kernel_args.seq for t in qs) == [1, 2],
      [(t.kernel_name, getattr(t.kernel_args, "seq", None)) for t in qs])
second = max(qs, key=lambda t: t.kernel_args.seq)
check("the second repeats the first's configuration: its start REUSEs it and starts (no CONFIGURE/START flag)",
      second.kernel_args.flags & 8 and not second.kernel_args.flags & (16 | 32), hex(second.kernel_args.flags))
check("the second keeps no configure task",
      not any(t.node_name == second.node_name + "_cfg" for t in d.node_list), [t.node_name for t in d.node_list])

print("\neach accelerator's splits kept only where they gain: long GEMMs split, a SIMD task configured ahead only when "
      "its input is late")
d = new_dfg(async_exec=True)
xs = [copy(d, f"x{i}", 4096) for i in range(4)]
gs = [gemm(d, f"g{i}", [xs[i]], m=16, k=64, n=16) for i in range(4)]
qs = [quant(d, f"q{i}", [gs[i]]) for i in range(4)]
d.bingo_schedule_programs()
names = {t.node_name for t in d.node_list}
check("the GEMMs run as start / wait parts", all(f"g{i}_w" in names for i in range(4)), sorted(names))
check("a quantise of a GEMM's output configures while the GEMM runs (the SIMD core is idle then)",
      "q3_cfg" in names, sorted(names))
d = new_dfg(async_exec=True)
xs = [copy(d, f"x{i}", 4096) for i in range(4)]
gs = [gemm(d, f"g{i}", [xs[i]], m=16, k=64, n=16) for i in range(4)]
qs = [quant(d, f"q{i}", [xs[i]]) for i in range(4)]
d.bingo_schedule_programs()
names = {t.node_name for t in d.node_list}
check("quantises whose inputs are there stay whole tasks (a split only adds dispatches)",
      all(q.kernel_name.endswith("fp16_to_int8") and getattr(q.kernel_args, "csr_mode", 0) == 0 for q in qs)
      and not any(n.endswith("_cfg") and n.startswith("q") for n in names), sorted(names))

print("\nL1: loads share space only where an edge orders the last reader of a tile before the next load")
from bingo_l1_packer import StaticL1Options                                             # noqa: E402
d = new_dfg()
d.static_l1_opts, d.l1_capacity_bytes = StaticL1Options(enable=True), 100 * 1024
ks = [buf(f"K{i}", 40 * 1024) for i in range(3)]
lds = [task(d, DM, f"ld{i}", "__snax_bingo_kernel_idma_1d_copy", SnaxBingoKernelIdma1dCopyArgs(0x10, ks[i], 40 * 1024))
       for i in range(3)]
gs = [task(d, GEMM, f"g{i}", "__snax_bingo_kernel_gemm_full",
           SnaxBingoKernelGemmFullArgs(ks[i], 0x200, 0, buf(f"O{i}", 1024), M=8, K=64, N=4, array_shape_idx=0,
                                       transpose_A=0, transpose_B=0, accumPrevC=0), [lds[i]]) for i in range(3)]
d.bingo_schedule_programs()
check("the three 40 KB tiles do not fit 100 KB at once, yet the compiler's order is kept",
      not d.kept_order and d.l1_budget is not None, (d.kept_order, d.l1_budget))
check("a later load follows a GEMM that read an earlier tile (an edge on the cluster, no new task)",
      any(d.has_edge(g, ld) for i, ld in enumerate(lds) for g in gs[:i]), program(d, DM))
check("no task was added", len(d.node_list) == 6, len(d.node_list))
check("the graph stays acyclic", nx.is_directed_acyclic_graph(d))

print("\nthe made order (the fallback where the compiler's order overflows L1): in-order cores, no passing")
from bingo_dfg_schedule import _Model                                                    # noqa: E402
d = new_dfg()
g = gemm(d, "g")
w = task(d, DM, "w", "__snax_bingo_kernel_stamp", None, [g])          # made first, waits for the GEMM
r = copy(d, "r", 4096)                                                 # made second, ready at once
seq = next(vs for vs in _Model(d, d.core_roles, {}, True)._made_order().values() if w in vs)
check("a ready task does not pass the core's earlier task that waits",
      [t.node_name for t in seq] == ["w", "r"], [t.node_name for t in seq])
d = new_dfg()
b0 = copy(d, "b0", 1024)
a = task(d, DM, "a", "__snax_bingo_kernel_stamp", None)               # made first, but waits for b (made later)
b = copy(d, "b", 1024, [b0])
g2 = gemm(d, "g2", [b])
d.bingo_add_edge(g2, a)
seq = next(vs for vs in _Model(d, d.core_roles, {}, True)._made_order().values() if a in vs)
check("where the made order would close a cycle, the task it waits for passes it",
      [t.node_name for t in seq] == ["b0", "b", "a"], [t.node_name for t in seq])

print(f"\n{'ALL PASS' if not FAILED else f'{len(FAILED)} FAILED: {FAILED}'}")
_sys.exit(1 if FAILED else 0)
