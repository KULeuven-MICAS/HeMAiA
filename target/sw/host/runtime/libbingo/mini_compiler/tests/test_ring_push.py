#!/usr/bin/env python3
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
#
# Tests for the weight rings' cluster mode (libs/blocks/load_stream.py WeightRings.push_from): every run a
# ring push on its cluster's DM core, and the edges that are the whole protocol -- the loads that last read
# the bytes a run overwrites before its push, the pushes of a ring in its order, the queue share, and each
# load after its run's push.
#
#   python3 test_ring_push.py

import os as _os
import sys as _sys

import networkx as nx

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _bingo_paths  # noqa: F401,E402
from bingo_dfg import BingoDFG                                        # noqa: E402
from bingo_kernel_args import SnaxBingoKernelGemmFullArgs, SnaxBingoKernelRingPushArgs  # noqa: E402
from bingo_mem_handle import BingoMemFixedAddr, BingoMemSymbol        # noqa: E402
from bingo_node import BingoNode                                      # noqa: E402
from libs.blocks import WeightRings                                   # noqa: E402

FAILED = []
GEMM, SIMD, XDMA, DM, HOST = 0, 1, 2, 3, 4
PUSH = SnaxBingoKernelRingPushArgs.KERNEL_NAME
LOAD = "__snax_bingo_kernel_idma_ring_load"
SEQ = 0x200000000000


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  {detail}" if not cond else ""))
    if not cond:
        FAILED.append(name)


def raises(fn, text):
    try:
        fn()
    except (ValueError, KeyError) as e:
        return text in str(e), str(e)
    return False, "no error"


def new_dfg():
    d = BingoDFG(num_chiplets=1, num_clusters_per_chiplet=2, num_cores_per_cluster=4, is_host_as_acc=True,
                 chiplet_ids=[0], dep_tag_width=5)
    d.core_roles = {"gemm": GEMM, "simd": SIMD, "xdma": XDMA, "dm": DM, "host": HOST}
    d.schedule_cores = True
    return d


def task(d, core, name, kernel, args, after=(), cluster=0):
    n = BingoNode(0, cluster, core, node_name=name, kernel_name=kernel, kernel_args=args)
    d.bingo_add_node(n)
    for a in after:
        d.bingo_add_edge(a, n)
    return n


def build(n_slots=4, slot_bytes=8192, batch=2, depth=8, chunks=12, sizes=(8192, 4096, 4096, 8192)):
    """Two clusters, each streaming `chunks` weight chunks through its ring into two L1 slabs, a GEMM reading each.
    Returns (dfg, rings, {cluster: [(load, ring byte lo, ring byte hi)]}, start)."""
    d = new_dfg()
    rings = WeightRings(slots=BingoMemSymbol("ring_slots"), flags=BingoMemSymbol("ring_flags"),
                        releases=BingoMemSymbol("ring_rel"), n_rings=2, n_slots=n_slots, slot_bytes=slot_bytes,
                        batch=batch)
    rings.push_from(memchip=0x20, engine=2, seq=BingoMemFixedAddr(SEQ, mem_level="HBM"), edge=d.bingo_add_edge,
                    depth=depth)
    start = task(d, DM, "start", "__snax_bingo_kernel_stamp", None)
    rings.start_after(start)
    out = {}
    for c in range(2):
        reader, src, out[c] = [None, None], 0x100000000000 + c * 0x1000000, []
        for j in range(chunks):
            nb = sizes[j % len(sizes)]
            i = j % 2
            ld, sl = rings.load_chunk(
                lambda n, core, kern, args, after, c=c: task(d, core, f"{n}_c{c}", kern, args, after, cluster=c), c,
                f"ld{j}", src=src, nbytes=nb, slab=0x10000 + i * 0x10000,
                war=[reader[i]] if reader[i] else [], dm=DM)
            lo = sl.offset - c * n_slots * slot_bytes
            out[c].append((ld, lo, lo + nb))
            reader[i] = task(d, GEMM, f"use{j}_c{c}", "__snax_bingo_kernel_gemm_full",
                             SnaxBingoKernelGemmFullArgs(0x100, 0x10000 + i * 0x10000, 0, 0x300, M=8, K=64, N=4,
                                                         array_shape_idx=0, transpose_A=0, transpose_B=0,
                                                         accumPrevC=0), [ld], cluster=c)
            src += nb
    return d, rings, out, start


def pushes_of(d, c):
    """Cluster c's pushes in the ring's order."""
    ps = [t for t in d.node_list if t.kernel_name == PUSH and t.assigned_cluster_id == c]
    return sorted(ps, key=lambda t: t.node_id)


print("\nthe runs: one push a run, on the ring's cluster's DM core")
d, rings, loads, start = build()
for c in range(2):
    ps = pushes_of(d, c)
    sizes = [nb for _, _, _, nb in rings.entries[c]]
    check(f"ring {c}: the pushes carry every chunk and byte",
          sum(p.kernel_args.chunks for p in ps) == len(sizes) and sum(p.kernel_args.size for p in ps) == sum(sizes),
          [(p.kernel_args.chunks, p.kernel_args.size) for p in ps])
    check(f"ring {c}: the first run is chunk 0 alone, then runs of at most the batch",
          ps[0].kernel_args.chunks == 1 and all(p.kernel_args.chunks <= 2 for p in ps), [p.kernel_args.chunks for p in ps])
    check(f"ring {c}: every push on the DM core of cluster {c}",
          all(p.assigned_core_id == DM and p.assigned_cluster_id == c for p in ps))
    check(f"ring {c}: the first push follows the start", d.has_edge(start, ps[0]))
    check(f"ring {c}: each push follows the one before", all(d.has_edge(a, b) for a, b in zip(ps, ps[1:])))
    lds = [ld for ld, _, _ in loads[c]]
    check(f"ring {c}: each load follows a push, and writes no release word",
          all(any(p.kernel_name == PUSH for p in d.predecessors(ld)) and ld.kernel_args.release_addr == 0
              for ld in lds))

print("\nthe overwrites: a run's push follows every load of the bytes it writes")
for c in range(2):
    run_of, ps, k = {}, pushes_of(d, c), 0
    for p in ps:
        for _ in range(p.kernel_args.chunks):
            run_of[k] = p
            k += 1
    late = []
    for kk, (_, lo, hi) in enumerate(loads[c]):
        for j in range(kk):
            ld, a, b = loads[c][j]
            if a < hi and lo < b and run_of[j] is not run_of[kk] and not nx.has_path(d, ld, run_of[kk]):
                late.append((j, kk))
    check(f"ring {c}: no push before a load of the bytes it overwrites", not late, late[:4])
    check(f"ring {c}: some chunk starts below its own slot (a short chunk), so the bytes matter",
          any(lo // 8192 != kk % 4 for kk, (_, lo, _) in enumerate(loads[c])))

print("\nthe queue share: two transfers a push, two rings on the engine, 8 entries -> 2 runs posted a ring")
for c in range(2):
    ps, firsts, k = pushes_of(d, c), [], 0
    for p in ps:
        firsts.append(loads[c][k][0])
        k += p.kernel_args.chunks
    check(f"ring {c}: push m follows the first load of run m - 2",
          all(nx.has_path(d, firsts[m - 2], ps[m]) for m in range(2, len(ps))))
check("the graph has no cycle", nx.is_directed_acyclic_graph(d))

print("\nthe args: the run's first chunk, its first slot and flag, its sequence-table entry")
p = pushes_of(d, 0)[1]
a = p.kernel_args.get_c_field_assignments({})
k0 = pushes_of(d, 0)[0].kernel_args.chunks
check("the engine: memory chip 0x20, engine 2", a["memchip"] == "0x20" and a["engine"] == "2", a)
check("the flag entry reads the sequence-table entry of the run's first chunk (k0 + 1, 64 B each)",
      a["flag_src_lo"] == f"(uint32_t)0x{SEQ + 64 * (k0 + 1):x}", a["flag_src_lo"])
check("the data entry carries the run's bytes, the flag entry 64 B a chunk",
      a["data_len_lo"] == str(p.kernel_args.size) and a["flag_len_lo"] == str(64 * p.kernel_args.chunks)
      and a["data_len_hi"] == a["flag_len_hi"] == "0", (a["data_len_lo"], a["flag_len_lo"]))
check("the doorbell and control words are 0", all(a[f] == "0" for f in ("data_doorbell", "data_ctrl",
                                                                        "flag_doorbell", "flag_ctrl")))

print("\nthe programs: each DM core posts a run before the loads of it")
d.bingo_schedule_programs()
for c in range(2):
    ts = [t for t in d.node_list if t.assigned_core_id == DM and t.assigned_cluster_id == c]
    h = nx.DiGraph()
    h.add_nodes_from(ts)
    h.add_edges_from((x, y) for x in ts for y in d.successors(x) if y in ts)
    order = list(nx.topological_sort(h))
    pos = {t: i for i, t in enumerate(order)}
    ok = all(pos[pp] < pos[ld] for ld, _, _ in loads[c] for pp in d.predecessors(ld) if pp.kernel_name == PUSH)
    check(f"cluster {c}: every push before its run's loads", ok)

print("\nwhat cluster mode refuses")
_, r2, _, _ = build(chunks=2)
check("a chunk taken outside load_chunk (it would have no push)",
      *raises(lambda: r2.take_ex(0, 0x1000, 4096), "load_chunk"))
check("a routed chunk", *raises(lambda: r2.load_chunk(lambda *a: None, 0, "r", src=0, nbytes=4096, slab=0, war=[],
                                                     kind=1, dm=DM), "routed"))
r3 = WeightRings(slots=BingoMemSymbol("s"), flags=BingoMemSymbol("f"), releases=BingoMemSymbol("r"), n_rings=2,
                 n_slots=4, slot_bytes=8192)
r3.plain_trailer = True
check("trailer chunks", *raises(lambda: r3.push_from(memchip=0x20, engine=0, seq=0, edge=None), "trailer"))

if FAILED:
    print(f"\n{len(FAILED)} FAILED: {FAILED}")
    _sys.exit(1)
print("\nall passed")
