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
from bingo_kernel_args import SnaxBingoKernelEnginePostArgs, SnaxBingoKernelGemmFullArgs  # noqa: E402
from bingo_mem_handle import BingoMemFixedAddr, BingoMemSymbol        # noqa: E402
from bingo_node import BingoNode                                      # noqa: E402
from libs.blocks import WeightRings                                   # noqa: E402

FAILED = []
GEMM, SIMD, XDMA, DM, HOST = 0, 1, 2, 3, 4
PUSH = SnaxBingoKernelEnginePostArgs.KERNEL_NAME
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

def shared_ring(defer):
    """One ring both clusters read: a post each run (post_chunk), each chunk read by both clusters, the releasing one
    last; that last read made `defer` chunks later. Returns (dfg, [(chunk, lo, hi, its post, its last read)], posts)."""
    d2 = new_dfg()
    r2s = WeightRings(slots=BingoMemSymbol("s2"), flags=BingoMemSymbol("f2"), releases=BingoMemSymbol("r2"), n_rings=1,
                      n_slots=4, slot_bytes=8192, batch=2)
    r2s.push_from(memchip=0x20, engine=1, seq=BingoMemFixedAddr(SEQ, mem_level="HBM"), edge=d2.bingo_add_edge, depth=8)
    st2 = task(d2, DM, "start", "__snax_bingo_kernel_stamp", None)
    r2s.start_after(st2)
    chunks2, late, src2 = [], [], 0x300000000000
    sizes2 = (8192, 4096, 4096, 8192)

    def finish(jj, tt, pp, ff, rr):
        last = task(d2, DM, f"ld{jj}_c{rr}", LOAD, None, [pp, ff], cluster=rr)
        r2s.read_chunk(0, tt[3] - 1, last, last=True)
        chunks2[jj] = chunks2[jj] + (last,)
    for j in range(14):
        nb = sizes2[j % len(sizes2)]
        c = j % 2
        tup, post = r2s.post_chunk(lambda n, core, kern, args, after, c=c: task(d2, core, f"{n}_c{c}", kern, args,
                                                                              after, cluster=c),
                                   0, f"ld{j}", src=src2, nbytes=nb, dm=DM)
        src2 += nb
        rel = (j // 3) % 2                      # the cluster that reads the chunk last
        first = task(d2, DM, f"ld{j}_c{1 - rel}", LOAD, None, [post], cluster=1 - rel)
        r2s.read_chunk(0, tup[3] - 1, first, last=False)
        late.append((j, tup, post, first, rel))
        chunks2.append((j, tup[0].offset, tup[0].offset + nb, post))
        for x in [x for x in late if x[0] <= j - defer]:
            finish(*x)
        late = [x for x in late if x[0] > j - defer]
    for x in late:
        finish(*x)
    posts2 = []
    for j, lo, hi, post, *_ in chunks2:
        if not posts2 or posts2[-1] is not post:
            posts2.append(post)
    return d2, chunks2, posts2


for defer, what in ((0, "each last read made right away"), (6, "the last reads made six chunks late")):
    print(f"\na ring both clusters read (post_chunk / read_chunk), {what}")
    d2, chunks2, posts2 = shared_ring(defer)
    overwriter = {j: next((q for q in posts2[posts2.index(post) + 1:] if q.kernel_args.dst_addr.offset < hi
                           and lo < q.kernel_args.dst_addr.offset + q.kernel_args.size), None)
                  for j, lo, hi, post, last in chunks2}
    missing = [(j, overwriter[j].node_name) for j, lo, hi, post, last in chunks2
               if overwriter[j] is not None and not nx.has_path(d2, last, overwriter[j])]
    check("every post follows the last reader of the bytes it overwrites", not missing, missing[:4])
    check("the shared ring's graph has no cycle", nx.is_directed_acyclic_graph(d2))
    if defer:
        check("some last read is made after the post that overwrites it (the late edge is needed)",
              any(q is not None and last.node_id > q.node_id for (j, lo, hi, post, last) in chunks2
                  for q in [overwriter[j]]))
    else:
        many = [q for q in posts2 if sum(1 for j, lo, hi, p, l in chunks2 if overwriter[j] is q) > 1]
        check("some post overwrites two chunks read last by different clusters (every last reader is needed)",
              any(len({l.assigned_cluster_id for j, lo, hi, p, l in chunks2 if overwriter[j] is q}) > 1 for q in many))
check("post_chunk outside cluster mode", *raises(lambda: WeightRings(
    slots=BingoMemSymbol("s3"), flags=BingoMemSymbol("f3"), releases=BingoMemSymbol("r3"), n_rings=1, n_slots=4,
    slot_bytes=8192).post_chunk(None, 0, "x", src=0, nbytes=4096, dm=DM), "cluster mode only"))

print("\nanother poster between a ring's runs: boundary, post_after, tail_before, runs_of")
d4 = new_dfg()
r4 = WeightRings(slots=BingoMemSymbol("s4"), flags=BingoMemSymbol("f4"), releases=BingoMemSymbol("r4"), n_rings=1,
                 n_slots=8, slot_bytes=8192, batch=4)
r4.push_from(memchip=0x20, engine=0, seq=BingoMemFixedAddr(SEQ, mem_level="HBM"), edge=d4.bingo_add_edge, depth=8)
r4.start_after(task(d4, DM, "start", "__snax_bingo_kernel_stamp", None))
cast_first = task(d4, HOST, "cast_post0", "__host_bingo_kernel_engine_post", None)
cast_last = task(d4, HOST, "cast_post1", "__host_bingo_kernel_engine_post", None, [cast_first])
route = task(d4, XDMA, "route", "__snax_bingo_kernel_moe_dispatch", None)
r4.boundary(0, 5, before=[cast_first], after=[cast_last])     # the cast between chunk 4's run and chunk 5's
r4.post_after(0, 8, [route])                                  # not taken yet: a boundary at 8 behind the route
lds4 = []
for j in range(12):
    ld4, _ = r4.load_chunk(lambda n, core, kern, args, after: task(d4, core, n, kern, args, after), 0, f"ld{j}",
                           src=0x100000000000 + j * 4096, nbytes=4096, slab=0x10000, war=lds4[-1:], dm=DM)
    lds4.append(ld4)
late_dep = task(d4, SIMD, "late_dep", "__snax_bingo_kernel_simd_mul_tiles", None, [lds4[0]])
r4.post_after(0, 2, [late_dep])                               # taken already: an edge into its run's post
tail = task(d4, HOST, "next_cast", "__host_bingo_kernel_engine_post", None)
r4.tail_before(0, [tail])
r4.close()
P = lambda k: r4.post_of(0, k)                                # noqa: E731
check("chunks 5 and 8 start runs (boundaries), chunk 2 joins chunk 1's",
      P(5) is not P(4) and P(8) is not P(7) and P(2) is P(1), [p.node_name for p in (P(1), P(2), P(4), P(5), P(8))])
check("the run before the boundary is posted before the cast, the run after it after the cast's last post",
      d4.has_edge(P(4), cast_first) and d4.has_edge(cast_last, P(5)))
check("post_after before the chunk is taken: its run's post follows the task", d4.has_edge(route, P(8)))
check("post_after once the chunk is taken: an edge into its run's post", d4.has_edge(late_dep, P(2)))
check("tail_before: the ring's last post before the task, once closed", d4.has_edge(P(11), tail))
runs4 = r4.runs_of(0)
check("runs_of: the runs' posts in ring order, each with its chunks' loads",
      [p for p, _ in runs4] == sorted({P(k) for k in range(12)}, key=lambda t: t.node_id)
      and [ld for _, ls in runs4 for ld in ls] == lds4, [(p.node_name, len(ls)) for p, ls in runs4])
check("the graph with the boundaries has no cycle", nx.is_directed_acyclic_graph(d4))
check("a boundary at a chunk taken already", *raises(lambda: r4.boundary(0, 3), "not taken yet"))

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
